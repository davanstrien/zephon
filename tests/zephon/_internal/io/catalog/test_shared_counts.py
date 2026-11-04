# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``node_shared_counts``: count-only discovery once per node."""

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import zephon._internal.io.catalog.handle as handle_mod
import zephon.io.dataset as dataset_mod
from zephon._internal.io.catalog import DatasetHeader, node_shared_counts

PINNED = "hf://org/name@" + "a" * 40 + "~original/default/train"


def _root(
    tmp_path: Path, name: str = "data", files: tuple[str, ...] = ("a.jsonl",)
) -> Path:
    root = tmp_path / name
    root.mkdir()
    for file in files:
        (root / file).write_text('{"i": 0}\n', encoding="utf-8")
    return root


def _header(root: Path | str, fmt: str = "jsonl") -> DatasetHeader:
    return DatasetHeader(name="demo", root=str(root), format=fmt, path=str(root))


def _counting(ids: list[int], counts: list[int]):
    calls: list[int] = []

    def compute() -> tuple[np.ndarray, np.ndarray]:
        calls.append(1)
        return np.array(ids, dtype=np.int64), np.array(counts, dtype=np.int64)

    return compute, calls


def _counts_files() -> list[Path]:
    bykey = handle_mod._schema_dir(handle_mod._catalog_dir()) / ".bykey"
    return sorted(bykey.glob(f"*{handle_mod._COUNTS_SUFFIX}"))


def test_second_call_reads_the_first_result(tmp_path: Path) -> None:
    header = _header(_root(tmp_path))
    compute, calls = _counting([0, 1, 2], [5, 6, 7])
    first = node_shared_counts(header, compute)
    second = node_shared_counts(header, compute)

    assert len(calls) == 1
    for ids, counts in (first, second):
        assert ids.tolist() == [0, 1, 2] and counts.tolist() == [5, 6, 7]
        assert ids.dtype == np.int64 and counts.dtype == np.int64
    assert len(_counts_files()) == 1


def test_format_root_and_listing_are_part_of_the_key(tmp_path: Path) -> None:
    root = _root(tmp_path)
    compute, calls = _counting([0], [1])
    node_shared_counts(_header(root), compute)
    node_shared_counts(_header(root, fmt="parquet"), compute)
    node_shared_counts(_header(_root(tmp_path, "other")), compute)
    assert len(calls) == 3

    # A changed file listing under the same root gets a new file.
    (root / "b.jsonl").write_text('{"i": 1}\n', encoding="utf-8")
    node_shared_counts(_header(root), compute)
    assert len(calls) == 4
    assert len(_counts_files()) == 4


def test_unreadable_file_is_recomputed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    header = _header(_root(tmp_path))
    compute, calls = _counting([0, 1], [3, 4])
    node_shared_counts(header, compute)
    (path,) = _counts_files()
    path.write_bytes(b"not an npz file")

    _, counts = node_shared_counts(header, compute)

    assert len(calls) == 2
    assert counts.tolist() == [3, 4]
    assert "unreadable shard counts file" in caplog.text
    # The recomputed result replaced the bad file.
    node_shared_counts(header, compute)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("ids", "counts"),
    [
        (np.array([0.0, 1.0]), np.array([3, 4])),  # float ids
        (np.array([1, 0]), np.array([3, 4])),  # unsorted ids
        (np.array([0, 0]), np.array([3, 4])),  # duplicate ids
        (np.array([0, 1]), np.array([3, -1])),  # negative count
        (np.array([0, 1]), np.array([3])),  # length mismatch
        (np.array([[0, 1]]), np.array([[3, 4]])),  # not 1-D
    ],
)
def test_malformed_file_is_recomputed(
    tmp_path: Path, ids: np.ndarray, counts: np.ndarray
) -> None:
    header = _header(_root(tmp_path))
    compute, calls = _counting([0, 1], [3, 4])
    node_shared_counts(header, compute)
    (path,) = _counts_files()
    with open(path, "wb") as fh:
        np.savez(fh, ids=ids, counts=counts)

    _, got = node_shared_counts(header, compute)

    assert len(calls) == 2
    assert got.tolist() == [3, 4]


def test_failed_compute_releases_the_lock_and_writes_nothing(tmp_path: Path) -> None:
    header = _header(_root(tmp_path))

    def failing() -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("scan failed")

    with pytest.raises(RuntimeError, match="scan failed"):
        node_shared_counts(header, failing)
    assert _counts_files() == []

    # The next process takes the lock, computes and shares the result.
    compute, calls = _counting([0], [7])
    assert node_shared_counts(header, compute)[1].tolist() == [7]
    assert node_shared_counts(header, compute)[1].tolist() == [7]
    assert len(calls) == 1


def test_uncreatable_catalog_dir_computes_in_each_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = _header(_root(tmp_path))
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setattr(handle_mod, "_catalog_dir", lambda: blocker / "catalog")
    compute, calls = _counting([0], [9])

    for _ in range(2):
        assert node_shared_counts(header, compute)[1].tolist() == [9]
    assert len(calls) == 2


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
    reason="needs POSIX permissions and a non-root user",
)
def test_read_only_catalog_dir_computes_in_each_process(tmp_path: Path) -> None:
    header = _header(_root(tmp_path))
    bykey = handle_mod._schema_dir(handle_mod._catalog_dir()) / ".bykey"
    bykey.mkdir(parents=True)
    bykey.chmod(0o555)
    compute, calls = _counting([0], [9])
    try:
        for _ in range(2):
            assert node_shared_counts(header, compute)[1].tolist() == [9]
    finally:
        bykey.chmod(0o755)
    assert len(calls) == 2


def test_concurrent_processes_compute_once(tmp_path: Path) -> None:
    """Processes that start together on one node run the scan once."""
    root = _root(tmp_path)
    marker = tmp_path / "computed"
    script = textwrap.dedent(
        f"""
        import time
        import numpy as np
        from zephon._internal.io.catalog import DatasetHeader, node_shared_counts

        def compute():
            with open({str(marker)!r}, "a") as fh:
                fh.write("x")
            time.sleep(0.5)
            return np.arange(4, dtype=np.int64), np.full(4, 10, dtype=np.int64)

        header = DatasetHeader(name="demo", root={str(root)!r}, format="jsonl", path=None)
        ids, counts = node_shared_counts(header, compute)
        print(int(counts.sum()))
        """
    )
    env = dict(os.environ, ZEPHON_CATALOG_DIR=str(tmp_path / "catalog"))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outputs = [proc.communicate(timeout=60) for proc in procs]

    for proc, (out, err) in zip(procs, outputs):
        assert proc.returncode == 0, err
        assert out.strip() == "40"
    assert marker.read_text() == "x"


class _Handler:
    kind = "parquet"

    def __init__(self) -> None:
        self.calls = 0

    def discover_counts(self, path: str, storage: object):
        self.calls += 1
        return [0, 1], [2, 3]


@pytest.mark.parametrize(
    ("path", "shared"),
    [
        (PINNED, True),
        ("hf://org/name@main~original/default/train", False),
        ("hf://org/name/train", False),
        ("s3://bucket/data", False),
        ("/local/data", False),
    ],
)
def test_from_path_shares_counts_only_for_pinned_hf_roots(
    path: str, shared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_calls: list[str] = []

    def fake_shared(
        header: DatasetHeader, compute: Callable[[], tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        shared_calls.append(header.root)
        return compute()

    monkeypatch.setattr(dataset_mod, "_node_shared_counts", fake_shared)
    handler = _Handler()
    ids, counts = dataset_mod._discover_counts(
        cast(Any, handler), _header(path, "parquet"), cast(Any, None)
    )

    assert ids.tolist() == [0, 1] and counts.tolist() == [2, 3]
    assert shared_calls == ([path] if shared else [])
    assert handler.calls == 1
