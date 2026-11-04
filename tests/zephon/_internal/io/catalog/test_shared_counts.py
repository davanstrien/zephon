# Copyright 2025 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``node_shared_counts``: count-only discovery once per node."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import zephon._internal.io.catalog.handle as handle_mod
import zephon.io.dataset as dataset_mod
from zephon._internal.io.catalog import node_shared_counts

PINNED = "hf://org/name@" + "a" * 40 + "~original/default/train"


def _counting(ids: list[int], counts: list[int]):
    calls: list[int] = []

    def compute() -> tuple[np.ndarray, np.ndarray]:
        calls.append(1)
        return np.array(ids, dtype=np.int64), np.array(counts, dtype=np.int64)

    return compute, calls


def _counts_files() -> list[Path]:
    counts_dir = handle_mod._schema_dir(handle_mod._catalog_dir()) / ".counts"
    return sorted(counts_dir.glob("*.npz"))


def test_second_call_reads_the_first_result() -> None:
    compute, calls = _counting([0, 1, 2], [5, 6, 7])
    first = node_shared_counts("parquet", PINNED, compute)
    second = node_shared_counts("parquet", PINNED, compute)

    assert len(calls) == 1
    for got in (first, second):
        np.testing.assert_array_equal(got[0], [0, 1, 2])
        np.testing.assert_array_equal(got[1], [5, 6, 7])
        assert got[0].dtype == np.int64 and got[1].dtype == np.int64
    assert len(_counts_files()) == 1


def test_format_and_root_are_part_of_the_key() -> None:
    compute, calls = _counting([0], [1])
    node_shared_counts("parquet", PINNED, compute)
    node_shared_counts("jsonl", PINNED, compute)
    node_shared_counts("parquet", PINNED.replace("train", "test"), compute)

    assert len(calls) == 3
    assert len(_counts_files()) == 3


def test_unreadable_file_is_recomputed(caplog: pytest.LogCaptureFixture) -> None:
    compute, calls = _counting([0, 1], [3, 4])
    node_shared_counts("parquet", PINNED, compute)
    (path,) = _counts_files()
    path.write_bytes(b"not an npz file")

    _, counts = node_shared_counts("parquet", PINNED, compute)

    assert len(calls) == 2
    np.testing.assert_array_equal(counts, [3, 4])
    assert "unreadable shard counts file" in caplog.text
    # The recomputed result replaced the bad file.
    assert node_shared_counts("parquet", PINNED, compute)[1].tolist() == [3, 4]
    assert len(calls) == 2


def test_unwritable_catalog_dir_computes_in_each_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setattr(handle_mod, "_catalog_dir", lambda: blocker / "catalog")
    compute, calls = _counting([0], [9])

    for _ in range(2):
        _, counts = node_shared_counts("parquet", PINNED, compute)
        assert counts.tolist() == [9]
    assert len(calls) == 2


def test_concurrent_processes_compute_once(tmp_path: Path) -> None:
    """Processes that start together on one node run the scan once."""
    marker = tmp_path / "computed"
    script = textwrap.dedent(
        f"""
        import time
        import numpy as np
        from zephon._internal.io.catalog import node_shared_counts

        def compute():
            with open({str(marker)!r}, "a") as fh:
                fh.write("x")
            time.sleep(0.5)
            return np.arange(4, dtype=np.int64), np.full(4, 10, dtype=np.int64)

        ids, counts = node_shared_counts("parquet", {PINNED!r}, compute)
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
def test_from_path_counts_are_shared_only_for_pinned_hf_roots(
    path: str, shared: bool
) -> None:
    handler = _Handler()
    for _ in range(2):
        ids, counts = dataset_mod._discover_counts(handler, path, storage=None)
        assert ids.tolist() == [0, 1] and counts.tolist() == [2, 3]
    assert handler.calls == (1 if shared else 2)
