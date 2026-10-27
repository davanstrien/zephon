"""The installed tokenizer engine against HF's recorded outputs.

Zephon loads with the tokenizers>=1 engine when that is installed and with HF
otherwise, and transformers pins tokenizers 0.x, so no environment runs both.
``tests/helpers/generate_tokenizer_reference.py`` records HF's outputs in
``reference.json``; the main test job checks HF still reproduces them, and the
tokenizers-v1 CI job runs this module alone to check the v1 engine does.
"""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import importlib.util
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pytest

from tests.helpers.work import FakeIndexableWorkSource, make_inmem_dataset
from zephon import Pipeline
from zephon._internal.ops.tokenize_text import TokenizeText
from zephon._internal.utils.tokenizer import (
    Refused,
    V1Tokenizer,
    fallback_tokenizer,
    installed_tokenizer_loader,
    load_v1_tokenizer,
)
from zephon.ops.base import OpContext, StageInfo
from zephon.types import SampleBatch, SampleMeta, SampleRecord

DATA = Path(__file__).resolve().parents[2] / "data" / "tokenizers"
DIRS = {"deepseek": DATA / "deepseek_like", "llama3": DATA / "llama3_like"}
REFERENCE_PATH = DATA / "reference.json"


def _tokenizers_major() -> int:
    try:
        return int(importlib.metadata.version("tokenizers").split(".")[0])
    except importlib.metadata.PackageNotFoundError:
        return 0


# Read from the environment, not from the engine choice under test, so a wrong
# choice fails these tests instead of skipping them.
V1_INSTALLED = _tokenizers_major() >= 1
# HF is skipped only where transformers is absent (the 3.14t job), as every
# transformers-backed test is.
NEEDS_ENGINE = pytest.mark.skipif(
    not V1_INSTALLED and importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)
NEEDS_V1 = pytest.mark.skipif(not V1_INSTALLED, reason="needs tokenizers>=1")

# Special and ``normalized: true`` added tokens of the tokenizer folders, as in
# DeepSeek V4's and Llama-3.2-1B's tokenizer.json (bos, eos, pad first).
DEEPSEEK_LIKE_SPECIALS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜▁pad▁｜>",
)
DEEPSEEK_LIKE_NORMALIZED = (
    "<｜fim▁hole｜>",
    "<｜fim▁begin｜>",
    "<｜fim▁end｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
)
LLAMA3_LIKE_SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|finetune_right_pad_id|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token_0|>",
)
_FAMILY_TOKENS = {
    "deepseek": (DEEPSEEK_LIKE_SPECIALS, DEEPSEEK_LIKE_NORMALIZED),
    "llama3": (LLAMA3_LIKE_SPECIALS, ()),
}


def adversarial_texts(family: str) -> dict[str, str]:
    """Strings where HF and the v1 engine are likeliest to disagree.

    Special-token text must stay content (``add_special_tokens=False``), and
    ``normalized: true`` added tokens match after normalization.
    """
    specials, normalized = _FAMILY_TOKENS[family]
    bos, eos, pad = specials[:3]
    return {
        "prose": "The quick brown fox jumps over the lazy dog.",
        "contractions": "It's what they'll say: we've done it, I'D agree.",
        "empty": "",
        "space": " ",
        "whitespace": "\n\n\t x\r\n",
        "leading_spaces": "  leading",
        "multibyte": "emoji 😀👍🏽 中文 日本語 ﬁ Ａ café",
        "digits": "1234567",
        "run_5000_chars": "lorem ipsum dolor " * 278,
        "eos_text": f"hello {eos} world",
        "bos_text": f"{bos}x",
        "pad_text": f"a{pad}b",
        **{f"special_{i}": f"x{t}y" for i, t in enumerate(specials[3:])},
        "unknown_special_text": "x<｜place▁holder▁no▁3｜>y<|reserved_special_token_9|>",
        **{f"normalized_{i}": f"q {t} z" for i, t in enumerate(normalized)},
    }


TRUNCATION_TEXTS = ["The quick brown fox jumps over the lazy dog.", "dog"]
OP_TEXTS = [
    "The quick brown fox jumps over the lazy dog, again and again. " * 4,
    "",
    "short 数据 😀",
    "q <｜fim▁hole｜> z",
]
OP_KWARGS: dict[str, dict[str, Any]] = {
    "bos_eos_split": {"split_long_samples": True, "max_length": 8},
    "truncation": {"truncation": True, "max_length": 8},
    "padding": {"padding": "max_length", "max_length": 64},
}
COUNT_TEXTS = ["The quick brown fox", "数据 😀 " * 20, "q <｜fim▁hole｜> z", "x"]


def as_json(value: Any) -> Any:
    """``value`` as it reads back from ``reference.json`` (arrays become lists)."""
    return json.loads(json.dumps(value, default=lambda a: a.tolist()))


@functools.cache
def load(family: str) -> Any:
    return installed_tokenizer_loader()(str(DIRS[family]))


def ids(family: str, text: str) -> Any:
    return load(family)([text], add_special_tokens=False)["input_ids"][0]


def special_ids(family: str) -> list[int | None]:
    tok = load(family)
    return [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id]


def truncated() -> Any:
    return load("deepseek")(
        TRUNCATION_TEXTS, add_special_tokens=False, truncation=True, max_length=4
    )["input_ids"]


def _setup(op: TokenizeText) -> None:
    op.setup(
        OpContext(
            {"record_node_metrics": lambda *a, **k: None},
            StageInfo(stage_index=0, stage_name="stage0", op_index=0),
        )
    )


def op_records(case: str) -> list[tuple[Any, Any, Any]]:
    """``TokenizeText.process_many`` output on ``OP_TEXTS`` under ``OP_KWARGS[case]``."""
    op = TokenizeText(
        tokenizer_id=str(DIRS["deepseek"]),
        field="text",
        **OP_KWARGS[case],
    )
    _setup(op)
    records = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"text": text},
        )
        for i, text in enumerate(OP_TEXTS)
    ]
    return [
        (rec.meta.sample_id, rec.meta.lineage, rec.payload)
        for rec in op.process_many(records)
    ]


def counts() -> list[int | None]:
    """Token counts of ``COUNT_TEXTS`` from the op's calibration counter."""
    op = TokenizeText(tokenizer_id=str(DIRS["deepseek"]), field="text")
    payloads = [{"text": t} for t in COUNT_TEXTS]
    plan = op.token_counting_spec().build_counter().plan(payloads)
    return [plan.count(p) for p in payloads]


def training_windows() -> list[dict[str, Any]]:
    """Packed training windows of a shuffled, tokenized pipeline."""
    rows = [
        {"text": f"Document {i}: the quick brown fox 数据 😀 " * (1 + i % 5)}
        for i in range(40)
    ]
    ws = FakeIndexableWorkSource(make_inmem_dataset("docs", rows), chunk_size=8)
    pipe = (
        Pipeline(ws)
        .decode_text()
        .tokenize(
            tokenizer_id=str(DIRS["deepseek"]),
            field="text",
            add_attention_mask=False,
        )
        .shuffle(buffer_size=16, seed=0)
        .pack_flat(max_length=32, algorithm="wrap")
        .batch(microbatch_size=4, drop_last=False)
        .options(deterministic=True, max_workers=1, default_stage_prefetch=0)
    )
    windows = []
    for batch in pipe:
        assert isinstance(batch, SampleBatch)
        out = batch.to_training(return_labels=True)
        windows.append({k: np.asarray(out[k]) for k in ("input_ids", "labels")})
    return windows


def compute() -> dict[str, Any]:
    """Every output the tests compare, keyed as in ``reference.json``."""
    return as_json(
        {
            "ids": {
                family: {
                    name: ids(family, text)
                    for name, text in adversarial_texts(family).items()
                }
                for family in DIRS
            },
            "special_ids": {family: special_ids(family) for family in DIRS},
            "truncated": truncated(),
            "op_records": {case: op_records(case) for case in OP_KWARGS},
            "counts": counts(),
            "training_windows": training_windows(),
        }
    )


@functools.cache
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Engine choice by installed tokenizers version
# ---------------------------------------------------------------------------


def _install_tokenizers(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    real = importlib.metadata.version

    def version_of(name: str) -> str:
        if name != "tokenizers":
            return real(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(importlib.metadata, "version", version_of)


@pytest.mark.parametrize(
    "version,engine",
    [(None, "hf"), ("0.23.3", "hf"), ("1.0.0rc3", "v1"), ("1.2.0", "v1")],
)
def test_engine_follows_installed_tokenizers(
    monkeypatch: pytest.MonkeyPatch, version: str | None, engine: str
) -> None:
    def loader(name: str) -> Any:
        def load(*args: Any, **kwargs: Any) -> Any:
            tok = fallback_tokenizer()
            tok.name_or_path = name
            return tok

        return load

    for name in ("hf", "v1"):
        monkeypatch.setattr(
            f"zephon._internal.utils.tokenizer.load_{name}_tokenizer", loader(name)
        )
    _install_tokenizers(monkeypatch, version)
    op = TokenizeText(tokenizer_id="some/dir", field="text")
    # Decided once at construction: worker setup and calibration route on it.
    _install_tokenizers(monkeypatch, "0.23.3" if engine == "v1" else "1.0.0")
    worker_op = cloudpickle.loads(cloudpickle.dumps(op))
    worker_op._setup_tokenizer()
    counter = op.token_counting_spec().build_counter()
    assert [worker_op.tok.name_or_path, counter.tokenizer.name_or_path] == [
        engine,
        engine,
    ]


@pytest.mark.parametrize(
    "kwargs,refused",
    [
        ({"special_tokens": "tokenizer_default"}, "special_tokens"),
        ({"use_fast": False}, "use_fast=False"),
        ({"tokenizer_id": None}, "tokenizer_id=None"),
    ],
    ids=["tokenizer_default", "use_fast", "no_tokenizer_id"],
)
def test_v1_refuses_at_construction(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], refused: str
) -> None:
    _install_tokenizers(monkeypatch, "1.0.0")
    kwargs = {"tokenizer_id": "some/dir", **kwargs}
    with pytest.raises(Refused, match=refused):
        TokenizeText(field="text", **kwargs)


def test_v1_accepts_any_options_with_a_prebuilt_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tokenizers(monkeypatch, "1.0.0")
    tok = fallback_tokenizer()
    op = TokenizeText(
        tok, field="text", special_tokens="tokenizer_default", use_fast=False
    )
    op._setup_tokenizer()
    assert op.tok is tok


# ---------------------------------------------------------------------------
# Installed engine (HF in the main job, tokenizers>=1 in the tokenizers-v1 job)
# ---------------------------------------------------------------------------


def test_committed_folder_marks_normalized_added_tokens() -> None:
    spec = json.loads((DIRS["deepseek"] / "tokenizer.json").read_text())
    normalized = {t["content"] for t in spec["added_tokens"] if t["normalized"]}
    assert normalized == set(DEEPSEEK_LIKE_NORMALIZED)


@NEEDS_ENGINE
@pytest.mark.parametrize(
    "family,name",
    [(family, name) for family in DIRS for name in adversarial_texts(family)],
)
def test_ids_match_reference(family: str, name: str) -> None:
    got = ids(family, adversarial_texts(family)[name])
    assert as_json(got) == reference()["ids"][family][name]


@NEEDS_ENGINE
@pytest.mark.parametrize("family", list(DIRS))
def test_special_token_ids_match_reference(family: str) -> None:
    assert special_ids(family) == reference()["special_ids"][family]


@NEEDS_ENGINE
def test_single_str_returns_one_sequence() -> None:
    text = adversarial_texts("deepseek")["prose"]
    got = load("deepseek")(text, add_special_tokens=False)["input_ids"]
    assert as_json(got) == reference()["ids"]["deepseek"]["prose"]


@NEEDS_ENGINE
def test_truncation_keeps_the_head() -> None:
    assert as_json(truncated()) == reference()["truncated"]


@NEEDS_ENGINE
@pytest.mark.parametrize(
    "roundtrip",
    [lambda t: pickle.loads(pickle.dumps(t)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_tokenizer_survives_copy(roundtrip: Any) -> None:
    texts = adversarial_texts("deepseek")
    clone = roundtrip(load("deepseek"))
    got = as_json(clone(list(texts.values()), add_special_tokens=False)["input_ids"])
    assert dict(zip(texts, got)) == reference()["ids"]["deepseek"]


@NEEDS_ENGINE
@pytest.mark.parametrize("case", list(OP_KWARGS))
def test_records_match_reference(case: str) -> None:
    assert as_json(op_records(case)) == reference()["op_records"][case]


@NEEDS_ENGINE
def test_none_text_raises() -> None:
    op = TokenizeText(tokenizer_id=str(DIRS["deepseek"]), field="text")
    _setup(op)
    records = [
        SampleRecord(
            meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
            payload={"text": text},
        )
        for i, text in enumerate(["fine", None])
    ]
    with pytest.raises(TypeError):
        op.process_many(records)


@NEEDS_ENGINE
def test_counts_through_the_op_match_reference() -> None:
    assert counts() == reference()["counts"]


@NEEDS_ENGINE
def test_training_windows_match_reference() -> None:
    windows = as_json(training_windows())
    assert len(windows) > 1
    assert windows == reference()["training_windows"]


# ---------------------------------------------------------------------------
# V1Tokenizer
# ---------------------------------------------------------------------------


@NEEDS_V1
def test_v1_installed_loads_v1_tokenizer() -> None:
    op = TokenizeText(tokenizer_id=str(DIRS["deepseek"]), field="text")
    op._setup_tokenizer()
    assert isinstance(op.tok, V1Tokenizer)


@NEEDS_V1
def test_v1_ids_are_int64() -> None:
    assert ids("deepseek", "The quick brown fox").dtype == np.int64


@NEEDS_V1
@pytest.mark.parametrize(
    "kwargs",
    [
        {"add_special_tokens": True},
        {"padding": True},
        {"truncation": True, "max_length": 4, "truncation_side": "left"},
    ],
    ids=["add_special_tokens", "padding", "truncation_side_left"],
)
def test_v1_refuses_call_options(kwargs: dict[str, Any]) -> None:
    refused = next(k for k in kwargs if k not in ("truncation", "max_length"))
    with pytest.raises(Refused, match=refused):
        load("deepseek")(["hello"], **kwargs)


@NEEDS_V1
def test_v1_refuses_disagreeing_special_token_files(tmp_path: Path) -> None:
    folder = tmp_path / "tok"
    shutil.copytree(DIRS["llama3"], folder)
    special_map = json.loads((folder / "special_tokens_map.json").read_text())
    special_map["pad_token"]["content"] = "<|eot_id|>"
    (folder / "special_tokens_map.json").write_text(json.dumps(special_map))
    with pytest.raises(Refused, match=r"pad_token.*finetune_right_pad_id.*eot_id"):
        load_v1_tokenizer(str(folder))
