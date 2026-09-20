"""Hermetic tests for G0 preload_timeline serialization (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_g0_timeline", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

serialize_preload_timeline = _mod.serialize_preload_timeline
serialize_preload_summary = _mod.serialize_preload_summary
ModelHandler = _mod.ModelHandler


def test_serialize_preload_timeline_valid_and_lost_chain():
    cache = {
        0: {"bbox": [1.0, 2.0, 3.0, 4.0], "lost": False},
        1: {"bbox": [2.0, 3.0, 4.0, 5.0], "lost": False},
        2: {"bbox": None, "lost": True},
        3: {"bbox": None, "lost": True},
    }
    timeline = serialize_preload_timeline(cache, base_frame=100)
    assert len(timeline) == 4
    assert timeline[0]["bbox_status"] == "valid"
    assert timeline[0]["points"] == [1.0, 2.0, 3.0, 4.0]
    assert timeline[0]["frame"] == 100
    assert timeline[2]["bbox_status"] == "lost"
    assert timeline[2]["lost"] is True
    assert timeline[3]["bbox_status"] == "skipped"


def test_serialize_preload_summary():
    sess = {
        "base_frame": 406,
        "preloaded_count": 80,
        "preloaded_until_frame": 485,
    }
    summary = serialize_preload_summary(sess)
    assert summary == {
        "base_frame": 406,
        "preloaded_count": 80,
        "preloaded_until_frame": 485,
    }


def test_infer_batch_g0_extras_are_explicitly_opt_in():
    handler = ModelHandler.__new__(ModelHandler)
    handler._infer_preload_init = lambda *_args, **_kwargs: (
        [[1.0, 2.0, 3.0, 4.0]],
        [{"session_key": "k"}],
        {"preload_timeline": [{"frame": 0}]},
    )

    legacy = handler.infer_batch(
        None,
        [[1.0, 2.0, 3.0, 4.0]],
        [],
        preload_images=["fixture"],
    )
    extended = handler.infer_batch(
        None,
        [[1.0, 2.0, 3.0, 4.0]],
        [],
        preload_images=["fixture"],
        include_g0_extras=True,
    )

    assert len(legacy) == 2
    assert len(extended) == 3
    assert extended[2]["preload_timeline"][0]["frame"] == 0
