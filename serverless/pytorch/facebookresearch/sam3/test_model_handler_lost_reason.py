"""Pure-lost-reason diagnostics tests for SAM3 handler (no GPU/checkpoint)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_lost_reason", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

ModelHandler = _mod.ModelHandler
Sam3Config = _mod.Sam3Config


@pytest.fixture
def captured(monkeypatch):
    records = []
    monkeypatch.setattr(_mod, "diag_enabled", lambda: True)
    monkeypatch.setattr(_mod, "log_frame_record", records.append)
    return records


def _dark_frame(size=200):
    return np.full((size, size, 3), 20, dtype=np.uint8)


def _handler(config):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = config
    return handler


def test_output_none_reason_logged(captured):
    handler = _handler(Sam3Config(ir_refine_enabled=False, output_policy="mask_bbox"))
    handler._output_to_bbox = lambda *_a, **_k: None
    sess = {"image_height": 200, "image_width": 200}

    bbox, lost, *_ = handler._postprocess_frame_outputs(
        sess,
        _dark_frame(),
        {"out_boxes_xywh": None},
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[40.0, 40.0, 60.0, 60.0],
        prev_lost=False,
        relative_frame=3,
    )

    assert bbox is None and lost is True
    record = captured[-1]
    assert record["event"] == "lost_frame"
    assert record["reason"] == "output_none"
    assert record["relativeFrame"] == 3
    assert record["rawSamBbox"] is None
    assert record["refinedBbox"] is None
    assert record["promptBbox"]["xtl"] == 10.0
    assert record["lastKnownBbox"]["xtl"] == 40.0
    assert record["refineCornerDeltas"] is None
    assert record["irRefineEnabled"] is False
    assert record["irStopOnMissing"] is False
    assert record["outputPolicy"] == "mask_bbox"
    assert "outputs" in record
    assert record["outputs"]["imageHeight"] == 200


def test_refine_none_reason_logged(captured):
    handler = _handler(
        Sam3Config(
            ir_refine_enabled=True,
            ir_stop_on_missing=True,
            output_policy="mask_bbox",
        )
    )
    handler._output_to_bbox = ModelHandler._output_to_bbox.__get__(handler, ModelHandler)
    mask = np.zeros((200, 200), dtype=bool)
    mask[90:110, 90:110] = True
    sess = {"image_height": 200, "image_width": 200}

    bbox, lost, *_ = handler._postprocess_frame_outputs(
        sess,
        _dark_frame(),
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[12.0, 12.0, 32.0, 32.0],
        prev_lost=False,
        relative_frame=5,
    )

    assert bbox is None and lost is True
    record = captured[-1]
    assert record["reason"] == "refine_none"
    assert record["relativeFrame"] == 5
    assert record["rawSamBbox"] is not None
    assert record["refinedBbox"] is None
    assert record["irRefineEnabled"] is True
    assert record["irStopOnMissing"] is True


def test_false_relock_reason_logged(captured):
    handler = _handler(Sam3Config(ir_refine_enabled=False, output_policy="mask_bbox"))
    relocked = [10.0, 10.0, 30.0, 30.0]
    handler._output_to_bbox = lambda *_a, **_k: list(relocked)
    sess = {"image_height": 200, "image_width": 200}

    bbox, lost, *_ = handler._postprocess_frame_outputs(
        sess,
        _dark_frame(),
        {"out_boxes_xywh": None},
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[100.0, 100.0, 120.0, 120.0],
        prev_lost=False,
        relative_frame=7,
    )

    assert bbox is None and lost is True
    record = captured[-1]
    assert record["reason"] == "false_relock"
    assert record["relativeFrame"] == 7
    assert record["rawSamBbox"]["xtl"] == 10.0
    assert record["refinedBbox"]["xtl"] == 10.0
    assert record["refinedNearPrompt"] is True
    assert record["refinedNearLastKnown"] is False
    assert record["lastKnownNearPrompt"] is False
    assert record["refineCornerDeltas"] == [0.0, 0.0, 0.0, 0.0]


def test_chunk_summary_reports_first_lost_reason(captured):
    handler = _handler(Sam3Config(ir_refine_enabled=False, output_policy="mask_bbox"))
    handler._output_to_bbox = lambda outputs, _sess, _ref: (
        None if outputs.get("dead") else [10.0, 10.0, 30.0, 30.0]
    )
    handler._save_preload_sequence = lambda *_a, **_k: None
    handler._start_session = lambda *_a, **_k: "fake"
    handler._add_prompt = lambda *_a, **_k: None
    handler._propagate_full_chunk = lambda *_a, **_k: {0: {"dead": False}, 1: {"dead": True}}
    handler._load_preloaded_frame = lambda *_a, **_k: _dark_frame(size=20)
    handler._cleanup_session_workspace = lambda *_a, **_k: None

    sess = {"temp_dir": None}
    cache = handler._build_frame_cache(sess, ["unused"], 2, [10.0, 10.0, 30.0, 30.0])

    assert cache[0]["lost"] is False
    assert cache[1]["lost"] is True
    assert cache[1]["bbox"] is None
    summary = [r for r in captured if r.get("event") == "chunk_lost_summary"][-1]
    assert summary["chunkLength"] == 2
    assert summary["preloadCount"] == 2
    assert summary["validCount"] == 1
    assert summary["lostCount"] == 1
    assert summary["firstLostRelativeFrame"] == 1
    assert summary["firstLostReason"] == "output_none"


def test_no_diagnostics_when_disabled(monkeypatch):
    records = []
    monkeypatch.setattr(_mod, "diag_enabled", lambda: False)
    monkeypatch.setattr(_mod, "log_frame_record", records.append)
    handler = _handler(Sam3Config(ir_refine_enabled=False, output_policy="mask_bbox"))
    handler._output_to_bbox = lambda *_a, **_k: None
    sess = {"image_height": 200, "image_width": 200}

    bbox, lost, *_ = handler._postprocess_frame_outputs(
        sess,
        _dark_frame(),
        {},
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[40.0, 40.0, 60.0, 60.0],
        prev_lost=False,
        relative_frame=1,
    )

    assert bbox is None and lost is True
    assert records == []
