"""Unit tests for SAM3 propagate frame-index filtering (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)
ModelHandler = _mod.ModelHandler


class _FakePredictor:
    def __init__(self, responses):
        self._responses = responses

    def handle_stream_request(self, request):
        assert request["propagation_direction"] == "forward"
        yield from self._responses


def test_propagate_frame_returns_only_target_index():
    handler = ModelHandler.__new__(ModelHandler)
    handler.predictor = _FakePredictor([
        {"frame_index": 0, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}},
        {"frame_index": 2, "outputs": {"out_boxes_xywh": [[0.3, 0.3, 0.05, 0.05]]}},
    ])
    sess = {"session_id": "fake-session"}
    out = handler._propagate_frame(sess, 0, 2)
    assert out is not None
    assert out["out_boxes_xywh"][0][0] == 0.3


def test_propagate_frame_returns_none_when_target_missing():
    handler = ModelHandler.__new__(ModelHandler)
    handler.predictor = _FakePredictor([
        {"frame_index": 0, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}},
    ])
    sess = {"session_id": "fake-session"}
    assert handler._propagate_frame(sess, 0, 1) is None
