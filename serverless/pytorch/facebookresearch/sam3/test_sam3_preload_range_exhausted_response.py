"""Tests for structured PreloadRangeExhaustedError HTTP responses."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_diagnostics = _load_module("sam3_diagnostics_for_main_test", _ROOT / "diagnostics.py")
sys.modules["diagnostics"] = _diagnostics
_model_handler = _load_module("sam3_model_handler_for_main_test", _ROOT / "model_handler.py")
sys.modules["model_handler"] = _model_handler
_main = _load_module("sam3_main_for_test", _ROOT / "main.py")

handler = _main.handler
PRELOAD_RANGE_EXHAUSTED_CODE = _main.PRELOAD_RANGE_EXHAUSTED_CODE
PreloadRangeExhaustedError = _main.PreloadRangeExhaustedError


class _FakeResponse:
    def __init__(self, body: str, status_code: int, **kwargs):
        self.body = body
        self.status_code = status_code
        self.headers = kwargs.get("headers", {})
        self.content_type = kwargs.get("content_type", "application/json")


def test_handler_returns_structured_preload_range_exhausted_payload():
    context = SimpleNamespace(
        logger=SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        ),
        user_data=SimpleNamespace(
            model=SimpleNamespace(
                infer_batch=mock.Mock(
                    side_effect=PreloadRangeExhaustedError(
                        "Preloaded frame range exhausted at frame 73 "
                        "(chunk base=0, count=73); re-seed tracking from a new annotation frame"
                    )
                ),
                _sessions={},
            )
        ),
        Response=_FakeResponse,
    )
    event = SimpleNamespace(
        body={
            "frame_index": 73,
            "states": [{"session_key": "s1", "base_frame": 0, "preloaded_count": 73}],
        }
    )

    response = handler(context, event)

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload == {
        "code": PRELOAD_RANGE_EXHAUSTED_CODE,
        "message": (
            "Preloaded frame range exhausted at frame 73 "
            "(chunk base=0, count=73); re-seed tracking from a new annotation frame"
        ),
    }
