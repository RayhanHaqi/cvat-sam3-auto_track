"""Unit tests for SAM3 lost-target / null bbox with preload cache (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_lost", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)
ModelHandler = _mod.ModelHandler
Sam3Config = _mod.Sam3Config
ValidationError = _mod.ValidationError
load_sam3_config = _mod.load_sam3_config
refine_bbox_with_ir_intensity = _mod.refine_bbox_with_ir_intensity


def _cached_handler(frame_cache, *, prev_lost=False, preloaded_count=96):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = SimpleNamespace(
        handle_stream_request=lambda *_args, **_kwargs: iter([]),
        handle_request=lambda *_args, **_kwargs: {},
    )
    key = "test_sess"
    handler._sessions = {
        key: {
            "temp_dir": None,
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "image_height": 100,
            "image_width": 100,
            "base_frame": 0,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": preloaded_count - 1,
            "frame_cache": frame_cache,
            "cache_ready": True,
        },
    }
    prev_state = {
        "session_key": key,
        "last_bbox": [10.0, 10.0, 30.0, 30.0],
        "base_frame": 0,
        "preloaded_count": preloaded_count,
        "preloaded_until_frame": preloaded_count - 1,
    }
    if prev_lost:
        prev_state["lost"] = True
        prev_state["last_bbox"] = None
    return handler, key, prev_state


def test_load_sam3_config_ir_stop_on_missing():
    cfg = load_sam3_config({"SAM3_IR_STOP_ON_MISSING": "1"})
    assert cfg.ir_stop_on_missing is True


def test_continue_returns_none_when_cached_entry_lost():
    cache = {1: {"bbox": None, "lost": True}}
    handler, _key, prev_state = _cached_handler(cache)
    shapes, states = handler.infer_batch(
        None,
        [None],
        [prev_state],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_continue_accepts_backend_padded_null_shapes():
    expected = [12.0, 12.0, 32.0, 32.0]
    cache = {1: {"bbox": expected, "lost": False}}
    handler, _key, prev_state = _cached_handler(cache)
    shapes, states = handler.infer_batch(
        None,
        [None],
        [prev_state],
        frame_index=1,
    )
    assert shapes == [expected]
    assert states[0]["last_bbox"] == expected
    assert states[0].get("lost") is not True


def test_continue_returns_none_when_prior_state_lost():
    cache = {1: {"bbox": [50.0, 50.0, 70.0, 70.0], "lost": False}}
    handler, _key, prev_state = _cached_handler(cache, prev_lost=True)
    shapes, states = handler.infer_batch(
        None,
        [None],
        [prev_state],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_init_rejects_all_null_shapes_without_session():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler._sessions = {}
    with pytest.raises(ValidationError, match="bounding box"):
        handler.infer_batch(
            None,
            [None],
            [{}],
            preload_images=["Zm9v"],
            preload_base_frame=0,
            preload_frame_count=1,
        )


def test_init_uses_seed_bbox_when_sam_output_missing():
    preload_count = 1
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = SimpleNamespace(
        handle_stream_request=lambda request: iter([
            {"frame_index": 0, "outputs": {}},
        ]),
        handle_request=lambda request: (
            {"session_id": "fake"} if request["type"] == "start_session" else {}
        ),
    )
    key = "init_sess"
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="sam3_test_")
    handler._sessions = {
        key: {
            "temp_dir": temp_dir,
            "frame_count": 0,
            "loaded_frame_count": 0,
            "session_id": None,
            "prompt_bbox": None,
            "image_height": None,
            "image_width": None,
            "base_frame": None,
            "preloaded_count": 0,
            "preloaded_until_frame": None,
            "frame_cache": {},
            "cache_ready": False,
        },
    }
    handler._output_to_bbox = lambda *_args, **_kwargs: None
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    preload_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    seed = [10.0, 10.0, 30.0, 30.0]
    shapes, states = handler.infer_batch(
        None,
        [seed],
        [{"session_key": key}],
        frame_index=0,
        preload_images=[preload_b64],
        preload_base_frame=0,
        preload_frame_count=preload_count,
    )
    assert shapes == [seed]
    assert states[0].get("lost") is not True


def test_refine_stop_on_missing_returns_none_without_support():
    image = np.full((200, 200, 3), 20, dtype=np.uint8)
    sam_bbox = [90.0, 90.0, 110.0, 110.0]
    assert refine_bbox_with_ir_intensity(image, sam_bbox, stop_on_missing=True) is None
    assert refine_bbox_with_ir_intensity(image, sam_bbox, stop_on_missing=False) == sam_bbox
