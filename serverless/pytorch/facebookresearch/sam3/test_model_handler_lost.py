"""Unit tests for SAM3 lost-target / null bbox continuation (no GPU)."""

import base64
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

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


def _tiny_image_b64():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _continue_handler(output_bbox, frame_index=1, base_frame=0, preloaded_count=96):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    key = "test_sess"
    temp_dir = tempfile.mkdtemp(prefix="sam3_test_")
    handler._sessions = {
        key: {
            "temp_dir": temp_dir,
            "frame_count": preloaded_count,
            "loaded_frame_count": preloaded_count,
            "session_id": "fake-session",
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "image_height": 100,
            "image_width": 100,
            "base_frame": base_frame,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": base_frame + preloaded_count - 1,
            "preloaded_session_ready": True,
        },
    }
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
        f"{temp_dir}/{frame_index - base_frame:05d}.jpg"
    )

    handler._reload_session = lambda _sess: (_ for _ in ()).throw(
        AssertionError("_reload_session must not be called")
    )
    handler._propagate_frame = lambda _sess, start, end: {}
    handler._output_to_bbox = lambda *_args, **_kwargs: output_bbox
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox
    return handler, key


def test_load_sam3_config_ir_stop_on_missing():
    cfg = load_sam3_config({"SAM3_IR_STOP_ON_MISSING": "1"})
    assert cfg.ir_stop_on_missing is True


def test_continue_returns_none_when_sam_output_missing():
    handler, key = _continue_handler(None)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{
            "session_key": key,
            "last_bbox": [10.0, 10.0, 30.0, 30.0],
            "base_frame": 0,
            "preloaded_count": 96,
            "preloaded_until_frame": 95,
        }],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_continue_accepts_backend_padded_null_shapes():
    expected = [12.0, 12.0, 32.0, 32.0]
    handler, key = _continue_handler(expected)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{
            "session_key": key,
            "last_bbox": [10.0, 10.0, 30.0, 30.0],
            "base_frame": 0,
            "preloaded_count": 96,
            "preloaded_until_frame": 95,
        }],
        frame_index=1,
    )
    assert shapes == [expected]
    assert states[0]["last_bbox"] == expected
    assert states[0].get("lost") is not True


def test_continue_returns_none_when_prior_state_lost():
    handler, key = _continue_handler([50.0, 50.0, 70.0, 70.0])
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{"session_key": key, "last_bbox": None, "lost": True, "preloaded_count": 96}],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_continue_returns_none_when_sam_snaps_back_to_seed():
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    handler, key = _continue_handler(seed)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{
            "session_key": key,
            "last_bbox": moved,
            "base_frame": 0,
            "preloaded_count": 96,
            "preloaded_until_frame": 95,
        }],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_init_rejects_all_null_shapes_without_session():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler._sessions = {}
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValidationError, match="bounding box"):
        handler.infer_batch(
            image,
            [None],
            [{}],
            frame_index=0,
            preload_images=[_tiny_image_b64()],
            preload_base_frame=0,
            preload_frame_count=1,
        )


def test_init_uses_seed_bbox_when_sam_output_missing():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    key = "init_sess"
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
        },
    }

    handler._start_preloaded_session = lambda sess: sess.__setitem__("preloaded_session_ready", True) or True
    handler._propagate_frame = lambda *_args: {}
    handler._output_to_bbox = lambda *_args, **_kwargs: None
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    seed = [10.0, 10.0, 30.0, 30.0]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [seed],
        [{"session_key": key}],
        frame_index=0,
        preload_images=[_tiny_image_b64()],
        preload_base_frame=0,
        preload_frame_count=1,
    )
    assert shapes == [seed]
    assert states[0].get("lost") is not True


def test_init_keeps_seed_bbox_when_ir_stop_on_missing_returns_none():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=True, ir_stop_on_missing=True)
    key = "init_ir_sess"
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
        },
    }

    handler._start_preloaded_session = lambda sess: sess.__setitem__("preloaded_session_ready", True) or True
    handler._propagate_frame = lambda *_args: {}
    handler._output_to_bbox = lambda *_args, **_kwargs: [12.0, 12.0, 32.0, 32.0]
    handler._maybe_refine_bbox = _mod.ModelHandler._maybe_refine_bbox.__get__(handler, ModelHandler)

    seed = [10.0, 10.0, 30.0, 30.0]
    image = np.full((200, 200, 3), 20, dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [seed],
        [{"session_key": key}],
        frame_index=0,
        preload_images=[_tiny_image_b64()],
        preload_base_frame=0,
        preload_frame_count=1,
    )
    assert shapes == [seed]
    assert states[0]["last_bbox"] == seed
    assert states[0].get("lost") is not True


def test_continue_returns_none_when_ir_stop_on_missing():
    handler, key = _continue_handler([90.0, 90.0, 110.0, 110.0])
    handler.config = Sam3Config(ir_refine_enabled=True, ir_stop_on_missing=True)
    handler._maybe_refine_bbox = _mod.ModelHandler._maybe_refine_bbox.__get__(handler, ModelHandler)
    image = np.full((200, 200, 3), 20, dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{
            "session_key": key,
            "last_bbox": [90.0, 90.0, 110.0, 110.0],
            "base_frame": 0,
            "preloaded_count": 96,
            "preloaded_until_frame": 95,
        }],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_refine_stop_on_missing_returns_none_without_support():
    image = np.full((200, 200, 3), 20, dtype=np.uint8)
    sam_bbox = [90.0, 90.0, 110.0, 110.0]
    assert refine_bbox_with_ir_intensity(image, sam_bbox, stop_on_missing=True) is None
    assert refine_bbox_with_ir_intensity(image, sam_bbox, stop_on_missing=False) == sam_bbox
