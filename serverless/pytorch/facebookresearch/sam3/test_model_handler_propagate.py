"""Unit tests for SAM3 propagate frame-index filtering (no GPU)."""

import base64
import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)
ModelHandler = _mod.ModelHandler
PreloadRangeExhaustedError = _mod.PreloadRangeExhaustedError
Sam3Config = _mod.Sam3Config
ValidationError = _mod.ValidationError


def _tiny_image_b64():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakePredictor:
    def __init__(self, responses):
        self._responses = responses
        self.stream_requests = []

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        assert request["propagation_direction"] == "forward"
        yield from self._responses

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "fake-session"}
        return {}


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


def test_init_saves_preload_sequence_and_starts_one_session():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([
        {"frame_index": 0, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}},
    ])
    key = "preload_init"
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
    handler._output_to_bbox = lambda *_args, **_kwargs: [12.0, 12.0, 32.0, 32.0]
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    seed = [10.0, 10.0, 30.0, 30.0]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    preload_images = [_tiny_image_b64() for _ in range(3)]
    shapes, states = handler.infer_batch(
        image,
        [seed],
        [{"session_key": key}],
        frame_index=0,
        preload_images=preload_images,
        preload_base_frame=0,
        preload_frame_count=3,
        preload_payload_bytes=1234,
    )

    sess = handler._sessions[key]
    assert sess["preloaded_count"] == 3
    assert sess["preloaded_session_ready"] is True
    assert sess["loaded_frame_count"] == 3
    assert handler.predictor.stream_requests[0]["start_frame_index"] == 0
    assert handler.predictor.stream_requests[0]["max_frame_num_to_track"] == 1
    assert shapes == [[12.0, 12.0, 32.0, 32.0]]
    assert states[0]["base_frame"] == 0
    assert states[0]["preloaded_count"] == 3
    assert states[0]["preloaded_until_frame"] == 2


def _preloaded_continue_handler(relative_frame):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    key = "preload_track"
    temp_dir = tempfile.mkdtemp(prefix="sam3_test_")
    preloaded_count = 96
    handler._sessions = {
        key: {
            "temp_dir": temp_dir,
            "frame_count": preloaded_count,
            "loaded_frame_count": preloaded_count,
            "session_id": "fake-session",
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "image_height": 100,
            "image_width": 100,
            "base_frame": 0,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": preloaded_count - 1,
            "preloaded_session_ready": True,
        },
    }
    for idx in range(preloaded_count):
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
            f"{temp_dir}/{idx:05d}.jpg"
        )

    reload_calls = []
    handler._reload_session = lambda _sess: reload_calls.append(True)

    propagate_calls = []
    expected_bbox = [12.0 + relative_frame, 12.0, 32.0 + relative_frame, 32.0]

    def _propagate(sess, start, end):
        propagate_calls.append((start, end))
        return {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}

    handler._propagate_frame = _propagate
    handler._output_to_bbox = lambda *_args, **_kwargs: expected_bbox
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    prev_state = {
        "session_key": key,
        "last_bbox": [10.0, 10.0, 30.0, 30.0],
        "base_frame": 0,
        "preloaded_count": preloaded_count,
        "preloaded_until_frame": preloaded_count - 1,
    }
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [prev_state],
        frame_index=relative_frame,
    )
    return handler, reload_calls, propagate_calls, shapes, states


def test_track_relative_frames_do_not_reload_session():
    for relative_frame in (1, 10, 70):
        _handler, reload_calls, propagate_calls, shapes, states = _preloaded_continue_handler(
            relative_frame
        )
        assert reload_calls == []
        assert propagate_calls == [(relative_frame, relative_frame)]
        assert shapes[0] is not None
        assert states[0]["last_propagated_rel_frame"] == relative_frame


def test_track_propagates_only_requested_relative_frame():
    _handler, _reload_calls, propagate_calls, _shapes, _states = _preloaded_continue_handler(42)
    assert propagate_calls == [(42, 42)]


def test_track_out_of_range_fails_without_replay_fallback():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    key = "preload_oob"
    temp_dir = tempfile.mkdtemp(prefix="sam3_test_")
    preloaded_count = 5
    handler._sessions = {
        key: {
            "temp_dir": temp_dir,
            "frame_count": preloaded_count,
            "loaded_frame_count": preloaded_count,
            "session_id": "fake-session",
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "image_height": 100,
            "image_width": 100,
            "base_frame": 0,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": preloaded_count - 1,
            "preloaded_session_ready": True,
        },
    }
    reload_calls = []
    propagate_calls = []
    handler._reload_session = lambda _sess: reload_calls.append(True)
    handler._propagate_frame = lambda *_args: propagate_calls.append(True) or {}
    handler._track_frame = lambda *_args: (_ for _ in ()).throw(
        AssertionError("_track_frame must not be called")
    )

    prev_state = {
        "session_key": key,
        "last_bbox": [10.0, 10.0, 30.0, 30.0],
        "base_frame": 0,
        "preloaded_count": preloaded_count,
        "preloaded_until_frame": preloaded_count - 1,
    }
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        handler.infer_batch(image, [None], [prev_state], frame_index=preloaded_count)
        raised = False
    except PreloadRangeExhaustedError:
        raised = True
    assert raised
    assert reload_calls == []
    assert propagate_calls == []


def test_track_without_preload_state_rejects_legacy_path():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        handler.infer_batch(image, [[1.0, 1.0, 2.0, 2.0]], [{}])
        raised = False
    except ValidationError:
        raised = True
    assert raised
