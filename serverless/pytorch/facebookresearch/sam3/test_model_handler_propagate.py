"""Unit tests for SAM3 preload-cache propagation (no GPU)."""

import base64
import importlib.util
import io
import os
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
    def __init__(self, stream_responses):
        self._stream_responses = stream_responses
        self.stream_requests = []
        self.handle_requests = []
        self.close_requests = []

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        assert request["propagation_direction"] == "forward"
        yield from self._stream_responses

    def handle_request(self, request):
        self.handle_requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "fake-session"}
        if request["type"] == "close_session":
            self.close_requests.append(request)
            return {}
        return {}


def _stream_for_count(count, box_x=0.1):
    responses = []
    for idx in range(count):
        responses.append({
            "frame_index": idx,
            "outputs": {"out_boxes_xywh": [[box_x + idx * 0.01, 0.1, 0.05, 0.05]]},
        })
    return responses


def test_init_performs_one_full_chunk_propagation_and_caches_each_frame():
    preload_count = 96
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor(_stream_for_count(preload_count))
    handler._sessions = {}
    handler._output_to_bbox = lambda *_args, **_kwargs: [12.0, 12.0, 32.0, 32.0]
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    seed = [10.0, 10.0, 30.0, 30.0]
    preload_images = [_tiny_image_b64() for _ in range(preload_count)]
    shapes, states = handler.infer_batch(
        None,
        [seed],
        [],
        frame_index=0,
        preload_images=preload_images,
        preload_base_frame=0,
        preload_frame_count=preload_count,
        preload_payload_bytes=1234,
    )

    key = states[0]["session_key"]
    sess = handler._sessions[key]
    assert len(handler.predictor.stream_requests) == 1
    assert handler.predictor.stream_requests[0]["start_frame_index"] == 0
    assert handler.predictor.stream_requests[0]["max_frame_num_to_track"] == preload_count - 1
    assert sess["cache_ready"] is True
    assert len(sess["frame_cache"]) == preload_count
    assert sess["session_id"] is None
    assert sess["temp_dir"] is None
    assert len(handler.predictor.close_requests) == 1
    assert shapes == [[12.0, 12.0, 32.0, 32.0]]
    assert states[0]["base_frame"] == 0
    assert states[0]["preloaded_count"] == preload_count


def _cached_continue_handler(relative_frame):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([])
    key = "preload_track"
    preloaded_count = 96
    expected_bbox = [12.0 + relative_frame, 12.0, 32.0 + relative_frame, 32.0]
    frame_cache = {
        idx: {"bbox": [12.0 + idx, 12.0, 32.0 + idx, 32.0], "lost": False}
        for idx in range(preloaded_count)
    }
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
    shapes, states = handler.infer_batch(
        None,
        [None],
        [prev_state],
        frame_index=relative_frame,
    )
    return handler, shapes, states, expected_bbox


def test_track_cached_frames_invoke_no_predictor():
    for relative_frame in (1, 10, 70, 95):
        handler, shapes, states, expected_bbox = _cached_continue_handler(relative_frame)
        assert handler.predictor.stream_requests == []
        assert handler.predictor.handle_requests == []
        assert shapes == [expected_bbox]
        assert states[0]["last_bbox"] == expected_bbox


def test_track_out_of_range_fails_without_side_effects():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([])
    key = "preload_oob"
    preloaded_count = 5
    handler._sessions = {
        key: {
            "temp_dir": None,
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "base_frame": 0,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": preloaded_count - 1,
            "frame_cache": {i: {"bbox": [1, 1, 2, 2], "lost": False} for i in range(preloaded_count)},
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
    try:
        handler.infer_batch(None, [None], [prev_state], frame_index=preloaded_count)
        raised = False
    except PreloadRangeExhaustedError:
        raised = True
    assert raised
    assert handler.predictor.stream_requests == []


def test_track_without_preload_state_rejects_legacy_path():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    try:
        handler.infer_batch(None, [[1.0, 1.0, 2.0, 2.0]], [{}])
        raised = False
    except ValidationError:
        raised = True
    assert raised


def test_failed_init_closes_predictor_and_deletes_temp_files():
    preload_count = 3
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)

    class _FailPredictor(_FakePredictor):
        def handle_stream_request(self, request):
            raise RuntimeError("propagation failed")

    handler.predictor = _FailPredictor(_stream_for_count(preload_count))
    handler._sessions = {}
    handler._output_to_bbox = lambda *_args, **_kwargs: [12.0, 12.0, 32.0, 32.0]
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox
    preload_images = [_tiny_image_b64() for _ in range(preload_count)]
    seed = [10.0, 10.0, 30.0, 30.0]

    sessions_before = dict(handler._sessions)
    try:
        handler.infer_batch(
            None,
            [seed],
            [],
            frame_index=0,
            preload_images=preload_images,
            preload_base_frame=0,
            preload_frame_count=preload_count,
        )
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert handler._sessions == sessions_before
    assert len(handler.predictor.close_requests) == 1


def test_cache_construction_preserves_prompt_snap_back_loss():
    preload_count = 2
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]

    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([
        {"frame_index": 0, "outputs": {"out_boxes_xywh": [[0.5, 0.5, 0.05, 0.05]]}},
        {"frame_index": 1, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}},
    ])
    handler._sessions = {}

    def _bbox(outputs, sess, reference_bbox):
        frame_idx = 0 if reference_bbox == seed else 1
        if frame_idx == 0:
            return moved
        return seed

    handler._output_to_bbox = _bbox
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    preload_images = [_tiny_image_b64() for _ in range(preload_count)]
    _shapes, states = handler.infer_batch(
        None,
        [seed],
        [],
        frame_index=0,
        preload_images=preload_images,
        preload_base_frame=0,
        preload_frame_count=preload_count,
    )
    key = states[0]["session_key"]

    entry = handler._sessions[key]["frame_cache"][1]
    assert entry["bbox"] is None
    assert entry["lost"] is True

    shapes, states = handler.infer_batch(
        None,
        [None],
        [{
            "session_key": key,
            "last_bbox": moved,
            "base_frame": 0,
            "preloaded_count": preload_count,
            "preloaded_until_frame": preload_count - 1,
        }],
        frame_index=1,
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_init_invalid_preload_count_leaves_no_session():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([])
    handler._sessions = {}
    seed = [10.0, 10.0, 30.0, 30.0]
    try:
        handler.infer_batch(
            None,
            [seed],
            [],
            frame_index=0,
            preload_images=[_tiny_image_b64()],
            preload_base_frame=0,
            preload_frame_count=0,
        )
        raised = False
    except ValidationError:
        raised = True
    assert raised
    assert handler._sessions == {}
    assert handler.predictor.stream_requests == []


def test_init_corrupt_preload_image_leaves_no_session():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([])
    handler._sessions = {}
    seed = [10.0, 10.0, 30.0, 30.0]
    try:
        handler.infer_batch(
            None,
            [seed],
            [],
            frame_index=0,
            preload_images=["not-valid-base64-image"],
            preload_base_frame=0,
            preload_frame_count=1,
        )
        raised = False
    except Exception:
        raised = True
    assert raised
    assert handler._sessions == {}
    assert handler.predictor.handle_requests == []


def test_init_failure_after_session_start_leaves_no_usable_session():
    preload_count = 2
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)

    class _FailAfterStart(_FakePredictor):
        def handle_stream_request(self, request):
            raise RuntimeError("propagation failed after start")

    handler.predictor = _FailAfterStart(_stream_for_count(preload_count))
    handler._sessions = {}
    handler._output_to_bbox = lambda *_args, **_kwargs: [12.0, 12.0, 32.0, 32.0]
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox
    preload_images = [_tiny_image_b64() for _ in range(preload_count)]
    seed = [10.0, 10.0, 30.0, 30.0]

    try:
        handler.infer_batch(
            None,
            [seed],
            [],
            frame_index=0,
            preload_images=preload_images,
            preload_base_frame=0,
            preload_frame_count=preload_count,
        )
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert handler._sessions == {}
    assert len(handler.predictor.close_requests) == 1


def test_track_relative_frame_95_with_nonzero_base_frame():
    base_frame = 50
    preloaded_count = 96
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(ir_refine_enabled=False)
    handler.predictor = _FakePredictor([])
    key = "nonzero_base"
    expected_bbox = [12.0 + 95, 12.0, 32.0 + 95, 32.0]
    frame_cache = {
        idx: {"bbox": [12.0 + idx, 12.0, 32.0 + idx, 32.0], "lost": False}
        for idx in range(preloaded_count)
    }
    handler._sessions = {
        key: {
            "temp_dir": None,
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
            "base_frame": base_frame,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": base_frame + preloaded_count - 1,
            "frame_cache": frame_cache,
            "cache_ready": True,
        },
    }
    prev_state = {
        "session_key": key,
        "last_bbox": [10.0, 10.0, 30.0, 30.0],
        "base_frame": base_frame,
        "preloaded_count": preloaded_count,
        "preloaded_until_frame": base_frame + preloaded_count - 1,
    }
    shapes, states = handler.infer_batch(
        None,
        [],
        [prev_state],
        frame_index=base_frame + 95,
    )
    assert handler.predictor.stream_requests == []
    assert shapes == [expected_bbox]
    assert states[0]["last_bbox"] == expected_bbox
