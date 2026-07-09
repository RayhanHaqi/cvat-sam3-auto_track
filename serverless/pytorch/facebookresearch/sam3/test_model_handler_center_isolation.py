"""Center-state isolation for adaptive_component_square_box (no GPU)."""

import base64
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_center_iso", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

ModelHandler = _mod.ModelHandler
Sam3Config = _mod.Sam3Config
bbox_center = _mod.bbox_center
bbox_side_length = _mod.bbox_side_length


def _dark_frame(height=200, width=200, background=20):
    return np.full((height, width, 3), background, dtype=np.uint8)


def _draw_disk(image, center_x, center_y, radius, intensity=250):
    cv2.circle(image, (center_x, center_y), radius, (intensity, intensity, intensity), -1)


def _mask_at(cx, cy, size=4, frame_size=200):
    mask = np.zeros((frame_size, frame_size), dtype=np.uint8)
    half = size // 2
    mask[cy - half : cy + half, cx - half : cx + half] = 1
    return mask


def _handler(policy, *, ir_refine=False):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(output_policy=policy, ir_refine_enabled=ir_refine)
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox
    return handler


def _sess():
    return {"image_height": 200, "image_width": 200}


def _postprocess_sequence(handler, masks, seed, prompt=None):
    prompt = prompt or seed
    sess = _sess()
    prev_bbox = None
    prev_accepted = None
    canonical_history = []
    emitted_history = []
    for rel, mask in enumerate(masks):
        frame_image = _dark_frame()
        _draw_disk(frame_image, 100, 100, radius=10, intensity=255)
        emitted, lost, _diag, canonical = handler._postprocess_frame_outputs(
            sess,
            frame_image,
            {"out_binary_masks": [mask]},
            is_init_frame=rel == 0,
            prompt_bbox=prompt,
            last_known_bbox=prev_bbox if prev_bbox is not None else prompt,
            prev_lost=False,
            relative_frame=rel,
            previous_accepted_target_side=prev_accepted,
        )
        assert not lost
        canonical_history.append(canonical)
        emitted_history.append(emitted)
        prev_bbox = canonical
        if _diag is not None:
            prev_accepted = _diag.get("accepted_target_side")
    return canonical_history, emitted_history


class _FakePredictor:
    def __init__(self, stream_responses):
        self._stream_responses = stream_responses
        self.stream_requests = []
        self.handle_requests = []

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        yield from self._stream_responses

    def handle_request(self, request):
        self.handle_requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "fake-session"}
        return {}


def _preload_b64(image):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_two_frame_state_isolation_next_tracking_uses_canonical():
    """Adaptive emitted side differs, but frame-2 tracking input matches baseline."""
    seed = [88.0, 88.0, 112.0, 112.0]
    mask0 = _mask_at(100, 100, size=6)
    mask1 = _mask_at(102, 98, size=6)

    base_handler = _handler("mask_center_box")
    adapt_handler = _handler("adaptive_component_square_box")

    frame_image0 = _dark_frame()
    _draw_disk(frame_image0, 100, 100, radius=12, intensity=255)
    frame_image1 = _dark_frame()
    _draw_disk(frame_image1, 102, 98, radius=12, intensity=255)

    base0, lost0, _, can0 = base_handler._postprocess_frame_outputs(
        _sess(), frame_image0, {"out_binary_masks": [mask0]}, is_init_frame=True,
        prompt_bbox=seed, last_known_bbox=seed, prev_lost=False,
    )
    adapt0, lost_a0, diag0, can_a0 = adapt_handler._postprocess_frame_outputs(
        _sess(), frame_image0, {"out_binary_masks": [mask0]}, is_init_frame=True,
        prompt_bbox=seed, last_known_bbox=seed, prev_lost=False,
    )
    assert not lost0 and not lost_a0
    assert bbox_side_length(adapt0) != bbox_side_length(base0) or adapt0 != base0
    assert bbox_center(can_a0) == pytest.approx(bbox_center(can0), abs=1e-6)

    tracking_input_base = can0
    tracking_input_adapt = can_a0
    assert bbox_center(tracking_input_adapt) == pytest.approx(
        bbox_center(tracking_input_base), abs=1e-6
    )

    base1, _, _, can1_base = base_handler._postprocess_frame_outputs(
        _sess(), frame_image1, {"out_binary_masks": [mask1]}, is_init_frame=False,
        prompt_bbox=seed, last_known_bbox=tracking_input_base, prev_lost=False,
    )
    adapt1, _, diag1, can1_adapt = adapt_handler._postprocess_frame_outputs(
        _sess(), frame_image1, {"out_binary_masks": [mask1]}, is_init_frame=False,
        prompt_bbox=seed, last_known_bbox=tracking_input_adapt, prev_lost=False,
        previous_accepted_target_side=diag0.get("accepted_target_side"),
    )
    assert bbox_center(can1_adapt) == pytest.approx(bbox_center(can1_base), abs=1e-3)
    assert bbox_center(adapt1) == pytest.approx(bbox_center(base1), abs=1e-3)


def test_multi_frame_center_invariance_emitted_midpoints():
    masks = [_mask_at(100 + i, 100 - i, size=6) for i in range(5)]
    seed = [88.0, 88.0, 112.0, 112.0]
    _, base_emitted = _postprocess_sequence(_handler("mask_center_box"), masks, seed)
    _, adapt_emitted = _postprocess_sequence(
        _handler("adaptive_component_square_box"), masks, seed
    )
    for rel, (b, a) in enumerate(zip(base_emitted, adapt_emitted)):
        bc = bbox_center(b)
        ac = bbox_center(a)
        assert abs(bc[0] - ac[0]) <= 0.001, f"frame {rel} cx delta"
        assert abs(bc[1] - ac[1]) <= 0.001, f"frame {rel} cy delta"


def test_adaptive_side_changes_while_center_identical():
    seed = [90.0, 90.0, 110.0, 110.0]
    mask = _mask_at(100, 100, size=8)
    frame_image = _dark_frame()
    _draw_disk(frame_image, 100, 100, radius=14, intensity=255)

    base_handler = _handler("mask_center_box")
    adapt_handler = _handler("adaptive_component_square_box")

    base_bbox, _, _, _ = base_handler._postprocess_frame_outputs(
        _sess(), frame_image, {"out_binary_masks": [mask]}, is_init_frame=True,
        prompt_bbox=seed, last_known_bbox=seed, prev_lost=False,
    )
    adapt_bbox, _, diag, canonical = adapt_handler._postprocess_frame_outputs(
        _sess(), frame_image, {"out_binary_masks": [mask]}, is_init_frame=True,
        prompt_bbox=seed, last_known_bbox=seed, prev_lost=False,
    )
    assert bbox_center(adapt_bbox) == pytest.approx(bbox_center(base_bbox), abs=1e-6)
    assert bbox_side_length(adapt_bbox) != pytest.approx(bbox_side_length(base_bbox), abs=0.5)
    assert diag is not None
    assert diag["canonical_center_x"] == pytest.approx(diag["emitted_center_x"], abs=1e-6)


def test_edge_clipped_visible_midpoint_may_differ_canonical_unchanged():
    seed = [0.0, 0.0, 20.0, 20.0]
    center_bbox = [-19.0, -19.0, 21.0, 21.0]
    frame_image = _dark_frame(100, 100)
    handler = _handler("adaptive_component_square_box")
    handler._maybe_refine_bbox = lambda *_a, **_k: list(center_bbox)

    emitted, lost, diag, canonical = handler._postprocess_frame_outputs(
        {"image_height": 100, "image_width": 100},
        frame_image,
        {"out_binary_masks": [_mask_at(1, 1, size=4, frame_size=100)]},
        is_init_frame=True,
        prompt_bbox=seed,
        last_known_bbox=seed,
        prev_lost=False,
        previous_accepted_target_side=40.0,
    )
    assert not lost
    assert diag["edge_clipped"] is True
    assert bbox_center(canonical) == pytest.approx((1.0, 1.0), abs=1e-6)
    assert bbox_center(emitted) != pytest.approx(bbox_center(canonical), abs=0.1)


def test_fallback_does_not_contaminate_canonical_tracking_bbox():
    seed = [90.0, 90.0, 110.0, 110.0]
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    frame_image = _dark_frame()
    handler = _handler("adaptive_component_square_box")
    handler._maybe_refine_bbox = lambda *_a, **_k: list(center_bbox)

    emitted, lost, diag, canonical = handler._postprocess_frame_outputs(
        _sess(),
        frame_image,
        {"out_binary_masks": [_mask_at(100, 100)]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=center_bbox,
        prev_lost=False,
        previous_accepted_target_side=30.0,
    )
    assert not lost
    assert diag["fallback_reason"] == "no_valid_component"
    assert canonical == center_bbox
    assert bbox_center(canonical) == pytest.approx(bbox_center(center_bbox), abs=1e-6)


def test_loss_and_snap_back_semantics_unchanged_for_baseline():
    handler = _handler("mask_center_box")
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    frame_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[18:22, 18:22] = 1

    emitted, lost, diag, canonical = handler._postprocess_frame_outputs(
        _sess(),
        frame_image,
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=moved,
        prev_lost=False,
    )
    assert emitted is None
    assert lost is True
    assert canonical is None
    assert diag is None


def test_cache_stores_emitted_only_not_canonical_in_payload():
    preload_count = 2
    stream = [
        {"frame_index": idx, "outputs": {"out_binary_masks": [_mask_at(100, 100)]}}
        for idx in range(preload_count)
    ]
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(
        output_policy="adaptive_component_square_box",
        ir_refine_enabled=False,
    )
    handler.predictor = _FakePredictor(stream)
    handler._sessions = {}
    handler._maybe_refine_bbox = lambda _i, bbox, previous_bbox=None: bbox
    seed = [40.0, 40.0, 60.0, 60.0]

    image = _dark_frame(100, 100)
    _draw_disk(image, 50, 50, radius=10, intensity=255)
    preload_b64 = _preload_b64(image)
    preload_images = [preload_b64 for _ in range(preload_count)]

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
    cache = handler._sessions[key]["frame_cache"]
    for entry in cache.values():
        assert set(entry.keys()) == {"bbox", "lost"}
    assert states[0].get("accepted_target_side") is None
    assert states[0].get("canonical_tracking_bbox") is None


def test_loss_and_snap_back_semantics_unchanged_for_adaptive():
    handler = _handler("adaptive_component_square_box")
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    frame_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[18:22, 18:22] = 1

    emitted, lost, diag, canonical = handler._postprocess_frame_outputs(
        {"image_height": 100, "image_width": 100},
        frame_image,
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=moved,
        prev_lost=False,
    )
    assert emitted is None
    assert lost is True
    assert canonical is None
    assert diag is None


def test_baseline_canonical_equals_emitted():
    seed = [40.0, 40.0, 60.0, 60.0]
    handler = _handler("mask_center_box")
    frame_image = _dark_frame()
    _draw_disk(frame_image, 50, 50, radius=8, intensity=255)

    emitted, lost, diag, canonical = handler._postprocess_frame_outputs(
        _sess(),
        frame_image,
        {"out_binary_masks": [_mask_at(50, 50)]},
        is_init_frame=True,
        prompt_bbox=seed,
        last_known_bbox=seed,
        prev_lost=False,
    )
    assert not lost
    assert emitted == canonical
    assert diag is None
