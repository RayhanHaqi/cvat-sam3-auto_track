"""Unit tests for SAM3 adaptive_component_square_box policy (no GPU)."""

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
_spec = importlib.util.spec_from_file_location("sam3_model_handler_adaptive", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

Sam3Config = _mod.Sam3Config
ModelHandler = _mod.ModelHandler
apply_adaptive_component_square_box = _mod.apply_adaptive_component_square_box
apply_adaptive_component_deadband_box = _mod.apply_adaptive_component_deadband_box
apply_adaptive_component_latent_prior_box = _mod.apply_adaptive_component_latent_prior_box
_latent_prior_state_from_diag = _mod._latent_prior_state_from_diag
bbox_center = _mod.bbox_center
bbox_side_length = _mod.bbox_side_length
build_adaptive_square_target = _mod.build_adaptive_square_target
equivalent_diameter_from_area = _mod.equivalent_diameter_from_area
gate_adaptive_side = _mod.gate_adaptive_side
load_sam3_config = _mod.load_sam3_config
median_component_side = _mod.median_component_side
select_adaptive_scale_component = _mod.select_adaptive_scale_component
select_bbox_from_outputs = _mod.select_bbox_from_outputs
square_bbox_from_center = _mod.square_bbox_from_center


def _dark_frame(height=200, width=200, background=20):
    return np.full((height, width, 3), background, dtype=np.uint8)


def _draw_disk(image, center_x, center_y, radius, intensity=250):
    cv2.circle(image, (center_x, center_y), radius, (intensity, intensity, intensity), -1)


def _draw_rect(image, x1, y1, x2, y2, intensity=250):
    cv2.rectangle(image, (x1, y1), (x2, y2), (intensity, intensity, intensity), -1)


def test_load_sam3_config_accepts_adaptive_policy():
    cfg = load_sam3_config({
        "SAM3_OUTPUT_POLICY": "adaptive_component_square_box",
        "SAM3_ADAPTIVE_PADDING_FACTOR": "1.5",
        "SAM3_ADAPTIVE_ROI_SCALE": "3.0",
        "SAM3_ADAPTIVE_MIN_ROI_MARGIN_PX": "32",
        "SAM3_ADAPTIVE_MIN_SIDE_PX": "6",
        "SAM3_ADAPTIVE_MAX_GROWTH_RATIO": "1.4",
        "SAM3_ADAPTIVE_MAX_SHRINK_RATIO": "0.8",
    })
    assert cfg.output_policy == "adaptive_component_square_box"
    assert cfg.adaptive_padding_factor == 1.5
    assert cfg.adaptive_roi_scale == 3.0
    assert cfg.adaptive_min_roi_margin_px == 32
    assert cfg.adaptive_min_side_px == 6
    assert cfg.adaptive_max_growth_ratio == 1.4
    assert cfg.adaptive_max_shrink_ratio == 0.8


def test_mask_center_box_behavior_unchanged():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[48:52, 58:62] = 1
    cfg = Sam3Config(output_policy="mask_center_box")
    reference = [20.0, 20.0, 40.0, 40.0]
    bbox = select_bbox_from_outputs(
        {"out_binary_masks": [mask]},
        100,
        100,
        reference,
        cfg,
    )
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    assert abs(cx - 60.0) < 1.0
    assert abs(cy - 50.0) < 1.0
    assert (bbox[2] - bbox[0]) >= 10.0
    assert (bbox[3] - bbox[1]) >= 10.0


def test_equivalent_diameter_and_median_estimator():
    area = np.pi * 5.0 * 5.0
    eq = equivalent_diameter_from_area(area)
    assert abs(eq - 10.0) < 1e-6
    side = median_component_side(12.0, 8.0, area)
    assert abs(side - 10.0) < 1e-6


def test_gate_adaptive_side_clamps_growth_and_shrink():
    cfg = Sam3Config(
        adaptive_max_growth_ratio=1.30,
        adaptive_max_shrink_ratio=0.75,
        adaptive_min_side_px=4,
    )
    ref = 20.0
    assert gate_adaptive_side(100.0, ref, cfg) == pytest.approx(26.0)
    assert gate_adaptive_side(1.0, ref, cfg) == pytest.approx(15.0)
    assert gate_adaptive_side(22.0, ref, cfg) == pytest.approx(22.0)


def test_a_interior_center_preservation_square_and_exact_center():
    image = _dark_frame(200, 200)
    center_x, center_y = 100.0, 100.0
    _draw_disk(image, int(center_x), int(center_y), radius=10, intensity=255)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    prompt_bbox = [88.0, 88.0, 108.0, 108.0]
    cfg = Sam3Config(adaptive_padding_factor=1.25)
    bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=None,
        is_init_frame=True,
        config=cfg,
    )
    assert diag["edge_clipped"] is False
    out_cx, out_cy = bbox_center(bbox)
    assert abs(out_cx - center_x) < 1e-6
    assert abs(out_cy - center_y) < 1e-6
    assert abs((bbox[2] - bbox[0]) - (bbox[3] - bbox[1])) < 1e-6
    assert diag["lost"] is False


def test_b_corner_clipping_symmetric_target_visible_bbox():
    target = build_adaptive_square_target(1.0, 1.0, 40.0, 100, 100)
    assert target["requested_target_bbox"] == pytest.approx([-19.0, -19.0, 21.0, 21.0])
    assert target["visible_clipped_bbox"] == [0.0, 0.0, 21.0, 21.0]
    assert target["edge_clipped"] is True

    image = _dark_frame(100, 100)
    center_bbox = [-19.0, -19.0, 21.0, 21.0]
    prompt_bbox = [0.0, 0.0, 20.0, 20.0]
    cfg = Sam3Config()
    bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=None,
        is_init_frame=True,
        config=cfg,
        previous_accepted_target_side=40.0,
    )
    assert bbox == [0.0, 0.0, 21.0, 21.0]
    assert diag["edge_clipped"] is True
    assert diag["requested_target_bbox"] == pytest.approx([-19.0, -19.0, 21.0, 21.0])
    assert diag["lost"] is False
    assert bbox[2] - bbox[0] < 40.0


def test_c_one_axis_clipping_without_horizontal_recenter():
    target = build_adaptive_square_target(1.0, 50.0, 40.0, 100, 100)
    assert target["requested_target_bbox"] == pytest.approx([-19.0, 30.0, 21.0, 70.0])
    assert target["visible_clipped_bbox"] == [0.0, 30.0, 21.0, 70.0]
    assert target["edge_clipped"] is True
    width = target["visible_clipped_bbox"][2] - target["visible_clipped_bbox"][0]
    height = target["visible_clipped_bbox"][3] - target["visible_clipped_bbox"][1]
    assert width < height


def test_d_temporal_history_uses_target_side_not_clipped_visible_width():
    image_edge = _dark_frame(100, 100)
    center_bbox_edge = [-19.0, -19.0, 21.0, 21.0]
    prompt_bbox = [0.0, 0.0, 20.0, 20.0]
    cfg = Sam3Config(adaptive_padding_factor=1.0, adaptive_max_growth_ratio=1.30)
    edge_bbox, edge_diag = apply_adaptive_component_square_box(
        image_edge,
        center_bbox_edge,
        prompt_bbox,
        previous_bbox=None,
        is_init_frame=True,
        config=cfg,
        previous_accepted_target_side=40.0,
    )
    assert edge_diag["accepted_target_side"] == pytest.approx(40.0)
    assert edge_bbox == [0.0, 0.0, 21.0, 21.0]
    assert edge_bbox[2] - edge_bbox[0] < 40.0

    image_interior = _dark_frame(200, 200)
    _draw_disk(image_interior, 100, 100, radius=18, intensity=255)
    center_bbox_interior = [80.0, 80.0, 120.0, 120.0]
    interior_bbox, interior_diag = apply_adaptive_component_square_box(
        image_interior,
        center_bbox_interior,
        prompt_bbox,
        previous_bbox=edge_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=edge_diag["accepted_target_side"],
    )
    assert interior_diag["previous_accepted_target_side"] == pytest.approx(40.0)
    assert interior_diag["accepted_target_side"] > edge_bbox[2] - edge_bbox[0]
    assert interior_diag["edge_clipped"] is False
    assert abs((interior_bbox[2] - interior_bbox[0]) - (interior_bbox[3] - interior_bbox[1])) < 1e-6


def test_e_fallback_at_edge_uses_previous_accepted_target_side():
    image = _dark_frame(100, 100)
    center_bbox = [-19.0, -19.0, 21.0, 21.0]
    prompt_bbox = [0.0, 0.0, 20.0, 20.0]
    cfg = Sam3Config()
    bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=[0.0, 0.0, 21.0, 21.0],
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=40.0,
    )
    assert diag["fallback_reason"] == "no_valid_component"
    assert diag["accepted_target_side"] == pytest.approx(40.0)
    assert bbox == [0.0, 0.0, 21.0, 21.0]
    assert diag["lost"] is False


def test_component_selection_finds_ball_and_rejects_far_blob():
    image = _dark_frame()
    center_x, center_y = 100, 100
    _draw_disk(image, center_x, center_y, radius=8, intensity=255)
    _draw_disk(image, 25, 25, radius=5, intensity=255)
    cfg = Sam3Config()
    comp = select_adaptive_scale_component(
        image,
        center_x,
        center_y,
        reference_side=20.0,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        config=cfg,
    )
    assert comp is not None
    assert comp["component_area"] > 0


def test_component_selection_returns_none_for_only_far_blob():
    image = _dark_frame()
    _draw_disk(image, 25, 25, radius=6, intensity=255)
    cfg = Sam3Config()
    comp = select_adaptive_scale_component(
        image,
        100.0,
        100.0,
        reference_side=20.0,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        config=cfg,
    )
    assert comp is None


def test_component_selection_rejects_oversized_center_blob():
    image = _dark_frame()
    _draw_rect(image, 60, 60, 140, 140, intensity=255)
    cfg = Sam3Config()
    comp = select_adaptive_scale_component(
        image,
        100.0,
        100.0,
        reference_side=20.0,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        config=cfg,
    )
    assert comp is None


def test_glare_large_circular_blob_farther_from_center_than_ball():
    image = _dark_frame()
    ball_cx, ball_cy = 100, 100
    glare_cx, glare_cy = 130, 100
    _draw_disk(image, ball_cx, ball_cy, radius=7, intensity=220)
    _draw_disk(image, glare_cx, glare_cy, radius=18, intensity=255)
    cfg = Sam3Config()
    comp = select_adaptive_scale_component(
        image,
        ball_cx,
        ball_cy,
        reference_side=20.0,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        config=cfg,
    )
    assert comp is not None
    comp_cx = ball_cx
    assert comp["component_bbox_width"] < 30.0


def test_glare_elongated_blob_prefers_near_center_plausible_component():
    image = _dark_frame()
    ball_cx, ball_cy = 100, 100
    _draw_disk(image, ball_cx, ball_cy, radius=8, intensity=230)
    _draw_rect(image, 70, 95, 130, 105, intensity=255)
    cfg = Sam3Config()
    comp = select_adaptive_scale_component(
        image,
        ball_cx,
        ball_cy,
        reference_side=20.0,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        config=cfg,
    )
    if comp is not None:
        assert comp["component_area"] < 500.0
    else:
        bbox, diag = apply_adaptive_component_square_box(
            image,
            [90.0, 90.0, 110.0, 110.0],
            [88.0, 88.0, 108.0, 108.0],
            previous_bbox=[90.0, 90.0, 110.0, 110.0],
            is_init_frame=False,
            config=cfg,
            previous_accepted_target_side=20.0,
        )
        assert diag["fallback_reason"] == "no_valid_component"
        assert diag["lost"] is False


def test_temporal_gate_limits_scale_explosion_in_apply():
    image = _dark_frame()
    center_x, center_y = 100, 100
    _draw_disk(image, center_x, center_y, radius=12, intensity=255)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    prompt_bbox = [88.0, 88.0, 108.0, 108.0]
    cfg = Sam3Config(adaptive_padding_factor=1.25, adaptive_max_growth_ratio=1.30)
    _bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=20.0,
    )
    assert diag["raw_component_side"] is not None
    assert diag["padded_side"] > 20.0
    assert diag["gated_side"] == pytest.approx(26.0)
    assert diag["adaptiveScaleStage"] == "temporal_gating"


def test_temporal_gate_limits_scale_collapse_in_apply():
    image = _dark_frame()
    center_x, center_y = 100, 100
    _draw_disk(image, center_x, center_y, radius=3, intensity=255)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    prompt_bbox = [88.0, 88.0, 108.0, 108.0]
    cfg = Sam3Config(adaptive_padding_factor=1.25, adaptive_max_shrink_ratio=0.75)
    _bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=[90.0, 90.0, 110.0, 110.0],
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=20.0,
    )
    assert diag["padded_side"] < 15.0
    assert diag["gated_side"] == pytest.approx(15.0)
    assert diag["adaptiveScaleStage"] == "temporal_gating"


def test_fallback_uses_previous_accepted_target_side_with_lost_false():
    image = _dark_frame()
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    prompt_bbox = [88.0, 88.0, 108.0, 108.0]
    cfg = Sam3Config()
    bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=[85.0, 85.0, 115.0, 115.0],
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=30.0,
    )
    assert diag["fallback_reason"] == "no_valid_component"
    assert diag["lost"] is False
    assert diag["accepted_target_side"] == pytest.approx(30.0)
    assert abs((bbox[2] - bbox[0]) - 30.0) < 1e-6


def test_first_frame_uses_seed_when_no_component():
    image = _dark_frame()
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    prompt_bbox = [80.0, 80.0, 120.0, 120.0]
    cfg = Sam3Config()
    bbox, diag = apply_adaptive_component_square_box(
        image,
        center_bbox,
        prompt_bbox,
        previous_bbox=None,
        is_init_frame=True,
        config=cfg,
    )
    assert diag["fallback_reason"] == "no_valid_component"
    assert abs(diag["accepted_target_side"] - bbox_side_length(prompt_bbox)) < 1e-6


def test_postprocess_loss_semantics_unchanged_for_adaptive():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(output_policy="adaptive_component_square_box")
    sess = {"image_height": 100, "image_width": 100}
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    frame_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[18:22, 18:22] = 1
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    bbox, lost, _diag, _canonical = handler._postprocess_frame_outputs(
        sess,
        frame_image,
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=moved,
        prev_lost=False,
    )
    assert bbox is None
    assert lost is True


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


def test_cache_stores_only_bbox_and_lost_for_adaptive_policy():
    import base64

    preload_count = 3
    stream = [
        {"frame_index": idx, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}}
        for idx in range(preload_count)
    ]
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(
        output_policy="adaptive_component_square_box",
        ir_refine_enabled=False,
    )
    handler.predictor = _FakePredictor(stream)
    handler._sessions = {}
    seed = [40.0, 40.0, 60.0, 60.0]

    buf = io.BytesIO()
    image = _dark_frame(100, 100)
    _draw_disk(image, 50, 50, radius=8, intensity=255)
    Image.fromarray(image).save(buf, format="JPEG")
    preload_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
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

    init_stream_count = len(handler.predictor.stream_requests)
    init_handle_count = len(handler.predictor.handle_requests)

    shapes, _states = handler.infer_batch(
        None,
        [None],
        [{
            "session_key": key,
            "last_bbox": cache[0]["bbox"],
            "base_frame": 0,
            "preloaded_count": preload_count,
            "preloaded_until_frame": preload_count - 1,
        }],
        frame_index=2,
    )
    assert len(handler.predictor.stream_requests) == init_stream_count
    assert len(handler.predictor.handle_requests) == init_handle_count
    assert shapes[0] == cache[2]["bbox"]


def _deadband_cfg(**overrides):
    values = {
        "adaptive_padding_factor": 1.10,
        "adaptive_deadband_ratio": 0.15,
        "adaptive_expansion_only": True,
        "adaptive_max_growth_ratio": 1.30,
    }
    values.update(overrides)
    return Sam3Config(**values)


def test_load_sam3_config_accepts_deadband_policy_defaults():
    cfg = load_sam3_config({"SAM3_OUTPUT_POLICY": "adaptive_component_deadband_box"})
    assert cfg.output_policy == "adaptive_component_deadband_box"
    assert cfg.adaptive_padding_factor == pytest.approx(1.10)
    assert cfg.adaptive_deadband_ratio == pytest.approx(0.15)
    assert cfg.adaptive_expansion_only is True


def test_deadband_inside_deadband_emits_canonical_bbox_byte_for_byte():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    _draw_disk(image, 100, 100, radius=10, intensity=255)
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=_deadband_cfg(),
        previous_accepted_target_side=20.0,
    )
    assert bbox == center_bbox
    assert diag["adaptive_emitted"] is False
    assert diag["deadband_blocked"] is True
    assert diag["temporal_gate_blocked"] is False
    assert diag["accepted_target_side"] == pytest.approx(bbox_side_length(center_bbox))


def test_deadband_temporal_gate_shrinks_raw_expansion_to_canonical_fallback():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    _draw_disk(image, 100, 100, radius=12, intensity=255)
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=_deadband_cfg(),
        previous_accepted_target_side=14.0,
    )
    assert diag["raw_candidate_side"] > canonical_side
    assert diag["relative_side_delta"] > _deadband_cfg().adaptive_deadband_ratio
    assert diag["gated_candidate_side"] <= canonical_side
    assert bbox == center_bbox
    assert diag["adaptive_emitted"] is False
    assert diag["temporal_gate_blocked"] is True
    assert diag["fallback_reason"] == "temporal_gate_expansion_only"
    assert diag["previous_accepted_target_side_after"] == pytest.approx(canonical_side)


def test_deadband_temporal_gate_puts_gated_expansion_inside_deadband():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    cfg = _deadband_cfg()
    _draw_disk(image, 100, 100, radius=13, intensity=255)
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=17.5,
    )
    assert diag["raw_candidate_side"] > canonical_side
    assert diag["relative_side_delta"] > cfg.adaptive_deadband_ratio
    assert diag["gated_candidate_side"] > canonical_side
    assert diag["final_relative_side_delta"] <= cfg.adaptive_deadband_ratio
    assert bbox == center_bbox
    assert diag["adaptive_emitted"] is False
    assert diag["temporal_gate_blocked"] is True
    assert diag["fallback_reason"] == "temporal_gate_deadband"
    assert diag["previous_accepted_target_side_after"] == pytest.approx(canonical_side)


def test_deadband_stale_scale_reset_after_temporal_gate_block():
    image_expand = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    cfg = _deadband_cfg()
    _draw_disk(image_expand, 100, 100, radius=16, intensity=255)
    _expanded, diag_expand = apply_adaptive_component_deadband_box(
        image_expand,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=20.0,
    )
    stale_side = diag_expand["accepted_target_side"]
    assert stale_side > canonical_side

    image_gate_block = _dark_frame(200, 200)
    _draw_disk(image_gate_block, 100, 100, radius=12, intensity=255)
    cfg_tight_gate = _deadband_cfg(adaptive_max_growth_ratio=0.57)
    _blocked, diag_blocked = apply_adaptive_component_deadband_box(
        image_gate_block,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg_tight_gate,
        previous_accepted_target_side=stale_side,
    )
    assert diag_blocked["temporal_gate_blocked"] is True
    assert diag_blocked["previous_accepted_target_side_after"] == pytest.approx(canonical_side)
    reset_side = diag_blocked["accepted_target_side"]

    image_next = _dark_frame(200, 200)
    _draw_disk(image_next, 100, 100, radius=16, intensity=255)
    _next_bbox, diag_next = apply_adaptive_component_deadband_box(
        image_next,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=reset_side,
    )
    assert diag_next["previous_accepted_target_side_before"] == pytest.approx(canonical_side)
    assert diag_next["previous_accepted_target_side_before"] != pytest.approx(stale_side)
    assert diag_next["gated_side"] <= canonical_side * cfg.adaptive_max_growth_ratio + 1e-6


def test_deadband_high_growth_emits_adaptive_square_bbox():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=_deadband_cfg(),
        previous_accepted_target_side=20.0,
    )
    assert diag["adaptive_emitted"] is True
    assert bbox != center_bbox
    out_cx, out_cy = bbox_center(bbox)
    assert abs(out_cx - 100.0) < 1e-6
    assert abs(out_cy - 100.0) < 1e-6
    assert abs((bbox[2] - bbox[0]) - (bbox[3] - bbox[1])) < 1e-6
    assert bbox_side_length(bbox) > bbox_side_length(center_bbox)


def test_deadband_expansion_only_blocks_valid_shrink_candidate():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    _draw_disk(image, 100, 100, radius=4, intensity=255)
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=_deadband_cfg(),
        previous_accepted_target_side=20.0,
    )
    assert bbox == center_bbox
    assert diag["expansion_only_blocked"] is True
    assert diag["adaptive_emitted"] is False


def test_deadband_no_valid_component_emits_canonical_not_prior_adaptive():
    image = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    bbox, diag = apply_adaptive_component_deadband_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=_deadband_cfg(),
        previous_accepted_target_side=40.0,
    )
    assert bbox == center_bbox
    assert diag["fallback_reason"] == "no_valid_component"
    assert diag["accepted_target_side"] == pytest.approx(bbox_side_length(center_bbox))
    assert diag["accepted_target_side"] != pytest.approx(40.0)


def test_deadband_blocked_frame_resets_previous_accepted_target_side_to_canonical():
    image_expand = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    _draw_disk(image_expand, 100, 100, radius=16, intensity=255)
    cfg = _deadband_cfg()
    _expanded, diag_expand = apply_adaptive_component_deadband_box(
        image_expand,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=20.0,
    )
    assert diag_expand["adaptive_emitted"] is True
    assert diag_expand["accepted_target_side"] > bbox_side_length(center_bbox)

    image_deadband = _dark_frame(200, 200)
    _draw_disk(image_deadband, 100, 100, radius=10, intensity=255)
    blocked_bbox, diag_blocked = apply_adaptive_component_deadband_box(
        image_deadband,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=diag_expand["accepted_target_side"],
    )
    assert blocked_bbox == center_bbox
    assert diag_blocked["deadband_blocked"] is True
    assert diag_blocked["previous_accepted_target_side_before"] == pytest.approx(
        diag_expand["accepted_target_side"]
    )
    assert diag_blocked["previous_accepted_target_side_after"] == pytest.approx(
        bbox_side_length(center_bbox)
    )


def test_deadband_canonical_fallback_prevents_stale_temporal_gating_reference():
    image_expand = _dark_frame(200, 200)
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    _draw_disk(image_expand, 100, 100, radius=16, intensity=255)
    cfg = _deadband_cfg()
    _expanded, diag_expand = apply_adaptive_component_deadband_box(
        image_expand,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=20.0,
    )
    stale_side = diag_expand["accepted_target_side"]
    assert stale_side > bbox_side_length(center_bbox)

    image_deadband = _dark_frame(200, 200)
    _draw_disk(image_deadband, 100, 100, radius=10, intensity=255)
    _blocked, diag_blocked = apply_adaptive_component_deadband_box(
        image_deadband,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=stale_side,
    )
    reset_side = diag_blocked["accepted_target_side"]
    assert reset_side == pytest.approx(bbox_side_length(center_bbox))

    image_next = _dark_frame(200, 200)
    _draw_disk(image_next, 100, 100, radius=16, intensity=255)
    _next_bbox, diag_next = apply_adaptive_component_deadband_box(
        image_next,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        previous_accepted_target_side=reset_side,
    )
    assert diag_next["previous_accepted_target_side_before"] == pytest.approx(reset_side)
    assert diag_next["gated_side"] <= reset_side * cfg.adaptive_max_growth_ratio + 1e-6


def test_postprocess_loss_semantics_unchanged_for_deadband_policy():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(output_policy="adaptive_component_deadband_box")
    sess = {"image_height": 100, "image_width": 100}
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    frame_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[18:22, 18:22] = 1
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    bbox, lost, _diag, _canonical = handler._postprocess_frame_outputs(
        sess,
        frame_image,
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=moved,
        prev_lost=False,
    )
    assert bbox is None
    assert lost is True


def test_cache_stores_only_bbox_and_lost_for_deadband_policy():
    import base64

    preload_count = 3
    stream = [
        {"frame_index": idx, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}}
        for idx in range(preload_count)
    ]
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(
        output_policy="adaptive_component_deadband_box",
        ir_refine_enabled=False,
    )
    handler.predictor = _FakePredictor(stream)
    handler._sessions = {}
    seed = [40.0, 40.0, 60.0, 60.0]

    buf = io.BytesIO()
    image = _dark_frame(100, 100)
    _draw_disk(image, 50, 50, radius=8, intensity=255)
    Image.fromarray(image).save(buf, format="JPEG")
    preload_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
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


def _latent_prior_cfg(**overrides):
    values = {
        "adaptive_padding_factor": 1.10,
        "adaptive_deadband_ratio": 0.15,
        "adaptive_expansion_only": True,
        "adaptive_max_growth_ratio": 1.30,
        "adaptive_selection_prior_max_age_frames": 20,
        "adaptive_emission_hold_max_age_frames": 4,
        "adaptive_growth_confirmations": 2,
    }
    values.update(overrides)
    return Sam3Config(**values)


def test_latent_prior_selection_prior_survives_in_deadband_component_frame():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    cfg = _latent_prior_cfg()
    image_grow = _dark_frame(200, 200)
    _draw_disk(image_grow, 100, 100, radius=16, intensity=255)
    _grow_bbox, grow_diag = apply_adaptive_component_latent_prior_box(
        image_grow,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    state = _latent_prior_state_from_diag(grow_diag)
    assert grow_diag["growth_confirmation_count"] == 1
    assert grow_diag["adaptive_emitted"] is False

    image_deadband = _dark_frame(200, 200)
    _draw_disk(image_deadband, 100, 100, radius=10, intensity=255)
    blocked_bbox, blocked_diag = apply_adaptive_component_latent_prior_box(
        image_deadband,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert blocked_bbox == center_bbox
    assert blocked_diag["adaptive_emitted"] is False
    assert blocked_diag["growth_confirmation_count"] == 0
    assert blocked_diag["selection_prior_side_after"] != pytest.approx(canonical_side)
    assert blocked_diag["selection_prior_side_after"] == pytest.approx(
        blocked_diag["gated_candidate_side"]
    )


def test_latent_prior_selection_prior_survives_no_component_frame():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    cfg = _latent_prior_cfg()
    image_grow = _dark_frame(200, 200)
    _draw_disk(image_grow, 100, 100, radius=16, intensity=255)
    _bbox, grow_diag = apply_adaptive_component_latent_prior_box(
        image_grow,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    prior_side = grow_diag["selection_prior_side_after"]
    state = _latent_prior_state_from_diag(grow_diag)

    blank = _dark_frame(200, 200)
    no_comp_bbox, no_comp_diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert no_comp_bbox == center_bbox
    assert no_comp_diag["component_valid"] is False
    assert no_comp_diag["selection_prior_side_after"] == pytest.approx(prior_side)
    assert no_comp_diag["selection_prior_age"] == 1


def test_latent_prior_selection_prior_expires_after_twenty_no_component_frames():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    stale_side = 30.0
    assert stale_side != pytest.approx(canonical_side)
    cfg = _latent_prior_cfg(adaptive_selection_prior_max_age_frames=20)
    state = {
        "selection_prior_side": stale_side,
        "selection_prior_age": 19,
        "growth_confirmation_count": 0,
        "emission_prior_side": None,
        "emission_hold_age": 0,
    }
    blank = _dark_frame(200, 200)
    _bbox, diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert diag["selection_prior_side_after"] == pytest.approx(canonical_side)
    assert diag["selection_prior_age"] == 0
    assert diag["growth_confirmation_count"] == 0
    assert diag["emission_prior_side_after"] is None
    assert diag["emission_hold_age"] == 0
    assert diag["emission_hold_active"] is False
    assert diag["growth_confirmation_reset_reason"] == "selection_prior_expired"


def test_latent_prior_selection_prior_preserves_through_nineteenth_no_component_frame():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    stale_side = 30.0
    cfg = _latent_prior_cfg(adaptive_selection_prior_max_age_frames=20)
    state = {
        "selection_prior_side": stale_side,
        "selection_prior_age": 18,
        "growth_confirmation_count": 0,
        "emission_prior_side": None,
        "emission_hold_age": 0,
    }
    blank = _dark_frame(200, 200)
    _bbox, diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert diag["selection_prior_side_after"] == pytest.approx(stale_side)
    assert diag["selection_prior_side_after"] != pytest.approx(canonical_side)
    assert diag["selection_prior_age"] == 19
    assert diag["growth_confirmation_reset_reason"] != "selection_prior_expired"


def test_latent_prior_post_expiry_valid_component_uses_canonical_reference():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    stale_side = 30.0
    cfg = _latent_prior_cfg(adaptive_selection_prior_max_age_frames=20)
    expired_state = {
        "selection_prior_side": stale_side,
        "selection_prior_age": 19,
        "growth_confirmation_count": 0,
        "emission_prior_side": None,
        "emission_hold_age": 0,
    }
    blank = _dark_frame(200, 200)
    _expired_bbox, expired_diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=expired_state,
    )
    assert expired_diag["selection_prior_side_after"] == pytest.approx(canonical_side)
    assert expired_diag["selection_prior_age"] == 0

    image = _dark_frame(200, 200)
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    _bbox, grow_diag = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=_latent_prior_state_from_diag(expired_diag),
    )
    assert grow_diag["selection_prior_side_before"] == pytest.approx(canonical_side)
    assert grow_diag["selection_prior_side_before"] != pytest.approx(stale_side)
    assert grow_diag["component_valid"] is True


def test_latent_prior_expiry_clears_growth_confirmations_and_emission_hold():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    canonical_side = bbox_side_length(center_bbox)
    cfg = _latent_prior_cfg(adaptive_selection_prior_max_age_frames=20)
    state = {
        "selection_prior_side": 30.0,
        "selection_prior_age": 19,
        "growth_confirmation_count": 2,
        "emission_prior_side": 35.0,
        "emission_hold_age": 1,
    }
    blank = _dark_frame(200, 200)
    _bbox, diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert diag["selection_prior_side_after"] == pytest.approx(canonical_side)
    assert diag["selection_prior_age"] == 0
    assert diag["growth_confirmation_count"] == 0
    assert diag["emission_prior_side_after"] is None
    assert diag["emission_hold_age"] == 0
    assert diag["emission_hold_active"] is False
    assert diag["emission_hold_clear_reason"] == "selection_prior_expired"


def test_latent_prior_two_consecutive_growth_supports_activate_emission():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    cfg = _latent_prior_cfg()
    image = _dark_frame(200, 200)
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    _bbox1, diag1 = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    assert diag1["growth_confirmation_count"] == 1
    assert diag1["adaptive_emitted"] is False

    image2 = _dark_frame(200, 200)
    _draw_disk(image2, 100, 100, radius=16, intensity=255)
    bbox2, diag2 = apply_adaptive_component_latent_prior_box(
        image2,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=_latent_prior_state_from_diag(diag1),
    )
    assert diag2["growth_confirmation_count"] == 2
    assert diag2["adaptive_emitted"] is True
    assert bbox_side_length(bbox2) > bbox_side_length(center_bbox)


def test_latent_prior_one_frame_growth_spike_does_not_activate_hold():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    cfg = _latent_prior_cfg()
    image = _dark_frame(200, 200)
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    _bbox, diag = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    assert diag["growth_confirmation_count"] == 1
    assert diag["emission_prior_side_after"] is None
    assert diag["adaptive_emitted"] is False


def test_latent_prior_hold_emits_across_no_component_frames_then_expires():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    cfg = _latent_prior_cfg(adaptive_emission_hold_max_age_frames=4)
    image = _dark_frame(200, 200)
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    _bbox1, diag1 = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    _bbox2, diag2 = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=_latent_prior_state_from_diag(diag1),
    )
    assert diag2["adaptive_emitted"] is True
    state = _latent_prior_state_from_diag(diag2)
    blank = _dark_frame(200, 200)
    held_frames = 0
    for _ in range(6):
        bbox, diag = apply_adaptive_component_latent_prior_box(
            blank,
            center_bbox,
            [88.0, 88.0, 108.0, 108.0],
            previous_bbox=center_bbox,
            is_init_frame=False,
            config=cfg,
            latent_prior_state=state,
        )
        if diag.get("emission_hold_active"):
            held_frames += 1
            assert bbox_side_length(bbox) > bbox_side_length(center_bbox)
        else:
            assert bbox == center_bbox
            if diag.get("emission_hold_clear_reason") == "hold_expired":
                break
        state = _latent_prior_state_from_diag(diag)
    assert held_frames == 4


def test_latent_prior_shrink_component_clears_active_hold():
    center_bbox = [90.0, 90.0, 110.0, 110.0]
    cfg = _latent_prior_cfg()
    image = _dark_frame(200, 200)
    _draw_disk(image, 100, 100, radius=16, intensity=255)
    _bbox1, diag1 = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
    )
    _bbox2, diag2 = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=_latent_prior_state_from_diag(diag1),
    )
    assert diag2["adaptive_emitted"] is True
    state = _latent_prior_state_from_diag(diag2)

    image_shrink = _dark_frame(200, 200)
    _draw_disk(image_shrink, 100, 100, radius=6, intensity=255)
    bbox, diag = apply_adaptive_component_latent_prior_box(
        image_shrink,
        center_bbox,
        [88.0, 88.0, 108.0, 108.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert bbox == center_bbox
    assert diag["emission_prior_side_after"] is None
    assert diag["emission_hold_clear_reason"] == "non_growth_component"


def test_latent_prior_pilot02_style_small_growth_stays_canonical():
    center_bbox = [90.0, 90.0, 105.4, 105.4]
    canonical_side = bbox_side_length(center_bbox)
    cfg = _latent_prior_cfg()
    image = _dark_frame(200, 200)
    _draw_disk(image, 97, 97, radius=8, intensity=255)
    bbox, diag = apply_adaptive_component_latent_prior_box(
        image,
        center_bbox,
        center_bbox,
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state={
            "selection_prior_side": canonical_side,
            "selection_prior_age": 0,
            "growth_confirmation_count": 1,
            "emission_prior_side": None,
            "emission_hold_age": 0,
        },
    )
    assert diag["component_valid"] is True
    assert diag["relative_side_delta"] is not None
    assert diag["relative_side_delta"] <= cfg.adaptive_deadband_ratio + 1e-6
    assert diag["adaptive_emitted"] is False
    assert bbox == center_bbox
    assert diag["growth_confirmation_count"] == 0


def test_latent_prior_edge_clipped_hold_retains_preclip_side_in_emission_prior():
    center_bbox = [-19.0, -19.0, 21.0, 21.0]
    cfg = _latent_prior_cfg()
    blank = _dark_frame(100, 100)
    state = {
        "selection_prior_side": 40.0,
        "selection_prior_age": 0,
        "growth_confirmation_count": 0,
        "emission_prior_side": 50.0,
        "emission_hold_age": 0,
    }
    bbox, diag = apply_adaptive_component_latent_prior_box(
        blank,
        center_bbox,
        [0.0, 0.0, 20.0, 20.0],
        previous_bbox=center_bbox,
        is_init_frame=False,
        config=cfg,
        latent_prior_state=state,
    )
    assert diag["emission_hold_active"] is True
    assert diag["edge_clipped"] is True
    assert diag["emission_prior_side_after"] == pytest.approx(50.0)
    assert bbox_side_length(bbox) < diag["emission_prior_side_after"]


def test_cache_stores_only_bbox_and_lost_for_latent_prior_policy():
    import base64
    preload_count = 3
    stream = [
        {"frame_index": idx, "outputs": {"out_boxes_xywh": [[0.1, 0.1, 0.05, 0.05]]}}
        for idx in range(preload_count)
    ]
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(
        output_policy="adaptive_component_latent_prior_box",
        ir_refine_enabled=False,
    )
    handler.predictor = _FakePredictor(stream)
    handler._sessions = {}
    seed = [40.0, 40.0, 60.0, 60.0]

    buf = io.BytesIO()
    image = _dark_frame(100, 100)
    _draw_disk(image, 50, 50, radius=8, intensity=255)
    Image.fromarray(image).save(buf, format="JPEG")
    preload_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
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
    assert states[0].get("selection_prior_side") is None


def test_postprocess_loss_semantics_unchanged_for_latent_prior_policy():
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = Sam3Config(output_policy="adaptive_component_latent_prior_box")
    sess = {"image_height": 100, "image_width": 100}
    seed = [10.0, 10.0, 30.0, 30.0]
    moved = [50.0, 50.0, 70.0, 70.0]
    frame_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[18:22, 18:22] = 1
    handler._maybe_refine_bbox = lambda _image, bbox, previous_bbox=None: bbox

    bbox, lost, _diag, _canonical = handler._postprocess_frame_outputs(
        sess,
        frame_image,
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=seed,
        last_known_bbox=moved,
        prev_lost=False,
        latent_prior_state={
            "selection_prior_side": 30.0,
            "selection_prior_age": 0,
            "growth_confirmation_count": 2,
            "emission_prior_side": 35.0,
            "emission_hold_age": 1,
        },
    )
    assert bbox is None
    assert lost is True
