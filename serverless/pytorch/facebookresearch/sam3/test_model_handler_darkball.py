"""Unit tests for the dark-ball calibration knobs (no GPU, no nuclio).

Covers SAM3_DETECTION_SCORE_THRESHOLD, SAM3_DETECTION_MAX_PROMPT_DISTANCE_PX,
SAM3_RECOVER_AFTER_MISS (+ SAM3_RECOVER_MAX_GAP gap interpolation),
SAM3_TRAILING_HOLD_FRAMES, SAM3_UI_GATE_CLAMP and SAM3_EDGE_HOLD.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_darkball", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)
ModelHandler = _mod.ModelHandler
Sam3Config = _mod.Sam3Config
load_sam3_config = _mod.load_sam3_config
select_bbox_from_outputs = _mod.select_bbox_from_outputs
EDGE_HOLD_MAX_DISP_SIDES = _mod.EDGE_HOLD_MAX_DISP_SIDES


def _mask_outputs(boxes, h=100, w=100):
    masks = []
    for x1, y1, x2, y2 in boxes:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        masks.append(mask)
    if not masks:
        return {
            "out_binary_masks": np.zeros((0, h, w), dtype=np.uint8),
            "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
        }
    return {
        "out_binary_masks": np.stack(masks),
        "out_boxes_xywh": np.zeros((len(masks), 4), dtype=np.float32),
    }


def _handler(config):
    handler = ModelHandler.__new__(ModelHandler)
    handler.config = config
    handler.predictor = SimpleNamespace(
        handle_request=lambda **_kwargs: {},
        handle_stream_request=lambda **_kwargs: iter(()),
    )
    handler._sessions = {}
    return handler


def _sess():
    return {
        "temp_dir": None,
        "image_height": 100,
        "image_width": 100,
        "base_frame": 0,
        "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
        "session_id": None,
    }


def test_load_sam3_config_darkball_defaults():
    cfg = load_sam3_config({})
    assert cfg.detection_score_threshold == 0.5
    assert cfg.detection_max_prompt_distance_px == 0.0
    assert cfg.recover_after_miss is False
    assert cfg.edge_hold_enabled is False


def test_load_sam3_config_darkball_env():
    cfg = load_sam3_config({
        "SAM3_DETECTION_SCORE_THRESHOLD": "0.35",
        "SAM3_DETECTION_MAX_PROMPT_DISTANCE_PX": "40",
        "SAM3_RECOVER_AFTER_MISS": "1",
        "SAM3_EDGE_HOLD": "true",
    })
    assert cfg.detection_score_threshold == 0.35
    assert cfg.detection_max_prompt_distance_px == 40.0
    assert cfg.recover_after_miss is True
    assert cfg.edge_hold_enabled is True


def test_load_sam3_config_darkball_invalid_threshold_keeps_default():
    cfg = load_sam3_config({"SAM3_DETECTION_SCORE_THRESHOLD": "not-a-number"})
    assert cfg.detection_score_threshold == 0.5


def test_apply_detection_score_threshold_sets_model():
    handler = _handler(Sam3Config(detection_score_threshold=0.35))
    handler.predictor = SimpleNamespace(model=SimpleNamespace(score_threshold_detection=0.5))
    handler._apply_detection_score_threshold()
    assert handler.predictor.model.score_threshold_detection == 0.35


def test_apply_detection_score_threshold_default_keeps_half():
    handler = _handler(Sam3Config())
    handler.predictor = SimpleNamespace(model=SimpleNamespace(score_threshold_detection=0.5))
    handler._apply_detection_score_threshold()
    assert handler.predictor.model.score_threshold_detection == 0.5


def test_apply_detection_score_threshold_tolerates_missing_attribute():
    handler = _handler(Sam3Config(detection_score_threshold=0.35))
    handler.predictor = SimpleNamespace()
    handler._apply_detection_score_threshold()
    handler.predictor = SimpleNamespace(model=SimpleNamespace())
    handler._apply_detection_score_threshold()


def test_select_bbox_distance_gate_off_picks_nearest():
    outputs = _mask_outputs([(42, 42, 52, 52), (12, 12, 22, 22)])
    reference = [10.0, 10.0, 30.0, 30.0]
    cfg = Sam3Config(output_policy="mask_center_box")
    bbox = select_bbox_from_outputs(outputs, 100, 100, reference, cfg)
    assert bbox is not None
    cx, cy = _mod.bbox_center(bbox)
    assert abs(cx - 17.0) < 1.0 and abs(cy - 17.0) < 1.0


def test_select_bbox_distance_gate_rejects_far_only():
    outputs = _mask_outputs([(70, 70, 80, 80)])
    reference = [10.0, 10.0, 30.0, 30.0]
    cfg = Sam3Config(output_policy="mask_center_box", detection_max_prompt_distance_px=20.0)
    assert select_bbox_from_outputs(outputs, 100, 100, reference, cfg) is None


def test_select_bbox_distance_gate_keeps_near_candidate():
    outputs = _mask_outputs([(70, 70, 80, 80), (12, 12, 22, 22)])
    reference = [10.0, 10.0, 30.0, 30.0]
    cfg = Sam3Config(output_policy="mask_center_box", detection_max_prompt_distance_px=20.0)
    bbox = select_bbox_from_outputs(outputs, 100, 100, reference, cfg)
    assert bbox is not None
    cx, cy = _mod.bbox_center(bbox)
    assert abs(cx - 17.0) < 1.0 and abs(cy - 17.0) < 1.0


def test_recover_after_miss_accepts_frame_with_output():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    sess = _sess()
    handler = _handler(Sam3Config(ir_refine_enabled=False, recover_after_miss=True))
    emitted, lost, _diag, canonical = handler._postprocess_frame_outputs(
        sess,
        np.zeros((100, 100, 3), dtype=np.uint8),
        outputs,
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[10.0, 10.0, 30.0, 30.0],
        prev_lost=True,
    )
    assert lost is False
    assert emitted is not None
    assert canonical is not None


def test_recover_after_miss_off_keeps_sticky_loss():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    sess = _sess()
    handler = _handler(Sam3Config(ir_refine_enabled=False))
    emitted, lost, _diag, _canonical = handler._postprocess_frame_outputs(
        sess,
        np.zeros((100, 100, 3), dtype=np.uint8),
        outputs,
        is_init_frame=False,
        prompt_bbox=[10.0, 10.0, 30.0, 30.0],
        last_known_bbox=[10.0, 10.0, 30.0, 30.0],
        prev_lost=True,
    )
    assert emitted is None
    assert lost is True


def test_recover_after_miss_cache_reacquires():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(Sam3Config(ir_refine_enabled=False, recover_after_miss=True))
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: None,
        2: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 3, [10.0, 10.0, 30.0, 30.0])
    assert cache[1]["lost"] is False and cache[1]["interpolated"] is True
    assert cache[2]["lost"] is False and cache[2]["bbox"] is not None


def test_recover_after_miss_cache_sticky_by_default():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(Sam3Config(ir_refine_enabled=False))
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: None,
        2: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 3, [10.0, 10.0, 30.0, 30.0])
    assert cache[1]["lost"] is True
    assert cache[2]["lost"] is True and cache[2]["bbox"] is None


def test_edge_hold_holds_far_border_box():
    handler = _handler(Sam3Config(edge_hold_enabled=True))
    sess = _sess()
    reference = [40.0, 40.0, 50.0, 50.0]
    far = [0.0, 10.0, 10.0, 20.0]
    held = handler._maybe_apply_edge_hold(far, reference, sess)
    assert held == reference


def test_edge_hold_keeps_close_border_box():
    handler = _handler(Sam3Config(edge_hold_enabled=True))
    sess = _sess()
    reference = [40.0, 5.0, 50.0, 15.0]
    close = [0.0, 5.0, 10.0, 15.0]
    assert handler._maybe_apply_edge_hold(close, reference, sess) == close


def test_edge_hold_ignores_non_border_box():
    handler = _handler(Sam3Config(edge_hold_enabled=True))
    sess = _sess()
    reference = [10.0, 10.0, 20.0, 20.0]
    interior = [60.0, 60.0, 70.0, 70.0]
    assert handler._maybe_apply_edge_hold(interior, reference, sess) == interior


def test_edge_hold_off_is_noop():
    handler = _handler(Sam3Config())
    sess = _sess()
    reference = [40.0, 40.0, 50.0, 50.0]
    far = [0.0, 10.0, 10.0, 20.0]
    assert handler._maybe_apply_edge_hold(far, reference, sess) == far


def test_edge_hold_threshold_constant_sane():
    assert 3.0 <= EDGE_HOLD_MAX_DISP_SIDES <= 5.0


def test_ui_gate_clamp_limits_center_jump():
    handler = _handler(Sam3Config(ui_gate_clamp=True))
    sess = _sess()
    sess["ui_emit_window"] = [[40.0, 40.0, 50.0, 50.0]]
    sess["ui_emit_prev_center"] = (45.0, 45.0)
    out = handler._clamp_emitted_to_ui_gate(
        sess, [90.0, 90.0, 100.0, 100.0], [40.0, 40.0, 50.0, 50.0], False
    )
    cx, cy = _mod.bbox_center(out)
    diagonal = _mod.bbox_side_length([40.0, 40.0, 50.0, 50.0]) * (2.0 ** 0.5)
    distance = ((cx - 45.0) ** 2 + (cy - 45.0) ** 2) ** 0.5
    assert distance <= diagonal * 2.4 + 1e-6
    assert out[0] >= 0.0 and out[2] <= 100.0


def test_ui_gate_clamp_limits_area_spike():
    handler = _handler(Sam3Config(ui_gate_clamp=True))
    sess = _sess()
    sess["ui_emit_window"] = [[40.0, 40.0, 50.0, 50.0]]
    sess["ui_emit_prev_center"] = (45.0, 45.0)
    out = handler._clamp_emitted_to_ui_gate(
        sess, [40.0, 40.0, 70.0, 70.0], [40.0, 40.0, 50.0, 50.0], False
    )
    side = _mod.bbox_side_length(out)
    assert side <= (2.4 * 100.0) ** 0.5 + 1e-6


def test_ui_gate_clamp_seeds_window_with_prompt_at_init():
    handler = _handler(Sam3Config(ui_gate_clamp=True))
    sess = _sess()
    out = handler._clamp_emitted_to_ui_gate(
        sess, [10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0], True
    )
    assert list(out) == [10.0, 10.0, 20.0, 20.0]
    assert sess["ui_emit_window"] == [[10.0, 10.0, 20.0, 20.0]]


def test_trailing_hold_applies_only_at_chunk_end():
    handler = _handler(Sam3Config(recover_after_miss=True, trailing_hold_frames=2))
    bbox = [10.0, 10.0, 20.0, 20.0]
    assert handler._should_hold_trailing_frame(96, 96, bbox) is True
    assert handler._should_hold_trailing_frame(94, 96, bbox) is True
    assert handler._should_hold_trailing_frame(93, 96, bbox) is False
    assert handler._should_hold_trailing_frame(96, 96, None) is False


def test_trailing_hold_off_by_default():
    handler = _handler(Sam3Config(recover_after_miss=True))
    assert handler._should_hold_trailing_frame(96, 96, [0.0, 0.0, 4.0, 4.0]) is False


def test_interpolates_short_gap_without_loss_marker():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(
        Sam3Config(
            ir_refine_enabled=False,
            recover_after_miss=True,
            recover_max_gap=2,
        )
    )
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: None,
        2: None,
        3: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 4, [10.0, 10.0, 30.0, 30.0])
    assert cache[1]["lost"] is False and cache[1]["interpolated"] is True
    assert cache[2]["lost"] is False and cache[2]["interpolated"] is True
    assert cache[3]["lost"] is False and "interpolated" not in cache[3]


def test_long_gap_keeps_loss():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(
        Sam3Config(
            ir_refine_enabled=False,
            recover_after_miss=True,
            recover_max_gap=2,
        )
    )
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: None,
        2: None,
        3: None,
        4: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 5, [10.0, 10.0, 30.0, 30.0])
    assert cache[1]["lost"] is True
    assert cache[2]["lost"] is True
    assert cache[3]["lost"] is True
    assert cache[4]["lost"] is False


def test_interpolation_off_without_recover():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(Sam3Config(ir_refine_enabled=False, recover_max_gap=2))
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: None,
        2: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 3, [10.0, 10.0, 30.0, 30.0])
    assert cache[1]["lost"] is True
    assert "interpolated" not in cache[1]


def test_ui_gate_clamp_caps_spike_and_keeps_verdict_valid():
    handler = _handler(Sam3Config(ui_gate_clamp=True))
    prompt = [100.0, 100.0, 104.0, 104.0]
    cache = {}
    for frame in range(1, 10):
        box = [100.0 + frame, 100.0, 104.0 + frame, 104.0]
        cache[frame] = {"bbox": box, "lost": False}
    cache[10] = {"bbox": [110.0, 100.0, 126.0, 116.0], "lost": False}
    sess = {"image_height": 480, "image_width": 848}
    capped = handler._apply_ui_gate_clamp(cache, 11, sess, prompt)
    assert capped == 1
    side = cache[10]["bbox"][2] - cache[10]["bbox"][0]
    assert side < 16.0
    verdict = handler._ui_gate_verdict(
        cache[10]["bbox"], sess["ui_emit_window"], sess["ui_emit_prev_center"]
    )
    assert verdict in ("valid", "soft")


def test_ui_gate_clamp_noop_when_disabled():
    handler = _handler(Sam3Config(ui_gate_clamp=False))
    prompt = [100.0, 100.0, 104.0, 104.0]
    cache = {1: {"bbox": [100.0, 100.0, 104.0, 104.0], "lost": False}}
    sess = {"image_height": 480, "image_width": 848}
    assert handler._apply_ui_gate_clamp(cache, 2, sess, prompt) == 0
    assert cache[1]["bbox"] == [100.0, 100.0, 104.0, 104.0]


def test_reseed_after_long_miss_repropagates():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(
        Sam3Config(
            ir_refine_enabled=False,
            recover_after_miss=True,
            recover_max_gap=2,
            reseed_after_miss=True,
        )
    )
    sess = _sess()
    outputs_by_frame = {
        0: outputs,
        1: outputs,
        2: None,
        3: None,
        4: None,
        5: outputs,
    }
    calls = []

    def _record_prompt(sess_arg, index, bbox=None):
        calls.append((int(index), list(bbox) if bbox is not None else None))

    handler._add_prompt = _record_prompt
    handler._propagate_range = lambda sess_arg, start, end: {
        4: outputs,
        5: outputs,
    }
    reseeds = handler._reseed_after_long_miss(sess, outputs_by_frame, 6)
    assert reseeds == 1
    assert calls == [(2, [9.5, 9.5, 29.5, 29.5])]
    assert outputs_by_frame[4] is outputs


def test_reseed_off_by_default():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(Sam3Config(ir_refine_enabled=False, recover_after_miss=True))
    sess = _sess()
    outputs_by_frame = {1: None, 2: None, 3: None, 4: outputs}
    calls = []
    handler._add_prompt = lambda sess_arg, index, bbox=None: calls.append(index)
    handler._propagate_range = lambda sess_arg, start, end: {1: outputs}
    assert handler._reseed_after_long_miss(sess, outputs_by_frame, 5) == 0
    assert calls == []
    assert outputs_by_frame[1] is None


def test_reseed_cache_fills_long_gap():
    outputs = _mask_outputs([(10, 10, 30, 30)])
    handler = _handler(
        Sam3Config(
            ir_refine_enabled=False,
            recover_after_miss=True,
            recover_max_gap=2,
            reseed_after_miss=True,
        )
    )
    handler._save_preload_sequence = lambda sess, preload, count: None
    handler._start_session = lambda sess: "session"
    handler._add_prompt = lambda sess, index, bbox=None: None
    handler._propagate_full_chunk = lambda sess, count: {
        0: outputs,
        1: outputs,
        2: None,
        3: None,
        4: None,
        5: None,
        6: outputs,
    }
    handler._propagate_range = lambda sess, start, end: {
        2: outputs,
        3: outputs,
        4: outputs,
        5: outputs,
    }
    handler._load_preloaded_frame = lambda sess, index: np.zeros(
        (100, 100, 3), dtype=np.uint8
    )
    handler._cleanup_session_workspace = lambda sess: None
    sess = _sess()
    cache = handler._build_frame_cache(sess, [""], 7, [10.0, 10.0, 30.0, 30.0])
    assert all(cache[frame]["lost"] is False for frame in range(7))


def _rod_mask(x0=40, y0=20, width=2, height=8, size=100):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y0 + height, x0:x0 + width] = 1
    return mask


def test_scaled_prompt_bbox_shrinks_around_center():
    handler = _handler(Sam3Config(prompt_box_scale=0.5))
    sess = {"image_height": 100, "image_width": 100}
    scaled = handler._scaled_prompt_bbox([10.0, 10.0, 30.0, 30.0], sess)
    assert scaled == [15.0, 15.0, 25.0, 25.0]
    plain = _handler(Sam3Config())
    assert plain._scaled_prompt_bbox([10.0, 10.0, 30.0, 30.0], sess) == [
        10.0, 10.0, 30.0, 30.0,
    ]


def test_add_prompt_uses_scaled_box():
    handler = _handler(Sam3Config(prompt_box_scale=0.5))
    seen = {}
    handler.predictor = SimpleNamespace(
        handle_request=lambda request: seen.update(request) or {},
        handle_stream_request=lambda **_kwargs: iter(()),
    )
    sess = {"session_id": "s", "image_height": 100, "image_width": 100,
            "prompt_bbox": [10.0, 10.0, 30.0, 30.0]}
    handler._add_prompt(sess, 0)
    rel = seen["bounding_boxes"][0]
    assert abs(rel[0] - 0.15) < 1e-6 and abs(rel[2] - 0.10) < 1e-6


def test_rod_mask_head_center_detects_thin_vertical_strip():
    handler = _handler(Sam3Config(rod_mask_anchor=True))
    strip = _rod_mask()
    anchor = handler._rod_mask_head_center(strip)
    assert anchor is not None
    assert abs(anchor[0] - 40.5) < 1e-6 and abs(anchor[1] - 20.5) < 1e-6
    round_mask = np.zeros((100, 100), dtype=np.uint8)
    round_mask[20:25, 40:45] = 1
    assert handler._rod_mask_head_center(round_mask) is None


def test_rod_mask_head_center_detects_lollipop():
    handler = _handler(Sam3Config(rod_mask_anchor=True))
    lollipop = np.zeros((100, 100), dtype=np.uint8)
    lollipop[20:25, 38:44] = 1  # ball head
    lollipop[25:35, 40:42] = 1  # rod tail
    anchor = handler._rod_mask_head_center(lollipop)
    assert anchor is not None
    assert abs(anchor[0] - 40.5) < 1.0
    assert abs(anchor[1] - 20.5) < 1e-6
    wide_strip = np.zeros((100, 100), dtype=np.uint8)
    wide_strip[20:34, 40:46] = 1  # uniform 6px strip: not a lollipop
    assert handler._rod_mask_head_center(wide_strip) is None


def test_rod_mask_anchor_moves_emitted_box_up():
    strip = _rod_mask()
    outputs = {"out_binary_masks": np.stack([strip])}
    reference = [36.0, 16.0, 46.0, 26.0]
    plain = _handler(Sam3Config(output_policy="motion_blur_center_box"))
    plain_box, _ = plain._motion_blur_target_bbox(
        {"image_height": 100, "image_width": 100},
        np.zeros((100, 100, 3), dtype=np.uint8),
        outputs,
        reference,
        None,
        None,
        relative_frame=1,
    )
    anchored = _handler(
        Sam3Config(output_policy="motion_blur_center_box", rod_mask_anchor=True)
    )
    anchored_box, _ = anchored._motion_blur_target_bbox(
        {"image_height": 100, "image_width": 100},
        np.zeros((100, 100, 3), dtype=np.uint8),
        outputs,
        reference,
        None,
        None,
        relative_frame=1,
    )
    plain_cy = (plain_box[1] + plain_box[3]) / 2.0
    anchored_cy = (anchored_box[1] + anchored_box[3]) / 2.0
    assert anchored_cy < plain_cy - 2.0
    assert abs(anchored_cy - 20.5) < 1.0


def _blob_scene():
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    image[20:26, 39:42] = 70  # ball above the rod tip
    image[30:60, 40:42] = 80  # rod below
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:38, 40:42] = 1  # the tracker's rod strip
    return image, mask


def test_dark_blob_above_mask_finds_ball():
    handler = _handler(Sam3Config(rod_mask_blob_reanchor=True))
    image, mask = _blob_scene()
    blob = handler._dark_blob_above_mask(image, mask)
    assert blob is not None
    assert abs(blob[0] - 40.0) < 1.0
    assert 21.0 < blob[1] < 24.0
    plain = np.full((100, 100, 3), 120, dtype=np.uint8)
    assert handler._dark_blob_above_mask(plain, mask) is None


def test_rod_blob_reanchor_moves_emitted_box_to_ball():
    image, mask = _blob_scene()
    outputs = {"out_binary_masks": np.stack([mask])}
    reference = [36.0, 26.0, 46.0, 36.0]
    handler = _handler(
        Sam3Config(
            output_policy="motion_blur_center_box", rod_mask_blob_reanchor=True
        )
    )
    box, diag = handler._motion_blur_target_bbox(
        {"image_height": 100, "image_width": 100},
        image,
        outputs,
        reference,
        None,
        None,
        relative_frame=5,
    )
    assert diag.get("rod_blob_reanchor") is True
    assert 21.0 < (box[1] + box[3]) / 2.0 < 24.0
