"""Unit tests for SAM3 IR centroid refinement helpers (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_ir", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

load_sam3_config = _mod.load_sam3_config
parse_bool_env = _mod.parse_bool_env
recenter_bbox = _mod.recenter_bbox
refine_bbox_with_ir_intensity = _mod.refine_bbox_with_ir_intensity


def _dark_frame(height=200, width=200, background=20):
    image = np.full((height, width, 3), background, dtype=np.uint8)
    return image


def _draw_disk(image, center_x, center_y, radius, intensity=250):
    cv2.circle(image, (center_x, center_y), radius, (intensity, intensity, intensity), -1)


def _draw_rect(image, x1, y1, x2, y2, intensity=250):
    cv2.rectangle(image, (x1, y1), (x2, y2), (intensity, intensity, intensity), -1)


def test_parse_bool_env_truthy_and_invalid():
    assert parse_bool_env("1") is True
    assert parse_bool_env("true") is True
    assert parse_bool_env("yes") is True
    assert parse_bool_env("on") is True
    assert parse_bool_env("0") is False
    assert parse_bool_env("maybe", default=False) is False


def test_load_sam3_config_ir_refine_defaults_off():
    cfg = load_sam3_config({})
    assert cfg.ir_refine_enabled is False


def test_load_sam3_config_ir_refine_enabled():
    cfg = load_sam3_config({"SAM3_IR_REFINE": "1"})
    assert cfg.ir_refine_enabled is True


def test_recenter_bbox_preserves_width_and_height():
    bbox = recenter_bbox([90.0, 90.0, 110.0, 110.0], 105.0, 95.0, 200, 200)
    assert abs((bbox[2] - bbox[0]) - 20.0) < 1e-6
    assert abs((bbox[3] - bbox[1]) - 20.0) < 1e-6
    assert abs(((bbox[0] + bbox[2]) / 2.0) - 105.0) < 1e-6
    assert abs(((bbox[1] + bbox[3]) / 2.0) - 95.0) < 1e-6


def test_refine_moves_center_to_nearby_compact_ball():
    image = _dark_frame()
    _draw_disk(image, 108, 92, radius=4, intensity=255)
    sam_bbox = [90.0, 90.0, 110.0, 110.0]
    refined = refine_bbox_with_ir_intensity(image, sam_bbox)

    sam_cx = (sam_bbox[0] + sam_bbox[2]) / 2.0
    sam_cy = (sam_bbox[1] + sam_bbox[3]) / 2.0
    ref_cx = (refined[0] + refined[2]) / 2.0
    ref_cy = (refined[1] + refined[3]) / 2.0

    assert abs((refined[2] - refined[0]) - 20.0) < 1e-6
    assert abs((refined[3] - refined[1]) - 20.0) < 1e-6
    assert ((ref_cx - sam_cx) ** 2 + (ref_cy - sam_cy) ** 2) ** 0.5 > 1.0
    assert abs(ref_cx - 108.0) < 2.0
    assert abs(ref_cy - 92.0) < 2.0


def test_refine_centers_on_full_ball_not_specular_highlight():
    image = _dark_frame()
    ball_cx, ball_cy = 100, 100
    highlight_cx, highlight_cy = 92, 88
    _draw_disk(image, ball_cx, ball_cy, radius=12, intensity=75)
    _draw_disk(image, highlight_cx, highlight_cy, radius=5, intensity=255)
    sam_bbox = [88.0, 88.0, 108.0, 108.0]
    refined = refine_bbox_with_ir_intensity(image, sam_bbox)

    ref_cx = (refined[0] + refined[2]) / 2.0
    ref_cy = (refined[1] + refined[3]) / 2.0
    dist_to_ball = ((ref_cx - ball_cx) ** 2 + (ref_cy - ball_cy) ** 2) ** 0.5
    dist_to_highlight = (
        (ref_cx - highlight_cx) ** 2 + (ref_cy - highlight_cy) ** 2
    ) ** 0.5

    assert abs((refined[2] - refined[0]) - 20.0) < 1e-6
    assert abs((refined[3] - refined[1]) - 20.0) < 1e-6
    assert dist_to_ball < dist_to_highlight
    assert dist_to_ball < 4.0


def test_refine_rejects_large_whiteboard_like_blob():
    image = _dark_frame()
    _draw_rect(image, 40, 70, 160, 130, intensity=255)
    sam_bbox = [95.0, 95.0, 105.0, 105.0]
    refined = refine_bbox_with_ir_intensity(image, sam_bbox)
    assert refined == sam_bbox


def test_refine_rejects_far_bright_blob():
    image = _dark_frame()
    _draw_disk(image, 30, 30, radius=5, intensity=255)
    sam_bbox = [90.0, 90.0, 110.0, 110.0]
    refined = refine_bbox_with_ir_intensity(image, sam_bbox)
    assert refined == sam_bbox
