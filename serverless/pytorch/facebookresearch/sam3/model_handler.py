import base64
import io
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch
from PIL import Image

from diagnostics import bbox_summary, enabled as diag_enabled, log_frame_record, summarize_outputs
from motion_blur import (
    MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT,
    MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT,
    MOTION_BLUR_TARGET_FRACTION_DEFAULT,
    estimate_motion_blur_center,
    intensity_weighted_centroid,
    mask_centroid,
    mask_points,
    median_sharp_diameter,
    motion_blur_report_row,
)
from sam3.model_builder import build_sam3_predictor

MAX_SESSIONS = 32
SAM3_PRELOAD_CHUNK_CAP = 128
VALID_PROMPT_MODES = ("box", "text_box", "text")
VALID_OUTPUT_POLICIES = (
    "sam_box",
    "mask_bbox",
    "mask_center_box",
    "adaptive_component_square_box",
    "adaptive_component_deadband_box",
    "adaptive_component_latent_prior_box",
    "motion_blur_center_box",
)
ADAPTIVE_COMPONENT_POLICIES = frozenset({
    "adaptive_component_square_box",
    "adaptive_component_deadband_box",
    "adaptive_component_latent_prior_box",
})
# Experimental calibration defaults for adaptive_component_square_box (v1).
ADAPTIVE_PADDING_FACTOR = 1.25
# Experimental calibration defaults for adaptive_component_deadband_box (v1).
ADAPTIVE_DEADBAND_PADDING_FACTOR = 1.10
ADAPTIVE_DEADBAND_RATIO = 0.15
ADAPTIVE_EXPANSION_ONLY_DEFAULT = True
# Experimental calibration defaults for adaptive_component_latent_prior_box (v1).
LATENT_PRIOR_SELECTION_PRIOR_MAX_AGE_FRAMES = 20
LATENT_PRIOR_EMISSION_HOLD_MAX_AGE_FRAMES = 4
LATENT_PRIOR_GROWTH_CONFIRMATIONS = 2
ADAPTIVE_ROI_SCALE = 2.5
ADAPTIVE_MIN_ROI_MARGIN_PX = 24
ADAPTIVE_MIN_SIDE_PX = 4
ADAPTIVE_MAX_GROWTH_RATIO = 1.30
ADAPTIVE_MAX_SHRINK_RATIO = 0.75
DEFAULT_TEXT_PROMPT = "ping pong ball"
DEFAULT_OUTPUT_POLICY = "motion_blur_center_box"
DEFAULT_OUTPUT_PROB_THRESH = 0.25
# SAM3 video model hard-codes the VG detection gate to 0.5; this is the
# handler override for the dark-ball calibration.
DEFAULT_DETECTION_SCORE_THRESHOLD = 0.5
# Maximum emitted-centre displacement (in box sides) that the edge hold
# tolerates before it holds the last accepted box.
EDGE_HOLD_MAX_DISP_SIDES = 4.0
# UI ball-geometry staging envelope (mirror of AUTO_TRACK_BALL_VALIDATION_CONFIG
# soft limits). Emitted boxes are clamped inside these so the rally segment is
# not discarded by the frontend gate on launch-acceleration or frame-exit
# frames. Margins are kept inside the soft thresholds.
UI_GATE_CLAMP_MAX_AREA_RATIO = 2.4
UI_GATE_CLAMP_MIN_AREA_RATIO = 0.4
UI_GATE_CLAMP_MAX_DISP_DIAGONALS = 2.4
UI_GATE_CLAMP_MAX_VELOCITY_RESIDUAL = 3.5
UI_GATE_EMIT_WINDOW = 8
# Frontend rally staging envelope (mirrored): soft limits count violations,
# hard limits discard the segment; two soft signals in one frame are hard.
UI_GATE_SOFT_AREA_MIN_RATIO = 0.35
UI_GATE_SOFT_AREA_MAX_RATIO = 2.5
UI_GATE_HARD_AREA_MAX_RATIO = 4.0
UI_GATE_HARD_AREA_MIN_RATIO = UI_GATE_SOFT_AREA_MIN_RATIO / 2.0
UI_GATE_SOFT_MAX_DISP_SCALE = 2.5
UI_GATE_HARD_MAX_DISP_SCALE = 5.0
UI_GATE_MAX_VELOCITY_RESIDUAL = 4.0
MIN_BOX_SIDE_PX = 4
ROD_BLOB_MIN_PIXELS = 3
ROD_BLOB_MAX_PIXELS = 150
IR_REFINE_BRIGHT_PERCENTILE = 88
IR_REFINE_ROI_SCALE = 1.5
IR_REFINE_MIN_ROI_MARGIN_PX = 24
IR_REFINE_MAX_DISP_RATIO = 0.6
IR_REFINE_MIN_AREA_RATIO = 0.02
IR_REFINE_MAX_AREA_RATIO = 3.0
IR_REFINE_MIN_CIRCULARITY = 0.25
IR_REFINE_MAX_ASPECT_RATIO = 2.5
IR_REFINE_MAX_TEMPORAL_DISP_RATIO = 2.5
# A false relock must snap back onto the prompt while the accepted box has moved
# away from it. Judged relative to the box side: a 2px corner test condemns a
# hand-drawn prompt on a 7px ball, where 3px is ordinary motion.
FALSE_RELOCK_MAX_DISP_RATIO = 1.0
IR_REFINE_SUPPORT_CONTRAST_FLOOR = 8.0
IR_REFINE_MIN_SEED_PIXELS = 3


@dataclass(frozen=True)
class Sam3Config:
    prompt_mode: str = "box"
    text_prompt: str = DEFAULT_TEXT_PROMPT
    output_policy: str = DEFAULT_OUTPUT_POLICY
    output_prob_thresh: float = DEFAULT_OUTPUT_PROB_THRESH
    # Dark-ball calibration: SAM3's VG detector gate is hard-coded to 0.5 in
    # the image (model_builder.build_sam3_video_model). Lowering it lets the
    # low-contrast dark ball on the launcher post be detected at the seed frame.
    detection_score_threshold: float = DEFAULT_DETECTION_SCORE_THRESHOLD
    # Optional fail-closed distance gate for candidate selection (0 disables).
    detection_max_prompt_distance_px: float = 0.0
    # Re-acquire after a single-frame miss instead of sticky loss.
    recover_after_miss: bool = False
    # Fail-soft near image borders: hold the last accepted box when an
    # edge-clamped box would jump far from it.
    edge_hold_enabled: bool = False
    # Clamp emitted boxes inside the UI geometry envelope so the rally segment
    # is not discarded by the frontend gate.
    ui_gate_clamp: bool = False
    # Hold the last known box for a miss in the last N preloaded frames (no
    # future frame can re-acquire there; avoids discarding the whole segment).
    trailing_hold_frames: int = 0
    # Fill up to N consecutive missing frames with linearly interpolated
    # boxes (the UI stops the segment on a lost frame and needs a box on every
    # frame). Longer gaps keep today's loss behaviour.
    recover_max_gap: int = 2
    # Re-prompt the tracker with the last accepted box when the model loses the
    # object for more than recover_max_gap frames, then re-propagate from the
    # first missed frame (model output instead of interpolated boxes).
    reseed_after_miss: bool = False
    reseed_max_attempts: int = 2
    # Scale applied to the SAM3 prompt box around its centre (1.0 = as drawn).
    # < 1.0 trims the dark rod/pole a resting ball sits on out of the prompt.
    prompt_box_scale: float = 1.0
    # Rod-like mask guard: a thin vertical mask (post/rod) is emitted at its
    # top edge (where the ball sits) instead of the strip centroid.
    rod_mask_anchor: bool = False
    rod_mask_max_width: int = 3
    rod_mask_min_aspect: float = 2.0
    # Rod-like mask: re-anchor the emitted centre on the dark blob just above
    # the rod tip (the ball that left the launcher) instead of the strip.
    rod_mask_blob_reanchor: bool = False
    rod_blob_search_px: int = 24
    rod_blob_margin: float = 15.0
    # Lollipop masks (wider ball head on a thin rod tail) are anchored at the
    # head; the head is >= this multiple of the tail width.
    rod_mask_lollipop_ratio: float = 2.0
    ir_refine_enabled: bool = False
    ir_stop_on_missing: bool = False
    adaptive_padding_factor: float = ADAPTIVE_PADDING_FACTOR
    adaptive_roi_scale: float = ADAPTIVE_ROI_SCALE
    adaptive_min_roi_margin_px: int = ADAPTIVE_MIN_ROI_MARGIN_PX
    adaptive_min_side_px: int = ADAPTIVE_MIN_SIDE_PX
    adaptive_max_growth_ratio: float = ADAPTIVE_MAX_GROWTH_RATIO
    adaptive_max_shrink_ratio: float = ADAPTIVE_MAX_SHRINK_RATIO
    adaptive_deadband_ratio: float = ADAPTIVE_DEADBAND_RATIO
    adaptive_expansion_only: bool = ADAPTIVE_EXPANSION_ONLY_DEFAULT
    adaptive_selection_prior_max_age_frames: int = LATENT_PRIOR_SELECTION_PRIOR_MAX_AGE_FRAMES
    adaptive_emission_hold_max_age_frames: int = LATENT_PRIOR_EMISSION_HOLD_MAX_AGE_FRAMES
    adaptive_growth_confirmations: int = LATENT_PRIOR_GROWTH_CONFIRMATIONS
    motion_blur_min_eigen_ratio: float = MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT
    motion_blur_min_mask_pixels: int = MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT
    motion_blur_target_fraction: float = MOTION_BLUR_TARGET_FRACTION_DEFAULT
    motion_blur_intensity_center: bool = False
    motion_blur_stabilized_diameter: bool = False
    motion_blur_diameter_window: int = 2


def parse_bool_env(value, default=False):
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    return default


def load_sam3_config(environ=None):
    env = environ if environ is not None else os.environ
    mode = env.get("SAM3_PROMPT_MODE", "box").strip().lower()
    if mode not in VALID_PROMPT_MODES:
        mode = "box"
    text_prompt = env.get("SAM3_TEXT_PROMPT", DEFAULT_TEXT_PROMPT).strip()
    policy = env.get("SAM3_OUTPUT_POLICY", DEFAULT_OUTPUT_POLICY).strip().lower()
    if policy not in VALID_OUTPUT_POLICIES:
        policy = DEFAULT_OUTPUT_POLICY
    try:
        prob_thresh = float(env.get("SAM3_OUTPUT_PROB_THRESH", str(DEFAULT_OUTPUT_PROB_THRESH)))
    except ValueError:
        prob_thresh = DEFAULT_OUTPUT_PROB_THRESH
    ir_refine_enabled = parse_bool_env(env.get("SAM3_IR_REFINE"), False)
    ir_stop_on_missing = parse_bool_env(env.get("SAM3_IR_STOP_ON_MISSING"), False)

    def _float_env(name, default):
        try:
            return float(env.get(name, str(default)))
        except ValueError:
            return default

    def _int_env(name, default):
        try:
            return int(env.get(name, str(default)))
        except ValueError:
            return default

    padding_default = (
        ADAPTIVE_DEADBAND_PADDING_FACTOR
        if policy in (
            "adaptive_component_deadband_box",
            "adaptive_component_latent_prior_box",
        )
        else ADAPTIVE_PADDING_FACTOR
    )
    return Sam3Config(
        prompt_mode=mode,
        text_prompt=text_prompt,
        output_policy=policy,
        output_prob_thresh=prob_thresh,
        detection_score_threshold=_float_env(
            "SAM3_DETECTION_SCORE_THRESHOLD", DEFAULT_DETECTION_SCORE_THRESHOLD
        ),
        detection_max_prompt_distance_px=_float_env(
            "SAM3_DETECTION_MAX_PROMPT_DISTANCE_PX", 0.0
        ),
        recover_after_miss=parse_bool_env(env.get("SAM3_RECOVER_AFTER_MISS"), False),
        edge_hold_enabled=parse_bool_env(env.get("SAM3_EDGE_HOLD"), False),
        ui_gate_clamp=parse_bool_env(env.get("SAM3_UI_GATE_CLAMP"), False),
        trailing_hold_frames=_int_env("SAM3_TRAILING_HOLD_FRAMES", 0),
        recover_max_gap=_int_env("SAM3_RECOVER_MAX_GAP", 2),
        reseed_after_miss=parse_bool_env(env.get("SAM3_RESEED_AFTER_MISS"), False),
        reseed_max_attempts=_int_env("SAM3_RESEED_MAX_ATTEMPTS", 2),
        prompt_box_scale=_float_env("SAM3_PROMPT_BOX_SCALE", 1.0),
        rod_mask_anchor=parse_bool_env(env.get("SAM3_ROD_MASK_ANCHOR"), False),
        rod_mask_max_width=_int_env("SAM3_ROD_MASK_MAX_WIDTH", 3),
        rod_mask_min_aspect=_float_env("SAM3_ROD_MASK_MIN_ASPECT", 2.0),
        rod_mask_blob_reanchor=parse_bool_env(
            env.get("SAM3_ROD_BLOB_REANCHOR"), False
        ),
        rod_blob_search_px=_int_env("SAM3_ROD_BLOB_SEARCH_PX", 24),
        rod_blob_margin=_float_env("SAM3_ROD_BLOB_MARGIN", 15.0),
        rod_mask_lollipop_ratio=_float_env("SAM3_ROD_MASK_LOLLIPOP_RATIO", 2.0),
        ir_refine_enabled=ir_refine_enabled,
        ir_stop_on_missing=ir_stop_on_missing,
        adaptive_padding_factor=_float_env("SAM3_ADAPTIVE_PADDING_FACTOR", padding_default),
        adaptive_roi_scale=_float_env("SAM3_ADAPTIVE_ROI_SCALE", ADAPTIVE_ROI_SCALE),
        adaptive_min_roi_margin_px=_int_env(
            "SAM3_ADAPTIVE_MIN_ROI_MARGIN_PX", ADAPTIVE_MIN_ROI_MARGIN_PX
        ),
        adaptive_min_side_px=_int_env("SAM3_ADAPTIVE_MIN_SIDE_PX", ADAPTIVE_MIN_SIDE_PX),
        adaptive_max_growth_ratio=_float_env(
            "SAM3_ADAPTIVE_MAX_GROWTH_RATIO", ADAPTIVE_MAX_GROWTH_RATIO
        ),
        adaptive_max_shrink_ratio=_float_env(
            "SAM3_ADAPTIVE_MAX_SHRINK_RATIO", ADAPTIVE_MAX_SHRINK_RATIO
        ),
        adaptive_deadband_ratio=_float_env(
            "SAM3_ADAPTIVE_DEADBAND_RATIO", ADAPTIVE_DEADBAND_RATIO
        ),
        adaptive_expansion_only=parse_bool_env(
            env.get("SAM3_ADAPTIVE_EXPANSION_ONLY"),
            ADAPTIVE_EXPANSION_ONLY_DEFAULT,
        ),
        adaptive_selection_prior_max_age_frames=_int_env(
            "SAM3_ADAPTIVE_SELECTION_PRIOR_MAX_AGE_FRAMES",
            LATENT_PRIOR_SELECTION_PRIOR_MAX_AGE_FRAMES,
        ),
        adaptive_emission_hold_max_age_frames=_int_env(
            "SAM3_ADAPTIVE_EMISSION_HOLD_MAX_AGE_FRAMES",
            LATENT_PRIOR_EMISSION_HOLD_MAX_AGE_FRAMES,
        ),
        adaptive_growth_confirmations=_int_env(
            "SAM3_ADAPTIVE_GROWTH_CONFIRMATIONS",
            LATENT_PRIOR_GROWTH_CONFIRMATIONS,
        ),
        motion_blur_min_eigen_ratio=_float_env(
            "SAM3_MOTION_BLUR_MIN_EIGEN_RATIO", MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT
        ),
        motion_blur_min_mask_pixels=_int_env(
            "SAM3_MOTION_BLUR_MIN_MASK_PIXELS", MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT
        ),
        motion_blur_target_fraction=_float_env(
            "SAM3_MOTION_BLUR_TARGET_FRACTION", MOTION_BLUR_TARGET_FRACTION_DEFAULT
        ),
        motion_blur_intensity_center=parse_bool_env(
            env.get("SAM3_MOTION_BLUR_INTENSITY_CENTER"), False
        ),
        motion_blur_stabilized_diameter=parse_bool_env(
            env.get("SAM3_MOTION_BLUR_STABILIZED_DIAMETER"), False
        ),
        motion_blur_diameter_window=_int_env("SAM3_MOTION_BLUR_DIAMETER_WINDOW", 2),
    )


def bbox_center(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def cv_box_to_sam3_input(bbox_abs, h, w):
    x1, y1, x2, y2 = [float(v) for v in bbox_abs]
    rw = (x2 - x1) / w
    rh = (y2 - y1) / h
    rx = x1 / w
    ry = y1 / h
    return [rx, ry, rw, rh]


def xywh_rel_to_abs(box, h, w):
    rx, ry, rw, rh = [float(v) for v in box]
    x1 = int(rx * w)
    y1 = int(ry * h)
    x2 = int((rx + rw) * w)
    y2 = int((ry + rh) * h)
    return _clamp_bbox([float(x1), float(y1), float(x2), float(y2)], h, w)


def _clamp_bbox(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def mask_tight_bbox(mask, h, w):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return _clamp_bbox(
        [float(x_min), float(y_min), float(x_max), float(y_max)],
        h,
        w,
    )


def mask_center_bbox(mask, reference_bbox, h, w, min_size_ratio=0.25):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    cy = float(coords[:, 0].mean())
    cx = float(coords[:, 1].mean())
    ref_w = max(float(reference_bbox[2] - reference_bbox[0]), MIN_BOX_SIDE_PX)
    ref_h = max(float(reference_bbox[3] - reference_bbox[1]), MIN_BOX_SIDE_PX)
    half_w = max(ref_w / 2.0, ref_w * min_size_ratio / 2.0, MIN_BOX_SIDE_PX / 2.0)
    half_h = max(ref_h / 2.0, ref_h * min_size_ratio / 2.0, MIN_BOX_SIDE_PX / 2.0)
    return _clamp_bbox(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        h,
        w,
    )


def build_add_prompt_request(
    session_id,
    frame_idx,
    prompt_bbox,
    image_height,
    image_width,
    config,
    *,
    force_box_only=False,
):
    request = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_idx,
        "output_prob_thresh": config.output_prob_thresh,
    }
    mode = "box" if force_box_only else config.prompt_mode
    if mode in ("box", "text_box"):
        rel_box = cv_box_to_sam3_input(prompt_bbox, image_height, image_width)
        request["bounding_boxes"] = [rel_box]
        request["bounding_box_labels"] = [1]
        request["rel_coordinates"] = True
    if mode in ("text", "text_box") and config.text_prompt:
        request["text"] = config.text_prompt
    return request


def select_bbox_from_outputs(outputs, image_height, image_width, reference_bbox, config):
    if outputs is None or reference_bbox is None:
        return None

    h, w = image_height, image_width
    candidates = []

    if config.output_policy in ("mask_bbox", "mask_center_box"):
        masks = outputs.get("out_binary_masks")
        if masks is not None:
            for mask in masks:
                if config.output_policy == "mask_center_box":
                    bbox = mask_center_bbox(mask, reference_bbox, h, w)
                else:
                    bbox = mask_tight_bbox(mask, h, w)
                if bbox is not None:
                    candidates.append(bbox)

    if not candidates or config.output_policy == "sam_box":
        boxes = outputs.get("out_boxes_xywh")
        if boxes is not None:
            for box in boxes:
                candidates.append(xywh_rel_to_abs(box, h, w))

    if not candidates and config.output_policy != "sam_box":
        boxes = outputs.get("out_boxes_xywh")
        if boxes is not None:
            for box in boxes:
                candidates.append(xywh_rel_to_abs(box, h, w))

    if not candidates:
        return None

    ref_cx, ref_cy = bbox_center(reference_bbox)
    max_distance = float(
        getattr(config, "detection_max_prompt_distance_px", 0.0) or 0.0
    )
    if max_distance > 0.0:
        candidates = [
            bbox
            for bbox in candidates
            if (
                (bbox_center(bbox)[0] - ref_cx) ** 2
                + (bbox_center(bbox)[1] - ref_cy) ** 2
            )
            ** 0.5
            <= max_distance
        ]
        if not candidates:
            return None
    best = min(
        candidates,
        key=lambda bbox: (bbox_center(bbox)[0] - ref_cx) ** 2
        + (bbox_center(bbox)[1] - ref_cy) ** 2,
    )
    return best


def select_mask_from_outputs(outputs, reference_bbox):
    """SAM3 mask nearest the reference-box centre.

    Applies the same nearest-to-reference rule as ``mask_center_box`` candidate
    selection.  Returns ``None`` when no usable mask candidate exists.
    """
    if outputs is None or reference_bbox is None:
        return None
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return None
    ref_cx, ref_cy = bbox_center(reference_bbox)
    best_mask = None
    best_distance = None
    for mask in masks:
        centroid = mask_centroid(mask)
        if centroid is None:
            continue
        distance = (centroid[0] - ref_cx) ** 2 + (centroid[1] - ref_cy) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_mask = mask
    return best_mask


def image_to_grayscale(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def recenter_bbox(bbox, center_x, center_y, image_height, image_width):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = x2 - x1
    box_h = y2 - y1
    return _clamp_bbox(
        [
            center_x - box_w / 2.0,
            center_y - box_h / 2.0,
            center_x + box_w / 2.0,
            center_y + box_h / 2.0,
        ],
        image_height,
        image_width,
    )


def _geometry_center_from_mask(component_mask, roi_x1, roi_y1):
    ys, xs = np.where(component_mask)
    return float(xs.mean()) + roi_x1, float(ys.mean()) + roi_y1


def _contour_shape_metrics(component_mask):
    component_u8 = (component_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        component_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    contour_area = float(cv2.contourArea(contour))
    circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
    _x, _y, comp_w, comp_h = cv2.boundingRect(contour)
    aspect_ratio = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)
    return circularity, aspect_ratio


def _score_support_component(
    component_mask,
    roi_x1,
    roi_y1,
    sam_cx,
    sam_cy,
    box_w,
    box_h,
    ref_area,
    previous_bbox,
    max_disp,
):
    area = float(component_mask.sum())
    min_area = ref_area * IR_REFINE_MIN_AREA_RATIO
    max_area = ref_area * IR_REFINE_MAX_AREA_RATIO
    if area < min_area or area > max_area:
        return None

    cx, cy = _geometry_center_from_mask(component_mask, roi_x1, roi_y1)
    displacement = ((cx - sam_cx) ** 2 + (cy - sam_cy) ** 2) ** 0.5
    if displacement > max_disp:
        return None

    shape = _contour_shape_metrics(component_mask)
    if shape is None:
        return None
    circularity, aspect_ratio = shape
    if circularity < IR_REFINE_MIN_CIRCULARITY:
        return None
    if aspect_ratio > IR_REFINE_MAX_ASPECT_RATIO:
        return None

    if previous_bbox is not None:
        prev_cx, prev_cy = bbox_center(previous_bbox)
        max_temporal_disp = max(box_w, box_h) * IR_REFINE_MAX_TEMPORAL_DISP_RATIO
        temporal_disp = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
        if temporal_disp > max_temporal_disp:
            return None

    score = displacement - circularity * 10.0
    return score, cx, cy


def refine_bbox_with_ir_intensity(image, sam_bbox, previous_bbox=None, stop_on_missing=False):
    """Fail-closed IR centroid refinement: move center only, preserve box size."""
    if sam_bbox is None:
        return None

    def _missing():
        return None if stop_on_missing else sam_bbox

    sam_bbox = [float(v) for v in sam_bbox]
    image_height, image_width = image.shape[:2]
    gray = image_to_grayscale(image)

    sam_cx, sam_cy = bbox_center(sam_bbox)
    box_w = max(sam_bbox[2] - sam_bbox[0], MIN_BOX_SIDE_PX)
    box_h = max(sam_bbox[3] - sam_bbox[1], MIN_BOX_SIDE_PX)
    ref_area = box_w * box_h

    margin_x = max(box_w * IR_REFINE_ROI_SCALE, IR_REFINE_MIN_ROI_MARGIN_PX)
    margin_y = max(box_h * IR_REFINE_ROI_SCALE, IR_REFINE_MIN_ROI_MARGIN_PX)
    roi_x1 = int(max(0.0, sam_cx - margin_x))
    roi_y1 = int(max(0.0, sam_cy - margin_y))
    roi_x2 = int(min(float(image_width), sam_cx + margin_x))
    roi_y2 = int(min(float(image_height), sam_cy + margin_y))

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return _missing()

    threshold = float(np.percentile(roi, IR_REFINE_BRIGHT_PERCENTILE))
    background = float(np.median(roi))
    bright_threshold = max(threshold, background + 15.0)
    if bright_threshold >= float(roi.max()):
        return _missing()

    support_threshold = min(
        bright_threshold - 1.0,
        background + IR_REFINE_SUPPORT_CONTRAST_FLOOR,
    )
    if support_threshold <= background:
        support_threshold = background + 1.0

    bright_binary = (roi >= bright_threshold).astype(np.uint8)
    num_bright, bright_labels, bright_stats, _bright_centroids = cv2.connectedComponentsWithStats(
        bright_binary,
        connectivity=8,
    )
    if num_bright <= 1:
        return _missing()

    support_binary = (roi >= support_threshold).astype(np.uint8)
    num_support, support_labels, _support_stats, _support_centroids = cv2.connectedComponentsWithStats(
        support_binary,
        connectivity=8,
    )
    if num_support <= 1:
        return _missing()

    max_disp = min(box_w, box_h) * IR_REFINE_MAX_DISP_RATIO
    candidates = []
    seen_support_labels = set()

    for seed_label_id in range(1, num_bright):
        seed_area = float(bright_stats[seed_label_id, cv2.CC_STAT_AREA])
        if seed_area < IR_REFINE_MIN_SEED_PIXELS:
            continue

        seed_mask = bright_labels == seed_label_id
        overlapping_support_ids = np.unique(support_labels[seed_mask])
        overlapping_support_ids = overlapping_support_ids[overlapping_support_ids != 0]

        for support_label_id in overlapping_support_ids:
            if support_label_id in seen_support_labels:
                continue
            support_mask = support_labels == support_label_id
            scored = _score_support_component(
                support_mask,
                roi_x1,
                roi_y1,
                sam_cx,
                sam_cy,
                box_w,
                box_h,
                ref_area,
                previous_bbox,
                max_disp,
            )
            if scored is None:
                continue
            seen_support_labels.add(support_label_id)
            candidates.append(scored)

    if not candidates:
        return _missing()

    _score, best_cx, best_cy = min(candidates, key=lambda item: item[0])
    return recenter_bbox(sam_bbox, best_cx, best_cy, image_height, image_width)


def bbox_side_length(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(x2 - x1, y2 - y1, MIN_BOX_SIDE_PX)


def square_target_bbox_from_center(center_x, center_y, side):
    """Full square target around center; coordinates may lie outside the image."""
    half = max(float(side) / 2.0, MIN_BOX_SIDE_PX / 2.0)
    return [
        float(center_x) - half,
        float(center_y) - half,
        float(center_x) + half,
        float(center_y) + half,
    ]


def _bbox_edge_clipped(requested_bbox, visible_bbox, tol=1e-6):
    return any(
        abs(float(requested_bbox[i]) - float(visible_bbox[i])) > tol for i in range(4)
    )


def build_adaptive_square_target(center_x, center_y, target_side, image_height, image_width):
    """Build adaptive square target; square before clipping, possibly rectangular after.

    Interior case: visible bbox is square and centered on the requested center.
    Edge-clipped case: clip the full square target to image bounds without recentering
    inward to preserve side length.
    """
    requested_target_side = max(float(target_side), MIN_BOX_SIDE_PX)
    requested_target_bbox = square_target_bbox_from_center(
        center_x,
        center_y,
        requested_target_side,
    )
    visible_clipped_bbox = _clamp_bbox(
        requested_target_bbox,
        image_height,
        image_width,
    )
    edge_clipped = _bbox_edge_clipped(requested_target_bbox, visible_clipped_bbox)
    return {
        "requested_target_side": requested_target_side,
        "requested_target_bbox": requested_target_bbox,
        "visible_clipped_bbox": visible_clipped_bbox,
        "edge_clipped": edge_clipped,
    }


def square_bbox_from_center(center_x, center_y, side, image_height, image_width):
    return build_adaptive_square_target(
        center_x,
        center_y,
        side,
        image_height,
        image_width,
    )["visible_clipped_bbox"]


def equivalent_diameter_from_area(area):
    area = max(float(area), 0.0)
    return math.sqrt(4.0 * area / math.pi)


def median_component_side(bbox_width, bbox_height, component_area):
    return float(
        np.median(
            [
                float(bbox_width),
                float(bbox_height),
                equivalent_diameter_from_area(component_area),
            ]
        )
    )


def _bbox_overlap_area(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _component_global_bbox(component_mask, roi_x1, roi_y1):
    ys, xs = np.where(component_mask)
    if xs.size == 0:
        return None
    return [
        float(xs.min()) + roi_x1,
        float(ys.min()) + roi_y1,
        float(xs.max()) + roi_x1,
        float(ys.max()) + roi_y1,
    ]


def _evaluate_adaptive_scale_component(
    component_mask,
    roi_x1,
    roi_y1,
    center_x,
    center_y,
    box_w,
    box_h,
    ref_area,
    previous_bbox,
    max_disp,
):
    area = float(component_mask.sum())
    min_area = ref_area * IR_REFINE_MIN_AREA_RATIO
    max_area = ref_area * IR_REFINE_MAX_AREA_RATIO
    if area < min_area or area > max_area:
        return None

    comp_bbox = _component_global_bbox(component_mask, roi_x1, roi_y1)
    if comp_bbox is None:
        return None
    comp_cx = (comp_bbox[0] + comp_bbox[2]) / 2.0
    comp_cy = (comp_bbox[1] + comp_bbox[3]) / 2.0

    if not (
        comp_bbox[0] <= center_x <= comp_bbox[2]
        and comp_bbox[1] <= center_y <= comp_bbox[3]
    ):
        displacement = ((comp_cx - center_x) ** 2 + (comp_cy - center_y) ** 2) ** 0.5
        if displacement > max_disp:
            return None
    else:
        displacement = ((comp_cx - center_x) ** 2 + (comp_cy - center_y) ** 2) ** 0.5

    shape = _contour_shape_metrics(component_mask)
    if shape is None:
        return None
    circularity, aspect_ratio = shape
    if circularity < IR_REFINE_MIN_CIRCULARITY:
        return None
    if aspect_ratio > IR_REFINE_MAX_ASPECT_RATIO:
        return None

    if previous_bbox is not None:
        prev_cx, prev_cy = bbox_center(previous_bbox)
        max_temporal_disp = max(box_w, box_h) * IR_REFINE_MAX_TEMPORAL_DISP_RATIO
        temporal_disp = ((comp_cx - prev_cx) ** 2 + (comp_cy - prev_cy) ** 2) ** 0.5
        if temporal_disp > max_temporal_disp:
            return None

    overlap_ratio = 0.0
    if previous_bbox is not None and ref_area > 0:
        overlap_ratio = _bbox_overlap_area(comp_bbox, previous_bbox) / ref_area

    score = displacement - circularity * 10.0 - overlap_ratio * 20.0
    comp_w = comp_bbox[2] - comp_bbox[0]
    comp_h = comp_bbox[3] - comp_bbox[1]
    return {
        "score": score,
        "area": area,
        "comp_w": comp_w,
        "comp_h": comp_h,
        "displacement": displacement,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "overlap_ratio": overlap_ratio,
    }


def _score_adaptive_scale_component(
    component_mask,
    roi_x1,
    roi_y1,
    center_x,
    center_y,
    box_w,
    box_h,
    ref_area,
    previous_bbox,
    max_disp,
):
    evaluated = _evaluate_adaptive_scale_component(
        component_mask,
        roi_x1,
        roi_y1,
        center_x,
        center_y,
        box_w,
        box_h,
        ref_area,
        previous_bbox,
        max_disp,
    )
    if evaluated is None:
        return None
    return (
        evaluated["score"],
        evaluated["area"],
        evaluated["comp_w"],
        evaluated["comp_h"],
    )


def _collect_adaptive_scale_candidates(
    image,
    center_x,
    center_y,
    reference_side,
    previous_bbox,
    config,
):
    image_height, image_width = image.shape[:2]
    gray = image_to_grayscale(image)
    box_w = max(float(reference_side), config.adaptive_min_side_px)
    box_h = box_w
    ref_area = box_w * box_h

    margin_x = max(box_w * config.adaptive_roi_scale, config.adaptive_min_roi_margin_px)
    margin_y = max(box_h * config.adaptive_roi_scale, config.adaptive_min_roi_margin_px)
    roi_x1 = int(max(0.0, center_x - margin_x))
    roi_y1 = int(max(0.0, center_y - margin_y))
    roi_x2 = int(min(float(image_width), center_x + margin_x))
    roi_y2 = int(min(float(image_height), center_y + margin_y))

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return []

    threshold = float(np.percentile(roi, IR_REFINE_BRIGHT_PERCENTILE))
    background = float(np.median(roi))
    bright_threshold = max(threshold, background + 15.0)
    if bright_threshold >= float(roi.max()):
        return []

    support_threshold = min(
        bright_threshold - 1.0,
        background + IR_REFINE_SUPPORT_CONTRAST_FLOOR,
    )
    if support_threshold <= background:
        support_threshold = background + 1.0

    bright_binary = (roi >= bright_threshold).astype(np.uint8)
    num_bright, bright_labels, bright_stats, _bright_centroids = cv2.connectedComponentsWithStats(
        bright_binary,
        connectivity=8,
    )
    if num_bright <= 1:
        return []

    support_binary = (roi >= support_threshold).astype(np.uint8)
    num_support, support_labels, _support_stats, _support_centroids = cv2.connectedComponentsWithStats(
        support_binary,
        connectivity=8,
    )
    if num_support <= 1:
        return []

    max_disp = min(box_w, box_h) * IR_REFINE_MAX_DISP_RATIO
    candidates = []
    seen_support_labels = set()

    for seed_label_id in range(1, num_bright):
        seed_area = float(bright_stats[seed_label_id, cv2.CC_STAT_AREA])
        if seed_area < IR_REFINE_MIN_SEED_PIXELS:
            continue

        seed_mask = bright_labels == seed_label_id
        overlapping_support_ids = np.unique(support_labels[seed_mask])
        overlapping_support_ids = overlapping_support_ids[overlapping_support_ids != 0]

        for support_label_id in overlapping_support_ids:
            if support_label_id in seen_support_labels:
                continue
            support_mask = support_labels == support_label_id
            evaluated = _evaluate_adaptive_scale_component(
                support_mask,
                roi_x1,
                roi_y1,
                center_x,
                center_y,
                box_w,
                box_h,
                ref_area,
                previous_bbox,
                max_disp,
            )
            if evaluated is None:
                continue
            seen_support_labels.add(support_label_id)
            candidates.append(evaluated)

    return candidates


def select_adaptive_scale_component(
    image,
    center_x,
    center_y,
    reference_side,
    previous_bbox,
    config,
):
    """Select a bright support component for scale estimation only (center fixed)."""
    candidates = _collect_adaptive_scale_candidates(
        image,
        center_x,
        center_y,
        reference_side,
        previous_bbox,
        config,
    )
    if not candidates:
        return None

    best = min(candidates, key=lambda item: item["score"])
    return {
        "component_area": best["area"],
        "component_bbox_width": best["comp_w"],
        "component_bbox_height": best["comp_h"],
        "equivalent_diameter": equivalent_diameter_from_area(best["area"]),
    }


def select_adaptive_scale_component_with_diagnostics(
    image,
    center_x,
    center_y,
    reference_side,
    previous_bbox,
    config,
):
    candidates = _collect_adaptive_scale_candidates(
        image,
        center_x,
        center_y,
        reference_side,
        previous_bbox,
        config,
    )
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda item: item["score"])
    best = ranked[0]
    score_margin = None
    if len(ranked) >= 2:
        score_margin = ranked[1]["score"] - ranked[0]["score"]
    return {
        "component_area": best["area"],
        "component_bbox_width": best["comp_w"],
        "component_bbox_height": best["comp_h"],
        "equivalent_diameter": equivalent_diameter_from_area(best["area"]),
        "component_distance": best["displacement"],
        "component_circularity": best["circularity"],
        "component_aspect_ratio": best["aspect_ratio"],
        "component_overlap": best["overlap_ratio"],
        "score_margin": score_margin,
    }


def gate_adaptive_side(padded_side, reference_side, config):
    reference_side = max(float(reference_side), config.adaptive_min_side_px)
    padded_side = max(float(padded_side), config.adaptive_min_side_px)
    lo = reference_side * config.adaptive_max_shrink_ratio
    hi = reference_side * config.adaptive_max_growth_ratio
    return max(lo, min(hi, padded_side))


def _relative_side_delta(candidate_side, reference_side):
    reference_side = float(reference_side)
    if reference_side <= 0:
        return None
    return (float(candidate_side) - reference_side) / reference_side


def _is_meaningful_growth(gated_side, canonical_side, deadband_ratio):
    if gated_side <= canonical_side:
        return False
    delta = _relative_side_delta(gated_side, canonical_side)
    return delta is not None and delta > deadband_ratio


def _bound_emission_side(side, canonical_side, config):
    return gate_adaptive_side(side, canonical_side, config)


def _default_latent_prior_state(canonical_reference_side):
    return {
        "selection_prior_side": canonical_reference_side,
        "selection_prior_age": 0,
        "growth_confirmation_count": 0,
        "emission_prior_side": None,
        "emission_hold_age": 0,
    }


def _latent_prior_state_from_diag(diag):
    return {
        "selection_prior_side": diag.get("selection_prior_side_after"),
        "selection_prior_age": diag.get("selection_prior_age", 0),
        "growth_confirmation_count": diag.get("growth_confirmation_count", 0),
        "emission_prior_side": diag.get("emission_prior_side_after"),
        "emission_hold_age": diag.get("emission_hold_age", 0),
    }


def apply_adaptive_component_square_box(
    image,
    center_bbox,
    prompt_bbox,
    previous_bbox,
    is_init_frame,
    config,
    *,
    previous_accepted_target_side=None,
):
    """Preserve SAM/IR center; adapt square side from local IR component.

    adaptive_component_square_box means square before image clipping; the returned
    visible bbox may be non-square after geometric clipping at image edges.
    """
    center_x, center_y = bbox_center(center_bbox)
    image_height, image_width = image.shape[:2]
    seed_side = bbox_side_length(prompt_bbox) if prompt_bbox is not None else None
    reference_side = previous_accepted_target_side
    if reference_side is None and not is_init_frame:
        reference_side = seed_side
    if reference_side is None:
        reference_side = seed_side
    if reference_side is None:
        reference_side = config.adaptive_min_side_px

    component = select_adaptive_scale_component(
        image,
        center_x,
        center_y,
        reference_side,
        previous_bbox,
        config,
    )

    diag = {
        "frame_index": None,
        "requested_center_x": center_x,
        "requested_center_y": center_y,
        "center_x": center_x,
        "center_y": center_y,
        "seed_side": seed_side,
        "previous_side": previous_accepted_target_side,
        "previous_accepted_target_side": previous_accepted_target_side,
        "component_area": None,
        "component_bbox_width": None,
        "component_bbox_height": None,
        "equivalent_diameter": None,
        "raw_component_side": None,
        "padded_side": None,
        "gated_side": None,
        "accepted_target_side": None,
        "requested_target_side": None,
        "requested_target_bbox": None,
        "visible_bbox": None,
        "edge_clipped": False,
        "fallback_reason": None,
        "lost": False,
        "adaptiveScaleStage": None,
    }

    if component is None:
        diag["fallback_reason"] = "no_valid_component"
        diag["adaptiveScaleStage"] = "component_selection"
        gated_side = reference_side
    else:
        raw_side = median_component_side(
            component["component_bbox_width"],
            component["component_bbox_height"],
            component["component_area"],
        )
        padded_side = raw_side * config.adaptive_padding_factor
        gated_side = gate_adaptive_side(padded_side, reference_side, config)
        diag["component_area"] = component["component_area"]
        diag["component_bbox_width"] = component["component_bbox_width"]
        diag["component_bbox_height"] = component["component_bbox_height"]
        diag["equivalent_diameter"] = component["equivalent_diameter"]
        diag["raw_component_side"] = raw_side
        diag["padded_side"] = padded_side
        diag["gated_side"] = gated_side
        if abs(gated_side - padded_side) > 1e-6:
            diag["adaptiveScaleStage"] = "temporal_gating"
        else:
            diag["adaptiveScaleStage"] = "raw_diameter_estimation"

    if component is None:
        diag["gated_side"] = gated_side

    target = build_adaptive_square_target(
        center_x,
        center_y,
        gated_side,
        image_height,
        image_width,
    )
    diag["accepted_target_side"] = gated_side
    diag["requested_target_side"] = target["requested_target_side"]
    diag["requested_target_bbox"] = target["requested_target_bbox"]
    diag["visible_bbox"] = target["visible_clipped_bbox"]
    diag["edge_clipped"] = target["edge_clipped"]

    return target["visible_clipped_bbox"], diag


def apply_adaptive_component_deadband_box(
    image,
    center_bbox,
    prompt_bbox,
    previous_bbox,
    is_init_frame,
    config,
    *,
    previous_accepted_target_side=None,
):
    """Preserve canonical center; adapt side only when evidence clears deadband gates."""
    center_x, center_y = bbox_center(center_bbox)
    image_height, image_width = image.shape[:2]
    seed_side = bbox_side_length(prompt_bbox) if prompt_bbox is not None else None
    canonical_reference_side = bbox_side_length(center_bbox)
    reference_side = previous_accepted_target_side
    if reference_side is None and not is_init_frame:
        reference_side = seed_side
    if reference_side is None:
        reference_side = seed_side
    if reference_side is None:
        reference_side = config.adaptive_min_side_px

    component = select_adaptive_scale_component_with_diagnostics(
        image,
        center_x,
        center_y,
        reference_side,
        previous_bbox,
        config,
    )

    component_valid = component is not None
    fallback_reason = None
    raw_component_side = None
    raw_candidate_side = None
    candidate_side = None
    gated_candidate_side = None
    relative_side_delta = None
    final_relative_side_delta = None
    deadband_blocked = False
    expansion_only_blocked = False
    temporal_gate_changed = False
    temporal_gate_blocked = False
    adaptive_emitted = False
    edge_clipped = False
    gated_side = None
    prev_before = previous_accepted_target_side
    gate_reference_side = prev_before
    if gate_reference_side is None and not is_init_frame:
        gate_reference_side = seed_side
    if gate_reference_side is None:
        gate_reference_side = seed_side
    if gate_reference_side is None:
        gate_reference_side = config.adaptive_min_side_px

    if not component_valid:
        fallback_reason = "no_valid_component"
        emitted_bbox = list(center_bbox)
        accepted_target_side = canonical_reference_side
    else:
        raw_component_side = median_component_side(
            component["component_bbox_width"],
            component["component_bbox_height"],
            component["component_area"],
        )
        raw_candidate_side = raw_component_side * config.adaptive_padding_factor
        candidate_side = raw_candidate_side
        gated_candidate_side = gate_adaptive_side(
            raw_candidate_side,
            gate_reference_side,
            config,
        )
        gated_side = gated_candidate_side
        temporal_gate_changed = abs(gated_candidate_side - raw_candidate_side) > 1e-6

        if canonical_reference_side > 0:
            relative_side_delta = (
                (raw_candidate_side - canonical_reference_side) / canonical_reference_side
            )
            final_relative_side_delta = (
                (gated_candidate_side - canonical_reference_side) / canonical_reference_side
            )

        raw_passes_expansion_only = (
            not config.adaptive_expansion_only
            or raw_candidate_side > canonical_reference_side
        )
        raw_passes_deadband = (
            canonical_reference_side <= 0
            or (
                (raw_candidate_side - canonical_reference_side) / canonical_reference_side
            ) > config.adaptive_deadband_ratio
        )

        emit_canonical = False
        if config.adaptive_expansion_only and gated_candidate_side <= canonical_reference_side:
            expansion_only_blocked = True
            emit_canonical = True
            if temporal_gate_changed and raw_passes_expansion_only:
                temporal_gate_blocked = True
                fallback_reason = "temporal_gate_expansion_only"
            else:
                fallback_reason = "expansion_only"
        elif canonical_reference_side > 0 and final_relative_side_delta <= config.adaptive_deadband_ratio:
            deadband_blocked = True
            emit_canonical = True
            if temporal_gate_changed and raw_passes_deadband:
                temporal_gate_blocked = True
                fallback_reason = "temporal_gate_deadband"
            else:
                fallback_reason = "deadband"

        if emit_canonical:
            emitted_bbox = list(center_bbox)
            accepted_target_side = canonical_reference_side
        else:
            target = build_adaptive_square_target(
                center_x,
                center_y,
                gated_candidate_side,
                image_height,
                image_width,
            )
            emitted_bbox = target["visible_clipped_bbox"]
            edge_clipped = target["edge_clipped"]
            accepted_target_side = gated_candidate_side
            adaptive_emitted = True

    diag = {
        "policy": "adaptive_component_deadband_box",
        "canonical_tracking_bbox": list(center_bbox),
        "canonical_reference_side": canonical_reference_side,
        "candidate_side": candidate_side,
        "raw_candidate_side": raw_candidate_side,
        "gated_candidate_side": gated_candidate_side,
        "relative_side_delta": relative_side_delta,
        "final_relative_side_delta": final_relative_side_delta,
        "component_valid": component_valid,
        "deadband_blocked": deadband_blocked,
        "expansion_only_blocked": expansion_only_blocked,
        "temporal_gate_changed": temporal_gate_changed,
        "temporal_gate_blocked": temporal_gate_blocked,
        "adaptive_emitted": adaptive_emitted,
        "previous_accepted_target_side_before": prev_before,
        "previous_accepted_target_side_after": accepted_target_side,
        "accepted_target_side": accepted_target_side,
        "score_margin": component.get("score_margin") if component else None,
        "component_distance": component.get("component_distance") if component else None,
        "component_overlap": component.get("component_overlap") if component else None,
        "component_circularity": component.get("component_circularity") if component else None,
        "component_aspect_ratio": component.get("component_aspect_ratio") if component else None,
        "component_area": component.get("component_area") if component else None,
        "component_bbox_width": component.get("component_bbox_width") if component else None,
        "component_bbox_height": component.get("component_bbox_height") if component else None,
        "equivalent_diameter": component.get("equivalent_diameter") if component else None,
        "raw_component_side": raw_component_side,
        "padded_side": raw_candidate_side,
        "gated_side": gated_side,
        "fallback_reason": fallback_reason,
        "lost": False,
        "edge_clipped": edge_clipped,
        "requested_center_x": center_x,
        "requested_center_y": center_y,
        "center_x": center_x,
        "center_y": center_y,
        "seed_side": seed_side,
        "previous_side": previous_accepted_target_side,
        "previous_accepted_target_side": prev_before,
    }

    return emitted_bbox, diag


def apply_adaptive_component_latent_prior_box(
    image,
    center_bbox,
    prompt_bbox,
    previous_bbox,
    is_init_frame,
    config,
    *,
    latent_prior_state=None,
):
    """Preserve canonical center; use bounded latent priors for selection and optional hold."""
    center_x, center_y = bbox_center(center_bbox)
    image_height, image_width = image.shape[:2]
    canonical_reference_side = bbox_side_length(center_bbox)
    state = latent_prior_state if latent_prior_state is not None else {}
    selection_prior_side_before = state.get("selection_prior_side")
    if selection_prior_side_before is None:
        selection_prior_side_before = canonical_reference_side
    selection_prior_age_before = int(state.get("selection_prior_age", 0))
    growth_confirmation_count_before = int(state.get("growth_confirmation_count", 0))
    emission_prior_side_before = state.get("emission_prior_side")
    emission_hold_age_before = int(state.get("emission_hold_age", 0))

    selection_prior_side = selection_prior_side_before
    selection_prior_age = selection_prior_age_before
    growth_confirmation_count = growth_confirmation_count_before
    emission_prior_side = emission_prior_side_before
    emission_hold_age = emission_hold_age_before
    growth_confirmation_reset_reason = None
    emission_hold_clear_reason = None
    fallback_reason = None
    adaptive_emitted = False
    edge_clipped = False
    component_valid = False
    raw_candidate_side = None
    gated_candidate_side = None
    relative_side_delta = None
    emission_hold_active = False

    if selection_prior_age >= config.adaptive_selection_prior_max_age_frames:
        selection_prior_side = canonical_reference_side
        selection_prior_age = 0
        growth_confirmation_count = 0
        growth_confirmation_reset_reason = "selection_prior_expired"
        emission_prior_side = None
        emission_hold_age = 0
        emission_hold_clear_reason = "selection_prior_expired"

    selection_reference = selection_prior_side
    component = select_adaptive_scale_component_with_diagnostics(
        image,
        center_x,
        center_y,
        selection_reference,
        previous_bbox,
        config,
    )
    component_valid = component is not None

    def _emit_canonical():
        return list(center_bbox)

    def _emit_adaptive(side):
        nonlocal edge_clipped, adaptive_emitted
        target = build_adaptive_square_target(
            center_x,
            center_y,
            side,
            image_height,
            image_width,
        )
        edge_clipped = target["edge_clipped"]
        adaptive_emitted = True
        return target["visible_clipped_bbox"], target["requested_target_side"]

    emitted_bbox = _emit_canonical()

    if component_valid:
        raw_component_side = median_component_side(
            component["component_bbox_width"],
            component["component_bbox_height"],
            component["component_area"],
        )
        raw_candidate_side = raw_component_side * config.adaptive_padding_factor
        gated_candidate_side = gate_adaptive_side(
            raw_candidate_side,
            selection_reference,
            config,
        )
        selection_prior_side = gated_candidate_side
        selection_prior_age = 0
        relative_side_delta = _relative_side_delta(
            gated_candidate_side,
            canonical_reference_side,
        )
        meaningful_growth = _is_meaningful_growth(
            gated_candidate_side,
            canonical_reference_side,
            config.adaptive_deadband_ratio,
        )
        shrink_or_equal = raw_candidate_side <= canonical_reference_side

        if shrink_or_equal or not meaningful_growth:
            if growth_confirmation_count > 0:
                growth_confirmation_reset_reason = "non_growth_component"
            growth_confirmation_count = 0
            if emission_prior_side is not None:
                emission_hold_clear_reason = "non_growth_component"
            emission_prior_side = None
            emission_hold_age = 0
            emitted_bbox = _emit_canonical()
            fallback_reason = "non_growth_component"
        elif meaningful_growth:
            growth_confirmation_count += 1
            if growth_confirmation_count >= config.adaptive_growth_confirmations:
                bounded_side = _bound_emission_side(
                    gated_candidate_side,
                    canonical_reference_side,
                    config,
                )
                emission_prior_side = bounded_side
                emission_hold_age = 0
                emitted_bbox, held_side = _emit_adaptive(bounded_side)
                emission_prior_side = held_side
                fallback_reason = None
            else:
                emitted_bbox = _emit_canonical()
                fallback_reason = "growth_confirmation_pending"
    else:
        if growth_confirmation_count > 0:
            growth_confirmation_reset_reason = "no_valid_component"
        growth_confirmation_count = 0
        selection_prior_age = selection_prior_age + 1
        fallback_reason = "no_valid_component"

        if (
            emission_prior_side is not None
            and emission_hold_age_before < config.adaptive_emission_hold_max_age_frames
        ):
            bounded_hold_side = _bound_emission_side(
                emission_prior_side,
                canonical_reference_side,
                config,
            )
            if _is_meaningful_growth(
                bounded_hold_side,
                canonical_reference_side,
                config.adaptive_deadband_ratio,
            ):
                emitted_bbox, held_side = _emit_adaptive(bounded_hold_side)
                emission_prior_side = held_side
                emission_hold_age = emission_hold_age_before + 1
                emission_hold_active = True
                fallback_reason = "emission_hold"
            else:
                emission_prior_side = None
                emission_hold_age = 0
                emission_hold_clear_reason = "hold_not_material"
                emitted_bbox = _emit_canonical()
        elif emission_prior_side is not None:
            emission_prior_side = None
            emission_hold_age = 0
            emission_hold_clear_reason = "hold_expired"
            emitted_bbox = _emit_canonical()

        if selection_prior_age >= config.adaptive_selection_prior_max_age_frames:
            selection_prior_side = canonical_reference_side
            selection_prior_age = 0
            growth_confirmation_count = 0
            if growth_confirmation_reset_reason is None:
                growth_confirmation_reset_reason = "selection_prior_expired"
            emission_prior_side = None
            emission_hold_age = 0
            emission_hold_active = False
            emission_hold_clear_reason = "selection_prior_expired"
            emitted_bbox = _emit_canonical()
            adaptive_emitted = False
            fallback_reason = "selection_prior_expired"

    if config.adaptive_expansion_only and component_valid and gated_candidate_side is not None:
        if gated_candidate_side <= canonical_reference_side and adaptive_emitted:
            emitted_bbox = _emit_canonical()
            adaptive_emitted = False
            if fallback_reason is None:
                fallback_reason = "expansion_only"

    diag = {
        "policy": "adaptive_component_latent_prior_box",
        "canonical_tracking_bbox": list(center_bbox),
        "canonical_reference_side": canonical_reference_side,
        "selection_prior_side_before": selection_prior_side_before,
        "selection_prior_side_after": selection_prior_side,
        "selection_prior_age": selection_prior_age,
        "raw_candidate_side": raw_candidate_side,
        "gated_candidate_side": gated_candidate_side,
        "component_valid": component_valid,
        "relative_side_delta": relative_side_delta,
        "growth_confirmation_count": growth_confirmation_count,
        "growth_confirmation_reset_reason": growth_confirmation_reset_reason,
        "emission_prior_side_before": emission_prior_side_before,
        "emission_prior_side_after": emission_prior_side,
        "emission_hold_age": emission_hold_age,
        "emission_hold_active": emission_hold_active,
        "emission_hold_clear_reason": emission_hold_clear_reason,
        "adaptive_emitted": adaptive_emitted,
        "fallback_reason": fallback_reason,
        "edge_clipped": edge_clipped,
        "lost": False,
        "component_area": component.get("component_area") if component else None,
        "component_bbox_width": component.get("component_bbox_width") if component else None,
        "component_bbox_height": component.get("component_bbox_height") if component else None,
        "equivalent_diameter": component.get("equivalent_diameter") if component else None,
        "raw_component_side": (
            median_component_side(
                component["component_bbox_width"],
                component["component_bbox_height"],
                component["component_area"],
            )
            if component
            else None
        ),
        "requested_center_x": center_x,
        "requested_center_y": center_y,
        "center_x": center_x,
        "center_y": center_y,
    }

    return emitted_bbox, diag


def log_adaptive_scale_diagnostics(diag, *, relative_frame, diag_meta=None):
    if not diag_enabled():
        return
    record = {
        "event": "adaptive_scale",
        "relativeFrame": relative_frame,
        **diag,
    }
    if diag_meta:
        record.update({
            "sessionId": diag_meta.get("sessionId"),
            "jobFrameIndex": diag_meta.get("jobFrameIndex"),
            "requestId": diag_meta.get("requestId"),
            "requestType": diag_meta.get("requestType"),
        })
    record["frame_index"] = relative_frame
    log_frame_record(record)


def log_motion_blur_diagnostics(diag, *, relative_frame=None, diag_meta=None):
    """Read-only motion-blur diagnostics; payload-free (no masks)."""
    if not diag_enabled() or diag is None:
        return
    record = {
        "event": "motion_blur_center",
        "relativeFrame": relative_frame,
        **diag,
    }
    if diag_meta:
        record.update({
            "sessionId": diag_meta.get("sessionId"),
            "jobFrameIndex": diag_meta.get("jobFrameIndex"),
            "requestId": diag_meta.get("requestId"),
            "requestType": diag_meta.get("requestType"),
        })
    log_frame_record(record)


def log_lost_frame_diagnostics(
    *,
    reason,
    relative_frame,
    sess,
    outputs,
    prompt_bbox,
    last_known_bbox,
    raw_sam_bbox,
    refined_bbox,
    config,
    diag_meta=None,
):
    """Read-only diagnostics for the first unusable/lost frame. No payloads."""
    if not diag_enabled():
        return
    corner_deltas = None
    if raw_sam_bbox is not None and refined_bbox is not None:
        try:
            corner_deltas = [
                float(refined_bbox[i]) - float(raw_sam_bbox[i]) for i in range(4)
            ]
        except Exception:
            corner_deltas = None
    record = {
        "event": "lost_frame",
        "reason": reason,
        "relativeFrame": relative_frame,
        "rawSamBbox": bbox_summary(raw_sam_bbox),
        "refinedBbox": bbox_summary(refined_bbox),
        "promptBbox": bbox_summary(prompt_bbox),
        "lastKnownBbox": bbox_summary(last_known_bbox),
        "refinedNearPrompt": _bboxes_near(refined_bbox, prompt_bbox),
        "refinedNearLastKnown": _bboxes_near(refined_bbox, last_known_bbox),
        "lastKnownNearPrompt": _bboxes_near(last_known_bbox, prompt_bbox),
        "refineCornerDeltas": corner_deltas,
        "irRefineEnabled": bool(getattr(config, "ir_refine_enabled", False)),
        "irStopOnMissing": bool(getattr(config, "ir_stop_on_missing", False)),
        "outputPolicy": getattr(config, "output_policy", None),
        "outputs": summarize_outputs(
            outputs, sess.get("image_height"), sess.get("image_width")
        ),
    }
    if diag_meta:
        record.update({
            "sessionId": diag_meta.get("sessionId"),
            "jobFrameIndex": diag_meta.get("jobFrameIndex"),
            "requestId": diag_meta.get("requestId"),
            "requestType": diag_meta.get("requestType"),
        })
    log_frame_record(record)


class SessionStaleError(Exception):
    """Raised when CVAT returns a session_key the worker no longer holds."""


class ValidationError(Exception):
    """Raised when the request payload is missing required fields."""


class PreloadRangeExhaustedError(Exception):
    """Raised when tracking requests a frame outside the bounded preload chunk."""

    def __init__(
        self,
        message: str,
        *,
        base_frame: int,
        preloaded_count: int,
        preloaded_until_frame: int,
        requested_frame: int,
    ) -> None:
        super().__init__(message)
        self.base_frame = base_frame
        self.preloaded_count = preloaded_count
        self.preloaded_until_frame = preloaded_until_frame
        self.requested_frame = requested_frame


def _patch_edt():
    try:
        from sam3.model import edt as edt_module

        def edt_cv2(data):
            assert data.dim() == 3 and data.is_cuda
            B, H, W = data.shape
            data_np = data.cpu().numpy().astype(np.uint8)
            out = np.zeros_like(data_np, dtype=np.float32)
            for b in range(B):
                out[b] = cv2.distanceTransform(data_np[b], cv2.DIST_L2, 0)
            return torch.from_numpy(out).to(data.device)

        edt_module.edt_triton = edt_cv2
    except ImportError:
        pass


def _patch_cc():
    try:
        from sam3.perflib import connected_components as cc_module
        from skimage.measure import label

        def cc_skimage(mask):
            mask_np = mask.cpu().numpy().astype(np.uint8).squeeze(1)
            B = mask_np.shape[0]
            labels = np.zeros_like(mask_np, dtype=np.int32)
            areas = np.zeros_like(mask_np, dtype=np.int32)
            for b in range(B):
                lbl, n = label(mask_np[b], return_num=True, connectivity=2)
                labels[b] = lbl
                for i in range(1, n + 1):
                    areas[b][lbl == i] = (lbl == i).sum()
            labels_t = torch.from_numpy(labels).unsqueeze(1).to(mask.device)
            areas_t = torch.from_numpy(areas).unsqueeze(1).to(mask.device)
            return labels_t, areas_t

        cc_module.connected_components = cc_skimage
    except ImportError:
        pass


_patch_edt()
_patch_cc()


def _bboxes_near(a, b, tol=2.0):
    if a is None or b is None:
        return False
    return all(abs(float(a[i]) - float(b[i])) <= tol for i in range(4))


def _sam_bbox_temporally_consistent(sam_bbox, previous_bbox, max_disp_ratio=None):
    """Reuse IR temporal gate: SAM center must stay within existing max displacement."""
    if previous_bbox is None or sam_bbox is None:
        return True
    box_w = max(float(sam_bbox[2]) - float(sam_bbox[0]), MIN_BOX_SIDE_PX)
    box_h = max(float(sam_bbox[3]) - float(sam_bbox[1]), MIN_BOX_SIDE_PX)
    ratio = IR_REFINE_MAX_TEMPORAL_DISP_RATIO if max_disp_ratio is None else max_disp_ratio
    max_temporal_disp = max(box_w, box_h) * ratio
    prev_cx, prev_cy = bbox_center(previous_bbox)
    sam_cx, sam_cy = bbox_center(sam_bbox)
    temporal_disp = ((sam_cx - prev_cx) ** 2 + (sam_cy - prev_cy) ** 2) ** 0.5
    return temporal_disp <= max_temporal_disp


def serialize_preload_timeline(cache, base_frame):
    """Build G0 preload_timeline[] from frame_cache (bbox/lost only, no secrets)."""
    timeline = []
    past_first_loss = False
    for relative_frame in range(len(cache)):
        entry = cache.get(relative_frame)
        if entry is None:
            raise ValidationError(f"SAM3 cache missing relative frame {relative_frame}")
        lost = bool(entry.get("lost"))
        bbox = entry.get("bbox")
        if lost or bbox is None:
            if past_first_loss:
                bbox_status = "skipped"
            else:
                bbox_status = "lost"
                past_first_loss = True
            points = None
        else:
            bbox_status = "valid"
            points = [float(v) for v in bbox]
        timeline.append({
            "frame": int(base_frame) + int(relative_frame),
            "relative_frame": int(relative_frame),
            "bbox_status": bbox_status,
            "points": points,
            "lost": lost or bbox is None,
        })
    return timeline


def serialize_preload_summary(sess):
    base_frame = int(sess["base_frame"])
    preloaded_count = int(sess["preloaded_count"])
    return {
        "base_frame": base_frame,
        "preloaded_count": preloaded_count,
        "preloaded_until_frame": int(
            sess.get("preloaded_until_frame", base_frame + preloaded_count - 1)
        ),
    }


class ModelHandler:
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config if config is not None else load_sam3_config()
        self.predictor = build_sam3_predictor(version="sam3", compile=False)
        self._apply_detection_score_threshold()
        self._sessions = {}

    def _apply_detection_score_threshold(self):
        """Apply the configured SAM3 VG detection gate to the built model.

        ``build_sam3_video_model`` hard-codes ``score_threshold_detection=0.5``
        and the handler cannot pass it as a build argument, but
        ``Sam3VideoBase.run_backbone_and_detection`` reads the attribute on
        every frame, so setting it after build is sufficient.
        """
        threshold = float(
            getattr(
                self.config,
                "detection_score_threshold",
                DEFAULT_DETECTION_SCORE_THRESHOLD,
            )
        )
        model = getattr(self.predictor, "model", None)
        if model is not None and hasattr(model, "score_threshold_detection"):
            model.score_threshold_detection = threshold

    def _new_session_record(self):
        return {
            "temp_dir": tempfile.mkdtemp(prefix="sam3_"),
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
            "ui_emit_window": [],
            "ui_emit_prev_center": None,
        }

    def _register_session(self, key, sess):
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = next(iter(self._sessions))
            self._destroy_session(oldest)
        self._sessions[key] = sess

    def _cleanup_session_workspace(self, sess):
        self._close_predictor_session(sess)
        temp_dir = sess.get("temp_dir")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            sess["temp_dir"] = None
        sess["session_id"] = None

    def _allocate_init_session_key(self, states):
        if states and len(states) > 0 and isinstance(states[0], dict):
            key = states[0].get("session_key")
            if key:
                if key in self._sessions:
                    raise ValidationError(
                        f"session_key {key!r} already exists; re-seed tracking from the annotation frame"
                    )
                raise SessionStaleError(
                    f"session_key {key!r} is stale; re-seed tracking from the annotation frame"
                )
        return uuid.uuid4().hex[:12]

    def _make_session(self):
        key = uuid.uuid4().hex[:12]
        self._register_session(key, self._new_session_record())
        return key

    def _get_key(self, states):
        if states and len(states) > 0 and isinstance(states[0], dict):
            key = states[0].get("session_key")
            if key:
                if key not in self._sessions:
                    raise SessionStaleError(
                        f"session_key {key!r} is stale; re-seed tracking from the annotation frame"
                    )
                return key
        return self._make_session()

    def _close_predictor_session(self, sess):
        if not sess["session_id"]:
            return
        try:
            self.predictor.handle_request(request={
                "type": "close_session",
                "session_id": sess["session_id"],
                "run_gc_collect": False,
            })
        except Exception:
            pass
        sess["session_id"] = None

    def _destroy_session(self, key):
        sess = self._sessions.pop(key, None)
        if sess is None:
            return
        self._close_predictor_session(sess)
        temp_dir = sess.get("temp_dir")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _output_to_bbox(self, outputs, sess, reference_bbox):
        return select_bbox_from_outputs(
            outputs,
            sess["image_height"],
            sess["image_width"],
            reference_bbox,
            self.config,
        )

    def _maybe_refine_bbox(self, image, bbox, previous_bbox=None):
        if not self.config.ir_refine_enabled or bbox is None:
            return bbox
        if (
            self.config.ir_stop_on_missing
            and previous_bbox is not None
            and not _sam_bbox_temporally_consistent(bbox, previous_bbox)
        ):
            return refine_bbox_with_ir_intensity(
                image,
                bbox,
                previous_bbox=previous_bbox,
                stop_on_missing=True,
            )
        return refine_bbox_with_ir_intensity(
            image,
            bbox,
            previous_bbox=previous_bbox,
            stop_on_missing=False,
        )

    def _build_state(
        self,
        sess_key,
        bbox,
        prompt_bbox,
        prev_state=None,
        lost=False,
        *,
        preload_meta=None,
    ):
        state = {
            "session_key": sess_key,
            "obj_id": 1 if prev_state is None else prev_state.get("obj_id", 1),
            "last_bbox": bbox,
            "prompt_bbox": prompt_bbox,
        }
        if lost:
            state["lost"] = True
        if preload_meta:
            state.update(preload_meta)
        elif prev_state:
            for key in (
                "base_frame",
                "preloaded_count",
                "preloaded_until_frame",
            ):
                if key in prev_state:
                    state[key] = prev_state[key]
        return state

    def _preload_state_fields(self, sess):
        return {
            "base_frame": sess["base_frame"],
            "preloaded_count": sess["preloaded_count"],
            "preloaded_until_frame": sess["preloaded_until_frame"],
        }

    def _decode_preload_image(self, image_b64):
        buf = io.BytesIO(base64.b64decode(image_b64))
        return np.array(Image.open(buf).convert("RGB"))

    def _save_preload_sequence(self, sess, preload_images_b64, preload_count):
        count = int(preload_count)
        if count <= 0:
            raise ValidationError("preload_frame_count must be positive")
        if count > SAM3_PRELOAD_CHUNK_CAP:
            raise ValidationError(
                f"preload_frame_count {count} exceeds SAM3_PRELOAD_CHUNK_CAP {SAM3_PRELOAD_CHUNK_CAP}"
            )
        if not preload_images_b64 or len(preload_images_b64) < count:
            raise ValidationError("preload_images length does not match preload_frame_count")

        first_image = self._decode_preload_image(preload_images_b64[0])
        h, w = first_image.shape[:2]
        sess["image_height"] = h
        sess["image_width"] = w

        for idx in range(count):
            image_array = self._decode_preload_image(preload_images_b64[idx])
            frame_path = os.path.join(sess["temp_dir"], f"{idx:05d}.jpg")
            Image.fromarray(image_array).save(frame_path)

        sess["frame_count"] = count
        sess["loaded_frame_count"] = count

    def _load_preloaded_frame(self, sess, relative_frame):
        temp_dir = sess.get("temp_dir")
        if not temp_dir:
            raise ValidationError(
                f"preloaded frame {relative_frame} is missing from session storage"
            )
        frame_path = os.path.join(temp_dir, f"{int(relative_frame):05d}.jpg")
        if not os.path.exists(frame_path):
            raise ValidationError(
                f"preloaded frame {relative_frame} is missing from session storage"
            )
        return np.array(Image.open(frame_path).convert("RGB"))

    def _shape_to_bbox(self, shape):
        if shape is None:
            return None
        if isinstance(shape, dict):
            points = shape.get("points")
            if points is not None:
                return [float(v) for v in points]
        return [float(v) for v in shape]

    def _first_seed_shape(self, shapes):
        if not shapes:
            return None
        for shape in shapes:
            if shape is not None:
                return self._shape_to_bbox(shape)
        return None

    def _can_continue_without_shapes(self, states):
        if not states or not isinstance(states[0], dict):
            return False
        key = states[0].get("session_key")
        if not key or key not in self._sessions:
            return False
        return bool(self._sessions[key].get("cache_ready"))

    def _propagate_full_chunk(self, sess, preload_count):
        outputs_by_frame = {}
        max_track = max(int(preload_count) - 1, 0)
        request = {
            "type": "propagate_in_video",
            "session_id": sess["session_id"],
            "start_frame_index": 0,
            "max_frame_num_to_track": max_track,
            "propagation_direction": "forward",
        }
        if self.config.prompt_mode == "box":
            request["force_tracker_propagation"] = True
        for response in self.predictor.handle_stream_request(request=request):
            frame_idx = response.get("frame_index")
            if frame_idx is not None:
                outputs_by_frame[int(frame_idx)] = response.get("outputs")
        return outputs_by_frame

    def _propagate_range(self, sess, start_frame, end_frame):
        """Forward propagation for one index range after a mid-rally re-prompt."""
        outputs_by_frame = {}
        max_track = max(int(end_frame) - int(start_frame), 0)
        request = {
            "type": "propagate_in_video",
            "session_id": sess["session_id"],
            "start_frame_index": int(start_frame),
            "max_frame_num_to_track": max_track,
            "propagation_direction": "forward",
        }
        if self.config.prompt_mode == "box":
            request["force_tracker_propagation"] = True
        for response in self.predictor.handle_stream_request(request=request):
            frame_idx = response.get("frame_index")
            if frame_idx is not None:
                outputs_by_frame[int(frame_idx)] = response.get("outputs")
        return outputs_by_frame

    def _stable_diameter_map(self, outputs_by_frame, frame_count):
        """Per-frame median minor extent of nearby non-elongated masks."""
        window = max(int(self.config.motion_blur_diameter_window), 0)
        stable = {}
        for relative_frame in range(int(frame_count)):
            masks = []
            for neighbour in range(relative_frame - window, relative_frame + window + 1):
                outputs = outputs_by_frame.get(neighbour)
                if not outputs:
                    continue
                candidates = outputs.get("out_binary_masks")
                if candidates is None:
                    continue
                for mask in candidates:
                    masks.append(mask)
            stable[relative_frame] = median_sharp_diameter(
                masks,
                max_ratio=self.config.motion_blur_min_eigen_ratio,
                min_mask_pixels=self.config.motion_blur_min_mask_pixels,
            )
        return stable

    def _motion_blur_target_bbox(
        self,
        sess,
        frame_image,
        outputs,
        reference_bbox,
        previous_outputs,
        next_outputs,
        relative_frame=None,
        diag_meta=None,
    ):
        """Emitted square centred on the motion-aware target.

        Falls back to the current-mask centroid square when the estimator
        cannot correct, and returns no box (canonical bbox kept) when even the
        mask is unusable.  Canonical tracking state is never changed here.
        """
        current_mask = select_mask_from_outputs(outputs, reference_bbox)
        previous_centroid = mask_centroid(
            select_mask_from_outputs(previous_outputs, reference_bbox)
        )
        next_centroid = mask_centroid(
            select_mask_from_outputs(next_outputs, reference_bbox)
        )
        center_xy = None
        if self.config.motion_blur_intensity_center:
            center_xy = intensity_weighted_centroid(frame_image, current_mask)
        diameter_override = None
        if self.config.motion_blur_stabilized_diameter:
            diameter_override = sess.get("motion_blur_stable_diameter", {}).get(
                relative_frame
            )
        estimate = estimate_motion_blur_center(
            current_mask,
            previous_centroid,
            next_centroid,
            min_eigen_ratio=self.config.motion_blur_min_eigen_ratio,
            min_mask_pixels=self.config.motion_blur_min_mask_pixels,
            target_fraction=self.config.motion_blur_target_fraction,
            center_xy=center_xy,
            diameter_override=diameter_override,
        )
        diag = motion_blur_report_row(estimate, frame_id=relative_frame)
        diag["center_source"] = "intensity" if center_xy is not None else "mask"
        log_motion_blur_diagnostics(
            diag, relative_frame=relative_frame, diag_meta=diag_meta
        )
        if estimate.target_xy is None:
            return None, diag
        target_x, target_y = estimate.target_xy
        if current_mask is not None and self._is_rod_like_mask(current_mask):
            applied = False
            if self.config.rod_mask_blob_reanchor and self._is_thin_mask(
                current_mask
            ):
                blob = self._dark_blob_above_mask(frame_image, current_mask)
                if blob is not None:
                    target_x, target_y = blob
                    diag["rod_blob_reanchor"] = True
                    applied = True
            if not applied and self.config.rod_mask_anchor:
                anchor = self._rod_mask_head_center(current_mask)
                if anchor is not None:
                    target_x, target_y = anchor
                    diag["rod_mask_anchor"] = True
        side = max(
            float(estimate.diameter_override or estimate.diameter or 0.0),
            MIN_BOX_SIDE_PX,
        )
        bbox = square_bbox_from_center(
            target_x,
            target_y,
            side,
            sess["image_height"],
            sess["image_width"],
        )
        return bbox, diag

    def _mask_width(self, mask):
        points = mask_points(mask)
        if points is None or len(points) == 0:
            return None
        return float(points[:, 0].max() - points[:, 0].min()) + 1.0

    def _is_thin_mask(self, mask):
        """Thin strip (rod width); excludes blurred-ball streaks (4+ px)."""
        width = self._mask_width(mask)
        if width is None:
            return False
        return width <= float(self.config.rod_mask_max_width)

    def _is_rod_like_mask(self, mask):
        """True for a vertically elongated mask (rod/ball-on-rod strip)."""
        points = mask_points(mask)
        if points is None or len(points) == 0:
            return False
        width = float(points[:, 0].max() - points[:, 0].min()) + 1.0
        height = float(points[:, 1].max() - points[:, 1].min()) + 1.0
        return height >= width * float(self.config.rod_mask_min_aspect)

    def _dark_blob_above_mask(self, image, mask):
        """Dark blob centroid just above a rod-like mask (the launched ball).

        The tracker can latch onto the rod under a dark ball resting on the
        launcher post; the ball is then the only dark blob above the rod tip.
        Returns ``None`` when the window holds no ball-sized dark blob.
        """
        points = mask_points(mask)
        if points is None or len(points) == 0:
            return None
        gray = image_to_grayscale(image)
        image_height, image_width = gray.shape[:2]
        top_y = float(points[:, 1].min())
        center_x = float(points[:, 0].mean())
        search = max(int(self.config.rod_blob_search_px), 4)
        x0 = max(0, int(center_x) - 8)
        x1 = min(image_width, int(center_x) + 9)
        y0 = max(0, int(top_y) - search)
        y1 = max(0, min(int(top_y) - 1, image_height))
        if x1 <= x0 or y1 <= y0:
            return None
        window = gray[y0:y1, x0:x1]
        background = float(np.median(window))
        dark = (window < (background - float(self.config.rod_blob_margin))).astype(
            np.uint8
        )
        count = int(dark.sum())
        if count < ROD_BLOB_MIN_PIXELS or count > ROD_BLOB_MAX_PIXELS:
            return None
        components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8
        )
        best = None
        for index in range(1, components):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < ROD_BLOB_MIN_PIXELS or area > ROD_BLOB_MAX_PIXELS:
                continue
            blob_h = int(stats[index, cv2.CC_STAT_HEIGHT])
            blob_w = int(stats[index, cv2.CC_STAT_WIDTH])
            if max(blob_h, blob_w) > 2 * max(min(blob_h, blob_w), 1) + 4:
                continue
            component = window[labels == index]
            if float(component.min()) > background - 2.0 * float(
                self.config.rod_blob_margin
            ):
                continue  # only noise-level darkness: skip
            centroid_y = centroids[index][1]
            if best is None or centroid_y < best[0]:
                best = (
                    centroid_y,
                    float(centroids[index][0]) + x0,
                    float(centroid_y) + y0,
                )
        if best is None:
            return None
        return best[1], best[2]

    def _mask_band_widths(self, mask):
        """Width of the mask's top and bottom thirds, in pixels."""
        points = mask_points(mask)
        if points is None or len(points) == 0:
            return None
        ys = points[:, 1]
        top = float(ys.min())
        bottom = float(ys.max())
        span = max(bottom - top + 1.0, 1.0)
        head = points[ys <= top + span * 0.3]
        tail = points[ys >= bottom - span * 0.3]
        if len(head) == 0 or len(tail) == 0:
            return None
        head_width = float(head[:, 0].max() - head[:, 0].min()) + 1.0
        tail_width = float(tail[:, 0].max() - tail[:, 0].min()) + 1.0
        return head_width, tail_width

    def _rod_mask_head_center(self, mask):
        """Top-edge centre of a rod-like mask whose head is a ball.

        Accepts thin strips (the rod alone) and lollipop masks (ball merged
        with the rod below it, head clearly wider than the tail).
        """
        points = mask_points(mask)
        if points is None or len(points) == 0:
            return None
        width = float(points[:, 0].max() - points[:, 0].min()) + 1.0
        height = float(points[:, 1].max() - points[:, 1].min()) + 1.0
        if height < width * float(self.config.rod_mask_min_aspect):
            return None
        if width > float(self.config.rod_mask_max_width):
            bands = self._mask_band_widths(mask)
            if bands is None:
                return None
            head_width, tail_width = bands
            ratio = head_width / max(tail_width, 1.0)
            if ratio < float(self.config.rod_mask_lollipop_ratio):
                return None
        centroid_x = float(points[:, 0].mean())
        head = points[points[:, 1] <= float(points[:, 1].min()) + 2.0]
        if len(head) > 0:
            centroid_x = float(head[:, 0].mean())
        top_y = float(points[:, 1].min()) + 0.5
        return centroid_x, top_y

    def _postprocess_frame_outputs(
        self,
        sess,
        frame_image,
        outputs,
        *,
        is_init_frame,
        prompt_bbox,
        last_known_bbox,
        prev_lost,
        relative_frame=None,
        diag_meta=None,
        previous_accepted_target_side=None,
        latent_prior_state=None,
        previous_outputs=None,
        next_outputs=None,
        lost_reason_out=None,
    ):
        if prev_lost and not self.config.recover_after_miss:
            if lost_reason_out is not None:
                lost_reason_out.append("sticky_prev_lost")
            return None, True, None, None

        reference_bbox = prompt_bbox if is_init_frame else last_known_bbox
        if self.config.output_policy in ADAPTIVE_COMPONENT_POLICIES or (
            self.config.output_policy == "motion_blur_center_box"
        ):
            center_cfg = replace(self.config, output_policy="mask_center_box")
            bbox = select_bbox_from_outputs(
                outputs,
                sess["image_height"],
                sess["image_width"],
                reference_bbox,
                center_cfg,
            )
        else:
            bbox = self._output_to_bbox(outputs, sess, reference_bbox)

        raw_sam_bbox = bbox

        if bbox is None:
            if is_init_frame:
                bbox = prompt_bbox
            else:
                log_lost_frame_diagnostics(
                    reason="output_none",
                    relative_frame=relative_frame,
                    sess=sess,
                    outputs=outputs,
                    prompt_bbox=prompt_bbox,
                    last_known_bbox=last_known_bbox,
                    raw_sam_bbox=None,
                    refined_bbox=None,
                    config=self.config,
                    diag_meta=diag_meta,
                )
                if lost_reason_out is not None:
                    lost_reason_out.append("output_none")
                return None, True, None, None

        bbox = self._maybe_refine_bbox(
            frame_image,
            bbox,
            previous_bbox=None if is_init_frame else last_known_bbox,
        )
        if bbox is None:
            if is_init_frame:
                bbox = prompt_bbox
            else:
                log_lost_frame_diagnostics(
                    reason="refine_none",
                    relative_frame=relative_frame,
                    sess=sess,
                    outputs=outputs,
                    prompt_bbox=prompt_bbox,
                    last_known_bbox=last_known_bbox,
                    raw_sam_bbox=raw_sam_bbox,
                    refined_bbox=None,
                    config=self.config,
                    diag_meta=diag_meta,
                )
                if lost_reason_out is not None:
                    lost_reason_out.append("refine_none")
                return None, True, None, None

        if (
            not is_init_frame
            and prompt_bbox is not None
            and last_known_bbox is not None
            and _bboxes_near(bbox, prompt_bbox)
            and not _bboxes_near(last_known_bbox, prompt_bbox)
            and not _sam_bbox_temporally_consistent(
                bbox, last_known_bbox, FALSE_RELOCK_MAX_DISP_RATIO
            )
        ):
            log_lost_frame_diagnostics(
                reason="false_relock",
                relative_frame=relative_frame,
                sess=sess,
                outputs=outputs,
                prompt_bbox=prompt_bbox,
                last_known_bbox=last_known_bbox,
                raw_sam_bbox=raw_sam_bbox,
                refined_bbox=bbox,
                config=self.config,
                diag_meta=diag_meta,
            )
            if lost_reason_out is not None:
                lost_reason_out.append("false_relock")
            return None, True, None, None

        canonical_tracking_bbox = list(bbox)
        emitted_bbox = canonical_tracking_bbox
        adaptive_diag = None
        if self.config.output_policy == "motion_blur_center_box":
            motion_bbox, adaptive_diag = self._motion_blur_target_bbox(
                sess,
                frame_image,
                outputs,
                reference_bbox,
                previous_outputs,
                next_outputs,
                relative_frame=relative_frame,
                diag_meta=diag_meta,
            )
            if motion_bbox is not None:
                emitted_bbox = motion_bbox
        elif self.config.output_policy == "adaptive_component_square_box":
            emitted_bbox, adaptive_diag = apply_adaptive_component_square_box(
                frame_image,
                canonical_tracking_bbox,
                prompt_bbox,
                None if is_init_frame else last_known_bbox,
                is_init_frame,
                self.config,
                previous_accepted_target_side=previous_accepted_target_side,
            )
            if adaptive_diag is not None:
                can_cx, can_cy = bbox_center(canonical_tracking_bbox)
                emit_cx, emit_cy = bbox_center(emitted_bbox)
                adaptive_diag["canonical_tracking_bbox"] = canonical_tracking_bbox
                adaptive_diag["canonical_center_x"] = can_cx
                adaptive_diag["canonical_center_y"] = can_cy
                adaptive_diag["emitted_bbox"] = emitted_bbox
                adaptive_diag["emitted_center_x"] = emit_cx
                adaptive_diag["emitted_center_y"] = emit_cy
                log_adaptive_scale_diagnostics(
                    adaptive_diag,
                    relative_frame=relative_frame,
                    diag_meta=diag_meta,
                )
        elif self.config.output_policy == "adaptive_component_deadband_box":
            emitted_bbox, adaptive_diag = apply_adaptive_component_deadband_box(
                frame_image,
                canonical_tracking_bbox,
                prompt_bbox,
                None if is_init_frame else last_known_bbox,
                is_init_frame,
                self.config,
                previous_accepted_target_side=previous_accepted_target_side,
            )
            if adaptive_diag is not None:
                can_cx, can_cy = bbox_center(canonical_tracking_bbox)
                emit_cx, emit_cy = bbox_center(emitted_bbox)
                adaptive_diag["canonical_center_x"] = can_cx
                adaptive_diag["canonical_center_y"] = can_cy
                adaptive_diag["emitted_bbox"] = emitted_bbox
                adaptive_diag["emitted_center_x"] = emit_cx
                adaptive_diag["emitted_center_y"] = emit_cy
                log_adaptive_scale_diagnostics(
                    adaptive_diag,
                    relative_frame=relative_frame,
                    diag_meta=diag_meta,
                )
        elif self.config.output_policy == "adaptive_component_latent_prior_box":
            emitted_bbox, adaptive_diag = apply_adaptive_component_latent_prior_box(
                frame_image,
                canonical_tracking_bbox,
                prompt_bbox,
                None if is_init_frame else last_known_bbox,
                is_init_frame,
                self.config,
                latent_prior_state=latent_prior_state,
            )
            if adaptive_diag is not None:
                can_cx, can_cy = bbox_center(canonical_tracking_bbox)
                emit_cx, emit_cy = bbox_center(emitted_bbox)
                adaptive_diag["canonical_center_x"] = can_cx
                adaptive_diag["canonical_center_y"] = can_cy
                adaptive_diag["emitted_bbox"] = emitted_bbox
                adaptive_diag["emitted_center_x"] = emit_cx
                adaptive_diag["emitted_center_y"] = emit_cy
                log_adaptive_scale_diagnostics(
                    adaptive_diag,
                    relative_frame=relative_frame,
                    diag_meta=diag_meta,
                )

        if not is_init_frame:
            emitted_bbox = self._maybe_apply_edge_hold(
                emitted_bbox,
                last_known_bbox,
                sess,
            )
        if (
            self.config.ui_gate_clamp
            and emitted_bbox is not None
            and not sess.get("_cache_build")
        ):
            emitted_bbox = self._clamp_emitted_to_ui_gate(
                sess,
                emitted_bbox,
                prompt_bbox,
                is_init_frame,
            )

        return emitted_bbox, False, adaptive_diag, canonical_tracking_bbox

    def _clamp_emitted_to_ui_gate(self, sess, emitted_bbox, prompt_bbox, is_init_frame):
        """Keep the emitted box inside the frontend ball-geometry envelope.

        The Track Rallies staging discards a segment when a box violates the
        median-of-8 area/aspect/displacement/velocity limits (launch
        acceleration or the ball leaving the frame). Clamping only the emitted
        geometry keeps the internal tracking reference untouched.
        """
        bbox = [float(value) for value in emitted_bbox]
        prompt = [float(value) for value in prompt_bbox]
        if is_init_frame or not sess.get("ui_emit_window"):
            window = [prompt]
            sess["ui_emit_window"] = window
            sess["ui_emit_prev_center"] = bbox_center(prompt)
            reference = prompt
        else:
            window = list(sess["ui_emit_window"])
            reference = window[-1]
        areas = [
            max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
            for box in window
        ]
        ref_area = float(np.median(areas)) if areas else 1.0
        diagonals = [
            max(
                bbox_side_length(box) * (2.0 ** 0.5),
                MIN_BOX_SIDE_PX,
            )
            for box in window
        ]
        ref_diagonal = float(np.median(diagonals)) if diagonals else MIN_BOX_SIDE_PX
        ref_cx, ref_cy = bbox_center(reference)
        max_step = ref_diagonal * UI_GATE_CLAMP_MAX_DISP_DIAGONALS
        prev_center = sess.get("ui_emit_prev_center")
        if prev_center is not None:
            prev_step = (
                (ref_cx - float(prev_center[0])) ** 2
                + (ref_cy - float(prev_center[1])) ** 2
            ) ** 0.5
            if prev_step > 1.0:
                max_step = min(
                    max_step, prev_step * UI_GATE_CLAMP_MAX_VELOCITY_RESIDUAL
                )
        cand_cx, cand_cy = bbox_center(bbox)
        step = ((cand_cx - ref_cx) ** 2 + (cand_cy - ref_cy) ** 2) ** 0.5
        if step > max_step and step > 0.0:
            scale = max_step / step
            cand_cx = ref_cx + (cand_cx - ref_cx) * scale
            cand_cy = ref_cy + (cand_cy - ref_cy) * scale
        side = bbox_side_length(bbox)
        area = side * side
        max_area = ref_area * UI_GATE_CLAMP_MAX_AREA_RATIO
        min_area = ref_area * UI_GATE_CLAMP_MIN_AREA_RATIO
        if area > max_area:
            side = max_area ** 0.5
        elif area < min_area:
            side = min_area ** 0.5
        side = max(float(side), MIN_BOX_SIDE_PX)
        clamped = square_bbox_from_center(
            cand_cx,
            cand_cy,
            side,
            sess["image_height"],
            sess["image_width"],
        )
        if not is_init_frame:
            verdict = self._ui_gate_verdict(
                clamped, window, sess.get("ui_emit_prev_center")
            )
            if verdict == "valid":
                window = (window + [clamped])[-UI_GATE_EMIT_WINDOW:]
                sess["ui_emit_window"] = window
                sess["ui_emit_prev_center"] = (cand_cx, cand_cy)
        return clamped

    def _ui_gate_verdict(self, candidate, window, prev_center):
        """Mirror the frontend rally staging verdict for one emitted box.

        The staging appends only boxes it accepts as valid to its reference
        window, so the clamp window must skip soft frames the same way or the
        two medians drift apart and the staging sees growth the clamp allowed.
        """
        if len(window) < 2:
            return "valid"
        reference = window[-1]
        areas = [
            max(
                1.0,
                (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])),
            )
            for box in window
        ]
        diagonals = [
            max(bbox_side_length(box) * (2.0 ** 0.5), MIN_BOX_SIDE_PX)
            for box in window
        ]
        ref_area = max(1.0, float(np.median(areas)))
        ref_diagonal = max(float(np.median(diagonals)), float(MIN_BOX_SIDE_PX))
        side = bbox_side_length(candidate)
        area_ratio = (side * side) / ref_area
        cand_cx, cand_cy = bbox_center(candidate)
        ref_cx, ref_cy = bbox_center(reference)
        distance = ((cand_cx - ref_cx) ** 2 + (cand_cy - ref_cy) ** 2) ** 0.5
        displacement_scale = distance / ref_diagonal
        if (
            area_ratio < UI_GATE_HARD_AREA_MIN_RATIO
            or area_ratio > UI_GATE_HARD_AREA_MAX_RATIO
            or displacement_scale > UI_GATE_HARD_MAX_DISP_SCALE
        ):
            return "hard"
        soft = 0
        if (
            area_ratio < UI_GATE_SOFT_AREA_MIN_RATIO
            or area_ratio > UI_GATE_SOFT_AREA_MAX_RATIO
        ):
            soft += 1
        if displacement_scale > UI_GATE_SOFT_MAX_DISP_SCALE:
            soft += 1
        if prev_center is not None:
            expected = (
                (ref_cx - float(prev_center[0])) ** 2
                + (ref_cy - float(prev_center[1])) ** 2
            ) ** 0.5
            if expected > 1.0 and distance > expected * UI_GATE_MAX_VELOCITY_RESIDUAL:
                soft += 1
        if soft >= 2:
            return "hard"
        return "soft" if soft == 1 else "valid"

    def _apply_ui_gate_clamp(self, cache, preload_count, sess, prompt_bbox):
        """Cap the emitted cache boxes to the frontend rally staging envelope.

        Runs over the final timeline: the staging reference window then holds
        exactly these boxes in this order, so no emitted frame trips the
        staging geometry checks; internal tracking references stay untouched.
        """
        if not self.config.ui_gate_clamp:
            return 0
        prompt = [float(value) for value in prompt_bbox]
        sess["ui_emit_window"] = [prompt]
        sess["ui_emit_prev_center"] = bbox_center(prompt)
        capped_count = 0
        for frame in range(1, int(preload_count)):
            entry = cache.get(frame) or {}
            box = entry.get("bbox")
            if entry.get("lost") or box is None:
                continue
            clamped = self._clamp_emitted_to_ui_gate(sess, box, prompt, False)
            if any(
                abs(float(a) - float(b)) > 1e-6 for a, b in zip(box, clamped)
            ):
                capped_count += 1
            entry["bbox"] = clamped
        return capped_count

    def _should_hold_trailing_frame(self, relative_frame, preload_count, prev_bbox):
        """True when a miss should hold the last box at the end of the chunk."""
        if not self.config.recover_after_miss or self.config.trailing_hold_frames <= 0:
            return False
        if prev_bbox is None:
            return False
        return int(relative_frame) >= int(preload_count) - int(
            self.config.trailing_hold_frames
        )

    def _interpolate_miss_gaps(self, cache, preload_count, sess):
        """Fill short internal gaps with linear box interpolation.

        The UI stops a rally on any frame without a box and needs a box on
        every frame, so a short dropout is filled rather than marked lost.
        Gaps longer than ``recover_max_gap`` keep today's loss behaviour.
        """
        if not self.config.recover_after_miss or self.config.recover_max_gap <= 0:
            return []
        max_gap = int(self.config.recover_max_gap)
        height = sess.get("image_height")
        width = sess.get("image_width")
        if height is None or width is None:
            return []
        interpolated = []
        frame = 1
        while frame < int(preload_count):
            entry = cache.get(frame) or {}
            if not entry.get("lost"):
                frame += 1
                continue
            start = frame
            end = start
            while end + 1 < int(preload_count) and (cache.get(end + 1) or {}).get(
                "lost"
            ):
                end += 1
            gap = end - start + 1
            before = cache.get(start - 1) if start > 0 else None
            after = cache.get(end + 1) if end + 1 < int(preload_count) else None
            if (
                gap <= max_gap
                and before is not None
                and after is not None
                and not before.get("lost")
                and not after.get("lost")
                and before.get("bbox") is not None
                and after.get("bbox") is not None
            ):
                box_before = [float(value) for value in before["bbox"]]
                box_after = [float(value) for value in after["bbox"]]
                start_cx, start_cy = bbox_center(box_before)
                end_cx, end_cy = bbox_center(box_after)
                start_side = bbox_side_length(box_before)
                end_side = bbox_side_length(box_after)
                for index in range(start, end + 1):
                    fraction = (index - start + 1) / (gap + 1.0)
                    center_x = start_cx + (end_cx - start_cx) * fraction
                    center_y = start_cy + (end_cy - start_cy) * fraction
                    side = start_side + (end_side - start_side) * fraction
                    cache[index] = {
                        "bbox": square_bbox_from_center(
                            center_x, center_y, side, height, width
                        ),
                        "lost": False,
                        "interpolated": True,
                    }
                    interpolated.append(index)
            frame = end + 1
        if interpolated and diag_enabled():
            log_frame_record({
                "event": "interpolated_frames",
                "relativeFrames": list(interpolated),
                "maxGap": max_gap,
            })
        return interpolated

    def _output_reference_box(self, outputs, height, width, reference):
        """Mask-centroid square for one frame's outputs, or None when empty."""
        mask = select_mask_from_outputs(outputs, reference)
        if mask is None:
            return None
        centroid = mask_centroid(mask)
        if centroid is None:
            return None
        side = bbox_side_length(reference)
        return square_bbox_from_center(
            centroid[0], centroid[1], side, height, width
        )

    def _reseed_after_long_miss(self, sess, outputs_by_frame, preload_count):
        """Re-prompt the tracker after long model dropouts.

        When the model returns no object for more than ``recover_max_gap``
        consecutive frames, the tracker is re-prompted with the last accepted
        box at the first missed frame and re-propagates forward, so the dropout
        is filled by model output instead of a straight-line interpolation.
        Returns the number of re-prompts performed.
        """
        if not self.config.reseed_after_miss:
            return 0
        height = sess.get("image_height")
        width = sess.get("image_width")
        prompt_bbox = sess.get("prompt_bbox")
        if height is None or width is None or prompt_bbox is None:
            return 0
        frame_count = int(preload_count)
        max_gap = max(int(self.config.recover_max_gap), 0)
        budget = max(int(self.config.reseed_max_attempts), 1)
        reference = [float(value) for value in prompt_bbox]
        reseeds = 0
        frame = 1
        while frame < frame_count and reseeds < budget:
            box = self._output_reference_box(
                outputs_by_frame.get(frame), height, width, reference
            )
            if box is not None:
                reference = box
                frame += 1
                continue
            start = frame
            end = start
            while end + 1 < frame_count:
                probe = self._output_reference_box(
                    outputs_by_frame.get(end + 1), height, width, reference
                )
                if probe is not None:
                    break
                end += 1
            if end - start + 1 <= max_gap or start == 0:
                frame = end + 1
                continue
            reseeds += 1
            self._add_prompt(sess, start, bbox=reference)
            refreshed = self._propagate_range(
                sess, start, frame_count - 1
            )
            for index, outputs in refreshed.items():
                if index >= start:
                    outputs_by_frame[index] = outputs
            if diag_enabled():
                log_frame_record({
                    "event": "reseed_after_miss",
                    "startFrame": int(start),
                    "gapFrames": int(end - start + 1),
                    "referenceBbox": bbox_summary(reference),
                    "refreshedFrames": len(refreshed),
                })
            frame = start
        return reseeds

    def _maybe_apply_edge_hold(self, emitted_bbox, reference_bbox, sess):
        """Fail-soft near image borders: hold the last accepted box.

        A mask that reaches the image border can produce an edge-clamped box
        whose centre is far outside the last accepted box; the UI geometry gate
        rejects such a frame and discards the whole segment. When enabled, keep
        the last accepted box for that frame instead.
        """
        if (
            not self.config.edge_hold_enabled
            or emitted_bbox is None
            or reference_bbox is None
        ):
            return emitted_bbox
        height = float(sess.get("image_height") or 0.0)
        width = float(sess.get("image_width") or 0.0)
        if height <= 0.0 or width <= 0.0:
            return emitted_bbox
        x1, y1, x2, y2 = [float(value) for value in emitted_bbox]
        touches_border = (
            x1 <= 0.0 or y1 <= 0.0 or x2 >= width - 1.0 or y2 >= height - 1.0
        )
        if not touches_border:
            return emitted_bbox
        ref_cx, ref_cy = bbox_center(reference_bbox)
        cand_cx, cand_cy = bbox_center(emitted_bbox)
        distance = ((cand_cx - ref_cx) ** 2 + (cand_cy - ref_cy) ** 2) ** 0.5
        if distance <= bbox_side_length(reference_bbox) * EDGE_HOLD_MAX_DISP_SIDES:
            return emitted_bbox
        return list(reference_bbox)

    def _build_frame_cache(
        self,
        sess,
        preload_images,
        preload_count,
        seed_shape,
    ):
        cache = {}
        try:
            self._save_preload_sequence(sess, preload_images, preload_count)
            sess["prompt_bbox"] = list(seed_shape)
            sess["session_id"] = self._start_session(sess)
            self._add_prompt(sess, 0)
            outputs_by_frame = self._propagate_full_chunk(sess, preload_count)
            self._reseed_after_long_miss(sess, outputs_by_frame, preload_count)
            if (
                self.config.output_policy == "motion_blur_center_box"
                and self.config.motion_blur_stabilized_diameter
            ):
                sess["motion_blur_stable_diameter"] = self._stable_diameter_map(
                    outputs_by_frame, preload_count
                )

            prev_bbox = None
            prev_lost = False
            prev_accepted_target_side = None
            prev_latent_prior_state = None
            prompt_bbox = sess["prompt_bbox"]
            lost_reason_out = []
            lost_count = 0
            last_emitted_bbox = None
            first_lost_relative_frame = None
            first_lost_reason = None

            sess["_cache_build"] = True
            for relative_frame in range(int(preload_count)):
                frame_image = self._load_preloaded_frame(sess, relative_frame)
                outputs = outputs_by_frame.get(relative_frame)
                is_init_frame = relative_frame == 0
                last_known_bbox = prev_bbox if prev_bbox is not None else prompt_bbox
                emitted_bbox, lost, adaptive_diag, canonical_tracking_bbox = (
                    self._postprocess_frame_outputs(
                        sess,
                        frame_image,
                        outputs,
                        is_init_frame=is_init_frame,
                        prompt_bbox=prompt_bbox,
                        last_known_bbox=last_known_bbox,
                        prev_lost=prev_lost,
                        relative_frame=relative_frame,
                        previous_accepted_target_side=prev_accepted_target_side,
                        latent_prior_state=prev_latent_prior_state,
                        previous_outputs=outputs_by_frame.get(relative_frame - 1),
                        next_outputs=outputs_by_frame.get(relative_frame + 1),
                        lost_reason_out=lost_reason_out,
                    )
                )
                if lost and self._should_hold_trailing_frame(
                    relative_frame, preload_count, prev_bbox
                ):
                    emitted_bbox = list(last_emitted_bbox or prev_bbox)
                    lost = False
                    lost_reason_out.clear()
                cache[relative_frame] = {"bbox": emitted_bbox, "lost": lost}
                if lost:
                    lost_count += 1
                    if first_lost_relative_frame is None:
                        first_lost_relative_frame = relative_frame
                        first_lost_reason = (
                            lost_reason_out[-1] if lost_reason_out else None
                        )
                    if self.config.recover_after_miss:
                        # Keep the last accepted box as the reference and let a
                        # later frame with usable output re-acquire it.
                        prev_lost = False
                    else:
                        prev_lost = True
                        prev_bbox = None
                        prev_accepted_target_side = None
                        prev_latent_prior_state = None
                else:
                    prev_lost = False
                    if canonical_tracking_bbox is not None:
                        prev_bbox = canonical_tracking_bbox
                    if self.config.output_policy == "adaptive_component_latent_prior_box":
                        if adaptive_diag is not None:
                            prev_latent_prior_state = _latent_prior_state_from_diag(
                                adaptive_diag
                            )
                    elif (
                        self.config.output_policy in ADAPTIVE_COMPONENT_POLICIES
                        and adaptive_diag is not None
                    ):
                        prev_accepted_target_side = adaptive_diag.get("accepted_target_side")
                if not lost and emitted_bbox is not None:
                    last_emitted_bbox = list(emitted_bbox)

            sess["_cache_build"] = False
            interpolated_frames = self._interpolate_miss_gaps(
                cache, preload_count, sess
            )
            clamped_frames = self._apply_ui_gate_clamp(
                cache, preload_count, sess, prompt_bbox
            )
            if clamped_frames and diag_enabled():
                log_frame_record({
                    "event": "ui_gate_clamp_summary",
                    "cappedFrames": int(clamped_frames),
                    "chunkLength": int(preload_count),
                })
            if interpolated_frames:
                lost_count = sum(
                    1 for index in range(int(preload_count)) if cache[index]["lost"]
                )
                if first_lost_relative_frame in interpolated_frames:
                    first_lost_relative_frame = next(
                        (
                            index
                            for index in range(int(preload_count))
                            if cache[index]["lost"]
                        ),
                        None,
                    )
                    first_lost_reason = None

            if diag_enabled():
                log_frame_record({
                    "event": "chunk_lost_summary",
                    "relativeFrame": first_lost_relative_frame,
                    "chunkLength": int(preload_count),
                    "preloadCount": int(preload_count),
                    "validCount": int(preload_count) - lost_count,
                    "lostCount": lost_count,
                    "firstLostRelativeFrame": first_lost_relative_frame,
                    "firstLostReason": first_lost_reason,
                })

            sess["frame_cache"] = cache
            sess["cache_ready"] = True
            return cache
        finally:
            self._cleanup_session_workspace(sess)

    def infer_batch(
        self,
        image,
        shapes,
        states,
        diag_meta=None,
        frame_index=None,
        preload_images=None,
        preload_base_frame=None,
        preload_frame_count=None,
        preload_payload_bytes=None,
        include_g0_extras=False,
    ):
        shapes = shapes or []
        prev_state = states[0] if states and isinstance(states[0], dict) else {}

        if preload_images is not None:
            shapes_out, states_out, g0_extras = self._infer_preload_init(
                shapes,
                states,
                diag_meta=diag_meta,
                frame_index=frame_index,
                preload_images=preload_images,
                preload_base_frame=preload_base_frame,
                preload_frame_count=preload_frame_count,
                preload_payload_bytes=preload_payload_bytes,
            )
            if include_g0_extras:
                return shapes_out, states_out, g0_extras
            return shapes_out, states_out

        if isinstance(prev_state, dict) and prev_state.get("preloaded_count") is not None:
            shapes_out, states_out = self._infer_preload_cached_track(
                states,
                diag_meta=diag_meta,
                frame_index=frame_index,
                preload_payload_bytes=preload_payload_bytes,
            )
            if include_g0_extras:
                return shapes_out, states_out, None
            return shapes_out, states_out

        raise ValidationError(
            "SAM3 tracker requires server-built preload_images on init; re-seed tracking"
        )

    def _infer_preload_init(
        self,
        shapes,
        states,
        diag_meta=None,
        frame_index=None,
        preload_images=None,
        preload_base_frame=None,
        preload_frame_count=None,
        preload_payload_bytes=None,
    ):
        batch_started = time.perf_counter()
        if self._first_seed_shape(shapes) is None:
            raise ValidationError("shapes must contain at least one bounding box")
        if preload_frame_count is None or preload_base_frame is None:
            raise ValidationError("preload_base_frame and preload_frame_count are required")
        if frame_index is None:
            frame_index = int(preload_base_frame)

        base_frame = int(preload_base_frame)
        preloaded_count = int(preload_frame_count)
        relative_frame = int(frame_index) - base_frame
        if relative_frame != 0:
            raise ValidationError("SAM3 init must start at preload_base_frame")

        sess_key = self._allocate_init_session_key(states)
        sess = self._new_session_record()
        sess["base_frame"] = base_frame
        sess["preloaded_count"] = preloaded_count
        sess["preloaded_until_frame"] = base_frame + preloaded_count - 1

        seed_shape = self._first_seed_shape(shapes)
        infer_started = time.perf_counter()
        try:
            cache = self._build_frame_cache(sess, preload_images, preloaded_count, seed_shape)
        except Exception:
            self._cleanup_session_workspace(sess)
            raise
        self._register_session(sess_key, sess)
        sam_inference_ms = (time.perf_counter() - infer_started) * 1000.0

        entry = cache[0]
        bbox = entry["bbox"]
        lost = entry["lost"]
        out_state = self._build_state(
            sess_key,
            bbox,
            sess["prompt_bbox"],
            lost=lost,
            preload_meta=self._preload_state_fields(sess),
        )

        if diag_enabled():
            log_frame_record({
                "sessionId": (diag_meta or {}).get("sessionId"),
                "jobFrameIndex": (diag_meta or {}).get("jobFrameIndex"),
                "requestId": (diag_meta or {}).get("requestId"),
                "requestType": (diag_meta or {}).get("requestType"),
                "samInferenceMs": sam_inference_ms,
                "preloadedFrameCount": preloaded_count,
                "relativeFrame": 0,
                "preloadPayloadBytes": preload_payload_bytes,
                "propagationStartFrame": 0,
                "propagationFrameCount": preloaded_count,
                "committedCandidate": bbox_summary(bbox),
                "lost": lost,
                "totalSamHandlerMs": (time.perf_counter() - batch_started) * 1000.0,
            })

        g0_extras = {
            "preload_timeline": serialize_preload_timeline(cache, base_frame),
            "preload_summary": serialize_preload_summary(sess),
        }
        return [bbox], [out_state], g0_extras

    def _infer_preload_cached_track(
        self,
        states,
        diag_meta=None,
        frame_index=None,
        preload_payload_bytes=None,
    ):
        batch_started = time.perf_counter()
        prev_state = states[0] if states and isinstance(states[0], dict) else {}
        if frame_index is None:
            raise ValidationError("frame_index is required for SAM3 track requests")

        sess_key = self._get_key(states)
        sess = self._sessions[sess_key]
        base_frame = int(prev_state.get("base_frame", sess.get("base_frame", 0)))
        preloaded_count = int(prev_state.get("preloaded_count", sess.get("preloaded_count", 0)))
        relative_frame = int(frame_index) - base_frame

        if relative_frame < 0 or relative_frame >= preloaded_count:
            preloaded_until_frame = int(
                prev_state.get("preloaded_until_frame")
                or sess.get("preloaded_until_frame")
                or base_frame + preloaded_count - 1
            )
            raise PreloadRangeExhaustedError(
                "Preloaded frame range exhausted at frame "
                f"{frame_index} (chunk base={base_frame}, count={preloaded_count}); "
                "re-seed tracking from a new annotation frame",
                base_frame=base_frame,
                preloaded_count=preloaded_count,
                preloaded_until_frame=preloaded_until_frame,
                requested_frame=int(frame_index),
            )
        if not sess.get("cache_ready"):
            raise ValidationError("SAM3 preload cache is not ready; re-seed tracking")

        if prev_state.get("lost"):
            out_state = self._build_state(
                sess_key,
                None,
                sess.get("prompt_bbox") or prev_state.get("prompt_bbox"),
                prev_state=prev_state,
                lost=True,
            )
            if diag_enabled():
                log_frame_record({
                    "sessionId": (diag_meta or {}).get("sessionId"),
                    "jobFrameIndex": (diag_meta or {}).get("jobFrameIndex"),
                    "requestId": (diag_meta or {}).get("requestId"),
                    "requestType": (diag_meta or {}).get("requestType"),
                    "relativeFrame": relative_frame,
                    "preloadPayloadBytes": preload_payload_bytes,
                    "lost": True,
                    "totalSamHandlerMs": (time.perf_counter() - batch_started) * 1000.0,
                })
            return [None], [out_state]

        entry = sess["frame_cache"].get(relative_frame)
        if entry is None:
            raise ValidationError(f"SAM3 cache missing relative frame {relative_frame}")

        bbox = entry["bbox"]
        lost = entry["lost"]
        prompt_bbox = sess.get("prompt_bbox") or prev_state.get("prompt_bbox")
        out_state = self._build_state(
            sess_key,
            bbox,
            prompt_bbox,
            prev_state=prev_state,
            lost=lost,
            preload_meta=self._preload_state_fields(sess),
        )

        if diag_enabled():
            log_frame_record({
                "sessionId": (diag_meta or {}).get("sessionId"),
                "jobFrameIndex": (diag_meta or {}).get("jobFrameIndex"),
                "requestId": (diag_meta or {}).get("requestId"),
                "requestType": (diag_meta or {}).get("requestType"),
                "relativeFrame": relative_frame,
                "preloadPayloadBytes": preload_payload_bytes,
                "committedCandidate": bbox_summary(bbox),
                "lost": lost,
                "totalSamHandlerMs": (time.perf_counter() - batch_started) * 1000.0,
            })

        return [bbox if not lost else None], [out_state]

    def _start_session(self, sess):
        response = self.predictor.handle_request(request={
            "type": "start_session",
            "resource_path": sess["temp_dir"],
        })
        return response["session_id"]

    def _scaled_prompt_bbox(self, bbox, sess):
        """Prompt box scaled about its centre (rod exclusion for small balls)."""
        scale = float(self.config.prompt_box_scale)
        if scale == 1.0 or scale <= 0.0:
            return bbox
        x1, y1, x2, y2 = [float(value) for value in bbox]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_w = max((x2 - x1) * scale / 2.0, MIN_BOX_SIDE_PX / 2.0)
        half_h = max((y2 - y1) * scale / 2.0, MIN_BOX_SIDE_PX / 2.0)
        return _clamp_bbox(
            [
                center_x - half_w,
                center_y - half_h,
                center_x + half_w,
                center_y + half_h,
            ],
            sess["image_height"],
            sess["image_width"],
        )

    def _add_prompt(self, sess, frame_idx, bbox=None):
        prompt_bbox = sess["prompt_bbox"] if bbox is None else bbox
        prompt_bbox = self._scaled_prompt_bbox(prompt_bbox, sess)
        request = build_add_prompt_request(
            sess["session_id"],
            frame_idx,
            prompt_bbox,
            sess["image_height"],
            sess["image_width"],
            self.config,
        )
        try:
            self.predictor.handle_request(request=request)
        except Exception:
            if self.config.prompt_mode == "text_box":
                fallback = build_add_prompt_request(
                    sess["session_id"],
                    frame_idx,
                    prompt_bbox,
                    sess["image_height"],
                    sess["image_width"],
                    self.config,
                    force_box_only=True,
                )
                self.predictor.handle_request(request=fallback)
            else:
                raise
