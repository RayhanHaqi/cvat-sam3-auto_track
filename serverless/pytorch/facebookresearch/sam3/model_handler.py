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
from sam3.model_builder import build_sam3_predictor

MAX_SESSIONS = 32
VALID_PROMPT_MODES = ("box", "text_box", "text")
VALID_OUTPUT_POLICIES = (
    "sam_box",
    "mask_bbox",
    "mask_center_box",
    "adaptive_component_square_box",
)
# Experimental calibration defaults for adaptive_component_square_box (v1).
ADAPTIVE_PADDING_FACTOR = 1.25
ADAPTIVE_ROI_SCALE = 2.5
ADAPTIVE_MIN_ROI_MARGIN_PX = 24
ADAPTIVE_MIN_SIDE_PX = 4
ADAPTIVE_MAX_GROWTH_RATIO = 1.30
ADAPTIVE_MAX_SHRINK_RATIO = 0.75
DEFAULT_TEXT_PROMPT = "ping pong ball"
DEFAULT_OUTPUT_PROB_THRESH = 0.25
MIN_BOX_SIDE_PX = 4
IR_REFINE_BRIGHT_PERCENTILE = 88
IR_REFINE_ROI_SCALE = 1.5
IR_REFINE_MIN_ROI_MARGIN_PX = 24
IR_REFINE_MAX_DISP_RATIO = 0.6
IR_REFINE_MIN_AREA_RATIO = 0.02
IR_REFINE_MAX_AREA_RATIO = 3.0
IR_REFINE_MIN_CIRCULARITY = 0.25
IR_REFINE_MAX_ASPECT_RATIO = 2.5
IR_REFINE_MAX_TEMPORAL_DISP_RATIO = 2.5
IR_REFINE_SUPPORT_CONTRAST_FLOOR = 8.0
IR_REFINE_MIN_SEED_PIXELS = 3


@dataclass(frozen=True)
class Sam3Config:
    prompt_mode: str = "box"
    text_prompt: str = DEFAULT_TEXT_PROMPT
    output_policy: str = "sam_box"
    output_prob_thresh: float = DEFAULT_OUTPUT_PROB_THRESH
    ir_refine_enabled: bool = False
    ir_stop_on_missing: bool = False
    adaptive_padding_factor: float = ADAPTIVE_PADDING_FACTOR
    adaptive_roi_scale: float = ADAPTIVE_ROI_SCALE
    adaptive_min_roi_margin_px: int = ADAPTIVE_MIN_ROI_MARGIN_PX
    adaptive_min_side_px: int = ADAPTIVE_MIN_SIDE_PX
    adaptive_max_growth_ratio: float = ADAPTIVE_MAX_GROWTH_RATIO
    adaptive_max_shrink_ratio: float = ADAPTIVE_MAX_SHRINK_RATIO


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
    policy = env.get("SAM3_OUTPUT_POLICY", "sam_box").strip().lower()
    if policy not in VALID_OUTPUT_POLICIES:
        policy = "sam_box"
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

    return Sam3Config(
        prompt_mode=mode,
        text_prompt=text_prompt,
        output_policy=policy,
        output_prob_thresh=prob_thresh,
        ir_refine_enabled=ir_refine_enabled,
        ir_stop_on_missing=ir_stop_on_missing,
        adaptive_padding_factor=_float_env("SAM3_ADAPTIVE_PADDING_FACTOR", ADAPTIVE_PADDING_FACTOR),
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
    best = min(
        candidates,
        key=lambda bbox: (bbox_center(bbox)[0] - ref_cx) ** 2
        + (bbox_center(bbox)[1] - ref_cy) ** 2,
    )
    return best


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

    displacement = ((comp_cx - center_x) ** 2 + (comp_cy - center_y) ** 2) ** 0.5
    score = displacement - circularity * 10.0 - overlap_ratio * 20.0
    comp_w = comp_bbox[2] - comp_bbox[0]
    comp_h = comp_bbox[3] - comp_bbox[1]
    return score, area, comp_w, comp_h


def select_adaptive_scale_component(
    image,
    center_x,
    center_y,
    reference_side,
    previous_bbox,
    config,
):
    """Select a bright support component for scale estimation only (center fixed)."""
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
        return None

    threshold = float(np.percentile(roi, IR_REFINE_BRIGHT_PERCENTILE))
    background = float(np.median(roi))
    bright_threshold = max(threshold, background + 15.0)
    if bright_threshold >= float(roi.max()):
        return None

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
        return None

    support_binary = (roi >= support_threshold).astype(np.uint8)
    num_support, support_labels, _support_stats, _support_centroids = cv2.connectedComponentsWithStats(
        support_binary,
        connectivity=8,
    )
    if num_support <= 1:
        return None

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
            scored = _score_adaptive_scale_component(
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
            if scored is None:
                continue
            seen_support_labels.add(support_label_id)
            candidates.append(scored)

    if not candidates:
        return None

    _score, area, comp_w, comp_h = min(candidates, key=lambda item: item[0])
    return {
        "component_area": area,
        "component_bbox_width": comp_w,
        "component_bbox_height": comp_h,
        "equivalent_diameter": equivalent_diameter_from_area(area),
    }


def gate_adaptive_side(padded_side, reference_side, config):
    reference_side = max(float(reference_side), config.adaptive_min_side_px)
    padded_side = max(float(padded_side), config.adaptive_min_side_px)
    lo = reference_side * config.adaptive_max_shrink_ratio
    hi = reference_side * config.adaptive_max_growth_ratio
    return max(lo, min(hi, padded_side))


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


class SessionStaleError(Exception):
    """Raised when CVAT returns a session_key the worker no longer holds."""


class ValidationError(Exception):
    """Raised when the request payload is missing required fields."""


class PreloadRangeExhaustedError(Exception):
    """Raised when tracking requests a frame outside the bounded preload chunk."""


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


class ModelHandler:
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config if config is not None else load_sam3_config()
        self.predictor = build_sam3_predictor(version="sam3", compile=False)
        self._sessions = {}

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
        return refine_bbox_with_ir_intensity(
            image,
            bbox,
            previous_bbox=previous_bbox,
            stop_on_missing=self.config.ir_stop_on_missing,
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
        for response in self.predictor.handle_stream_request(request={
            "type": "propagate_in_video",
            "session_id": sess["session_id"],
            "start_frame_index": 0,
            "max_frame_num_to_track": max_track,
            "propagation_direction": "forward",
        }):
            frame_idx = response.get("frame_index")
            if frame_idx is not None:
                outputs_by_frame[int(frame_idx)] = response.get("outputs")
        return outputs_by_frame

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
    ):
        if prev_lost:
            return None, True, None, None

        reference_bbox = prompt_bbox if is_init_frame else last_known_bbox
        if self.config.output_policy == "adaptive_component_square_box":
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

        if bbox is None:
            if is_init_frame:
                bbox = prompt_bbox
            else:
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
                return None, True, None, None

        if (
            not is_init_frame
            and prompt_bbox is not None
            and last_known_bbox is not None
            and _bboxes_near(bbox, prompt_bbox)
            and not _bboxes_near(last_known_bbox, prompt_bbox)
        ):
            return None, True, None, None

        canonical_tracking_bbox = list(bbox)
        emitted_bbox = canonical_tracking_bbox
        adaptive_diag = None
        if self.config.output_policy == "adaptive_component_square_box":
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

        return emitted_bbox, False, adaptive_diag, canonical_tracking_bbox

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

            prev_bbox = None
            prev_lost = False
            prev_accepted_target_side = None
            prompt_bbox = sess["prompt_bbox"]

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
                    )
                )
                cache[relative_frame] = {"bbox": emitted_bbox, "lost": lost}
                if lost:
                    prev_lost = True
                    prev_bbox = None
                    prev_accepted_target_side = None
                else:
                    prev_lost = False
                    prev_bbox = canonical_tracking_bbox
                    if (
                        self.config.output_policy == "adaptive_component_square_box"
                        and adaptive_diag is not None
                    ):
                        prev_accepted_target_side = adaptive_diag.get("accepted_target_side")

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
    ):
        shapes = shapes or []
        prev_state = states[0] if states and isinstance(states[0], dict) else {}

        if preload_images is not None:
            return self._infer_preload_init(
                shapes,
                states,
                diag_meta=diag_meta,
                frame_index=frame_index,
                preload_images=preload_images,
                preload_base_frame=preload_base_frame,
                preload_frame_count=preload_frame_count,
                preload_payload_bytes=preload_payload_bytes,
            )

        if isinstance(prev_state, dict) and prev_state.get("preloaded_count") is not None:
            return self._infer_preload_cached_track(
                states,
                diag_meta=diag_meta,
                frame_index=frame_index,
                preload_payload_bytes=preload_payload_bytes,
            )

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

        return [bbox], [out_state]

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
            raise PreloadRangeExhaustedError(
                "Preloaded frame range exhausted at frame "
                f"{frame_index} (chunk base={base_frame}, count={preloaded_count}); "
                "re-seed tracking from a new annotation frame"
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

    def _add_prompt(self, sess, frame_idx):
        request = build_add_prompt_request(
            sess["session_id"],
            frame_idx,
            sess["prompt_bbox"],
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
                    sess["prompt_bbox"],
                    sess["image_height"],
                    sess["image_width"],
                    self.config,
                    force_box_only=True,
                )
                self.predictor.handle_request(request=fallback)
            else:
                raise
