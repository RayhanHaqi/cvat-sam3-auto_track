import base64
import io
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from diagnostics import bbox_summary, enabled as diag_enabled, log_frame_record, summarize_outputs
from sam3.model_builder import build_sam3_predictor

MAX_SESSIONS = 32
VALID_PROMPT_MODES = ("box", "text_box", "text")
VALID_OUTPUT_POLICIES = ("sam_box", "mask_bbox", "mask_center_box")
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
    return Sam3Config(
        prompt_mode=mode,
        text_prompt=text_prompt,
        output_policy=policy,
        output_prob_thresh=prob_thresh,
        ir_refine_enabled=ir_refine_enabled,
        ir_stop_on_missing=ir_stop_on_missing,
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

    def _make_session(self):
        if len(self._sessions) >= MAX_SESSIONS:
            oldest = next(iter(self._sessions))
            self._destroy_session(oldest)

        key = uuid.uuid4().hex[:12]
        self._sessions[key] = {
            "temp_dir": tempfile.mkdtemp(prefix="sam3_"),
            "frame_count": 0,
            "loaded_frame_count": 0,
            "session_id": None,
            "prompt_bbox": None,
            "image_height": None,
            "image_width": None,
        }
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
        shutil.rmtree(sess["temp_dir"], ignore_errors=True)

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
                "last_propagated_rel_frame",
            ):
                if key in prev_state:
                    state[key] = prev_state[key]
        return state

    def _preload_state_fields(self, sess, relative_frame):
        return {
            "base_frame": sess["base_frame"],
            "preloaded_count": sess["preloaded_count"],
            "preloaded_until_frame": sess["preloaded_until_frame"],
            "last_propagated_rel_frame": relative_frame,
        }

    def _uses_preload_path(self, preload_images, prev_state):
        if preload_images is not None:
            return True
        if isinstance(prev_state, dict) and prev_state.get("preloaded_count") is not None:
            return True
        return False

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
        sess["loaded_frame_count"] = 0
        sess["preloaded_session_ready"] = False

    def _load_preloaded_frame(self, sess, relative_frame):
        frame_path = os.path.join(sess["temp_dir"], f"{int(relative_frame):05d}.jpg")
        if not os.path.exists(frame_path):
            raise ValidationError(
                f"preloaded frame {relative_frame} is missing from session storage"
            )
        return np.array(Image.open(frame_path).convert("RGB"))

    def _start_preloaded_session(self, sess):
        self._close_predictor_session(sess)
        sess["session_id"] = self._start_session(sess)
        sess["loaded_frame_count"] = sess["frame_count"]
        sess["preloaded_session_ready"] = True
        self._add_prompt(sess, 0)
        return True

    def _first_seed_shape(self, shapes):
        if not shapes:
            return None
        for shape in shapes:
            if shape is not None:
                return shape
        return None

    def _can_continue_without_shapes(self, states):
        if not states or not isinstance(states[0], dict):
            return False
        key = states[0].get("session_key")
        if not key or key not in self._sessions:
            return False
        prev_state = states[0]
        if prev_state.get("preloaded_count") is not None:
            return True
        return self._sessions[key]["frame_count"] >= 1

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
        if not self._uses_preload_path(preload_images, prev_state):
            raise ValidationError(
                "SAM3 tracker requires server-built preload_images on init; re-seed tracking"
            )
        return self._infer_batch_preloaded(
            image,
            shapes,
            states,
            diag_meta=diag_meta,
            frame_index=frame_index,
            preload_images=preload_images,
            preload_base_frame=preload_base_frame,
            preload_frame_count=preload_frame_count,
            preload_payload_bytes=preload_payload_bytes,
        )

    def _infer_batch_preloaded(
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
        batch_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        prev_state = states[0] if states and isinstance(states[0], dict) else {}
        is_init = preload_images is not None

        if is_init:
            if self._first_seed_shape(shapes) is None:
                raise ValidationError("shapes must contain at least one bounding box")
            if preload_frame_count is None or preload_base_frame is None:
                raise ValidationError("preload_base_frame and preload_frame_count are required")
            if frame_index is None:
                frame_index = int(preload_base_frame)
            sess_key = self._get_key(states)
            sess = self._sessions[sess_key]
            self._save_preload_sequence(sess, preload_images, preload_frame_count)
            base_frame = int(preload_base_frame)
            preloaded_count = int(preload_frame_count)
            sess["base_frame"] = base_frame
            sess["preloaded_count"] = preloaded_count
            sess["preloaded_until_frame"] = base_frame + preloaded_count - 1
            relative_frame = int(frame_index) - base_frame
            if relative_frame != 0:
                raise ValidationError("SAM3 init must start at preload_base_frame")
            seed_shape = self._first_seed_shape(shapes)
            sess["prompt_bbox"] = list(seed_shape)
            sam_session_reloaded = self._start_preloaded_session(sess)
            propagation_start_frame = 0
            propagation_frame_count = 1
        else:
            if self._first_seed_shape(shapes) is None and not self._can_continue_without_shapes(states):
                raise ValidationError("shapes must contain at least one bounding box")
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
            if not sess.get("preloaded_session_ready"):
                raise ValidationError("SAM3 preload session is not ready; re-seed tracking")
            sam_session_reloaded = False
            propagation_start_frame = relative_frame
            propagation_frame_count = 1

        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        refine_image = self._load_preloaded_frame(sess, relative_frame)

        def _emit_frame_record(**extra):
            if not diag_enabled():
                return
            log_frame_record({
                "sessionId": (diag_meta or {}).get("sessionId"),
                "jobFrameIndex": (diag_meta or {}).get("jobFrameIndex"),
                "requestId": (diag_meta or {}).get("requestId"),
                "requestType": (diag_meta or {}).get("requestType"),
                "samPreprocessMs": preprocess_ms,
                "samInferenceMs": extra.get("samInferenceMs"),
                "samPostprocessMs": extra.get("samPostprocessMs"),
                "geometryConversionMs": extra.get("geometryConversionMs"),
                "samSessionCount": len(self._sessions),
                "samMemoryEntryCount": len(self._sessions),
                "samTrackedObjectCount": sess.get("preloaded_count"),
                "preloadedFrameCount": sess.get("preloaded_count"),
                "relativeFrame": relative_frame,
                "preloadPayloadBytes": preload_payload_bytes,
                "samSessionReloaded": extra.get(
                    "samSessionReloaded",
                    sam_session_reloaded,
                ),
                "propagationStartFrame": extra.get(
                    "propagationStartFrame",
                    propagation_start_frame,
                ),
                "propagationFrameCount": extra.get(
                    "propagationFrameCount",
                    propagation_frame_count,
                ),
                "rawSamOutput": extra.get("rawSamOutput"),
                "selectedCandidate": extra.get("selectedCandidate"),
                "postRefineCandidate": extra.get("postRefineCandidate"),
                "committedCandidate": extra.get("committedCandidate"),
                "lost": extra.get("lost", False),
                "totalSamHandlerMs": (time.perf_counter() - batch_started) * 1000.0,
            })

        if is_init:
            infer_started = time.perf_counter()
            result = self._propagate_frame(sess, 0, 0)
            sam_inference_ms = (time.perf_counter() - infer_started) * 1000.0
            reference_bbox = sess["prompt_bbox"]
            geom_started = time.perf_counter()
            raw_summary = summarize_outputs(result, sess["image_height"], sess["image_width"])
            bbox = self._output_to_bbox(result, sess, reference_bbox)
            selected_summary = bbox_summary(bbox)
            geometry_ms = (time.perf_counter() - geom_started) * 1000.0
            if bbox is None:
                bbox = reference_bbox
            refine_started = time.perf_counter()
            bbox = self._maybe_refine_bbox(refine_image, bbox)
            if bbox is None:
                bbox = reference_bbox
            post_refine_summary = bbox_summary(bbox)
            postprocess_ms = (time.perf_counter() - refine_started) * 1000.0
            out_state = self._build_state(
                sess_key,
                bbox,
                sess["prompt_bbox"],
                preload_meta=self._preload_state_fields(sess, relative_frame),
            )
            _emit_frame_record(
                samInferenceMs=sam_inference_ms,
                samPostprocessMs=postprocess_ms,
                geometryConversionMs=geometry_ms,
                rawSamOutput=raw_summary,
                selectedCandidate=selected_summary,
                postRefineCandidate=post_refine_summary,
                committedCandidate=bbox_summary(bbox),
                samSessionReloaded=sam_session_reloaded,
                propagationStartFrame=propagation_start_frame,
                propagationFrameCount=propagation_frame_count,
            )
            return [bbox], [out_state]

        if prev_state.get("lost"):
            out_state = self._build_state(
                sess_key,
                None,
                sess.get("prompt_bbox"),
                prev_state=prev_state,
                lost=True,
            )
            _emit_frame_record(
                lost=True,
                committedCandidate=None,
                samSessionReloaded=False,
                propagationStartFrame=propagation_start_frame,
                propagationFrameCount=propagation_frame_count,
            )
            return [None], [out_state]

        prompt_bbox = sess.get("prompt_bbox") or prev_state.get("prompt_bbox")
        seed_shape = self._first_seed_shape(shapes)
        if prompt_bbox is None and seed_shape is not None:
            sess["prompt_bbox"] = list(seed_shape)
            prompt_bbox = sess["prompt_bbox"]

        last_known_bbox = prev_state.get("last_bbox", prompt_bbox)
        infer_started = time.perf_counter()
        result = self._propagate_frame(sess, relative_frame, relative_frame)
        sam_inference_ms = (time.perf_counter() - infer_started) * 1000.0
        geom_started = time.perf_counter()
        raw_summary = summarize_outputs(result, sess["image_height"], sess["image_width"])
        bbox = self._output_to_bbox(result, sess, last_known_bbox)
        selected_summary = bbox_summary(bbox)
        geometry_ms = (time.perf_counter() - geom_started) * 1000.0
        if bbox is None:
            out_state = self._build_state(
                sess_key,
                None,
                prompt_bbox,
                prev_state=prev_state,
                lost=True,
                preload_meta=self._preload_state_fields(sess, relative_frame),
            )
            _emit_frame_record(
                samInferenceMs=sam_inference_ms,
                geometryConversionMs=geometry_ms,
                rawSamOutput=raw_summary,
                selectedCandidate=None,
                lost=True,
                samSessionReloaded=False,
                propagationStartFrame=propagation_start_frame,
                propagationFrameCount=propagation_frame_count,
            )
            return [None], [out_state]

        refine_started = time.perf_counter()
        bbox = self._maybe_refine_bbox(refine_image, bbox, previous_bbox=last_known_bbox)
        post_refine_summary = bbox_summary(bbox)
        postprocess_ms = (time.perf_counter() - refine_started) * 1000.0
        if bbox is None:
            out_state = self._build_state(
                sess_key,
                None,
                prompt_bbox,
                prev_state=prev_state,
                lost=True,
                preload_meta=self._preload_state_fields(sess, relative_frame),
            )
            _emit_frame_record(
                samInferenceMs=sam_inference_ms,
                samPostprocessMs=postprocess_ms,
                geometryConversionMs=geometry_ms,
                rawSamOutput=raw_summary,
                selectedCandidate=selected_summary,
                postRefineCandidate=None,
                lost=True,
                samSessionReloaded=False,
                propagationStartFrame=propagation_start_frame,
                propagationFrameCount=propagation_frame_count,
            )
            return [None], [out_state]

        if (
            prompt_bbox is not None
            and last_known_bbox is not None
            and _bboxes_near(bbox, prompt_bbox)
            and not _bboxes_near(last_known_bbox, prompt_bbox)
        ):
            out_state = self._build_state(
                sess_key,
                None,
                prompt_bbox,
                prev_state=prev_state,
                lost=True,
                preload_meta=self._preload_state_fields(sess, relative_frame),
            )
            _emit_frame_record(
                samInferenceMs=sam_inference_ms,
                samPostprocessMs=postprocess_ms,
                geometryConversionMs=geometry_ms,
                rawSamOutput=raw_summary,
                selectedCandidate=selected_summary,
                postRefineCandidate=post_refine_summary,
                lost=True,
                samSessionReloaded=False,
                propagationStartFrame=propagation_start_frame,
                propagationFrameCount=propagation_frame_count,
            )
            return [None], [out_state]

        new_state = self._build_state(
            sess_key,
            bbox,
            prompt_bbox,
            prev_state=prev_state,
            preload_meta=self._preload_state_fields(sess, relative_frame),
        )
        _emit_frame_record(
            samInferenceMs=sam_inference_ms,
            samPostprocessMs=postprocess_ms,
            geometryConversionMs=geometry_ms,
            rawSamOutput=raw_summary,
            selectedCandidate=selected_summary,
            postRefineCandidate=post_refine_summary,
            committedCandidate=bbox_summary(bbox),
            samSessionReloaded=False,
            propagationStartFrame=propagation_start_frame,
            propagationFrameCount=propagation_frame_count,
        )
        return [bbox], [new_state]

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

    def _propagate_frame(self, sess, start_frame, end_frame):
        target_output = None
        frame_count = end_frame - start_frame + 1
        for response in self.predictor.handle_stream_request(request={
            "type": "propagate_in_video",
            "session_id": sess["session_id"],
            "start_frame_index": start_frame,
            "max_frame_num_to_track": frame_count,
            "propagation_direction": "forward",
        }):
            if response.get("frame_index") == end_frame:
                target_output = response["outputs"]
        return target_output
