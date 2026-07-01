"""Read-only Auto Track diagnostics for SAM3 Nuclio handler."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator


def enabled() -> bool:
    value = os.environ.get("AUTO_TRACK_DIAGNOSTICS", "").strip().lower()
    return value in ("1", "true", "yes")


def diag_correlation_fields(diag_meta: dict[str, Any] | None) -> dict[str, Any]:
    if not diag_meta:
        return {
            "sessionId": None,
            "jobFrameIndex": None,
            "requestId": None,
            "requestType": None,
        }
    return {
        "sessionId": diag_meta.get("sessionId"),
        "jobFrameIndex": diag_meta.get("jobFrameIndex"),
        "requestId": diag_meta.get("requestId"),
        "requestType": diag_meta.get("requestType"),
    }


def log_event(event_name: str, **fields: Any) -> None:
    if not enabled():
        return
    diag_meta = fields.pop("diagMeta", None)
    record = {
        "event": event_name,
        "timestamp": time.time(),
        **diag_correlation_fields(diag_meta if isinstance(diag_meta, dict) else None),
        **fields,
    }
    if diag_meta is not None:
        record["diagMeta"] = diag_meta
    print(f"AUTO_TRACK_DIAG {json.dumps(record, default=str)}", flush=True)


@contextmanager
def timed_stage(stage_name: str, **meta: Any) -> Iterator[dict[str, Any]]:
    diag_meta = meta.pop("diagMeta", None)
    correlation = diag_correlation_fields(diag_meta if isinstance(diag_meta, dict) else None)
    timings: dict[str, Any] = {"stage": stage_name, **correlation, **meta}
    start = time.perf_counter()
    log_event(f"{stage_name}_start", diagMeta=diag_meta, **correlation, **meta)
    try:
        yield timings
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings["durationMs"] = elapsed_ms
        log_event(f"{stage_name}_end", diagMeta=diag_meta, durationMs=elapsed_ms, **correlation, **meta)


def summarize_mask(mask) -> dict[str, Any] | None:
    if mask is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(mask)
        if arr.size == 0:
            return None
        area = int(arr.astype(bool).sum())
        h, w = arr.shape[:2]
        ys, xs = np.where(arr.astype(bool))
        if xs.size == 0:
            return {"maskWidth": w, "maskHeight": h, "maskArea": area, "componentCount": 0}
        return {
            "maskWidth": w,
            "maskHeight": h,
            "maskArea": area,
            "componentCount": 1,
            "boundsXtl": float(xs.min()),
            "boundsYtl": float(ys.min()),
            "boundsXbr": float(xs.max()),
            "boundsYbr": float(ys.max()),
        }
    except Exception:
        return None


def summarize_outputs(outputs: dict[str, Any] | None, image_height: int, image_width: int) -> dict[str, Any]:
    if not outputs:
        return {"candidateCount": 0}
    summary: dict[str, Any] = {
        "hasOutBoxesXywh": outputs.get("out_boxes_xywh") is not None,
        "hasOutBinaryMasks": outputs.get("out_binary_masks") is not None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
    boxes = outputs.get("out_boxes_xywh")
    if boxes is not None:
        try:
            summary["boxCandidateCount"] = len(list(boxes))
        except Exception:
            summary["boxCandidateCount"] = None
    masks = outputs.get("out_binary_masks")
    if masks is not None:
        try:
            mask_list = list(masks)
            summary["maskCandidateCount"] = len(mask_list)
            if mask_list:
                summary["firstMask"] = summarize_mask(mask_list[0])
        except Exception:
            summary["maskCandidateCount"] = None
    return summary


def bbox_summary(bbox) -> dict[str, Any] | None:
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return {
            "xtl": x1,
            "ytl": y1,
            "xbr": x2,
            "ybr": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "centerX": (x1 + x2) / 2.0,
            "centerY": (y1 + y2) / 2.0,
        }
    except Exception:
        return None


def log_frame_record(record: dict[str, Any]) -> None:
    if not enabled():
        return
    print(f"AUTO_TRACK_DIAG_FRAME {json.dumps(record, default=str)}", flush=True)
