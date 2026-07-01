#!/usr/bin/env python3
"""Experimental GPU capability test for SAM3 propagate modes.

NOT production code. Run inside the SAM3 Nuclio GPU container against a real video.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw

NUCLIO_ROOT = Path(os.environ.get("SAM3_CAPABILITY_NUCLIO_ROOT", "/opt/nuclio"))
if str(NUCLIO_ROOT) not in sys.path:
    sys.path.insert(0, str(NUCLIO_ROOT))

from model_handler import (  # noqa: E402
    build_add_prompt_request,
    load_sam3_config,
    select_bbox_from_outputs,
)
from sam3.model_builder import build_sam3_predictor  # noqa: E402

COMPARE_FRAMES = (1, 10, 70, 95)
MAX_FRAMES_DEFAULT = 96


@dataclass
class MemorySnapshot:
    host_rss_mb: float
    cuda_allocated_mb: float | None
    cuda_reserved_mb: float | None


@dataclass
class PropagateCallRecord:
    mode: str
    call_index: int
    requested_start_frame_index: int
    requested_max_frame_num_to_track: int
    returned_frame_indices: list[int]
    selected_bbox: list[float] | None
    elapsed_s: float


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 propagate capability test (GPU).")
    parser.add_argument("--video", required=True, help="Path to rally .mp4")
    parser.add_argument("--max-frame", type=int, default=95, help="Last frame index to test")
    parser.add_argument("--frame-count", type=int, default=MAX_FRAMES_DEFAULT, help="Preload frame count")
    parser.add_argument("--init-bbox", default="", help="x1,y1,x2,y2; auto-detect if omitted")
    parser.add_argument("--output", required=True, help="Output directory for JSON and overlays")
    parser.add_argument(
        "--production-max",
        type=int,
        default=1,
        help="max_frame_num_to_track used for mode B production pattern",
    )
    return parser.parse_args()


def memory_snapshot() -> MemorySnapshot:
    proc = psutil.Process()
    cuda_allocated = None
    cuda_reserved = None
    if torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        cuda_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return MemorySnapshot(
        host_rss_mb=proc.memory_info().rss / (1024 * 1024),
        cuda_allocated_mb=cuda_allocated,
        cuda_reserved_mb=cuda_reserved,
    )


def extract_frames(video_path: Path, frame_count: int, out_dir: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    saved = 0
    while saved < frame_count:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(out_dir / f"{saved:05d}.jpg")
        saved += 1
    cap.release()
    if saved < frame_count:
        raise RuntimeError(f"video ended early at frame {saved - 1}, need {frame_count}")
    return width, height


def detect_init_bbox(frame_rgb: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape[:2]
    max_area = max(500, int(h_img * w_img * 0.002))
    best = None
    best_score = -1.0
    for percentile in (99.9, 99.5, 99.0):
        threshold = float(np.percentile(gray, percentile))
        bright = (gray >= threshold).astype(np.uint8)
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < 5 or area > max_area:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 6.0:
                continue
            score = float(area) / aspect
            if score > best_score:
                best_score = score
                pad = 8
                best = [
                    float(max(0, x - pad)),
                    float(max(0, y - pad)),
                    float(min(w_img - 1, x + w + pad)),
                    float(min(h_img - 1, y + h + pad)),
                ]
    if best is None:
        raise RuntimeError("auto init bbox failed: no compact bright contours on frame 0")
    return best


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def outputs_to_bbox(outputs, width: int, height: int, reference_bbox, config):
    if outputs is None:
        return None
    return select_bbox_from_outputs(outputs, height, width, reference_bbox, config)


def start_session(predictor, resource_dir: Path) -> str:
    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": str(resource_dir)}
    )
    return response["session_id"]


def close_session(predictor, session_id: str):
    try:
        predictor.handle_request(
            request={
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": True,
            }
        )
    except Exception:
        pass


def add_prompt(predictor, session_id: str, frame_idx: int, prompt_bbox, width: int, height: int, config):
    request = build_add_prompt_request(
        session_id,
        frame_idx,
        prompt_bbox,
        height,
        width,
        config,
    )
    try:
        predictor.handle_request(request=request)
    except Exception:
        if config.prompt_mode == "text_box":
            fallback = build_add_prompt_request(
                session_id,
                frame_idx,
                prompt_bbox,
                height,
                width,
                config,
                force_box_only=True,
            )
            predictor.handle_request(request=fallback)
        else:
            raise


def propagate_stream(
    predictor,
    session_id: str,
    start_frame_index: int,
    max_frame_num_to_track: int,
) -> tuple[list[dict[str, Any]], float]:
    responses: list[dict[str, Any]] = []
    started = time.perf_counter()
    for response in predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagation_direction": "forward",
        }
    ):
        responses.append(
            {
                "frame_index": response.get("frame_index"),
                "outputs": response.get("outputs"),
            }
        )
    return responses, time.perf_counter() - started


def frame_bbox_map(
    responses,
    width: int,
    height: int,
    reference_bbox,
    config,
) -> dict[int, list[float] | None]:
    out: dict[int, list[float] | None] = {}
    for item in responses:
        frame_idx = item["frame_index"]
        if frame_idx is None:
            continue
        out[int(frame_idx)] = outputs_to_bbox(item["outputs"], width, height, reference_bbox, config)
    return out


def draw_overlay(frame_path: Path, bbox: list[float] | None, label: str, out_path: Path):
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 64, 64), width=2)
    draw.text((8, 8), label, fill=(255, 255, 0))
    image.save(out_path)


def fresh_session(predictor, frames_dir: Path, prompt_bbox, width: int, height: int, config) -> str:
    session_id = start_session(predictor, frames_dir)
    add_prompt(predictor, session_id, 0, prompt_bbox, width, height, config)
    return session_id


def run_mode_a(predictor, frames_dir: Path, prompt_bbox, width: int, height: int, config, max_frame: int):
    session_id = fresh_session(predictor, frames_dir, prompt_bbox, width, height, config)
    responses, elapsed = propagate_stream(
        predictor,
        session_id,
        0,
        max_frame + 1,
    )
    close_session(predictor, session_id)
    bbox_map = frame_bbox_map(responses, width, height, prompt_bbox, config)
    return {
        "elapsed_s": elapsed,
        "returned_frame_indices": sorted(bbox_map.keys()),
        "bboxes": {str(k): v for k, v in bbox_map.items()},
        "compare": {str(f): bbox_map.get(f) for f in COMPARE_FRAMES if f <= max_frame},
    }


def run_mode_b(
    predictor,
    frames_dir: Path,
    prompt_bbox,
    width: int,
    height: int,
    config,
    max_frame: int,
    production_max: int,
):
    session_id = fresh_session(predictor, frames_dir, prompt_bbox, width, height, config)
    records: list[PropagateCallRecord] = []
    bbox_map: dict[int, list[float] | None] = {}
    reference_bbox = prompt_bbox
    memory_points: dict[str, MemorySnapshot] = {
        "start": memory_snapshot(),
    }

    for rel_frame in range(0, max_frame + 1):
        responses, elapsed = propagate_stream(
            predictor,
            session_id,
            rel_frame,
            production_max,
        )
        returned_indices = [int(r["frame_index"]) for r in responses if r.get("frame_index") is not None]
        selected = None
        for item in responses:
            if item.get("frame_index") == rel_frame:
                selected = outputs_to_bbox(
                    item.get("outputs"),
                    width,
                    height,
                    reference_bbox,
                    config,
                )
                break
        if selected is not None:
            reference_bbox = selected
        bbox_map[rel_frame] = selected
        records.append(
            PropagateCallRecord(
                mode="B",
                call_index=rel_frame,
                requested_start_frame_index=rel_frame,
                requested_max_frame_num_to_track=production_max,
                returned_frame_indices=returned_indices,
                selected_bbox=selected,
                elapsed_s=elapsed,
            )
        )
        if rel_frame in (0, 50, max_frame):
            memory_points[f"frame_{rel_frame}"] = memory_snapshot()

    close_session(predictor, session_id)
    return {
        "production_max": production_max,
        "records": [asdict(r) for r in records],
        "compare": {str(f): bbox_map.get(f) for f in COMPARE_FRAMES if f <= max_frame},
        "memory_points": {k: asdict(v) for k, v in memory_points.items()},
        "latency_summary": {
            "count": len(records),
            "min_s": min(r.elapsed_s for r in records),
            "max_s": max(r.elapsed_s for r in records),
            "mean_s": sum(r.elapsed_s for r in records) / len(records),
            "frame_1_s": records[1].elapsed_s if len(records) > 1 else None,
            "frame_10_s": records[10].elapsed_s if len(records) > 10 else None,
            "frame_70_s": records[70].elapsed_s if len(records) > 70 else None,
            "frame_95_s": records[95].elapsed_s if len(records) > 95 else None,
        },
    }


def run_mode_c(predictor, frames_dir: Path, prompt_bbox, width: int, height: int, config, targets):
    results = {}
    for target in targets:
        session_id = fresh_session(predictor, frames_dir, prompt_bbox, width, height, config)
        responses, elapsed = propagate_stream(predictor, session_id, target, 1)
        close_session(predictor, session_id)
        bbox_map = frame_bbox_map(responses, width, height, prompt_bbox, config)
        results[str(target)] = {
            "elapsed_s": elapsed,
            "returned_frame_indices": sorted(bbox_map.keys()),
            "selected_bbox": bbox_map.get(target),
            "all_bboxes": {str(k): v for k, v in bbox_map.items()},
        }
    return results


def run_max_frame_probe(predictor, frames_dir: Path, prompt_bbox, width: int, height: int, config, frame_idx: int):
    probe = {}
    for max_value in (0, 1):
        session_id = fresh_session(predictor, frames_dir, prompt_bbox, width, height, config)
        responses, elapsed = propagate_stream(predictor, session_id, frame_idx, max_value)
        close_session(predictor, session_id)
        probe[str(max_value)] = {
            "elapsed_s": elapsed,
            "returned_frame_indices": [int(r["frame_index"]) for r in responses if r.get("frame_index") is not None],
            "selected_bbox": frame_bbox_map(responses, width, height, prompt_bbox, config).get(frame_idx),
        }
    return probe


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_sam3_config()
    video_path = Path(args.video)
    frame_count = min(args.frame_count, args.max_frame + 1)

    frames_dir = Path(tempfile.mkdtemp(prefix="sam3_capability_"))
    try:
        width, height = extract_frames(video_path, frame_count, frames_dir)
        frame0 = np.array(Image.open(frames_dir / "00000.jpg").convert("RGB"))
        if args.init_bbox:
            prompt_bbox = [float(v) for v in args.init_bbox.split(",")]
        else:
            prompt_bbox = detect_init_bbox(frame0)

        predictor = build_sam3_predictor(version="sam3", compile=False)

        results: dict[str, Any] = {
            "video": str(video_path),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "prompt_bbox": prompt_bbox,
            "config": asdict(config),
            "cuda_available": torch.cuda.is_available(),
            "memory_start": asdict(memory_snapshot()),
        }

        results["mode_a"] = run_mode_a(
            predictor,
            frames_dir,
            prompt_bbox,
            width,
            height,
            config,
            args.max_frame,
        )
        results["mode_b"] = run_mode_b(
            predictor,
            frames_dir,
            prompt_bbox,
            width,
            height,
            config,
            args.max_frame,
            args.production_max,
        )
        results["mode_c"] = run_mode_c(
            predictor,
            frames_dir,
            prompt_bbox,
            width,
            height,
            config,
            [f for f in COMPARE_FRAMES if f <= args.max_frame],
        )
        results["max_frame_probe_frame_10"] = run_max_frame_probe(
            predictor,
            frames_dir,
            prompt_bbox,
            width,
            height,
            config,
            10,
        )

        compare_report = {}
        mode_a = results["mode_a"]["compare"]
        mode_b = results["mode_b"]["compare"]
        mode_c = results["mode_c"]
        for frame in COMPARE_FRAMES:
            if frame > args.max_frame:
                continue
            key = str(frame)
            a_bbox = mode_a.get(key)
            b_bbox = mode_b.get(key)
            c_bbox = mode_c.get(key, {}).get("selected_bbox")
            compare_report[key] = {
                "mode_a_bbox": a_bbox,
                "mode_b_bbox": b_bbox,
                "mode_c_bbox": c_bbox,
                "iou_a_vs_b": bbox_iou(a_bbox, b_bbox),
                "iou_a_vs_c": bbox_iou(a_bbox, c_bbox),
                "lost_b": b_bbox is None,
                "lost_c": c_bbox is None,
            }
            overlay_a = output_dir / f"overlay_A_frame_{frame:05d}.jpg"
            overlay_b = output_dir / f"overlay_B_frame_{frame:05d}.jpg"
            overlay_c = output_dir / f"overlay_C_frame_{frame:05d}.jpg"
            draw_overlay(frames_dir / f"{frame:05d}.jpg", a_bbox, f"A frame {frame}", overlay_a)
            draw_overlay(frames_dir / f"{frame:05d}.jpg", b_bbox, f"B frame {frame}", overlay_b)
            draw_overlay(frames_dir / f"{frame:05d}.jpg", c_bbox, f"C frame {frame}", overlay_c)

        results["compare_report"] = compare_report
        results["memory_end"] = asdict(memory_snapshot())

        out_json = output_dir / "capability_results.json"
        out_json.write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(results, indent=2, default=str))
        print(f"Wrote {out_json}")
        return 0
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
