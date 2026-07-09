#!/usr/bin/env python3
"""Offline scaffolding to compare SAM3 bbox output policies against manual labels.

This utility does not run SAM3 inference. It compares exported labeled CSVs or
JSON frame records produced by two policy runs (or one run vs ground truth).

Expected later benchmark target:
  - 10-12 short segments
  - 40-50 frames each
  - ~400-600 labeled frames total

Input format (CSV columns):
  frame_index,gt_xtl,gt_ytl,gt_xbr,gt_ybr,pred_xtl,pred_ytl,pred_xbr,pred_ybr[,lost]

Usage:
  python eval_adaptive_bbox_policies.py baseline.csv adaptive.csv
  python eval_adaptive_bbox_policies.py --gt ground_truth.csv --pred adaptive.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_side(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
        raise ValueError(f"JSON must be a list of records: {path}")
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _row_bbox(row, prefix):
    return [
        float(row[f"{prefix}_xtl"]),
        float(row[f"{prefix}_ytl"]),
        float(row[f"{prefix}_xbr"]),
        float(row[f"{prefix}_ybr"]),
    ]


def evaluate(gt_rows: Iterable[dict], pred_rows: Iterable[dict]) -> dict:
    gt_list = list(gt_rows)
    pred_list = list(pred_rows)
    if len(gt_list) != len(pred_list):
        raise ValueError(f"row count mismatch: gt={len(gt_list)} pred={len(pred_list)}")

    center_errors = []
    ious = []
    side_rel_errors = []
    side_jitter = []
    wrong_object = 0
    false_loss = 0
    prev_side = None

    for gt, pred in zip(gt_list, pred_list):
        lost = str(pred.get("lost", "0")).lower() in ("1", "true", "yes")
        if lost:
            false_loss += 1
            continue
        gt_bbox = _row_bbox(gt, "gt") if "gt_xtl" in gt else _row_bbox(gt, "pred")
        pred_bbox = _row_bbox(pred, "pred")
        gt_cx, gt_cy = _bbox_center(gt_bbox)
        pred_cx, pred_cy = _bbox_center(pred_bbox)
        center_errors.append(math.hypot(pred_cx - gt_cx, pred_cy - gt_cy))
        ious.append(_iou(gt_bbox, pred_bbox))
        gt_side = _bbox_side(gt_bbox)
        pred_side = _bbox_side(pred_bbox)
        if gt_side > 0:
            side_rel_errors.append(abs(pred_side - gt_side) / gt_side)
        if prev_side is not None:
            side_jitter.append(abs(pred_side - prev_side) / max(prev_side, 1e-6))
        prev_side = pred_side
        if ious[-1] < 0.1:
            wrong_object += 1

    def _mean(values):
        return sum(values) / len(values) if values else float("nan")

    return {
        "frames": len(gt_list),
        "mean_center_error_px": _mean(center_errors),
        "mean_bbox_iou": _mean(ious),
        "mean_side_relative_error": _mean(side_rel_errors),
        "mean_temporal_scale_jitter": _mean(side_jitter),
        "wrong_object_rate": wrong_object / len(gt_list) if gt_list else 0.0,
        "false_loss_rate": false_loss / len(gt_list) if gt_list else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path, help="Baseline policy CSV")
    parser.add_argument("adaptive", nargs="?", type=Path, help="Adaptive policy CSV")
    parser.add_argument("--gt", type=Path, help="Ground-truth CSV")
    parser.add_argument("--pred", type=Path, help="Prediction CSV to compare against --gt")
    args = parser.parse_args(argv)

    if args.gt and args.pred:
        metrics = evaluate(_read_rows(args.gt), _read_rows(args.pred))
        print(json.dumps({"comparison": "gt_vs_pred", **metrics}, indent=2))
        return 0

    if not args.baseline or not args.adaptive:
        parser.error("provide baseline+adaptive CSVs or --gt and --pred")

    baseline_rows = _read_rows(args.baseline)
    adaptive_rows = _read_rows(args.adaptive)
    baseline_metrics = evaluate(baseline_rows, baseline_rows)
    adaptive_metrics = evaluate(baseline_rows, adaptive_rows)
    print(json.dumps({
        "mask_center_box_self_check": baseline_metrics,
        "mask_center_box_vs_adaptive_component_square_box": adaptive_metrics,
        "procedure": {
            "step_1": "Run preload cache export twice with SAM3_OUTPUT_POLICY=mask_center_box and adaptive_component_square_box.",
            "step_2": "Label 10-12 segments (~40-50 frames each) with manual ball boxes.",
            "step_3": "Merge exports into CSV with gt_* and pred_* columns per frame.",
            "step_4": "Run this script with --gt and --pred to compute metrics.",
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
