#!/usr/bin/env python3
"""Drag a box on a video frame; print --init-bbox coordinates."""

import argparse
import sys

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick init bbox on a video frame (same box as CVAT Track seed)."
    )
    parser.add_argument("--video", required=True, help="Path to .mp4")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (0-based)")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}", file=sys.stderr)
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame < 0 or (total > 0 and args.frame >= total):
        print(f"ERROR: --frame {args.frame} out of range (0..{max(total - 1, 0)})", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = cap.read()
    cap.release()
    if not ok or image is None:
        print(f"ERROR: cannot read frame {args.frame}", file=sys.stderr)
        return 1

    window = f"Frame {args.frame}: drag box around ball, Enter=confirm, c=cancel"
    roi = cv2.selectROI(window, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        print("Cancelled or empty selection.", file=sys.stderr)
        return 1

    x1, y1, x2, y2 = x, y, x + w, y + h
    bbox = f"{x1},{y1},{x2},{y2}"
    print(bbox)
    print(f"\nUse with evaluate_video.py:\n  --init-bbox {bbox}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
