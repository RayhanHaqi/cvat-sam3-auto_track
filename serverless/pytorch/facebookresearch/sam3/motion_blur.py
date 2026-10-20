"""Canonical motion-blur ball-center estimator for CVAT SAM3 masks.

Used by the default ``motion_blur_center_box`` output policy (C).  The former
``sam_box`` policy remains an explicit override.  This module is also
importable standalone for offline geometric validation on saved masks.

Method per frame:
  * PCA on current-mask pixel coordinates gives the centroid, a major axis
    ``e_major`` and a minor axis ``e_minor`` (eigenvector sign is arbitrary).
  * Apparent ball diameter ``D`` = minor-axis extent of the mask.  No physical
    ball size and no P5/P95 clipping is used.
  * Neighbouring mask centroids give the temporal displacement
    ``v = c_{t+1} - c_{t-1}``, used only to orient ``e_major`` with ``v``.
  * Leading boundary ``u_head`` = max projection on the oriented major axis.
  * Target = centroid + (u_head - D/2 - (1-alpha)(L-D)) * e_major.
    Default alpha=0.5 targets the major-axis midpoint (mid-exposure).

Any invalid/tiny mask, insufficient elongation (eigenvalue ratio below the
explicit configured threshold), missing neighbour, or degenerate motion
falls back to the current-mask centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Validated C parameters; the handler can override them from env.
MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT = 1.44  # variance ratio (~1.2 length ratio)
MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT = 6
# 1.0 = end of exposure (original formula); 0.5 = mid exposure (validated on
# tasks 25-27: median 0.44 px vs annotation centre, against 2.08 px at 1.0).
MOTION_BLUR_TARGET_FRACTION_DEFAULT = 0.5
_DEGENERATE_EPS = 1e-6


@dataclass(frozen=True)
class MotionBlurEstimate:
    used_correction: bool
    fallback_reason: str | None
    mask_pixels: int
    centroid_xy: tuple[float, float] | None
    lambda_major: float | None
    lambda_minor: float | None
    eigen_ratio: float | None
    diameter: float | None
    major_axis_length: float | None
    motion_vector: tuple[float, float] | None
    e_major: tuple[float, float] | None
    e_minor: tuple[float, float] | None
    leading_edge: float | None
    target_xy: tuple[float, float] | None
    centroid_to_target: float | None
    target_fraction: float | None
    exposure_fraction_estimate: float | None
    diameter_override: float | None


def mask_points(mask):
    """Mask pixel coordinates as float ``(N, 2)`` in ``(x, y)`` order."""
    if mask is None:
        return np.empty((0, 2), dtype=np.float64)
    coords = np.argwhere(np.asarray(mask) > 0)
    return coords[:, ::-1].astype(np.float64)


def mask_centroid(mask):
    points = mask_points(mask)
    if points.size == 0:
        return None
    return tuple(float(v) for v in points.mean(axis=0))


def _to_grayscale(image):
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    return (
        0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    ).astype(np.float64)


def intensity_weighted_centroid(image, mask, *, margin=6.0):
    """Intensity-weighted centre inside a mask-anchored ROI.

    The SAM3 mask usually covers the bright core only; weighting the grayscale
    pixels above the ROI background keeps the ball's darker rim contributable
    without thresholding it into the mask.  Returns ``(cx, cy)`` or ``None``.
    """
    if image is None or mask is None:
        return None
    gray = _to_grayscale(image)
    binary = np.asarray(mask) > 0
    coords = np.argwhere(binary)
    if coords.size == 0:
        return None
    height, width = gray.shape
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = int(round(max(float(margin), 2.0)))
    y0 = max(0, y_min - pad)
    y1 = min(height, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(width, x_max + pad + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    background = float(np.median(roi))
    weights = np.clip(roi - background, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None
    yy, xx = np.nonzero(weights)
    values = weights[yy, xx]
    return (
        float((xx * values).sum() / total) + x0,
        float((yy * values).sum() / total) + y0,
    )


def _fallback(points, reason, *, mask_pixels=None, **fields):
    centroid = fields.pop("centroid_xy", None)
    if centroid is None and points.size:
        centroid = tuple(float(v) for v in points.mean(axis=0))
    base = {
        "used_correction": False,
        "fallback_reason": reason,
        "mask_pixels": len(points) if mask_pixels is None else int(mask_pixels),
        "centroid_xy": centroid,
        "lambda_major": None,
        "lambda_minor": None,
        "eigen_ratio": None,
        "diameter": None,
        "major_axis_length": None,
        "motion_vector": None,
        "e_major": None,
        "e_minor": None,
        "leading_edge": None,
        "target_xy": centroid,
        "centroid_to_target": 0.0 if centroid is not None else None,
        "target_fraction": None,
        "exposure_fraction_estimate": None,
        "diameter_override": None,
    }
    base.update(fields)
    return MotionBlurEstimate(**base)


def estimate_motion_blur_center(
    mask,
    previous_centroid=None,
    next_centroid=None,
    *,
    min_eigen_ratio=MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT,
    min_mask_pixels=MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT,
    target_fraction=MOTION_BLUR_TARGET_FRACTION_DEFAULT,
    center_xy=None,
    diameter_override=None,
):
    """Return a :class:`MotionBlurEstimate` for one SAM3 mask.

    ``center_xy`` optionally overrides the mask centroid as the projection
    origin (e.g. an intensity-weighted centre); the PCA axes are unchanged.
    ``diameter_override`` optionally replaces the mask's own minor extent as
    the effective ball diameter (e.g. a value stabilised from sharp frames).
    """
    points = mask_points(mask)
    count = len(points)
    if count == 0:
        return _fallback(points, "invalid_mask", mask_pixels=0, centroid_xy=center_xy)
    if count < int(min_mask_pixels):
        return _fallback(points, "tiny_mask", centroid_xy=center_xy)

    mean = points.mean(axis=0)
    origin = mean if center_xy is None else np.asarray(center_xy, dtype=np.float64)
    centered = points - origin
    covariance = centered.T @ centered / float(count)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        return _fallback(points, "degenerate_covariance")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    lambda_major = float(eigenvalues[0])
    lambda_minor = float(eigenvalues[1])
    if lambda_major <= 0.0 or lambda_minor <= 0.0:
        return _fallback(points, "degenerate_covariance")

    ratio = lambda_major / lambda_minor
    e_major = eigenvectors[:, 0]
    e_minor = eigenvectors[:, 1]
    pca_fields = {
        "centroid_xy": (float(origin[0]), float(origin[1])),
        "lambda_major": lambda_major,
        "lambda_minor": lambda_minor,
        "eigen_ratio": ratio,
        "e_major": (float(e_major[0]), float(e_major[1])),
        "e_minor": (float(e_minor[0]), float(e_minor[1])),
        "diameter_override": None
        if diameter_override is None
        else float(diameter_override),
    }
    projections_major = centered @ e_major
    projections_minor = centered @ e_minor
    diameter = float(projections_minor.max() - projections_minor.min())
    major_axis_length = float(projections_major.max() - projections_major.min())
    if diameter <= 0.0:
        return _fallback(points, "degenerate_diameter", **pca_fields)
    extent_fields = dict(
        pca_fields, diameter=diameter, major_axis_length=major_axis_length
    )
    motion = None
    if previous_centroid is not None and next_centroid is not None:
        motion = np.asarray(next_centroid, dtype=np.float64) - np.asarray(
            previous_centroid, dtype=np.float64
        )
        extent_fields["motion_vector"] = (float(motion[0]), float(motion[1]))
    if ratio < float(min_eigen_ratio):
        return _fallback(points, "insufficient_elongation", **extent_fields)
    if motion is None:
        return _fallback(points, "missing_neighbor_frame", **extent_fields)
    motion_norm = float(np.linalg.norm(motion))
    if motion_norm < _DEGENERATE_EPS:
        return _fallback(points, "degenerate_motion", **extent_fields)

    orientation = float(e_major @ motion)
    if abs(orientation) <= _DEGENERATE_EPS * motion_norm:
        return _fallback(points, "degenerate_motion", **extent_fields)
    if orientation < 0.0:
        e_major = -e_major
        projections_major = -projections_major

    leading_edge = float(projections_major.max())
    effective_diameter = diameter if diameter_override is None else float(diameter_override)
    # Blur displacement in frame units: (L - D) over the per-frame displacement
    # |v| / 2 (v is measured across two frame intervals).
    exposure_fraction = (major_axis_length - effective_diameter) / (motion_norm / 2.0)
    offset = leading_edge - effective_diameter / 2.0 - (1.0 - float(target_fraction)) * (
        major_axis_length - effective_diameter
    )
    target = origin + offset * e_major
    return MotionBlurEstimate(
        used_correction=True,
        fallback_reason=None,
        mask_pixels=count,
        centroid_xy=(float(origin[0]), float(origin[1])),
        lambda_major=lambda_major,
        lambda_minor=lambda_minor,
        eigen_ratio=ratio,
        diameter=diameter,
        major_axis_length=major_axis_length,
        motion_vector=(float(motion[0]), float(motion[1])),
        e_major=(float(e_major[0]), float(e_major[1])),
        e_minor=(float(e_minor[0]), float(e_minor[1])),
        leading_edge=leading_edge,
        target_xy=(float(target[0]), float(target[1])),
        centroid_to_target=float(abs(offset)),
        target_fraction=float(target_fraction),
        exposure_fraction_estimate=float(exposure_fraction),
        diameter_override=None
        if diameter_override is None
        else float(diameter_override),
    )


def median_sharp_diameter(
    masks,
    *,
    max_ratio=MOTION_BLUR_MIN_EIGEN_RATIO_DEFAULT,
    min_mask_pixels=MOTION_BLUR_MIN_MASK_PIXELS_DEFAULT,
):
    """Median minor extent of nearby non-elongated masks, or ``None``.

    Pure-mask stabiliser for the emitted box size: blurred frames inflate the
    measured width, so only masks with eigen ratio at or below ``max_ratio``
    contribute.  Returns ``None`` when no sharp mask is available.
    """
    diameters = []
    for mask in masks:
        estimate = estimate_motion_blur_center(
            mask, min_mask_pixels=min_mask_pixels
        )
        if estimate.eigen_ratio is None or estimate.diameter is None:
            continue
        if estimate.eigen_ratio <= float(max_ratio):
            diameters.append(estimate.diameter)
    if not diameters:
        return None
    return float(np.median(diameters))


def motion_blur_report_row(estimate, frame_id=None):
    """Per-frame report row required by the experiment evidence."""
    return {
        "frame_id": frame_id,
        "lambda_major": estimate.lambda_major,
        "lambda_minor": estimate.lambda_minor,
        "eigenvalue_ratio": estimate.eigen_ratio,
        "estimated_diameter": estimate.diameter,
        "major_axis_mask_length": estimate.major_axis_length,
        "motion_vector": list(estimate.motion_vector)
        if estimate.motion_vector is not None
        else None,
        "mask_centroid": list(estimate.centroid_xy)
        if estimate.centroid_xy is not None
        else None,
        "experimental_target": list(estimate.target_xy)
        if estimate.target_xy is not None
        else None,
        "centroid_to_target_displacement": estimate.centroid_to_target,
        "target_fraction": estimate.target_fraction,
        "exposure_fraction_estimate": estimate.exposure_fraction_estimate,
        "diameter_override": estimate.diameter_override,
        "used_correction": estimate.used_correction,
        "fallback_reason": estimate.fallback_reason,
    }
