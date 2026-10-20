"""Deterministic geometry, canonical C defaults and legacy override (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _load("sam3_motion_blur", _HERE / "motion_blur.py")
sys.modules.setdefault(
    "sam3.model_builder", SimpleNamespace(build_sam3_predictor=lambda **_: None)
)
handler_mod = _load("sam3_model_handler_motion_blur", _HERE / "model_handler.py")


def _rect_mask(cx, cy, half_len, half_width, frame=200):
    mask = np.zeros((frame, frame), dtype=np.uint8)
    mask[cy - half_width : cy + half_width, cx - half_len : cx + half_len] = 1
    return mask


def _disk_mask(cx, cy, radius, frame=200):
    yy, xx = np.ogrid[:frame, :frame]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2).astype(np.uint8)


def _diagonal_mask(cx=100, cy=100, half_len=20, half_width=4, frame=200):
    mask = np.zeros((frame, frame), dtype=np.uint8)
    for t in range(-half_len, half_len):
        x, y = cx + t, cy + t
        mask[y - half_width : y + half_width + 1, x - half_width : x + half_width + 1] = 1
    return mask


def _handler(policy=None):
    handler = handler_mod.ModelHandler.__new__(handler_mod.ModelHandler)
    handler.config = handler_mod.load_sam3_config(
        {} if policy is None else {"SAM3_OUTPUT_POLICY": policy}
    )
    handler._maybe_refine_bbox = lambda image, bbox, previous_bbox=None: bbox
    return handler


def test_circular_mask_falls_back_to_centroid():
    mask = _disk_mask(100, 100, 8)
    est = mb.estimate_motion_blur_center(mask, (94.0, 100.0), (106.0, 100.0))
    assert est.used_correction is False
    assert est.fallback_reason == "insufficient_elongation"
    assert est.target_xy == pytest.approx(est.centroid_xy)
    assert est.eigen_ratio == pytest.approx(1.0, abs=0.05)


def test_horizontal_mask_moving_right_targets_right_edge_minus_half_diameter():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    est = mb.estimate_motion_blur_center(
        mask, (90.0, 100.0), (110.0, 100.0), target_fraction=1.0
    )
    assert est.used_correction is True
    right_edge = 100 + 20 - 1
    assert est.target_xy[0] == pytest.approx(right_edge - est.diameter / 2.0, abs=0.51)
    assert est.target_xy[1] == pytest.approx(100.0, abs=0.51)
    assert est.motion_vector[0] > 0.0
    assert est.e_major[0] > 0.0


def test_horizontal_mask_moving_left_targets_left_edge_plus_half_diameter():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    est = mb.estimate_motion_blur_center(
        mask, (110.0, 100.0), (90.0, 100.0), target_fraction=1.0
    )
    assert est.used_correction is True
    left_edge = 100 - 20
    assert est.target_xy[0] == pytest.approx(left_edge + est.diameter / 2.0, abs=0.51)
    assert est.motion_vector[0] < 0.0
    assert est.e_major[0] < 0.0


def test_diagonal_mask_uses_pca_axes_and_orients_with_motion():
    mask = _diagonal_mask()
    est = mb.estimate_motion_blur_center(
        mask, (90.0, 90.0), (110.0, 110.0), target_fraction=1.0
    )
    assert est.used_correction is True
    diagonal = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert abs(float(np.dot(est.e_major, diagonal))) == pytest.approx(1.0, abs=1e-6)
    assert est.target_xy[0] - est.centroid_xy[0] > 0.0
    assert est.target_xy[1] - est.centroid_xy[1] > 0.0
    assert 0.0 < est.centroid_to_target < est.major_axis_length


def test_eigenvector_sign_does_not_change_result_after_orientation(monkeypatch):
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    before = mb.estimate_motion_blur_center(mask, (90.0, 100.0), (110.0, 100.0))
    real_eigh = np.linalg.eigh

    def flipped(covariance):
        values, vectors = real_eigh(covariance)
        return values, -vectors

    monkeypatch.setattr(np.linalg, "eigh", flipped)
    after = mb.estimate_motion_blur_center(mask, (90.0, 100.0), (110.0, 100.0))
    assert after.target_xy == pytest.approx(before.target_xy)
    assert after.e_major == pytest.approx(before.e_major)
    assert after.leading_edge == pytest.approx(before.leading_edge)


def test_missing_neighbour_information_falls_back_to_centroid():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    est = mb.estimate_motion_blur_center(mask, None, None)
    assert est.used_correction is False
    assert est.fallback_reason == "missing_neighbor_frame"
    assert est.target_xy == pytest.approx(est.centroid_xy)
    assert est.diameter is not None


def test_invalid_and_tiny_masks_fall_back_safely():
    empty = np.zeros((50, 50), dtype=np.uint8)
    est = mb.estimate_motion_blur_center(empty, (1.0, 1.0), (2.0, 2.0))
    assert est.used_correction is False
    assert est.fallback_reason == "invalid_mask"
    assert est.target_xy is None

    tiny = np.zeros((50, 50), dtype=np.uint8)
    tiny[10:12, 10:12] = 1
    est = mb.estimate_motion_blur_center(tiny, (1.0, 1.0), (2.0, 2.0))
    assert est.used_correction is False
    assert est.fallback_reason == "tiny_mask"
    assert est.target_xy == pytest.approx((10.5, 10.5))


def test_c_is_default_and_sam_box_is_an_explicit_rollback():
    assert handler_mod.Sam3Config().output_policy == "motion_blur_center_box"
    default = handler_mod.load_sam3_config({})
    assert default.output_policy == "motion_blur_center_box"
    assert default.motion_blur_target_fraction == 0.5
    assert default.motion_blur_min_eigen_ratio == 1.44
    assert default.motion_blur_min_mask_pixels == 6
    assert default.motion_blur_intensity_center is False
    assert default.motion_blur_stabilized_diameter is False
    for policy in ("", "unknown"):
        assert handler_mod.load_sam3_config(
            {"SAM3_OUTPUT_POLICY": policy}
        ).output_policy == default.output_policy
    legacy = handler_mod.load_sam3_config({"SAM3_OUTPUT_POLICY": "sam_box"})
    assert legacy.output_policy == "sam_box"


def test_default_policy_emits_mid_exposure_mask_geometry_through_handler():
    mask = _rect_mask(100, 100, 20, 4)
    previous = {"out_binary_masks": [_rect_mask(90, 100, 20, 4)]}
    following = {"out_binary_masks": [_rect_mask(110, 100, 20, 4)]}
    prompt = [80.0, 96.0, 120.0, 104.0]
    sess = {"image_height": 200, "image_width": 200}
    bbox, lost, diag, _canonical = _handler()._postprocess_frame_outputs(
        sess,
        np.zeros((200, 200, 3), dtype=np.uint8),
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=prompt,
        last_known_bbox=prompt,
        prev_lost=False,
        relative_frame=1,
        previous_outputs=previous,
        next_outputs=following,
    )
    assert lost is False
    assert diag["used_correction"] is True
    assert bbox is not None
    assert diag["target_fraction"] == 0.5
    assert diag["center_source"] == "mask"
    assert handler_mod.bbox_center(bbox) == pytest.approx((99.5, 99.5))
    assert bbox[2] - bbox[0] == pytest.approx(7.0)
    assert bbox[3] - bbox[1] == pytest.approx(7.0)
    assert _canonical != pytest.approx(bbox)  # Emission never feeds back as reference.


def test_legacy_policy_ignores_motion_blur_branch():
    prompt = [80.0, 80.0, 120.0, 120.0]
    sess = {"image_height": 200, "image_width": 200}
    outputs = {"out_boxes_xywh": [[0.4, 0.4, 0.2, 0.2]]}
    bbox, lost, diag, _canonical = _handler("sam_box")._postprocess_frame_outputs(
        sess,
        np.zeros((200, 200, 3), dtype=np.uint8),
        outputs,
        is_init_frame=False,
        prompt_bbox=prompt,
        last_known_bbox=prompt,
        prev_lost=False,
        relative_frame=1,
        previous_outputs={"out_binary_masks": [_rect_mask(90, 100, 20, 4)]},
        next_outputs={"out_binary_masks": [_rect_mask(110, 100, 20, 4)]},
    )
    assert lost is False
    assert diag is None
    assert bbox == pytest.approx(handler_mod.xywh_rel_to_abs([0.4, 0.4, 0.2, 0.2], 200, 200))


def test_target_fraction_controls_exposure_anchor():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    neighbors = ((90.0, 100.0), (110.0, 100.0))
    end = mb.estimate_motion_blur_center(mask, *neighbors, target_fraction=1.0)
    mid = mb.estimate_motion_blur_center(mask, *neighbors, target_fraction=0.5)
    start = mb.estimate_motion_blur_center(mask, *neighbors, target_fraction=0.0)
    assert end.target_xy[0] > mid.target_xy[0] > start.target_xy[0]
    assert mid.target_xy == pytest.approx(mid.centroid_xy, abs=0.51)
    assert end.target_xy[0] == pytest.approx(119.0 - end.diameter / 2.0, abs=0.51)
    assert start.target_xy[0] == pytest.approx(80.0 + start.diameter / 2.0, abs=0.51)
    assert (end.target_fraction, mid.target_fraction, start.target_fraction) == (
        1.0,
        0.5,
        0.0,
    )


def test_exposure_fraction_estimate():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    estimate = mb.estimate_motion_blur_center(mask, (90.0, 100.0), (110.0, 100.0))
    # L=39 px, D=7 px, |v|=20 px over two frame intervals -> 10 px/frame
    assert estimate.exposure_fraction_estimate == pytest.approx(3.2, abs=0.05)


def test_target_fraction_config_env():
    assert handler_mod.Sam3Config().motion_blur_target_fraction == 0.5
    configured = handler_mod.load_sam3_config(
        {"SAM3_MOTION_BLUR_TARGET_FRACTION": "1.0"}
    )
    assert configured.motion_blur_target_fraction == 1.0


def test_intensity_weighted_centroid_follows_bright_blob():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    image = np.zeros((200, 200), dtype=np.uint8)
    image[97:103, 101:107] = 200
    binary = mb.mask_centroid(mask)
    weighted = mb.intensity_weighted_centroid(image, mask, margin=4)
    assert weighted is not None
    assert weighted[0] > binary[0] + 2.0
    assert weighted[1] == pytest.approx(binary[1], abs=1.0)


def test_estimator_center_override_preserves_perpendicular_offset():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    neighbors = ((90.0, 100.0), (110.0, 100.0))
    baseline = mb.estimate_motion_blur_center(mask, *neighbors, target_fraction=0.5)
    shifted = mb.estimate_motion_blur_center(
        mask, *neighbors, target_fraction=0.5, center_xy=(100.0, 96.0)
    )
    assert shifted.centroid_xy == pytest.approx((100.0, 96.0))
    assert shifted.target_xy[1] == pytest.approx(96.0, abs=0.01)
    assert shifted.target_xy[0] == pytest.approx(baseline.target_xy[0], abs=0.51)


def test_intensity_center_config_env():
    assert handler_mod.Sam3Config().motion_blur_intensity_center is False
    enabled = handler_mod.load_sam3_config(
        {"SAM3_MOTION_BLUR_INTENSITY_CENTER": "1"}
    )
    assert enabled.motion_blur_intensity_center is True


def test_median_sharp_diameter_ignores_elongated_masks():
    sharp_small = _disk_mask(100, 100, 5)
    sharp_big = _disk_mask(100, 100, 6)
    elongated = _rect_mask(100, 100, half_len=20, half_width=4)
    value = mb.median_sharp_diameter([sharp_small, elongated, sharp_big])
    assert value == pytest.approx(11.0, abs=0.51)


def test_diameter_override_changes_target_scale_not_measurement():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    neighbors = ((90.0, 100.0), (110.0, 100.0))
    baseline = mb.estimate_motion_blur_center(mask, *neighbors, target_fraction=1.0)
    overridden = mb.estimate_motion_blur_center(
        mask, *neighbors, target_fraction=1.0, diameter_override=10.0
    )
    assert overridden.diameter == pytest.approx(baseline.diameter)
    assert overridden.diameter_override == 10.0
    assert overridden.target_xy[0] == pytest.approx(114.0, abs=0.51)
    assert overridden.target_xy[0] < baseline.target_xy[0]


def test_handler_uses_stabilized_diameter_from_session():
    mask = _rect_mask(100, 100, half_len=20, half_width=4)
    prompt = [80.0, 96.0, 120.0, 104.0]
    sess = {
        "image_height": 200,
        "image_width": 200,
        "motion_blur_stable_diameter": {1: 10.0},
    }
    handler = handler_mod.ModelHandler.__new__(handler_mod.ModelHandler)
    handler.config = handler_mod.Sam3Config(
        output_policy="motion_blur_center_box",
        motion_blur_stabilized_diameter=True,
    )
    handler._maybe_refine_bbox = lambda image, bbox, previous_bbox=None: bbox
    bbox, lost, diag, _canonical = handler._postprocess_frame_outputs(
        sess,
        np.zeros((200, 200, 3), dtype=np.uint8),
        {"out_binary_masks": [mask]},
        is_init_frame=False,
        prompt_bbox=prompt,
        last_known_bbox=prompt,
        prev_lost=False,
        relative_frame=1,
        previous_outputs={"out_binary_masks": [_rect_mask(90, 100, 20, 4)]},
        next_outputs={"out_binary_masks": [_rect_mask(110, 100, 20, 4)]},
    )
    assert lost is False
    assert diag["diameter_override"] == 10.0
    assert bbox[2] - bbox[0] == pytest.approx(10.0, abs=0.51)


def test_stabilized_diameter_config_env():
    default = handler_mod.Sam3Config()
    assert default.motion_blur_stabilized_diameter is False
    assert default.motion_blur_diameter_window == 2
    enabled = handler_mod.load_sam3_config(
        {
            "SAM3_MOTION_BLUR_STABILIZED_DIAMETER": "1",
            "SAM3_MOTION_BLUR_DIAMETER_WINDOW": "4",
        }
    )
    assert enabled.motion_blur_stabilized_diameter is True
    assert enabled.motion_blur_diameter_window == 4
