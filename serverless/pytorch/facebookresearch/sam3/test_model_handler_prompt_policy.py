"""Unit tests for SAM3 prompt/output policy helpers (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam3_model_handler_policy", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)
_spec.loader.exec_module(_mod)

Sam3Config = _mod.Sam3Config
build_add_prompt_request = _mod.build_add_prompt_request
load_sam3_config = _mod.load_sam3_config
select_bbox_from_outputs = _mod.select_bbox_from_outputs


def test_load_sam3_config_defaults_to_box_mode():
    cfg = load_sam3_config({})
    assert cfg.prompt_mode == "box"
    assert cfg.output_policy == "sam_box"
    assert cfg.text_prompt == "ping pong ball"


def test_load_sam3_config_reads_text_box_experiment():
    cfg = load_sam3_config({
        "SAM3_PROMPT_MODE": "text_box",
        "SAM3_TEXT_PROMPT": "table tennis ball",
        "SAM3_OUTPUT_POLICY": "mask_center_box",
        "SAM3_OUTPUT_PROB_THRESH": "0.45",
    })
    assert cfg.prompt_mode == "text_box"
    assert cfg.text_prompt == "table tennis ball"
    assert cfg.output_policy == "mask_center_box"
    assert cfg.output_prob_thresh == 0.45


def test_load_sam3_config_accepts_adaptive_component_square_box():
    cfg = load_sam3_config({"SAM3_OUTPUT_POLICY": "adaptive_component_square_box"})
    assert cfg.output_policy == "adaptive_component_square_box"


def test_build_add_prompt_request_box_mode():
    cfg = Sam3Config(prompt_mode="box")
    req = build_add_prompt_request(
        "sess-1", 0, [100.0, 80.0, 120.0, 100.0], 200, 400, cfg
    )
    assert req["type"] == "add_prompt"
    assert "bounding_boxes" in req
    assert "text" not in req
    assert req["output_prob_thresh"] == 0.3


def test_build_add_prompt_request_text_box_mode():
    cfg = Sam3Config(prompt_mode="text_box", text_prompt="ping pong ball")
    req = build_add_prompt_request(
        "sess-1", 0, [10.0, 10.0, 30.0, 30.0], 100, 100, cfg
    )
    assert req["text"] == "ping pong ball"
    assert req["bounding_boxes"] == [[0.1, 0.1, 0.2, 0.2]]
    assert req["bounding_box_labels"] == [1]


def test_build_add_prompt_request_text_only_mode():
    cfg = Sam3Config(prompt_mode="text", text_prompt="ping pong ball")
    req = build_add_prompt_request(
        "sess-1", 0, [10.0, 10.0, 30.0, 30.0], 100, 100, cfg
    )
    assert req["text"] == "ping pong ball"
    assert "bounding_boxes" not in req


def test_build_add_prompt_request_force_box_only_strips_text():
    cfg = Sam3Config(prompt_mode="text_box", text_prompt="ping pong ball")
    req = build_add_prompt_request(
        "sess-1", 0, [10.0, 10.0, 30.0, 30.0], 100, 100, cfg, force_box_only=True
    )
    assert "bounding_boxes" in req
    assert "text" not in req


def test_select_bbox_prefers_nearest_candidate_not_first():
    cfg = Sam3Config(output_policy="sam_box")
    outputs = {
        "out_boxes_xywh": [
            [0.7, 0.7, 0.05, 0.05],
            [0.1, 0.1, 0.05, 0.05],
        ]
    }
    reference = [5.0, 5.0, 25.0, 25.0]
    bbox = select_bbox_from_outputs(outputs, 100, 100, reference, cfg)
    assert bbox == [10.0, 10.0, 15.0, 15.0]


def test_select_bbox_mask_center_box_uses_reference_size():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[48:52, 58:62] = 1
    cfg = Sam3Config(output_policy="mask_center_box")
    reference = [20.0, 20.0, 40.0, 40.0]
    bbox = select_bbox_from_outputs(
        {"out_binary_masks": [mask]},
        100,
        100,
        reference,
        cfg,
    )
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    assert abs(cx - 60.0) < 1.0
    assert abs(cy - 50.0) < 1.0
    assert (bbox[2] - bbox[0]) >= 10.0
    assert (bbox[3] - bbox[1]) >= 10.0
