"""Unit tests for SAM2 lost-target / null bbox continuation (no GPU)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

_HANDLER_PATH = Path(__file__).resolve().parent / "model_handler.py"
_spec = importlib.util.spec_from_file_location("sam2_model_handler_lost", _HANDLER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sam2.build_sam"] = SimpleNamespace(build_sam2_camera_predictor=lambda *_args, **_kwargs: MagicMock())
_spec.loader.exec_module(_mod)
ModelHandler = _mod.ModelHandler


def _empty_logits():
    return torch.zeros((1, 1, 32, 32), dtype=torch.float32)


def test_propagate_returns_none_when_mask_empty():
    handler = ModelHandler.__new__(ModelHandler)
    handler.predictor = MagicMock()
    handler.predictor.track.return_value = ([1], [_empty_logits()])

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{"obj_id": 1, "last_bbox": [1.0, 1.0, 5.0, 5.0]}],
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_propagate_returns_none_when_object_missing():
    handler = ModelHandler.__new__(ModelHandler)
    handler.predictor = MagicMock()
    handler.predictor.track.return_value = ([], [])

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{"obj_id": 1, "last_bbox": [1.0, 1.0, 5.0, 5.0]}],
    )
    assert shapes == [None]
    assert states[0]["lost"] is True


def test_propagate_keeps_lost_state():
    handler = ModelHandler.__new__(ModelHandler)
    handler.predictor = MagicMock()

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    shapes, states = handler.infer_batch(
        image,
        [None],
        [{"obj_id": 1, "lost": True, "last_bbox": None}],
    )
    assert shapes == [None]
    assert states[0]["lost"] is True
    handler.predictor.track.assert_not_called()
