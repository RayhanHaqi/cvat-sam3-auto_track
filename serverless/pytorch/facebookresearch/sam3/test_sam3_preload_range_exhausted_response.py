import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_main_module():
    sam3_dir = Path(__file__).resolve().parent
    sys.modules.setdefault("torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    sys.modules["sam3.model_builder"] = SimpleNamespace(build_sam3_predictor=lambda **_: None)

    for module_name in ("diagnostics", "model_handler"):
        dep_path = sam3_dir / f"{module_name}.py"
        dep_key = f"sam3_terminal_dep_{module_name}_{id(dep_path)}"
        if dep_key not in sys.modules:
            spec = importlib.util.spec_from_file_location(dep_key, dep_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dep_key] = mod
            if module_name == "diagnostics":
                sys.modules["diagnostics"] = mod
            else:
                sys.modules["model_handler"] = mod
            spec.loader.exec_module(mod)

    main_path = sam3_dir / "main.py"
    main_key = f"sam3_terminal_main_{id(main_path)}"
    spec = importlib.util.spec_from_file_location(main_key, main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[main_key] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeContext:
    def __init__(self, model):
        self.user_data = SimpleNamespace(model=model)
        self.logger = SimpleNamespace(info=lambda *_a, **_k: None, warn=lambda *_a, **_k: None)

    class Response:
        def __init__(self, body, headers, content_type, status_code):
            self.body = body
            self.headers = headers
            self.content_type = content_type
            self.status_code = status_code


def test_preload_range_exhausted_returns_terminal_http_200():
    main = _load_main_module()
    PreloadRangeExhaustedError = main.PreloadRangeExhaustedError

    class _Model:
        def infer_batch(self, *_args, **_kwargs):
            raise PreloadRangeExhaustedError(
                "exhausted",
                base_frame=10,
                preloaded_count=12,
                preloaded_until_frame=21,
                requested_frame=22,
            )

    context = _FakeContext(_Model())
    event = SimpleNamespace(
        body={
            "frame_index": 22,
            "states": [{"session_key": "k", "base_frame": 10, "preloaded_count": 12}],
        }
    )
    response = main.handler(context, event)
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["tracking_status"] == "preload_exhausted"
    assert payload["tracking_stop_reason"] == "tracking_window_complete"
    assert payload["preloaded_until_frame"] == 21
    assert payload["shapes"] == []
    assert payload["states"] == []


def test_preload_range_exhausted_error_carries_server_frame_metadata():
    main = _load_main_module()
    PreloadRangeExhaustedError = main.PreloadRangeExhaustedError

    exc = PreloadRangeExhaustedError(
        "exhausted",
        base_frame=10,
        preloaded_count=12,
        preloaded_until_frame=21,
        requested_frame=22,
    )
    assert exc.preloaded_until_frame == 21
    assert exc.base_frame == 10
    assert exc.preloaded_count == 12
    assert exc.requested_frame == 22


def test_validation_error_still_returns_http_400():
    main = _load_main_module()

    class _Model:
        def infer_batch(self, *_args, **_kwargs):
            raise main.ValidationError("bad request")

    context = _FakeContext(_Model())
    event = SimpleNamespace(body={"frame_index": 1, "states": [{}]})
    response = main.handler(context, event)
    assert response.status_code == 400
    payload = json.loads(response.body)
    assert "error" in payload
    assert "tracking_status" not in payload
