# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rest_framework import status

from cvat.apps.lambda_manager.tests.test_lambda import (
    LAMBDA_FUNCTIONS_PATH,
    _LambdaTestCaseBase,
    functions,
    id_function_sam3_tracker,
    id_function_tracker,
    tasks,
)
from cvat.apps.lambda_manager.views import (
    SAM3_PRELOAD_CHUNK_CAP,
    SAM3_TRACKER_FUNCTION_ID,
    LambdaFunction,
    apply_sam3_source_timeline_guard,
    parse_sam3_raw_source_frame_index,
)


def _load_sam3_handler_module():
    handler_path = (
        Path(__file__).resolve().parents[4]
        / "serverless/pytorch/facebookresearch/sam3/model_handler.py"
    )
    module_name = f"sam3_model_handler_timeline_{id(handler_path)}"
    sys.modules.setdefault("torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    sys.modules.setdefault(
        "sam3.model_builder",
        SimpleNamespace(build_sam3_predictor=lambda **_: None),
    )
    sys.modules.setdefault(
        "diagnostics",
        SimpleNamespace(
            bbox_summary=lambda bbox: bbox,
            enabled=lambda: False,
            log_frame_record=lambda *_args, **_kwargs: None,
            summarize_outputs=lambda *_args, **_kwargs: {},
        ),
    )
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task9_style_paths(cvat_start: int = 100, cvat_count: int = 96) -> dict[int, str]:
    paths: dict[int, str] = {}
    raw_index = 11848
    for offset in range(cvat_count):
        cvat_frame = cvat_start + offset
        if cvat_frame == 112:
            raw_index = 12004
        paths[cvat_frame] = f"annotate_test/frame_{raw_index}.png"
        if cvat_frame < 112:
            raw_index += 1
        else:
            raw_index += 1
    return paths


class Sam3SourceTimelineGuardUnitTests(_LambdaTestCaseBase):
    def test_parse_raw_source_frame_index_from_basename(self):
        self.assertEqual(parse_sam3_raw_source_frame_index("frame_11848.png"), 11848)
        self.assertEqual(
            parse_sam3_raw_source_frame_index("annotate_test/frame_11859.PNG"),
            11859,
        )
        self.assertIsNone(parse_sam3_raw_source_frame_index("image_0.jpg"))
        self.assertIsNone(parse_sam3_raw_source_frame_index(None))

    def test_contiguous_raw_sequence_keeps_chunk_cap(self):
        candidates = list(range(0, SAM3_PRELOAD_CHUNK_CAP))
        paths = {frame: f"frame_{1000 + frame}.png" for frame in candidates}

        included, diag = apply_sam3_source_timeline_guard(
            candidates,
            lambda frame_idx: paths.get(frame_idx),
        )

        self.assertEqual(len(included), SAM3_PRELOAD_CHUNK_CAP)
        self.assertTrue(diag["sourceTimelineVerified"])
        self.assertEqual(diag["preloadStartRawIndex"], 1000)
        self.assertEqual(diag["preloadEndRawIndex"], 1000 + SAM3_PRELOAD_CHUNK_CAP - 1)
        self.assertIsNone(diag["firstSourceGapAtCvatFrame"])
        self.assertEqual(diag["preloadStopReason"], "chunk_cap")

    def test_source_jump_truncates_before_discontinuity(self):
        paths = _task9_style_paths()
        candidates = list(range(100, 100 + 96))

        included, diag = apply_sam3_source_timeline_guard(
            candidates,
            lambda frame_idx: paths.get(frame_idx),
        )

        self.assertEqual(included, list(range(100, 112)))
        self.assertEqual(len(included), 12)
        self.assertTrue(diag["sourceTimelineVerified"])
        self.assertEqual(diag["preloadStartRawIndex"], 11848)
        self.assertEqual(diag["preloadEndRawIndex"], 11859)
        self.assertEqual(diag["firstSourceGapAtCvatFrame"], 112)
        self.assertEqual(diag["firstSourceGapDelta"], 145)
        self.assertEqual(diag["preloadStopReason"], "source_timeline_gap")

    def test_unparseable_filenames_keep_cvat_contiguous_behavior(self):
        candidates = list(range(0, 5))
        paths = {
            0: "frame_10.png",
            1: "frame_11.png",
            2: "not_a_frame_name.jpg",
            3: "frame_13.png",
            4: "frame_14.png",
        }

        included, diag = apply_sam3_source_timeline_guard(
            candidates,
            lambda frame_idx: paths.get(frame_idx),
        )

        self.assertEqual(included, candidates)
        self.assertFalse(diag["sourceTimelineVerified"])
        self.assertEqual(diag["preloadStopReason"], "continuity_unverified")


class Sam3SourceTimelineGuardIntegrationTests(_LambdaTestCaseBase):
    def setUp(self):
        super().setUp()
        images_main_task = self._generate_task_images(3)
        self.main_task = self._create_task(tasks["main"], images_main_task)

    def _sam3_function(self) -> LambdaFunction:
        from cvat.apps.lambda_manager.views import LambdaGateway

        gateway = LambdaGateway()
        return LambdaFunction(gateway, functions["positive"][SAM3_TRACKER_FUNCTION_ID])

    def test_gap_excluded_frames_are_not_fetched_for_preload(self):
        from cvat.apps.engine.models import Job, Task

        func = self._sam3_function()
        db_task = Task.objects.get(pk=self.main_task["id"])
        db_job = Job.objects.get(segment__task_id=db_task.id)
        paths = _task9_style_paths()
        captured_frames: list[int] = []
        frame_b64 = base64.b64encode(b"jpeg").decode("ascii")

        def capture_get_image(_db_task, frame_idx):
            captured_frames.append(frame_idx)
            return frame_b64

        with mock.patch.object(
            func,
            "_sam3_preload_candidate_frame_indices",
            return_value=list(range(100, 196)),
        ):
            with mock.patch.object(
                func,
                "_sam3_source_frame_path",
                side_effect=lambda _task, frame_idx: paths.get(frame_idx),
            ):
                with mock.patch.object(func, "_get_image", side_effect=capture_get_image):
                    fields = func._build_sam3_preload_fields(db_task, db_job, 100)

        self.assertEqual(fields["preload_base_frame"], 100)
        self.assertEqual(fields["preload_frame_count"], 12)
        self.assertEqual(captured_frames, list(range(100, 112)))
        self.assertNotIn(112, captured_frames)

    def test_sam3_init_resolves_timeline_before_any_get_image(self):
        from cvat.apps.engine.models import Job, Task

        job = Job.objects.get(segment__task_id=self.main_task["id"])
        db_task = Task.objects.get(pk=self.main_task["id"])
        func = self._sam3_function()
        paths = {
            0: "frame_10.png",
            1: "frame_11.png",
            2: "frame_20.png",
            3: "frame_21.png",
        }
        call_log: list[str | tuple[str, int]] = []
        frame_b64 = base64.b64encode(b"jpeg").decode("ascii")
        original_resolve = LambdaFunction._resolve_sam3_preload_frames

        def tracked_resolve(self, db_task_arg, db_job, base_frame):
            call_log.append("resolve")
            return original_resolve(self, db_task_arg, db_job, base_frame)

        def capture_get_image(_db_task, frame_idx):
            call_log.append(("get_image", frame_idx))
            return frame_b64

        captured: list[dict] = []

        def capture_gateway_invoke(_func_arg, payload):
            captured.append(dict(payload))
            return {
                "shapes": [[12.34, 34.0, 35.01, 41.99]],
                "states": [{"session_key": "sam3-order-test"}],
            }

        with mock.patch.object(
            func,
            "_sam3_preload_candidate_frame_indices",
            return_value=[0, 1, 2],
        ):
            with mock.patch.object(
                func,
                "_sam3_source_frame_path",
                side_effect=lambda _task, frame_idx: paths.get(frame_idx),
            ):
                with mock.patch.object(
                    LambdaFunction,
                    "_resolve_sam3_preload_frames",
                    autospec=True,
                    side_effect=tracked_resolve,
                ):
                    with mock.patch.object(func, "_get_image", side_effect=capture_get_image):
                        with mock.patch.object(
                            func.gateway,
                            "invoke",
                            side_effect=capture_gateway_invoke,
                        ):
                            func.invoke(
                                db_task,
                                {
                                    "frame": 0,
                                    "shapes": [{
                                        "type": "rectangle",
                                        "points": [12.12, 34.45, 54.0, 76.12],
                                    }],
                                },
                                db_job=job,
                            )

        self.assertEqual(call_log[0], "resolve")
        resolve_index = call_log.index("resolve")
        fetched_frames = [entry[1] for entry in call_log if isinstance(entry, tuple)]
        self.assertEqual(fetched_frames, [0, 1])
        self.assertTrue(
            all(call_log.index(("get_image", frame_idx)) > resolve_index for frame_idx in fetched_frames)
        )
        self.assertNotIn(2, fetched_frames)
        self.assertEqual(captured[0]["preload_frame_count"], 2)
        self.assertEqual(captured[0]["image"], captured[0]["preload_images"][0])

    def test_task9_style_init_fetches_exactly_twelve_frames_including_seed(self):
        from cvat.apps.engine.models import Job, Task

        db_task = Task.objects.get(pk=self.main_task["id"])
        db_job = Job.objects.get(segment__task_id=db_task.id)
        func = self._sam3_function()
        paths = _task9_style_paths()
        captured_frames: list[int] = []
        frame_b64 = base64.b64encode(b"jpeg").decode("ascii")

        with mock.patch.object(
            func,
            "_sam3_preload_candidate_frame_indices",
            return_value=list(range(100, 196)),
        ):
            with mock.patch.object(
                func,
                "_sam3_source_frame_path",
                side_effect=lambda _task, frame_idx: paths.get(frame_idx),
            ):
                with mock.patch.object(
                    func,
                    "_get_image",
                    side_effect=lambda _task, frame_idx: (
                        captured_frames.append(frame_idx) or frame_b64
                    ),
                ):
                    fields = func._build_sam3_preload_fields(db_task, db_job, 100)

        self.assertEqual(fields["preload_frame_count"], 12)
        self.assertEqual(captured_frames, list(range(100, 112)))
        self.assertEqual(len(captured_frames), 12)

    def test_cache_continuation_through_final_pre_gap_frame(self):
        mod = _load_sam3_handler_module()

        handler = mod.ModelHandler.__new__(mod.ModelHandler)
        handler.config = mod.Sam3Config(ir_refine_enabled=False)
        handler.predictor = SimpleNamespace(stream_requests=[])
        preloaded_count = 12
        handler._sessions = {
            "gap-test": {
                "temp_dir": None,
                "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
                "base_frame": 100,
                "preloaded_count": preloaded_count,
                "preloaded_until_frame": 111,
                "frame_cache": {
                    i: {"bbox": [1, 1, 2, 2], "lost": False}
                    for i in range(preloaded_count)
                },
                "cache_ready": True,
            },
        }
        prev_state = {
            "session_key": "gap-test",
            "base_frame": 100,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": 111,
        }
        shapes, states = handler.infer_batch(None, [None], [prev_state], frame_index=111)
        self.assertEqual(shapes, [[1, 1, 2, 2]])
        self.assertNotIn("lost", states[0])

    def test_first_post_gap_frame_returns_range_exhaustion(self):
        mod = _load_sam3_handler_module()

        handler = mod.ModelHandler.__new__(mod.ModelHandler)
        handler.config = mod.Sam3Config(ir_refine_enabled=False)
        handler.predictor = SimpleNamespace(stream_requests=[])
        preloaded_count = 12
        handler._sessions = {
            "gap-test": {
                "temp_dir": None,
                "prompt_bbox": [10.0, 10.0, 30.0, 30.0],
                "base_frame": 100,
                "preloaded_count": preloaded_count,
                "preloaded_until_frame": 111,
                "frame_cache": {
                    i: {"bbox": [1, 1, 2, 2], "lost": False}
                    for i in range(preloaded_count)
                },
                "cache_ready": True,
            },
        }
        prev_state = {
            "session_key": "gap-test",
            "base_frame": 100,
            "preloaded_count": preloaded_count,
            "preloaded_until_frame": 111,
        }
        with self.assertRaises(mod.PreloadRangeExhaustedError):
            handler.infer_batch(None, [None], [prev_state], frame_index=112)

    def test_generic_tracker_payload_unchanged(self):
        captured = []

        def capture_invoke(func, payload):
            captured.append((func.id, dict(payload)))
            return self._invoke_function(func, payload)

        with mock.patch(
            "cvat.apps.lambda_manager.views.LambdaGateway.invoke",
            side_effect=capture_invoke,
        ):
            response = self._post_request(
                f"{LAMBDA_FUNCTIONS_PATH}/{id_function_tracker}",
                self.admin,
                data={
                    "task": self.main_task["id"],
                    "frame": 0,
                    "shapes": [{"type": "rectangle", "points": [12.12, 34.45, 54.0, 76.12]}],
                },
            )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        generic_payload = captured[-1][1]
        self.assertIn("image", generic_payload)
        self.assertNotIn("preload_images", generic_payload)

    def test_timeline_diagnostics_logged_on_init(self):
        from cvat.apps.engine.models import Job, Task

        func = self._sam3_function()
        db_task = Task.objects.get(pk=self.main_task["id"])
        db_job = Job.objects.get(segment__task_id=db_task.id)
        paths = _task9_style_paths()
        frame_b64 = base64.b64encode(b"jpeg").decode("ascii")

        with mock.patch.object(
            func,
            "_sam3_preload_candidate_frame_indices",
            return_value=list(range(100, 196)),
        ):
            with mock.patch.object(
                func,
                "_sam3_source_frame_path",
                side_effect=lambda _task, frame_idx: paths.get(frame_idx),
            ):
                with mock.patch.object(func, "_get_image", return_value=frame_b64):
                    with mock.patch(
                        "cvat.apps.lambda_manager.views.slogger.glob.info",
                    ) as log_mock:
                        func._build_sam3_preload_fields(db_task, db_job, 100)

        timeline_logs = [
            call.args[1]
            for call in log_mock.call_args_list
            if call.args and call.args[0] == "SAM3 preload source timeline: %s"
        ]
        self.assertEqual(len(timeline_logs), 1)
        diag = json.loads(timeline_logs[0])
        self.assertTrue(diag["sourceTimelineVerified"])
        self.assertEqual(diag["preloadStartRawIndex"], 11848)
        self.assertEqual(diag["preloadEndRawIndex"], 11859)
        self.assertEqual(diag["firstSourceGapAtCvatFrame"], 112)
        self.assertEqual(diag["firstSourceGapDelta"], 145)
        self.assertEqual(diag["preloadStopReason"], "source_timeline_gap")
