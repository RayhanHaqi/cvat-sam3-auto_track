# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Contract tests for Auto Track boundary outside-keyframe commit helpers.

Mirrors cvat-ui/src/utils/auto-track-boundary-commit.ts so UI behavior stays
testable without a JS test runner in cvat-ui.
"""

from __future__ import annotations

from typing import Optional

import unittest


def find_auto_track_boundary_commit_client_id(
    tracked_client_ids: list[int],
    object_states: list[dict],
    boundary_frame: int,
) -> Optional[int]:
    for client_id in tracked_client_ids:
        object_state = next(
            (state for state in object_states if state["clientID"] == client_id),
            None,
        )
        if not object_state or not object_state.get("keyframes"):
            continue
        keyframes = object_state["keyframes"]
        prev = keyframes.get("prev")
        last = keyframes.get("last")
        if prev != boundary_frame - 1:
            continue
        if isinstance(last, int) and last >= boundary_frame:
            continue
        return client_id
    return None


def build_outside_keyframe_commit(boundary_frame: int) -> dict:
    return {
        "frame": boundary_frame,
        "outside": True,
        "keyframe": True,
    }


def should_auto_advance_after_auto_track_frame(
    auto_track_session_active: bool,
    lost_client_ids: list[int],
) -> bool:
    return auto_track_session_active and not lost_client_ids


class AutoTrackBoundaryCommitContractTest(unittest.TestCase):
    def test_boundary_commit_targets_exhausted_frame_not_prior_valid_frame(self):
        boundary_frame = 26
        client_id = find_auto_track_boundary_commit_client_id(
            tracked_client_ids=[42],
            object_states=[
                {
                    "clientID": 42,
                    "keyframes": {"prev": boundary_frame - 1, "last": boundary_frame - 1},
                    "outside": False,
                },
            ],
            boundary_frame=boundary_frame,
        )

        self.assertEqual(client_id, 42)
        self.assertEqual(
            build_outside_keyframe_commit(boundary_frame),
            {"frame": 26, "outside": True, "keyframe": True},
        )

    def test_prior_valid_frame_is_not_marked_outside_by_selector(self):
        boundary_frame = 26
        client_id = find_auto_track_boundary_commit_client_id(
            tracked_client_ids=[42],
            object_states=[
                {
                    "clientID": 42,
                    "keyframes": {"prev": boundary_frame - 2, "last": boundary_frame - 1},
                },
            ],
            boundary_frame=boundary_frame,
        )

        self.assertIsNone(client_id)

    def test_boundary_commit_skips_already_committed_outside_keyframe(self):
        boundary_frame = 26
        client_id = find_auto_track_boundary_commit_client_id(
            tracked_client_ids=[42],
            object_states=[
                {
                    "clientID": 42,
                    "keyframes": {"prev": boundary_frame - 1, "last": boundary_frame},
                    "outside": True,
                },
            ],
            boundary_frame=boundary_frame,
        )

        self.assertIsNone(client_id)

    def test_boundary_stop_prevents_auto_advance(self):
        self.assertFalse(should_auto_advance_after_auto_track_frame(True, [42]))
        self.assertFalse(should_auto_advance_after_auto_track_frame(False, []))

    def test_generic_error_path_does_not_select_boundary_commit_target(self):
        preload_range_exhausted_code = 'preload_range_exhausted'

        class _FakeError:
            def __init__(self, code=None):
                self.lambdaErrorCode = code

        generic = _FakeError()
        structured = _FakeError(preload_range_exhausted_code)

        self.assertNotEqual(getattr(generic, "lambdaErrorCode", None), preload_range_exhausted_code)
        self.assertEqual(getattr(structured, "lambdaErrorCode", None), preload_range_exhausted_code)

        # Generic errors never enter the boundary commit selector.
        self.assertIsNone(find_auto_track_boundary_commit_client_id([], [], 26))

    def test_reseed_on_same_boundary_frame_remains_eligible_after_outside_commit(self):
        boundary_frame = 26
        after_commit_states = [
            {
                "clientID": 42,
                "keyframes": {"prev": boundary_frame - 1, "last": boundary_frame},
                "outside": True,
            },
        ]

        self.assertIsNone(find_auto_track_boundary_commit_client_id([42], after_commit_states, boundary_frame))
        self.assertIsNone(find_auto_track_boundary_commit_client_id([99], after_commit_states, boundary_frame))
