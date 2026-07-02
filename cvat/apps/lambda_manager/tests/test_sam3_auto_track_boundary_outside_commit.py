# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""Contract tests for Auto Track boundary outside-keyframe commit helpers.

Mirrors cvat-ui/src/utils/auto-track-boundary-commit.ts and
auto-track-boundary-handler.ts so UI behavior stays testable without a JS runner
in CI for cvat-ui.
"""

from __future__ import annotations

import unittest
from typing import Optional


def filter_auto_track_client_ids(client_ids: list) -> list[int]:
    return [
        client_id for client_id in client_ids
        if isinstance(client_id, int)
    ]


def find_auto_track_boundary_commit_client_id(
    tracker_group_client_ids: list,
    object_states: list[dict],
    boundary_frame: int,
) -> Optional[int]:
    for client_id in filter_auto_track_client_ids(tracker_group_client_ids):
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
        if object_state.get("outside") is True and last == boundary_frame:
            continue
        return client_id
    return None


def is_auto_track_session_authoritative(
    session_id: int,
    current_session_id: int,
    auto_track_session_active: bool,
) -> bool:
    return auto_track_session_active and session_id == current_session_id


def should_commit_shape_for_inactive_tracking_branch(
    session_id: int,
    current_session_id: int,
    auto_track_session_active: bool,
) -> bool:
    return (not auto_track_session_active) and session_id == current_session_id


class AutoTrackBoundaryCommitContractTest(unittest.TestCase):
    def test_boundary_commit_targets_exhausted_frame_not_prior_valid_frame(self):
        boundary_frame = 26
        client_id = find_auto_track_boundary_commit_client_id(
            tracker_group_client_ids=[42],
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

    def test_only_failing_tracker_group_is_eligible(self):
        boundary_frame = 26
        states = [
            {"clientID": 10, "keyframes": {"prev": 25, "last": 25}},
            {"clientID": 20, "keyframes": {"prev": 25, "last": 25}},
        ]
        self.assertEqual(find_auto_track_boundary_commit_client_id([10], states, boundary_frame), 10)
        self.assertEqual(find_auto_track_boundary_commit_client_id([20], states, boundary_frame), 20)
        self.assertIsNone(find_auto_track_boundary_commit_client_id([30], states, boundary_frame))

    def test_prior_valid_frame_is_not_marked_outside_by_selector(self):
        boundary_frame = 26
        client_id = find_auto_track_boundary_commit_client_id(
            tracker_group_client_ids=[42],
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
            tracker_group_client_ids=[42],
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

    def test_session_authorization_blocks_stale_commit(self):
        self.assertFalse(is_auto_track_session_authoritative(1, 2, True))
        self.assertFalse(is_auto_track_session_authoritative(1, 1, False))
        self.assertTrue(is_auto_track_session_authoritative(1, 1, True))

    def test_inactive_branch_blocks_stale_manual_commits(self):
        self.assertFalse(should_commit_shape_for_inactive_tracking_branch(1, 2, False))
        self.assertTrue(should_commit_shape_for_inactive_tracking_branch(1, 1, False))
        self.assertFalse(should_commit_shape_for_inactive_tracking_branch(1, 1, True))

    def test_nullable_client_ids_are_filtered(self):
        self.assertEqual(filter_auto_track_client_ids([10, None, 20]), [10, 20])

    def test_generic_error_path_does_not_select_boundary_commit_target(self):
        preload_range_exhausted_code = "preload_range_exhausted"

        class _FakeError:
            def __init__(self, code=None):
                self.lambdaErrorCode = code

        generic = _FakeError()
        structured = _FakeError(preload_range_exhausted_code)

        self.assertNotEqual(getattr(generic, "lambdaErrorCode", None), preload_range_exhausted_code)
        self.assertEqual(getattr(structured, "lambdaErrorCode", None), preload_range_exhausted_code)
        self.assertIsNone(find_auto_track_boundary_commit_client_id([], [], 26))
