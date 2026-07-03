// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
    SAM3_TRACKER_ID,
    buildTrackerSeedInteractionSettings,
    isSam3PreloadExhaustedTerminalResponse,
    resolveSam3AutoTrackTerminalAction,
} from '../../cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/sam3_auto_track_terminal.ts';

const TERMINAL_RESPONSE = {
    tracking_status: 'preload_exhausted',
    tracking_stop_reason: 'tracking_window_complete',
    shapes: [],
    states: [],
} as const;

describe('sam3_auto_track_terminal', () => {
    it('detects SAM3 preload_exhausted terminal responses', () => {
        assert.equal(
            isSam3PreloadExhaustedTerminalResponse(SAM3_TRACKER_ID, TERMINAL_RESPONSE),
            true,
        );
        assert.equal(
            resolveSam3AutoTrackTerminalAction(SAM3_TRACKER_ID, TERMINAL_RESPONSE).kind,
            'terminal_complete',
        );
    });

    it('routes generic missing-box responses to shape processing', () => {
        const genericResponse = { tracking_status: undefined, shapes: [] };
        assert.equal(
            resolveSam3AutoTrackTerminalAction(SAM3_TRACKER_ID, genericResponse).kind,
            'process_shapes',
        );
        assert.equal(
            resolveSam3AutoTrackTerminalAction('other-tracker', TERMINAL_RESPONSE).kind,
            'process_shapes',
        );
    });

    it('enables transparent seed fill only for meta-sam3-tracker-v4', () => {
        assert.deepEqual(buildTrackerSeedInteractionSettings(SAM3_TRACKER_ID), {
            crosshair: true,
            transparentBoxFill: true,
        });
        assert.deepEqual(buildTrackerSeedInteractionSettings('other-tracker'), {
            crosshair: true,
        });
    });
});
