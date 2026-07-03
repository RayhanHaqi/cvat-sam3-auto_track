// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
    SAM3_TRACKER_ID,
    resolveSam3AutoTrackTerminalAction,
} from '../../cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/sam3_auto_track_terminal.ts';

describe('tools-control terminal guard contract', () => {
    it('terminal branch short-circuits before empty-shape loss handling', () => {
        const response = {
            tracking_status: 'preload_exhausted',
            tracking_stop_reason: 'tracking_window_complete',
            shapes: [] as unknown[],
        };

        const terminalAction = resolveSam3AutoTrackTerminalAction(SAM3_TRACKER_ID, response);
        assert.equal(terminalAction.kind, 'terminal_complete');
        assert.equal(response.shapes.length, 0);
        const wouldReachLossPath = terminalAction.kind === 'process_shapes' && response.shapes[0] === undefined;
        assert.equal(wouldReachLossPath, false);
    });
});
