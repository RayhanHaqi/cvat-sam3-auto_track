// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
    clearTransparentBoxFillOnRelease,
    interactionRectangleOpacity,
    mergeInteractionSettings,
    usesTransparentInteractionBoxFill,
} from '../../cvat-canvas/src/typescript/interaction_box_style.ts';

const DEFAULT_SETTINGS = {
    crosshair: false,
    points_type: 'any' as const,
    removalStrategy: 'any' as const,
    appendCursorPositionAsPoint: false,
    transparentBoxFill: false,
};

describe('interaction_box_style', () => {
    it('uses default opacity when transparentBoxFill is absent', () => {
        assert.equal(interactionRectangleOpacity(undefined, 0.5), 0.5);
        assert.equal(usesTransparentInteractionBoxFill(undefined), false);
    });

    it('uses transparent fill mode only when transparentBoxFill is true', () => {
        assert.equal(interactionRectangleOpacity(true, 0.5), null);
        assert.equal(usesTransparentInteractionBoxFill(true), true);
        assert.equal(usesTransparentInteractionBoxFill(false), false);
    });

    it('enables transparent fill only when interact payload sets transparentBoxFill true', () => {
        const sam3 = mergeInteractionSettings(DEFAULT_SETTINGS, {
            crosshair: true,
            transparentBoxFill: true,
        });
        assert.equal(sam3.transparentBoxFill, true);
        assert.equal(interactionRectangleOpacity(sam3.transparentBoxFill, 0.5), null);
    });

    it('restores default fill after generic draw_box without the flag', () => {
        const afterSam3 = mergeInteractionSettings(DEFAULT_SETTINGS, {
            crosshair: true,
            transparentBoxFill: true,
        });
        const generic = mergeInteractionSettings(afterSam3, { crosshair: true });
        assert.equal(generic.transparentBoxFill, false);
        assert.equal(interactionRectangleOpacity(generic.transparentBoxFill, 0.5), 0.5);
    });

    it('honors explicit transparentBoxFill false in the payload', () => {
        const cleared = mergeInteractionSettings(
            { ...DEFAULT_SETTINGS, transparentBoxFill: true },
            { transparentBoxFill: false },
        );
        assert.equal(cleared.transparentBoxFill, false);
    });

    it('clears transparentBoxFill on release', () => {
        const released = clearTransparentBoxFillOnRelease({
            ...DEFAULT_SETTINGS,
            transparentBoxFill: true,
        });
        assert.equal(released.transparentBoxFill, false);
    });
});
