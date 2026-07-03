// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/** Whether the interaction rectangle should keep a visible stroke but no fill overlay. */
export function usesTransparentInteractionBoxFill(transparentBoxFill?: boolean): boolean {
    return transparentBoxFill === true;
}

/**
 * Selected-shape opacity for interaction rectangles.
 * Returns null when the fill must stay transparent (stroke/handles use full opacity).
 */
export function interactionRectangleOpacity(
    transparentBoxFill: boolean | undefined,
    selectedShapeOpacity: number,
): number | null {
    if (usesTransparentInteractionBoxFill(transparentBoxFill)) {
        return null;
    }
    return selectedShapeOpacity;
}

export const TRANSPARENT_INTERACTION_RECT_FILL_STYLE = 'fill:none!important;fill-opacity:0!important;';

export interface InteractionSettingsSlice {
    crosshair?: boolean;
    points_type?: 'any' | 'positive' | 'negative';
    removalStrategy?: 'any' | 'last';
    appendCursorPositionAsPoint?: boolean;
    transparentBoxFill?: boolean;
}

/** Merge interaction settings; transparentBoxFill is opt-in per interact() call only. */
export function mergeInteractionSettings<T extends InteractionSettingsSlice>(
    previous: T,
    incoming: InteractionSettingsSlice | undefined,
): T {
    const next = incoming ?? {};
    return {
        ...previous,
        ...next,
        transparentBoxFill: next.transparentBoxFill === true,
    };
}

export function clearTransparentBoxFillOnRelease<T extends InteractionSettingsSlice>(
    settings: T,
): T {
    return {
        ...settings,
        transparentBoxFill: false,
    };
}
