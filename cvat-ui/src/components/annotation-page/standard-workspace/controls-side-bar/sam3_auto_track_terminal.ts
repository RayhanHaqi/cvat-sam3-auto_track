// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export const SAM3_TRACKER_ID = 'meta-sam3-tracker-v4';

export interface Sam3TrackerTerminalFields {
    tracking_status?: string;
    tracking_stop_reason?: string;
    shapes?: unknown[];
    states?: unknown[];
}

export type Sam3TerminalHandlingAction =
    | { kind: 'terminal_complete'; stopReason: string }
    | { kind: 'process_shapes' };

export function isSam3TrackerId(trackerId: string | number | undefined): boolean {
    return String(trackerId) === SAM3_TRACKER_ID;
}

export function isSam3PreloadExhaustedTerminalResponse(
    trackerId: string | number,
    response: Sam3TrackerTerminalFields,
): boolean {
    return isSam3TrackerId(trackerId) && response.tracking_status === 'preload_exhausted';
}

export function resolveSam3AutoTrackTerminalAction(
    trackerId: string | number,
    response: Sam3TrackerTerminalFields,
): Sam3TerminalHandlingAction {
    if (isSam3PreloadExhaustedTerminalResponse(trackerId, response)) {
        return {
            kind: 'terminal_complete',
            stopReason: response.tracking_stop_reason || 'preload_exhausted',
        };
    }
    return { kind: 'process_shapes' };
}

export function buildTrackerSeedInteractionSettings(trackerId: string | number): {
    crosshair: boolean;
    transparentBoxFill?: boolean;
} {
    return {
        crosshair: true,
        ...(isSam3TrackerId(trackerId) ? { transparentBoxFill: true } : {}),
    };
}
