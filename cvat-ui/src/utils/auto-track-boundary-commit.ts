// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export interface AutoTrackKeyframeInfo {
    prev?: number;
    last?: number;
}

export interface AutoTrackObjectSnapshot {
    clientID: number;
    keyframes?: AutoTrackKeyframeInfo | null;
    outside?: boolean;
}

export interface OutsideKeyframeCommit {
    frame: number;
    outside: true;
    keyframe: true;
}

export function findAutoTrackBoundaryCommitClientID(
    trackedClientIDs: number[],
    objectStates: AutoTrackObjectSnapshot[],
    boundaryFrame: number,
): number | null {
    for (const clientID of trackedClientIDs) {
        const objectState = objectStates.find((state) => state.clientID === clientID);
        if (!objectState?.keyframes) {
            continue;
        }
        const { prev, last } = objectState.keyframes;
        if (prev !== boundaryFrame - 1) {
            continue;
        }
        if (typeof last === 'number' && last >= boundaryFrame) {
            continue;
        }
        return clientID;
    }
    return null;
}

export function buildOutsideKeyframeCommit(boundaryFrame: number): OutsideKeyframeCommit {
    return {
        frame: boundaryFrame,
        outside: true,
        keyframe: true,
    };
}

export function shouldAutoAdvanceAfterAutoTrackFrame(
    autoTrackSessionActive: boolean,
    lostClientIDs: number[],
): boolean {
    return autoTrackSessionActive && lostClientIDs.length === 0;
}

export function isGenericLambdaTrackingError(error: unknown): boolean {
    return !((error as { lambdaErrorCode?: string } | null)?.lambdaErrorCode === 'preload_range_exhausted');
}
