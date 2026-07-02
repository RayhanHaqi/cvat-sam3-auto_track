// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export interface AutoTrackKeyframeInfo {
    prev?: number | null;
    last?: number | null;
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

export function filterAutoTrackClientIDs(clientIDs: ReadonlyArray<number | null | undefined>): number[] {
    return clientIDs.filter((clientID): clientID is number => (
        typeof clientID === 'number' && Number.isFinite(clientID)
    ));
}

export function toAutoTrackObjectSnapshots(
    objectStates: ReadonlyArray<{
        clientID: number | null;
        keyframes?: AutoTrackKeyframeInfo | null;
        outside?: boolean;
    }>,
): AutoTrackObjectSnapshot[] {
    return objectStates.flatMap((objectState) => {
        if (typeof objectState.clientID !== 'number' || !Number.isFinite(objectState.clientID)) {
            return [];
        }
        return [{
            clientID: objectState.clientID,
            keyframes: objectState.keyframes,
            outside: objectState.outside,
        }];
    });
}

export function findAutoTrackBoundaryCommitClientID(
    trackerGroupClientIDs: ReadonlyArray<number | null | undefined>,
    objectStates: ReadonlyArray<AutoTrackObjectSnapshot>,
    boundaryFrame: number,
): number | null {
    const normalizedGroupClientIDs = filterAutoTrackClientIDs(trackerGroupClientIDs);
    for (const clientID of normalizedGroupClientIDs) {
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
        if (objectState.outside === true && typeof last === 'number' && last === boundaryFrame) {
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

export function isAutoTrackSessionAuthoritative(
    sessionId: number,
    currentSessionId: number,
    autoTrackSessionActive: boolean,
): boolean {
    return autoTrackSessionActive && sessionId === currentSessionId;
}

export function shouldCommitShapeForInactiveTrackingBranch(
    sessionId: number,
    currentSessionId: number,
    autoTrackSessionActive: boolean,
): boolean {
    return !autoTrackSessionActive && sessionId === currentSessionId;
}
