// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import {
    AutoTrackObjectSnapshot,
    buildOutsideKeyframeCommit,
    findAutoTrackBoundaryCommitClientID,
    isAutoTrackSessionAuthoritative,
    toAutoTrackObjectSnapshots,
} from './auto-track-boundary-commit';

export type PreloadRangeExhaustedBoundaryResult =
    | 'committed'
    | 'warn_only'
    | 'stale';

export interface PersistableAutoTrackObjectState {
    outside: boolean;
    keyframe: boolean;
    save: () => Promise<void>;
}

export interface PreloadRangeExhaustedBoundaryDeps {
    boundaryFrame: number;
    sessionId: number;
    trackerGroupClientIDs: ReadonlyArray<number | null | undefined>;
    objectStates: ReadonlyArray<{
        clientID: number | null;
        keyframes?: {
            prev?: number | null;
            last?: number | null;
        } | null;
        outside?: boolean;
    }>;
    getCurrentSessionId: () => number;
    isAutoTrackSessionActive: () => boolean;
    resolveObjectState: (clientID: number) => PersistableAutoTrackObjectState | null;
    warnOnlyStop: () => void;
    onOutsideCommitted: (clientID: number) => Promise<void>;
}

function isSessionAuthoritative(deps: PreloadRangeExhaustedBoundaryDeps): boolean {
    return isAutoTrackSessionAuthoritative(
        deps.sessionId,
        deps.getCurrentSessionId(),
        deps.isAutoTrackSessionActive(),
    );
}

export function resolvePreloadRangeExhaustedCommitClientID(
    trackerGroupClientIDs: ReadonlyArray<number | null | undefined>,
    objectStates: ReadonlyArray<AutoTrackObjectSnapshot>,
    boundaryFrame: number,
): number | null {
    return findAutoTrackBoundaryCommitClientID(
        trackerGroupClientIDs,
        objectStates,
        boundaryFrame,
    );
}

export async function handlePreloadRangeExhaustedBoundaryCommit(
    deps: PreloadRangeExhaustedBoundaryDeps,
): Promise<PreloadRangeExhaustedBoundaryResult> {
    if (!isSessionAuthoritative(deps)) {
        return 'stale';
    }

    const snapshots = toAutoTrackObjectSnapshots(deps.objectStates);
    const commitClientID = resolvePreloadRangeExhaustedCommitClientID(
        deps.trackerGroupClientIDs,
        snapshots,
        deps.boundaryFrame,
    );

    if (commitClientID === null) {
        deps.warnOnlyStop();
        return 'warn_only';
    }

    if (!isSessionAuthoritative(deps)) {
        return 'stale';
    }

    const objectState = deps.resolveObjectState(commitClientID);
    if (!objectState) {
        deps.warnOnlyStop();
        return 'warn_only';
    }

    if (!isSessionAuthoritative(deps)) {
        return 'stale';
    }

    const outsideCommit = buildOutsideKeyframeCommit(deps.boundaryFrame);
    objectState.outside = outsideCommit.outside;
    objectState.keyframe = outsideCommit.keyframe;
    await objectState.save();

    if (!isSessionAuthoritative(deps)) {
        return 'stale';
    }

    await deps.onOutsideCommitted(commitClientID);
    return 'committed';
}
