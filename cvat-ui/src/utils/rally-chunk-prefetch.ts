// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { FramesMetaData, Job } from 'cvat-core-wrapper';
import {
    computeFilenameRallyContract,
    parseRawSourceIndexBig,
} from 'components/annotation-page/standard-workspace/controls-side-bar/multi_rally_segments';

/** How many upcoming rally starts are prepared by the background chunk prefetch. */
export const RALLY_START_PREFETCH_COUNT = 3;

interface RallyStartCacheEntry {
    frameNumbers: number[];
    // null when the job frame names do not satisfy the rally filename contract
    rallyStartFrames: number[] | null;
}

// Job frames are stable while a job is open, so the contract is computed once per
// frames meta data object and reused on every frame change.
const rallyStartsCache = new WeakMap<FramesMetaData, RallyStartCacheEntry>();

function computeRallyStartFrames(meta: FramesMetaData, frameNumbers: number[]): number[] | null {
    const cached = rallyStartsCache.get(meta);
    if (cached && cached.frameNumbers === frameNumbers) {
        return cached.rallyStartFrames;
    }

    // meta.frames is ordered exactly like the job frame numbers
    const nameByFrame = new Map<number, string | undefined>(
        frameNumbers.map((frame, index) => [frame, meta.frames[index]?.name]),
    );
    const contract = computeFilenameRallyContract(
        frameNumbers,
        (frame) => parseRawSourceIndexBig(nameByFrame.get(frame)),
    );
    const rallyStartFrames = contract.ok ? contract.expectedStartFrames : null;
    rallyStartsCache.set(meta, { frameNumbers, rallyStartFrames });
    return rallyStartFrames;
}

/**
 * The frames where a rally starts strictly after `currentFrame`, up to `limit` of them.
 * Returns an empty array when the job frame names do not satisfy the rally filename
 * contract, so jobs without a rally structure are not affected.
 */
export function nextRallyStartFrames(
    meta: FramesMetaData | null,
    frameNumbers: number[],
    currentFrame: number,
    limit = RALLY_START_PREFETCH_COUNT,
): number[] {
    if (!meta || !frameNumbers.length || !Number.isInteger(currentFrame)) {
        return [];
    }

    const rallyStartFrames = computeRallyStartFrames(meta, frameNumbers);
    if (!rallyStartFrames) {
        return [];
    }

    return rallyStartFrames
        .filter((frame) => frame > currentFrame)
        .slice(0, limit);
}

/**
 * Asks cvat-core to download and decode chunks which hold the upcoming rally starts in
 * the background. Best effort: it never blocks the player and never fails the page.
 */
export function prefetchNextRallyChunks(
    job: Job | null,
    meta: FramesMetaData | null,
    frameNumbers: number[],
    currentFrame: number,
): void {
    if (!job) {
        return;
    }

    const rallyStartFrames = nextRallyStartFrames(meta, frameNumbers, currentFrame);
    if (!rallyStartFrames.length) {
        return;
    }

    job.frames.prefetchChunksForFrames(rallyStartFrames, currentFrame).catch(() => {
        // the background prefetch must never affect the annotation page
    });
}
