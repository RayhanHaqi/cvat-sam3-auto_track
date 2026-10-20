// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * Pure helpers for the multi-rally Auto Track workflow.
 *
 * Deliberately dependency free so the filename contract, segment and preflight
 * math can be checked without a test framework:
 *
 *   node -e "import('./multi_rally_segments.ts').then((m) => m.runMultiRallySelfCheck())"
 *
 * The raw-source-frame rule mirrors `SAM3_RAW_SOURCE_FRAME_RE` and
 * `apply_sam3_source_timeline_guard` in `cvat/apps/lambda_manager/views.py`, so the
 * client contract describes exactly the rally boundaries the tracker preload honors.
 * Source frame ids are compared as BigInt: long ids must never lose precision.
 */

/** Mirrors `SAM3_RAW_SOURCE_FRAME_RE` on the server. */
export const SAM3_RAW_SOURCE_FRAME_PATTERN = /frame_(\d+)\.png$/i;

/** Exact trailing source frame number, e.g. 9007199254740993 for `frame_9007199254740993.png`. */
export function parseRawSourceIndexBig(filename: string | null | undefined): bigint | null {
    if (!filename) {
        return null;
    }
    const base = filename.replace(/\\/g, '/').split('/').pop() ?? '';
    const match = SAM3_RAW_SOURCE_FRAME_PATTERN.exec(base);
    return match ? BigInt(match[1]) : null;
}

export type FilenameContractResult =
    | { ok: true; frames: number[]; expectedStartFrames: number[] }
    | { ok: false; frame: number | null; reason: string };

/**
 * Fail-closed filename contract over the whole job:
 * every frame must parse, and a rally starts at the job start plus every frame
 * whose source index is not exactly the previous one + 1.
 */
export function computeFilenameRallyContract(
    frames: number[],
    rawIndexAt: (frame: number) => bigint | null,
): FilenameContractResult {
    if (!frames.length) {
        return { ok: false, frame: null, reason: 'empty_frame_range' };
    }

    const parsed: bigint[] = [];
    for (const frame of frames) {
        const raw = rawIndexAt(frame);
        if (raw === null) {
            return { ok: false, frame, reason: 'filename_unparsable' };
        }
        parsed.push(raw);
    }

    const expectedStartFrames = [frames[0]];
    for (let index = 1; index < frames.length; index++) {
        if (parsed[index] !== parsed[index - 1] + 1n) {
            expectedStartFrames.push(frames[index]);
        }
    }

    return { ok: true, frames, expectedStartFrames };
}

export interface RallySeedRecord {
    clientID: number;
    frame: number;
    points: number[];
    labelID: number;
    trackerID: string | number;
}

/** A discovered seed enriched with the live annotation facts preflight needs. */
export interface RallySeedCandidate extends RallySeedRecord {
    objectType?: string | null;
    shapeType?: string | null;
    outside?: boolean | null;
    lock?: boolean | null;
    keyframeFirst?: number | null;
    keyframeLast?: number | null;
    /** Facts read at seed+1 when the track declares a trailing keyframe there. */
    boundaryFrame?: number | null;
    boundaryOutside?: boolean | null;
    boundaryPoints?: number[] | null;
}

/** Facts read from a track's OWN first keyframe; interpolated facts must never be used. */
export interface RallySeedStateFacts {
    points: number[];
    objectType: string;
    shapeType: string;
    outside: boolean;
    lock: boolean;
    keyframeFirst: number | null;
    keyframeLast: number | null;
    /** Facts read at seed+1 when the track declares a trailing keyframe there. */
    boundaryFrame: number | null;
    boundaryOutside: boolean | null;
    boundaryPoints: number[] | null;
}

/**
 * Builds a candidate from the track's own first-frame facts. `null` facts mean the
 * state could not be read at that frame; the candidate is then deliberately left
 * invalid so preflight fails closed instead of silently dropping the track.
 */
export function buildRallySeedCandidate(
    clientID: number,
    seedFrame: number,
    labelID: number,
    trackerID: string | number,
    ownFirstFrameFacts: RallySeedStateFacts | null,
): RallySeedCandidate {
    if (!ownFirstFrameFacts) {
        return {
            clientID,
            frame: seedFrame,
            points: [],
            labelID,
            trackerID,
            objectType: null,
            shapeType: null,
            outside: null,
            lock: null,
            keyframeFirst: null,
            keyframeLast: null,
            boundaryFrame: null,
            boundaryOutside: null,
            boundaryPoints: null,
        };
    }
    return {
        clientID,
        frame: seedFrame,
        labelID,
        trackerID,
        ...ownFirstFrameFacts,
    };
}

export interface RallySegment {
    index: number;
    /** The seed frame. Never rewritten by the run. */
    anchor: number;
    /** Exclusive end: the next expected rally start, or `stopFrame + 1` when none remains. */
    endExclusive: number;
    isFinal: boolean;
}

export interface RallySegmentWriteFrames {
    /** Keyframes to add after the seed, strictly inside the segment. */
    keyframes: number[];
    /** `outside=true` boundary frame for non-final segments, otherwise null. */
    boundary: number | null;
}

/**
 * Segments are aligned 1:1 with the sorted present seeds. Each segment ends at the
 * next expected filename-derived rally start (or stopFrame + 1 when the seed is the
 * last expected start), never at the next present seed, so rallies whose seed is
 * missing are skipped instead of crossed.
 */
export function buildRallySegments(
    seedFrames: number[],
    expectedStartFrames: number[],
    stopFrame: number,
): RallySegment[] {
    const sorted = [...seedFrames].sort((a, b) => a - b);
    const expected = [...expectedStartFrames].sort((a, b) => a - b);
    return sorted.map((anchor, index) => {
        const expectedIndex = expected.indexOf(anchor);
        const nextExpected = expectedIndex >= 0 && expectedIndex + 1 < expected.length ?
            expected[expectedIndex + 1] : null;
        return {
            index,
            anchor,
            endExclusive: nextExpected === null ? stopFrame + 1 : nextExpected,
            isFinal: nextExpected === null,
        };
    });
}

/**
 * Frames a segment writes: seed+1 .. endExclusive-1, plus an outside boundary at
 * endExclusive for non-final segments. A one-frame rally yields no keyframes
 * (non-final: boundary only; final: nothing to write).
 */
export function rallySegmentWriteFrames(segment: RallySegment, stopFrame: number): RallySegmentWriteFrames {
    const keyframes: number[] = [];
    const lastInterior = Math.min(segment.endExclusive - 1, stopFrame);
    for (let frame = segment.anchor + 1; frame <= lastInterior; frame++) {
        keyframes.push(frame);
    }
    return {
        keyframes,
        boundary: segment.isFinal ? null : segment.endExclusive,
    };
}

export function sameRallyPoints(
    left: number[] | null | undefined,
    right: number[] | null | undefined,
): boolean {
    if (!left || !right || left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index++) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}

/** Loose view of a seed's own-frame facts, satisfied by both candidates and live reads. */
export interface RallySeedShapeFacts {
    frame: number;
    points?: number[] | null;
    objectType?: string | null;
    shapeType?: string | null;
    outside?: boolean | null;
    lock?: boolean | null;
    keyframeFirst?: number | null;
    keyframeLast?: number | null;
    boundaryFrame?: number | null;
    boundaryOutside?: boolean | null;
    boundaryPoints?: number[] | null;
}

export type RallySeedShapeVerdict =
    | { ok: true; trailingBoundary: boolean }
    | { ok: false; reason: 'invalid_seed_object' | 'invalid_seed_boundary' };

/**
 * A seed is accepted in exactly two shapes:
 *  - legacy: one keyframe at the seed frame and nothing after it;
 *  - current: the seed keyframe plus exactly one trailing `outside=true` keyframe at
 *    seed+1 with the same points (what the Collect Seeds commit writes so the
 *    rectangle is only visible on its seed frame).
 * Any later keyframe, or a seed+1 keyframe that is not a verified outside boundary,
 * fails closed. A seed on the last job frame has no room for a boundary.
 */
export function verifyRallySeedShape(
    facts: RallySeedShapeFacts,
    stopFrame: number,
): RallySeedShapeVerdict {
    const baseValid = facts.objectType === 'track' &&
        facts.shapeType === 'rectangle' &&
        facts.outside === false &&
        facts.lock === false &&
        facts.keyframeFirst === facts.frame;
    if (!baseValid) {
        return { ok: false, reason: 'invalid_seed_object' };
    }
    if (facts.keyframeLast === facts.frame) {
        return { ok: true, trailingBoundary: false };
    }
    if (facts.frame + 1 > stopFrame || facts.keyframeLast !== facts.frame + 1) {
        return { ok: false, reason: 'invalid_seed_boundary' };
    }
    const boundaryValid = facts.boundaryFrame === facts.frame + 1 &&
        facts.boundaryOutside === true &&
        sameRallyPoints(facts.boundaryPoints, facts.points);
    return boundaryValid ?
        { ok: true, trailingBoundary: true } :
        { ok: false, reason: 'invalid_seed_boundary' };
}

/** Segment for one anchor, or null when the anchor is off the filename contract. */
export function segmentForAnchor(
    anchor: number,
    expectedStartFrames: number[],
    stopFrame: number,
): RallySegment | null {
    if (!expectedStartFrames.includes(anchor)) {
        return null;
    }
    const [segment] = buildRallySegments([anchor], expectedStartFrames, stopFrame);
    return segment ?? null;
}

/** Terminal keyframe a completed track must end on for its segment. */
export function completedTerminalForSegment(segment: RallySegment, stopFrame: number): number {
    const write = rallySegmentWriteFrames(segment, stopFrame);
    if (write.boundary !== null) {
        return write.boundary;
    }
    if (write.keyframes.length) {
        return stopFrame;
    }
    return segment.anchor;
}

export interface RallyCompletedFrameFact {
    frame: number;
    keyframe: boolean;
    outside: boolean;
}

export type RallyCompletedVerdict =
    | { ok: true }
    | { ok: false; frame: number | null; reason: 'incomplete_completed_track' };

/**
 * Strict frame-structure check for a completed rally track: every interior write
 * frame must be a non-outside keyframe and the non-final boundary must be an
 * outside keyframe. Points are tracker output and are not re-verified here.
 */
export function verifyRallyCompletedFrames(
    segment: RallySegment,
    stopFrame: number,
    facts: RallyCompletedFrameFact[],
): RallyCompletedVerdict {
    const write = rallySegmentWriteFrames(segment, stopFrame);
    const byFrame = new Map(facts.map((fact) => [fact.frame, fact]));
    for (const frame of write.keyframes) {
        const fact = byFrame.get(frame);
        if (!fact || fact.keyframe !== true || fact.outside !== false) {
            return { ok: false, frame, reason: 'incomplete_completed_track' };
        }
    }
    if (write.boundary !== null) {
        const fact = byFrame.get(write.boundary);
        if (!fact || fact.keyframe !== true || fact.outside !== true) {
            return { ok: false, frame: write.boundary, reason: 'incomplete_completed_track' };
        }
    }
    return { ok: true };
}

export type SeedPromptDecision =
    | { action: 'ignore'; committedCount: number }
    | { action: 'commit'; index: number; committedCount: number };

/**
 * Picks the prompt a `canvas.interact` event must commit for Collect Seeds.
 *
 * CVAT reports every `draw_box` drawstop with `finished=false` and the full
 * accumulated prompt list, so the committed count — not `finished` and never
 * `shapes[0]` — is the only reliable marker. A list shorter than the count means
 * the draw was released/cleared, so the newest prompt is a fresh one. An already
 * committed prompt is ignored replay.
 */
export function decideSeedPrompt(shapesLength: number, committedCount: number): SeedPromptDecision {
    if (shapesLength <= 0) {
        // Nothing in the buffer: keep the session marker, the next prompt decides.
        return { action: 'ignore', committedCount };
    }
    const effectiveCount = shapesLength < committedCount ? 0 : committedCount;
    if (shapesLength <= effectiveCount) {
        return { action: 'ignore', committedCount: effectiveCount };
    }
    return { action: 'commit', index: shapesLength - 1, committedCount: shapesLength };
}

export interface RallyPreflightContext {
    stopFrame: number;
    activeLabelID: number;
    activeTrackerID: string | number;
    /** Exact rally-start frames derived from `computeFilenameRallyContract`. */
    expectedStartFrames: number[];
    isFrameDeleted: (frame: number) => boolean;
}

export type RallyPreflightResult =
    | { ok: true; seeds: RallySeedRecord[]; segments: RallySegment[] }
    | { ok: false; frame: number | null; reason: string };

/**
 * Validates discovered seeds against the full filename contract before any tracking.
 * Every present seed must sit exactly on an expected rally start; expected starts with
 * no seed are skipped, and each present seed's segment ends at the next expected start
 * so a skipped rally is never crossed. Fails closed with the exact frame and reason.
 */
export function preflightRallySeeds(
    candidates: RallySeedCandidate[],
    context: RallyPreflightContext,
): RallyPreflightResult {
    const eligible = candidates.filter((candidate) => (
        candidate.labelID === context.activeLabelID &&
        candidate.trackerID === context.activeTrackerID
    ));
    if (!eligible.length) {
        return { ok: false, frame: null, reason: 'no_seeds_for_active_label_tracker' };
    }

    const countsByFrame = new Map<number, number>();
    for (const candidate of eligible) {
        countsByFrame.set(candidate.frame, (countsByFrame.get(candidate.frame) ?? 0) + 1);
    }
    for (const [frame, count] of countsByFrame) {
        if (count > 1) {
            return { ok: false, frame, reason: 'duplicate_seed_frame' };
        }
    }

    const sorted = [...eligible].sort((a, b) => a.frame - b.frame);
    const expectedFrameSet = new Set(context.expectedStartFrames);
    for (const seed of sorted) {
        if (!expectedFrameSet.has(seed.frame)) {
            return { ok: false, frame: seed.frame, reason: 'unexpected_seed_frame' };
        }
    }

    for (const seed of sorted) {
        if (context.isFrameDeleted(seed.frame)) {
            return { ok: false, frame: seed.frame, reason: 'deleted_seed_frame' };
        }
        const verdict = verifyRallySeedShape(seed, context.stopFrame);
        if (!verdict.ok) {
            return { ok: false, frame: seed.frame, reason: verdict.reason };
        }
    }

    const segments = buildRallySegments(
        sorted.map((seed) => seed.frame),
        context.expectedStartFrames,
        context.stopFrame,
    );
    for (const segment of segments) {
        const write = rallySegmentWriteFrames(segment, context.stopFrame);
        const requested = write.boundary === null ?
            write.keyframes : [...write.keyframes, write.boundary];
        for (const frame of requested) {
            if (context.isFrameDeleted(frame)) {
                return { ok: false, frame, reason: 'deleted_frame_in_range' };
            }
        }
    }

    return { ok: true, seeds: sorted, segments };
}

/** Carries the exact frame and reason of a failed/partially committed rally run. */
export class RallyRunStop extends Error {
    public readonly frame: number | null;
    public readonly reason: string;
    /** Highest frame already written, or null when nothing was written. */
    public committedThrough: number | null;

    constructor(frame: number | null, reason: string, committedThrough: number | null = null) {
        super(frame === null ? reason : `frame ${frame}: ${reason}`);
        this.name = 'RallyRunStop';
        this.frame = frame;
        this.reason = reason;
        this.committedThrough = committedThrough;
    }
}

/* istanbul ignore next */
function assertSelfCheck(condition: unknown, message: string): void {
    if (!condition) {
        throw new Error(`multi-rally self-check failed: ${message}`);
    }
}

/* istanbul ignore next */
export function runMultiRallySelfCheck(): void {
    // --- filename ids are parsed exactly, including unsafe/long numbers ---
    assertSelfCheck(parseRawSourceIndexBig('/data/frame_000123.png') === 123n, 'parse raw index');
    assertSelfCheck(parseRawSourceIndexBig('frame_7.PNG') === 7n, 'parse is case-insensitive');
    assertSelfCheck(parseRawSourceIndexBig('clip.mp4') === null, 'non-image name has no index');
    assertSelfCheck(
        parseRawSourceIndexBig('frame_9007199254740993.png') === 9007199254740993n,
        'long id keeps precision',
    );
    assertSelfCheck(
        parseRawSourceIndexBig('frame_9007199254740993.png') !== BigInt('9007199254740992'),
        'long id is not rounded through Number',
    );

    // --- exact expected rally-start set ---
    const names: Record<number, string> = {
        0: 'frame_000001.png',
        1: 'frame_000002.png',
        2: 'frame_000003.png',
        3: 'frame_000001.png',
        4: 'frame_000002.png',
        5: 'frame_000003.png',
        6: 'frame_000004.png',
    };
    const rawIndexAt = (frame: number): bigint | null => parseRawSourceIndexBig(names[frame]);
    const frameRange = Object.keys(names).map((frame) => +frame);
    const contract = computeFilenameRallyContract(frameRange, rawIndexAt);
    assertSelfCheck(contract.ok === true, 'contract accepts a parseable range');
    assertSelfCheck(
        contract.ok === true && contract.expectedStartFrames.join(',') === '0,3',
        'expected rally starts are the job start plus each non-+1 transition',
    );

    const longNames: Record<number, string> = {
        0: 'frame_9007199254740992.png',
        1: 'frame_9007199254740993.png',
        2: 'frame_9007199254740993.png',
    };
    const longContract = computeFilenameRallyContract(
        Object.keys(longNames).map((frame) => +frame),
        (frame) => parseRawSourceIndexBig(longNames[frame]),
    );
    assertSelfCheck(
        longContract.ok === true && longContract.expectedStartFrames.join(',') === '0,2',
        'long ids are compared exactly (reset detected, +1 not a boundary)',
    );

    const unparsable = computeFilenameRallyContract([0, 1], () => null);
    assertSelfCheck(
        unparsable.ok === false && unparsable.reason === 'filename_unparsable' && unparsable.frame === 0,
        'contract fails closed on unparsable names',
    );

    // --- segments, including one-frame rallies ---
    const segments = buildRallySegments([0, 3], [0, 3], 6);
    assertSelfCheck(segments.length === 2, 'segment count');
    assertSelfCheck(segments[0].endExclusive === 3 && segments[0].isFinal === false, 'first boundary');
    assertSelfCheck(segments[1].endExclusive === 7 && segments[1].isFinal === true, 'final to stopFrame+1');
    assertSelfCheck(
        rallySegmentWriteFrames(segments[0], 6).keyframes.join(',') === '1,2',
        'non-final interior keyframes',
    );
    assertSelfCheck(rallySegmentWriteFrames(segments[0], 6).boundary === 3, 'non-final boundary');
    assertSelfCheck(rallySegmentWriteFrames(segments[1], 6).boundary === null, 'final segment has no boundary');

    const oneFrameSegments = buildRallySegments([0, 1], [0, 1], 5);
    const oneFrameWrite = rallySegmentWriteFrames(oneFrameSegments[0], 5);
    assertSelfCheck(oneFrameWrite.keyframes.length === 0, 'one-frame rally has no interior keyframes');
    assertSelfCheck(oneFrameWrite.boundary === 1, 'one-frame non-final rally writes the boundary');
    const trailingFrameSegments = buildRallySegments([5], [5], 5);
    assertSelfCheck(
        rallySegmentWriteFrames(trailingFrameSegments[0], 5).keyframes.length === 0 &&
            rallySegmentWriteFrames(trailingFrameSegments[0], 5).boundary === null,
        'one-frame final rally writes nothing',
    );

    // --- a skipped expected start is never crossed: each present seed ends at the
    // next expected start, not at the next present seed ---
    const sparseSegments = buildRallySegments([0, 6], [0, 3, 6], 6);
    assertSelfCheck(sparseSegments.length === 2, 'sparse segment count');
    assertSelfCheck(
        sparseSegments[0].endExclusive === 3 && sparseSegments[0].isFinal === false,
        'a present seed stops at the next expected start, skipping the missing one',
    );
    assertSelfCheck(
        sparseSegments[1].endExclusive === 7 && sparseSegments[1].isFinal === true,
        'the last present expected start runs to stopFrame+1',
    );

    // --- preflight ---
    const seed = (
        clientID: number, frame: number, overrides: Partial<RallySeedCandidate> = {},
    ): RallySeedCandidate => ({
        clientID,
        frame,
        points: [0, 0, 1, 1],
        labelID: 5,
        trackerID: 't',
        objectType: 'track',
        shapeType: 'rectangle',
        outside: false,
        lock: false,
        keyframeFirst: frame,
        keyframeLast: frame,
        ...overrides,
    });
    const context: RallyPreflightContext = {
        stopFrame: 6,
        activeLabelID: 5,
        activeTrackerID: 't',
        expectedStartFrames: [0, 3],
        isFrameDeleted: () => false,
    };

    assertSelfCheck(
        preflightRallySeeds([seed(1, 0), seed(2, 3)], context).ok === true,
        'accepts exactly one seed per expected rally start',
    );

    // A missing expected start is skipped: the present seed stops at the missing
    // contract boundary, never running into the skipped rally.
    const missingMiddle = preflightRallySeeds([seed(1, 0)], context);
    assertSelfCheck(
        missingMiddle.ok === true && missingMiddle.segments.length === 1 &&
            missingMiddle.segments[0].endExclusive === 3 && missingMiddle.segments[0].isFinal === false,
        'a missing middle start is skipped and the present seed stops at that contract boundary',
    );

    const extra = preflightRallySeeds([seed(1, 0), seed(2, 3), seed(3, 5)], context);
    assertSelfCheck(
        extra.ok === false && extra.reason === 'unexpected_seed_frame' && extra.frame === 5,
        'rejects an off-contract seed',
    );

    const duplicate = preflightRallySeeds([seed(1, 0), seed(2, 0), seed(3, 3)], context);
    assertSelfCheck(
        duplicate.ok === false && duplicate.reason === 'duplicate_seed_frame',
        'rejects duplicate seeds on one frame',
    );

    const offStart = preflightRallySeeds([seed(1, 1), seed(2, 3)], context);
    assertSelfCheck(
        offStart.ok === false && offStart.reason === 'unexpected_seed_frame' && offStart.frame === 1,
        'rejects a present seed off an expected start',
    );

    const locked = preflightRallySeeds([seed(1, 0), seed(2, 3, { lock: true })], context);
    assertSelfCheck(locked.ok === false && locked.reason === 'invalid_seed_object', 'rejects a locked seed');

    const multi = preflightRallySeeds([seed(1, 0), seed(2, 3, { keyframeFirst: 2 })], context);
    assertSelfCheck(multi.ok === false && multi.reason === 'invalid_seed_object', 'rejects a multi-keyframe seed');

    const outside = preflightRallySeeds([seed(1, 0), seed(2, 3, { outside: true })], context);
    assertSelfCheck(outside.ok === false && outside.reason === 'invalid_seed_object', 'rejects an outside seed');

    // --- a seed may carry exactly one verified trailing outside boundary at seed+1 ---
    const trailingBoundarySeed = seed(2, 3, {
        keyframeLast: 4,
        boundaryFrame: 4,
        boundaryOutside: true,
        boundaryPoints: [0, 0, 1, 1],
    });
    assertSelfCheck(
        preflightRallySeeds([seed(1, 0), trailingBoundarySeed], context).ok === true,
        'accepts a seed with a verified trailing outside boundary',
    );
    const trailingVerdict = verifyRallySeedShape(trailingBoundarySeed, 6);
    assertSelfCheck(
        trailingVerdict.ok === true && trailingVerdict.trailingBoundary === true,
        'a verified trailing boundary is reported as such',
    );

    const boundarySeed = (
        boundaryOutside: boolean,
        points: number[],
        keyframeLast = 4,
    ): RallySeedCandidate => seed(2, 3, {
        keyframeLast,
        boundaryFrame: 4,
        boundaryOutside,
        boundaryPoints: points,
    });

    const notOutsideBoundary = preflightRallySeeds(
        [seed(1, 0), boundarySeed(false, [0, 0, 1, 1])],
        context,
    );
    assertSelfCheck(
        notOutsideBoundary.ok === false && notOutsideBoundary.reason === 'invalid_seed_boundary',
        'rejects a seed+1 keyframe that is not outside',
    );

    const boundaryPointsChanged = preflightRallySeeds(
        [seed(1, 0), boundarySeed(true, [0, 0, 2, 2])],
        context,
    );
    assertSelfCheck(
        boundaryPointsChanged.ok === false && boundaryPointsChanged.reason === 'invalid_seed_boundary',
        'rejects a trailing boundary whose points differ from the seed',
    );

    const unverifiedBoundary = preflightRallySeeds(
        [seed(1, 0), seed(2, 3, { keyframeLast: 4 })],
        context,
    );
    assertSelfCheck(
        unverifiedBoundary.ok === false && unverifiedBoundary.reason === 'invalid_seed_boundary',
        'rejects a declared seed+1 keyframe that was not read/verified',
    );

    const laterKeyframe = preflightRallySeeds(
        [seed(1, 0), boundarySeed(true, [0, 0, 1, 1], 5)],
        context,
    );
    assertSelfCheck(
        laterKeyframe.ok === false && laterKeyframe.reason === 'invalid_seed_boundary',
        'rejects a keyframe later than seed+1',
    );

    // A seed on the last job frame has no room for a boundary; the legacy
    // one-keyframe shape stays valid there.
    const stopFrameContext = { ...context, stopFrame: 3 };
    assertSelfCheck(
        preflightRallySeeds([seed(1, 0), seed(2, 3)], stopFrameContext).ok === true,
        'keeps a seed on the last job frame valid as a single keyframe',
    );
    const beyondStopFrame = preflightRallySeeds(
        [seed(1, 0), boundarySeed(true, [0, 0, 1, 1])],
        stopFrameContext,
    );
    assertSelfCheck(
        beyondStopFrame.ok === false && beyondStopFrame.reason === 'invalid_seed_boundary',
        'rejects a trailing keyframe beyond the job stop frame',
    );

    // Missing starts are skipped, including the first one, and each present seed
    // stops at its next expected start.
    const sparseContext = { ...context, expectedStartFrames: [0, 3, 6] };
    const skippedFirst = preflightRallySeeds([seed(1, 3), seed(2, 6)], sparseContext);
    assertSelfCheck(
        skippedFirst.ok === true && skippedFirst.segments.length === 2 &&
            skippedFirst.segments[0].anchor === 3 &&
            skippedFirst.segments[0].endExclusive === 6 && skippedFirst.segments[0].isFinal === false &&
            skippedFirst.segments[1].anchor === 6 &&
            skippedFirst.segments[1].endExclusive === 7 && skippedFirst.segments[1].isFinal === true,
        'skipping the first start succeeds; each present seed stops at the next expected start',
    );

    const nonadjacent = preflightRallySeeds([seed(1, 0), seed(2, 6)], sparseContext);
    assertSelfCheck(
        nonadjacent.ok === true && nonadjacent.segments.length === 2 &&
            nonadjacent.segments[0].endExclusive === 3 && nonadjacent.segments[0].isFinal === false &&
            nonadjacent.segments[1].endExclusive === 7 && nonadjacent.segments[1].isFinal === true,
        'nonadjacent present seeds each stop at their next expected boundary (missing 3 skipped)',
    );

    const deletedSeed = preflightRallySeeds(
        [seed(1, 0), seed(2, 3)],
        { ...context, isFrameDeleted: (frame) => frame === 3 },
    );
    assertSelfCheck(
        deletedSeed.ok === false && deletedSeed.reason === 'deleted_seed_frame',
        'rejects a deleted seed frame',
    );

    const deletedRange = preflightRallySeeds(
        [seed(1, 0), seed(2, 3)],
        { ...context, isFrameDeleted: (frame) => frame === 5 },
    );
    assertSelfCheck(
        deletedRange.ok === false && deletedRange.reason === 'deleted_frame_in_range' && deletedRange.frame === 5,
        'rejects a deleted frame inside a requested range',
    );

    const otherTracker = preflightRallySeeds([seed(1, 0), seed(2, 3)], {
        ...context, activeTrackerID: 'other',
    });
    assertSelfCheck(
        otherTracker.ok === false && otherTracker.reason === 'no_seeds_for_active_label_tracker',
        'ignores seeds from another tracker',
    );

    assertSelfCheck(
        sameRallyPoints([1, 2, 3], [1, 2, 3]) && !sameRallyPoints([1, 2, 3], [1, 2, 4]) &&
            !sameRallyPoints([1, 2], [1, 2, 3]) && !sameRallyPoints(null, [1]),
        'live point comparison is exact',
    );

    // --- seed prompt selection is count-based, never finished/shapes[0] based ---
    const firstPrompt = decideSeedPrompt(1, 0);
    assertSelfCheck(
        firstPrompt.action === 'commit' && firstPrompt.index === 0 && firstPrompt.committedCount === 1,
        'the first prompt commits at index 0',
    );
    const replayedPrompt = decideSeedPrompt(1, 1);
    assertSelfCheck(
        replayedPrompt.action === 'ignore' && replayedPrompt.committedCount === 1,
        'a replayed prompt is ignored',
    );
    const secondPrompt = decideSeedPrompt(2, 1);
    assertSelfCheck(
        secondPrompt.action === 'commit' && secondPrompt.index === 1 && secondPrompt.committedCount === 2,
        'a second prompt commits the newest index',
    );
    const releasedPrompt = decideSeedPrompt(1, 3);
    assertSelfCheck(
        releasedPrompt.action === 'commit' && releasedPrompt.index === 0 && releasedPrompt.committedCount === 1,
        'a released/cleared prompt list commits the fresh newest prompt',
    );
    const emptyPrompt = decideSeedPrompt(0, 3);
    assertSelfCheck(
        emptyPrompt.action === 'ignore' && emptyPrompt.committedCount === 3,
        'an empty prompt list is ignored without dropping the session count',
    );

    // --- candidates come from the track's OWN first frame, never from interpolation ---
    const ownFacts = buildRallySeedCandidate(7, 3, 5, 't', {
        points: [1, 2, 3, 4],
        objectType: 'track',
        shapeType: 'rectangle',
        outside: false,
        lock: false,
        keyframeFirst: 3,
        keyframeLast: 3,
        boundaryFrame: null,
        boundaryOutside: null,
        boundaryPoints: null,
    });
    assertSelfCheck(ownFacts.frame === 3 && ownFacts.points.join(',') === '1,2,3,4', 'own-frame points are kept');
    assertSelfCheck(
        ownFacts.outside === false && ownFacts.keyframeFirst === 3 && ownFacts.keyframeLast === 3,
        'own-frame flags are kept',
    );
    const unreadableOwnFrame = buildRallySeedCandidate(8, 4, 5, 't', null);
    assertSelfCheck(
        unreadableOwnFrame.objectType === null && unreadableOwnFrame.keyframeFirst === null,
        'an unreadable own frame stays invalid',
    );
    assertSelfCheck(
        preflightRallySeeds(
            [seed(1, 0), unreadableOwnFrame],
            { ...context, expectedStartFrames: [0, 4] },
        ).ok === false,
        'an unreadable own frame is rejected by preflight',
    );

    // --- completed tracks end exactly on their filename-derived terminal ---
    const completedSegment = segmentForAnchor(0, [0, 3], 6);
    assertSelfCheck(
        completedSegment !== null && completedSegment.endExclusive === 3,
        'a single anchor maps to its next expected boundary',
    );
    assertSelfCheck(segmentForAnchor(1, [0, 3], 6) === null, 'an off-contract anchor has no segment');
    assertSelfCheck(
        completedSegment !== null && completedTerminalForSegment(completedSegment, 6) === 3,
        'a non-final completed track ends on its boundary',
    );
    const finalSegment = segmentForAnchor(3, [0, 3], 6);
    assertSelfCheck(
        finalSegment !== null && completedTerminalForSegment(finalSegment, 6) === 6,
        'a final completed track ends on stopFrame',
    );
    assertSelfCheck(
        completedSegment !== null &&
            verifyRallyCompletedFrames(completedSegment, 6, [
                { frame: 1, keyframe: true, outside: false },
                { frame: 2, keyframe: true, outside: false },
                { frame: 3, keyframe: true, outside: true },
            ]).ok === true,
        'a fully keyframed segment verifies as completed',
    );
    assertSelfCheck(
        completedSegment !== null &&
            verifyRallyCompletedFrames(completedSegment, 6, [
                { frame: 1, keyframe: true, outside: false },
                { frame: 2, keyframe: false, outside: false },
                { frame: 3, keyframe: true, outside: true },
            ]).ok === false,
        'a completed track with a missing interior keyframe fails closed',
    );

    // eslint-disable-next-line no-console
    console.log('multi-rally self-check passed');
}
