/** Conservative geometry gates for CVAT Auto Track on ping_pong_ball (2D boxes only). */

import type {
    MinimalShape,
    PreloadSummary,
    PreloadTimelineEntry,
    TrackerResults,
} from 'cvat-core-wrapper';

export const SAM3_V4_TRACKER_ID = 'meta-sam3-tracker-v4';

/** PERF2: enabled unless explicitly disabled at build/runtime. */
export const CVAT_UI_SAM3_G0_TIMELINE_ENABLED = process.env.CVAT_UI_SAM3_G0_TIMELINE !== '0';

export type AutoTrackValidationVerdict = 'valid' | 'soft_invalid' | 'hard_invalid';

export interface AutoTrackBallValidationConfig {
    maxConsecutiveInvalidFrames: number;
    minAreaRatio: number;
    maxAreaRatio: number;
    hardMaxAreaRatio: number;
    maxAspectRatio: number;
    maxAspectRatioDelta: number;
    maxCenterDisplacementScale: number;
    hardMaxCenterDisplacementScale: number;
    maxVelocityResidualScale: number;
}

export const AUTO_TRACK_BALL_VALIDATION_CONFIG: AutoTrackBallValidationConfig = {
    maxConsecutiveInvalidFrames: 1,
    minAreaRatio: 0.35,
    maxAreaRatio: 2.5,
    hardMaxAreaRatio: 4.0,
    maxAspectRatio: 2.8,
    maxAspectRatioDelta: 1.6,
    maxCenterDisplacementScale: 2.5,
    hardMaxCenterDisplacementScale: 5.0,
    maxVelocityResidualScale: 4.0,
};

/**
 * Robust size reference: number of recent valid boxes kept per track, the seed
 * box being the first entry. Gates compare against the median of this window,
 * so a few transiently tiny masks (SAM3 reported 2.9 px next to 7 px ones)
 * cannot collapse the reference the way a last-valid-box reference does: eight
 * entries mean a three-frame excursion cannot move the median, and the seed box
 * only leaves the window once eight boxes were accepted.
 */
export const AUTO_TRACK_BALL_REFERENCE_WINDOW = 8;

export interface AutoTrackBallHistory {
    lastValidPoints: number[] | null;
    previousCenter: [number, number] | null;
    consecutiveInvalidFrames: number;
    /** Recent valid boxes, oldest first, seeded with the seed box. */
    recentValidPoints: number[][];
}

export interface LossTerminationShapeSource {
    autoTrackHistory?: AutoTrackBallHistory;
    shapePoints: number[];
}

export interface LossTerminationObjectStateSource {
    points: number[] | null;
}

export function boxCenter(points: number[]): [number, number] {
    return [(points[0] + points[2]) / 2, (points[1] + points[3]) / 2];
}

export function boxArea(points: number[]): number {
    const w = Math.max(0, points[2] - points[0]);
    const h = Math.max(0, points[3] - points[1]);
    return w * h;
}

export function boxAspectRatio(points: number[]): number {
    const w = Math.max(1, points[2] - points[0]);
    const h = Math.max(1, points[3] - points[1]);
    return Math.max(w, h) / Math.min(w, h);
}

export function boxDiagonal(points: number[]): number {
    const w = Math.max(1, points[2] - points[0]);
    const h = Math.max(1, points[3] - points[1]);
    return Math.hypot(w, h);
}

function centerDistance(a: number[], b: number[]): number {
    const [ax, ay] = boxCenter(a);
    const [bx, by] = boxCenter(b);
    return Math.hypot(ax - bx, ay - by);
}

/** Median of a numeric list; 0 for an empty list. */
export function medianValue(values: number[]): number {
    if (!values.length) {
        return 0;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) {
        return sorted[middle];
    }
    return (sorted[middle - 1] + sorted[middle]) / 2;
}

export interface AutoTrackBallReferenceGeometry {
    area: number;
    diagonal: number;
    aspectRatio: number;
}

/**
 * Median geometry over recent valid boxes: area, diagonal and aspect ratio are
 * each reduced by their own median, so one outlier box cannot skew the size,
 * displacement or shape reference. `null` for an empty list.
 */
export function medianReferenceGeometry(pointsList: number[][]): AutoTrackBallReferenceGeometry | null {
    if (!pointsList.length) {
        return null;
    }
    return {
        area: medianValue(pointsList.map((points) => boxArea(points))),
        diagonal: medianValue(pointsList.map((points) => boxDiagonal(points))),
        aspectRatio: medianValue(pointsList.map((points) => boxAspectRatio(points))),
    };
}

/**
 * Reference window of a track: recent valid boxes with the seed box first.
 * Histories created before this window existed fall back to the last valid box.
 */
export function getAutoTrackReferenceWindow(history: AutoTrackBallHistory): number[][] {
    if (history.recentValidPoints && history.recentValidPoints.length > 0) {
        return history.recentValidPoints;
    }
    return history.lastValidPoints ? [history.lastValidPoints] : [];
}

export function createEmptyAutoTrackHistory(seedPoints: number[]): AutoTrackBallHistory {
    const center = boxCenter(seedPoints);
    return {
        lastValidPoints: [...seedPoints],
        previousCenter: center,
        consecutiveInvalidFrames: 0,
        recentValidPoints: [[...seedPoints]],
    };
}

export function validateAutoTrackCandidate(
    candidatePoints: number[],
    history: AutoTrackBallHistory,
    config: AutoTrackBallValidationConfig = AUTO_TRACK_BALL_VALIDATION_CONFIG,
): AutoTrackValidationVerdict {
    const reference = history.lastValidPoints;
    const geometry = medianReferenceGeometry(getAutoTrackReferenceWindow(history));
    if (!reference || reference.length < 4 || !geometry) {
        return 'valid';
    }

    const refArea = Math.max(1, geometry.area);
    const candArea = boxArea(candidatePoints);
    const areaRatio = candArea / refArea;

    if (areaRatio < config.minAreaRatio / 2 || areaRatio > config.hardMaxAreaRatio) {
        return 'hard_invalid';
    }

    const refDiag = geometry.diagonal;
    const displacement = centerDistance(reference, candidatePoints);
    const displacementScale = displacement / refDiag;

    if (displacementScale > config.hardMaxCenterDisplacementScale) {
        return 'hard_invalid';
    }

    const candAspect = boxAspectRatio(candidatePoints);
    if (candAspect > config.maxAspectRatio * 1.5) {
        return 'hard_invalid';
    }

    let softSignals = 0;
    if (areaRatio < config.minAreaRatio || areaRatio > config.maxAreaRatio) {
        softSignals += 1;
    }
    if (Math.abs(candAspect - geometry.aspectRatio) > config.maxAspectRatioDelta) {
        softSignals += 1;
    }
    if (displacementScale > config.maxCenterDisplacementScale) {
        softSignals += 1;
    }

    if (history.previousCenter) {
        const [cx, cy] = boxCenter(candidatePoints);
        const [px, py] = history.previousCenter;
        const velocity = Math.hypot(cx - px, cy - py);
        const expected = Math.hypot(
            boxCenter(reference)[0] - px,
            boxCenter(reference)[1] - py,
        );
        if (expected > 1 && velocity > expected * config.maxVelocityResidualScale) {
            softSignals += 1;
        }
        if (velocity > refDiag * config.hardMaxCenterDisplacementScale) {
            return 'hard_invalid';
        }
    }

    if (softSignals >= 2) {
        return 'hard_invalid';
    }
    if (softSignals >= 1) {
        return 'soft_invalid';
    }
    return 'valid';
}

export function recordValidAutoTrackHistory(
    history: AutoTrackBallHistory,
    validPoints: number[],
): AutoTrackBallHistory {
    const center = boxCenter(validPoints);
    const window = [...getAutoTrackReferenceWindow(history), [...validPoints]];
    return {
        lastValidPoints: [...validPoints],
        previousCenter: center,
        consecutiveInvalidFrames: 0,
        recentValidPoints: window.slice(-AUTO_TRACK_BALL_REFERENCE_WINDOW),
    };
}

export function recordInvalidAutoTrackHistory(
    history: AutoTrackBallHistory,
): AutoTrackBallHistory {
    return {
        ...history,
        consecutiveInvalidFrames: history.consecutiveInvalidFrames + 1,
    };
}

export function shouldConfirmAutoTrackLoss(
    verdict: AutoTrackValidationVerdict,
    history: AutoTrackBallHistory,
    config: AutoTrackBallValidationConfig = AUTO_TRACK_BALL_VALIDATION_CONFIG,
): boolean {
    if (verdict === 'hard_invalid') {
        return true;
    }
    if (verdict === 'soft_invalid') {
        return history.consecutiveInvalidFrames + 1 > config.maxConsecutiveInvalidFrames;
    }
    return false;
}

/** Finite positive-area rectangle; optional frame bounds when known. */
export function isFinitePositiveBBox(
    points: number[] | null | undefined,
    frameWidth?: number,
    frameHeight?: number,
): points is number[] {
    if (!points || points.length < 4) {
        return false;
    }
    const [x1, y1, x2, y2] = points;
    if (![x1, y1, x2, y2].every((value) => Number.isFinite(value))) {
        return false;
    }
    if (x1 >= x2 || y1 >= y2) {
        return false;
    }
    if (boxArea(points) <= 0) {
        return false;
    }
    if (frameWidth !== undefined && (x1 < 0 || x2 > frameWidth)) {
        return false;
    }
    if (frameHeight !== undefined && (y1 < 0 || y2 > frameHeight)) {
        return false;
    }
    return true;
}

export function isShapeUnusable(
    shape: MinimalShape | null | undefined,
    frameWidth?: number,
    frameHeight?: number,
): boolean {
    if (shape === null || shape === undefined) {
        return true;
    }
    return !isFinitePositiveBBox(shape.points, frameWidth, frameHeight);
}

export function isExplicitSam3Loss(
    trackerId: string,
    response: Pick<TrackerResults, 'lost' | 'tracking_status'>,
    state: { lost?: boolean } | null | undefined,
    shape: MinimalShape | null | undefined,
    frameWidth?: number,
    frameHeight?: number,
): boolean {
    if (trackerId !== SAM3_V4_TRACKER_ID) {
        return false;
    }
    if (response.tracking_status === 'preload_exhausted') {
        return false;
    }
    const explicitLoss = response.lost === true || state?.lost === true;
    if (!explicitLoss) {
        return false;
    }
    return isShapeUnusable(shape, frameWidth, frameHeight);
}

export function getLossTerminationPoints(
    trackedShape: LossTerminationShapeSource,
    objectState: LossTerminationObjectStateSource,
    frameWidth?: number,
    frameHeight?: number,
): number[] | null {
    const fromHistory = trackedShape.autoTrackHistory?.lastValidPoints;
    if (isFinitePositiveBBox(fromHistory, frameWidth, frameHeight)) {
        return [...fromHistory];
    }
    if (isFinitePositiveBBox(trackedShape.shapePoints, frameWidth, frameHeight)) {
        return [...trackedShape.shapePoints];
    }
    if (isFinitePositiveBBox(objectState.points, frameWidth, frameHeight)) {
        return [...objectState.points];
    }
    return null;
}

export function isPreloadTimelineResponse(
    response: TrackerResults,
): response is TrackerResults & { preload_timeline: PreloadTimelineEntry[]; preload_summary: PreloadSummary } {
    return Array.isArray(response.preload_timeline) &&
        response.preload_timeline.length > 0 &&
        typeof response.preload_summary === 'object' &&
        response.preload_summary !== null;
}

export function validatePreloadTimeline(
    response: TrackerResults,
): { valid: true; timeline: PreloadTimelineEntry[]; summary: PreloadSummary } | { valid: false; reason: string } {
    if (!isPreloadTimelineResponse(response)) {
        return { valid: false, reason: 'missing_preload_timeline' };
    }
    const { preload_timeline: timeline, preload_summary: summary } = response;
    if (timeline.length !== summary.preloaded_count) {
        return { valid: false, reason: 'timeline_length_mismatch' };
    }
    for (let i = 0; i < timeline.length; i++) {
        const entry = timeline[i];
        if (entry.relative_frame !== i) {
            return { valid: false, reason: 'relative_frame_gap' };
        }
        if (entry.frame !== summary.base_frame + i) {
            return { valid: false, reason: 'absolute_frame_mismatch' };
        }
        if (entry.bbox_status === 'valid') {
            if (!entry.points || entry.points.length !== 4 || !isFinitePositiveBBox(entry.points)) {
                return { valid: false, reason: 'invalid_valid_entry_points' };
            }
        }
    }
    const expectedUntil = summary.base_frame + summary.preloaded_count - 1;
    if (summary.preloaded_until_frame !== expectedUntil) {
        return { valid: false, reason: 'preloaded_until_mismatch' };
    }
    return { valid: true, timeline, summary };
}

/* istanbul ignore next */
function assertAutoTrackBallSelfCheck(condition: unknown, message: string): void {
    if (!condition) {
        throw new Error(`auto-track ball validation self-check failed: ${message}`);
    }
}

/**
 * The robust-reference contract, checkable without a test framework (same style
 * as `runMultiRallySelfCheck`):
 *
 *   node --experimental-strip-types -e \
 *     "import('./auto_track_ball_validation.ts').then((m) => m.runAutoTrackBallValidationSelfCheck())"
 */
/* istanbul ignore next */
export function runAutoTrackBallValidationSelfCheck(): void {
    const square = (side: number, cx: number, cy: number = 0): number[] => [
        cx - side / 2,
        cy - side / 2,
        cx + side / 2,
        cy + side / 2,
    ];

    // --- medians ---
    assertAutoTrackBallSelfCheck(medianValue([]) === 0, 'median of an empty list is 0');
    assertAutoTrackBallSelfCheck(medianValue([3]) === 3, 'median of a single value');
    assertAutoTrackBallSelfCheck(medianValue([3, 1, 2]) === 2, 'odd-length median');
    assertAutoTrackBallSelfCheck(medianValue([4, 1, 3, 2]) === 2.5, 'even-length median averages the middle pair');
    assertAutoTrackBallSelfCheck(medianReferenceGeometry([]) === null, 'an empty window has no geometry');

    // --- the production rally: tiny transient masks, then recovery ---
    // Seed 6x6, then 7.5, 7.2, 7.8, 5.0, 5.3, 3.7, 2.9, 3.2, 7.6, 8.5, 10.0 px
    // squares moving about 2 px/frame, then 6.2 and 8.5 px/frame.
    const sides = [7.5, 7.2, 7.8, 5.0, 5.3, 3.7, 2.9, 3.2, 7.6, 8.5, 10.0];
    const steps = [2, 2, 2, 2, 2, 2, 2, 2, 6.2, 8.5, 8.5];
    const expected: AutoTrackValidationVerdict[] = [
        'valid', 'valid', 'valid', 'valid', 'valid',
        'soft_invalid', 'soft_invalid', 'soft_invalid', 'valid', 'valid', 'valid',
    ];
    let history = createEmptyAutoTrackHistory(square(6, 0));
    let cursor = 0;
    for (let index = 0; index < sides.length; index += 1) {
        cursor += steps[index];
        const candidate = square(sides[index], cursor);
        const verdict = validateAutoTrackCandidate(candidate, history);
        assertAutoTrackBallSelfCheck(
            verdict === expected[index],
            `production rally frame ${index} (side ${sides[index]} px) is ${verdict}, expected ${expected[index]}`,
        );
        if (verdict === 'valid') {
            history = recordValidAutoTrackHistory(history, candidate);
        } else {
            history = recordInvalidAutoTrackHistory(history);
        }
    }

    // --- a stable median still rejects real jumps ---
    let stable = createEmptyAutoTrackHistory(square(8, 0));
    let stableCursor = 0;
    for (let index = 0; index < AUTO_TRACK_BALL_REFERENCE_WINDOW; index += 1) {
        stableCursor += 2;
        const box = square(8, stableCursor);
        assertAutoTrackBallSelfCheck(
            validateAutoTrackCandidate(box, stable) === 'valid',
            `stable frame ${index} stays valid`,
        );
        stable = recordValidAutoTrackHistory(stable, box);
    }
    const medianDiagonal = boxDiagonal(square(8, 0));
    assertAutoTrackBallSelfCheck(
        validateAutoTrackCandidate(square(medianDiagonal * 5, stableCursor), stable) === 'hard_invalid',
        'a 5x diagonal jump from a stable median is hard-invalid',
    );
    assertAutoTrackBallSelfCheck(
        validateAutoTrackCandidate(square(medianDiagonal / 5 / Math.SQRT2, stableCursor), stable) === 'hard_invalid',
        'a 5x diagonal shrink from a stable median is hard-invalid',
    );
    assertAutoTrackBallSelfCheck(
        validateAutoTrackCandidate(square(8, stableCursor + medianDiagonal * 5 + 1), stable) === 'hard_invalid',
        'a move beyond the hard displacement limit is hard-invalid',
    );
    assertAutoTrackBallSelfCheck(
        validateAutoTrackCandidate(square(8 * Math.SQRT1_2, stableCursor), stable) === 'valid',
        'a half-area box stays valid, inside the soft band',
    );
    assertAutoTrackBallSelfCheck(
        validateAutoTrackCandidate(square(8 * Math.sqrt(0.3), stableCursor), stable) === 'soft_invalid',
        'a 0.3-area box is soft-invalid, not hard-invalid',
    );

    // --- window mechanics ---
    assertAutoTrackBallSelfCheck(
        createEmptyAutoTrackHistory(square(6, 0)).recentValidPoints.length === 1,
        'the reference window starts with the seed box',
    );
    let windowed = createEmptyAutoTrackHistory(square(6, 0));
    for (let index = 0; index < AUTO_TRACK_BALL_REFERENCE_WINDOW + 4; index += 1) {
        windowed = recordValidAutoTrackHistory(windowed, square(6 + index, index * 2));
    }
    assertAutoTrackBallSelfCheck(
        windowed.recentValidPoints.length === AUTO_TRACK_BALL_REFERENCE_WINDOW,
        'the reference window is capped',
    );
    assertAutoTrackBallSelfCheck(
        windowed.recentValidPoints[0][2] - windowed.recentValidPoints[0][0] === 10,
        'the oldest entry leaves the window first',
    );
    const invalidOnly = recordInvalidAutoTrackHistory(windowed);
    assertAutoTrackBallSelfCheck(
        invalidOnly.recentValidPoints === windowed.recentValidPoints && invalidOnly.consecutiveInvalidFrames === 1,
        'an invalid frame never enters the reference window',
    );

    // eslint-disable-next-line no-console
    console.log('auto-track ball validation self-check passed');
}
