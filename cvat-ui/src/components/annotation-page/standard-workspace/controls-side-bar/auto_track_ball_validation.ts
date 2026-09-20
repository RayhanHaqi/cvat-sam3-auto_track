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

export interface AutoTrackBallHistory {
    lastValidPoints: number[] | null;
    previousCenter: [number, number] | null;
    consecutiveInvalidFrames: number;
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

export function createEmptyAutoTrackHistory(seedPoints: number[]): AutoTrackBallHistory {
    const center = boxCenter(seedPoints);
    return {
        lastValidPoints: [...seedPoints],
        previousCenter: center,
        consecutiveInvalidFrames: 0,
    };
}

export function validateAutoTrackCandidate(
    candidatePoints: number[],
    history: AutoTrackBallHistory,
    config: AutoTrackBallValidationConfig = AUTO_TRACK_BALL_VALIDATION_CONFIG,
): AutoTrackValidationVerdict {
    const reference = history.lastValidPoints;
    if (!reference || reference.length < 4) {
        return 'valid';
    }

    const refArea = Math.max(1, boxArea(reference));
    const candArea = boxArea(candidatePoints);
    const areaRatio = candArea / refArea;

    if (areaRatio < config.minAreaRatio / 2 || areaRatio > config.hardMaxAreaRatio) {
        return 'hard_invalid';
    }

    const refDiag = boxDiagonal(reference);
    const displacement = centerDistance(reference, candidatePoints);
    const displacementScale = displacement / refDiag;

    if (displacementScale > config.hardMaxCenterDisplacementScale) {
        return 'hard_invalid';
    }

    const refAspect = boxAspectRatio(reference);
    const candAspect = boxAspectRatio(candidatePoints);
    if (candAspect > config.maxAspectRatio * 1.5) {
        return 'hard_invalid';
    }

    let softSignals = 0;
    if (areaRatio < config.minAreaRatio || areaRatio > config.maxAreaRatio) {
        softSignals += 1;
    }
    if (Math.abs(candAspect - refAspect) > config.maxAspectRatioDelta) {
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
    return {
        lastValidPoints: [...validPoints],
        previousCenter: center,
        consecutiveInvalidFrames: 0,
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
