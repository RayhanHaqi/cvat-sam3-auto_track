/** Conservative geometry gates for CVAT Auto Track on ping_pong_ball (2D boxes only). */

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
