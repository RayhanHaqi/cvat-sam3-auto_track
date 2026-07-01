/**
 * Structured diagnostics for CVAT SAM3 Auto Track (ping_pong_ball).
 * Disabled by default. No tracking behavior changes.
 */

export type AutoTrackDiagEventName =
    | 'session_start'
    | 'seed_committed'
    | 'frame_start'
    | 'lambda_request_start'
    | 'lambda_request_end'
    | 'sam_inference_start'
    | 'sam_inference_end'
    | 'candidate_received'
    | 'candidate_validated'
    | 'shape_commit_start'
    | 'shape_commit_end'
    | 'annotations_fetch_start'
    | 'annotations_fetch_end'
    | 'reducer_or_store_update_end'
    | 'canvas_setup_start'
    | 'canvas_setup_end'
    | 'frame_end'
    | 'session_stop'
    | 'session_cancel'
    | 'session_lost'
    | 'session_complete'
    | 'session_error'
    | 'stale_result_dropped';

export type AutoTrackDiagTrendLabel = 'stable' | 'increasing' | 'decreasing' | 'insufficient_data';

export interface BoxGeometrySummary {
    xtl: number | null;
    ytl: number | null;
    xbr: number | null;
    ybr: number | null;
    width: number | null;
    height: number | null;
    centerX: number | null;
    centerY: number | null;
}

export interface AutoTrackFrameRecord {
    sessionId: number;
    jobFrameIndex: number;
    requestId: string | null;
    sourceFrameIdentifier: string | null;
    sourceFrameDelta: number | null;

    activeTrackedObjectCount: number | null;
    committedShapeCountForCurrentTrack: number | null;
    currentTrackKeyframeCount: number | null;
    currentTrackFrameSpan: number | null;
    totalAnnotationStateCount: number | null;
    totalVisibleCanvasShapeCount: number | null;
    canvasObjectCount: number | null;
    visibleShapeCount: number | null;

    frameAcquireMs: number | null;
    lambdaRoundTripMs: number | null;
    samInferenceMs: number | null;
    samPreprocessMs: number | null;
    samPostprocessMs: number | null;
    candidateExtractionMs: number | null;
    geometryConversionMs: number | null;
    validationMs: number | null;
    shapeSaveMs: number | null;
    annotationsFetchMs: number | null;
    storeUpdateMs: number | null;
    canvasSetupMs: number | null;
    totalFrameMs: number | null;

    pendingPromiseCount: number | null;
    pendingQueueLength: number | null;
    activeRequestCount: number | null;

    samSessionCount: number | null;
    samMemoryEntryCount: number | null;
    samTrackedObjectCount: number | null;

    candidateType: string | null;
    candidateXtl: number | null;
    candidateYtl: number | null;
    candidateXbr: number | null;
    candidateYbr: number | null;
    candidateWidth: number | null;
    candidateHeight: number | null;
    candidateCenterX: number | null;
    candidateCenterY: number | null;
    candidateConfidence: number | null;

    committedXtl: number | null;
    committedYtl: number | null;
    committedXbr: number | null;
    committedYbr: number | null;
    committedWidth: number | null;
    committedHeight: number | null;
    committedCenterX: number | null;
    committedCenterY: number | null;

    candidateAccepted: boolean | null;
    validationResult: string | null;
    validationReason: string | null;
    stopReason: string | null;
    cancelled: boolean | null;

    requestPayloadBytes: number | null;
    responsePayloadBytes: number | null;
    annotationStateCount: number | null;
    fullStateReplacement: boolean | null;
}

export interface AutoTrackCorrelationSlice {
    totalFrameMs: number | null;
    lambdaRoundTripMs: number | null;
    annotationsFetchMs: number | null;
    canvasSetupMs: number | null;
    annotationStateCount: number | null;
    currentTrackKeyframeCount: number | null;
    currentTrackFrameSpan: number | null;
}

function nowMs(): number {
    return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function isAutoTrackDiagnosticsEnabled(): boolean {
    try {
        if (typeof localStorage !== 'undefined') {
            const ls = localStorage.getItem('AUTO_TRACK_DIAGNOSTICS');
            if (ls !== null && ['1', 'true', 'yes'].includes(ls.toLowerCase())) {
                return true;
            }
        }
    } catch {
        // ignore
    }
    if (typeof process !== 'undefined' && process.env?.AUTO_TRACK_DIAGNOSTICS) {
        const v = String(process.env.AUTO_TRACK_DIAGNOSTICS).toLowerCase();
        return ['1', 'true', 'yes'].includes(v);
    }
    return false;
}

export function boxGeometryFromPoints(points: number[] | null | undefined): BoxGeometrySummary {
    if (!points || points.length < 4) {
        return {
            xtl: null, ytl: null, xbr: null, ybr: null,
            width: null, height: null, centerX: null, centerY: null,
        };
    }
    const xtl = points[0];
    const ytl = points[1];
    const xbr = points[2];
    const ybr = points[3];
    const width = xbr - xtl;
    const height = ybr - ytl;
    return {
        xtl, ytl, xbr, ybr, width, height,
        centerX: (xtl + xbr) / 2,
        centerY: (ytl + ybr) / 2,
    };
}

function percentile(values: number[], p: number): number | null {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
    return sorted[idx];
}

function avg(values: number[]): number | null {
    if (!values.length) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
}

function sliceThirds<T>(values: T[]): { early: T[]; middle: T[]; late: T[] } {
    if (!values.length) {
        return { early: [], middle: [], late: [] };
    }
    const n = values.length;
    const third = Math.max(1, Math.floor(n / 3));
    return {
        early: values.slice(0, third),
        middle: values.slice(third, third * 2),
        late: values.slice(third * 2),
    };
}

function avgNumericField(
    records: AutoTrackFrameRecord[],
    field: keyof AutoTrackFrameRecord,
): number | null {
    const values = records
        .map((record) => record[field])
        .filter((value): value is number => typeof value === 'number');
    return avg(values);
}

function buildCorrelationSlice(records: AutoTrackFrameRecord[]): AutoTrackCorrelationSlice {
    return {
        totalFrameMs: avgNumericField(records, 'totalFrameMs'),
        lambdaRoundTripMs: avgNumericField(records, 'lambdaRoundTripMs'),
        annotationsFetchMs: avgNumericField(records, 'annotationsFetchMs'),
        canvasSetupMs: avgNumericField(records, 'canvasSetupMs'),
        annotationStateCount: avgNumericField(records, 'annotationStateCount'),
        currentTrackKeyframeCount: avgNumericField(records, 'currentTrackKeyframeCount'),
        currentTrackFrameSpan: avgNumericField(records, 'currentTrackFrameSpan'),
    };
}

function trendLabel(values: Array<number | null | undefined>): AutoTrackDiagTrendLabel {
    const numeric = values.filter((value): value is number => typeof value === 'number');
    if (numeric.length < 3) {
        return 'insufficient_data';
    }
    const { early, late } = sliceThirds(numeric);
    const earlyAvg = avg(early);
    const lateAvg = avg(late);
    if (earlyAvg === null || lateAvg === null) {
        return 'insufficient_data';
    }
    const delta = lateAvg - earlyAvg;
    const tolerance = Math.max(Math.abs(earlyAvg) * 0.1, 1);
    if (Math.abs(delta) <= tolerance) {
        return 'stable';
    }
    return delta > 0 ? 'increasing' : 'decreasing';
}

export class AutoTrackDiagnosticsController {
    private sessionId: number | null = null;

    private frameStartMs: number | null = null;

    private currentFrame: Partial<AutoTrackFrameRecord> | null = null;

    private frameRecords: AutoTrackFrameRecord[] = [];

    private activeLambdaRequests = 0;

    private maxActiveLambdaRequests = 0;

    private acceptedFrameCount = 0;

    private softInvalidCount = 0;

    private hardInvalidCount = 0;

    private lostCount = 0;

    private cancelledCount = 0;

    private maxActiveTrackedObjectCount = 0;

    private maxCommittedShapeCountForCurrentTrack = 0;

    private maxTotalAnnotationStateCount = 0;

    private maxTotalVisibleCanvasShapeCount = 0;

    private maxCurrentTrackFrameSpan = 0;

    private maxPendingPromiseCount = 0;

    private maxSamMemoryEntryCount = 0;

    public isEnabled(): boolean {
        return isAutoTrackDiagnosticsEnabled();
    }

    public getSessionId(): number | null {
        return this.sessionId;
    }

    public getActiveLambdaRequestCount(): number {
        return this.activeLambdaRequests;
    }

    public beginSession(sessionId: number): void {
        if (!this.isEnabled()) return;
        this.sessionId = sessionId;
        this.frameRecords = [];
        this.acceptedFrameCount = 0;
        this.softInvalidCount = 0;
        this.hardInvalidCount = 0;
        this.lostCount = 0;
        this.cancelledCount = 0;
        this.maxActiveTrackedObjectCount = 0;
        this.maxCommittedShapeCountForCurrentTrack = 0;
        this.maxTotalAnnotationStateCount = 0;
        this.maxTotalVisibleCanvasShapeCount = 0;
        this.maxCurrentTrackFrameSpan = 0;
        this.maxActiveLambdaRequests = 0;
        this.maxPendingPromiseCount = 0;
        this.maxSamMemoryEntryCount = 0;
        this.emit('session_start', { sessionId });
    }

    public endSession(event: 'session_stop' | 'session_cancel' | 'session_lost' | 'session_complete' | 'session_error', reason?: string): void {
        if (!this.isEnabled() || this.sessionId === null) return;
        if (event === 'session_cancel') {
            this.cancelledCount += 1;
        }
        this.emit(event, { sessionId: this.sessionId, reason: reason ?? null });
        this.printSessionSummary();
        this.sessionId = null;
        this.currentFrame = null;
        this.frameStartMs = null;
    }

    public emit(eventName: AutoTrackDiagEventName, fields: Record<string, unknown> = {}): void {
        if (!this.isEnabled()) return;
        const record = {
            event: eventName,
            timestamp: new Date().toISOString(),
            monotonicMs: nowMs(),
            sessionId: this.sessionId,
            ...fields,
        };
        // eslint-disable-next-line no-console
        console.info('[AUTO_TRACK_DIAG]', JSON.stringify(record));
    }

    public beginFrame(meta: {
        jobFrameIndex: number;
        requestId?: string | null;
        sourceFrameIdentifier?: string | null;
        sourceFrameDelta?: number | null;
        activeTrackedObjectCount?: number;
        committedShapeCountForCurrentTrack?: number;
        currentTrackKeyframeCount?: number | null;
        currentTrackFrameSpan?: number | null;
        totalAnnotationStateCount?: number;
        totalVisibleCanvasShapeCount?: number | null;
        annotationStateCount?: number;
    }): void {
        if (!this.isEnabled() || this.sessionId === null) return;
        this.frameStartMs = nowMs();
        const activeTrackedObjectCount = meta.activeTrackedObjectCount ?? null;
        const committedShapeCountForCurrentTrack = meta.committedShapeCountForCurrentTrack ?? null;
        const totalAnnotationStateCount = meta.totalAnnotationStateCount ?? meta.annotationStateCount ?? null;
        const currentTrackFrameSpan = meta.currentTrackFrameSpan ?? null;

        if (activeTrackedObjectCount !== null) {
            this.maxActiveTrackedObjectCount = Math.max(this.maxActiveTrackedObjectCount, activeTrackedObjectCount);
        }
        if (committedShapeCountForCurrentTrack !== null) {
            this.maxCommittedShapeCountForCurrentTrack = Math.max(
                this.maxCommittedShapeCountForCurrentTrack,
                committedShapeCountForCurrentTrack,
            );
        }
        if (totalAnnotationStateCount !== null) {
            this.maxTotalAnnotationStateCount = Math.max(this.maxTotalAnnotationStateCount, totalAnnotationStateCount);
        }
        if (currentTrackFrameSpan !== null) {
            this.maxCurrentTrackFrameSpan = Math.max(this.maxCurrentTrackFrameSpan, currentTrackFrameSpan);
        }

        this.currentFrame = {
            sessionId: this.sessionId,
            jobFrameIndex: meta.jobFrameIndex,
            requestId: meta.requestId ?? null,
            sourceFrameIdentifier: meta.sourceFrameIdentifier ?? null,
            sourceFrameDelta: meta.sourceFrameDelta ?? null,
            activeTrackedObjectCount,
            committedShapeCountForCurrentTrack,
            currentTrackKeyframeCount: meta.currentTrackKeyframeCount ?? null,
            currentTrackFrameSpan,
            totalAnnotationStateCount,
            totalVisibleCanvasShapeCount: meta.totalVisibleCanvasShapeCount ?? null,
            canvasObjectCount: null,
            visibleShapeCount: null,
            frameAcquireMs: null,
            lambdaRoundTripMs: null,
            samInferenceMs: null,
            samPreprocessMs: null,
            samPostprocessMs: null,
            candidateExtractionMs: null,
            geometryConversionMs: null,
            validationMs: null,
            shapeSaveMs: null,
            annotationsFetchMs: null,
            storeUpdateMs: null,
            canvasSetupMs: null,
            totalFrameMs: null,
            pendingPromiseCount: null,
            pendingQueueLength: null,
            activeRequestCount: this.activeLambdaRequests,
            samSessionCount: null,
            samMemoryEntryCount: null,
            samTrackedObjectCount: null,
            candidateType: null,
            candidateXtl: null,
            candidateYtl: null,
            candidateXbr: null,
            candidateYbr: null,
            candidateWidth: null,
            candidateHeight: null,
            candidateCenterX: null,
            candidateCenterY: null,
            candidateConfidence: null,
            committedXtl: null,
            committedYtl: null,
            committedXbr: null,
            committedYbr: null,
            committedWidth: null,
            committedHeight: null,
            committedCenterX: null,
            committedCenterY: null,
            candidateAccepted: null,
            validationResult: null,
            validationReason: null,
            stopReason: null,
            cancelled: false,
            requestPayloadBytes: null,
            responsePayloadBytes: null,
            annotationStateCount: meta.annotationStateCount ?? totalAnnotationStateCount ?? null,
            fullStateReplacement: true,
        };
        this.emit('frame_start', {
            jobFrameIndex: meta.jobFrameIndex,
            requestId: meta.requestId ?? null,
            activeTrackedObjectCount,
            committedShapeCountForCurrentTrack,
            currentTrackFrameSpan,
            totalAnnotationStateCount,
        });
    }

    public patchCurrentFrame(patch: Partial<AutoTrackFrameRecord>): void {
        if (!this.isEnabled() || !this.currentFrame) return;
        Object.assign(this.currentFrame, patch);
        if (typeof patch.totalVisibleCanvasShapeCount === 'number') {
            this.maxTotalVisibleCanvasShapeCount = Math.max(
                this.maxTotalVisibleCanvasShapeCount,
                patch.totalVisibleCanvasShapeCount,
            );
        }
    }

    public endFrame(patch: Partial<AutoTrackFrameRecord> = {}): void {
        if (!this.isEnabled() || !this.currentFrame || this.frameStartMs === null) return;
        const totalFrameMs = nowMs() - this.frameStartMs;
        const record = {
            ...(this.currentFrame as AutoTrackFrameRecord),
            ...patch,
            totalFrameMs: patch.totalFrameMs ?? totalFrameMs,
            activeRequestCount: this.activeLambdaRequests,
        };
        this.frameRecords.push(record);
        this.emit('frame_end', record);
        // eslint-disable-next-line no-console
        console.info('[AUTO_TRACK_DIAG_FRAME]', JSON.stringify(record));
        this.currentFrame = null;
        this.frameStartMs = null;
    }

    public onLambdaRequestStart(meta: Record<string, unknown>): void {
        if (!this.isEnabled()) return;
        this.activeLambdaRequests += 1;
        this.maxActiveLambdaRequests = Math.max(this.maxActiveLambdaRequests, this.activeLambdaRequests);
        if (typeof meta.requestId === 'string') {
            this.patchCurrentFrame({ requestId: meta.requestId });
        }
        this.patchCurrentFrame({ activeRequestCount: this.activeLambdaRequests });
        this.emit('lambda_request_start', meta);
    }

    public onLambdaRequestEnd(meta: Record<string, unknown>): void {
        if (!this.isEnabled()) return;
        this.activeLambdaRequests = Math.max(0, this.activeLambdaRequests - 1);
        this.patchCurrentFrame({
            activeRequestCount: this.activeLambdaRequests,
            lambdaRoundTripMs: typeof meta.durationMs === 'number' ? meta.durationMs : null,
            requestPayloadBytes: typeof meta.requestPayloadBytes === 'number' ? meta.requestPayloadBytes : null,
            responsePayloadBytes: typeof meta.responsePayloadBytes === 'number' ? meta.responsePayloadBytes : null,
            samInferenceMs: typeof meta.samInferenceMs === 'number' ? meta.samInferenceMs : null,
            samPreprocessMs: typeof meta.samPreprocessMs === 'number' ? meta.samPreprocessMs : null,
            samPostprocessMs: typeof meta.samPostprocessMs === 'number' ? meta.samPostprocessMs : null,
            samSessionCount: typeof meta.samSessionCount === 'number' ? meta.samSessionCount : null,
            samMemoryEntryCount: typeof meta.samMemoryEntryCount === 'number' ? meta.samMemoryEntryCount : null,
        });
        if (typeof meta.samMemoryEntryCount === 'number') {
            this.maxSamMemoryEntryCount = Math.max(this.maxSamMemoryEntryCount, meta.samMemoryEntryCount);
        }
        this.emit('lambda_request_end', meta);
    }

    public recordCandidate(points: number[] | null, candidateType: string, confidence: number | null = null): void {
        if (!this.isEnabled()) return;
        const g = boxGeometryFromPoints(points);
        this.patchCurrentFrame({
            candidateType,
            candidateConfidence: confidence,
            candidateXtl: g.xtl,
            candidateYtl: g.ytl,
            candidateXbr: g.xbr,
            candidateYbr: g.ybr,
            candidateWidth: g.width,
            candidateHeight: g.height,
            candidateCenterX: g.centerX,
            candidateCenterY: g.centerY,
        });
        this.emit('candidate_received', { candidateType, geometry: g });
    }

    public recordValidation(verdict: string, reason: string | null, accepted: boolean): void {
        if (!this.isEnabled()) return;
        if (verdict === 'soft_invalid') this.softInvalidCount += 1;
        if (verdict === 'hard_invalid') this.hardInvalidCount += 1;
        if (accepted) this.acceptedFrameCount += 1;
        this.patchCurrentFrame({
            validationResult: verdict,
            validationReason: reason,
            candidateAccepted: accepted,
        });
        this.emit('candidate_validated', { verdict, reason, accepted });
    }

    public recordCommittedShape(points: number[] | null): void {
        if (!this.isEnabled()) return;
        const g = boxGeometryFromPoints(points);
        this.patchCurrentFrame({
            committedXtl: g.xtl,
            committedYtl: g.ytl,
            committedXbr: g.xbr,
            committedYbr: g.ybr,
            committedWidth: g.width,
            committedHeight: g.height,
            committedCenterX: g.centerX,
            committedCenterY: g.centerY,
        });
    }

    public recordStaleDrop(meta: Record<string, unknown>): void {
        if (!this.isEnabled()) return;
        this.emit('stale_result_dropped', meta);
    }

    public recordCanvasSetup(
        durationMs: number,
        metrics: {
            visibleShapeCount: number;
            canvasObjectCount: number;
            totalVisibleCanvasShapeCount?: number;
        },
    ): void {
        if (!this.isEnabled()) return;
        const totalVisibleCanvasShapeCount = metrics.totalVisibleCanvasShapeCount ?? metrics.visibleShapeCount;
        this.patchCurrentFrame({
            canvasSetupMs: durationMs,
            visibleShapeCount: metrics.visibleShapeCount,
            canvasObjectCount: metrics.canvasObjectCount,
            totalVisibleCanvasShapeCount,
        });
        this.emit('canvas_setup_end', {
            durationMs,
            visibleShapeCount: metrics.visibleShapeCount,
            canvasObjectCount: metrics.canvasObjectCount,
            totalVisibleCanvasShapeCount,
        });
    }

    public printSessionSummary(): void {
        if (!this.isEnabled() || this.sessionId === null) return;
        const totals = this.frameRecords.map((r) => r.totalFrameMs).filter((v): v is number => v !== null);
        const lambda = this.frameRecords.map((r) => r.lambdaRoundTripMs).filter((v): v is number => v !== null);
        const sam = this.frameRecords.map((r) => r.samInferenceMs).filter((v): v is number => v !== null);
        const save = this.frameRecords.map((r) => r.shapeSaveMs).filter((v): v is number => v !== null);
        const fetch = this.frameRecords.map((r) => r.annotationsFetchMs).filter((v): v is number => v !== null);
        const store = this.frameRecords.map((r) => r.storeUpdateMs).filter((v): v is number => v !== null);
        const canvas = this.frameRecords.map((r) => r.canvasSetupMs).filter((v): v is number => v !== null);
        const frameThirds = sliceThirds(this.frameRecords);

        const summary = {
            sessionId: this.sessionId,
            processedFrameCount: this.frameRecords.length,
            acceptedFrameCount: this.acceptedFrameCount,
            softInvalidCount: this.softInvalidCount,
            hardInvalidCount: this.hardInvalidCount,
            lostCount: this.lostCount,
            cancelledCount: this.cancelledCount,
            counterDefinitions: {
                activeTrackedObjectCount: 'Number of CVAT track objects registered in ToolsControl trackedShapes for this Auto Track session.',
                committedShapeCountForCurrentTrack: 'Number of objectState.save() commits for the active Auto Track object during this session.',
                currentTrackKeyframeCount: 'Exact CVAT keyframe count for the track object; null when not safely observable without full track traversal.',
                currentTrackFrameSpan: 'Inclusive CVAT frame span (keyframes.last - keyframes.first + 1) for the active track object on the current frame.',
                totalAnnotationStateCount: 'Length of annotations.states loaded in the annotation workspace.',
                totalVisibleCanvasShapeCount: 'Count of annotation objects passed to canvas.setup for the current frame after workspace filters.',
                canvasObjectCount: 'Same as visibleShapeCount: objects handed to canvas.setup on the current frame.',
                visibleShapeCount: 'Objects on the current frame after filterAnnotations, excluding tags.',
            },
            sessionMaximums: {
                activeTrackedObjectCount: this.maxActiveTrackedObjectCount,
                committedShapeCountForCurrentTrack: this.maxCommittedShapeCountForCurrentTrack,
                totalAnnotationStateCount: this.maxTotalAnnotationStateCount,
                totalVisibleCanvasShapeCount: this.maxTotalVisibleCanvasShapeCount,
                currentTrackFrameSpan: this.maxCurrentTrackFrameSpan,
            },
            percentiles: {
                lambdaRoundTripMs: { p50: percentile(lambda, 50), p95: percentile(lambda, 95), max: lambda.length ? Math.max(...lambda) : null },
                samInferenceMs: { p50: percentile(sam, 50), p95: percentile(sam, 95), max: sam.length ? Math.max(...sam) : null },
                shapeSaveMs: { p50: percentile(save, 50), p95: percentile(save, 95), max: save.length ? Math.max(...save) : null },
                annotationsFetchMs: { p50: percentile(fetch, 50), p95: percentile(fetch, 95), max: fetch.length ? Math.max(...fetch) : null },
                storeUpdateMs: { p50: percentile(store, 50), p95: percentile(store, 95), max: store.length ? Math.max(...store) : null },
                canvasSetupMs: { p50: percentile(canvas, 50), p95: percentile(canvas, 95), max: canvas.length ? Math.max(...canvas) : null },
                totalFrameMs: { p50: percentile(totals, 50), p95: percentile(totals, 95), max: totals.length ? Math.max(...totals) : null },
            },
            firstThirdAverage: buildCorrelationSlice(frameThirds.early),
            middleThirdAverage: buildCorrelationSlice(frameThirds.middle),
            lastThirdAverage: buildCorrelationSlice(frameThirds.late),
            observedTrends: {
                totalFrameMs: trendLabel(this.frameRecords.map((r) => r.totalFrameMs)),
                lambdaRoundTripMs: trendLabel(this.frameRecords.map((r) => r.lambdaRoundTripMs)),
                annotationsFetchMs: trendLabel(this.frameRecords.map((r) => r.annotationsFetchMs)),
                canvasSetupMs: trendLabel(this.frameRecords.map((r) => r.canvasSetupMs)),
                annotationStateCount: trendLabel(this.frameRecords.map((r) => r.annotationStateCount)),
                currentTrackKeyframeCount: trendLabel(this.frameRecords.map((r) => r.currentTrackKeyframeCount)),
                currentTrackFrameSpan: trendLabel(this.frameRecords.map((r) => r.currentTrackFrameSpan)),
            },
            maximumPendingPromiseCount: this.maxPendingPromiseCount,
            maximumPendingQueueLength: null,
            maximumSamMemoryEntryCount: this.maxSamMemoryEntryCount,
            maximumActiveRequestCount: this.maxActiveLambdaRequests,
        };
        // eslint-disable-next-line no-console
        console.info('[AUTO_TRACK_DIAG_SESSION_SUMMARY]', JSON.stringify(summary));
    }

    public noteLost(): void {
        this.lostCount += 1;
    }
}

export const autoTrackDiagnostics = new AutoTrackDiagnosticsController();
