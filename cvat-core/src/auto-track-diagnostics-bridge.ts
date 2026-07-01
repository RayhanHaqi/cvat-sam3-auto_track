// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/** Optional bridge from cvat-ui Auto Track diagnostics into cvat-core lambda calls. */

export interface AutoTrackLambdaDiagMeta {
    sessionId: number;
    jobFrameIndex: number;
    requestType: string;
    requestId: string;
}

export interface AutoTrackDiagBridge {
    isEnabled(): boolean;
    onLambdaRequestStart(meta: Record<string, unknown>): void;
    onLambdaRequestEnd(meta: Record<string, unknown>): void;
}

let bridge: AutoTrackDiagBridge | null = null;

export function setAutoTrackDiagBridge(next: AutoTrackDiagBridge | null): void {
    bridge = next;
}

export function getAutoTrackDiagBridge(): AutoTrackDiagBridge | null {
    return bridge;
}

export function createAutoTrackRequestId(): string {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return `atd_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

export function extractAutoTrackDiagMeta(body: Record<string, unknown>): AutoTrackLambdaDiagMeta | null {
    const raw = body._autoTrackDiag;
    if (!raw || typeof raw !== 'object') {
        return null;
    }
    const meta = raw as AutoTrackLambdaDiagMeta;
    if (
        typeof meta.sessionId !== 'number' ||
        typeof meta.jobFrameIndex !== 'number' ||
        typeof meta.requestId !== 'string'
    ) {
        return null;
    }
    return meta;
}

export function estimateJsonBytes(value: unknown): number | null {
    try {
        return new TextEncoder().encode(JSON.stringify(value)).length;
    } catch {
        return null;
    }
}
