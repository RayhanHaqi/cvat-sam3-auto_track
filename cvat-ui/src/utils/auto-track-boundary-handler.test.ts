// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import assert from 'node:assert/strict';
import test from 'node:test';

import {
    findAutoTrackBoundaryCommitClientID,
    filterAutoTrackClientIDs,
    shouldCommitShapeForInactiveTrackingBranch,
    toAutoTrackObjectSnapshots,
} from './auto-track-boundary-commit';
import {
    handlePreloadRangeExhaustedBoundaryCommit,
    resolvePreloadRangeExhaustedCommitClientID,
} from './auto-track-boundary-handler';

function makeObjectState(clientID: number, prev: number, last: number, outside = false): {
    clientID: number;
    keyframes: { prev: number; last: number };
    outside: boolean;
    outsideCommitted: boolean;
    keyframe: boolean;
    save(): Promise<void>;
} {
    return {
        clientID,
        keyframes: { prev, last },
        outside,
        outsideCommitted: false,
        keyframe: false,
        async save(): Promise<void> {
            this.outsideCommitted = this.outside;
        },
    };
}

function makeDeps(overrides: Partial<Parameters<typeof handlePreloadRangeExhaustedBoundaryCommit>[0]> = {}): {
    deps: Parameters<typeof handlePreloadRangeExhaustedBoundaryCommit>[0];
    saves: number[];
    warnings: string[];
    committed: number[];
    setSessionInvalid: () => void;
    cancelSession: () => void;
    objectStates: ReturnType<typeof makeObjectState>[];
} {
    let currentSessionId = 1;
    let autoTrackActive = true;
    const saves: number[] = [];
    const warnings: string[] = [];
    const committed: number[] = [];

    const objectStates = [
        makeObjectState(10, 25, 25),
        makeObjectState(20, 25, 25),
    ];

    const deps = {
        boundaryFrame: 26,
        sessionId: 1,
        trackerGroupClientIDs: [10],
        objectStates,
        getCurrentSessionId: () => currentSessionId,
        isAutoTrackSessionActive: () => autoTrackActive,
        resolveObjectState: (clientID: number) => {
            const state = objectStates.find((item) => item.clientID === clientID);
            if (!state) {
                return null;
            }
            return {
                get outside() { return state.outside; },
                set outside(value: boolean) { state.outside = value; },
                get keyframe() { return state.keyframe; },
                set keyframe(value: boolean) { state.keyframe = value; },
                save: async () => {
                    saves.push(clientID);
                    await state.save();
                },
            };
        },
        warnOnlyStop: () => {
            warnings.push('warn_only');
            autoTrackActive = false;
            currentSessionId += 1;
        },
        onOutsideCommitted: async (clientID: number) => {
            committed.push(clientID);
            autoTrackActive = false;
            currentSessionId += 1;
        },
        ...overrides,
    };

    return {
        deps,
        saves,
        warnings,
        committed,
        setSessionInvalid: () => {
            currentSessionId += 1;
        },
        cancelSession: () => {
            autoTrackActive = false;
            currentSessionId += 1;
        },
        objectStates,
    };
}

test('A: only failing tracker-group object is selected when two tracks share prev === B - 1', () => {
    const boundaryFrame = 26;
    const snapshots = toAutoTrackObjectSnapshots([
        { clientID: 10, keyframes: { prev: 25, last: 25 } },
        { clientID: 20, keyframes: { prev: 25, last: 25 } },
    ]);

    assert.equal(
        resolvePreloadRangeExhaustedCommitClientID([10], snapshots, boundaryFrame),
        10,
    );
    assert.equal(
        resolvePreloadRangeExhaustedCommitClientID([20], snapshots, boundaryFrame),
        20,
    );
    assert.equal(
        resolvePreloadRangeExhaustedCommitClientID([10, 20], snapshots, boundaryFrame),
        10,
    );
});

test('A: handler commits outside only for failing group target', async () => {
    const harness = makeDeps();
    const result = await handlePreloadRangeExhaustedBoundaryCommit(harness.deps);

    assert.equal(result, 'committed');
    assert.deepEqual(harness.saves, [10]);
    assert.deepEqual(harness.committed, [10]);
    assert.equal(harness.objectStates[0].outside, true);
    assert.equal(harness.objectStates[0].keyframe, true);
    assert.equal(harness.objectStates[1].outside, false);
});

test('B: stale typed exhaustion after Esc does not call save', async () => {
    const harness = makeDeps();
    harness.cancelSession();

    const result = await handlePreloadRangeExhaustedBoundaryCommit(harness.deps);

    assert.equal(result, 'stale');
    assert.deepEqual(harness.saves, []);
    assert.deepEqual(harness.committed, []);
    assert.deepEqual(harness.warnings, []);
});

test('C: superseded session does not mutate new track', async () => {
    const harness = makeDeps({
        trackerGroupClientIDs: [99],
        objectStates: [
            makeObjectState(99, 25, 25),
        ],
    });
    harness.setSessionInvalid();

    const result = await handlePreloadRangeExhaustedBoundaryCommit(harness.deps);

    assert.equal(result, 'stale');
    assert.deepEqual(harness.saves, []);
});

test('D: inactive branch guard blocks stale commits after stop', () => {
    assert.equal(shouldCommitShapeForInactiveTrackingBranch(1, 2, false), false);
    assert.equal(shouldCommitShapeForInactiveTrackingBranch(1, 1, false), true);
    assert.equal(shouldCommitShapeForInactiveTrackingBranch(1, 1, true), false);
});

test('E: repeated boundary handling skips already-outside keyframe at B', () => {
    const boundaryFrame = 26;
    const snapshots = toAutoTrackObjectSnapshots([
        { clientID: 10, keyframes: { prev: 25, last: 26 }, outside: true },
    ]);

    assert.equal(
        findAutoTrackBoundaryCommitClientID([10], snapshots, boundaryFrame),
        null,
    );
});

test('E: nullable client IDs are filtered safely', () => {
    assert.deepEqual(filterAutoTrackClientIDs([10, null, undefined, 20]), [10, 20]);
    const nullableSnapshots = toAutoTrackObjectSnapshots([
        { clientID: null, keyframes: { prev: 25 } },
        { clientID: 10, keyframes: { prev: 25, last: 25 } },
    ]);
    assert.equal(
        findAutoTrackBoundaryCommitClientID([null, 10], nullableSnapshots, 26),
        10,
    );
});

test('F: warn-only path when no eligible target in failing group', async () => {
    const harness = makeDeps({
        trackerGroupClientIDs: [10],
        objectStates: [
            makeObjectState(10, 24, 25),
        ],
    });

    const result = await handlePreloadRangeExhaustedBoundaryCommit(harness.deps);

    assert.equal(result, 'warn_only');
    assert.deepEqual(harness.saves, []);
    assert.deepEqual(harness.warnings, ['warn_only']);
});

test('F: session invalidated before save performs no outside commit callback', async () => {
    const harness = makeDeps();
    const originalResolve = harness.deps.resolveObjectState;
    harness.deps.resolveObjectState = (clientID: number) => {
        const resolved = originalResolve(clientID);
        if (!resolved) {
            return null;
        }
        return {
            get outside() { return resolved.outside; },
            set outside(value: boolean) { resolved.outside = value; },
            get keyframe() { return resolved.keyframe; },
            set keyframe(value: boolean) { resolved.keyframe = value; },
            save: async () => {
                harness.setSessionInvalid();
                await resolved.save();
            },
        };
    };

    const result = await handlePreloadRangeExhaustedBoundaryCommit(harness.deps);

    assert.equal(result, 'stale');
    assert.deepEqual(harness.committed, []);
});
