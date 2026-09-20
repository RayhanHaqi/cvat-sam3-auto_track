// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

// Discriminates why the next/previous rally frame was (or was not) resolved:
// - gap: a later rally boundary was found at the given index (may be the last index);
// - end: no later gap, so the last frame index is the target;
// - start: an earlier rally start was found at the given index;
// - fallback: the decision cannot be made, use regular step navigation.
export type RallyNavigationTarget =
    | { kind: 'gap'; index: number }
    | { kind: 'end'; index: number }
    | { kind: 'start'; index: number }
    | { kind: 'fallback' };

/**
 * Extracts the trailing digit sequence immediately before the final extension,
 * e.g. "frame_0008.jpg" -> 8n. Returns null when there is no such sequence.
 * BigInt is used so long numeric tokens keep exact precision.
 */
export function parseFrameToken(filename: string): bigint | null {
    const lastDot = filename.lastIndexOf('.');
    const stem = lastDot > 0 ? filename.slice(0, lastDot) : filename;
    const match = /(\d+)$/.exec(stem);
    return match ? BigInt(match[1]) : null;
}

/**
 * Given job-ordered frame filenames and the index of the current frame, returns
 * the next rally target. Frame-specific names are required: a video job expands
 * one physical name to every frame, so identical names carry no rally boundary
 * and yield fallback. The current or any scanned filename needed for the
 * decision failing to parse also yields fallback.
 */
export function findNextRallyFrame(filenames: string[], currentIndex: number): RallyNavigationTarget {
    if (currentIndex < 0 || currentIndex >= filenames.length) {
        return { kind: 'fallback' };
    }

    if (filenames.length > 1 && filenames.every((name) => name === filenames[0])) {
        return { kind: 'fallback' };
    }

    let previous = parseFrameToken(filenames[currentIndex]);
    if (previous === null) {
        return { kind: 'fallback' };
    }

    for (let index = currentIndex + 1; index < filenames.length; index++) {
        const value = parseFrameToken(filenames[index]);
        if (value === null) {
            return { kind: 'fallback' };
        }
        if (value !== previous + 1n) {
            return { kind: 'gap', index };
        }
        previous = value;
    }

    return { kind: 'end', index: filenames.length - 1 };
}

/**
 * Given job-ordered frame filenames and the index of the current frame, returns
 * the nearest rally start strictly before the current frame. A rally starts at
 * index 0 or wherever the token is not the previous token + 1. When the current
 * frame is itself a rally start, that start is skipped and the prior one is
 * returned; when there is no earlier boundary, index 0 is the target. The same
 * fail-closed guards as findNextRallyFrame apply: identical names, an
 * out-of-range index or any malformed scanned token yield fallback.
 */
export function findPrevRallyFrame(filenames: string[], currentIndex: number): RallyNavigationTarget {
    if (currentIndex < 0 || currentIndex >= filenames.length) {
        return { kind: 'fallback' };
    }

    if (filenames.length > 1 && filenames.every((name) => name === filenames[0])) {
        return { kind: 'fallback' };
    }

    if (parseFrameToken(filenames[currentIndex]) === null) {
        return { kind: 'fallback' };
    }

    for (let index = currentIndex - 1; index >= 1; index--) {
        const value = parseFrameToken(filenames[index]);
        const previous = parseFrameToken(filenames[index - 1]);
        if (value === null || previous === null) {
            return { kind: 'fallback' };
        }
        if (value !== previous + 1n) {
            return { kind: 'start', index };
        }
    }

    return { kind: 'start', index: 0 };
}

/* istanbul ignore next */
function assertSelfCheck(condition: unknown, message: string): void {
    if (!condition) {
        throw new Error(`rally-navigation self-check failed: ${message}`);
    }
}

/* istanbul ignore next */
export function runRallyNavigationSelfCheck(): void {
    function names(values: Array<number | string>): string[] {
        return values.map((value) => (typeof value === 'number' ? `frame_${value}.jpg` : value));
    }

    function nextIndex(values: Array<number | string>, currentIndex: number): unknown {
        return findNextRallyFrame(names(values), currentIndex);
    }

    function prevIndex(values: Array<number | string>, currentIndex: number): unknown {
        return findPrevRallyFrame(names(values), currentIndex);
    }

    // --- next: contiguous run reaches end, first boundary is the gap ---
    assertSelfCheck(
        JSON.stringify(nextIndex([0, 1, 2], 0)) === JSON.stringify({ kind: 'end', index: 2 }),
        'contiguous next hits end',
    );

    const gaps = [0, 1, 2, 10, 11, 20, 21, 22];
    assertSelfCheck(
        JSON.stringify(nextIndex(gaps, 0)) === JSON.stringify({ kind: 'gap', index: 3 }),
        'next first gap',
    );
    assertSelfCheck(
        JSON.stringify(nextIndex(gaps, 3)) === JSON.stringify({ kind: 'gap', index: 5 }),
        'next second gap',
    );

    // --- prev: mid-rally resolves to the current rally start ---
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, 4)) === JSON.stringify({ kind: 'start', index: 3 }),
        'prev mid-rally',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, 2)) === JSON.stringify({ kind: 'start', index: 0 }),
        'prev first rally',
    );
    // --- prev: an exact rally start is skipped for the prior one ---
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, 3)) === JSON.stringify({ kind: 'start', index: 0 }),
        'prev skips exact start',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, 5)) === JSON.stringify({ kind: 'start', index: 3 }),
        'prev skips second start',
    );
    // --- prev: job start and one element target index 0 ---
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, 0)) === JSON.stringify({ kind: 'start', index: 0 }),
        'prev at job start',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex([0], 0)) === JSON.stringify({ kind: 'start', index: 0 }),
        'prev one element',
    );
    // --- BigInt precision beyond Number.MAX_SAFE_INTEGER ---
    const big = [
        'frame_9007199254740991.jpg', 'frame_9007199254740992.jpg',
        'frame_9007199254740993.jpg', 'frame_9007199254740995.jpg',
    ];
    assertSelfCheck(
        JSON.stringify(nextIndex(big, 0)) === JSON.stringify({ kind: 'gap', index: 3 }),
        'next long id gap',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex(big, 3)) === JSON.stringify({ kind: 'start', index: 0 }),
        'prev long id start',
    );
    // --- fail-closed guards ---
    assertSelfCheck(
        JSON.stringify(prevIndex([5, 5, 5], 1)) === JSON.stringify({ kind: 'fallback' }),
        'prev identical names',
    );
    assertSelfCheck(
        JSON.stringify(nextIndex([5, 5, 5], 1)) === JSON.stringify({ kind: 'fallback' }),
        'next identical names',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex([0, 1, 'clip.mp4'], 2)) === JSON.stringify({ kind: 'fallback' }),
        'prev malformed current',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex([0, 'clip.mp4', 5], 2)) === JSON.stringify({ kind: 'fallback' }),
        'prev malformed scanned',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, -1)) === JSON.stringify({ kind: 'fallback' }),
        'prev below bounds',
    );
    assertSelfCheck(
        JSON.stringify(prevIndex(gaps, gaps.length)) === JSON.stringify({ kind: 'fallback' }),
        'prev above bounds',
    );
    // --- extensionless names still parse ---
    assertSelfCheck(
        JSON.stringify(prevIndex(['000', '001', '002', '010'], 3)) ===
            JSON.stringify({ kind: 'start', index: 0 }),
        'extensionless parsed',
    );
}
