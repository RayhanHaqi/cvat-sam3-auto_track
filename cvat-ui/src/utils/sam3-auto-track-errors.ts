// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

export const SAM3_PRELOAD_RANGE_EXHAUSTED_CODE = 'preload_range_exhausted';

export const SAM3_PRELOAD_RANGE_EXHAUSTED_USER_MESSAGE =
    'Auto Track stopped at a source-timeline boundary.\n' +
    'Re-seed on the next visible ball frame to continue.';

export function isSam3PreloadRangeExhaustedError(error: unknown): boolean {
    return (error as { lambdaErrorCode?: string } | null)?.lambdaErrorCode ===
        SAM3_PRELOAD_RANGE_EXHAUSTED_CODE;
}
