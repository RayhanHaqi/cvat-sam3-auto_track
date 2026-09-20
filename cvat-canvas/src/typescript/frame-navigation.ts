// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * Dependency-free decision used by `CanvasModel.isAbleToChangeFrame`.
 *
 * An active interaction normally blocks frame navigation. A deliberately opted-in
 * interaction (for example the rally seed draw, which must let the operator move
 * between rallies without toggling collection off and on) may set
 * `settings.allowFrameChange` to keep frame navigation available. Anything else
 * keeps the default block, so DRAW/EDIT/INTERACT are never widened by accident.
 *
 * Runnable without a test framework:
 *   node -e "import('./frame-navigation.ts').then((m) => m.runFrameNavigationSelfCheck())"
 */

export interface InteractionFrameNavigationData {
    enabled?: boolean;
    settings?: object | null;
}

export function interactionAllowsFrameChange(
    data: InteractionFrameNavigationData | null | undefined,
): boolean {
    const settings = data?.settings as { allowFrameChange?: unknown } | undefined | null;
    return data?.enabled === true && settings?.allowFrameChange === true;
}

/* istanbul ignore next */
export function runFrameNavigationSelfCheck(): void {
    const assert = (condition: unknown, message: string): void => {
        if (!condition) {
            throw new Error(`frame-navigation self-check failed: ${message}`);
        }
    };

    assert(
        interactionAllowsFrameChange({ enabled: true, settings: { allowFrameChange: true } }),
        'an opted-in interaction allows frame navigation',
    );
    assert(
        !interactionAllowsFrameChange({ enabled: true, settings: { crosshair: true } }),
        'a plain interaction stays blocked',
    );
    assert(
        !interactionAllowsFrameChange({ enabled: true }),
        'a missing settings object stays blocked',
    );
    assert(
        !interactionAllowsFrameChange({ enabled: true, settings: {} }),
        'an empty settings object stays blocked',
    );
    assert(
        !interactionAllowsFrameChange({ enabled: true, settings: { allowFrameChange: false } }),
        'an explicit false stays blocked',
    );
    assert(
        !interactionAllowsFrameChange({ enabled: false, settings: { allowFrameChange: true } }),
        'a disabled interaction never opts in',
    );
    assert(!interactionAllowsFrameChange(null), 'no interaction never opts in');
    assert(!interactionAllowsFrameChange(undefined), 'undefined interaction never opts in');

    // eslint-disable-next-line no-console
    console.log('frame-navigation self-check passed');
}
