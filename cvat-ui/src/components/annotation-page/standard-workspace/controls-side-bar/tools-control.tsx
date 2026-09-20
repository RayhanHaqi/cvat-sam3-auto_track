// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { ReactPortal } from 'react';
import ReactDOM from 'react-dom';
import { connect } from 'react-redux';
import Icon, {
    EnvironmentFilled,
    EnvironmentOutlined,
    LoadingOutlined,
    QuestionCircleOutlined,
} from '@ant-design/icons';
import Popover from 'antd/lib/popover';
import Select from 'antd/lib/select';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import Text from 'antd/lib/typography/Text';
import Tabs from 'antd/lib/tabs';
import { Row, Col } from 'antd/lib/grid';
import notification from 'antd/lib/notification';
import message from 'antd/lib/message';
import Switch from 'antd/lib/switch';
import lodash from 'lodash';

import { AIToolsIcon } from 'icons';
import { Canvas, convertShapesForInteractor, InteractionResult } from 'cvat-canvas-wrapper';
import {
    getCore, Label, MLModel, ObjectState, ObjectType, ShapeType, Job,
    MinimalShape, InteractorResults, PreloadSummary, PreloadTimelineEntry, TrackerResults,
} from 'cvat-core-wrapper';
import openCVWrapper from 'utils/opencv-wrapper/opencv-wrapper';
import {
    CombinedState, ActiveControl, ToolsBlockerState, PluginComponent,
} from 'reducers';
import {
    interactWithCanvas,
    switchNavigationBlocked as switchNavigationBlockedAction,
    fetchAnnotationsAsync,
    updateAnnotationsAsync,
    createAnnotationsAsync,
    changeFrameAsync,
} from 'actions/annotation-actions';
import DetectorRunner, { AnnotateTaskRequestBody } from 'components/model-runner-modal/detector-runner';
import LabelSelector from 'components/label-selector/label-selector';
import CVATTooltip from 'components/common/cvat-tooltip';
import CVATMarkdown from 'components/common/cvat-markdown';

import ApproximationAccuracy, {
    thresholdFromAccuracy,
} from 'components/annotation-page/standard-workspace/controls-side-bar/approximation-accuracy';
import ConfidenceThreshold from 'components/annotation-page/standard-workspace/controls-side-bar/confidence-threshold';
import { switchToolsBlockerState } from 'actions/settings-actions';
import { autoTrackDiagnostics, boxGeometryFromPoints } from 'utils/auto-track-diagnostics';
import {
    createAutoTrackRequestId,
    setAutoTrackDiagBridge,
} from 'cvat-core/src/auto-track-diagnostics-bridge';
import { getFramesMeta } from 'cvat-core/src/frames';
import withVisibilityHandling from './handle-popover-visibility';
import ToolsTooltips from './interactor-tooltips';
import {
    buildRallySeedCandidate,
    computeFilenameRallyContract,
    decideSeedPrompt,
    parseRawSourceIndexBig,
    preflightRallySeeds,
    RallyRunStop,
    rallySegmentWriteFrames,
    RallySegment,
    RallySeedCandidate,
    RallySeedRecord,
    sameRallyPoints,
    verifyRallySeedShape,
} from './multi_rally_segments';
import {
    AutoTrackBallHistory,
    createEmptyAutoTrackHistory,
    CVAT_UI_SAM3_G0_TIMELINE_ENABLED,
    SAM3_V4_TRACKER_ID,
    getLossTerminationPoints,
    isExplicitSam3Loss,
    isShapeUnusable,
    recordInvalidAutoTrackHistory,
    recordValidAutoTrackHistory,
    shouldConfirmAutoTrackLoss,
    validateAutoTrackCandidate,
    validatePreloadTimeline,
} from './auto_track_ball_validation';

interface StateToProps {
    canvasInstance: Canvas;
    labels: Label[];
    states: ObjectState[];
    activeLabelID: number | null;
    jobInstance: Job;
    isActivated: boolean;
    frame: number;
    interactors: MLModel[];
    detectors: MLModel[];
    trackers: MLModel[];
    curZOrder: number;
    defaultApproxPolyAccuracy: number;
    toolsBlockerState: ToolsBlockerState;
    frameIsDeleted: boolean;
    interactorExtras: PluginComponent[];
    /** Current shared navigation block, captured so a run can restore the prior value. */
    navigationBlocked: boolean;
}

interface DispatchToProps {
    updateAnnotations: (states: ObjectState[]) => Promise<void>;
    createAnnotations: (states: ObjectState[]) => Promise<void>;
    fetchAnnotations: () => Promise<void>;
    onInteractionStart: typeof interactWithCanvas;
    onSwitchToolsBlockerState: typeof switchToolsBlockerState;
    switchNavigationBlocked: typeof switchNavigationBlockedAction;
    changeFrame: typeof changeFrameAsync;
}

const MIN_SUPPORTED_INTERACTOR_VERSION = 2;
const core = getCore();
const CustomPopover = withVisibilityHandling(Popover, 'tools-control');

function mapStateToProps(state: CombinedState): StateToProps {
    const {
        annotation: {
            job: { instance: jobInstance, labels },
            canvas: { instance: canvasInstance, activeControl },
            player: {
                frame: { number: frame, data: { deleted: frameIsDeleted } },
                navigationBlocked,
            },
            annotations: {
                zLayer: { cur: curZOrder },
                states,
            },
            drawing: { activeLabelID },
        },
        models: {
            interactors, detectors, trackers,
        },
        settings: {
            workspace: { toolsBlockerState, defaultApproxPolyAccuracy },
        },
        plugins: {
            components: {
                aiTools: {
                    interactors: {
                        extras: interactorExtras,
                    },
                },
            },
        },
    } = state;

    return {
        interactors,
        detectors,
        trackers,
        isActivated: activeControl === ActiveControl.AI_TOOLS,
        activeLabelID,
        labels,
        states,
        canvasInstance: canvasInstance as Canvas,
        jobInstance: jobInstance as Job,
        frame,
        curZOrder,
        defaultApproxPolyAccuracy,
        toolsBlockerState,
        frameIsDeleted,
        interactorExtras,
        navigationBlocked,
    };
}

const mapDispatchToProps = {
    onInteractionStart: interactWithCanvas,
    updateAnnotations: updateAnnotationsAsync,
    createAnnotations: createAnnotationsAsync,
    fetchAnnotations: fetchAnnotationsAsync,
    onSwitchToolsBlockerState: switchToolsBlockerState,
    switchNavigationBlocked: switchNavigationBlockedAction,
    changeFrame: changeFrameAsync,
};

type Props = StateToProps & DispatchToProps;
interface TrackedShape {
    clientID: number;
    serverlessState: any;
    shapePoints: number[];
    trackerModel: MLModel;
    autoTrackHistory?: AutoTrackBallHistory;
}

interface RallyStagedFrame {
    frame: number;
    points: number[];
}

interface RallyStageContext {
    token: number;
    tracker: MLModel;
    labelID: number;
    segment: RallySegment;
    segmentCount: number;
    /** Live seed track captured when the segment started. */
    seedClientID: number;
    seedPoints: number[];
    staged: RallyStagedFrame[];
    history: AutoTrackBallHistory;
    lastStagedPoints: number[];
    isFrameDeleted: (frame: number) => boolean;
}

interface State {
    activeInteractor: MLModel | null;
    activeLabelID: number | null;
    activeTracker: MLModel | null;
    startInteractingWithBox: boolean;
    convertMasksToPolygons: boolean;
    trackedShapes: TrackedShape[];
    fetching: boolean;
    interactorResponseReceived: boolean;
    showConfidenceControl: boolean;
    approxPolyAccuracy: number;
    thresholdValue: number;
    mode: 'detection' | 'interaction' | 'tracking' | 'seed';
    portals: React.ReactPortal[];
    autoTrackActive: boolean;
    collectingSeeds: boolean;
    rallySeeds: RallySeedRecord[];
    rallyRunActive: boolean;
    rallyStatus: string | null;
}

type DetectorResults = Extract<Awaited<ReturnType<typeof core.lambda.call>>, { version: number }>;

function trackedRectangleMapper(shape: MinimalShape | null): MinimalShape | null {
    if (shape === null) {
        return null;
    }
    return {
        type: ShapeType.RECTANGLE,
        points: shape.points.reduce(
            (acc: number[], value: number, index: number): number[] => {
                if (index % 2) {
                // y
                    acc[1] = Math.min(acc[1], value);
                    acc[3] = Math.max(acc[3], value);
                } else {
                // x
                    acc[0] = Math.min(acc[0], value);
                    acc[2] = Math.max(acc[2], value);
                }
                return acc;
            },
            [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER, Number.MIN_SAFE_INTEGER],
        ),
    };
}

function registerPlugin(): (callback: null | (() => void)) => void {
    let onTrigger: null | (() => void) = null;
    const listener = {
        name: 'Remove annotations listener',
        description: 'Tracker needs to know when annotations is reset in the job',
        cvat: {
            classes: {
                Job: {
                    prototype: {
                        annotations: {
                            clear: {
                                leave(self: any, result: any) {
                                    if (typeof onTrigger === 'function') {
                                        onTrigger();
                                    }
                                    return result;
                                },
                            },
                        },
                    },
                },
            },
        },
    };

    core.plugins.register(listener);

    return (callback: null | (() => void)) => {
        onTrigger = callback;
    };
}

const onRemoveAnnotations = registerPlugin();

export class ToolsControlComponent extends React.PureComponent<Props, State> {
    private autoTrackSessionActive = false;

    private autoTrackSessionId = 0;

    private autoTrackCommittedSaveCount = 0;

    private autoTrackPreloadTimelineByFrame: Map<number, PreloadTimelineEntry> | null = null;

    private autoTrackPreloadSummary: PreloadSummary | null = null;

    private autoTrackUseG0Timeline = false;

    private suppressNextTrackingCancel = false;

    /** Scoped true while frame navigation resets seed prompt overlays. */
    private suppressNextSeedCancel = false;

    private multiRallyRunActive = false;

    private multiRallyRunToken = 0;

    private multiRallyPhase: 'idle' | 'staging' | 'cancelling' | 'committing' = 'idle';

    private multiRallyCancelRequested = false;

    /** Shared navigation block value to restore, captured when the run acquires it. */
    private multiRallyPriorNavigationBlocked = false;

    private mounted = false;

    /** Synchronous twin of `state.collectingSeeds` (setState is async). */
    private seedCollecting = false;

    /** Prompts already committed to tracks in the current Collect Seeds session. */
    private seedCommittedPromptCount = 0;

    private interaction: {
        id: string | null;
        isAborted: boolean;
        latestPostponedEvent: Event | null;
        latestResponse: {
            rle: Int32Array;
            points: [number, number][];
            approximatedPoints: [number, number][];
            confidence: number;
        }[];
        latestRequest: null | {
            interactor: MLModel;
            data: {
                frame: number;
                neg_points: number[][];
                pos_points: number[][];
                obj_bbox: number[][];
            };
        } | null;
        closeFetchingMessage: (() => void) | null;
        noShapesMessage: (() => void) | null;
    };

    public constructor(props: Props) {
        super(props);

        const supportedTrackers = this.getSupportedTrackers();

        this.state = {
            convertMasksToPolygons: false,
            startInteractingWithBox: false,
            activeInteractor: props.interactors.length ? props.interactors[0] : null,
            activeTracker: supportedTrackers.length ? supportedTrackers[0] : null,
            activeLabelID: props.labels.length ? props.labels[0].id as number : null,
            approxPolyAccuracy: props.defaultApproxPolyAccuracy,
            thresholdValue: 0.5,
            trackedShapes: [],
            fetching: false,
            interactorResponseReceived: false,
            showConfidenceControl: false,
            mode: 'interaction',
            portals: [],
            autoTrackActive: false,
            collectingSeeds: false,
            rallySeeds: [],
            rallyRunActive: false,
            rallyStatus: null,
        };

        this.interaction = {
            id: null,
            isAborted: false,
            latestPostponedEvent: null,
            latestResponse: [],
            latestRequest: null,
            closeFetchingMessage: null,
            noShapesMessage: null,
        };
    }

    public componentDidMount(): void {
        const { canvasInstance } = this.props;
        this.mounted = true;
        onRemoveAnnotations(() => {
            this.setState({ trackedShapes: [], rallySeeds: [] });
        });

        this.setState({
            portals: this.collectTrackerPortals(),
        });

        canvasInstance.html().addEventListener('canvas.interacted', this.interactionListener);
        canvasInstance.html().addEventListener('canvas.canceled', this.cancelListener);
        window.addEventListener('keydown', this.autoTrackEscapeListener, true);
        setAutoTrackDiagBridge(autoTrackDiagnostics);
    }

    public componentDidUpdate(prevProps: Props, prevState: State): void {
        const {
            isActivated, defaultApproxPolyAccuracy, states, toolsBlockerState, frame, canvasInstance,
        } = this.props;
        const {
            approxPolyAccuracy, mode, activeTracker, thresholdValue,
        } = this.state;

        if (prevProps.states !== states || prevState.activeTracker !== activeTracker) {
            this.setState({
                portals: this.collectTrackerPortals(),
            });
        }

        if (prevProps.frame !== frame && this.seedCollecting && mode === 'seed') {
            // draw_box stays active across frames, so re-arming alone would keep the
            // old translucent prompt overlay. canvas.cancel() synchronously clears the
            // prompts; its 'canvas.canceled' event is this reset, not a user cancel.
            this.suppressNextSeedCancel = true;
            try {
                canvasInstance.cancel();
            } finally {
                this.suppressNextSeedCancel = false;
            }
            this.seedCommittedPromptCount = 0;
            this.armSeedDraw();
        }

        if (prevProps.isActivated && !isActivated) {
            window.removeEventListener('contextmenu', this.contextmenuDisabler);
            if (!this.autoTrackSessionActive) {
                this.setState({ autoTrackActive: false });
            }
            if (this.seedCollecting && !this.multiRallyRunActive) {
                this.stopSeedCollection();
            }

            // hide interaction messages if exists
            for (const messageCallback of ['closeFetchingMessage', 'noShapesMessage'] as const) {
                if (this.interaction[messageCallback]) {
                    this.interaction[messageCallback]?.();
                    this.interaction[messageCallback] = null;
                }
            }
        } else if (!prevProps.isActivated && isActivated) {
            // reset flags when start interaction/tracking
            this.interaction = {
                id: null,
                isAborted: false,
                latestPostponedEvent: null,
                latestResponse: [],
                latestRequest: null,
                closeFetchingMessage: null,
                noShapesMessage: null,
            };

            this.setState({
                approxPolyAccuracy: defaultApproxPolyAccuracy,
                interactorResponseReceived: false,
                showConfidenceControl: false,
            });
            window.addEventListener('contextmenu', this.contextmenuDisabler);
        }

        if (
            prevProps.toolsBlockerState.algorithmsLocked &&
            !toolsBlockerState.algorithmsLocked &&
            isActivated && mode === 'interaction' && this.interaction.latestPostponedEvent
        ) {
            this.onInteraction(this.interaction.latestPostponedEvent);
        }

        if (prevState.thresholdValue !== thresholdValue) {
            if (isActivated && mode === 'interaction') {
                this.drawIntermediateShapesOnCanvas();
            }
        }

        if (prevState.approxPolyAccuracy !== approxPolyAccuracy) {
            if (isActivated && mode === 'interaction') {
                this.interaction.latestResponse.forEach(({ points }, idx) => {
                    const approximated = this.approximateResponsePoints(points);
                    this.interaction.latestResponse[idx].approximatedPoints = approximated;
                });

                this.drawIntermediateShapesOnCanvas();
            }
        }

        this.checkTrackedStates(prevProps);
    }

    public componentWillUnmount(): void {
        const { canvasInstance, switchNavigationBlocked } = this.props;
        this.mounted = false;
        // Hard-invalidate any in-flight rally run; no setState is allowed here.
        const wasRunning = this.multiRallyRunActive;
        this.multiRallyRunToken += 1;
        this.multiRallyRunActive = false;
        this.multiRallyCancelRequested = true;
        this.multiRallyPhase = 'idle';
        if (wasRunning) {
            // Synchronously release the shared blocks; the owner would otherwise be gone.
            switchNavigationBlocked(this.multiRallyPriorNavigationBlocked);
        }
        onRemoveAnnotations(null);
        canvasInstance.html().removeEventListener('canvas.interacted', this.interactionListener);
        canvasInstance.html().removeEventListener('canvas.canceled', this.cancelListener);
        window.removeEventListener('keydown', this.autoTrackEscapeListener, true);
        setAutoTrackDiagBridge(null);
    }

    /* eslint-disable react/sort-comp -- Keep the Auto Track safety lifecycle together for auditability. */
    private noteAutoTrackCommittedSave(): void {
        this.autoTrackCommittedSaveCount += 1;
        if (autoTrackDiagnostics.isEnabled()) {
            autoTrackDiagnostics.patchCurrentFrame({
                committedShapeCountForCurrentTrack: this.autoTrackCommittedSaveCount,
            });
        }
    }

    private buildAutoTrackDiagPayload(
        sessionId: number,
        frame: number,
        requestType: string,
        requestId: string,
    ): Record<string, unknown> {
        if (!autoTrackDiagnostics.isEnabled() || !this.autoTrackSessionActive) {
            return {};
        }
        return {
            _autoTrackDiag: {
                sessionId,
                jobFrameIndex: frame,
                requestType,
                requestId,
            },
        };
    }

    private stopAutoTrackSession = (): void => {
        this.clearG0TimelineSession();
        this.autoTrackSessionActive = false;
        this.suppressNextTrackingCancel = false;
        this.setState({ autoTrackActive: false });
    };

    private invalidateAutoTrackSession = (): void => {
        this.clearG0TimelineSession();
        this.autoTrackSessionId += 1;
    };

    private clearG0TimelineSession = (): void => {
        this.autoTrackPreloadTimelineByFrame = null;
        this.autoTrackPreloadSummary = null;
        this.autoTrackUseG0Timeline = false;
    };

    private tryIngestG0PreloadTimeline(tracker: MLModel, response: TrackerResults): void {
        this.clearG0TimelineSession();
        if (!CVAT_UI_SAM3_G0_TIMELINE_ENABLED || tracker.id !== SAM3_V4_TRACKER_ID) {
            return;
        }
        const validated = validatePreloadTimeline(response);
        if (!validated.valid || validated.summary.preloaded_count <= 1) {
            return;
        }
        const timelineByFrame = new Map<number, PreloadTimelineEntry>();
        validated.timeline.forEach((entry) => timelineByFrame.set(entry.frame, entry));
        this.autoTrackPreloadTimelineByFrame = timelineByFrame;
        this.autoTrackPreloadSummary = validated.summary;
        this.autoTrackUseG0Timeline = true;
        if (autoTrackDiagnostics.isEnabled()) {
            autoTrackDiagnostics.noteG0TimelineIngested(
                validated.summary.preloaded_count,
                validated.summary.stop_reason ?? null,
            );
        }
    }

    private shouldUseLocalG0Timeline(tracker: MLModel, numObjects: number): boolean {
        return (
            this.autoTrackUseG0Timeline &&
            CVAT_UI_SAM3_G0_TIMELINE_ENABLED &&
            tracker.id === SAM3_V4_TRACKER_ID &&
            numObjects === 1 &&
            this.autoTrackPreloadTimelineByFrame !== null &&
            this.autoTrackPreloadSummary !== null
        );
    }

    private buildLocalTimelineTrackerResponse(
        frame: number,
        serverlessState: any,
    ): TrackerResults | 'skipped' | null {
        const summary = this.autoTrackPreloadSummary;
        const timelineByFrame = this.autoTrackPreloadTimelineByFrame;
        if (!summary || !timelineByFrame) {
            return null;
        }
        const entry = timelineByFrame.get(frame);
        if (!entry) {
            return null;
        }
        if (entry.bbox_status === 'skipped') {
            return 'skipped';
        }
        const shape = entry.bbox_status === 'valid' && entry.points ?
            { type: ShapeType.RECTANGLE, points: entry.points } as MinimalShape :
            null;
        const syntheticState = entry.lost ?
            { ...serverlessState, lost: true } :
            serverlessState;
        return {
            shapes: [shape],
            states: [syntheticState],
            ...(entry.lost ? { lost: true } : {}),
        };
    }

    private async handleTrackingWindowComplete(
        autoTrackSessionId: number,
        trackableObjects: { clientIDs: number[] },
        objectStates: ObjectState[],
        trackedShapes: TrackedShape[],
        lostClientIDs: number[],
    ): Promise<boolean> {
        let windowStopHandled = false;
        for (let i = 0; i < trackableObjects.clientIDs.length; i++) {
            const clientID = trackableObjects.clientIDs[i];
            const [objectState] = objectStates.filter(
                (_state: any): boolean => _state.clientID === clientID,
            );
            const trackedShape = trackedShapes.find(
                (_trackedShape: TrackedShape) => _trackedShape.clientID === clientID,
            );
            if (!objectState || !trackedShape) {
                continue;
            }
            const lossPoints = getLossTerminationPoints(trackedShape, objectState);
            if (!lossPoints) {
                this.stopAutoTrackWithoutOutside(
                    'Tracking window complete, but no prior valid box was available. Please manually mark or re-seed from this frame.',
                    'window_complete_no_prior_bbox',
                );
                windowStopHandled = true;
                break;
            }
            // eslint-disable-next-line no-await-in-loop
            await this.commitAutoTrackLoss(
                objectState,
                trackedShape.clientID,
                lostClientIDs,
                lossPoints,
                'Tracking window complete. Shape marked outside at the last valid box.',
            );
            windowStopHandled = true;
            break;
        }
        if (!windowStopHandled) {
            if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                autoTrackDiagnostics.endSession('session_complete', 'tracking_window_complete');
            }
            this.stopAutoTrackSession();
            notification.info({
                message: 'Tracking window complete',
                description: 'Start a new Auto Track session for the next segment.',
                duration: 4,
            });
        }
        return windowStopHandled;
    }

    private isAutoTrackSessionCurrent = (sessionId: number): boolean => (
        sessionId === this.autoTrackSessionId
    );

    private cancelAutoTrackSession = (): void => {
        if (!this.autoTrackSessionActive) {
            return;
        }
        if (autoTrackDiagnostics.isEnabled()) {
            autoTrackDiagnostics.endSession('session_cancel');
        }
        this.invalidateAutoTrackSession();
        this.stopAutoTrackSession();
        const { canvasInstance } = this.props;
        canvasInstance.cancel();
    };

    private autoTrackEscapeListener = (event: KeyboardEvent): void => {
        if (!this.autoTrackSessionActive && !this.multiRallyRunActive) {
            return;
        }
        if (event.key !== 'Escape' && event.key !== 'Esc' && event.keyCode !== 27) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        if (this.autoTrackSessionActive) {
            this.cancelAutoTrackSession();
        }
        if (this.multiRallyRunActive) {
            this.stopMultiRallyRun();
        }
    };

    private getSupportedTrackers(): MLModel[] {
        const { trackers } = this.props;
        return trackers.filter((tracker: MLModel) => (
            tracker.supportedShapeTypes!.includes(ShapeType.RECTANGLE) &&
            !/sam\s*2/i.test(tracker.name) &&
            !String(tracker.id).toLowerCase().includes('sam2')
        ));
    }

    private stopAutoTrackWithoutOutside(
        reason: string,
        diagTag: string,
    ): void {
        notification.warning({
            message: 'Auto Track stopped',
            description: <CVATMarkdown>{reason}</CVATMarkdown>,
            duration: 8,
        });
        if (autoTrackDiagnostics.isEnabled()) {
            autoTrackDiagnostics.endSession('session_lost', diagTag);
        }
        this.invalidateAutoTrackSession();
        this.stopAutoTrackSession();
    }

    private async commitAutoTrackLoss(
        objectState: ObjectState,
        clientID: number,
        lostClientIDs: number[],
        lossPoints: number[],
        reason: string,
    ): Promise<void> {
        const { fetchAnnotations } = this.props;
        objectState.points = [...lossPoints];
        objectState.outside = true;
        objectState.keyframe = true;
        autoTrackDiagnostics.emit('shape_commit_start', { clientID, reason });
        const saveStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
        try {
            await objectState.save();
        } catch (error: unknown) {
            const errMsg = error instanceof Error ? error.message : String(error);
            const saveFailureMessage = [
                'SAM3 reported target loss, but the outside keyframe could not be saved',
                `(${errMsg}). Reload annotations and review this frame before retrying.`,
            ].join(' ');
            notification.error({
                message: 'Auto Track stopped',
                description: (
                    <CVATMarkdown>
                        {saveFailureMessage}
                    </CVATMarkdown>
                ),
                duration: null,
            });
            try {
                await fetchAnnotations();
            } catch {
                // Best-effort reload after failed outside save.
            }
            if (autoTrackDiagnostics.isEnabled()) {
                autoTrackDiagnostics.endSession('session_lost', 'outside_save_failed');
            }
            this.invalidateAutoTrackSession();
            this.stopAutoTrackSession();
            return;
        }
        this.noteAutoTrackCommittedSave();
        autoTrackDiagnostics.patchCurrentFrame({
            shapeSaveMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - saveStart,
        });
        autoTrackDiagnostics.emit('shape_commit_end', { clientID, reason });
        lostClientIDs.push(clientID);
        notification.warning({
            message: 'Auto Track stopped',
            description: <CVATMarkdown>{reason}</CVATMarkdown>,
            duration: 8,
        });
        autoTrackDiagnostics.noteLost();
        autoTrackDiagnostics.endSession('session_lost', reason);
        this.invalidateAutoTrackSession();
        this.stopAutoTrackSession();
    }

    private async processAutoTrackShapeResult(
        sessionId: number,
        tracker: MLModel,
        response: TrackerResults,
        shape: MinimalShape | null,
        state: any,
        objectState: ObjectState,
        trackedShape: TrackedShape,
        lostClientIDs: number[],
    ): Promise<'lost' | 'soft_skip' | 'valid' | 'stale' | 'stopped'> {
        if (!this.isAutoTrackSessionCurrent(sessionId)) {
            autoTrackDiagnostics.recordStaleDrop({
                requestSessionId: sessionId,
                currentSessionId: this.autoTrackSessionId,
                jobFrameIndex: objectState.frame,
            });
            return 'stale';
        }

        const history = trackedShape.autoTrackHistory ??
            createEmptyAutoTrackHistory(trackedShape.shapePoints);

        if (isExplicitSam3Loss(String(tracker.id), response, state, shape)) {
            autoTrackDiagnostics.recordCandidate(null, 'lost', null);
            autoTrackDiagnostics.recordValidation('lost', 'explicit_sam3_loss', false);
            const lossPoints = getLossTerminationPoints(trackedShape, objectState);
            if (!lossPoints) {
                this.stopAutoTrackWithoutOutside(
                    'SAM3 reported target loss, but no prior valid box was available. Please manually mark or re-seed from this frame.',
                    'explicit_loss_no_prior_bbox',
                );
                return 'stopped';
            }
            await this.commitAutoTrackLoss(
                objectState,
                trackedShape.clientID,
                lostClientIDs,
                lossPoints,
                'SAM3 reported target loss. Shape marked outside at the last valid box.',
            );
            return 'lost';
        }

        if (isShapeUnusable(shape)) {
            autoTrackDiagnostics.recordCandidate(null, 'lost', null);
            autoTrackDiagnostics.recordValidation('lost', 'protocol_error_unusable_bbox', false);
            this.stopAutoTrackWithoutOutside(
                'Tracker returned an unusable box without explicit SAM3 loss. Auto Track stopped without marking outside.',
                'protocol_error_unusable_bbox',
            );
            return 'stopped';
        }

        if (!shape) {
            return 'stopped';
        }

        autoTrackDiagnostics.recordCandidate(shape.points, 'rectangle', null);
        const validationStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
        const verdict = validateAutoTrackCandidate(shape.points, history);
        autoTrackDiagnostics.patchCurrentFrame({
            validationMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - validationStart,
        });
        if (verdict === 'valid') {
            if (!this.isAutoTrackSessionCurrent(sessionId)) {
                autoTrackDiagnostics.recordStaleDrop({
                    requestSessionId: sessionId,
                    currentSessionId: this.autoTrackSessionId,
                    jobFrameIndex: objectState.frame,
                    stage: 'before_valid_save',
                });
                return 'stale';
            }
            objectState.outside = false;
            objectState.keyframe = true;
            objectState.points = shape.points;
            autoTrackDiagnostics.emit('shape_commit_start', { clientID: trackedShape.clientID });
            const saveStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
            await objectState.save();
            this.noteAutoTrackCommittedSave();
            autoTrackDiagnostics.patchCurrentFrame({
                shapeSaveMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - saveStart,
            });
            autoTrackDiagnostics.emit('shape_commit_end', { clientID: trackedShape.clientID });
            autoTrackDiagnostics.recordCommittedShape(shape.points);
            autoTrackDiagnostics.recordValidation('valid', null, true);
            trackedShape.serverlessState = state;
            trackedShape.shapePoints = shape.points;
            trackedShape.autoTrackHistory = recordValidAutoTrackHistory(history, shape.points);
            return 'valid';
        }

        const nextHistory = recordInvalidAutoTrackHistory(history);
        trackedShape.autoTrackHistory = nextHistory;

        if (shouldConfirmAutoTrackLoss(verdict, history)) {
            if (!this.isAutoTrackSessionCurrent(sessionId)) {
                autoTrackDiagnostics.recordStaleDrop({
                    requestSessionId: sessionId,
                    currentSessionId: this.autoTrackSessionId,
                    jobFrameIndex: objectState.frame,
                    stage: 'before_validation_stop',
                });
                return 'stale';
            }
            autoTrackDiagnostics.recordValidation(verdict, 'validation_stop', false);
            const reason = verdict === 'hard_invalid' ?
                'Implausible ball box (large jump/size change). Auto Track stopped without marking outside.' :
                'Ball box failed validation repeatedly. Auto Track stopped without marking outside.';
            this.stopAutoTrackWithoutOutside(reason, 'geometry_validation_stop');
            return 'stopped';
        }

        autoTrackDiagnostics.recordValidation(verdict, 'soft_skip', false);
        return 'soft_skip';
    }
    /* eslint-enable react/sort-comp */

    private contextmenuDisabler = (e: MouseEvent): void => {
        if (
            e.target &&
            (e.target as Element).classList &&
            (e.target as Element).classList.toString().includes('ant-modal')
        ) {
            e.preventDefault();
        }
    };

    private cancelListener = async (): Promise<void> => {
        const { fetching, mode } = this.state;
        if (fetching) {
            // user pressed ESC
            this.setState({ fetching: false });
            this.interaction.isAborted = true;
        }
        if (mode === 'seed') {
            if (this.suppressNextSeedCancel) {
                // Our own frame-change overlay reset; collection stays active.
                return;
            }
            this.stopSeedCollection();
            return;
        }
        if (mode === 'tracking') {
            if (this.suppressNextTrackingCancel) {
                this.suppressNextTrackingCancel = false;
                return;
            }
            if (this.autoTrackSessionActive) {
                this.cancelAutoTrackSession();
                return;
            }
            this.stopAutoTrackSession();
        }
    };

    private startTrackerDraw = (autoTrack: boolean): void => {
        const { activeTracker, activeLabelID, fetching } = this.state;
        const {
            canvasInstance, frame, jobInstance, onInteractionStart, onSwitchToolsBlockerState,
        } = this.props;

        if (!activeTracker || !activeLabelID || fetching || frame === jobInstance.stopFrame) {
            return;
        }

        canvasInstance.cancel();
        if (autoTrack) {
            this.invalidateAutoTrackSession();
            this.autoTrackCommittedSaveCount = 0;
            if (autoTrackDiagnostics.isEnabled()) {
                autoTrackDiagnostics.beginSession(this.autoTrackSessionId);
            }
        }
        this.autoTrackSessionActive = autoTrack;
        this.suppressNextTrackingCancel = false;
        this.setState({ mode: 'tracking', autoTrackActive: autoTrack });
        const parameters = { command: 'draw_box' as const, settings: { crosshair: true } };
        canvasInstance.interact({ enabled: true, ...parameters });
        onInteractionStart(activeTracker, activeLabelID, parameters);
        onSwitchToolsBlockerState({ buttonVisible: false });
    };

    private runInteractionRequest = async (interactionId: string): Promise<void> => {
        const { jobInstance } = this.props;
        const { activeInteractor, fetching } = this.state;

        const { id, latestRequest } = this.interaction;
        if (id !== interactionId || !latestRequest || fetching) {
            // id !== interactionId: request not relevant anymore (new session has started)
            // !latestRequest: nothing to process
            // fetching: another request is already running
            return;
        }

        const { interactor, data } = latestRequest;
        this.interaction.latestRequest = null;

        try {
            this.interaction.closeFetchingMessage = message.loading({
                content: `Waiting for a response from ${activeInteractor?.name}`,
                duration: 0,
                className: 'cvat-tracking-notice',
            });

            try {
                // run server request
                this.setState({ fetching: true });

                await this.initializeOpenCV();
                const response = await core.lambda.call(
                    jobInstance.taskId,
                    interactor,
                    { ...data, job: jobInstance.id },
                ) as InteractorResults;

                if (this.interaction.id !== interactionId || this.interaction.isAborted) {
                    // new interaction session or the session is aborted
                    return;
                }

                const latestResponse: ToolsControlComponent['interaction']['latestResponse'] = [];
                let showConfidenceControl = false;
                for (const item of response.shapes) {
                    const polygonPoints = this.receivePointsFromMask(item.points);
                    if (polygonPoints.length < 3) {
                        continue;
                    }

                    const approximated = this.approximateResponsePoints(polygonPoints!);
                    const confidenceAttr = item.attributes.find((attr) => attr.spec_id === 0);
                    const confidence = confidenceAttr ? +confidenceAttr.value : 1;
                    showConfidenceControl = showConfidenceControl || !!confidenceAttr;
                    latestResponse.push({
                        rle: item.points,
                        points: polygonPoints,
                        approximatedPoints: approximated,
                        confidence,
                    });
                }

                this.interaction.latestResponse = latestResponse;
                this.setState({
                    interactorResponseReceived: !!latestResponse.length,
                    showConfidenceControl,
                });
            } finally {
                if (this.interaction.id === interactionId) {
                    this.interaction.closeFetchingMessage?.();
                    this.interaction.closeFetchingMessage = null;
                }

                this.setState({ fetching: false });
            }

            this.drawIntermediateShapesOnCanvas();
            setTimeout(() => this.runInteractionRequest(interactionId));
        } catch (error: any) {
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Interaction error occurred',
                duration: null,
            });
        }
    };

    private onInteraction = (e: Event): void => {
        const { frame, isActivated } = this.props;
        const { activeInteractor } = this.state;

        if (!isActivated) {
            return;
        }

        if (!this.interaction.id) {
            this.interaction.id = lodash.uniqueId('interaction_');
        }

        const { shapes } = (e as CustomEvent).detail;
        const interactor = activeInteractor as MLModel;
        const boxes = convertShapesForInteractor(shapes, 'rectangle', 'positive');
        const posPoints = convertShapesForInteractor(shapes, 'points', 'positive');
        const negPoints = convertShapesForInteractor(shapes, 'points', 'negative');
        this.interaction.latestRequest = {
            interactor,
            data: {
                frame,
                obj_bbox: boxes,
                pos_points: posPoints,
                neg_points: negPoints,
            },
        };

        this.runInteractionRequest(this.interaction.id);
    };

    private onTracking = async (e: Event): Promise<void> => {
        const { trackedShapes, activeTracker, activeLabelID } = this.state;
        const {
            isActivated, jobInstance, frame, curZOrder, fetchAnnotations,
        } = this.props;

        if ((!isActivated && !this.autoTrackSessionActive) || !activeLabelID) {
            return;
        }

        const [label] = jobInstance.labels.filter((_label: any): boolean => _label.id === activeLabelID);

        const { finished } = (e as CustomEvent).detail;
        if (!finished) {
            return;
        }

        // TODO: support more rectangles at the same time
        // OR: drop this tracking method

        try {
            const { points } = (e as CustomEvent).detail.shapes[0];
            const state = new core.classes.ObjectState({
                shapeType: ShapeType.RECTANGLE,
                objectType: ObjectType.TRACK,
                source: core.enums.Source.SEMI_AUTO,
                zOrder: curZOrder,
                label,
                points,
                frame,
                occluded: false,
                attributes: {},
                descriptions: [`Trackable (${activeTracker?.name})`],
            });

            const [clientID] = await jobInstance.annotations.put([state]);
            if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                autoTrackDiagnostics.emit('seed_committed', {
                    frame,
                    clientID,
                    geometry: boxGeometryFromPoints(points),
                });
            }
            const newTrackedShape: TrackedShape = {
                clientID,
                serverlessState: null,
                shapePoints: points,
                trackerModel: activeTracker as MLModel,
                autoTrackHistory: this.autoTrackSessionActive ?
                    createEmptyAutoTrackHistory(points) : undefined,
            };
            await new Promise<void>((resolve) => {
                this.setState({
                    trackedShapes: [...trackedShapes, newTrackedShape],
                }, resolve);
            });

            // update annotations on a canvas
            await fetchAnnotations();

            const { changeFrame } = this.props;
            if (this.autoTrackSessionActive) {
                if (frame < jobInstance.stopFrame) {
                    changeFrame(frame + 1);
                } else {
                    if (autoTrackDiagnostics.isEnabled()) {
                        autoTrackDiagnostics.endSession('session_complete', 'last_frame');
                    }
                    this.stopAutoTrackSession();
                }
            }
        } catch (error: any) {
            this.stopAutoTrackSession();
            notification.error({
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                message: 'Tracking error occurred',
                duration: null,
            });
        }
    };

    private interactionListener = async (e: Event): Promise<void> => {
        const { toolsBlockerState, isActivated, canvasInstance } = this.props;
        const { activeInteractor, mode } = this.state;

        const allowWhileAutoTrack = mode === 'tracking' && this.autoTrackSessionActive;
        if (!isActivated && !allowWhileAutoTrack) {
            return;
        }

        if (mode === 'interaction') {
            if (!activeInteractor) {
                return;
            }
            const { shapes, finished } = (e as CustomEvent<{ shapes: InteractionResult[], finished: boolean }>).detail;

            if (finished) {
                // make an object from current result
                // do not make one more request
                // prevent future requests if possible
                this.interaction.isAborted = true;
                this.interaction.latestRequest = null;
                this.constructFromLatestResponse();
            } else {
                const isRectangleRequired = activeInteractor!.params.canvas.startWithBox === true;
                const minPosPoints = activeInteractor!.params.canvas.minPosVertices ?? 0;
                const minNegPoints = activeInteractor!.params.canvas.minNegVertices ?? 0;

                const boxes = convertShapesForInteractor(shapes, 'rectangle', 'positive');
                const posPoints = convertShapesForInteractor(shapes, 'points', 'positive');
                const negPoints = convertShapesForInteractor(shapes, 'points', 'negative');

                if (isRectangleRequired && !boxes.length) {
                    // there should be at least one box to proceed
                    canvasInstance.interact({
                        enabled: true,
                        command: 'draw_box',
                        settings: { crosshair: true },
                    });
                    return;
                }

                // auto-switch to points when something is already drawn
                canvasInstance.interact({
                    enabled: true,
                    command: 'draw_points',
                    settings: { crosshair: false },
                });

                if (posPoints.length < minPosPoints || negPoints.length < minNegPoints) {
                    // there should be enough points to proceed
                    return;
                }

                // request data is enough, but it is postponed
                if (toolsBlockerState.algorithmsLocked) {
                    this.interaction.latestPostponedEvent = e;
                    return;
                }

                this.onInteraction(e);
            }
        }

        if (mode === 'tracking') {
            const { finished, shapes } = (e as CustomEvent<{
                shapes: InteractionResult[] | null;
                finished: boolean;
            }>).detail;
            if (
                this.autoTrackSessionActive &&
                !finished &&
                Array.isArray(shapes) &&
                shapes.length
            ) {
                this.suppressNextTrackingCancel = true;
                canvasInstance.interact({ enabled: false });
                return;
            }
            this.onTracking(e);
        }

        if (mode === 'seed') {
            this.onSeedDraw(e);
        }
    };

    private setActiveInteractor = (value: string): void => {
        const { interactors } = this.props;
        const [interactor] = interactors.filter((_interactor: MLModel) => _interactor.id === value);

        if (interactor.version < MIN_SUPPORTED_INTERACTOR_VERSION) {
            notification.warning({
                message: 'Interactor API is outdated',
                description: 'Probably, you should consider updating the serverless function',
            });
        }

        this.setState({
            activeInteractor: interactor,
        });
    };

    private setActiveTracker = (value: string): void => {
        const { trackers } = this.props;
        this.setState({
            activeTracker: trackers.filter((tracker: MLModel) => tracker.id === value)[0],
        });
    };

    private collectAutoTrackTelemetry(
        objectStates: ObjectState[],
        trackedShapes: TrackedShape[],
    ): {
            activeTrackedObjectCount: number;
            committedShapeCountForCurrentTrack: number;
            currentTrackKeyframeCount: null;
            currentTrackFrameSpan: number | null;
            totalAnnotationStateCount: number;
            totalVisibleCanvasShapeCount: null;
        } {
        const activeTrackedObjectCount = trackedShapes.length;
        const primaryTracked = trackedShapes[0];
        const clientState = primaryTracked ?
            objectStates.find((state) => state.clientID === primaryTracked.clientID) :
            undefined;
        const keyframes = clientState?.keyframes ?? null;
        let currentTrackFrameSpan: number | null = null;
        if (
            keyframes &&
            typeof keyframes.first === 'number' &&
            typeof keyframes.last === 'number'
        ) {
            currentTrackFrameSpan = keyframes.last - keyframes.first + 1;
        }
        return {
            activeTrackedObjectCount,
            committedShapeCountForCurrentTrack: this.autoTrackCommittedSaveCount,
            currentTrackKeyframeCount: null,
            currentTrackFrameSpan,
            totalAnnotationStateCount: objectStates.length,
            totalVisibleCanvasShapeCount: null,
        };
    }

    private drawIntermediateShapesOnCanvas(): void {
        const { canvasInstance } = this.props;
        const { convertMasksToPolygons, thresholdValue } = this.state;
        const shapesToBeDrawn = this.interaction.latestResponse
            .filter(({ confidence }) => typeof confidence !== 'number' || confidence >= thresholdValue)
            .filter(({ approximatedPoints }) => !convertMasksToPolygons || approximatedPoints.length >= 3)
            .map(({ rle, approximatedPoints }) => ({
                shapeType: convertMasksToPolygons ? ShapeType.POLYGON : ShapeType.MASK,
                points: convertMasksToPolygons ? approximatedPoints.flat() : rle,
            }));

        canvasInstance.interact({
            enabled: true,
            command: 'put_shapes',
            payload: {
                shapes: shapesToBeDrawn,
            },
        });

        if (!shapesToBeDrawn.length) {
            if (!this.interaction.noShapesMessage) {
                this.interaction.noShapesMessage = message.info({
                    content: 'No shapes to display',
                    duration: 0,
                });
            }
        } else if (this.interaction.noShapesMessage) {
            this.interaction.noShapesMessage();
            this.interaction.noShapesMessage = null;
        }
    }

    private collectTrackerPortals(): React.ReactPortal[] {
        const { states, fetchAnnotations } = this.props;
        const { trackedShapes, activeTracker } = this.state;

        const trackedClientIDs = trackedShapes.map((trackedShape: TrackedShape) => trackedShape.clientID);
        const portals = !activeTracker ?
            [] :
            states
                .filter((objectState) => objectState.objectType === 'track' && objectState.shapeType === 'rectangle')
                .map((objectState: any): React.ReactPortal | null => {
                    const { clientID } = objectState;
                    const selectorID = `#cvat-objects-sidebar-state-item-${clientID}`;
                    let targetElement = window.document.querySelector(
                        `${selectorID} .cvat-object-item-button-prev-keyframe`,
                    ) as HTMLElement;

                    const isTracked = trackedClientIDs.includes(clientID);
                    if (targetElement) {
                        targetElement = targetElement.parentElement?.parentElement as HTMLElement;
                        return ReactDOM.createPortal(
                            <Col>
                                {isTracked ? (
                                    <CVATTooltip overlay='Disable tracking'>
                                        <EnvironmentFilled
                                            onClick={() => {
                                                const filteredStates = trackedShapes.filter(
                                                    (trackedShape: TrackedShape) => trackedShape.clientID !== clientID,
                                                );
                                                /* eslint no-param-reassign: ["error", { "props": false }] */
                                                objectState.descriptions = [];
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: filteredStates,
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                ) : (
                                    <CVATTooltip overlay={`Enable tracking using ${activeTracker.name}`}>
                                        <EnvironmentOutlined
                                            onClick={() => {
                                                objectState.descriptions = [`Trackable (${activeTracker.name})`];
                                                objectState.keyframe = true;
                                                objectState.save().then(() => {
                                                    this.setState({
                                                        trackedShapes: [
                                                            ...trackedShapes,
                                                            {
                                                                clientID,
                                                                serverlessState: null,
                                                                shapePoints: objectState.points,
                                                                trackerModel: activeTracker,
                                                            },
                                                        ],
                                                    });
                                                    fetchAnnotations();
                                                });
                                            }}
                                        />
                                    </CVATTooltip>
                                )}
                            </Col>,
                            targetElement,
                        );
                    }

                    return null;
                })
                .filter((portal: ReactPortal | null) => portal !== null);

        return portals as React.ReactPortal[];
    }

    private async checkTrackedStates(prevProps: Props): Promise<void> {
        const {
            frame,
            jobInstance,
            states: objectStates,
            trackers,
            fetchAnnotations,
            switchNavigationBlocked,
        } = this.props;
        const { trackedShapes } = this.state;
        let shouldAutoAdvance = false;
        const lostClientIDs: number[] = [];
        let timelineSkippedFrame = false;

        let withServerRequest = false;

        type AccumulatorType = {
            // These maps are indexed by tracker ID.
            stateful: Map<string | number, {
                clientIDs: number[];
                states: any[];
                shapes: MinimalShape[];
            }>;
            stateless: Map<string | number, {
                clientIDs: number[];
                shapes: MinimalShape[];
            }>;
        };

        if (prevProps.frame !== frame && trackedShapes.length) {
            const { autoTrackSessionId } = this;
            // 1. find all trackable objects on the current frame
            // 2. divide them into two groups: with relevant state, without relevant state
            const trackingData = trackedShapes.reduce<AccumulatorType>(
                (acc: AccumulatorType, trackedShape: TrackedShape): AccumulatorType => {
                    const {
                        serverlessState, shapePoints, clientID, trackerModel,
                    } = trackedShape;
                    const clientState = objectStates.find((_state): boolean => _state.clientID === clientID);
                    const keyframes = clientState?.keyframes;

                    if (
                        !clientState || !keyframes ||
                        keyframes?.prev !== frame - 1 ||
                        (typeof keyframes?.last === 'number' && keyframes?.last >= frame)
                    ) {
                        return acc;
                    }

                    const needsInit = serverlessState === null;
                    const trackableOnFrame = clientState && (
                        !clientState.outside || (needsInit && keyframes?.prev === frame - 1)
                    );

                    if (trackableOnFrame) {
                        const points = (needsInit ? shapePoints : clientState.points) as number[];
                        withServerRequest = true;
                        const stateIsRelevant =
                            serverlessState !== null &&
                            points.length === shapePoints.length &&
                            points.every((coord: number, i: number) => coord === shapePoints[i]);
                        if (stateIsRelevant) {
                            const container = acc.stateful.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                                states: [],
                            };
                            container.clientIDs.push(clientID);
                            container.shapes.push({ type: clientState.shapeType, points });
                            container.states.push(serverlessState);
                            acc.stateful.set(trackerModel.id, container);
                        } else {
                            const container = acc.stateless.get(trackerModel.id) ?? {
                                clientIDs: [],
                                shapes: [],
                            };
                            container.clientIDs.push(clientID);
                            container.shapes.push({ type: clientState.shapeType, points });
                            acc.stateless.set(trackerModel.id, container);
                        }
                    }

                    return acc;
                },
                {
                    stateful: new Map(),
                    stateless: new Map(),
                },
            );

            try {
                if (withServerRequest) {
                    switchNavigationBlocked(true);
                }
                if (this.autoTrackSessionActive && withServerRequest && autoTrackDiagnostics.isEnabled()) {
                    const telemetry = this.collectAutoTrackTelemetry(objectStates, trackedShapes);
                    autoTrackDiagnostics.beginFrame({
                        jobFrameIndex: frame,
                        sourceFrameIdentifier: String(frame),
                        sourceFrameDelta: frame - prevProps.frame,
                        annotationStateCount: objectStates.length,
                        ...telemetry,
                    });
                }
                // 3. get relevant state for the second group
                for (const [trackerID, trackableObjects] of trackingData.stateless) {
                    let hideMessage = null;
                    try {
                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: states are being initialized for ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects' : 'object'
                            } ..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });

                        const initRequestId = createAutoTrackRequestId();
                        const response = await core.lambda.call(jobInstance.taskId, tracker, {
                            type: 'init_tracking',
                            frame: frame - 1,
                            shapes: trackableObjects.shapes,
                            job: jobInstance.id,
                            ...this.buildAutoTrackDiagPayload(
                                autoTrackSessionId,
                                frame,
                                'init_tracking',
                                initRequestId,
                            ),
                        }) as TrackerResults;

                        if (
                            this.autoTrackSessionActive &&
                            !this.isAutoTrackSessionCurrent(autoTrackSessionId)
                        ) {
                            continue;
                        }

                        if (
                            this.autoTrackSessionActive &&
                            trackableObjects.clientIDs.length === 1
                        ) {
                            this.tryIngestG0PreloadTimeline(tracker, response);
                        }

                        const { states: serverlessStates } = response;
                        const statefulContainer = trackingData.stateful.get(trackerID) ?? {
                            clientIDs: [],
                            shapes: [],
                            states: [],
                        };

                        Array.prototype.push.apply(statefulContainer.clientIDs, trackableObjects.clientIDs);
                        Array.prototype.push.apply(statefulContainer.shapes, trackableObjects.shapes);
                        Array.prototype.push.apply(statefulContainer.states, serverlessStates);
                        trackingData.stateful.set(trackerID, statefulContainer);
                        trackingData.stateless.delete(trackerID);
                    } catch (error: any) {
                        notification.error({
                            message: 'Tracker initialization error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                    }
                }

                for (const [trackerID, trackableObjects] of trackingData.stateful) {
                    // 4. run tracking for all the objects
                    let hideMessage = null;
                    try {
                        const [tracker] = trackers.filter((_tracker: MLModel) => _tracker.id === trackerID);
                        if (!tracker) {
                            throw new Error(`Suitable tracker with ID ${trackerID} not found in tracker list`);
                        }

                        const numOfObjects = trackableObjects.clientIDs.length;
                        hideMessage = message.loading({
                            content: `${tracker.name}: ${numOfObjects} ${
                                numOfObjects > 1 ? 'objects are' : 'object is'
                            } being tracked..`,
                            duration: 0,
                            className: 'cvat-tracking-notice',
                        });
                        let response: TrackerResults | undefined;

                        if (this.shouldUseLocalG0Timeline(tracker, numOfObjects)) {
                            const summary = this.autoTrackPreloadSummary;
                            if (summary && frame > summary.preloaded_until_frame) {
                                // eslint-disable-next-line no-await-in-loop
                                await this.handleTrackingWindowComplete(
                                    autoTrackSessionId,
                                    trackableObjects,
                                    objectStates,
                                    trackedShapes,
                                    lostClientIDs,
                                );
                                break;
                            }
                            const localResponse = this.buildLocalTimelineTrackerResponse(
                                frame,
                                trackableObjects.states[0],
                            );
                            if (localResponse === null) {
                                this.stopAutoTrackWithoutOutside(
                                    'Preload timeline entry missing or corrupt for this frame. Auto Track stopped.',
                                    'timeline_entry_missing',
                                );
                                break;
                            }
                            if (localResponse === 'skipped') {
                                timelineSkippedFrame = true;
                                if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                                    autoTrackDiagnostics.patchCurrentFrame({
                                        resultSource: 'g0_timeline_skipped',
                                        lambdaRoundTripMs: 0,
                                    });
                                }
                            } else {
                                response = localResponse;
                                if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                                    autoTrackDiagnostics.patchCurrentFrame({
                                        resultSource: 'g0_timeline_local',
                                        lambdaRoundTripMs: 0,
                                    });
                                }
                            }
                        }

                        if (!timelineSkippedFrame) {
                            if (response === undefined) {
                                // eslint-disable-next-line no-await-in-loop
                                const trackRequestId = createAutoTrackRequestId();
                                response = await core.lambda.call(jobInstance.taskId, tracker, {
                                    type: 'track',
                                    frame,
                                    states: trackableObjects.states,
                                    job: jobInstance.id,
                                    ...this.buildAutoTrackDiagPayload(
                                        autoTrackSessionId,
                                        frame,
                                        'track',
                                        trackRequestId,
                                    ),
                                }) as TrackerResults;
                                if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                                    autoTrackDiagnostics.patchCurrentFrame({
                                        resultSource: 'track_http',
                                    });
                                }
                            }

                            if (
                                this.autoTrackSessionActive &&
                                !this.isAutoTrackSessionCurrent(autoTrackSessionId)
                            ) {
                                break;
                            }

                            if (response.tracking_status === 'preload_exhausted') {
                                // eslint-disable-next-line no-await-in-loop
                                await this.handleTrackingWindowComplete(
                                    autoTrackSessionId,
                                    trackableObjects,
                                    objectStates,
                                    trackedShapes,
                                    lostClientIDs,
                                );
                                break;
                            }

                            response.shapes = response.shapes.map(trackedRectangleMapper);
                        }

                        const updatedTrackedShapes = [...trackedShapes];
                        if (!timelineSkippedFrame) {
                            if (response === undefined) {
                                throw new Error('Tracker response is missing for a non-skipped timeline frame');
                            }
                            for (let i = 0; i < trackableObjects.clientIDs.length; i++) {
                                const clientID = trackableObjects.clientIDs[i];
                                const shape = response.shapes[i];
                                const state = response.states[i];
                                const [objectState] = objectStates.filter(
                                    (_state: any): boolean => _state.clientID === clientID,
                                );
                                const trackedShapeIndex = updatedTrackedShapes.findIndex(
                                    (_trackedShape: TrackedShape) => _trackedShape.clientID === clientID,
                                );
                                const trackedShape = trackedShapeIndex >= 0 ?
                                    updatedTrackedShapes[trackedShapeIndex] : null;
                                if (!objectState || !trackedShape) {
                                    continue;
                                }
                                if (this.autoTrackSessionActive) {
                                    // eslint-disable-next-line no-await-in-loop
                                    const result = await this.processAutoTrackShapeResult(
                                        autoTrackSessionId,
                                        tracker,
                                        response,
                                        shape,
                                        state,
                                        objectState,
                                        trackedShape,
                                        lostClientIDs,
                                    );
                                    if (result === 'stale' || result === 'stopped' || result === 'lost') {
                                        break;
                                    }
                                } else if (shape === null || shape === undefined) {
                                    objectState.outside = true;
                                    // eslint-disable-next-line no-await-in-loop
                                    await objectState.save();
                                    lostClientIDs.push(clientID);
                                } else {
                                    objectState.outside = false;
                                    objectState.keyframe = true;
                                    objectState.points = shape.points;
                                    // eslint-disable-next-line no-await-in-loop
                                    await objectState.save();
                                    trackedShape.serverlessState = state;
                                    trackedShape.shapePoints = shape.points;
                                }
                            }
                        }
                        if (!timelineSkippedFrame) {
                            this.setState({ trackedShapes: updatedTrackedShapes });
                        }
                    } catch (error: any) {
                        if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                            autoTrackDiagnostics.endSession('session_error', error.message);
                        }
                        this.stopAutoTrackSession();
                        notification.error({
                            message: 'Tracking error',
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            duration: null,
                        });
                    } finally {
                        if (hideMessage) hideMessage();
                    }
                }

                const fetchStart = typeof performance !== 'undefined' ? performance.now() : Date.now();
                if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                    autoTrackDiagnostics.emit('annotations_fetch_start', { frame });
                }
                await fetchAnnotations();
                if (this.autoTrackSessionActive && autoTrackDiagnostics.isEnabled()) {
                    autoTrackDiagnostics.patchCurrentFrame({
                        annotationsFetchMs: (typeof performance !== 'undefined' ? performance.now() : Date.now()) - fetchStart,
                        annotationStateCount: objectStates.length,
                        fullStateReplacement: true,
                    });
                    autoTrackDiagnostics.emit('annotations_fetch_end', { frame });
                    autoTrackDiagnostics.emit('reducer_or_store_update_end', {
                        frame,
                        annotationStateCount: objectStates.length,
                        note: 'redux_update_observed_on_next_render',
                    });
                }

                if (lostClientIDs.length) {
                    this.setState({
                        trackedShapes: trackedShapes.filter(
                            (tracked: TrackedShape) => !lostClientIDs.includes(tracked.clientID),
                        ),
                    });
                } else if (
                    this.autoTrackSessionActive &&
                    this.isAutoTrackSessionCurrent(autoTrackSessionId)
                ) {
                    shouldAutoAdvance = true;
                }

                if (
                    timelineSkippedFrame &&
                    this.autoTrackSessionActive &&
                    this.isAutoTrackSessionCurrent(autoTrackSessionId) &&
                    !lostClientIDs.length
                ) {
                    shouldAutoAdvance = true;
                }

                if (
                    this.autoTrackSessionActive &&
                    withServerRequest &&
                    autoTrackDiagnostics.isEnabled()
                ) {
                    autoTrackDiagnostics.endFrame({
                        cancelled: !this.isAutoTrackSessionCurrent(autoTrackSessionId),
                        stopReason: lostClientIDs.length ? 'lost' : null,
                        pendingPromiseCount: null,
                        pendingQueueLength: null,
                    });
                }
            } finally {
                if (withServerRequest) {
                    switchNavigationBlocked(false);
                }
                if (
                    shouldAutoAdvance &&
                    this.autoTrackSessionActive &&
                    this.isAutoTrackSessionCurrent(autoTrackSessionId)
                ) {
                    const { changeFrame } = this.props;
                    if (frame < jobInstance.stopFrame) {
                        changeFrame(frame + 1);
                    } else {
                        if (autoTrackDiagnostics.isEnabled()) {
                            autoTrackDiagnostics.endSession('session_complete', 'last_frame');
                        }
                        this.stopAutoTrackSession();
                    }
                }
            }
        }
    }

    /* eslint-disable react/sort-comp -- Keep the multi-rally lifecycle together for auditability. */
    private armSeedDraw = (): void => {
        const { canvasInstance, onInteractionStart } = this.props;
        const { activeTracker, activeLabelID } = this.state;
        if (!this.seedCollecting || !activeTracker || !activeLabelID) {
            return;
        }
        // The seed draw is the only interaction that opts into frame navigation:
        // the operator must be able to move between rally starts without leaving
        // Collect Seeds. Track Rallies staging/commit sets `navigationBlocked`,
        // which stays authoritative, and no unrelated interaction sets this flag.
        const parameters = {
            command: 'draw_box' as const,
            settings: { crosshair: true, allowFrameChange: true },
        };
        try {
            canvasInstance.interact({ enabled: true, ...parameters });
            onInteractionStart(activeTracker, activeLabelID, parameters);
        } catch (error: any) {
            // Never swallow an arm failure: leave seed mode and surface it.
            this.stopSeedCollection();
            notification.error({
                message: 'Rally seed mode stopped',
                description: (
                    <CVATMarkdown>
                        {`The canvas rejected the seed draw tool: ${error.message}`}
                    </CVATMarkdown>
                ),
                duration: 8,
            });
        }
    };

    private startSeedCollection = (): void => {
        const { activeTracker, activeLabelID, fetching } = this.state;
        const { canvasInstance } = this.props;
        if (!activeTracker || !activeLabelID || fetching) {
            return;
        }
        this.suppressNextSeedCancel = false;
        canvasInstance.cancel();
        this.autoTrackSessionActive = false;
        this.seedCollecting = true;
        this.seedCommittedPromptCount = 0;
        this.setState({ mode: 'seed', collectingSeeds: true, autoTrackActive: false }, this.armSeedDraw);
    };

    private stopSeedCollection = (): void => {
        if (!this.seedCollecting) {
            return;
        }
        const { canvasInstance } = this.props;
        this.suppressNextSeedCancel = false;
        this.seedCollecting = false;
        if (this.mounted) {
            this.setState({ collectingSeeds: false, mode: 'interaction' });
        }
        try {
            canvasInstance.interact({ enabled: false });
        } catch {
            // The interaction was already released; nothing else to cancel.
        }
    };

    private onSeedDraw = async (e: Event): Promise<void> => {
        const { activeTracker, activeLabelID } = this.state;
        const {
            frame, curZOrder, jobInstance, fetchAnnotations,
        } = this.props;

        const { shapes } = (e as CustomEvent<{
            shapes: InteractionResult[] | null;
            finished: boolean;
        }>).detail;
        if (!this.seedCollecting || !activeTracker || !activeLabelID) {
            return;
        }

        // CVAT reports every draw_box drawstop with `finished=false` and the full
        // accumulated prompt list. The committed count is the only reliable marker:
        // claim the new length synchronously (before any await) so closely spaced
        // events cannot commit the same prompt twice, and always take the newest
        // prompt, never shapes[0].
        const decision = decideSeedPrompt(shapes?.length ?? 0, this.seedCommittedPromptCount);
        this.seedCommittedPromptCount = decision.committedCount;
        if (decision.action !== 'commit' || !shapes) {
            return;
        }

        const [label] = jobInstance.labels.filter((_label: any): boolean => _label.id === activeLabelID);
        const { points } = shapes[decision.index];
        try {
            const state = new core.classes.ObjectState({
                shapeType: ShapeType.RECTANGLE,
                objectType: ObjectType.TRACK,
                source: core.enums.Source.SEMI_AUTO,
                zOrder: curZOrder,
                label,
                points,
                frame,
                occluded: false,
                attributes: {},
            });
            const [clientID] = await jobInstance.annotations.put([state]);

            // A single-keyframe track is interpolated onto every later frame, so the
            // seed rectangle would be drawn across the whole rest of the job. The
            // native CVAT way to confine it to its seed frame is a trailing
            // outside=true keyframe at seed+1. Seeds on the last job frame cannot
            // have one and stay single-keyframe (accepted as the legacy seed shape).
            if (frame < jobInstance.stopFrame) {
                const boundaryStates = await jobInstance.annotations.get(frame + 1, true, []);
                const boundaryState = boundaryStates.find(
                    (_state: ObjectState): boolean => _state.clientID === clientID,
                );
                if (boundaryState) {
                    boundaryState.outside = true;
                    boundaryState.keyframe = true;
                    boundaryState.points = [...points];
                    await boundaryState.save();
                }
            }

            if (this.mounted) {
                this.setState((previous) => ({
                    rallySeeds: [
                        ...previous.rallySeeds.filter((seed) => seed.clientID !== clientID),
                        {
                            clientID, frame, points: [...points], labelID: activeLabelID, trackerID: activeTracker.id,
                        },
                    ],
                }));
            }
            await fetchAnnotations();
        } catch (error: any) {
            notification.error({
                message: 'Rally seed not saved',
                description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                duration: null,
            });
        } finally {
            if (this.mounted && this.seedCollecting) {
                // Stay armed so the next rally start can be seeded right away.
                this.armSeedDraw();
            }
        }
    };

    /** True while this token still owns the run and the run is not cancelling. */
    /** Pure token ownership: the owning run keeps cleanup even while cancelling. */
    private tokenOwnsMultiRallyRun = (token: number): boolean => token === this.multiRallyRunToken;

    private ownsMultiRallyRun = (token: number): boolean => (
        this.multiRallyRunActive && this.tokenOwnsMultiRallyRun(token)
    );

    private isMultiRallyRunCurrent = (token: number): boolean => (
        this.ownsMultiRallyRun(token) && this.multiRallyPhase !== 'cancelling'
    );

    /**
     * Staging and cancelling abort; a committing phase ignores a Stop request
     * because the already-validated segment must be written to completion.
     */
    private assertMultiRallyRunCurrent(token: number): void {
        if (!this.tokenOwnsMultiRallyRun(token)) {
            throw new RallyRunStop(null, 'cancelled');
        }
        if (this.multiRallyPhase === 'cancelling') {
            throw new RallyRunStop(null, 'cancelled');
        }
    }

    private stopMultiRallyRun = (): void => {
        if (!this.multiRallyRunActive) {
            return;
        }
        if (this.multiRallyPhase === 'committing') {
            // A validated segment is already being written; finish it, then stop.
            this.multiRallyCancelRequested = true;
            if (this.mounted) {
                this.setState({ rallyStatus: 'Stopping after this rally…' });
            }
            return;
        }
        // Staging (or between segments): abort now, but keep the run owned and the UI
        // busy until the in-flight operation reaches its finally. No new run may start.
        this.multiRallyCancelRequested = true;
        this.multiRallyPhase = 'cancelling';
        if (this.mounted) {
            this.setState({ rallyStatus: 'Stopping…' });
        }
    };

    private updateRallyStatus(context: RallyStageContext, frame: number): void {
        if (!this.mounted || !this.isMultiRallyRunCurrent(context.token)) {
            return;
        }
        this.setState({
            rallyStatus: `Rally ${context.segment.index + 1}/${context.segmentCount} · frame ${frame}`,
        });
    }

    private describeRallyPreflightFailure(frame: number | null, reason: string): string {
        const where = frame === null ? '' : ` (frame ${frame})`;
        switch (reason) {
            case 'no_seeds_for_active_label_tracker':
                return 'No rally seed rectangle tracks found for the active label. Use Collect Seeds first.';
            case 'duplicate_seed_frame':
                return `Two seed tracks share the same frame${where}.`;
            case 'deleted_seed_frame':
                return `A seed is on a deleted frame${where}.`;
            case 'invalid_seed_object':
                return `A seed is not an unlocked, non-outside, single-keyframe rectangle${where}.`;
            case 'invalid_seed_boundary':
                return [
                    `A seed has an extra or unverified keyframe after it${where}.`,
                    'Expected nothing, or exactly one outside boundary at seed+1.',
                ].join(' ');
            case 'unexpected_seed_frame':
                return `A seed track sits where the source file names show no rally start${where}.`;
            case 'deleted_frame_in_range':
                return `A requested rally frame is deleted${where}.`;
            case 'empty_frame_range':
                return 'The job reports no frames.';
            case 'filename_unparsable':
                return `A job frame name is not frame_<number>.png${where}. The full filename contract is required.`;
            case 'frame_range_mismatch':
                return `Frame metadata does not match the job range${where}.`;
            default:
                return `Rally preflight failed${where} (${reason}).`;
        }
    }

    private describeRallyStop(frame: number | null, reason: string): string {
        const where = frame === null ? '' : ` at frame ${frame}`;
        const discarded = 'The current rally was discarded before writing.';
        switch (reason) {
            case 'target_lost':
                return `Tracker lost the ball${where}. ${discarded}`;
            case 'frame_skipped':
                return `Tracker skipped a frame${where}. ${discarded}`;
            case 'frame_missing':
                return `No tracker result for the frame${where}. ${discarded}`;
            case 'deleted_frame':
                return `Frame${where} is deleted. ${discarded}`;
            case 'unusable_box':
                return `Tracker returned an unusable box${where}. ${discarded}`;
            case 'geometry_rejected':
                return `Box failed geometry validation${where}. ${discarded}`;
            case 'premature_source_gap':
                return `Source frame numbering jumps before the next rally boundary${where}. ${discarded}`;
            case 'premature_job_end':
                return `Tracker preload ended before the next rally boundary${where}. ${discarded}`;
            case 'boundary_mismatch':
                return `Tracker preload missed the expected rally boundary${where}. ${discarded}`;
            case 'continuity_unverified':
                return `Source frame numbering could not be verified${where}. ${discarded}`;
            case 'zero_progress':
                return `Tracker produced no new frames${where}. ${discarded}`;
            case 'seed_missing_before_commit':
                return `The seed track disappeared before commit${where}. Nothing was written.`;
            case 'seed_changed_before_commit':
                return `The seed track changed before commit${where}. Nothing was written.`;
            case 'seed_points_changed':
                return `The seed box was edited before commit${where}. Nothing was written.`;
            case 'timeline_unusable':
                return `Tracker preload timeline was unusable${where}. Nothing was written.`;
            default:
                return `Tracking stopped${where} (${reason}). ${discarded}`;
        }
    }

    private describeRallyPartialCommit(committedThrough: number, frame: number | null): string {
        const where = frame === null ? '' : ` Frame ${frame} failed.`;
        return [
            `Track Rallies stopped during commit.${where}`,
            `Partial annotations may exist on the current rally up to frame ${committedThrough}.`,
            'Review them and undo if needed; completed rallies are kept.',
        ].join(' ');
    }

    private buildJobFrameRange(meta: Awaited<ReturnType<typeof getFramesMeta>>): number[] {
        const frames: number[] = [];
        const step = Math.max(1, meta.frameStep);
        for (let frame = meta.startFrame; frame <= meta.stopFrame; frame += step) {
            frames.push(frame);
        }
        return frames;
    }

    /**
     * Enumerates every active-label rectangle track of the job (one `allTracks` read
     * returns all tracks) and then re-reads each candidate at its own first keyframe.
     * When the first keyframe is followed by a keyframe at seed+1, that trailing
     * state is read too so preflight can verify the outside boundary without ever
     * trusting interpolated facts.
     *
     * `annotations.get(frame, true, [])` interpolates: a track whose first keyframe is
     * after `frame` comes back with `outside=true` and interpolated points, so facts
     * taken from the enumeration read are never reused. Descriptions are not used as a
     * marker (they do not survive `Track.toJSON`/reload); every active-label rectangle
     * track is returned so preflight can reject duplicate/extra/invalid candidates.
     */
    private async discoverRallySeeds(
        tracker: MLModel,
        labelID: number,
        snapshotFrame: number,
    ): Promise<RallySeedCandidate[]> {
        const { jobInstance } = this.props;
        const snapshot = await jobInstance.annotations.get(snapshotFrame, true, []);
        const candidates: RallySeedCandidate[] = [];

        for (const state of snapshot) {
            if (String(state.objectType) !== 'track' || String(state.shapeType) !== 'rectangle') {
                continue;
            }
            if (state.label?.id !== labelID || state.clientID === null) {
                continue;
            }
            const seedFrame = state.keyframes?.first;
            if (typeof seedFrame !== 'number' ||
                seedFrame < jobInstance.startFrame || seedFrame > jobInstance.stopFrame) {
                continue;
            }

            const ownStates = await jobInstance.annotations.get(seedFrame, true, []);
            const ownState = ownStates.find((_state: ObjectState): boolean => _state.clientID === state.clientID);
            const ownLastKeyframe = ownState?.keyframes?.last ?? null;
            // Only a track that declares a keyframe at seed+1 needs its boundary read.
            let boundaryState: ObjectState | undefined;
            if (ownLastKeyframe === seedFrame + 1 && seedFrame + 1 <= jobInstance.stopFrame) {
                const boundaryStates = await jobInstance.annotations.get(seedFrame + 1, true, []);
                boundaryState = boundaryStates.find(
                    (_state: ObjectState): boolean => _state.clientID === state.clientID,
                );
            }
            candidates.push(buildRallySeedCandidate(
                state.clientID,
                seedFrame,
                labelID,
                tracker.id,
                ownState ? {
                    points: [...ownState.points],
                    objectType: String(ownState.objectType),
                    shapeType: String(ownState.shapeType),
                    outside: !!ownState.outside,
                    lock: !!ownState.lock,
                    keyframeFirst: ownState.keyframes?.first ?? null,
                    keyframeLast: ownLastKeyframe,
                    boundaryFrame: boundaryState ? seedFrame + 1 : null,
                    boundaryOutside: boundaryState ? !!boundaryState.outside : null,
                    boundaryPoints: boundaryState ? [...boundaryState.points] : null,
                } : null,
            ));
        }

        return candidates;
    }

    private async runRallySegment(context: RallyStageContext): Promise<void> {
        const { jobInstance } = this.props;
        const write = rallySegmentWriteFrames(context.segment, jobInstance.stopFrame);
        if (!write.keyframes.length && write.boundary === null) {
            // One-frame final rally: the seed already covers it.
            return;
        }
        if (write.keyframes.length) {
            this.multiRallyPhase = 'staging';
            await this.stageRallySegment(context);
            this.assertMultiRallyRunCurrent(context.token);
            const stagedFrames = context.staged.map((entry) => entry.frame).join(',');
            if (stagedFrames !== write.keyframes.join(',')) {
                throw new RallyRunStop(context.segment.anchor, 'zero_progress');
            }
        }
        // The whole segment validated; only now do we write anything.
        this.multiRallyPhase = 'committing';
        await this.commitRallySegment(context);
    }

    private async stageRallySegment(context: RallyStageContext): Promise<void> {
        await this.stageRallySegmentFromTimeline(context);
    }

    private checkRallyGeometry(context: RallyStageContext, frame: number, points: number[]): void {
        const shape: MinimalShape = { type: ShapeType.RECTANGLE, points };
        if (isShapeUnusable(shape)) {
            throw new RallyRunStop(frame, 'unusable_box');
        }
        const verdict = validateAutoTrackCandidate(points, context.history);
        if (verdict !== 'valid') {
            throw new RallyRunStop(frame, 'geometry_rejected');
        }
    }

    private stageFrame(context: RallyStageContext, frame: number, points: number[]): void {
        if (context.isFrameDeleted(frame)) {
            throw new RallyRunStop(frame, 'deleted_frame');
        }
        this.checkRallyGeometry(context, frame, points);
        context.staged.push({ frame, points: [...points] });
        context.lastStagedPoints = [...points];
        context.history = recordValidAutoTrackHistory(context.history, points);
        this.updateRallyStatus(context, frame);
    }

    private async stageRallySegmentFromTimeline(context: RallyStageContext): Promise<void> {
        const { jobInstance } = this.props;
        const { token, tracker, segment } = context;
        let { anchor } = segment;
        let anchorPoints = [...context.seedPoints];

        for (;;) {
            this.assertMultiRallyRunCurrent(token);
            const response = await core.lambda.call(jobInstance.taskId, tracker, {
                type: 'init_tracking',
                frame: anchor,
                shapes: [{ type: ShapeType.RECTANGLE, points: anchorPoints }],
                job: jobInstance.id,
            }) as TrackerResults;
            this.assertMultiRallyRunCurrent(token);

            const validated = validatePreloadTimeline(response);
            if (!validated.valid) {
                throw new RallyRunStop(anchor, 'timeline_unusable');
            }
            const { summary, timeline } = validated;
            // The preload must prove source continuity before any coverage success.
            if (summary.source_timeline_verified !== true || summary.stop_reason === 'continuity_unverified') {
                throw new RallyRunStop(anchor, 'continuity_unverified');
            }
            const coverageEnd = summary.preloaded_until_frame;
            const targetEnd = segment.endExclusive - 1;
            const stageUntil = Math.min(coverageEnd, targetEnd);

            let progressed = false;
            for (const entry of timeline) {
                this.assertMultiRallyRunCurrent(token);
                if (entry.frame <= anchor || entry.frame > stageUntil) {
                    continue;
                }
                if (entry.bbox_status === 'skipped') {
                    throw new RallyRunStop(entry.frame, 'frame_skipped');
                }
                if (entry.bbox_status === 'lost' || entry.lost) {
                    throw new RallyRunStop(entry.frame, 'target_lost');
                }
                if (entry.bbox_status !== 'valid' || !entry.points) {
                    throw new RallyRunStop(entry.frame, 'frame_missing');
                }
                this.stageFrame(context, entry.frame, entry.points);
                progressed = true;
            }

            if (coverageEnd >= targetEnd) {
                if (coverageEnd !== targetEnd) {
                    // Preload ran past a boundary the filename contract requires.
                    throw new RallyRunStop(coverageEnd + 1, 'boundary_mismatch');
                }
                // The expected gap at the segment end is the only accepted ending.
                return;
            }
            if (!progressed) {
                throw new RallyRunStop(coverageEnd + 1, 'zero_progress');
            }
            if (summary.stop_reason === 'chunk_cap') {
                // Chain from the last validated frame with no overlap.
                const [last] = context.staged.slice(-1);
                anchor = last.frame;
                anchorPoints = [...last.points];
                continue;
            }
            if (summary.stop_reason === 'source_timeline_gap') {
                throw new RallyRunStop(coverageEnd + 1, 'premature_source_gap');
            }
            throw new RallyRunStop(coverageEnd + 1, 'premature_job_end');
        }
    }

    private async fetchRallyTrackState(clientID: number, frame: number): Promise<ObjectState | null> {
        const { jobInstance } = this.props;
        const states = await jobInstance.annotations.get(frame, true, []);
        return states.find((_state: ObjectState): boolean => _state.clientID === clientID) ?? null;
    }

    private async commitRallySegment(context: RallyStageContext): Promise<void> {
        const { jobInstance, fetchAnnotations } = this.props;
        const {
            token, segment, seedClientID, seedPoints, staged,
        } = context;
        const write = rallySegmentWriteFrames(segment, jobInstance.stopFrame);
        let committedThrough: number | null = null;

        try {
            // Re-read the live seed immediately before commit; cached seeds are never trusted.
            const seedState = await this.fetchRallyTrackState(seedClientID, segment.anchor);
            this.assertMultiRallyRunCurrent(token);
            if (!seedState) {
                throw new RallyRunStop(segment.anchor, 'seed_missing_before_commit');
            }
            const seedKeyframes = seedState.keyframes;
            if (String(seedState.objectType) !== 'track' ||
                String(seedState.shapeType) !== 'rectangle' ||
                seedState.label?.id !== context.labelID) {
                throw new RallyRunStop(segment.anchor, 'seed_changed_before_commit');
            }
            // The live seed may carry the trailing outside boundary written by
            // Collect Seeds; read it so the shape check verifies it fail-closed.
            let seedBoundaryState: ObjectState | null = null;
            if (seedKeyframes?.last === segment.anchor + 1) {
                seedBoundaryState = await this.fetchRallyTrackState(seedClientID, segment.anchor + 1);
                this.assertMultiRallyRunCurrent(token);
            }
            const seedShape = verifyRallySeedShape({
                frame: segment.anchor,
                points: seedState.points,
                objectType: String(seedState.objectType),
                shapeType: String(seedState.shapeType),
                outside: !!seedState.outside,
                lock: !!seedState.lock,
                keyframeFirst: seedKeyframes?.first ?? null,
                keyframeLast: seedKeyframes?.last ?? null,
                boundaryFrame: seedBoundaryState ? segment.anchor + 1 : null,
                boundaryOutside: seedBoundaryState ? !!seedBoundaryState.outside : null,
                boundaryPoints: seedBoundaryState ? [...seedBoundaryState.points] : null,
            }, jobInstance.stopFrame);
            if (!seedShape.ok) {
                throw new RallyRunStop(segment.anchor, 'seed_changed_before_commit');
            }
            if (!sameRallyPoints(seedState.points, seedPoints)) {
                throw new RallyRunStop(segment.anchor, 'seed_points_changed');
            }

            for (let index = 0; index < write.keyframes.length; index++) {
                const frame = write.keyframes[index];
                this.assertMultiRallyRunCurrent(token);
                const objectState = await this.fetchRallyTrackState(seedClientID, frame);
                this.assertMultiRallyRunCurrent(token);
                if (!objectState) {
                    throw new RallyRunStop(frame, 'state_missing_on_commit');
                }
                objectState.outside = false;
                objectState.keyframe = true;
                objectState.points = staged[index].points;
                this.assertMultiRallyRunCurrent(token);
                await objectState.save();
                committedThrough = frame;
            }

            if (write.boundary !== null) {
                this.assertMultiRallyRunCurrent(token);
                const boundaryState = await this.fetchRallyTrackState(seedClientID, write.boundary);
                this.assertMultiRallyRunCurrent(token);
                if (!boundaryState) {
                    throw new RallyRunStop(write.boundary, 'state_missing_on_commit');
                }
                boundaryState.outside = true;
                boundaryState.keyframe = true;
                boundaryState.points = context.lastStagedPoints.length ?
                    context.lastStagedPoints : seedPoints;
                this.assertMultiRallyRunCurrent(token);
                await boundaryState.save();
                committedThrough = write.boundary;
            }

            await fetchAnnotations();
        } catch (error) {
            const stop = error instanceof RallyRunStop ? error : new RallyRunStop(null, 'commit_failed');
            stop.committedThrough = committedThrough;
            if (committedThrough !== null) {
                try {
                    // Surface whatever partial state exists so the operator can review it.
                    await fetchAnnotations();
                } catch {
                    // Refreshing annotations is best effort.
                }
            }
            throw stop;
        }
    }

    /**
     * Captures the current shared navigation block so the run restores the prior value
     * instead of a blind default, then takes it for the whole preflight/run.
     */
    private acquireMultiRallyWriteLock = (): void => {
        const { navigationBlocked, switchNavigationBlocked } = this.props;
        this.multiRallyPriorNavigationBlocked = navigationBlocked;
        switchNavigationBlocked(true);
    };

    private releaseMultiRallyWriteLock = (): void => {
        const { switchNavigationBlocked } = this.props;
        switchNavigationBlocked(this.multiRallyPriorNavigationBlocked);
    };

    private startMultiRallyRun = async (): Promise<void> => {
        const {
            jobInstance, fetchAnnotations, canvasInstance,
        } = this.props;
        const { activeLabelID, activeTracker, fetching } = this.state;
        if (!this.mounted || this.multiRallyRunActive || fetching || !activeTracker || !activeLabelID) {
            return;
        }
        if (!CVAT_UI_SAM3_G0_TIMELINE_ENABLED || activeTracker.id !== SAM3_V4_TRACKER_ID) {
            notification.warning({
                message: 'Track Rallies unavailable',
                description: 'Track Rallies requires the SAM3 v4 G0 timeline tracker.',
                duration: 8,
            });
            return;
        }

        // This run owns the token until its own finally runs (or until unmount).
        this.multiRallyRunActive = true;
        this.multiRallyRunToken += 1;
        const token = this.multiRallyRunToken;
        this.multiRallyCancelRequested = false;
        this.multiRallyPhase = 'staging';

        try {
            canvasInstance.cancel();
        } catch {
            // Nothing to cancel.
        }
        // The blocks are acquired before the discovery snapshot and held for the run.
        this.acquireMultiRallyWriteLock();
        this.setState({ rallyRunActive: true, rallyStatus: 'Preparing rally seeds…' });

        try {
            // Let the blocking overlay commit before the first annotation read.
            await new Promise<void>((resolve) => {
                window.setTimeout(resolve, 0);
            });

            let meta: Awaited<ReturnType<typeof getFramesMeta>>;
            try {
                meta = await getFramesMeta('job', jobInstance.id);
            } catch (error: any) {
                notification.error({
                    message: 'Track Rallies not started',
                    description: <CVATMarkdown>{`Frame metadata is unavailable: ${error.message}`}</CVATMarkdown>,
                    duration: 8,
                });
                return;
            }

            const frames = this.buildJobFrameRange(meta);
            if (!frames.length || frames[0] !== jobInstance.startFrame) {
                notification.warning({
                    message: 'Track Rallies not started',
                    description: (
                        <CVATMarkdown>
                            {this.describeRallyPreflightFailure(frames[0] ?? null, 'frame_range_mismatch')}
                        </CVATMarkdown>
                    ),
                    duration: 8,
                });
                return;
            }
            const rawIndexAt = (frame: number): bigint | null => {
                try {
                    return parseRawSourceIndexBig(meta.frames[meta.getFrameIndex(frame)]?.name);
                } catch {
                    return null;
                }
            };
            const isFrameDeleted = (frame: number): boolean => !!meta.deletedFrames[frame];

            const contract = computeFilenameRallyContract(frames, rawIndexAt);
            if (!contract.ok) {
                notification.warning({
                    message: 'Track Rallies not started',
                    description: (
                        <CVATMarkdown>
                            {this.describeRallyPreflightFailure(contract.frame, contract.reason)}
                        </CVATMarkdown>
                    ),
                    duration: 8,
                });
                return;
            }

            const snapshotFrame = frames.find((frame) => !isFrameDeleted(frame)) ?? frames[0];
            const candidates = await this.discoverRallySeeds(activeTracker, activeLabelID, snapshotFrame);
            this.assertMultiRallyRunCurrent(token);

            const preflight = preflightRallySeeds(candidates, {
                stopFrame: jobInstance.stopFrame,
                activeLabelID,
                activeTrackerID: activeTracker.id,
                expectedStartFrames: contract.expectedStartFrames,
                isFrameDeleted,
            });
            if (!preflight.ok) {
                notification.warning({
                    message: 'Track Rallies not started',
                    description: (
                        <CVATMarkdown>
                            {this.describeRallyPreflightFailure(preflight.frame, preflight.reason)}
                        </CVATMarkdown>
                    ),
                    duration: 8,
                });
                return;
            }

            const segmentCount = preflight.segments.length;
            const skippedCount = Math.max(0, contract.expectedStartFrames.length - segmentCount);
            const skippedSuffix = skippedCount ?
                ` · ${skippedCount} ${skippedCount > 1 ? 'starts' : 'start'} skipped` : '';
            this.setState({
                rallyStatus: `${segmentCount} ${segmentCount > 1 ? 'rallies' : 'rally'} queued${skippedSuffix}`,
            });

            for (const segment of preflight.segments) {
                if (!this.ownsMultiRallyRun(token)) {
                    return;
                }
                if (this.multiRallyCancelRequested) {
                    notification.info({
                        message: 'Track Rallies stopped',
                        description: 'Stopped. Completed rallies are kept.',
                        duration: 4,
                    });
                    return;
                }
                // Between segments nothing is being written, so staging is the safe phase.
                this.multiRallyPhase = 'staging';
                const seed = preflight.seeds[segment.index];
                const context: RallyStageContext = {
                    token,
                    tracker: activeTracker,
                    labelID: activeLabelID,
                    segment,
                    segmentCount,
                    seedClientID: seed.clientID,
                    seedPoints: [...seed.points],
                    staged: [],
                    history: createEmptyAutoTrackHistory(seed.points),
                    lastStagedPoints: [...seed.points],
                    isFrameDeleted,
                };
                await this.runRallySegment(context);
            }

            this.assertMultiRallyRunCurrent(token);
            await fetchAnnotations();
            notification.success({
                message: 'Track Rallies complete',
                description: `Tracked ${segmentCount} ${segmentCount > 1 ? 'rallies' : 'rally'}` +
                    `${skippedCount ? `; skipped ${skippedCount} ${skippedCount > 1 ? 'starts' : 'start'}` : ''}.`,
                duration: 4,
            });
        } catch (error: any) {
            if (this.ownsMultiRallyRun(token)) {
                const stop = error instanceof RallyRunStop ? error : null;
                if (stop && stop.reason === 'cancelled') {
                    notification.info({
                        message: 'Track Rallies stopped',
                        description: 'Stopped. Completed rallies are kept.',
                        duration: 4,
                    });
                } else {
                    const committedThrough = stop?.committedThrough ?? null;
                    if (committedThrough === null) {
                        notification.warning({
                            message: 'Track Rallies stopped',
                            description: (
                                <CVATMarkdown>
                                    {stop ? this.describeRallyStop(stop.frame, stop.reason) : String(error.message)}
                                </CVATMarkdown>
                            ),
                            duration: 8,
                        });
                    } else {
                        notification.error({
                            message: 'Track Rallies stopped during commit',
                            description: (
                                <CVATMarkdown>
                                    {this.describeRallyPartialCommit(committedThrough, stop?.frame ?? null)}
                                </CVATMarkdown>
                            ),
                            duration: null,
                        });
                    }
                }
            }
        } finally {
            if (this.ownsMultiRallyRun(token)) {
                this.releaseMultiRallyWriteLock();
                this.multiRallyPhase = 'idle';
                this.multiRallyCancelRequested = false;
                this.multiRallyRunActive = false;
                // Invalidate so late responses from this run write nothing.
                this.multiRallyRunToken += 1;
                if (this.mounted) {
                    this.setState({ rallyRunActive: false, rallyStatus: null });
                }
            }
        }
    };
    /* eslint-enable react/sort-comp */

    private async constructFromLatestResponse(): Promise<void> {
        const { convertMasksToPolygons, thresholdValue } = this.state;
        const {
            frame, labels, curZOrder, activeLabelID, createAnnotations,
        } = this.props;

        if (!this.interaction.latestResponse.length) {
            return;
        }

        const common = {
            frame,
            objectType: ObjectType.SHAPE,
            source: core.enums.Source.SEMI_AUTO,
            label: labels.find((label) => label.id === activeLabelID as number) as Label,
            occluded: false,
            zOrder: curZOrder,
        };

        const objectsToConstruct = this.interaction.latestResponse.filter(
            ({ confidence }) => typeof confidence !== 'number' || confidence >= thresholdValue,
        );

        let objects: ObjectState[] = [];
        if (convertMasksToPolygons) {
            objects = objectsToConstruct
                .filter(({ approximatedPoints }) => approximatedPoints.length >= 3)
                .map(({ approximatedPoints }) => (
                    new core.classes.ObjectState({
                        shapeType: ShapeType.POLYGON,
                        points: approximatedPoints.flat(),
                        ...common,
                    })
                ));
        } else {
            objects = objectsToConstruct
                .map(({ rle }) => (
                    new core.classes.ObjectState({
                        shapeType: ShapeType.MASK,
                        points: Array.from(rle),
                        ...common,
                    })
                ));
        }
        createAnnotations(objects);
    }

    private async initializeOpenCV(): Promise<void> {
        if (!openCVWrapper.isInitialized) {
            const hide = message.loading('Initializing contour utilities..', 0);
            try {
                await openCVWrapper.initialize(() => {});
            } catch (error: any) {
                notification.error({
                    message: 'Could not initialize contour utilities',
                    description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                    duration: null,
                });
            } finally {
                hide();
            }
        }
    }

    private receivePointsFromMask(mask: Int32Array): [number, number][] {
        if (!openCVWrapper.isInitialized) {
            throw new Error('OpenCV was not initialized');
        }

        const polygons = openCVWrapper.getContoursFromStateSync({ points: mask, shapeType: ShapeType.MASK });
        if (polygons.length) {
            return polygons[0].map<[number, number]>((val) => [val[0], val[1]]);
        }

        return [];
    }

    private approximateResponsePoints(points: [number, number][]): [number, number][] {
        if (!openCVWrapper.isInitialized) {
            throw new Error('OpenCV was not initialized');
        }

        const { approxPolyAccuracy } = this.state;
        if (points.length > 3) {
            const threshold = thresholdFromAccuracy(approxPolyAccuracy);
            return openCVWrapper.contours.approxPoly(points, threshold);
        }

        return points;
    }

    private renderLabelBlock(): JSX.Element {
        const { labels } = this.props;
        const { activeLabelID, collectingSeeds, rallyRunActive } = this.state;
        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Label</Text>
                    </Col>
                </Row>
                <Row justify='center'>
                    <Col span={24}>
                        <LabelSelector
                            style={{ width: '100%' }}
                            labels={labels}
                            value={activeLabelID}
                            disabled={collectingSeeds || rallyRunActive}
                            onChange={(value: any) => this.setState({ activeLabelID: value.id })}
                        />
                    </Col>
                </Row>
            </>
        );
    }

    private renderTrackerBlock(): JSX.Element {
        const {
            activeTracker, fetching, autoTrackActive, collectingSeeds, rallySeeds, rallyRunActive, rallyStatus,
        } = this.state;

        const supportedTrackers = this.getSupportedTrackers();
        const rallyBusy = collectingSeeds || rallyRunActive;
        const collectDisabled = !activeTracker || fetching || autoTrackActive || rallyRunActive;
        const sam3Ready = CVAT_UI_SAM3_G0_TIMELINE_ENABLED && activeTracker?.id === SAM3_V4_TRACKER_ID;
        const trackRalliesDisabled = !activeTracker || fetching || autoTrackActive || collectingSeeds || !sam3Ready;

        if (!supportedTrackers.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available trackers found
                        </Text>
                    </Col>
                </Row>
            );
        }

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Tracker</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='center'>
                    <Col span={24}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={supportedTrackers[0].name}
                            disabled={rallyBusy}
                            onChange={this.setActiveTracker}
                        >
                            {supportedTrackers.map(
                                (tracker: MLModel): JSX.Element => (
                                    <Select.Option value={tracker.id} title={tracker.description} key={tracker.id}>
                                        {tracker.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                </Row>
                <Row align='middle' justify='end' gutter={8}>
                    <Col>
                        <CVATTooltip overlay='Draw one box at each rally start. Each box is saved as a single-keyframe track; use frame navigation between rallies.'>
                            <Button
                                type={collectingSeeds ? 'default' : 'primary'}
                                danger={collectingSeeds}
                                className='cvat-tools-collect-seeds-button'
                                disabled={collectingSeeds ? false : collectDisabled}
                                onClick={() => (
                                    collectingSeeds ? this.stopSeedCollection() : this.startSeedCollection()
                                )}
                            >
                                {collectingSeeds ? 'Stop Collecting' : 'Collect Seeds'}
                            </Button>
                        </CVATTooltip>
                    </Col>
                    <Col>
                        <CVATTooltip overlay='Track every collected seed in order, one rally at a time. Requires the SAM3 v4 G0 timeline tracker; stops at the first failed frame.'>
                            <Button
                                type={rallyRunActive ? 'default' : 'primary'}
                                danger={rallyRunActive}
                                className='cvat-tools-track-rallies-button'
                                disabled={rallyRunActive ? false : trackRalliesDisabled}
                                onClick={() => (
                                    rallyRunActive ? this.stopMultiRallyRun() : this.startMultiRallyRun()
                                )}
                            >
                                {rallyRunActive ? 'Stop Rallies' : 'Track Rallies'}
                            </Button>
                        </CVATTooltip>
                    </Col>
                </Row>
                {(collectingSeeds || rallyRunActive || rallySeeds.length > 0) && (
                    <Row justify='end'>
                        <Col>
                            <Text type='secondary' className='cvat-text-color'>
                                {rallyRunActive && rallyStatus ? rallyStatus : `Seeds: ${rallySeeds.length}`}
                            </Text>
                        </Col>
                    </Row>
                )}
            </>
        );
    }

    private renderInteractorBlock(): JSX.Element {
        const {
            interactors, canvasInstance, labels, onInteractionStart, interactorExtras,
        } = this.props;
        const {
            activeInteractor, activeLabelID, fetching, startInteractingWithBox, convertMasksToPolygons,
        } = this.state;

        if (!interactors.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available interactors found
                        </Text>
                    </Col>
                </Row>
            );
        }

        const minNegVertices = activeInteractor?.params?.canvas?.minNegVertices ?? -1;
        const renderStartWithBox = activeInteractor?.params?.canvas?.startWithBoxOptional ?? false;

        const renderedInteractorExtras = interactorExtras
            .sort((a, b) => a.data.weight - b.data.weight)
            .filter((plugin) => plugin.data.shouldBeRendered(this.props, this.state))
            .map(({ component: Component }, index) => (
                <Component targetProps={this.props} targetState={this.state} key={index} />
            ));

        return (
            <>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color'>Interactor</Text>
                    </Col>
                </Row>
                <Row align='middle' justify='space-between'>
                    <Col span={22}>
                        <Select
                            style={{ width: '100%' }}
                            defaultValue={interactors[0].name}
                            onChange={this.setActiveInteractor}
                        >
                            {interactors.map(
                                (interactor: MLModel): JSX.Element => (
                                    <Select.Option
                                        value={interactor.id}
                                        title={interactor.description}
                                        key={interactor.id}
                                    >
                                        {interactor.name}
                                    </Select.Option>
                                ),
                            )}
                        </Select>
                    </Col>
                    <Col span={2} className='cvat-interactors-tips-icon-container'>
                        <Popover
                            destroyTooltipOnHide
                            content={(
                                <ToolsTooltips
                                    name={activeInteractor?.name}
                                    withNegativePoints={minNegVertices >= 0}
                                    {...(activeInteractor?.tip || {})}
                                />
                            )}
                        >
                            <QuestionCircleOutlined />
                        </Popover>
                    </Col>
                </Row>
                <div className='cvat-tools-interactor-setups'>
                    <div>
                        <Switch
                            checked={convertMasksToPolygons}
                            onChange={(checked: boolean) => {
                                this.setState({ convertMasksToPolygons: checked });
                            }}
                        />
                        <Text>Convert masks to polygons</Text>
                    </div>

                    {renderStartWithBox && (
                        <div>
                            <Switch
                                checked={startInteractingWithBox}
                                onChange={(value: boolean) => this.setState({ startInteractingWithBox: value })}
                            />
                            <Text>Start with a bounding box</Text>
                        </div>
                    )}
                </div>
                <div className='cvat-tools-interactor-extras'>
                    {renderedInteractorExtras}
                </div>
                <Row align='middle' justify='end'>
                    <Col>
                        <Button
                            type='primary'
                            loading={fetching}
                            className='cvat-tools-interact-button'
                            disabled={!activeInteractor ||
                                fetching ||
                                activeInteractor.version < MIN_SUPPORTED_INTERACTOR_VERSION}
                            onClick={() => {
                                if (activeInteractor && activeLabelID && labels.length) {
                                    this.setState({ mode: 'interaction' });
                                    canvasInstance.cancel();
                                    const startWithBox = activeInteractor.params.canvas.startWithBoxOptional ? (
                                        startInteractingWithBox
                                    ) : activeInteractor.params.canvas.startWithBox ?? false;

                                    const parameters = {
                                        command: startWithBox ? 'draw_box' as const : 'draw_points' as const,
                                        settings: {
                                            appendCursorPositionAsPoint: false,
                                            removalStrategy: 'any' as const,
                                            points_type: 'any' as const,
                                            crosshair: startWithBox,
                                        },
                                    };
                                    canvasInstance.interact({ enabled: true, ...parameters });
                                    onInteractionStart(activeInteractor, activeLabelID, parameters);
                                }
                            }}
                        >
                            Interact
                        </Button>
                    </Col>
                </Row>
            </>
        );
    }

    private renderDetectorBlock(): JSX.Element {
        const {
            jobInstance, detectors, curZOrder, frame, labels, createAnnotations,
        } = this.props;

        if (!detectors.length) {
            return (
                <Row justify='center' align='middle' style={{ marginTop: '5px' }}>
                    <Col>
                        <Text type='warning' className='cvat-text-color'>
                            No available detectors found
                        </Text>
                    </Col>
                </Row>
            );
        }

        return (
            <DetectorRunner
                withCleanup={false}
                models={detectors}
                labels={labels}
                dimension={jobInstance.dimension}
                runInference={async (model: MLModel, body: AnnotateTaskRequestBody) => {
                    function loadAttributes(
                        attributes: { spec_id: number; value: string }[],
                    ): Record<number, string> {
                        return Object.fromEntries(attributes.map((a) => [a.spec_id, a.value]));
                    }

                    try {
                        this.setState({ mode: 'detection', fetching: true });

                        // The function call endpoint doesn't support the cleanup parameter.
                        const { cleanup, ...restOfBody } = body;

                        const result = await core.lambda.call(jobInstance.taskId, model, {
                            ...restOfBody, type: 'annotate_frame', frame, job: jobInstance.id,
                        }) as DetectorResults;

                        const tagStates = result.tags.map((tag) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === tag.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(tag.attributes),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.TAG,
                                source: core.enums.Source.AUTO,
                            });
                        });

                        const shapeStates = result.shapes.map((shape) => {
                            const jobLabel = jobInstance.labels
                                .find((jLabel) => jLabel.id === shape.label_id)!;

                            return new core.classes.ObjectState({
                                attributes: loadAttributes(shape.attributes),
                                elements: shape.elements?.map((element) => {
                                    const jobSublabel = jobLabel.structure!.sublabels
                                        .find((sublabel) => sublabel.id === element.label_id)!;

                                    return {
                                        attributes: loadAttributes(element.attributes),
                                        frame,
                                        label: jobSublabel,
                                        objectType: ObjectType.SHAPE,
                                        occluded: element.occluded,
                                        outside: element.outside,
                                        points: element.points,
                                        shapeType: element.type,
                                        source: core.enums.Source.AUTO,
                                    };
                                }),
                                frame,
                                label: jobLabel,
                                objectType: ObjectType.SHAPE,
                                occluded: shape.occluded,
                                points: shape.points,
                                rotation: shape.rotation,
                                shapeType: shape.type,
                                source: core.enums.Source.AUTO,
                                zOrder: curZOrder,
                            });
                        });

                        createAnnotations([...tagStates, ...shapeStates]);
                    } catch (error: any) {
                        notification.error({
                            description: <CVATMarkdown>{error.message}</CVATMarkdown>,
                            message: 'Detection error occurred',
                            duration: null,
                        });
                    } finally {
                        this.setState({ fetching: false });
                    }
                }}
            />
        );
    }

    private renderPopoverContent(): JSX.Element {
        return (
            <div className='cvat-tools-control-popover-content'>
                <Row justify='start'>
                    <Col>
                        <Text className='cvat-text-color' strong>
                            AI Tools
                        </Text>
                    </Col>
                </Row>
                <Tabs
                    type='card'
                    tabBarGutter={8}
                    items={[{
                        key: 'interactors',
                        label: 'Interactors',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderInteractorBlock()}
                            </>
                        ),
                    }, {
                        key: 'detectors',
                        label: 'Detectors',
                        children: this.renderDetectorBlock(),
                    }, {
                        key: 'trackers',
                        label: 'Trackers',
                        children: (
                            <>
                                {this.renderLabelBlock()}
                                {this.renderTrackerBlock()}
                            </>
                        ),
                    }]}
                />
            </div>
        );
    }

    public render(): JSX.Element | null {
        const {
            interactors, detectors, trackers, isActivated,
            canvasInstance, labels, frameIsDeleted,
        } = this.props;
        const {
            fetching, approxPolyAccuracy, interactorResponseReceived, thresholdValue,
            showConfidenceControl, mode, portals, convertMasksToPolygons,
            rallyRunActive, rallyStatus,
        } = this.state;

        if (![...interactors, ...detectors, ...trackers].length) return null;

        const dynamicPopoverProps = isActivated ?
            {
                overlayStyle: {
                    display: 'none',
                },
            } :
            {};

        const dynamicIconProps = isActivated ?
            {
                className: 'cvat-tools-control cvat-active-canvas-control',
                onClick: (): void => {
                    canvasInstance.interact({ enabled: false });
                },
            } :
            {
                className: 'cvat-tools-control',
            };

        const showAnyContent = labels.length && !frameIsDeleted;
        const showInteractionContent = isActivated && mode === 'interaction' && interactorResponseReceived;
        const showDetectionContent = fetching && mode === 'detection';

        const interactionContent: JSX.Element | null = showInteractionContent ? (
            <>
                { convertMasksToPolygons && (
                    <ApproximationAccuracy
                        approxPolyAccuracy={approxPolyAccuracy}
                        onChange={(value: number) => {
                            this.setState({ approxPolyAccuracy: value });
                        }}
                    />
                )}
                { showConfidenceControl && (
                    <ConfidenceThreshold
                        thresholdValue={thresholdValue}
                        onChange={(value: number) => {
                            this.setState({ thresholdValue: value });
                        }}
                    />
                )}
            </>
        ) : null;

        const detectionContent: JSX.Element | null = showDetectionContent ? (
            <Modal
                title='Making a server request'
                zIndex={Number.MAX_SAFE_INTEGER}
                open
                destroyOnClose
                closable={false}
                footer={[]}
            >
                <Text>Waiting for a server response..</Text>
                <LoadingOutlined style={{ marginLeft: '10px' }} />
            </Modal>
        ) : null;

        /*
         * While a rally run owns the job, this modal is the shared write block:
         * `Mousetrap.prototype.stopCallback` stops every keyboard shortcut while an
         * `ant-modal` is visible, and the modal mask swallows canvas/sidebar/top-bar
         * pointer edits. `switchNavigationBlocked` covers programmatic frame changes
         * and undo/redo. The Stop button (and Escape) end the run.
         */
        const rallyBlockContent: JSX.Element | null = rallyRunActive ? (
            <Modal
                title='Tracking rallies'
                zIndex={Number.MAX_SAFE_INTEGER}
                open
                destroyOnClose
                closable={false}
                keyboard={false}
                maskClosable={false}
                footer={[]}
            >
                <Text>{rallyStatus ?? 'Preparing rally seeds…'}</Text>
                <LoadingOutlined style={{ marginLeft: '10px' }} />
                <Row justify='end' style={{ marginTop: '16px' }}>
                    <Col>
                        <Button danger onClick={() => this.stopMultiRallyRun()}>
                            Stop Rallies
                        </Button>
                    </Col>
                </Row>
            </Modal>
        ) : null;

        return showAnyContent ? (
            <>
                <CustomPopover {...dynamicPopoverProps} placement='right' content={this.renderPopoverContent()}>
                    <Icon {...dynamicIconProps} component={AIToolsIcon} />
                </CustomPopover>
                {interactionContent}
                {detectionContent}
                {rallyBlockContent}
                {portals}
            </>
        ) : (
            <>
                <Icon className=' cvat-tools-control cvat-disabled-canvas-control' component={AIToolsIcon} />
                {rallyBlockContent}
            </>
        );
    }
}

export default connect(mapStateToProps, mapDispatchToProps)(ToolsControlComponent);
