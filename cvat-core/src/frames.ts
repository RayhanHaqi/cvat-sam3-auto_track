// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import _, { range, sortedIndexOf } from 'lodash';
import {
    FrameDecoder, BlockType, DimensionType, ChunkQuality, decodeContextImages, RequestOutdatedError,
} from 'cvat-data';
import PluginRegistry from './plugins';
import serverProxy from './server-proxy';
import { SerializedChapterMetaData, SerializedFramesMetaData } from './server-response-types';
import { ArgumentError } from './exceptions';
import { FieldUpdateTrigger } from './common';
import config from './config';

// frame storage by job id
const frameDataCache: Record<string, {
    metaFetchedTimestamp: number;
    chunkSize: number;
    mode: 'annotation' | 'interpolation';
    jobStartFrame: number;
    decodeForward: boolean;
    forwardStep: number;
    latestFrameDecodeRequest: number | null;
    latestContextImagesRequest: number | null;
    provider: FrameDecoder;
    prefetchAnalyzer: PrefetchAnalyzer;
    decodedBlocksCacheSize: number;
    activeChunkRequest: Promise<void> | null;
    activeContextRequest: Promise<Record<number, ImageBitmap>> | null;
    segmentFrameNumbers: number[];
    contextCache: Record<number, {
        data: Record<number, ImageBitmap>;
        timestamp: number;
        size: number;
    }>;
    getChunk: (chunkIndex: number, quality: ChunkQuality) => Promise<ArrayBuffer>;
    getMeta: () => Promise<FramesMetaData>;
    backgroundPrefetch: BackgroundChunkPrefetch;
}> = {};

type FrameDataCacheEntry = (typeof frameDataCache)[string];

// how many chunks the background prefetch may hold decoded for the requested frames
const BACKGROUND_PREFETCH_CHUNKS_LIMIT = 3;
// how long to wait before re-checking the decoder when it is busy, but there is no
// visible frame chunk request to wait for
const BACKGROUND_PREFETCH_RETRY_TIMEOUT = 500;

/**
 * State of the background chunk prefetch requested by `prefetchChunksForFrames`.
 * It is intentionally kept separate from `activeChunkRequest` and
 * `latestFrameDecodeRequest`, which belong to the visible frame path.
 */
interface BackgroundChunkPrefetch {
    // chunk indexes we want to keep decoded for the latest request
    targets: number[];
    // targets which are neither decoded nor are being downloaded/decoded right now
    queued: number[];
    // the chunk which is being downloaded/decoded by the background prefetch right now
    inFlight: number | null;
    // chunks decoded by the background prefetch and still held in the decoder
    decodedChunks: number[];
    // the chunk with the frame currently opened in the player
    visibleChunk: number | null;
    // a retry is already scheduled
    retryScheduled: boolean;
    // resolves the chunk request announced for the chunk which is being decoded
    releaseDecodeRequest: (() => void) | null;
}

// frame meta data storage by job id
const frameMetaCacheSync: Record<string, FramesMetaData> = {};
const frameMetaCache: Record<string, Promise<FramesMetaData>> = new Proxy({}, {
    set(target, prop, value): boolean {
        if (typeof prop === 'string' && value instanceof Promise) {
            const result = Reflect.set(target, prop, value);

            // automatically update synced storage each time new promise set
            if (result) {
                value.then((metaData: FramesMetaData) => {
                    if (target[prop]) {
                        frameMetaCacheSync[prop] = metaData;
                    }
                }).catch(() => {
                    // do nothing
                });
            }

            return result;
        }
        return Reflect.set(target, prop, value);
    },
    deleteProperty(target, prop): boolean {
        if (typeof prop === 'string') {
            const result = Reflect.deleteProperty(target, prop);
            if (result) {
                delete frameMetaCacheSync[prop];
            }
            return result;
        }

        return Reflect.deleteProperty(target, prop);
    },
});

enum DeletedFrameState {
    DELETED = 'deleted',
    RESTORED = 'restored',
}

interface FramesMetaDataUpdatedData {
    cloudStorageId?: boolean;
    deletedFrames?: Record<number, DeletedFrameState>;
}

export class ChapterMetaData {
    readonly #title: string;

    constructor(initialData: SerializedChapterMetaData) {
        this.#title = initialData.title;
    }

    get title(): string {
        return this.#title;
    }
}

export class Chapter {
    public id: number;
    public start: number;
    public stop: number;
    public metadata: ChapterMetaData;
}

export class FramesMetaData {
    public chunkSize: number;
    public chapters: Chapter[] | null;
    public deletedFrames: Record<number, boolean>;
    public includedFrames: number[] | null;
    public frameFilter: string;
    public frames: {
        width: number;
        height: number;
        name: string;
        related_files: number;
    }[];
    public imageQuality: number;
    public size: number;
    public startFrame: number;
    public stopFrame: number;
    public frameStep: number;
    public chunkCount: number;
    public chunksUpdatedDate: string;
    public storage: string;
    public cloudStorageId: number | null;

    #updateTrigger: FieldUpdateTrigger;

    constructor(initialData: Omit<SerializedFramesMetaData, 'deleted_frames'> & { deleted_frames: Record<number, boolean> }) {
        const data: typeof initialData = {
            chunk_size: undefined,
            chapters: [],
            deleted_frames: {},
            included_frames: null,
            frame_filter: undefined,
            frames: [],
            image_quality: undefined,
            size: undefined,
            start_frame: undefined,
            stop_frame: undefined,
            chunks_updated_date: undefined,
            storage: undefined,
            cloud_storage_id: undefined,
        };

        this.#updateTrigger = new FieldUpdateTrigger();

        for (const property in data) {
            if (Object.prototype.hasOwnProperty.call(data, property) && property in initialData) {
                if (property === 'deleted_frames') {
                    const update = (frame: string, remove: boolean): void => {
                        const [state, oppositeState] = remove ?
                            [DeletedFrameState.DELETED, DeletedFrameState.RESTORED] :
                            [DeletedFrameState.RESTORED, DeletedFrameState.DELETED];
                        if (this.#updateTrigger.get(`deletedFrames:${frame}:${oppositeState}`)) {
                            this.#updateTrigger.resetField(`deletedFrames:${frame}:${oppositeState}`);
                        } else {
                            this.#updateTrigger.update(`deletedFrames:${frame}:${state}`);
                        }
                    };

                    const handler = {
                        set: (target, prop, value) => {
                            update(prop, value);
                            return Reflect.set(target, prop, value);
                        },
                        deleteProperty: (target, prop) => {
                            if (prop in target) {
                                update(prop, false);
                            }
                            return Reflect.deleteProperty(target, prop);
                        },
                    };
                    data[property] = new Proxy(initialData[property], handler);
                } else {
                    data[property] = initialData[property];
                }
            }
        }

        const frameStep: number = (() => {
            if (data.frame_filter) {
                const frameStepParts = data.frame_filter.split('=', 2);
                if (frameStepParts.length !== 2) {
                    throw new ArgumentError(`Invalid frame filter '${data.frame_filter}'`);
                }
                return +frameStepParts[1];
            }
            return 1;
        })();

        Object.defineProperties(
            this,
            Object.freeze({
                chunkSize: {
                    get: () => data.chunk_size,
                },
                chapters: {
                    get: () => data.chapters,
                },
                deletedFrames: {
                    get: () => data.deleted_frames,
                },
                includedFrames: {
                    get: () => data.included_frames,
                },
                frameFilter: {
                    get: () => data.frame_filter,
                },
                imageQuality: {
                    get: () => data.image_quality,
                },
                size: {
                    get: () => data.size,
                },
                startFrame: {
                    get: () => data.start_frame,
                },
                stopFrame: {
                    get: () => data.stop_frame,
                },
                frameStep: {
                    get: () => frameStep,
                },
                chunksUpdatedDate: {
                    get: () => data.chunks_updated_date,
                },
                storage: {
                    get: () => data.storage,
                },
                cloudStorageId: {
                    get: () => data.cloud_storage_id,
                    set: (cloudStorageId) => {
                        this.#updateTrigger.update('cloudStorageId');
                        data.cloud_storage_id = cloudStorageId;
                    },
                },
            }),
        );

        const frameNumbers = this.getDataFrameNumbers();
        const chunkCount: number = Math.ceil(frameNumbers.length / this.chunkSize);

        let framesInfo = [];
        if (initialData.frames.length === 1) {
            // it may be a videofile or one image
            framesInfo = frameNumbers.map(() => initialData.frames[0]);
        } else if (this.includedFrames?.length) {
            // Only simple GT jobs have includedFrames, so this condition works only for them
            const framesNumbers = new Set(this.includedFrames.map((dataFrameNumber: number) => (
                Math.floor((dataFrameNumber - this.startFrame) / this.frameStep)
            )));
            // Frames must contain placeholders on positions out of includedFrames.
            // That is, we can use frames with indices corresponding to frame numbers
            framesInfo = initialData.frames.filter((frame, idx) => framesNumbers.has(idx));
        } else {
            framesInfo = initialData.frames;
        }

        Object.defineProperties(
            this,
            Object.freeze({
                chunkCount: {
                    get: () => chunkCount,
                },
                frames: {
                    get: () => framesInfo,
                },
            }),
        );
    }

    getDataFrameNumber(jobRelativeFrame: number): number {
        return this.frameStep * jobRelativeFrame + this.startFrame;
    }

    getJobRelativeFrameNumber(dataFrameNumber: number): number {
        return (dataFrameNumber - this.startFrame) / this.frameStep;
    }

    getUpdated(): FramesMetaDataUpdatedData {
        const updatedFields: FramesMetaDataUpdatedData = {};
        for (const key of Object.keys(this.#updateTrigger.getUpdated(this))) {
            if (key.startsWith('deletedFrames')) {
                updatedFields.deletedFrames = updatedFields.deletedFrames ?? {};
                const [, frame, state] = key.split(':');
                updatedFields.deletedFrames[frame] = state;
            } else if (key === 'cloudStorageId') {
                updatedFields.cloudStorageId = true;
            }
        }
        return updatedFields;
    }

    resetUpdated(): void {
        this.#updateTrigger.reset();
    }

    getFrameIndex(dataFrameNumber: number): number {
        // Here we use absolute (task source data) frame numbers.
        // TODO: migrate from data frame numbers to local frame numbers to simplify code.
        // Requires server changes in api/jobs/{id}/data/meta/
        // for included_frames, start_frame, stop_frame fields

        if (dataFrameNumber < this.startFrame || dataFrameNumber > this.stopFrame) {
            throw new ArgumentError(`Frame number ${dataFrameNumber} doesn't belong to the job`);
        }

        let frameIndex = null;
        if (this.includedFrames) {
            frameIndex = sortedIndexOf(this.includedFrames, dataFrameNumber);
            if (frameIndex === -1) {
                throw new ArgumentError(`Frame number ${dataFrameNumber} doesn't belong to the job`);
            }
        } else {
            frameIndex = Math.floor((dataFrameNumber - this.startFrame) / this.frameStep);
        }
        return frameIndex;
    }

    getFrameChunkIndex(dataFrameNumber: number): number {
        return Math.floor(this.getFrameIndex(dataFrameNumber) / this.chunkSize);
    }

    getSegmentFrameNumbers(jobStartFrame: number): number[] {
        const frames = this.getDataFrameNumbers();
        return frames.map((frame) => this.getJobRelativeFrameNumber(frame) + jobStartFrame);
    }

    getDataFrameNumbers(): number[] {
        if (this.includedFrames) {
            return this.includedFrames.slice(0);
        }

        return range(this.startFrame, this.stopFrame + 1, this.frameStep);
    }
}

export class FrameData {
    public readonly filename: string;
    public readonly width: number;
    public readonly height: number;
    public readonly number: number;
    public readonly relatedFiles: number;
    public readonly deleted: boolean;
    public readonly jobID: number;

    constructor({
        width,
        height,
        name,
        jobID,
        frameNumber,
        deleted,
        related_files: relatedFiles,
    }) {
        Object.defineProperties(
            this,
            Object.freeze({
                filename: {
                    value: name,
                    writable: false,
                },
                width: {
                    value: width,
                    writable: false,
                },
                height: {
                    value: height,
                    writable: false,
                },
                jobID: {
                    value: jobID,
                    writable: false,
                },
                number: {
                    value: frameNumber,
                    writable: false,
                },
                relatedFiles: {
                    value: relatedFiles,
                    writable: false,
                },
                deleted: {
                    value: deleted,
                    writable: false,
                },
            }),
        );
    }

    async data(onServerRequest = () => {}): Promise<{
        renderWidth: number;
        renderHeight: number;
        imageData: ImageBitmap | Blob;
    }> {
        const result = await PluginRegistry.apiWrapper.call(this, FrameData.prototype.data, onServerRequest);
        return result;
    }
}

class PrefetchAnalyzer {
    #requestedFrames: number[];
    #meta: FramesMetaData;
    #getDataFrameNumber: (frameNumber: number) => number;

    constructor(meta: FramesMetaData, dataFrameNumberGetter: (frameNumber: number) => number) {
        this.#requestedFrames = [];
        this.#meta = meta;
        this.#getDataFrameNumber = dataFrameNumberGetter;
    }

    shouldPrefetchNext(current: number, isPlaying: boolean, isChunkCached: (chunk) => boolean): boolean {
        if (isPlaying) {
            return true;
        }

        const currentDataFrameNumber = this.#getDataFrameNumber(current);
        const currentChunk = this.#meta.getFrameChunkIndex(currentDataFrameNumber);
        const { length } = this.#requestedFrames;
        const isIncreasingOrder = this.#requestedFrames
            .every((val, index) => index === 0 || val > this.#requestedFrames[index - 1]);
        if (
            length && (isIncreasingOrder && current > this.#requestedFrames[length - 1]) &&
            (
                this.#meta.getFrameIndex(currentDataFrameNumber) % this.#meta.chunkSize
            ) >= Math.ceil(this.#meta.chunkSize / 2) &&
            !isChunkCached(currentChunk + 1)
        ) {
            // is increasing order including the current frame
            // if current is in the middle+ of the chunk
            // if the next chunk was not cached yet
            return true;
        }

        return false;
    }

    addRequested(frame: number): void {
        // latest requested frame is bubbling (array is unique)
        const indexOf = this.#requestedFrames.indexOf(frame);
        if (indexOf !== -1) {
            this.#requestedFrames.splice(indexOf, 1);
        }

        this.#requestedFrames.push(frame);

        // only half of chunk size is considered in this logic
        const limit = Math.ceil(this.#meta.chunkSize / 2);
        if (this.#requestedFrames.length > limit) {
            this.#requestedFrames.shift();
        }
    }
}

Object.defineProperty(FrameData.prototype.data, 'implementation', {
    async value(this: FrameData, onServerRequest) {
        const {
            provider, prefetchAnalyzer, chunkSize, jobStartFrame,
            decodeForward, forwardStep, decodedBlocksCacheSize, segmentFrameNumbers,
        } = frameDataCache[this.jobID];
        const meta = await frameDataCache[this.jobID].getMeta();

        return new Promise<{
            renderWidth: number;
            renderHeight: number;
            imageData: ImageBitmap | Blob;
        }>((resolve, reject) => {
            const requestId = +_.uniqueId();
            const requestedDataFrameNumber = meta.getDataFrameNumber(this.number - jobStartFrame);
            const chunkIndex = meta.getFrameChunkIndex(requestedDataFrameNumber);
            const frame = provider.frame(this.number);

            function findTheNextNotDecodedChunk(currentFrameIndex: number): number | null {
                const { chunkCount } = meta;
                let nextFrameIndex = currentFrameIndex + forwardStep;
                let nextChunkIndex = Math.floor(nextFrameIndex / chunkSize);
                while (nextChunkIndex === chunkIndex) {
                    nextFrameIndex += forwardStep;
                    nextChunkIndex = Math.floor(nextFrameIndex / chunkSize);
                }

                if (nextChunkIndex < 0 || chunkCount <= nextChunkIndex) {
                    return null;
                }

                if (provider.isChunkCached(nextChunkIndex)) {
                    return findTheNextNotDecodedChunk(nextFrameIndex);
                }

                return nextChunkIndex;
            }

            if (frame) {
                if (
                    prefetchAnalyzer.shouldPrefetchNext(
                        this.number,
                        decodeForward,
                        (chunk) => provider.isChunkCached(chunk),
                    ) && decodedBlocksCacheSize > 1 && !frameDataCache[this.jobID].activeChunkRequest
                ) {
                    const nextChunkIndex = findTheNextNotDecodedChunk(
                        meta.getFrameIndex(requestedDataFrameNumber),
                    );
                    const predecodeChunksMax = Math.floor(decodedBlocksCacheSize / 2);
                    if (nextChunkIndex !== null &&
                        nextChunkIndex <= chunkIndex + predecodeChunksMax
                    ) {
                        frameDataCache[this.jobID].activeChunkRequest = new Promise((resolveForward) => {
                            const releasePromise = (): void => {
                                resolveForward();
                                frameDataCache[this.jobID].activeChunkRequest = null;
                            };

                            frameDataCache[this.jobID].getChunk(
                                nextChunkIndex, ChunkQuality.COMPRESSED,
                            ).then((chunk: ArrayBuffer) => {
                                if (!(this.jobID in frameDataCache)) {
                                    // check if frameDataCache still exist
                                    // as it may be released during chunk request
                                    resolveForward();
                                    return;
                                }

                                provider.cleanup(1);
                                provider.requestDecodeBlock(
                                    chunk,
                                    nextChunkIndex,
                                    segmentFrameNumbers.slice(
                                        nextChunkIndex * chunkSize,
                                        (nextChunkIndex + 1) * chunkSize,
                                    ),
                                    () => {},
                                    releasePromise,
                                    releasePromise,
                                );
                            }).catch(() => {
                                releasePromise();
                            });
                        });
                    }
                }

                resolve({
                    renderWidth: this.width,
                    renderHeight: this.height,
                    imageData: frame,
                });
                prefetchAnalyzer.addRequested(this.number);
                return;
            }

            onServerRequest();
            frameDataCache[this.jobID].latestFrameDecodeRequest = requestId;
            (frameDataCache[this.jobID].activeChunkRequest || Promise.resolve()).finally(() => {
                if (frameDataCache[this.jobID]?.latestFrameDecodeRequest !== requestId) {
                    // not relevant request anymore
                    reject(this.number);
                    return;
                }

                // it might appear during previous decoding, so, check again
                const currentFrame = provider.frame(this.number);
                if (currentFrame) {
                    resolve({
                        renderWidth: this.width,
                        renderHeight: this.height,
                        imageData: currentFrame,
                    });
                    prefetchAnalyzer.addRequested(this.number);
                    return;
                }

                frameDataCache[this.jobID].activeChunkRequest = new Promise<void>((
                    resolveLoadAndDecode,
                ) => {
                    let wasResolved = false;
                    frameDataCache[this.jobID].getChunk(
                        chunkIndex, ChunkQuality.COMPRESSED,
                    ).then((chunk: ArrayBuffer) => {
                        try {
                            if (!(this.jobID in frameDataCache)) {
                                // check if frameDataCache still exist
                                // as it may be released during chunk request
                                resolveLoadAndDecode();
                                reject(this.number);
                                return;
                            }

                            provider
                                .requestDecodeBlock(
                                    chunk,
                                    chunkIndex,
                                    segmentFrameNumbers.slice(
                                        chunkIndex * chunkSize,
                                        (chunkIndex + 1) * chunkSize,
                                    ),
                                    (_frame: number, bitmap: ImageBitmap | Blob) => {
                                        if (decodeForward) {
                                            // resolve immediately only if is not playing
                                            return;
                                        }

                                        if (frameDataCache[this.jobID].latestFrameDecodeRequest === requestId &&
                                            this.number === _frame
                                        ) {
                                            wasResolved = true;
                                            resolve({
                                                renderWidth: this.width,
                                                renderHeight: this.height,
                                                imageData: bitmap,
                                            });
                                            prefetchAnalyzer.addRequested(this.number);
                                        }
                                    },
                                    () => {
                                        frameDataCache[this.jobID].activeChunkRequest = null;
                                        resolveLoadAndDecode();
                                        const decodedFrame = provider.frame(this.number);
                                        if (decodeForward) {
                                            // resolve after decoding everything if playing
                                            resolve({
                                                renderWidth: this.width,
                                                renderHeight: this.height,
                                                imageData: decodedFrame,
                                            });
                                        } else if (!wasResolved) {
                                            reject(this.number);
                                        }
                                    },
                                    (error: Error | RequestOutdatedError) => {
                                        frameDataCache[this.jobID].activeChunkRequest = null;
                                        resolveLoadAndDecode();
                                        if (error instanceof RequestOutdatedError) {
                                            reject(this.number);
                                        } else {
                                            reject(error);
                                        }
                                    },
                                );
                        } catch (error) {
                            reject(error);
                        }
                    }).catch((error) => {
                        reject(error);
                        resolveLoadAndDecode(error);
                    });
                });
            });
        });
    },
    writable: false,
});

function mergeMetaData(
    nextData: SerializedFramesMetaData,
    previousData?: Promise<FramesMetaData>,
): Promise<FramesMetaData> {
    const framesMetaData = new FramesMetaData({
        ...nextData,
        deleted_frames: Object.fromEntries(nextData.deleted_frames.map((_frame) => [_frame, true])),
    });

    if (previousData instanceof Promise) {
        return previousData.then((prevMeta) => {
            const updatedFields = prevMeta.getUpdated();
            if (updatedFields.deletedFrames) {
                for (const [frame, state] of Object.entries(updatedFields.deletedFrames)) {
                    if (state === DeletedFrameState.DELETED) {
                        framesMetaData.deletedFrames[frame] = true;
                    } else if (state === DeletedFrameState.RESTORED) {
                        delete framesMetaData.deletedFrames[frame];
                    }
                }
            }

            if (updatedFields.cloudStorageId) {
                framesMetaData.cloudStorageId = prevMeta.cloudStorageId;
            }

            return framesMetaData;
        });
    }

    return Promise.resolve(framesMetaData);
}

export function getJobFramesMetaSync(jobID: number): FramesMetaData {
    const cached = frameMetaCacheSync[jobID];
    if (!cached) {
        throw new Error('Frames meta cache was not initialized for this job');
    }
    return cached;
}

export function getFramesMeta(type: 'job' | 'task', id: number, forceReload = false): Promise<FramesMetaData> {
    if (type === 'task') {
        // we do not cache task meta currently. So, each new call will results to the server request
        return serverProxy.frames.getMeta('task', id).then((serialized) => (
            new FramesMetaData({
                ...serialized,
                deleted_frames: Object.fromEntries(serialized.deleted_frames.map((_frame) => [_frame, true])),
            })
        ));
    }

    if (!(id in frameMetaCache) || forceReload) {
        const previousCache = frameMetaCache[id];
        frameMetaCache[id] = new Promise((resolve, reject) => {
            serverProxy.frames.getMeta('job', id).then((serialized) => {
                // When we get new framesMetaData from server there can be some unsaved data
                // here we merge new meta data with cached one
                mergeMetaData(serialized, previousCache).then((mergedData) => {
                    resolve(mergedData);
                });
            }).catch((error: unknown) => {
                delete frameMetaCache[id];
                if (previousCache instanceof Promise) {
                    frameMetaCache[id] = previousCache;
                }
                reject(error);
            });
        });
    }

    return frameMetaCache[id];
}

function saveMeta(meta: FramesMetaData, session: 'job' | 'task', id: number): Promise<FramesMetaData> {
    const newMeta = new Promise<FramesMetaData>((resolve, reject) => {
        serverProxy.frames.saveMeta(session, id, {
            deleted_frames: Object.keys(meta.deletedFrames).map((frame) => +frame),
            cloud_storage_id: meta.cloudStorageId,
        }).then((serverMeta) => {
            const updatedMetaData = new FramesMetaData({
                ...serverMeta,
                deleted_frames: Object.fromEntries(serverMeta.deleted_frames.map((_frame) => [_frame, true])),
            });
            resolve(updatedMetaData);
        }).catch((error) => {
            if (session === 'job') {
                frameMetaCache[id] = Promise.resolve(meta);
            }
            reject(error);
        });
    });

    if (session === 'job') {
        frameMetaCache[id] = newMeta;
    }

    return newMeta;
}

async function refreshJobCacheIfOutdated(jobID: number): Promise<void> {
    const cached = frameDataCache[jobID];
    if (!cached) {
        throw new Error('Frames meta cache was not initialized for this job');
    }

    const isOutdated = (Date.now() - cached.metaFetchedTimestamp) > config.jobMetaDataReloadPeriod;

    if (isOutdated) {
        // get metadata again if outdated
        const prevMeta = await cached.getMeta();
        const meta = await getFramesMeta('job', jobID, true);
        if (new Date(meta.chunksUpdatedDate) > new Date(prevMeta.chunksUpdatedDate)) {
            // chunks were re-defined. Existing data not relevant anymore
            // currently we only re-write meta, remove all cached frames from provider and clear cached context images
            // other parameters (e.g. chunkSize) are not supposed to be changed
            cached.provider.cleanup(Number.MAX_SAFE_INTEGER);
            for (const frame of Object.keys(cached.contextCache)) {
                for (const image of Object.values(cached.contextCache[+frame].data)) {
                    // close images to immediate memory release
                    image.close();
                }
            }
            cached.contextCache = {};
        }

        cached.metaFetchedTimestamp = Date.now();
    }
}

export async function getContextImage(
    jobID: number,
    frame: number,
    getImageContext: (frame: number) => Promise<ArrayBuffer>,
): Promise<Record<string, ImageBitmap>> {
    if (!(jobID in frameDataCache)) {
        throw new Error(
            'Frame data was not initialized for this job. Try first requesting any frame.',
        );
    }

    const frameData = frameDataCache[jobID];
    const meta = await frameData.getMeta();
    const requestId = frame;
    const { jobStartFrame } = frameData;
    const dataFrameNumber = meta.getDataFrameNumber(frame - jobStartFrame);
    const frameIndex = meta.getFrameIndex(dataFrameNumber);
    const { related_files: relatedFiles } = meta.frames[frameIndex];
    return new Promise<Record<string, ImageBitmap>>((resolve, reject) => {
        if (relatedFiles === 0) {
            resolve({});
        } else if (frame in frameData.contextCache) {
            resolve(frameData.contextCache[frame].data);
        } else {
            frameData.latestContextImagesRequest = requestId;
            const executor = (): void => {
                if (frameData.latestContextImagesRequest !== requestId) {
                    reject(frame);
                } else if (frame in frameData.contextCache) {
                    resolve(frameData.contextCache[frame].data);
                } else {
                    frameData.activeContextRequest = getImageContext(frame)
                        .then((encodedImages) => decodeContextImages(encodedImages, 0, relatedFiles));
                    frameData.activeContextRequest.then((images) => {
                        const size = Object.values(images)
                            .reduce((acc, image) => acc + image.width * image.height * 4, 0);
                        const totalSize = Object.values(frameData.contextCache)
                            .reduce((acc, item) => acc + item.size, 0);
                        if (totalSize > 512 * 1024 * 1024) {
                            const [leastTimestampFrame] = Object.entries(frameData.contextCache)
                                .sort(([, item1], [, item2]) => item1.timestamp - item2.timestamp)[0];
                            delete frameData.contextCache[leastTimestampFrame];
                        }

                        frameData.contextCache[frame] = {
                            data: images,
                            timestamp: Date.now(),
                            size,
                        };

                        if (frameData.latestContextImagesRequest !== requestId) {
                            reject(frame);
                        } else {
                            resolve(images);
                        }
                    }).finally(() => {
                        frameData.activeContextRequest = null;
                    });
                }
            };

            if (!frameData.activeContextRequest) {
                executor();
            } else {
                const checkAndExecute = (): void => {
                    if (frameData.activeContextRequest) {
                        // if we just execute in finally
                        // it might raise multiple server requests for context images
                        // if the promise was pending before and several requests came for the same frame
                        // all these requests will stuck on "finally"
                        // and when the promise is fulfilled, it will run all the microtasks
                        // since they all have the same request id, all they will perform in executor()
                        frameData.activeContextRequest.finally(() => setTimeout(checkAndExecute));
                    } else {
                        executor();
                    }
                };

                setTimeout(checkAndExecute);
            }
        }
    });
}

export function decodePreview(preview: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            resolve(reader.result as string);
        };
        reader.onerror = (error) => {
            reject(error);
        };
        reader.readAsDataURL(preview);
    });
}

export async function getFrame(
    jobID: number,
    chunkSize: number,
    chunkType: 'video' | 'imageset',
    mode: 'interpolation' | 'annotation', // todo: obsolete, need to remove
    frame: number,
    jobStartFrame: number,
    isPlaying: boolean,
    step: number,
    dimension: DimensionType,
    getChunk: (chunkIndex: number, quality: ChunkQuality) => Promise<ArrayBuffer>,
): Promise<FrameData> {
    const dataCacheExists = jobID in frameDataCache;

    if (!dataCacheExists) {
        const blockType = chunkType === 'video' ? BlockType.MP4VIDEO : BlockType.ARCHIVE;
        const meta = await getFramesMeta('job', jobID);

        const mean = meta.frames.reduce((a, b) => a + b.width * b.height, 0) / meta.frames.length;
        const stdDev = Math.sqrt(
            meta.frames.map((x) => (x.width * x.height - mean) ** 2).reduce((a, b) => a + b) /
            meta.frames.length,
        );

        // limit of decoded frames cache by 2GB
        const decodedBlocksCacheSize = Math.min(
            Math.floor((2048 * 1024 * 1024) / ((mean + stdDev) * 4 * chunkSize)) || 1, 10,
        );

        const dataFrameNumberGetter = (frameNumber: number): number => (
            meta.getDataFrameNumber(frameNumber - jobStartFrame)
        );

        frameDataCache[jobID] = {
            metaFetchedTimestamp: Date.now(),
            chunkSize,
            mode,
            jobStartFrame,
            segmentFrameNumbers: meta.getSegmentFrameNumbers(jobStartFrame),
            decodeForward: isPlaying,
            forwardStep: step,
            provider: new FrameDecoder(
                blockType,
                decodedBlocksCacheSize,
                (frameNumber: number): number => (
                    meta.getFrameChunkIndex(dataFrameNumberGetter(frameNumber))
                ),
                dimension,
            ),
            prefetchAnalyzer: new PrefetchAnalyzer(meta, dataFrameNumberGetter),
            decodedBlocksCacheSize,
            activeChunkRequest: null,
            activeContextRequest: null,
            latestFrameDecodeRequest: null,
            latestContextImagesRequest: null,
            contextCache: {},
            getChunk,
            getMeta: () => {
                const cached = frameMetaCache[jobID];
                if (!(cached instanceof Promise)) {
                    throw new Error('Frame meta data is not initialized');
                }
                return cached;
            },
            backgroundPrefetch: {
                targets: [],
                queued: [],
                inFlight: null,
                decodedChunks: [],
                visibleChunk: null,
                retryScheduled: false,
                releaseDecodeRequest: null,
            },
        };
    }

    // basically the following functions may be affected if job cache is outdated
    // - getFrame
    // - getContextImage
    // - getCachedChunks
    // And from this idea we should call refreshJobCacheIfOutdated from each one
    // However, following from the order, these methods are usually called
    // it may lead to even more confusing behaviour
    //
    // Usually user first receives frame, then user receives ranges and finally user receives context images
    // In this case (extremely rare, but nevertheless possible) user may get context images related to another frame
    // - if cache gets outdated after getFrame() call
    // - and before getContextImage() call
    // - and both calls refer to the same frame that is refreshed honeypot frame and this frame has context images
    // Thus, it is better to only call `refreshJobCacheIfOutdated` from getFrame()
    await refreshJobCacheIfOutdated(jobID);

    const framesMetaData = await frameDataCache[jobID].getMeta();
    const dataFrameNumber = framesMetaData.getDataFrameNumber(frame - jobStartFrame);
    const frameIndex = framesMetaData.getFrameIndex(dataFrameNumber);
    const frameMeta = framesMetaData.frames[frameIndex];
    frameDataCache[jobID].provider.setRenderSize(frameMeta.width, frameMeta.height);
    frameDataCache[jobID].decodeForward = isPlaying;
    frameDataCache[jobID].forwardStep = step;

    const meta = await frameDataCache[jobID].getMeta();

    return new FrameData({
        width: frameMeta.width,
        height: frameMeta.height,
        name: frameMeta.name,
        related_files: frameMeta.related_files,
        frameNumber: frame,
        deleted: frame in meta.deletedFrames,
        jobID,
    });
}

export async function deleteFrame(jobID: number, frame: number): Promise<void> {
    const meta = await frameMetaCache[jobID];
    meta.deletedFrames[frame] = true;
}

export async function restoreFrame(jobID: number, frame: number): Promise<void> {
    const meta = await frameMetaCache[jobID];
    delete meta.deletedFrames[frame];
}

export async function patchMeta(id: number, meta?: FramesMetaData, session: 'job' | 'task' = 'job'): Promise<FramesMetaData> {
    const oldMeta = (session === 'job' ? await frameMetaCache[id] : meta);
    const updatedFields = oldMeta.getUpdated();
    if (Object.keys(updatedFields).length) {
        const newMeta = await saveMeta(oldMeta, session, id);
        return session === 'job' ? frameMetaCache[id] : newMeta;
    }
    return oldMeta;
}

export async function findFrame(
    jobID: number,
    frameFrom: number,
    frameTo: number,
    filters: { offset?: number, notDeleted: boolean, chapterMark?: boolean },
): Promise<number | null> {
    const offset = filters.offset || 1;
    const chapterMark = filters.chapterMark || false;
    const meta = await getFramesMeta('job', jobID);

    const sign = Math.sign(frameTo - frameFrom);
    const predicate = sign > 0 ? (frame) => frame <= frameTo : (frame) => frame >= frameTo;
    const update = sign > 0 ? (frame) => frame + 1 : (frame) => frame - 1;
    let framesCounter = 0;
    let lastUndeletedFrame = null;
    const check = (frame): boolean => {
        if (meta.includedFrames) {
            // meta.includedFrames contains absolute frame numbers
            const jobStartFrame = 0; // this is only true when includedFrames is set
            return (
                meta.includedFrames.includes(meta.getDataFrameNumber(frame - jobStartFrame))
            ) && (!filters.notDeleted || !(frame in meta.deletedFrames));
        }

        if (filters.notDeleted) {
            return !(frame in meta.deletedFrames);
        }

        if (chapterMark) {
            return meta.chapters.some((chapter) => chapter.start === frame);
        }

        return true;
    };
    for (let frame = frameFrom; predicate(frame); frame = update(frame)) {
        if (check(frame)) {
            lastUndeletedFrame = frame;
            framesCounter++;
            if (framesCounter === offset) {
                return lastUndeletedFrame;
            }
        }
    }

    return lastUndeletedFrame;
}

export function getCachedChunks(jobID: number): number[] {
    if (!(jobID in frameDataCache)) {
        return [];
    }

    return frameDataCache[jobID].provider.cachedChunks(true);
}

export async function getJobFrameNumbers(jobID: number): Promise<number[]> {
    if (!(jobID in frameDataCache)) {
        return [];
    }

    const { segmentFrameNumbers } = frameDataCache[jobID];
    return segmentFrameNumbers.slice(0);
}

function dropStalePrefetchedChunks(entry: FrameDataCacheEntry): void {
    const { backgroundPrefetch } = entry;
    const staleChunks = backgroundPrefetch.decodedChunks.filter((chunkIndex) => (
        // a chunk may be released by the regular decoded chunks cache
        !entry.provider.isChunkCached(chunkIndex) ||
        // the chunk left the requested window, but never release the displayed one
        (chunkIndex !== backgroundPrefetch.visibleChunk &&
            !backgroundPrefetch.targets.includes(chunkIndex))
    ));

    if (!staleChunks.length) {
        return;
    }

    entry.provider.dropChunks(staleChunks);
    backgroundPrefetch.decodedChunks = backgroundPrefetch.decodedChunks
        .filter((chunkIndex) => !staleChunks.includes(chunkIndex));
}

function announceDecodedChunkRequest(entry: FrameDataCacheEntry): void {
    // The decoder decodes one chunk at a time and replaces the callbacks of an in-flight
    // decode as soon as the same chunk is requested again. To keep the background decode
    // intact, it is announced as activeChunkRequest before being requested: then the
    // visible frame path waits for it instead of asking for the same chunk once more,
    // exactly like it waits for the chunk request of its own prefetch.
    const { backgroundPrefetch } = entry;
    let request: Promise<void> | null = null;
    let release: (() => void) | null = null;

    request = new Promise<void>((resolve) => {
        release = (): void => {
            resolve();
            backgroundPrefetch.releaseDecodeRequest = null;
            if (entry.activeChunkRequest === request) {
                entry.activeChunkRequest = null;
            }
        };
    });

    entry.activeChunkRequest = request;
    backgroundPrefetch.releaseDecodeRequest = release;
}

function runBackgroundPrefetch(jobID: number): void {
    const entry = frameDataCache[jobID];
    if (!entry) {
        return;
    }

    const { backgroundPrefetch } = entry;
    if (backgroundPrefetch.inFlight !== null) {
        // one chunk at a time. Also, a decode request which is not known to the decoder
        // anymore must release the chunk request announced for it, otherwise the visible
        // frame path would wait for it forever
        if (
            backgroundPrefetch.releaseDecodeRequest &&
            entry.provider.pendingChunkIndex() !== backgroundPrefetch.inFlight
        ) {
            backgroundPrefetch.inFlight = null;
            backgroundPrefetch.releaseDecodeRequest();
        } else {
            return;
        }
    }

    // wait until the visible frame path releases the decoder, then continue with the queue
    const retryWhenDecoderIsFree = (): void => {
        if (
            backgroundPrefetch.retryScheduled ||
            backgroundPrefetch.inFlight !== null ||
            !backgroundPrefetch.queued.length
        ) {
            return;
        }

        backgroundPrefetch.retryScheduled = true;
        const retry = (): void => {
            const current = frameDataCache[jobID];
            if (current) {
                current.backgroundPrefetch.retryScheduled = false;
                runBackgroundPrefetch(jobID);
            }
        };

        const visibleChunkRequest = entry.activeChunkRequest;
        if (visibleChunkRequest) {
            // the visible frame path has priority, wait until it releases the decoder
            visibleChunkRequest.then(retry, retry);
        } else {
            // the decoder is busy with a request which is not tracked by activeChunkRequest
            setTimeout(retry, BACKGROUND_PREFETCH_RETRY_TIMEOUT);
        }
    };

    while (backgroundPrefetch.queued.length) {
        if (entry.decodeForward) {
            // the player is running, the visible frame path owns the decoder then
            retryWhenDecoderIsFree();
            return;
        }

        if (entry.activeChunkRequest || entry.provider.pendingChunkIndex() !== null) {
            // the visible frame path owns the decoder, retry when it is done
            retryWhenDecoderIsFree();
            return;
        }

        const chunkIndex = backgroundPrefetch.queued.shift() as number;
        if (entry.provider.isChunkCached(chunkIndex)) {
            // the chunk is already decoded, it does not need to be accounted as
            // prefetched: only chunks decoded here are released when they become stale
            continue;
        }

        backgroundPrefetch.inFlight = chunkIndex;
        entry.getChunk(chunkIndex, ChunkQuality.COMPRESSED).then((chunk: ArrayBuffer) => {
            if (frameDataCache[jobID]?.backgroundPrefetch !== backgroundPrefetch) {
                // the frame data cache entry is not relevant anymore
                return;
            }

            if (!backgroundPrefetch.targets.includes(chunkIndex)) {
                // the user moved on while the chunk was downloading
                backgroundPrefetch.inFlight = null;
                runBackgroundPrefetch(jobID);
                return;
            }

            if (
                entry.decodeForward || entry.activeChunkRequest ||
                entry.provider.pendingChunkIndex() !== null ||
                entry.provider.isChunkCached(chunkIndex)
            ) {
                // the visible frame path took the decoder over while the chunk was downloading
                backgroundPrefetch.inFlight = null;
                backgroundPrefetch.queued.unshift(chunkIndex);
                retryWhenDecoderIsFree();
                return;
            }

            announceDecodedChunkRequest(entry);
            entry.provider.requestDecodeBlock(
                chunk,
                chunkIndex,
                entry.segmentFrameNumbers.slice(
                    chunkIndex * entry.chunkSize,
                    (chunkIndex + 1) * entry.chunkSize,
                ),
                // decoded frames are not needed here, the chunk is prefetched for later use
                () => {},
                () => {
                    if (backgroundPrefetch.releaseDecodeRequest) {
                        backgroundPrefetch.releaseDecodeRequest();
                    }
                    if (frameDataCache[jobID]?.backgroundPrefetch === backgroundPrefetch) {
                        if (backgroundPrefetch.inFlight === chunkIndex) {
                            backgroundPrefetch.inFlight = null;
                        }
                        if (
                            entry.provider.isChunkCached(chunkIndex) &&
                            !backgroundPrefetch.decodedChunks.includes(chunkIndex)
                        ) {
                            backgroundPrefetch.decodedChunks.push(chunkIndex);
                        }

                        dropStalePrefetchedChunks(entry);
                        runBackgroundPrefetch(jobID);
                    }
                },
                () => {
                    // the request was rejected, most likely because the visible frame path
                    // requested another chunk in the meantime
                    if (backgroundPrefetch.releaseDecodeRequest) {
                        backgroundPrefetch.releaseDecodeRequest();
                    }
                    if (frameDataCache[jobID]?.backgroundPrefetch === backgroundPrefetch) {
                        if (backgroundPrefetch.inFlight === chunkIndex) {
                            backgroundPrefetch.inFlight = null;
                        }
                        runBackgroundPrefetch(jobID);
                    }
                },
            );
        }).catch(() => {
            if (frameDataCache[jobID]?.backgroundPrefetch !== backgroundPrefetch) {
                return;
            }

            backgroundPrefetch.inFlight = null;
            // a failed download drops the queued work, the next request recreates it
            backgroundPrefetch.queued = [];
        });
        return;
    }
}

/**
 * Requests background download and decode of the chunks which hold the given frames.
 * Intended for frames which are likely to be opened soon, for example upcoming rally
 * starts. The chunks are decoded one at a time, only when the visible frame path is
 * idle, and at most `BACKGROUND_PREFETCH_CHUNKS_LIMIT` of them are held. Chunks which
 * are already decoded or in progress are skipped. The call does not wait for the
 * downloads to finish, it resolves with the chunk indexes which were queued.
 *
 * @param jobID the job to prefetch frames in
 * @param frameNumbers frames to prepare chunks for
 * @param currentFrame the frame currently opened in the player, its chunk is never
 * prefetched and never evicted by the background prefetch
 */
export async function prefetchChunksForFrames(
    jobID: number,
    frameNumbers: number[],
    currentFrame?: number,
): Promise<number[]> {
    if (!Array.isArray(frameNumbers) || frameNumbers.some((frame) => !Number.isInteger(frame))) {
        throw new ArgumentError('Frame numbers must be an array of integers');
    }

    const entry = frameDataCache[jobID];
    if (!entry) {
        // no frame was requested for this job yet, nothing can be prefetched
        return [];
    }

    const meta = await entry.getMeta();
    const { jobStartFrame, backgroundPrefetch } = entry;
    const chunkIndexForFrame = (frame: number): number | null => {
        try {
            return meta.getFrameChunkIndex(meta.getDataFrameNumber(frame - jobStartFrame));
        } catch (error) {
            // the frame does not belong to the job
            return null;
        }
    };

    if (Number.isInteger(currentFrame)) {
        const visibleChunk = chunkIndexForFrame(currentFrame as number);
        if (visibleChunk !== null) {
            backgroundPrefetch.visibleChunk = visibleChunk;
        }
    }

    backgroundPrefetch.targets = _
        .uniq(frameNumbers
            .map((frame) => chunkIndexForFrame(frame))
            .filter((chunkIndex): chunkIndex is number => chunkIndex !== null))
        .filter((chunkIndex) => chunkIndex !== backgroundPrefetch.visibleChunk)
        .sort((chunkIndex1, chunkIndex2) => chunkIndex1 - chunkIndex2)
        .slice(0, BACKGROUND_PREFETCH_CHUNKS_LIMIT);

    // an in-flight chunk cannot be cancelled, it is accounted for in the completion callbacks
    backgroundPrefetch.queued = backgroundPrefetch.targets.filter((chunkIndex) => (
        chunkIndex !== backgroundPrefetch.inFlight && !entry.provider.isChunkCached(chunkIndex)
    ));

    const queuedChunks = backgroundPrefetch.queued.slice(0);
    dropStalePrefetchedChunks(entry);
    runBackgroundPrefetch(jobID);

    return queuedChunks;
}

export function clear(jobID: number): void {
    if (jobID in frameDataCache) {
        frameDataCache[jobID].provider.close();
        for (const contextImagesByFrame of Object.values(frameDataCache[jobID].contextCache)) {
            for (const image of Object.values(contextImagesByFrame.data)) {
                image.close();
            }
        }

        delete frameDataCache[jobID];
    }

    if (jobID in frameMetaCache) {
        delete frameMetaCache[jobID];
    }
}
