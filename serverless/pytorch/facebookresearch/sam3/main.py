import base64
import ctypes
import io
import json
import time

import numpy as np
from PIL import Image

from diagnostics import log_event, timed_stage
from model_handler import (
    ModelHandler,
    PreloadRangeExhaustedError,
    SessionStaleError,
    ValidationError,
    load_sam3_config,
)


def _set_process_name(name: str) -> None:
    """Set process title for ps and nvidia-smi (argv[0] + Linux comm)."""
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except Exception:
        pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(15, name.encode()[:15], 0, 0, 0)  # PR_SET_NAME
    except Exception:
        pass


def init_context(context):
    _set_process_name("sam3")
    config = load_sam3_config()
    context.logger.info("Init SAM 3.1 context...")
    context.logger.info(
        "SAM3 config: prompt_mode=%s output_policy=%s prob_thresh=%s ir_refine=%s ir_stop_on_missing=%s text_prompt=%r",
        config.prompt_mode,
        config.output_policy,
        config.output_prob_thresh,
        config.ir_refine_enabled,
        config.ir_stop_on_missing,
        config.text_prompt if config.prompt_mode != "box" else "",
    )
    context.user_data.model = ModelHandler(config=config)
    context.logger.info("SAM 3.1 ready.")


def handler(context, event):
    context.logger.info("Run SAM 3.1 tracker")
    try:
        data = event.body
        diag_meta = data.pop("_autoTrackDiag", None)
        decode_ms = 0.0
        image = None
        if "image" in data and data["image"] is not None:
            decode_start = time.perf_counter()
            buf = io.BytesIO(base64.b64decode(data["image"]))
            image = np.array(Image.open(buf).convert("RGB"))
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
        shapes = data.get("shapes") or []
        states = data.get("states") or []
        preload_images = data.get("preload_images")
        preload_base_frame = data.get("preload_base_frame")
        preload_frame_count = data.get("preload_frame_count")
        frame_index = data.get("frame_index")
        preload_payload_bytes = None
        if preload_images is not None:
            preload_payload_bytes = len(
                json.dumps(
                    {
                        "preload_images": preload_images,
                        "preload_base_frame": preload_base_frame,
                        "preload_frame_count": preload_frame_count,
                    },
                    default=str,
                )
            )

        log_event(
            "sam_handler_start",
            diagMeta=diag_meta,
            imageDecodeMs=decode_ms,
            shapeCount=len(shapes),
            stateCount=len(states),
            preloadFrameCount=preload_frame_count,
            preloadPayloadBytes=preload_payload_bytes,
            frameIndex=frame_index,
        )

        with timed_stage("sam_inference", diagMeta=diag_meta) as stage:
            out_shapes, out_states = context.user_data.model.infer_batch(
                image,
                shapes,
                states,
                diag_meta=diag_meta,
                frame_index=frame_index,
                preload_images=preload_images,
                preload_base_frame=preload_base_frame,
                preload_frame_count=preload_frame_count,
                preload_payload_bytes=preload_payload_bytes,
            )
            stage["samSessionCount"] = len(context.user_data.model._sessions)

        response_body = {"shapes": out_shapes, "states": out_states}
        log_event(
            "sam_handler_end",
            diagMeta=diag_meta,
            responsePayloadBytes=len(json.dumps(response_body, default=str)),
            samSessionCount=len(context.user_data.model._sessions),
        )

        return context.Response(
            body=json.dumps(response_body),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except SessionStaleError as exc:
        context.logger.warn(str(exc))
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=409,
        )
    except ValidationError as exc:
        context.logger.warn(str(exc))
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    except PreloadRangeExhaustedError as exc:
        context.logger.warn(str(exc))
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    except Exception as exc:
        context.logger.error(f"SAM3 tracker failed: {exc}")
        return context.Response(
            body=json.dumps({"error": "internal tracker error"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
