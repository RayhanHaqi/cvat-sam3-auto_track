import base64
import ctypes
import io
import json

import numpy as np
from PIL import Image
from model_handler import ModelHandler


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
    _set_process_name("sam2")
    context.logger.info("Init SAM 2.1 context...")
    context.user_data.model = ModelHandler()
    context.logger.info("SAM 2.1 ready.")


def handler(context, event):
    context.logger.info("Run SAM 2.1 tracker")
    try:
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        image = np.array(Image.open(buf).convert("RGB"))
        shapes = data.get("shapes") or []
        states = data.get("states") or []

        out_shapes, out_states = context.user_data.model.infer_batch(image, shapes, states)

        return context.Response(
            body=json.dumps({"shapes": out_shapes, "states": out_states}),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except Exception as exc:
        context.logger.error(f"SAM2 tracker failed: {exc}")
        return context.Response(
            body=json.dumps({"error": "internal tracker error"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )