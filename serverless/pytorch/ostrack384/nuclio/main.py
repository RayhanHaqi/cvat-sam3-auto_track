import json
import base64
import io

import numpy as np
from PIL import Image
from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Init OSTrack 384 context...")
    context.user_data.model = ModelHandler()
    context.logger.info("OSTrack 384 ready.")


def handler(context, event):
    context.logger.info("Run OSTrack 384 tracker")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = np.array(Image.open(buf).convert("RGB"))[:, :, ::-1].copy()
    shapes = data.get("shapes") or []
    states = data.get("states") or []

    out_shapes, out_states = context.user_data.model.infer_batch(image, shapes, states)

    return context.Response(
        body=json.dumps({"shapes": out_shapes, "states": out_states}),
        headers={},
        content_type="application/json",
        status_code=200,
    )
