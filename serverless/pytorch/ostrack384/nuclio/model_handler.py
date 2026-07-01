import sys

import jsonpickle
import numpy as np

sys.path.append('/opt/nuclio/OSTrack')
sys.path.append('/opt/nuclio/OSTrack/lib')

local_py_path = '/opt/nuclio/OSTrack/lib/test/evaluation/local.py'
with open(local_py_path, 'w') as f:
    f.write("from lib.test.evaluation.environment import EnvSettings\n\n")
    f.write("def local_env_settings():\n")
    f.write("    settings = EnvSettings()\n")
    f.write("    settings.prj_dir = '/opt/nuclio/OSTrack'\n")
    f.write("    settings.save_dir = '/opt/nuclio/OSTrack/output'\n")
    f.write("    settings.network_path = '/opt/nuclio/OSTrack/output/test/networks'\n")
    f.write("    settings.results_path = '/opt/nuclio/OSTrack/output/test/tracking_results'\n")
    f.write("    settings.segmentation_path = '/opt/nuclio/OSTrack/output/test/segmentation_results'\n")
    f.write("    settings.result_plot_path = '/opt/nuclio/OSTrack/output/test/result_plots'\n")
    f.write("    return settings\n")

from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters


class ModelHandler:
    def __init__(self):
        params = parameters('vitb_384_mae_ce_32x4_ep300')
        params.checkpoint = '/ostrack384.pth'
        params.debug = False
        self.tracker = OSTrack(params, "otb")

    def decode_state(self, state):
        decode = jsonpickle.decode
        self.tracker.z_dict1 = decode(state["model.z_dict1"])
        self.tracker.state = decode(state["model.state"])

    def encode_state(self):
        state = {}
        state["model.z_dict1"] = jsonpickle.encode(self.tracker.z_dict1)
        state["model.state"] = jsonpickle.encode(self.tracker.state)
        return state

    def infer_batch(self, image, shapes, states):
        full_states = [states[i] if i < len(states) else None for i in range(len(shapes))]
        is_init = all(s is None for s in full_states)

        out_shapes, out_states = [], []

        if is_init:
            shape = shapes[0]
            bbox_xywh = [shape[0], shape[1], shape[2] - shape[0], shape[3] - shape[1]]
            self.tracker.initialize(image, {"init_bbox": bbox_xywh})
            encoded = self.encode_state()
            for shape in shapes:
                out_shapes.append(list(shape))
                out_states.append(encoded)
        else:
            state = full_states[0]
            self.decode_state(state)
            outputs = self.tracker.track(image)
            pred = outputs["target_bbox"]
            bbox = [pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3]]
            encoded = self.encode_state()
            for i, shape in enumerate(shapes):
                fallback = state.get("last_bbox", list(shape)) if state else list(shape)
                out_shapes.append(bbox)
                out_states.append({**encoded, "last_bbox": bbox})

        return out_shapes, out_states