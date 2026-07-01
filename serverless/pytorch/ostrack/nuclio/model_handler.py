import jsonpickle
import numpy as np
import torch
import sys
import os

# 1. Tambahkan path lib ke environment
sys.path.append('/opt/nuclio/OSTrack')
sys.path.append('/opt/nuclio/OSTrack/lib')

# 2. TRIK PAKSA: Tulis ulang file local.py secara dinamis (Dengan fungsi yang diminta!)
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

# 3. Sekarang import OSTrack
from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters

class ModelHandler:
    def __init__(self):
        params = parameters('vitb_256_mae_ce_32x4_ep300')
        params.checkpoint = '/ostrack.pth'
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

    def init_tracker(self, img, bbox):
        init_info = {"init_bbox": bbox}
        self.tracker.initialize(img, init_info)

    def track(self, img):
        outputs = self.tracker.track(img)
        prediction_bbox = outputs["target_bbox"]
        left = prediction_bbox[0]
        top = prediction_bbox[1]
        right = prediction_bbox[0] + prediction_bbox[2]
        bottom = prediction_bbox[1] + prediction_bbox[3]
        return (left, top, right, bottom)

    def infer(self, image, shape, state):
        if state is None:
            init_shape = [shape[0], shape[1], shape[2] - shape[0], shape[3] - shape[1]]
            self.init_tracker(image, init_shape)
            state = self.encode_state()
        else:
            self.decode_state(state)
            shape = self.track(image)
            state = self.encode_state()

        return shape, state