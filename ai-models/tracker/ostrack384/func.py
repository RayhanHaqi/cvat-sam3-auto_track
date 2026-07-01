import dataclasses
import sys

sys.path.insert(0, '/opt/ostrack/OSTrack')
sys.path.insert(0, '/opt/ostrack/OSTrack/lib')

# OSTrack requires local.py to define environment paths before importing the tracker
_local_py = '/opt/ostrack/OSTrack/lib/test/evaluation/local.py'
with open(_local_py, 'w') as _f:
    _f.write(
        "from lib.test.evaluation.environment import EnvSettings\n\n"
        "def local_env_settings():\n"
        "    settings = EnvSettings()\n"
        "    settings.prj_dir = '/opt/ostrack/OSTrack'\n"
        "    settings.save_dir = '/opt/ostrack/OSTrack/output'\n"
        "    settings.network_path = '/opt/ostrack/OSTrack/output/test/networks'\n"
        "    settings.results_path = '/opt/ostrack/OSTrack/output/test/tracking_results'\n"
        "    settings.segmentation_path = '/opt/ostrack/OSTrack/output/test/segmentation_results'\n"
        "    settings.result_plot_path = '/opt/ostrack/OSTrack/output/test/result_plots'\n"
        "    return settings\n"
    )

import numpy as np
import PIL.Image
import cvat_sdk.auto_annotation as cvataa

from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters

_CHECKPOINT = '/opt/ostrack/ostrack384.pth'
_VARIANT = 'vitb_384_mae_ce_32x4_ep300'


@dataclasses.dataclass(kw_only=True)
class _TrackingState:
    tracker: object  # OSTrack instance carrying z_dict1 and state internally


class _OSTrack384Tracker:
    def __init__(self, device: str = 'cpu') -> None:
        self._device = device

    spec = cvataa.TrackingFunctionSpec(supported_shape_types=['rectangle'])

    def _new_tracker(self) -> OSTrack:
        params = parameters(_VARIANT)
        params.checkpoint = _CHECKPOINT
        params.debug = False
        return OSTrack(params, 'otb')

    def preprocess_image(
        self,
        context: cvataa.TrackingFunctionContext,
        image: PIL.Image.Image,
    ) -> np.ndarray:
        return np.array(image.convert('RGB'))[:, :, ::-1].copy()

    def init_tracking_state(
        self,
        context: cvataa.TrackingFunctionShapeContext,
        pp_image: np.ndarray,
        shape: cvataa.TrackableShape,
    ) -> _TrackingState:
        x1, y1, x2, y2 = (float(v) for v in shape.points[:4])
        tracker = self._new_tracker()
        tracker.initialize(pp_image, {'init_bbox': [x1, y1, x2 - x1, y2 - y1]})
        return _TrackingState(tracker=tracker)

    def track(
        self,
        context: cvataa.TrackingFunctionShapeContext,
        pp_image: np.ndarray,
        state: _TrackingState,
    ) -> cvataa.TrackableShape | None:
        outputs = state.tracker.track(pp_image)
        x, y, w, h = outputs['target_bbox']
        return cvataa.TrackableShape(type='rectangle', points=[x, y, x + w, y + h])


create = _OSTrack384Tracker
