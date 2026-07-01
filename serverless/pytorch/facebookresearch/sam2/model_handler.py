import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor


class ModelHandler:
    # SAM2 camera predictor state lives in GPU memory and is not serialized into
    # CVAT states (only bbox metadata is returned). Requires numWorkers: 1 and a
    # warm, sticky Nuclio worker; re-seed from the annotation frame after restart.
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam2_camera_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "/opt/nuclio/sam2/checkpoints/sam2.1_hiera_large.pt",
        )

    def _mask_to_bbox(self, logits, fallback):
        mask = (logits > 0.0).cpu().numpy().squeeze()
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return list(fallback) if fallback is not None else None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h, w = mask.shape[:2]
        return [
            float(max(0, x_min)),
            float(max(0, y_min)),
            float(min(w - 1, x_max)),
            float(min(h - 1, y_max)),
        ]

    def infer_batch(self, image, shapes, states):
        full_states = [states[i] if i < len(states) else None for i in range(len(shapes))]
        is_init = all(s is None for s in full_states)

        if is_init:
            self.predictor.load_first_frame(image)
            last_ids, last_logits = [], None
            for i, shape in enumerate(shapes):
                box = np.array(shape, dtype=np.float32)
                _, last_ids, last_logits = self.predictor.add_new_prompt(
                    frame_idx=0, obj_id=i + 1, bbox=box
                )
            out_shapes, out_states = [], []
            for i, shape in enumerate(shapes):
                oid = i + 1
                ids_list = list(last_ids) if last_ids is not None else []
                if oid in ids_list:
                    idx = ids_list.index(oid)
                    bbox = self._mask_to_bbox(last_logits[idx], shape)
                else:
                    bbox = list(shape)
                out_shapes.append(bbox)
                out_states.append({"obj_id": oid, "last_bbox": bbox})
            return out_shapes, out_states

        # propagate — call track once for the whole frame
        active_indices = [
            i for i in range(len(shapes))
            if not (full_states[i] or {}).get("lost")
        ]
        if not active_indices:
            out_shapes, out_states = [], []
            for i, _shape in enumerate(shapes):
                state = full_states[i] or {"obj_id": i + 1}
                lost_state = dict(state)
                lost_state["lost"] = True
                out_shapes.append(None)
                out_states.append(lost_state)
            return out_shapes, out_states

        out_ids, out_logits = self.predictor.track(image)
        ids_list = list(out_ids) if out_ids is not None else []
        out_shapes, out_states = [], []
        for i, shape in enumerate(shapes):
            state = full_states[i] or {"obj_id": i + 1}
            if state.get("lost"):
                lost_state = dict(state)
                lost_state["lost"] = True
                out_shapes.append(None)
                out_states.append(lost_state)
                continue

            oid = state.get("obj_id", i + 1)
            if oid in ids_list:
                idx = ids_list.index(oid)
                bbox = self._mask_to_bbox(out_logits[idx], None)
            else:
                bbox = None

            new_state = dict(state)
            if bbox is None:
                new_state["lost"] = True
                new_state["last_bbox"] = None
                out_shapes.append(None)
            else:
                new_state.pop("lost", None)
                new_state["last_bbox"] = bbox
                out_shapes.append(bbox)
            out_states.append(new_state)
        return out_shapes, out_states
