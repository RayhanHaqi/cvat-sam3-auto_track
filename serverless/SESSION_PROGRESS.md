# Session Progress — CVAT Serverless SAM Trackers

**Date:** 2026-06-24  
**Scope:** Review, fixes, HF token setup, Nuclio redeploy prep

---

## 1. Code Review Summary

Reviewed the `serverless/` SAM tracker work (SAM2, SAM3, OSTrack) for ping-pong ball labeling in CVAT.

**Verdict:** Functional for warm, single-worker tracking. SAM3 had performance/resource issues and a critical security finding.

| Severity | Count |
|----------|-------|
| Bugs | 8 |
| Suggestions | 5 |
| Nits | 3 |

Full review artifacts (if still present): `/tmp/grok-review-33a77e5f.md`, `/tmp/grok-review-summary-33a77e5f.md`

---

## 2. Fixes Implemented

### Security
- **Removed hardcoded `HF_TOKEN`** from `pytorch/facebookresearch/sam3/function-gpu.yaml`
- Token must be passed at deploy time via `HF_TOKEN` env var
- **Action still required:** Revoke the old leaked token on Hugging Face (it remains in git history)

### SAM3 (`model_handler.py`, `main.py`, `function-gpu.yaml`)
- Session persistence: try incremental propagation before full session reload
- Temp-dir cleanup via `_destroy_session()` + LRU cap (32 sessions)
- Stale `session_key` → HTTP 409; empty `shapes` → HTTP 400
- `main.py` uses `context.user_data.model` with structured error handling
- Removed default `CUDA_LAUNCH_BLOCKING=1` (debug-only now)
- `flash-attn` install failure logs a warning instead of silent `|| true`
- Description updated to "single-object video tracker"
- Python 3.10 in build spec (was 3.11)

### Deploy (`deploy_gpu.sh`)
- Default allowlist: **SAM2 + SAM3 only** (OSTrack skipped unless `./deploy_gpu.sh . all`)
- SAM3 deploy requires `HF_TOKEN`; script exits if unset

### Repo hygiene
- `.gitignore` entries for `serverless/**/*.log`, `*.pth`, `*.pth.tar`, `weights/`
- Removed ~600 MB of tracked logs/binaries from git index (files remain on disk)

### OSTrack
- `ostrack/nuclio/main.py`: `or []` defaults on shapes/states
- `ostrack384/nuclio/model_handler.py`: `jsonpickle` encode/decode state
- Removed placeholder comment in `ostrack/nuclio/model_handler.py`

### SAM2
- Comment documenting single-worker / non-serialized predictor state requirement

### Docs (`AGENTS.md`)
- Updated session management, Python 3.10, deploy commands, `CUDA_LAUNCH_BLOCKING` note, `HF_TOKEN` deploy flag

---

## 3. Hugging Face Token Setup

SAM3 downloads the gated checkpoint from [facebook/sam3](https://huggingface.co/facebook/sam3).

### Account access (required first)
1. Log in at https://huggingface.co/facebook/sam3
2. Agree to Meta's gated access form
3. Confirm model files are visible (not "request access")

### Token configuration
- **Type:** Fine-grained, **Read** only
- **Name:** `CVAT SAM3`
- **Minimum permission:** Read access to contents of all public gated repos you can access  
  **OR** repo-scoped read on `facebook/sam3` only
- **Do not enable:** Write, Inference, Webhooks, Org permissions, etc.

### Verify locally
```bash
conda activate pingpong
pip install huggingface_hub   # if missing
export HF_TOKEN=hf_your_token_here

hf auth whoami
# or: /home/tilakoid/miniconda3/envs/pingpong/bin/hf auth whoami

# Lightweight access check (no full download):
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
print('User:', api.whoami().get('name'))
info = api.model_info('facebook/sam3')
print('Model access OK')
"
```

---

## 4. Nuclio State (2026-06-24)

### Before redeploy
| Function | State | Notes |
|----------|-------|-------|
| `meta-sam3-tracker-v3` | ready | Healthy, port 32792 |
| `meta-sam2-tracker-v2` | building | Immortal provisioning bug; container exited |

### Fix for stuck function
Normal delete fails with "Function is being provisioned". Use force:

```bash
nuctl delete function meta-sam2-tracker-v2 --platform local --force
```

**Do not remove the Nuclio platform itself** unless force delete keeps failing.

### Full reset (last resort only)
```bash
docker stop nuclio nuclio-local-storage-reader
docker rm nuclio-local-storage-reader
docker volume rm nuclio-local-storage
docker start nuclio
# Redeploy; if still stuck, use new metadata.name + image tag in function-gpu.yaml
```

---

## 5. Redeploy Procedure

```bash
conda activate pingpong
export HF_TOKEN=hf_your_token_here
cd ~/pingpong/cvat/serverless
./deploy_gpu.sh

nuctl get function --platform local
docker restart cvat_server cvat_worker_annotation
```

Expected result: both `meta-sam2-tracker-v2` and `meta-sam3-tracker-v3` in **ready** state.

Deploy all GPU functions (not just SAM):
```bash
./deploy_gpu.sh . all
```

---

## 6. Tracker Reference

| Function | Port (example) | Image | VRAM (idle) |
|----------|----------------|-------|-------------|
| `meta-sam2-tracker-v2` | varies | `cvat.meta.sam2.tracker:v2-gpu` | ~1.2 GB |
| `meta-sam3-tracker-v3` | 32792 | `cvat.meta.sam3.tracker:v3-gpu` | ~4.0 GB |

CVAT invoke method must be `dashboard` (see `AGENTS.md`).

Management:
```bash
sam-off    # stop both trackers (free VRAM)
sam-on     # start both trackers
nuctl get function --platform local
```

---

## 7. Remaining / Follow-up

- [ ] Revoke old `HF_TOKEN` on Hugging Face (was committed in yaml)
- [ ] Redeploy SAM2 + SAM3 after force-delete (user confirmed `--force` done)
- [ ] Confirm both functions show **ready** after `./deploy_gpu.sh`
- [ ] Test tracking in CVAT UI on a short annotate sequence
- [ ] Optional: `git commit` the code changes when satisfied

---

## 8. Files Changed This Session

```
.gitignore
AGENTS.md
serverless/deploy_gpu.sh
serverless/pytorch/facebookresearch/sam2/model_handler.py
serverless/pytorch/facebookresearch/sam3/function-gpu.yaml
serverless/pytorch/facebookresearch/sam3/main.py
serverless/pytorch/facebookresearch/sam3/model_handler.py
serverless/pytorch/ostrack/nuclio/main.py
serverless/pytorch/ostrack/nuclio/model_handler.py
serverless/pytorch/ostrack384/nuclio/model_handler.py
```

Git index also removed (still on disk, now gitignored):
- `serverless/sam2_build.log`, `ostrack_build.log`, `transt_build.log`, `latest.log`
- `serverless/pytorch/ostrack/nuclio/OSTrack_ep0300.pth.tar`
- `serverless/weights/deaot.pth`