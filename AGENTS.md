# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

CVAT (Computer Vision Annotation Tool) is a multi-component annotation platform. The repo is a polyglot monorepo combining a Django/Python backend, a Yarn workspace of TypeScript packages for the frontend, a Python SDK/CLI, and Nuclio-based serverless functions for auto-annotation.

## Architecture

### Backend (Django) — `cvat/`
- `cvat/apps/` — Django apps. Key ones:
  - `engine` — core: tasks, jobs, projects, annotations, media handling, frame extraction (`frame_provider.py`, `media_extractors.py`), backup/import/export.
  - `dataset_manager` — dataset format conversion and task/job import/export workflows. Format plugins live in `formats/`. Note `bindings.py` is excluded from Black in pyproject.toml.
  - `lambda_manager` — integrates with Nuclio serverless functions (auto-annotation, trackers, interactors).
  - `iam` — auth, Open Policy Agent (OPA) rules for access control.
  - `quality_control`, `consensus` — QA features. `events` — activity logging. `webhooks`, `organizations`, `health`.
- `cvat/settings/` — Django settings (base/development/production/testing).
- `cvat/schema.yml` — generated OpenAPI schema (source for the SDK).
- Async work runs on Redis-backed RQ workers (`cvat/rqworker.py`, `rqscheduler.py`). Media/exports/imports are offloaded to queues, not request handlers.

### Frontend (Yarn workspaces, TypeScript)
Workspaces declared in the root `package.json`: `cvat-data`, `cvat-core`, `cvat-canvas`, `cvat-canvas3d`, `cvat-ui`. Packages link to each other via `link:./../<pkg>`, so the build order matters (data → core → canvas/canvas3d → ui).
- `cvat-ui` — React/Redux single-page app (Ant Design 5). Webpack-built.
- `cvat-core` — non-UI TS API layer that talks to the Django REST API; the domain model (tasks, jobs, annotations, collections).
- `cvat-canvas`, `cvat-canvas3d` — SVG/Three.js rendering and interaction for 2D/3D annotation workspaces.
- `cvat-data` — shared data decoders (e.g. video frame extraction in the browser).

### Python SDK & CLI
- `cvat-sdk/` — generated + hand-written Python client against `cvat/schema.yml`; published as `cvat-sdk`.
- `cvat-cli/` — CLI that depends on cvat-sdk; published as `cvat-cli`.

### Serverless (`serverless/`)
Auto-annotation functions packaged as Nuclio deployments (one folder per model with a `nuclio/` subdir containing `function.yaml` / `function-gpu.yaml` and `main.py`). Deployed with `nuctl` against a Nuclio cluster wired into the backend via `lambda_manager`.

### Tests (`tests/`)
- `tests/python/` — Pytest REST API / SDK / CLI integration tests. They spin up a dedicated `test_cvat_*_1` docker-compose stack and restore a fixture DB + data volume per test function (see `tests/python/README.md`).
- `tests/cypress/` — end-to-end UI tests. Configs: `cypress.config.js`, `cypress_canvas3d.config.js`.

## This Project's Labeling Workflow

CVAT is used to label ball position in ping pong frames. Only **annotate** frames (frames containing ball action) are uploaded — background frames stay local and are merged back after export via `tracknet_env/preprocess/merge_background.py`.

**Import:** upload `annotate_train.zip` + `annotate_val.zip` from `tracknet_env/datasets/raw_dataset/<session>/`
**Label:** use SAM2 or SAM3 auto-tracker to track the ball across frames (see SAM Tracker Deployments below)
**Export:** YOLO Detection format → `tracknet_env/datasets/formatted_dataset/YOLO_Detection/`

## SAM Tracker Deployments

Two Nuclio GPU functions deployed for CVAT auto-tracking. Both run in Docker and consume ~5.3 GB VRAM total when idle.

| Function | Port | Image | Docker Container | VRAM | Source |
|----------|------|-------|-----------------|------|--------|
| `meta-sam2-tracker-v2` | 32826 | `cvat.meta.sam2.tracker:v2-gpu` | `nuclio-meta-sam2-tracker-v2-*` | ~1.2 GB | `serverless/pytorch/facebookresearch/sam2/` |
| `meta-sam3-tracker-v4` | 32827 | `cvat.meta.sam3.tracker:v4-gpu` | `nuclio-meta-sam3-tracker-v4-*` | ~4.0 GB | `serverless/pytorch/facebookresearch/sam3/` |

### CVAT Invocation: Dashboard Mode

CVAT talks to Nuclio functions via `lambda_manager`. The `INVOKE_METHOD` controls routing:

- **`direct`** (default on Docker) — calls `http://host.docker.internal:{port}`, but Nuclio's reported `httpPort` mismatches Docker's actual mapped port, causing `500 Server Error`.
- **`dashboard`** — routes via `http://nuclio:8070/api/function_invocations` with `x-nuclio-function-name` header. This works reliably.

Set in `components/serverless/docker-compose.serverless.yml`:
```yaml
environment:
  CVAT_NUCLIO_INVOKE_METHOD: dashboard
```
Must be set on both `cvat_server` and `cvat_worker_annotation` services.

### Nuclio State Reset

Nuclio can enter an immortal **"building"** state that survives container removal, volume deletion, and Nuclio restart. The only reliable fix: redeploy with a new, never-before-used function name + image tag.

```bash
# Full reset procedure:
docker stop nuclio nuclio-local-storage-reader
docker rm nuclio-local-storage-reader
docker volume rm nuclio-local-storage
docker start nuclio
# Then redeploy with UNIQUE metadata.name and spec.build.image values
```

### SAM3 GPU Compatibility (RTX 5060 Ti / sm_120 Blackwell)

PyTorch Triton on Blackwell (`sm_120`) causes kernel segfaults. Workarounds in `function-gpu.yaml`:

```yaml
env:
  - name: TRITON_INTERPRET
    value: "1"
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: expandable_segments:True
```

Set `CUDA_LAUNCH_BLOCKING=1` only when debugging CUDA errors (it serializes kernels and slows inference).

`model_handler.py` monkey-patches two Triton-dependent SAM3 modules:

| Module | Replacement | Reason |
|--------|-------------|--------|
| `sam3.model.edt.edt_triton` | `cv2.distanceTransform` | Euclidean distance transform |
| `sam3.perflib.connected_components` | `skimage.measure.label` | Connected components labeling |

Also use `version="sam3"` (not `"sam3.1"`) in `build_sam3_predictor` — SAM3.1's multiplex `init_state()` doesn't accept the `offload_state_to_cpu` kwarg that `Sam3BasePredictor.start_session` passes.

### SAM3 Session Management

SAM3's `start_session(resource_path)` expects a directory of all frames upfront. CVAT sends frames one-at-a-time, so `model_handler.py`:
1. Saves each incoming frame as a JPEG to a temp directory
2. Tries incremental propagation on the warm session when a single new frame arrives
3. Reloads the SAM3 session only when incremental tracking fails (loads all accumulated frames)
4. Cleans up temp dirs when sessions are evicted or destroyed; returns HTTP 409 on stale `session_key`

SAM2 predictor state is not serialized into CVAT states — it requires `numWorkers: 1` and a warm worker.

### Build Environment

SAM3 image (`python:3.10` on `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`):
- **Deps:** `torch>=2.11`, `torchvision`, `opencv-python-headless`, `pycocotools`, `psutil`, `scikit-image`, `setuptools`
- **System:** `build-essential` (needed for Triton JIT `CC` compiler)
- **SAM3:** installed via `pip install -e .` from `facebookresearch/sam3` repo
- **Checkpoint:** HF model `sam3.pt` requires `HF_TOKEN` env var

### Management Commands

```bash
cd serverless
./sam-off.sh               # stop both trackers (frees ~5 GB VRAM)
./sam-on.sh                # start both trackers (no rebuild)
nuctl get function --platform local
```

```bash
# Deploy SAM2 + SAM3 (default allowlist)
cd serverless && HF_TOKEN=... ./deploy_gpu.sh

# Deploy SAM3 only
cd serverless/pytorch/facebookresearch/sam3
nuctl deploy --project-name cvat --path . --file function-gpu.yaml \
    --platform local --namespace nuclio \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --env HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before deploy}" \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```

After deploying new trackers, restart CVAT to discover them:
```bash
docker restart cvat_server cvat_worker_annotation
```

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `500 Server Error` on tracker call | Wrong `INVOKE_METHOD` | Set `CVAT_NUCLIO_INVOKE_METHOD=dashboard` |
| New tracker not showing in CVAT UI | CVAT hasn't discovered it | `docker restart cvat_server cvat_worker_annotation` |
| Segfault / CUDA error on SAM3 inference | Triton JIT on sm_120 | Set `TRITON_INTERPRET=1` + monkey-patches |
| "Function is in state: building" forever | Immortal Nuclio state | Full reset + redeploy with unique name |
| SAM3 `init_state()` TypeError | Wrong version string | Use `version="sam3"` not `"sam3.1"` |
| Triton JIT fails with "No such file: cc" | Missing C compiler | `apt install build-essential` |
| OOM during SAM3 inference | expandable_segments not set | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| SAM3 crash loop, `403` on `facebook/sam3` | HF token lacks gated-repo read | Fine-grained token: enable **public gated repositories**; redeploy with `HF_TOKEN` |
| SAM3 `401` / `GatedRepoError` | No or invalid `HF_TOKEN` at deploy | `export HF_TOKEN=...` then `./deploy_gpu.sh` |
| CVAT won't invoke tracker (not found) | Wrong network | `--platform-config '{"attributes": {"network": "cvat_cvat"}}'` |

**Security:** `HF_TOKEN` is stored in the Nuclio function spec and may appear in container debug logs. Never commit tokens; rotate after sharing logs. `token-hf.txt` is gitignored.

## Common commands

### Frontend
```sh
yarn install                      # install workspaces
yarn build:cvat-data && yarn build:cvat-core && \
  yarn build:cvat-canvas && yarn build:cvat-canvas3d && yarn build:cvat-ui
yarn start:cvat-ui                # dev server; proxies API to http://localhost:7000
yarn workspace cvat-ui run type-check
yarn workspace cvat-ui run lint           # or lint:fix
```
Per-package scripts live in each workspace's `package.json`. Lint runs via Husky + lint-staged on commit.

### Backend
```sh
# Full stack (recommended)
docker compose up -d              # adds -f docker-compose.dev.yml for dev

# Direct Django invocations (inside a configured env)
python manage.py migrate
python manage.py runserver 7000
python manage.py test cvat/apps   # Django unit tests

# Formatting
./dev/format_python_code.sh       # runs black + isort (line-length 100, py310)
```

### Python tests
```sh
pip install -r tests/python/requirements.txt
pytest ./tests/python                          # full integration suite (starts docker)
pytest ./tests/python --start-services         # bring up stack without running tests
pytest ./tests/python/rest_api/test_tasks.py::TestClass::test_name
pytest ./tests/python --rebuild                # rebuild images
```
Fixture DB lives in `tests/python/shared/assets/cvat_db/`; update via `python tests/python/shared/utils/dump_objects.py` after a schema change.

### Cypress
```sh
cd tests && npx cypress open --config-file cypress.config.js
cd tests && npx cypress run --spec cypress/e2e/<path>
```

## Working conventions

- **Python style**: Black + isort with `line-length = 100`, target `py310`. The `serverless/` tree is excluded from isort (each function has its own runtime). `cvat/apps/dataset_manager/bindings.py` and several `engine/*.py` files are currently Black-excluded — don't auto-format them unless the change is intentional.
- **Changelog**: user-visible changes require a fragment in `changelog.d/` (see existing files for format); `CHANGELOG.md` is assembled at release.
- **OpenAPI schema**: backend changes that touch serializers/views regenerate `cvat/schema.yml`; the Python SDK is generated from it, so regenerate SDK stubs when the schema changes.
- **Serverless function edits**: each `function.yaml` pins its own base image and deps. Don't share Python deps across functions.
- **Node/Yarn**: this repo uses Yarn 4 (`packageManager: yarn@4.9.2`) — use `yarn`, not `npm`, especially for workspace commands.

## Primary branch

`develop` is the integration branch (also the PR target). Releases are cut to versioned branches.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Judgment, Not Automation

**Use the model for synthesis and reasoning. Don't delegate deterministic logic.**

- Use me for: classification, drafting, summarization, extraction from unstructured text.
- Do NOT use me for: routing, retries, status-code handling, deterministic transforms.
- If plain code can answer, plain code answers.

## 6. Hard Token Budgets

**Every loop can spiral. Set limits before you start.**

- Per-task budget: 4,000 tokens. Per-session: 30,000 tokens.
- If approaching budget, summarize and start fresh. Do not push through.
- Surfacing the breach is better than silently overrunning.

## 7. Surface Conflicts, Don't Average

**When two parts of the codebase disagree, don't blend them.**

- If two existing patterns contradict, pick one (the more recent / more tested).
- Explain why and flag the other for cleanup.
- "Average" code that satisfies both rules is the worst code.

## 8. Read Before You Write

**Understand adjacent code before adding to it.**

- Before adding code in a file, read the file's exports, callers, and shared utilities.
- If you don't understand why existing code is structured the way it is, ask before adding.
- "Looks orthogonal to me" is the most dangerous phrase in this codebase.

## 9. Tests Verify Intent, Not Just Behavior

**Tests pass is not the goal. Correct behavior is the goal.**

- Every test must encode WHY the behavior matters, not just WHAT it does.
- A test that can't fail when business logic changes is wrong.
- If you can't write a test that would fail on incorrect output, the function is wrong.

## 10. Checkpoint After Every Step

**Multi-step work needs intermediate verification.**

- After completing each step: summarize what was done, what's verified, what's left.
- Don't continue from a state you can't describe back.
- If you lose track, stop and restate.

## 11. Convention Beats Novelty

**Inside the codebase, conformance trumps taste.**

- If the codebase uses snake_case and you'd prefer camelCase: snake_case.
- If the codebase uses class-based patterns and you'd prefer hooks: class-based.
- Disagreement is a separate conversation. Don't fork conventions silently.

## 12. Fail Loud

**Silent failures are the most expensive ones.**

- "Completed" is wrong if anything was skipped silently.
- "Tests pass" is wrong if any were skipped.
- "Feature works" is wrong if you didn't verify the edge case.
- Default to surfacing uncertainty, not hiding it.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
