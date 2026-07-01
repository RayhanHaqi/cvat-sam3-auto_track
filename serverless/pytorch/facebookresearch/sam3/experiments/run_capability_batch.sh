#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${SAM3_CAPABILITY_CONTAINER:-nuclio-nuclio-meta-sam3-tracker-v4}"
SCRIPT_HOST="/home/tilakoid/pingpong/cvat/serverless/pytorch/facebookresearch/sam3/experiments/capability_test_propagate_modes.py"
OUT_HOST="/home/tilakoid/pingpong/cvat/serverless/pytorch/facebookresearch/sam3/experiments/outputs"
SAM_SS="/home/tilakoid/pingpong/SAM_SS"

docker exec "${CONTAINER}" mkdir -p /opt/nuclio/experiments /tmp/sam3_videos
docker cp "${SCRIPT_HOST}" "${CONTAINER}:/opt/nuclio/experiments/capability_test_propagate_modes.py"

run_case() {
  local name="$1"
  local video_host="$2"
  local production_max="$3"
  local init_bbox="${4:-}"
  local out_name="${name}_max${production_max}"

  docker cp "${video_host}" "${CONTAINER}:/tmp/sam3_videos/${name}.mp4"
  local cmd=(python3 /opt/nuclio/experiments/capability_test_propagate_modes.py
    --video "/tmp/sam3_videos/${name}.mp4"
    --max-frame 95
    --frame-count 96
    --output "/tmp/sam3_capability_${out_name}"
    --production-max "${production_max}")
  if [[ -n "${init_bbox}" ]]; then
    cmd+=(--init-bbox "${init_bbox}")
  fi
  echo "=== RUN ${out_name} ==="
  docker exec "${CONTAINER}" "${cmd[@]}"
  mkdir -p "${OUT_HOST}/${out_name}"
  docker cp "${CONTAINER}:/tmp/sam3_capability_${out_name}/." "${OUT_HOST}/${out_name}/"
}

for video in SAM3 SAM3_2 SAM3_3; do
  run_case "${video}" "${SAM_SS}/${video}.mp4" 1 ""
  run_case "${video}" "${SAM_SS}/${video}.mp4" 0 ""
done

echo "All capability runs complete. Outputs under ${OUT_HOST}"
