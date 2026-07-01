#!/bin/bash
# Stop SAM3 Nuclio tracker container to free GPU VRAM (~4 GB idle).
set -eu
ids=$(docker ps -q --filter 'name=nuclio-meta-sam3')
if [ -z "$ids" ]; then
  echo "No SAM3 tracker container running."
  exit 0
fi
docker stop $ids
echo "Stopped SAM3 tracker. Free VRAM with: nvidia-smi"
