#!/bin/bash
# Start SAM3 Nuclio tracker container (no rebuild). SAM2 is not deployed by default.
set -eu
ids=$(docker ps -aq --filter 'name=nuclio-meta-sam3')
if [ -z "$ids" ]; then
  echo "No SAM3 tracker container found. Deploy first: HF_TOKEN=... ./deploy_gpu.sh"
  exit 1
fi
docker start $ids
echo "Started SAM3 tracker. Check: nuctl get function --platform local"
