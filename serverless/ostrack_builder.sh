#!/bin/bash

LOG_FILE="latest.log"
FUNC_PATH="pytorch/botaoye/ostrack/nuclio/"

nuctl delete function pth-botaoye-ostrack-384
docker rmi cvat.pth.botaoye.ostrack_384:latest 2>/dev/null
docker builder prune -f

{
    # Jalankan deploy
    ./deploy_gpu.sh "$FUNC_PATH"

    sleep 5
    docker ps | grep ostrack
    
} 2>&1 | tee "$LOG_FILE"