#!/bin/bash
# Sample commands to deploy nuclio functions on GPU

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

# Default allowlist: SAM3 tracker only for ping-pong ball labeling.
# Pass "all" as second arg to deploy every function-gpu.yaml under FUNCTIONS_DIR.
# To deploy SAM2 as well: ALLOWLIST+=(pytorch/facebookresearch/sam2) before running, or use "all".
ALLOWLIST=(
    pytorch/facebookresearch/sam3
)

nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function-gpu.yaml
do
    case "$func_config" in
        */hkchengrex/*) echo "Skipping $func_config (not needed for this project)"; continue ;;
    esac

    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$func_root")"

    if [ "${2:-}" != "all" ]; then
        allowed=false
        for entry in "${ALLOWLIST[@]}"; do
            if [ "$func_rel_path" = "$entry" ]; then
                allowed=true
                break
            fi
        done
        if [ "$allowed" = false ]; then
            echo "Skipping $func_rel_path (not in default allowlist; pass 'all' to deploy everything)"
            continue
        fi
    fi

    echo "Deploying $func_rel_path function..."
    deploy_env=(
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk
        --env CVAT_FUNCTIONS_REDIS_PORT=6666
    )
    if [ "$func_rel_path" = "pytorch/facebookresearch/sam3" ]; then
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "ERROR: HF_TOKEN must be set to deploy SAM3" >&2
            exit 1
        fi
        deploy_env+=(
            --env "HF_TOKEN=${HF_TOKEN}"
            --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
        )
        for var in SAM3_PROMPT_MODE SAM3_TEXT_PROMPT SAM3_OUTPUT_POLICY SAM3_OUTPUT_PROB_THRESH SAM3_IR_REFINE SAM3_IR_STOP_ON_MISSING; do
            if [ -n "${!var:-}" ]; then
                deploy_env+=(--env "$var=${!var}")
            fi
        done
    fi

    nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        "${deploy_env[@]}" \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

nuctl get function --platform local
