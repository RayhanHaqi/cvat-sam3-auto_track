FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    wget \
    && rm -rf /var/lib/apt/lists/*

ARG USE_GPU=false
ARG TARGETARCH
ADD --chmod=+x https://dl.k8s.io/release/v1.34.0/bin/linux/${TARGETARCH}/kubectl /usr/local/bin/

RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid 1000 --create-home agent

RUN mkdir /shared && chown agent:agent /shared

WORKDIR /opt/ostrack

RUN git clone https://github.com/botaoye/OSTrack.git OSTrack && \
    sed -i 's/from torch._six import string_classes/string_classes = (str, bytes)/g' OSTrack/lib/train/data/loader.py

COPY ai-models/tracker/ostrack384/requirements.txt .

RUN if [ "$USE_GPU" = "true" ] && [ "$TARGETARCH" = "amd64" ]; then \
        echo "Building GPU image" && pip install --no-cache-dir cvat-cli -r requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu124; \
    else \
        echo "Building CPU only image" && pip install --no-cache-dir cvat-cli -r requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN gdown 19tj5NmwZr9ylDaqOwfNVOLijr389ZDBi -O /opt/ostrack/ostrack384.pth

WORKDIR /app

COPY ai-models/tracker/ostrack384/ .
COPY --chmod=755 ai-models/agents_deployment/ostrack384/function_registration.sh .
COPY --chmod=755 ai-models/agents_deployment/ostrack384/function_deregistration.sh .
COPY --chmod=755 ai-models/agents_deployment/ostrack384/check_env.sh .
COPY --chmod=755 ai-models/agents_deployment/ostrack384/entrypoint.sh .

USER agent

ENTRYPOINT ["./entrypoint.sh"]
