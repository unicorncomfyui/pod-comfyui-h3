# RunPod ComfyUI Pod with VSCode - MiniMax H3 / RTX 5090 (Blackwell sm_120)
#
# Single parameterised Dockerfile. The CUDA target is selected at build time so
# the same source produces the whole matrix:
#
#   cu130 (default)  -> requires NVIDIA driver 580+ -> full NVFP4 tensor-core path
#   cu129 (fallback) -> requires NVIDIA driver 575+ -> wider RunPod availability
#
#   docker build --build-arg CUDA_BASE=12.9.2-cudnn-runtime-ubuntu24.04 \
#                --build-arg TORCH_INDEX=cu129 .
#
# Nothing is compiled from source: comfy-kitchen (kernels) and comfy-aimdo
# (dynamic VRAM offloader) ship as prebuilt abi3 wheels, so the image is based
# on -runtime rather than -devel and needs no SageAttention build step.

ARG CUDA_BASE=13.3.1-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_BASE}

ARG TORCH_INDEX=cu130
ARG TORCH_VERSION=2.13.0
ARG PYTHON_VERSION=3.13
ARG COMFYUI_VERSION=v0.30.0
ARG CODE_SERVER_VERSION=4.131.0

LABEL maintainer="ComfyUI Pod VSCode - MiniMax H3" \
      description="RunPod Pod: ComfyUI + VSCode, MiniMax H3 video generation, RTX 5090 Blackwell" \
      gpu.target="RTX 5090 (sm_120)" \
      model.primary="MiniMax-H3"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="12.0" \
    COMFYUI_PORT=3000 \
    VSCODE_PORT=8080 \
    HF_HOME=/workspace/.cache/huggingface \
    DOWNLOAD_MODELS=true \
    MODEL_SETS="minimax-h3-fl2va,minimax-h3-ref2va"

# Record the build parameters so a running pod can report exactly what it is.
ENV BUILD_TORCH_INDEX=${TORCH_INDEX} \
    BUILD_TORCH_VERSION=${TORCH_VERSION} \
    BUILD_COMFYUI_VERSION=${COMFYUI_VERSION} \
    BUILD_PYTHON_VERSION=${PYTHON_VERSION}

# ---------------------------------------------------------------------------
# Layer 1 - runtime system packages, Python and code-server.
# No build toolchain here: it is installed and purged inside layer 2 so it
# never lands in a published layer.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
        git git-lfs wget curl unzip nano vim jq \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg \
        openssh-server rsync \
        google-perftools libtcmalloc-minimal4 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=${CODE_SERVER_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ---------------------------------------------------------------------------
# Layer 2 - PyTorch, ComfyUI and the pruned custom-node set.
# The build toolchain is installed and removed within this single RUN so the
# resulting layer carries none of it (removing it in a later RUN would save
# nothing, since layers are additive).
# ---------------------------------------------------------------------------
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config \
    # PyTorch for the selected CUDA target. Installed first so that ComfyUI's
    # bare `torch` requirement resolves as already-satisfied instead of pulling
    # the default PyPI build.
    && pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    # ComfyUI. The project moved from comfyanonymous/ComfyUI to Comfy-Org/ComfyUI.
    && git clone https://github.com/Comfy-Org/ComfyUI.git comfyui \
    && cd comfyui \
    && git checkout ${COMFYUI_VERSION} \
    && rm -rf .git \
    # Pulls comfy-kitchen (NVFP4 / int8-convrot kernels) and comfy-aimdo
    # (dynamic VRAM offloader) - both required by MiniMax H3.
    && pip install --no-cache-dir -r requirements.txt \
    # Fast, resumable, chunk-deduplicated model downloads (H3 is 63 GB).
    && pip install --no-cache-dir "huggingface_hub[hf_xet]>=1.26.0" \
    && cd custom_nodes \
    # Pruned node set: video-oriented only. The image-era nodes (WAS suite,
    # Impact Pack, Comfyroll, RES4LYF, ...) are intentionally dropped - they
    # weigh a lot and conflict more often on each ComfyUI core bump.
    && git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
    && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git \
    && git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    && git clone --depth 1 https://github.com/chrisgoringe/cg-use-everywhere.git \
    # ComfyUI-Easy-Use is deliberately absent: it depends on clip_interrogator
    # 0.6.0, last released March 2023, which is not a safe bet against
    # transformers 5.x. It is image-workflow QoL with little value on a video pod.
    && for dir in /app/comfyui/custom_nodes/*/; do \
        if [ -f "${dir}requirements.txt" ]; then \
            pip install --no-cache-dir -r "${dir}requirements.txt" || \
                echo "[WARN] requirements failed for ${dir}"; \
        fi; \
    done \
    # Frame-Interpolation ships no requirements.txt - it offers a cupy and a
    # no-cupy variant. Take no-cupy: cupy would need nvcc, which a -runtime
    # base image does not have.
    && pip install --no-cache-dir \
        -r /app/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt \
        || echo "[WARN] Frame-Interpolation requirements failed" \
    # VideoHelperSuite asks for opencv-python while KJNodes asks for
    # opencv-python-headless; both provide cv2 and whichever lands last wins.
    # Settle it explicitly on the headless build - this is a container.
    && pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless \
    # Strip node git metadata (~1-2 GB) and any bundled weights.
    && find /app/comfyui -name ".git" -type d -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /app/comfyui/custom_nodes \( -name "*.pth" -o -name "*.safetensors" \) -size +50M -delete \
    # Purge the toolchain inside the same layer.
    && apt-get purge -y build-essential cmake ninja-build pkg-config \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/*

# ---------------------------------------------------------------------------
# Layer 3 - upscaler, configs and entrypoint scripts.
# Model weights are NOT baked in: 63 GB of H3 lands on the network volume at
# first boot. Measured result: 6.27 GB compressed for cu130, 8.46 GB for cu129.
# ---------------------------------------------------------------------------
RUN mkdir -p /app/comfyui/models/upscale_models \
             /app/comfyui/user/default/workflows \
             /root/.config/code-server \
             /root/.local/share/code-server/User \
    && wget -q -O /app/comfyui/models/upscale_models/4x-UltraSharp.pth \
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"

COPY models/manifest.json /app/models/manifest.json
COPY scripts/download_models.py /app/scripts/download_models.py
COPY init.sh start.sh fetch_models.sh /app/
COPY config/code-server-config.yaml /root/.config/code-server/config.yaml
COPY config/vscode-settings.json /root/.local/share/code-server/User/settings.json

# Normalise line endings in case of a checkout from Windows.
RUN sed -i 's/\r$//' /app/init.sh /app/start.sh /app/fetch_models.sh \
    && chmod +x /app/init.sh /app/start.sh /app/fetch_models.sh

EXPOSE 8080 3000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${COMFYUI_PORT}/ || exit 1

CMD ["/app/start.sh"]
