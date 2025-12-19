# RunPod ComfyUI Pod with VSCode - RTX 5090 (OPTIMIZED)
# Optimized for RTX 5090 with CUDA 12.8.1, SageAttention, and code-server
# Base: Ubuntu 22.04 + CUDA 12.8.1-cudnn

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Metadata
LABEL maintainer="ComfyUI Pod VSCode RTX5090"
LABEL description="RunPod Pod with ComfyUI, VSCode (code-server), SageAttention - Optimized for RTX 5090"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="8.9" \
    COMFYUI_PORT=3000 \
    VSCODE_PORT=8080 \
    CHECK_MODELS=true

# Install system dependencies, Python 3.11, and code-server in ONE layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials (will be removed later)
    build-essential cmake ninja-build pkg-config \
    # Python 3.11
    software-properties-common \
    # Git and tools
    git git-lfs wget curl unzip nano vim \
    # Media libraries
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    # Network tools
    openssh-server rsync \
    # Performance optimization
    google-perftools libtcmalloc-minimal4 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=4.96.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install PyTorch nightly + ComfyUI + ALL custom nodes in ONE optimized layer
WORKDIR /app
ARG COMFYUI_COMMIT=36357bb
RUN pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && git clone https://github.com/comfyanonymous/ComfyUI.git comfyui \
    && cd comfyui \
    && git reset --hard ${COMFYUI_COMMIT} \
    && rm -rf .git \
    && pip install --no-cache-dir -r requirements.txt \
    && cd custom_nodes \
    # Clone ALL custom nodes
    && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git \
    && git clone --depth 1 https://github.com/theUpsider/ComfyUI-Logic.git \
    && git clone --depth 1 https://github.com/chrisgoringe/cg-use-everywhere.git \
    && git clone --depth 1 https://github.com/chrisgoringe/cg-image-picker.git \
    && git clone --depth 1 https://github.com/M1kep/ComfyLiterals.git \
    && git clone --depth 1 https://github.com/Jordach/comfy-plasma.git \
    && git clone --depth 1 https://github.com/ClownsharkBatwing/RES4LYF.git \
    && git clone --depth 1 https://github.com/JPS-GER/ComfyUI_JPS-Nodes.git \
    && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
    && git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git \
    && git clone --depth 1 https://github.com/Jonseed/ComfyUI-Detail-Daemon.git \
    && git clone --depth 1 https://github.com/bash-j/mikey_nodes.git \
    && git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone --depth 1 --recursive https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git \
    && git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
    && git clone --depth 1 https://github.com/WASasquatch/was-node-suite-comfyui.git \
    && git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git \
    && export SKIP_MODEL_DOWNLOAD=1 \
    && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    && git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git \
    # Install requirements for all custom nodes
    && for dir in /app/comfyui/custom_nodes/*; do \
        if [ -f "$dir/requirements.txt" ]; then \
            pip install --no-cache-dir -r "$dir/requirements.txt" || true; \
        fi; \
        if [ -f "$dir/install.py" ]; then \
            (cd "$dir" && python install.py) || true; \
        fi; \
    done \
    # Install additional useful packages
    && pip install --no-cache-dir jupyter ipython matplotlib pandas opencv-python pillow scikit-image scipy tqdm \
    # CRITICAL: Remove ALL .git directories (saves 1-2GB)
    && find /app/comfyui -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true \
    # Remove large model files from custom nodes
    && find /app/comfyui/custom_nodes -name "*.pth" -size +100M -delete \
    && find /app/comfyui/custom_nodes -name "*.safetensors" -size +100M -delete \
    # Clean up
    && rm -rf /tmp/* /var/tmp/* /root/.cache/*

# Download UltraSharp upscaler + Copy configs in ONE layer
RUN mkdir -p /app/comfyui/models/upscale_models \
    && wget -q -O /app/comfyui/models/upscale_models/4x-UltraSharp.pth \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
    && mkdir -p /app/comfyui/user/default/workflows \
    && mkdir -p /root/.config/code-server \
    && mkdir -p /root/.local/share/code-server/User

# Copy all config files
COPY workflows/z_image_turbo_upscaler.json /app/comfyui/user/default/workflows/
COPY init.sh start.sh /app/
COPY config/code-server-config.yaml /root/.config/code-server/config.yaml
COPY config/vscode-settings.json /root/.local/share/code-server/User/settings.json

RUN chmod +x /app/init.sh /app/start.sh

# Remove build dependencies to save space (keep runtime libs)
RUN apt-get remove -y --purge build-essential cmake ninja-build pkg-config \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose ports
EXPOSE 8080 3000 22

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Start services
CMD ["/app/start.sh"]
