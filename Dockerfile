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
# LAYER ORDER IS LOAD-BEARING. A rebuilt layer rebuilds every layer after it,
# so these run from least to most likely to change, and torch sits early
# because it is by far the most expensive thing to fetch. Raising the Pillow
# floor must not re-download 4 GB of torch; bumping code-server must not
# either. That is the whole reason this file is split the way it is.
#
#   1  system packages, Python, C compiler   ~never
#   2  torch                                 on a torch bump      expensive
#   3  code-server                           on a version bump
#   4  ComfyUI core                          on a ComfyUI bump
#   5  custom nodes                          when the set moves
#   6  advisory floors                       on every CVE         cheap
#   7  bundled assets
#   8  scripts and configs                   constantly           cheapest
#
# Nothing is compiled AT BUILD TIME: comfy-kitchen (kernels) and comfy-aimdo
# (dynamic VRAM offloader) ship as prebuilt abi3 wheels, so the image is based
# on -runtime rather than -devel and needs no SageAttention build step. That is
# why layer 1 installs gcc alone instead of build-essential - see the note
# there before adding to it.

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
    CC=gcc \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="12.0" \
    COMFYUI_PORT=3000 \
    VSCODE_PORT=8080 \
    HF_HOME=/workspace/.cache/huggingface \
    DOWNLOAD_MODELS=true \
    MODEL_SETS="minimax-h3-fl2va,minimax-h3-ref2va,minimax-h3-turbo-lora"

# Record the build parameters so a running pod can report exactly what it is.
ENV BUILD_TORCH_INDEX=${TORCH_INDEX} \
    BUILD_TORCH_VERSION=${TORCH_VERSION} \
    BUILD_COMFYUI_VERSION=${COMFYUI_VERSION} \
    BUILD_PYTHON_VERSION=${PYTHON_VERSION}

# ---------------------------------------------------------------------------
# 1 - System packages, Python, and the C compiler.
#
# gcc and libc6-dev are permanent, and they are not a build-time convenience:
# Triton compiles its kernels JIT, at RUNTIME. torch 2.13 routes torch._native
# ops - bmm_outer_product, reached by the H3 text encoder's RoPE - through
# Triton, which shells out to cc to build driver.c on first use. Without them
# the pod loads the model fine and then dies mid-generation with "Failed to
# find C compiler".
#
# build-essential is deliberately NOT installed. It exists to compile things,
# and this image compiles nothing at build time - every wheel it installs is
# prebuilt. If a future dependency ships as an sdist and needs g++ or make,
# the build will fail loudly and name it; add exactly what it asks for, here,
# rather than restoring the whole meta-package on suspicion.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
        git git-lfs wget curl unzip nano vim jq \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg \
        openssh-server rsync \
        google-perftools libtcmalloc-minimal4 \
        gcc libc6-dev \
    # The openssh-server postinst generates host keys at install time. A host
    # key identifies a deployment, not an image, so it has no business in a
    # published layer: start.sh creates the set on the volume at first boot.
    && rm -f /etc/ssh/ssh_host_* \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ---------------------------------------------------------------------------
# 2 - PyTorch. Its own layer, and early: ~4 GB that almost never changes.
# Installed before ComfyUI so its bare `torch` requirement resolves as
# already-satisfied instead of pulling the default PyPI build.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# ---------------------------------------------------------------------------
# 3 - code-server. After torch on purpose: both change rarely, but when this
# one does, torch must stay cached.
# ---------------------------------------------------------------------------
RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --version=${CODE_SERVER_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/*

# ---------------------------------------------------------------------------
# 4 - ComfyUI core. Pulls comfy-kitchen (NVFP4 / int8-convrot kernels) and
# comfy-aimdo (dynamic VRAM offloader), both required by MiniMax H3.
# The project moved from comfyanonymous/ComfyUI to Comfy-Org/ComfyUI.
# ---------------------------------------------------------------------------
WORKDIR /app
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI.git comfyui \
    && git -C comfyui checkout -q ${COMFYUI_VERSION} \
    && rm -rf comfyui/.git \
    && pip install --no-cache-dir -r comfyui/requirements.txt \
    # Fast, resumable, chunk-deduplicated model downloads (H3 is 63 GB).
    && pip install --no-cache-dir "huggingface_hub[hf_xet]>=1.26.0" \
    && rm -rf /root/.cache/*

# ---------------------------------------------------------------------------
# 5 - Custom nodes, every one pinned to a commit.
#
# `git clone --depth 1` with no ref takes whatever HEAD happens to be at build
# time, which means two builds of the same source produce different images and
# a node can break against a pinned ComfyUI without anything changing here.
# That is not hypothetical: VideoHelperSuite has not moved since 2026-05-14 and
# calls helpDOM.addHelp(), removed from the 0.30.0 frontend, so any workflow
# holding a VHS_ node fails to load. Pinning does not fix that one - no
# published commit of it works - but it stops the other five drifting into the
# same state unnoticed.
#
# To bump one: change its SHA here, rebuild, and re-run the bench. Never by
# rebuilding and hoping.
#
# The set is video-oriented only. The image-era nodes (WAS suite, Impact Pack,
# Comfyroll, RES4LYF, ...) are intentionally absent - they weigh a lot and
# conflict more often on each ComfyUI core bump. ComfyUI-Easy-Use too: it needs
# clip_interrogator 0.6.0, last released March 2023, which is not a safe bet
# against transformers 5.x.
# ---------------------------------------------------------------------------
WORKDIR /app/comfyui/custom_nodes
RUN git clone --filter=blob:none https://github.com/Comfy-Org/ComfyUI-Manager.git \
    && git -C ComfyUI-Manager checkout -q d47c9346190397e1c316bc5a82155faaf9f5d700 \
    && git clone --filter=blob:none https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git -C ComfyUI-VideoHelperSuite checkout -q 4ee72c065db22c9d96c2427954dc69e7b908444b \
    && git clone --filter=blob:none https://github.com/kijai/ComfyUI-KJNodes.git \
    && git -C ComfyUI-KJNodes checkout -q 35e5956193769d18a13136cdedb73a36a05c73e6 \
    && git clone --filter=blob:none https://github.com/rgthree/rgthree-comfy.git \
    && git -C rgthree-comfy checkout -q 6b76ee6f2c5a007710b5a16f97c94330d6ecc871 \
    && git clone --filter=blob:none https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    && git -C ComfyUI-Frame-Interpolation checkout -q 26545cc2dd95bc3d27f056016300673bdeee78f5 \
    && git clone --filter=blob:none https://github.com/chrisgoringe/cg-use-everywhere.git \
    && git -C cg-use-everywhere checkout -q 50ae9f8c5d8b9538589663c90a15d4067a02969c \
    # Sampler for the step-distillation LoRA: 4 steps instead of 12.
    && git clone --filter=blob:none https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git \
    && git -C ComfyUI-MiniMax-H3-Turbo checkout -q 96cc1ddc001617da132dd73f31cd43666bf1d8d4 \
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
    # Strip node git metadata and any bundled weights.
    && find /app/comfyui -name ".git" -type d -prune -exec rm -rf {} + 2>/dev/null || true \
    && find /app/comfyui/custom_nodes \( -name "*.pth" -o -name "*.safetensors" \) -size +50M -delete \
    && rm -rf /root/.cache/*

# ---------------------------------------------------------------------------
# 6 - Advisory floors. Its own layer, and late, for two reasons: nothing above
# can pull an older pin back in, and answering a CVE costs one small layer
# instead of the whole install. Pillow matters most - ComfyUI hands it every
# image that is loaded, so it sits on the widest input surface in the stack.
# Held below 13 to stay an API-compatible bump: a floor, not an upgrade.
# Raise these when the scan says to; do not drop them to "simplify" the file.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade "pillow>=12.3.0,<13" setuptools \
    && rm -rf /root/.cache/*

# ---------------------------------------------------------------------------
# 6b - SageAttention, optional and prebuilt.
#
# INT8-quantised attention. Measured on this exact stack, against the standard
# implementation, same seed, same graph:
#
#   0.88 MP / 12 steps   148.20 s -> 104.21 s   (-29.7%)
#   0.40 MP /  6 steps    24.39 s ->  19.42 s   (-20.4%)
#
# The gain grows with size because attention is quadratic in latent tokens
# while everything else is linear: it is 59% of the clock at 0.88 MP and 41%
# at 0.40 MP, and Sage halves it. Enable at runtime with SAGE_ATTENTION=true;
# nothing here turns it on.
#
# The URL points at a wheel WE built, because none exists for Linux: upstream
# publishes no wheels at all, and the well-known third-party builds are
# win_amd64 only. See .github/workflows/sageattention-wheel.yml - it compiles
# against this same torch and CUDA, which is not optional. A wheel built for
# another torch links against another libtorch ABI and fails at import with an
# undefined symbol naming a mangled C++ function rather than the real cause.
#
# Empty by default, so a build that has not been given a wheel still produces
# a working image - ComfyUI simply falls back to PyTorch attention and says so
# in its log. That keeps the file's invariant intact: nothing is compiled here.
# ---------------------------------------------------------------------------
# No `#` inside the RUN below. Docker joins continuation lines into a single
# command, and whether a commented line is stripped by the parser or handed to
# the shell decides between a no-op and commenting out everything after it.
# Explanation belongs here, where that question does not arise.
#
# `set -eu` plus one statement per line, each announcing itself: a chain of &&
# reports only "exit code 1" for whichever link broke, and on a twelve-minute
# build that costs a round trip per guess. The zip check exists because a 404
# or an HTML error page is still a file, and pip describes such a thing badly.
#
# The download KEEPS the wheel's own filename. pip parses that name for the
# five segments name-version-python-abi-platform, and refuses anything else -
# saving it as sageattention.whl earned exactly that: "Invalid wheel filename
# (wrong number of parts)". A wheel's name is metadata, not decoration.
ARG SAGEATTENTION_WHEEL=""
RUN set -eu; \
    if [ -z "${SAGEATTENTION_WHEEL}" ]; then \
        echo "[INFO] No SAGEATTENTION_WHEEL given; attention stays on PyTorch."; \
        exit 0; \
    fi; \
    mkdir -p /tmp/whl; \
    WHL="/tmp/whl/$(basename "${SAGEATTENTION_WHEEL}")"; \
    echo "==> downloading ${SAGEATTENTION_WHEEL}"; \
    curl -fSL --retry 3 -o "${WHL}" "${SAGEATTENTION_WHEEL}"; \
    echo "==> downloaded $(stat -c%s "${WHL}") bytes as $(basename "${WHL}")"; \
    python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "${WHL}" \
        || { echo "[ERROR] not a wheel:"; head -c 400 "${WHL}"; exit 1; }; \
    echo "==> installing"; \
    pip install --no-cache-dir "${WHL}"; \
    rm -rf /tmp/whl; \
    echo "==> installed:"; \
    pip show sageattention | head -2; \
    rm -rf /root/.cache/*

# ---------------------------------------------------------------------------
# 7 - Bundled assets. Model weights are NOT baked in: 64 GB of H3 lands on the
# network volume at first boot.
# ---------------------------------------------------------------------------
RUN mkdir -p /app/comfyui/models/upscale_models \
             /app/comfyui/user/default/workflows \
             /root/.config/code-server \
             /root/.local/share/code-server/User \
    && wget -q -O /app/comfyui/models/upscale_models/4x-UltraSharp.pth \
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth" \
    # Ready-made turbo workflow, staged onto the volume by start.sh - ComfyUI
    # reads workflows from --user-directory, which points at the volume, so a
    # copy left in the image would never be listed in the UI.
    && wget -q -O /app/comfyui/user/default/workflows/minimax_h3_t2v_turbo.json \
        "https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_t2v_turbo.json"

# ---------------------------------------------------------------------------
# 8 - Scripts and configs. Last because they change on almost every commit,
# and this way that costs a few kilobytes rather than a rebuild.
# ---------------------------------------------------------------------------
WORKDIR /app
COPY models/manifest.json /app/models/manifest.json
COPY scripts/download_models.py /app/scripts/download_models.py
COPY scripts/bench.py /app/scripts/bench.py
COPY scripts/worker.py /app/scripts/worker.py
# API-format templates the worker injects into, plus a sample job. Kept out of
# ComfyUI's own workflow directory on purpose: these are the /prompt payload
# shape, not the editor's, and the UI would list them as broken graphs.
COPY workflows/ /app/workflows/
COPY examples/ /app/examples/
COPY init.sh start.sh fetch_models.sh /app/
COPY config/code-server-config.yaml /root/.config/code-server/config.yaml
COPY config/vscode-settings.json /root/.local/share/code-server/User/settings.json

# Normalise line endings in case of a checkout from Windows.
RUN sed -i 's/\r$//' /app/init.sh /app/start.sh /app/fetch_models.sh \
    && chmod +x /app/init.sh /app/start.sh /app/fetch_models.sh

# Declare the driver we actually need, not the one the base tag implies.
#
# nvidia/cuda:13.3.x stamps NVIDIA_REQUIRE_CUDA=cuda>=13.3, and the container
# runtime's prestart hook refuses to start on any host below that - the pod
# never boots, with "unsatisfied condition: cuda>=13.3". But nothing here needs
# 13.3: torch is built for cu130, whose floor is driver 580, which is the same
# number init.sh checks. The 13.3 runtime libraries ship INSIDE the image; only
# the driver comes from the host, and CUDA minor version compatibility is
# exactly the guarantee that a 13.x driver runs a 13.y runtime.
#
# Overriding it here rather than rebasing on 13.0 widens the pool of machines
# RunPod can schedule us on by a lot. Deliberately placed in the volatile zone:
# only the image's FINAL environment is read, so setting it last costs one
# metadata layer instead of invalidating torch.
#
# Note for the pod template: RunPod's "Allowed CUDA versions" is a whitelist of
# host versions, not a minimum. Ticking 13.0 permits a 13.0 host.
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"

EXPOSE 8080 3000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${COMFYUI_PORT}/ || exit 1

CMD ["/app/start.sh"]
