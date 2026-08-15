#!/bin/bash
# Serve a local vision-language model to the H3 Prompt Writer extension.
#
# WHY THIS IS A SCRIPT AND NOT PART OF THE IMAGE
#
# The runtime weighs 1.4 GB and the weights another 18, for a feature that is
# an experiment. Baking either into the image would make every pod pay for
# something most pods will never start. So both land on the volume, on demand,
# and survive across pods the same way the H3 weights do.
#
# WHY OLLAMA AND NOT llama-server
#
# llama.cpp publishes no CUDA build for Linux. Its release page carries x64
# (CPU), Vulkan, SYCL, OpenVINO and arm64 - the CUDA binaries are Windows only.
# That leaves three bad options on this pod: run a 27B model on the CPU, add a
# Vulkan loader and an ICD to a -runtime CUDA image and hope the container
# runtime exposes the graphics capability, or compile llama.cpp against CUDA 13
# in an image that deliberately carries no toolchain.
#
# Ollama ships its own CUDA runtime libraries inside the tarball. It therefore
# does not care that this image is CUDA 13 while most prebuilt inference
# binaries are still linked against CUDA 12 - the same mismatch that made
# faster-whisper fall back to the CPU here. Nothing is compiled, nothing is
# resolved against the system, and the Prompt Writer speaks to it natively.
#
# Usage:
#   bash /app/scripts/promptgen.sh start      install if needed, then serve
#   bash /app/scripts/promptgen.sh status     is it up, and what is resident
#   bash /app/scripts/promptgen.sh unload     free the card, leave the server up
#   bash /app/scripts/promptgen.sh stop       stop the server entirely
#   bash /app/scripts/promptgen.sh pull TAG   fetch another model
#
# Then: ComfyUI > Extensions > H3 Prompt Writer > Settings > Ollama.

set -euo pipefail

DATA_DIR="${COMFYUI_DATA_DIR:-/workspace/comfyui-data}"
PREFIX="${PROMPTGEN_PREFIX:-$DATA_DIR/promptgen}"

# Pinned, for the same reason every custom node in the Dockerfile is pinned: a
# runtime that changes under you turns "it worked last week" into a bisect.
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.32.13}"

# Default tag. qwen3.8:27b is the q4_K_M build at about 18 GB - a dense 27B
# vision-language model, so it reads the reference images rather than only the
# brief. 27b-nvfp4 is 16.9 GB and is the format this card runs natively; it is
# the first thing to try if 18 GB is tight, but it is newer and less walked.
PROMPTGEN_MODEL="${PROMPTGEN_MODEL:-qwen3.8:27b}"

# Loopback, and not negotiable: the Prompt Writer accepts local providers only
# and rejects any other host. Exposing this port would also publish an
# unauthenticated inference endpoint on a pod that already exposes two ports.
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$PREFIX/models}"

# How long a model stays resident after its last request. Ollama's own default
# is 5m and it is kept: shorter means reloading 18 GB from the network volume
# between two prompts, which is slow enough to change how you work. The cost is
# that the card stays occupied - see the VRAM note in vram_note().
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"

BIN="$PREFIX/bin/ollama"
LOG="/var/log/promptgen.log"
PIDFILE="$PREFIX/ollama.pid"

log() { echo "$*"; }

vram_note() {
    log ""
    log "  The card holds one of these at a time, not both:"
    log "    H3          20.97 GB diffusion + 15.69 GB text encoder"
    log "    this model  ~18 GB + context"
    log "  Write your prompts, then 'promptgen.sh unload' before rendering."
    log "  Skipping that does not fail cleanly - it looks like an H3 problem."
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
ensure_zstd() {
    command -v zstd >/dev/null 2>&1 && return 0
    # Not in the Dockerfile on purpose. zstd is 1 MB, but it belongs in the
    # system-packages layer, and that layer sits below torch - adding it there
    # would re-download 4 GB of wheels on the next build to serve an opt-in
    # experiment. Installing it here costs a few seconds on a pod that is
    # already about to pull 20 GB over the network.
    log "[..] Installing zstd (needed to unpack the runtime)"
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq --no-install-recommends zstd >/dev/null 2>&1 || {
        log "[ERROR] Could not install zstd. Install it by hand, then re-run:"
        log "          apt-get update && apt-get install -y zstd"
        return 1
    }
}

ensure_installed() {
    if [ -x "$BIN" ]; then
        return 0
    fi

    # 1.4 GB of runtime plus the weights. Check before spending the download,
    # not after - a volume that fills up mid-pull leaves a partial blob store
    # that the next run reports as a corrupt manifest.
    local avail_gb
    avail_gb=$(df -BG --output=avail "$DATA_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 25 ]; then
        log "[ERROR] ${avail_gb} GB free on $DATA_DIR; the runtime and one model"
        log "        need about 25. Free space or point PROMPTGEN_PREFIX"
        log "        somewhere larger."
        return 1
    fi

    ensure_zstd || return 1

    local url="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
    log "[..] Fetching the runtime ${OLLAMA_VERSION} (about 1.4 GB)"
    mkdir -p "$PREFIX"
    # --retry on a throttled response, the same reason the Dockerfile's two
    # Hugging Face fetches carry it: a 429 in the middle of a build or a boot
    # is not a reason to fail the whole thing.
    if ! curl -fSL --retry 5 --retry-delay 5 --retry-all-errors \
            -o "$PREFIX/ollama.tar.zst" "$url"; then
        log "[ERROR] Download failed. Check the pod has outbound network."
        return 1
    fi

    log "[..] Unpacking into $PREFIX"
    # The tarball carries bin/ and lib/ side by side and the binary finds its
    # bundled CUDA libraries relative to itself - keep the layout intact.
    tar --use-compress-program=unzstd -xf "$PREFIX/ollama.tar.zst" -C "$PREFIX"
    rm -f "$PREFIX/ollama.tar.zst"

    if [ ! -x "$BIN" ]; then
        log "[ERROR] $BIN missing after unpacking - the archive layout changed."
        return 1
    fi
    log "[OK] Runtime installed at $PREFIX"
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
server_up() {
    curl -fsS --max-time 3 "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1
}

start_server() {
    if server_up; then
        log "[OK] Already serving on ${OLLAMA_HOST}"
        return 0
    fi
    mkdir -p "$OLLAMA_MODELS"
    log "[..] Starting the server (log: $LOG)"
    nohup "$BIN" serve >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"

    local i
    for i in $(seq 1 30); do
        if server_up; then
            log "[OK] Serving on ${OLLAMA_HOST} (PID $(cat "$PIDFILE"))"
            return 0
        fi
        sleep 1
    done
    log "[ERROR] The server did not answer within 30 s. Last lines of $LOG:"
    tail -20 "$LOG" 2>/dev/null || true
    return 1
}

have_model() {
    "$BIN" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$1"
}

pull_model() {
    local tag="$1"
    if have_model "$tag"; then
        log "[OK] $tag already on the volume"
        return 0
    fi
    log "[..] Pulling $tag - first time only, and it is tens of gigabytes"
    "$BIN" pull "$tag"
}

# ---------------------------------------------------------------------------
case "${1:-start}" in
    start)
        ensure_installed
        start_server
        pull_model "$PROMPTGEN_MODEL"
        log ""
        log "[OK] Ready. In ComfyUI: Extensions > H3 Prompt Writer > Settings"
        log "     > Ollama > Check now, then pick ${PROMPTGEN_MODEL}."
        vram_note
        ;;

    stop)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            kill "$(cat "$PIDFILE")"
            rm -f "$PIDFILE"
            log "[OK] Server stopped; the card is free."
        else
            # Started by hand, or by a previous pod whose PID means nothing now.
            pkill -f "$BIN serve" 2>/dev/null && log "[OK] Server stopped." \
                || log "[OK] Nothing was running."
            rm -f "$PIDFILE"
        fi
        ;;

    unload)
        # keep_alive 0 evicts the weights and leaves the server listening, so
        # the next prompt costs a model load rather than a process start. This
        # is the one to run before queueing an H3 render.
        if ! server_up; then
            log "[OK] Nothing is serving, so nothing is resident."
            exit 0
        fi
        curl -fsS "http://${OLLAMA_HOST}/api/generate" \
            -d "{\"model\":\"${PROMPTGEN_MODEL}\",\"keep_alive\":0}" >/dev/null \
            && log "[OK] ${PROMPTGEN_MODEL} evicted; the card is free." \
            || log "[WARN] Eviction request refused - check 'promptgen.sh status'."
        ;;

    status)
        if server_up; then
            log "server     up on ${OLLAMA_HOST}"
            log "models     $OLLAMA_MODELS"
            log ""
            log "--- installed ---"
            "$BIN" list 2>/dev/null || true
            log ""
            log "--- resident on the GPU ---"
            "$BIN" ps 2>/dev/null || true
        else
            log "server     down"
            [ -x "$BIN" ] && log "runtime    installed at $PREFIX" \
                          || log "runtime    not installed - run 'promptgen.sh start'"
        fi
        log ""
        # || true, and not decoration: pipefail is on, so a pod without
        # nvidia-smi would make the status command exit non-zero after having
        # printed everything correctly.
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
            | while read -r line; do log "GPU        $line"; done || true
        ;;

    pull)
        ensure_installed
        start_server
        pull_model "${2:?usage: promptgen.sh pull TAG}"
        ;;

    *)
        log "usage: promptgen.sh {start|stop|unload|status|pull TAG}"
        exit 2
        ;;
esac
