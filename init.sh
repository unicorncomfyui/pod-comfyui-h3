#!/bin/bash
# Initialization for the MiniMax H3 ComfyUI pod.
# Runs diagnostics, validates the H3 runtime stack, then fetches model weights.
# Expects COMFYUI_DIR to be exported by start.sh.

set -e

echo ""
echo "=========================================="
echo "SYSTEM DIAGNOSTICS"
echo "=========================================="

echo ""
echo "--- Image build parameters ---"
echo "PyTorch:  ${BUILD_TORCH_VERSION:-?} (${BUILD_TORCH_INDEX:-?})"
echo "ComfyUI:  ${BUILD_COMFYUI_VERSION:-?}"
echo "Python:   ${BUILD_PYTHON_VERSION:-?}"

echo ""
echo "--- GPU ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
        --format=csv,noheader || echo "[ERROR] nvidia-smi query failed"

    # MiniMax H3 ships its text encoder in NVFP4, which needs Blackwell
    # tensor cores. Anything below sm_120 falls back to a much slower path.
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    if [ "$COMPUTE_CAP" != "12.0" ]; then
        echo "[WARN] Compute capability is $COMPUTE_CAP, not 12.0 (Blackwell)."
        echo "       The NVFP4 text encoder will run on an emulated path."
        echo "       This image targets the RTX 5090."
    fi

    # cu130 builds need a 580+ host driver; RunPod still has 575 machines.
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "${BUILD_TORCH_INDEX}" = "cu130" ] && [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
        echo "[ERROR] Host driver is ${DRIVER_MAJOR}.x but this is a CUDA 13 image (needs 580+)."
        echo "        Redeploy on a CUDA 13 machine, or use the :cu129-* image tag."
    fi
else
    echo "[ERROR] nvidia-smi not found"
fi

echo ""
echo "--- CPU / RAM ---"
lscpu | grep -E "^Model name|^CPU\(s\)" || true
free -h | grep -E "Mem:|Swap:" || true

# Host RAM, not VRAM, is the bottleneck on a 32 GB card: the offloader streams
# weights through it. Only one diffusion model is resident per workflow, so the
# figure that matters is ~42.5 GB (one diffusion model + text encoder + both
# VAEs), not the 63.4 GB total that a full fl2va+ref2va install occupies on disk.
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "${RAM_GB:-0}" -lt 48 ]; then
    echo "[WARN] Only ${RAM_GB} GB of host RAM detected."
    echo "       A single H3 workflow needs ~42.5 GB resident. Below 48 GB expect"
    echo "       heavy swapping or OOM. Consider FAST_DISK=true to trade speed"
    echo "       for survival, or move to a larger pod."
elif [ "${RAM_GB:-0}" -lt 80 ]; then
    echo "[INFO] ${RAM_GB} GB of host RAM - workable but with little margin."
    echo "       If you hit host-memory exhaustion, try COMFYUI_EXTRA_ARGS=--disable-pinned-memory"
else
    echo "[OK]   ${RAM_GB} GB of host RAM - comfortable for H3."
fi

if [ "${FAST_DISK:-false}" = "true" ] && [ "${RAM_GB:-0}" -ge 80 ]; then
    echo "[WARN] FAST_DISK=true with ${RAM_GB} GB of RAM is likely a pessimisation:"
    echo "       the weights fit in RAM, and on RunPod they sit on a network"
    echo "       volume rather than local NVMe. Consider FAST_DISK=false."
fi

echo ""
echo "--- Disk ---"
df -h / /workspace 2>/dev/null || df -h /

echo ""
echo "--- PyTorch / CUDA ---"
python - <<'PY' || echo "[ERROR] Python diagnostics failed"
import sys
print(f"Python: {sys.version.split()[0]}")
try:
    import torch
    print(f"torch:  {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA (torch): {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name} | sm_{p.major}{p.minor} | {p.total_memory/1024**3:.1f} GB")
    else:
        import os
        print("[ERROR] CUDA is not available to PyTorch")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'unset')}")
except Exception as e:
    print(f"[ERROR] torch check failed: {e}")
PY

echo ""
echo "--- MiniMax H3 runtime stack ---"
# These two packages are what make H3 fit on a 32 GB card. Without them the
# model loads but either fails on int8-convrot weights or OOMs immediately.
python - <<'PY' || true
for mod, label in (
    ("comfy_kitchen", "comfy-kitchen (NVFP4 / int8-convrot kernels)"),
    ("comfy_aimdo", "comfy-aimdo (dynamic VRAM offloader)"),
):
    try:
        m = __import__(mod)
        print(f"[OK]   {label}: {getattr(m, '__version__', 'installed')}")
    except ImportError as e:
        print(f"[ERROR] {label} missing - H3 will not run correctly ({e})")
PY

echo ""
echo "=========================================="
echo "END DIAGNOSTICS"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
if [ "${DOWNLOAD_MODELS:-true}" != "true" ]; then
    echo "[SKIP] Model download disabled (DOWNLOAD_MODELS=false)"
elif [ -z "${COMFYUI_DIR}" ]; then
    echo "[SKIP] COMFYUI_DIR not set - skipping model download"
elif [ "${NETWORK_VOLUME}" != "true" ]; then
    # 63 GB of weights will not fit in container storage.
    echo "[SKIP] No network volume detected - skipping model download."
    echo "       MiniMax H3 needs up to 63 GB; attach a network volume."
else
    echo "Fetching model weights (MODEL_SETS=${MODEL_SETS:-default})..."
    python /app/scripts/download_models.py || echo "[WARN] Model download reported errors"
fi

# ---------------------------------------------------------------------------
# Optional page-cache warm-up.
#
# On a RunPod network volume, reads are network-bound and RunPod rates their
# throughput as "variable". Left alone, those 42 GB stream in during the first
# generation, as unpredictable stalls mid-sampling. Reading them once up front
# moves that cost to boot, where it is visible in the log, and subsequent
# faults are served from RAM.
#
# Name ONE set: prewarming more than fits in RAM just evicts itself.
# ---------------------------------------------------------------------------
if [ -n "${PREWARM_SET}" ] && [ -n "${COMFYUI_DIR}" ]; then
    echo ""
    echo "Warming page cache for '${PREWARM_SET}'..."
    PREWARM_LIST=$(mktemp)
    python /app/scripts/download_models.py --list "${PREWARM_SET}" > "$PREWARM_LIST" 2>/dev/null || true

    if [ ! -s "$PREWARM_LIST" ]; then
        echo "[WARN] Nothing to prewarm - set unknown, or its files are not downloaded yet"
    else
        # Read line-by-line rather than word-splitting, so a path containing a
        # space cannot silently turn into two bogus paths.
        PREWARM_BYTES=0
        while IFS= read -r f; do
            SZ=$(stat -c %s "$f" 2>/dev/null || echo 0)
            PREWARM_BYTES=$(( PREWARM_BYTES + SZ ))
        done < "$PREWARM_LIST"
        PREWARM_GB=$(( PREWARM_BYTES / 1000000000 ))

        AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')   # "available", not "free"
        if [ "${AVAIL_GB:-0}" -lt "$PREWARM_GB" ]; then
            echo "[SKIP] Needs ${PREWARM_GB} GB but only ${AVAIL_GB} GB available - would thrash"
        else
            START=$(date +%s)
            while IFS= read -r f; do
                cat "$f" > /dev/null 2>&1 || true
            done < "$PREWARM_LIST"
            echo "[OK]   ${PREWARM_GB} GB cached in $(( $(date +%s) - START ))s"
        fi
    fi
    rm -f "$PREWARM_LIST"
fi

echo ""
echo "[OK] Initialization completed"
exit 0
