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

    # A host driver too old for the image's CUDA runtime is fatal, not advisory:
    # torch reports "CUDA available: False" and every later failure is a
    # downstream symptom. Abort here rather than let the pod look healthy, boot
    # ComfyUI, and pull 42 GB of weights it can never execute.
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    case "${BUILD_TORCH_INDEX}" in
        cu130|cu131|cu132) DRIVER_MIN=580 ;;
        cu129)             DRIVER_MIN=575 ;;
        *)                 DRIVER_MIN=0   ;;
    esac

    if [ "${DRIVER_MAJOR:-0}" -lt "$DRIVER_MIN" ]; then
        echo ""
        echo "=============================================================="
        echo "[FATAL] Host driver ${DRIVER_MAJOR}.x is too old for this image."
        echo ""
        echo "  image built for : ${BUILD_TORCH_INDEX}  (needs driver ${DRIVER_MIN}+)"
        echo "  host driver     : ${DRIVER_MAJOR}.x"
        echo ""
        echo "  RunPod scheduled this pod on an incompatible machine. In the"
        echo "  template, under GPU Compatibility > Allowed CUDA versions,"
        echo "  untick every version below the one this image needs."
        echo ""
        echo "  Set ALLOW_DRIVER_MISMATCH=true to boot anyway (CPU only)."
        echo "=============================================================="
        echo ""
        if [ "${ALLOW_DRIVER_MISMATCH:-false}" != "true" ]; then
            exit 1
        fi
        echo "[WARN] ALLOW_DRIVER_MISMATCH=true - continuing without a usable GPU"
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

# Triton compiles its kernels JIT, at runtime. torch._native routes ops through
# it (the H3 text encoder's RoPE hits bmm_outer_product), so a missing compiler
# does not surface until mid-generation, as a RuntimeError buried in a stack
# trace. Check it here instead.
if command -v "${CC:-gcc}" > /dev/null 2>&1; then
    echo "[OK]   C compiler for Triton JIT: $(command -v "${CC:-gcc}")"
else
    echo "[ERROR] No C compiler (${CC:-gcc}) - Triton cannot JIT its kernels."
    echo "        Generation will fail with 'Failed to find C compiler'."
fi

echo ""
echo "=========================================="
echo "END DIAGNOSTICS"
echo "=========================================="
echo ""

echo "[OK] Diagnostics completed"
exit 0
