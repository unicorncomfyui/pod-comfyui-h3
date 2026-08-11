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
# `free` reads /proc/meminfo, which inside a container reports the HOST's
# memory, not this pod's share of it - a 92 GB pod on a big machine cheerfully
# announces 386 GB. The cgroup limit is the real figure, so prefer it and fall
# back to /proc only when there is no limit set (cgroup v2 writes "max", v1
# writes an absurdly large sentinel).
read_cgroup_ram_gb() {
    local raw=""
    if [ -r /sys/fs/cgroup/memory.max ]; then                       # cgroup v2
        raw=$(cat /sys/fs/cgroup/memory.max)
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then    # cgroup v1
        raw=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    fi
    case "$raw" in
        ''|max|*[!0-9]*) return 1 ;;
    esac
    # Anything past ~1 PB is "unlimited" dressed up as a number.
    [ "$raw" -gt 1000000000000000 ] && return 1
    echo $((raw / 1024 / 1024 / 1024))
}

if RAM_GB=$(read_cgroup_ram_gb); then
    echo "[INFO] Pod memory limit read from cgroup: ${RAM_GB} GB"
else
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    echo "[INFO] No cgroup memory limit - reporting host RAM: ${RAM_GB} GB"
    echo "       ComfyUI will print this same host figure at startup."
fi
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
# This block decides whether the pod boots. The driver-version check above is a
# proxy; this is the outcome. It covers every way a GPU goes missing at once -
# an old driver, an empty CUDA_VISIBLE_DEVICES that hides all devices, a broken
# runtime injection on a community host, a driver updated under a running
# container. All of them end here, and torch is the only thing that can tell.
CUDA_OK=true

# /dev/nvidia-uvm is created by the container runtime, not by this image, and on
# a busy host it can appear a few seconds after the entrypoint does. nvidia-smi
# never touches it - it talks to the kernel module directly - but cuInit cannot
# work without it, which is how a pod ends up with a perfectly healthy GPU and
# "CUDA unknown error". Wait a little rather than kill a pod for losing a
# startup race. If the node is genuinely never coming, thirty seconds is a cheap
# price for being sure, and the evidence below says so explicitly.
if [ -e /dev/nvidiactl ] && [ ! -e /dev/nvidia-uvm ]; then
    echo "[WARN] /dev/nvidia-uvm not present yet - waiting up to 30 s"
    for _ in 1 2 3 4 5 6; do
        sleep 5
        [ -e /dev/nvidia-uvm ] && break
    done
    [ -e /dev/nvidia-uvm ] && echo "[OK] /dev/nvidia-uvm appeared"
fi

python - <<'PY' || CUDA_OK=false
import sys
print(f"Python: {sys.version.split()[0]}")
try:
    import torch
except Exception as e:
    print(f"[ERROR] torch import failed: {e}")
    sys.exit(1)

print(f"torch:  {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    import os
    print("[ERROR] CUDA is not available to PyTorch")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # An empty value is not "unset": it hides every device from CUDA while
        # leaving nvidia-smi perfectly happy, which is a confusing place to end.
        shown = repr(cvd) if cvd.strip() == "" else cvd
        print(f"CUDA_VISIBLE_DEVICES is set to {shown}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'unset')}")
    sys.exit(1)

print(f"CUDA (torch): {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | sm_{p.major}{p.minor} | {p.total_memory/1024**3:.1f} GB")

# is_available() only enumerates. User-space libraries that no longer match the
# loaded kernel module - a host driver updated under a running container - can
# enumerate a healthy GPU and then fail on the first real kernel, which is a
# generation fifteen minutes from now. Launch one here instead: the .item()
# forces a synchronisation, so an async launch failure surfaces now.
try:
    t = torch.randn(64, 64, device="cuda")
    float((t @ t).sum().item())
    print("CUDA smoke test: ok")
except Exception as e:
    print(f"[ERROR] CUDA enumerates but cannot execute: {e}")
    sys.exit(1)
PY

if [ "$CUDA_OK" != "true" ]; then
    echo ""
    echo "=============================================================="
    echo "[FATAL] PyTorch cannot use a GPU on this host."
    echo ""
    echo "  nvidia-smi may still look perfectly healthy - it does not go"
    echo "  through the CUDA runtime."
    echo ""
    # Telling the reader to run `ls /dev/nvidia*` is useless advice: by the time
    # anyone reads this the container has exited and taken the evidence with it.
    # Collect it here, while the failing host is still under our feet, so a
    # recurring failure can be attributed instead of guessed at.
    echo "  --- evidence, collected on the failing host ---"
    echo ""
    if ls /dev/nvidia* >/dev/null 2>&1; then
        ls -l /dev/nvidia* 2>&1 | sed 's/^/    /'
    else
        echo "    no /dev/nvidia* device nodes at all"
    fi
    echo ""

    # The kernel module's version. The user-space libcuda carries its own in the
    # filename. They are installed by different halves of the container runtime,
    # and a host driver upgraded under a running container is exactly the case
    # where they stop agreeing - libs and module mismatched, GPU still visible.
    KMOD_VER=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)
    LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.[0-9]* /usr/local/nvidia/lib64/libcuda.so.[0-9]* 2>/dev/null | head -1)
    LIB_VER=${LIBCUDA##*libcuda.so.}
    echo "    kernel module : ${KMOD_VER:-unknown}"
    echo "    libcuda       : ${LIB_VER:-not found}  (${LIBCUDA:-no libcuda.so.* on the path})"
    echo ""

    echo "  --- reading ---"
    echo ""
    if [ ! -e /dev/nvidiactl ]; then
        echo "    No device nodes. The container was started without GPU access."
        echo "    This is a template or scheduling problem, not a driver one."
    elif [ ! -e /dev/nvidia-uvm ]; then
        echo "    nvidia-uvm is MISSING and it did not appear after 30 s."
        echo "    CUDA cannot initialise without it; nvidia-smi does not need it,"
        echo "    which is why the GPU looks fine. Nothing in this image can"
        echo "    create that node - it comes from the host's container runtime."
        echo "    REDEPLOY ON ANOTHER HOST. Retrying here will not help."
    elif [ -n "$KMOD_VER" ] && [ -n "$LIB_VER" ] && [ "$KMOD_VER" != "$LIB_VER" ]; then
        echo "    Kernel module ${KMOD_VER} against libcuda ${LIB_VER}: the host"
        echo "    driver was updated under this running container. Restarting the"
        echo "    pod re-injects matching libraries and usually fixes it."
    else
        echo "    Device nodes and driver versions both look consistent, so the"
        echo "    cause is none of the usual three. Keep this block: it is the"
        echo "    part worth reporting."
    fi
    echo ""
    echo "  An EMPTY CUDA_VISIBLE_DEVICES also hides every device while leaving"
    echo "  nvidia-smi happy. Unset it - do not set it to 0."
    echo ""
    echo "  Refusing to boot: the pod would pull 42 GB of weights and start"
    echo "  ComfyUI, then fail on the first generation."
    echo ""
    echo "  Set ALLOW_DRIVER_MISMATCH=true to boot anyway (CPU only)."
    echo "=============================================================="
    echo ""
    if [ "${ALLOW_DRIVER_MISMATCH:-false}" != "true" ]; then
        exit 1
    fi
    echo "[WARN] ALLOW_DRIVER_MISMATCH=true - continuing without a usable GPU"
fi

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
