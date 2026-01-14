#!/bin/bash
# Initialization script for RunPod ComfyUI Pod with VSCode
# Handles SageAttention compilation with network volume caching

set -e

echo "Starting initialization..."

# System diagnostics
echo ""
echo "=========================================="
echo "SYSTEM DIAGNOSTICS"
echo "=========================================="

# GPU Information
echo ""
echo "--- GPU Information ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || echo "[ERROR] nvidia-smi query failed"
    echo ""
    echo "CUDA Driver Version:"
    nvidia-smi | grep "CUDA Version" || echo "[ERROR] Could not detect CUDA driver version"
else
    echo "[ERROR] nvidia-smi not found"
fi

# CPU Information
echo ""
echo "--- CPU Information ---"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core" || echo "[ERROR] Could not get CPU info"

# Memory Information
echo ""
echo "--- Memory Information ---"
free -h | grep -E "Mem:|Swap:" || echo "[ERROR] Could not get memory info"

# Disk Space
echo ""
echo "--- Disk Space ---"
df -h / /workspace 2>/dev/null || df -h /

# Environment Variables
echo ""
echo "--- NVIDIA Environment Variables ---"
env | grep -E "CUDA|NVIDIA" | sort || echo "No NVIDIA environment variables found"

# PyTorch & CUDA Information
echo ""
echo "--- PyTorch & CUDA Information ---"
python -c "
import sys
print(f'Python version: {sys.version.split()[0]}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version (PyTorch): {torch.version.cuda}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            props = torch.cuda.get_device_properties(i)
            print(f'  Compute capability: {props.major}.{props.minor}')
            print(f'  Total memory: {props.total_memory / 1024**3:.2f} GB')
    else:
        print('[ERROR] CUDA is not available to PyTorch')
        import os
        print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"Not set\")}')
        print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"Not set\")}')
except Exception as e:
    print(f'[ERROR] PyTorch check failed: {e}')
" || echo "[ERROR] Python diagnostics failed"

# CUDA Toolkit Version
echo ""
echo "--- CUDA Toolkit Version ---"
if [ -f "/usr/local/cuda/version.json" ]; then
    cat /usr/local/cuda/version.json | grep -E "cuda_version|name" || echo "/usr/local/cuda/version.json found but could not parse"
elif command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" || echo "nvcc found but could not get version"
else
    echo "CUDA toolkit not found in standard locations"
fi

echo ""
echo "=========================================="
echo "END DIAGNOSTICS"
echo "=========================================="
echo ""

# Verify SageAttention installation (pre-compiled in Docker image)
echo "--- SageAttention Verification ---"
if python -c "import sageattention; print(f'✅ SageAttention installed: {sageattention.__version__ if hasattr(sageattention, \"__version__\") else \"v2.2.0 (eb615cf)\"}')"; then
    echo "[OK] SageAttention ready (pre-compiled in image)"
else
    echo "[WARN] SageAttention not found - image may need rebuild"
fi
echo ""

# Z-Image-Turbo model checking and downloading
if [ "${CHECK_MODELS:-true}" = "true" ]; then
    echo "Checking Z-Image-Turbo models..."

    # ONLY download models if network volume is available
    # Container storage doesn't have enough space (~10GB needed)
    if [ -d "/workspace/ComfyUI" ]; then
        COMFYUI_DIR="/workspace/ComfyUI"
        echo "Network volume detected - using $COMFYUI_DIR for models"
    else
        echo "[SKIP] No network volume detected - skipping model downloads (container has insufficient storage)"
        echo "       Models will need to be provided manually or use network volume"
        COMFYUI_DIR=""
    fi

    if [ -n "$COMFYUI_DIR" ]; then
        # Define model paths
        DIFFUSION_MODEL="$COMFYUI_DIR/models/diffusion_models/z_image_turbo_bf16.safetensors"
        TEXT_ENCODER="$COMFYUI_DIR/models/clip/qwen_3_4b.safetensors"
        VAE_MODEL="$COMFYUI_DIR/models/vae/ae.safetensors"

        MODELS_MISSING=false

        # Check diffusion model
        if [ ! -f "$DIFFUSION_MODEL" ]; then
            echo "Downloading Z-Image-Turbo diffusion model (3GB)..."
            mkdir -p "$COMFYUI_DIR/models/diffusion_models"
            wget --progress=bar:force:noscroll -O "$DIFFUSION_MODEL" \
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
                || { echo "[ERROR] Failed to download diffusion model"; MODELS_MISSING=true; }
        else
            echo "[OK] Diffusion model found: z_image_turbo_bf16.safetensors"
        fi

        # Check text encoder
        if [ ! -f "$TEXT_ENCODER" ]; then
            echo "Downloading Z-Image-Turbo text encoder (7GB)..."
            mkdir -p "$COMFYUI_DIR/models/clip"
            wget --progress=bar:force:noscroll -O "$TEXT_ENCODER" \
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
                || { echo "[ERROR] Failed to download text encoder"; MODELS_MISSING=true; }
        else
            echo "[OK] Text encoder found: qwen_3_4b.safetensors"
        fi

        # Check VAE
        if [ ! -f "$VAE_MODEL" ]; then
            echo "Downloading Z-Image-Turbo VAE (200MB)..."
            mkdir -p "$COMFYUI_DIR/models/vae"
            wget --progress=bar:force:noscroll -O "$VAE_MODEL" \
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
                || { echo "[ERROR] Failed to download VAE"; MODELS_MISSING=true; }
        else
            echo "[OK] VAE found: ae.safetensors"
        fi

        if [ "$MODELS_MISSING" = "false" ]; then
            echo "[OK] All Z-Image-Turbo models are available"
        else
            echo "[WARN] Some models failed to download, but continuing..."
        fi
    fi
else
    echo "[SKIP] Model checking disabled (CHECK_MODELS=false)"
fi

echo "[OK] Initialization completed successfully"
exit 0
