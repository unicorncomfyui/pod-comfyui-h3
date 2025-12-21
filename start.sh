#!/bin/bash
# Startup script for RunPod ComfyUI Pod with VSCode
# Starts code-server and ComfyUI with optimizations

set -e

echo "Starting RunPod ComfyUI Pod with VSCode..."

# Network Volume detection
# RunPod mounts network volumes at /workspace automatically
NETWORK_VOLUME_EXISTS=false

if [ -d "/workspace" ] && [ "$(ls -A /workspace 2>/dev/null)" ]; then
    echo "Network Volume detected at /workspace"
    NETWORK_VOLUME_EXISTS=true
elif [ -d "/runpod-volume" ] && [ ! -L "/workspace" ]; then
    echo "Network Volume detected at /runpod-volume"
    ln -sf /runpod-volume /workspace
    echo "[OK] Symlink created: /workspace -> /runpod-volume"
    NETWORK_VOLUME_EXISTS=true
elif [ -d "/workspace" ]; then
    echo "Empty network volume detected at /workspace"
    NETWORK_VOLUME_EXISTS=true
else
    echo "No network volume found, using container storage"
    mkdir -p /workspace
fi

# Initialize ComfyUI on network volume if available
if [ "$NETWORK_VOLUME_EXISTS" = "true" ] && [ ! -d "/workspace/ComfyUI" ]; then
    echo "First run: Copying ComfyUI to network volume for persistence..."
    echo "This will take 1-2 minutes..."
    cp -r /app/comfyui /workspace/ComfyUI
    echo "[OK] ComfyUI copied to /workspace/ComfyUI"
fi

# Determine ComfyUI location - PRIORITIZE network volume over container
if [ -d "/workspace/ComfyUI" ]; then
    COMFYUI_DIR="/workspace/ComfyUI"
    echo "Using network volume ComfyUI: $COMFYUI_DIR"
elif [ -d "/app/comfyui" ]; then
    COMFYUI_DIR="/app/comfyui"
    echo "Using container ComfyUI: $COMFYUI_DIR"
else
    echo "[ERROR] ComfyUI not found!"
    exit 1
fi

# Run initialization (SageAttention compilation/cache, VSCode extensions, etc.)
echo "Running initialization script..."
if ! bash /app/init.sh; then
    echo "[ERROR] CRITICAL: Initialization failed!"
    exit 1
fi

# Configure SSH if enabled
if [ "${ENABLE_SSH:-false}" = "true" ]; then
    echo "Configuring SSH server..."
    mkdir -p /var/run/sshd
    if [ -n "$SSH_PASSWORD" ]; then
        echo "root:$SSH_PASSWORD" | chpasswd
    fi
    /usr/sbin/sshd -D &
    echo "[OK] SSH server started on port 22"
fi

# Enable tcmalloc for memory optimization
if [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
    echo "[OK] tcmalloc enabled for memory optimization"
fi

# Start code-server in background
echo "Starting code-server (VSCode) on port ${VSCODE_PORT}..."
echo "Opening directory: $COMFYUI_DIR"

code-server \
    --bind-addr 0.0.0.0:${VSCODE_PORT} \
    --auth none \
    --disable-telemetry \
    --disable-update-check \
    "$COMFYUI_DIR" > /var/log/code-server.log 2>&1 &

CODE_SERVER_PID=$!
echo "[OK] code-server started (PID: $CODE_SERVER_PID)"
echo "     URL: http://0.0.0.0:${VSCODE_PORT}"

# Wait for code-server to be ready
sleep 5

# Start ComfyUI
echo "Starting ComfyUI on port ${COMFYUI_PORT}..."
cd "$COMFYUI_DIR"

python main.py \
    --listen 0.0.0.0 \
    --port ${COMFYUI_PORT} \
    --preview-method auto \
    2>&1 | tee /var/log/comfyui.log &

COMFYUI_PID=$!
echo "[OK] ComfyUI started (PID: $COMFYUI_PID)"
echo "     URL: http://0.0.0.0:${COMFYUI_PORT}"

# Monitor processes
echo ""
echo "[OK] All services started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Available Services:"
echo "  VSCode:  http://0.0.0.0:${VSCODE_PORT}  (no authentication)"
echo "  ComfyUI: http://0.0.0.0:${COMFYUI_PORT}"
if [ "${ENABLE_SSH:-false}" = "true" ]; then
    echo "  SSH:     port 22"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep container running and monitor processes
while true; do
    # Check if code-server is still running
    if ! kill -0 $CODE_SERVER_PID 2>/dev/null; then
        echo "[ERROR] code-server died, restarting..."
        code-server \
            --bind-addr 0.0.0.0:${VSCODE_PORT} \
            --auth none \
            --disable-telemetry \
            --disable-update-check \
            "$COMFYUI_DIR" > /var/log/code-server.log 2>&1 &
        CODE_SERVER_PID=$!
    fi

    # Check if ComfyUI is still running
    if ! kill -0 $COMFYUI_PID 2>/dev/null; then
        echo "[ERROR] ComfyUI died, restarting..."
        cd "$COMFYUI_DIR"
        python main.py \
            --listen 0.0.0.0 \
            --port ${COMFYUI_PORT} \
            --preview-method auto \
            2>&1 | tee /var/log/comfyui.log &
        COMFYUI_PID=$!
    fi

    sleep 30
done
