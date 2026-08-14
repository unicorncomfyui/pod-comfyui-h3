#!/bin/bash
# Model acquisition, run in the BACKGROUND by start.sh once code-server and
# ComfyUI are already listening.
#
# This used to live in init.sh, which start.sh waits on - so a fresh volume
# left the pod with no shell and no UI for the 15+ minutes it takes to pull
# 42 GB. Services first, weights second: ComfyUI simply does not list the
# model until its file lands, and a refresh picks it up.

set -u

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
if [ -n "${PREWARM_SET:-}" ] && [ -n "${COMFYUI_DIR}" ]; then
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


echo "[OK] Model acquisition finished"
