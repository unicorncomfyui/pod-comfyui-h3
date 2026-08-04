#!/usr/bin/env python3
"""Benchmark ComfyUI performance levers, one configuration at a time.

The levers (--fast, --cache-lru, --async-offload) are command-line arguments,
so comparing them means restarting ComfyUI. The pod's own instance is watched
by start.sh and would be restarted with its original arguments, so this script
spawns its OWN short-lived instances on a scratch port instead - one per
configuration - and leaves the running pod alone.

Timing comes from ComfyUI's own history entries (execution_start ->
execution_success), not from wall-clock around the HTTP call, so queue waits
and network latency stay out of the numbers.

Usage:
    python scripts/bench.py workflow_api.json
    python scripts/bench.py workflow_api.json --reps 5 --seed 42
    python scripts/bench.py workflow_api.json --config "fp16" --config "baseline"

Export the workflow from the ComfyUI menu with **Export (API)**, not the plain
save - the normal format is not accepted by /prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFYUI_HOME = os.environ.get("COMFYUI_HOME", "/app/comfyui")
DATA_DIR = os.environ.get("COMFYUI_DATA_DIR", "/workspace/comfyui-data")
PORT = int(os.environ.get("BENCH_PORT", "3111"))

# Each entry is (label, extra CLI args). "baseline" must stay first: every
# other configuration is reported as a delta against it.
CONFIGS: dict[str, list[str]] = {
    "baseline": [],
    "fp16": ["--fast", "fp16_accumulation"],
    "cublas": ["--fast", "cublas_ops"],
    "autotune": ["--fast", "autotune"],
    "fp8mm": ["--fast", "fp8_matrix_mult"],
    "fast-all": ["--fast"],
    "lru": ["--cache-lru", "10"],
    "offload4": ["--async-offload", "4"],
    "offload8": ["--async-offload", "8"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def api(path: str, payload: dict | None = None) -> dict:
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or "{}")


def wait_ready(proc: subprocess.Popen, timeout: int = 300) -> bool:
    """Poll until the server answers, or the process dies first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"[ERROR] ComfyUI exited early with code {proc.returncode}")
            return False
        try:
            api("/system_stats")
            return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(2)
    log("[ERROR] ComfyUI did not become ready in time")
    return False


def force_seed(workflow: dict, seed: int) -> int:
    """Pin every seed-like widget so runs are actually comparable."""
    n = 0
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs and not isinstance(inputs[key], list):
                inputs[key] = seed
                n += 1
    return n


def set_input(workflow: dict, name: str, value) -> int:
    """Set every widget called `name` across the workflow.

    Values that are lists are node links, not widgets - never overwrite those.
    """
    n = 0
    for node in workflow.values():
        inputs = node.get("inputs", {})
        if name in inputs and not isinstance(inputs[name], list):
            current = inputs[name]
            inputs[name] = type(current)(value) if current is not None else value
            n += 1
    return n


def parse_sweep(specs: list[str]) -> list[tuple[str, list]]:
    """'steps=8,12,20' -> ('steps', [8, 12, 20]), numbers kept numeric."""
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--sweep expects NAME=v1,v2,...  (got {spec!r})")
        name, raw = spec.split("=", 1)
        values = []
        for v in raw.split(","):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
        out.append((name.strip(), values))
    return out


def run_prompt(workflow: dict) -> float | None:
    """Queue one prompt and return ComfyUI's own execution time in seconds."""
    prompt_id = api("/prompt", {"prompt": workflow})["prompt_id"]

    while True:
        time.sleep(1)
        hist = api(f"/history/{prompt_id}")
        entry = hist.get(prompt_id)
        if not entry:
            continue

        status = entry.get("status", {})
        if not status.get("completed") and status.get("status_str") != "error":
            continue
        if status.get("status_str") == "error":
            log("[ERROR] Prompt failed - check the ComfyUI log")
            return None

        # messages: [["execution_start", {"timestamp": ms}], ...]
        stamps = {
            m[0]: m[1].get("timestamp")
            for m in status.get("messages", [])
            if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict)
        }
        start = stamps.get("execution_start")
        end = stamps.get("execution_success")
        if start and end:
            return (end - start) / 1000.0

        log("[WARN] No timestamps in history entry")
        return None


def sweep_in_instance(workflow: dict, sweeps: list[tuple[str, list]],
                      reps: int) -> list[dict]:
    """Sweep workflow parameters inside ONE running ComfyUI.

    Deliberately not one instance per point: parameter sweeps do not change
    the command line, so reloading the models for every combination would add
    minutes per point and buy nothing.
    """
    import itertools

    names = [n for n, _ in sweeps]
    rows = []
    for combo in itertools.product(*[v for _, v in sweeps]):
        wf = json.loads(json.dumps(workflow))  # deep copy per point
        label_parts, applied = [], True
        for name, value in zip(names, combo):
            hits = set_input(wf, name, value)
            if hits == 0:
                log(f"[WARN] No widget named '{name}' in this workflow - skipping")
                applied = False
            label_parts.append(f"{name}={value}")
        label = " ".join(label_parts)
        if not applied:
            rows.append({"label": label, "times": [], "error": "input not found"})
            continue

        log(f"\n--- {label} ---")
        times = []
        for i in range(reps + 1):
            t = run_prompt(wf)
            if t is None:
                break
            log(f"  {'warmup' if i == 0 else f'run {i}':>8}: {t:7.2f} s"
                + ("   (discarded)" if i == 0 else ""))
            if i > 0:
                times.append(t)
        rows.append({"label": label, "times": times})
    return rows


def bench_config(label: str, extra: list[str], workflow: dict, reps: int,
                 sweeps: list[tuple[str, list]] | None = None):
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(PORT),
        "--extra-model-paths-config", f"{DATA_DIR}/extra_model_paths.yaml",
        "--output-directory", f"{DATA_DIR}/output",
        "--input-directory", f"{DATA_DIR}/input",
        "--user-directory", f"{DATA_DIR}/user",
        "--preview-method", "none",   # previews would skew the measurement
        "--disable-all-custom-nodes", # keep third-party code out of the numbers
        *extra,
    ]
    log(f"\n{'=' * 62}\n{label}: {' '.join(extra) or '(no extra args)'}\n{'=' * 62}")

    proc = subprocess.Popen(
        cmd, cwd=COMFYUI_HOME,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        if not wait_ready(proc):
            fail = {"label": label, "times": [], "error": "startup failed"}
            return [fail] if sweeps else fail

        if sweeps:
            return sweep_in_instance(workflow, sweeps, reps)

        times: list[float] = []
        # +1: the first run pays model staging and any cold Triton kernels.
        # It is measured and shown, then dropped from the statistics.
        for i in range(reps + 1):
            t = run_prompt(workflow)
            if t is None:
                break
            tag = "warmup" if i == 0 else f"run {i}"
            log(f"  {tag:>8}: {t:7.2f} s" + ("   (discarded)" if i == 0 else ""))
            if i > 0:
                times.append(t)
        return {"label": label, "times": times}
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        time.sleep(5)  # let VRAM actually come back before the next config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workflow", help="workflow exported with Export (API)")
    p.add_argument("--reps", type=int, default=3, help="measured runs per config")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--config", action="append", dest="configs",
                   help="config name, repeatable; default is all of them")
    p.add_argument("--sweep", action="append",
                   help="NAME=v1,v2,... sweep a workflow widget; repeatable. "
                        "e.g. --sweep steps=8,12,20 --sweep megapixels=0.3,0.6,1.0")
    p.add_argument("--list", action="store_true", help="list configs and exit")
    args = p.parse_args()

    if args.list:
        for name, extra in CONFIGS.items():
            print(f"  {name:<12} {' '.join(extra) or '(no extra args)'}")
        return 0

    workflow = json.loads(Path(args.workflow).read_text(encoding="utf-8"))
    if not all(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
        log("[ERROR] This is not an API-format workflow.")
        log("        Re-export it from ComfyUI with Export (API).")
        return 1

    log(f"Pinned {force_seed(workflow, args.seed)} seed widget(s) to {args.seed}")

    if args.sweep:
        # Sweeping parameters shares one instance; sweeping CLI levers cannot.
        try:
            sweeps = parse_sweep(args.sweep)
        except ValueError as e:
            log(f"[ERROR] {e}")
            return 1
        total = 1
        for _, v in sweeps:
            total *= len(v)
        log(f"Sweeping {' x '.join(f'{n}({len(v)})' for n, v in sweeps)} "
            f"= {total} point(s), {args.reps} run(s) each")
        results = bench_config("sweep", CONFIGS[args.configs[0]] if args.configs
                               else [], workflow, args.reps, sweeps=sweeps)
    else:
        names = args.configs or list(CONFIGS)
        unknown = [n for n in names if n not in CONFIGS]
        if unknown:
            log(f"[ERROR] Unknown config(s): {', '.join(unknown)}")
            return 1
        if "baseline" not in names:
            names.insert(0, "baseline")
        results = [bench_config(n, CONFIGS[n], workflow, args.reps) for n in names]

    log(f"\n{'=' * 62}\nRESULTS  ({args.reps} runs each, warm-up discarded)\n{'=' * 62}")
    log(f"{'config':<12} {'median':>9} {'min':>9} {'max':>9} {'spread':>8}  vs base")

    base = None
    for r in results:
        if not r["times"]:
            log(f"{r['label']:<12} {'FAILED':>9}  {r.get('error', '')}")
            continue
        med = statistics.median(r["times"])
        lo, hi = min(r["times"]), max(r["times"])
        if base is None:
            base, delta = med, "  --"
        else:
            pct = (med - base) / base * 100
            delta = f"{pct:+6.1f}%"
        # Spread wider than the effect means the result is noise.
        log(f"{r['label']:<12} {med:8.2f}s {lo:8.2f}s {hi:8.2f}s "
            f"{(hi - lo) / med * 100:7.1f}% {delta}")

    log("\nRead spread before believing a delta: if it exceeds the difference")
    log("between two configs, you measured noise, not an improvement.")
    log("--fast features can degrade quality - watch the video, not just the clock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
