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
import re
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

# Set from --keep-custom-nodes. Custom nodes are excluded by default so that a
# third-party upscaler or frame interpolator cannot end up inside the numbers.
# That rule is too coarse on its own: workflows routinely route one widget
# through a small utility node - a math expression feeding a frame count, a
# resolution helper - and excluding those does not protect the measurement, it
# only stops you from benchmarking the graph you actually run.
KEEP_CUSTOM_NODES = False

# Base seed; repetition i runs with BASE_SEED + i. See force_seed().
BASE_SEED = 1234

# Measured on this project, by eye rather than by metric: below this the output
# loses too much to be worth shipping, so a timing taken there describes a
# setting nobody would use. It is a floor for BENCHMARKS, not a limit on what
# the pipeline accepts.
MEGAPIXEL_FLOOR = float(os.environ.get("BENCH_MEGAPIXEL_FLOOR", "0.6"))


def _sh(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - a missing tool must not sink the run
        return ""


def hardware() -> list[str]:
    """What these numbers were produced on.

    Absolute timings are NOT portable between pods. Two RTX 5090s measured
    here differed by 20% per sampling step - 2.37 s against 2.85 s - on
    nominally identical hardware. Board partners set their own power limits
    and fan curves, hosts differ in CPU and PCIe, and a machine can be shared.
    That spread is larger than most changes worth testing, so a table that
    does not say what it ran on is a table that cannot be compared to
    anything, including itself three weeks later.

    Every line is optional: this is documentation, and failing to collect it
    must never cost a measurement.
    """
    out: list[str] = []

    gpu = _sh(["nvidia-smi", "--format=csv,noheader",
               "--query-gpu=name,driver_version,vbios_version,"
               "power.max_limit,memory.total,clocks.max.sm"])
    if gpu:
        f = [x.strip() for x in gpu.splitlines()[0].split(",")]
        while len(f) < 6:
            f.append("?")
        out.append(f"GPU        {f[0]}  {f[4]}  cap {f[3]}  max SM {f[5]}")
        # VBIOS is the closest thing to a board-partner fingerprint that
        # nvidia-smi exposes; two cards of the same model rarely share one.
        out.append(f"driver     {f[1]}   vbios {f[2]}")

    part = ""
    for line in _sh(["nvidia-smi", "-q"]).splitlines():
        if "Board Part Number" in line:
            part = line.split(":", 1)[1].strip()
            break
    if part and part not in ("N/A", "Unknown"):
        out.append(f"board      {part}")

    try:
        import torch
        cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0)) \
            if torch.cuda.is_available() else "?"
        out.append(f"torch      {torch.__version__}  cuda {torch.version.cuda}"
                   f"  sm_{cap.replace('.', '')}")
    except Exception:  # noqa: BLE001
        pass

    try:
        from importlib.metadata import version
        out.append(f"sage       {version('sageattention')}")
    except Exception:  # noqa: BLE001
        out.append("sage       not installed (attention stays on PyTorch)")

    try:
        cpus = 0
        model = ""
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("processor"):
                cpus += 1
            elif line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
        # Decimal GB, not GiB. RunPod sells a "92 GB" pod and the cgroup holds
        # 86 GiB - the same number, and a stamp that disagrees with the invoice
        # is a stamp you stop trusting.
        mem = ""
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem = f"{int(line.split()[1]) * 1024 / 1e9:.0f} GB"
                break
        # /proc shows the HOST inside a container, not what this pod was
        # given: a 94 GB pod reported 756 GB and 224 vCPU here. The cgroup is
        # the allocation, and the allocation is what explains a slow run.
        quota = ""
        try:
            cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text().split()
            if cpu_max[0] != "max":
                quota = f"{int(cpu_max[0]) / int(cpu_max[1]):.0f} vCPU"
        except Exception:  # noqa: BLE001 - cgroup v1, or not containerised
            pass
        limit = ""
        for f in ("/sys/fs/cgroup/memory.max",
                  "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:
                raw = Path(f).read_text().strip()
                if raw != "max" and int(raw) < (1 << 50):
                    limit = f"{int(raw) / 1e9:.0f} GB"
                break
            except Exception:  # noqa: BLE001
                continue
        if model:
            out.append(f"host       {cpus} vCPU {model}  {mem} RAM")
            if quota or limit:
                out.append(f"pod limits {quota or 'no cpu quota'}"
                           f"  {limit or 'no memory limit'}")
    except Exception:  # noqa: BLE001
        pass

    pod = os.environ.get("RUNPOD_POD_ID") or os.environ.get("HOSTNAME", "")
    tag = os.environ.get("IMAGE_TAG", "")
    if pod or tag:
        out.append(f"pod        {pod or '?'}" + (f"   image {tag}" if tag else ""))
    return out


def fit_steps(results: list[dict]) -> list[str]:
    """Split the time into what the step count buys and what it does not.

    A sweep over `steps` gives this for free, and the two numbers are worth
    more than the totals: the fixed part (prompt encode, VAE, decode, write)
    does not move when you change step count, so a percentage against a total
    understates what more steps actually cost. Least squares, so three points
    are better than two - with only two the line passes through them exactly
    and says nothing about whether the model holds.
    """
    import re
    pts = []
    for r in results:
        m = re.search(r"steps=(\d+)", r.get("label", ""))
        if m and r.get("times"):
            pts.append((int(m.group(1)), statistics.median(r["times"])))
    if len({s for s, _ in pts}) < 2:
        return []
    n = len(pts)
    mx = sum(s for s, _ in pts) / n
    my = sum(t for _, t in pts) / n
    var = sum((s - mx) ** 2 for s, _ in pts)
    if var == 0:
        return []
    k = sum((s - mx) * (t - my) for s, t in pts) / var
    fixed = my - k * mx
    lines = [f"\nsteps model  {fixed:.2f}s fixed + {k:.2f}s per step "
             f"({n} point(s))"]
    if fixed > 0:
        lines.append("             predicted: " + "  ".join(
            f"{s}->{fixed + k * s:.1f}s" for s in (4, 6, 8, 12)))
    if n == 2:
        lines.append("             two points fit any line exactly - add a "
                     "third before trusting the split")
    return lines

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
    # Needs the sageattention package; ComfyUI falls back to PyTorch attention
    # and says so if it is missing, so this config degrades to a second
    # baseline rather than failing. Measure it at 0.88 MP: attention cost grows
    # as pixels^1.84, so at 0.4 MP most of the clock is elsewhere and the
    # comparison answers a question you are not asking.
    "sage": ["--use-sage-attention"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def api(path: str, payload: dict | None = None) -> dict:
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        # /prompt refuses a graph with a 400 whose BODY carries the reason -
        # which node, which input. Letting the exception through printed a
        # bare "HTTP Error 400: Bad Request" and sent you reading tracebacks
        # for something ComfyUI had already explained.
        body = e.read().decode("utf-8", "replace")
        log(f"[ERROR] {path} -> {e.code}")
        try:
            err = json.loads(body).get("error", {})
            log(f"        {err.get('type', '?')}: {err.get('message', body[:300])}")
            for d in json.loads(body).get("node_errors", {}).values():
                for m in d.get("errors", []):
                    log(f"        {m.get('details', m)}")
        except json.JSONDecodeError:
            log(f"        {body[:400]}")
        raise


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
    """Set every seed-like widget.

    Called with a DIFFERENT value before each repetition, which is the point:
    ComfyUI caches per node on its inputs, so resubmitting a byte-identical
    prompt executes nothing at all and reports a runtime of zero. Varying the
    seed invalidates the sampler's cache entry without changing the amount of
    work done - the value a seed holds has no effect on how long sampling
    takes - so the runs stay comparable and actually happen.
    """
    n = 0
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs and not isinstance(inputs[key], list):
                inputs[key] = seed
                n += 1
    return n


def set_input(workflow: dict, name: str, value) -> int:
    """Set a widget across the workflow, optionally on one node only.

    `name` may be `field` or `target.field`, where target is a node id or a
    class_type. Some fields legitimately appear on several nodes that must NOT
    move together: this graph loads two VAEs through the same `VAELoader`
    class, and sweeping `vae_name` would hand the video VAE's replacement to
    the audio one, which has no such file and fails the run. Naming the node
    is the only way to say which of the two you meant.

    Values that are lists are node links, not widgets - never overwrite those.
    """
    target, _, field = name.rpartition(".")
    n = 0
    for node_id, node in workflow.items():
        if target and target not in (node_id, node.get("class_type")):
            continue
        inputs = node.get("inputs", {})
        if field in inputs and not isinstance(inputs[field], list):
            current = inputs[field]
            inputs[field] = type(current)(value) if current is not None else value
            n += 1
    return n


def label_outputs(workflow: dict, label: str) -> None:
    """Name the files after the point that produced them.

    Every run otherwise lands under the template's own prefix, so a sweep
    leaves a pile of `MiniMax_H3_00042.mp4` and the only way to tell which
    setting made which clip is the modification time. That turns the visual
    half of a benchmark - the half that actually decides anything - into
    guesswork.

    The graph's own prefix is KEPT as a suffix, not replaced. A workflow with
    two save nodes distinguishes them there - the VAE comparison writes `fp16`
    and `int8_convrot` - and overwriting both with the config name destroyed
    exactly the distinction the run existed to make. Eight files, four per
    config, none of them attributable.
    """
    safe = re.sub(r"[^A-Za-z0-9.=-]+", "_", label).strip("_") or "run"
    for node in workflow.values():
        inputs = node.get("inputs", {})
        current = inputs.get("filename_prefix")
        if not isinstance(current, str):
            continue
        leaf = re.sub(r"[^A-Za-z0-9.=-]+", "_", current.split("/")[-1]).strip("_")
        inputs["filename_prefix"] = f"bench/{safe}/{safe}_{leaf}" if leaf \
            else f"bench/{safe}/{safe}"


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


class VramWatch:
    """Peak VRAM across a run, sampled by nvidia-smi at 200 ms.

    The first version polled /system_stats inside the one-second loop that
    watches the queue. It was free, and it was too coarse: repeated runs of
    the SAME configuration reported peaks 1-2 GB apart, while the effect being
    measured - the upstream H3 VAE rework - is worth about 3.5 GB. An
    instrument whose noise is half the signal cannot settle the question.

    nvidia-smi in its own process at 200 ms costs nothing on the GPU, does not
    contend for the GIL, and does not add HTTP load to the server being timed.
    Total used on the device, not torch's accounting: the offloader and the
    VAE both count against the same 32 GB.

    Still a high-water mark of what was SEEN. A spike shorter than 200 ms can
    hide - but VAE decode, the phase this measures, lasts seconds.
    """

    def __init__(self) -> None:
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:  # noqa: BLE001 - instrumentation never fails a run
            self.proc = None
        return self

    def __exit__(self, *exc) -> None:
        pass

    def peak_mb(self) -> float | None:
        if not self.proc:
            return None
        try:
            self.proc.terminate()
            out, _ = self.proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
            return None
        vals = [float(x) for x in (out or "").split() if x.strip().isdigit()]
        return max(vals) if vals else None


# Peak seen during the last run_prompt(), in MB. A module global because
# run_prompt returns one number and its callers already build rows from it.
PEAK_VRAM_MB: float | None = None


def run_prompt(workflow: dict) -> float | None:
    """Queue one prompt and return ComfyUI's own execution time in seconds.

    Also records the peak VRAM seen while it ran. The polling loop below
    already wakes every second, so sampling there is free - but a spike
    shorter than a second can be missed. VAE decode, the phase this project
    cares about, lasts several seconds, so it is seen; treat the number as a
    high-water mark of what was observed, not a proven maximum.
    """
    global PEAK_VRAM_MB
    PEAK_VRAM_MB = None
    watch = VramWatch().__enter__()
    try:
        return _run_prompt(workflow)
    finally:
        PEAK_VRAM_MB = watch.peak_mb()


def _run_prompt(workflow: dict) -> float | None:
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
            # The history entry already carries the exception, the node class
            # that raised it and a traceback. Printing "check the ComfyUI log"
            # instead sent the reader to a spawned instance whose output is
            # not even on screen - the answer was in hand and thrown away.
            log("[ERROR] Prompt failed:")
            for m in status.get("messages", []):
                if not (isinstance(m, list) and len(m) == 2):
                    continue
                if m[0] != "execution_error":
                    continue
                d = m[1] if isinstance(m[1], dict) else {}
                log(f"        node  {d.get('node_type', '?')} "
                    f"(#{d.get('node_id', '?')})")
                log(f"        error {str(d.get('exception_message', ''))[:300]}")
                for line in (d.get("traceback") or [])[-4:]:
                    log(f"        | {str(line).rstrip()[:160]}")
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
        label_outputs(wf, label)
        times, peaks = [], []
        for i in range(reps + 1):
            force_seed(wf, BASE_SEED + i)
            t = run_prompt(wf)
            if t is None:
                break
            log(f"  {'warmup' if i == 0 else f'run {i}':>8}: {t:7.2f} s"
                + (f"  peak {PEAK_VRAM_MB / 1024:.1f} GB" if PEAK_VRAM_MB else "")
                + ("   (discarded)" if i == 0 else ""))
            if i > 0:
                times.append(t)
                if PEAK_VRAM_MB:
                    peaks.append(PEAK_VRAM_MB)
        rows.append({"label": label, "times": times, "peaks": peaks})
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
    ]
    if not KEEP_CUSTOM_NODES:
        cmd.append("--disable-all-custom-nodes")
    cmd += extra
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

        label_outputs(workflow, label)
        times: list[float] = []
        peaks: list[float] = []
        # +1: the first run pays model staging and any cold Triton kernels.
        # It is measured and shown, then dropped from the statistics.
        for i in range(reps + 1):
            force_seed(workflow, BASE_SEED + i)
            t = run_prompt(workflow)
            if t is None:
                break
            tag = "warmup" if i == 0 else f"run {i}"
            log(f"  {tag:>8}: {t:7.2f} s"
                + (f"  peak {PEAK_VRAM_MB / 1024:.1f} GB" if PEAK_VRAM_MB else "")
                + ("   (discarded)" if i == 0 else ""))
            if i > 0:
                times.append(t)
                if PEAK_VRAM_MB:
                    peaks.append(PEAK_VRAM_MB)
        return {"label": label, "times": times, "peaks": peaks}
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        time.sleep(5)  # let VRAM actually come back before the next config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Optional, then checked by hand: as a required positional it would also be
    # demanded by --list, which needs no workflow and never reads one.
    p.add_argument("workflow", nargs="?",
                   help="workflow exported with Export (API)")
    p.add_argument("--reps", type=int, default=3, help="measured runs per config")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--config", action="append", dest="configs",
                   help="config name, repeatable; default is all of them")
    p.add_argument("--sweep", action="append",
                   help="NAME=v1,v2,... sweep a workflow widget; repeatable. "
                        "e.g. --sweep steps=8,12,20 --sweep megapixels=0.3,0.6,1.0. "
                        "Prefix with a node id or class to target one node: "
                        "--sweep 105:11.vae_name=a.safetensors,b.safetensors")
    p.add_argument("--keep-custom-nodes", action="store_true",
                   help="leave custom nodes enabled. Needed when the workflow "
                        "routes a widget through a utility node, which would "
                        "otherwise make every run fail on an unknown class_type")
    p.add_argument("--list", action="store_true", help="list configs and exit")
    args = p.parse_args()

    global KEEP_CUSTOM_NODES
    KEEP_CUSTOM_NODES = args.keep_custom_nodes

    if args.list:
        for name, extra in CONFIGS.items():
            print(f"  {name:<12} {' '.join(extra) or '(no extra args)'}")
        return 0

    if not args.workflow:
        p.error("a workflow is required (or use --list on its own)")

    # A bare `bench/x.json` is how the shipped graphs are named in every
    # example, but the path resolves against the working directory - and there
    # is no reason to be sitting in /app when running this. Fall back to the
    # image's own workflow directory before giving up, and say which file was
    # actually opened so a stale copy elsewhere cannot masquerade as this one.
    wf_path = Path(args.workflow)
    if not wf_path.is_file():
        candidate = Path(COMFYUI_HOME).parent / "workflows" / args.workflow
        if candidate.is_file():
            wf_path = candidate
        else:
            log(f"[ERROR] Workflow not found: {args.workflow}")
            log(f"        Tried {Path(args.workflow).resolve()}")
            log(f"        and   {candidate}")
            return 1
    log(f"Workflow: {wf_path}")
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    if not all(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
        log("[ERROR] This is not an API-format workflow.")
        log("        Re-export it from ComfyUI with Export (API).")
        return 1

    # Below 0.6 MP this project judges the output too degraded to act on, so a
    # timing measured there is a number nobody will use. Warn rather than
    # refuse: measuring the cheap end on purpose is legitimate, forgetting
    # where you are is not.
    sizes = [n["inputs"].get("megapixels") for n in workflow.values()
             if n.get("class_type") == "ResolutionSelector"]
    if args.sweep:
        swept = [v for name, vals in parse_sweep(args.sweep)
                 if name == "megapixels" for v in vals]
        sizes = swept or sizes
    for mp in sizes:
        if isinstance(mp, (int, float)) and mp < MEGAPIXEL_FLOOR:
            log(f"[WARN] {mp} MP is below the {MEGAPIXEL_FLOOR} MP floor this "
                f"project judges usable - the timing will not describe a "
                f"setting you would ship.")

    # Refuse to share the scratch port. wait_ready() only asks whether anything
    # answers there, so a leftover instance - or a second copy of this script -
    # would be mistaken for our own: every prompt would go to a server started
    # with someone else's flags, and the numbers would look plausible and mean
    # nothing. Two runs writing one log is how that gets noticed, far too late.
    try:
        api("/system_stats")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        pass
    else:
        log(f"[ERROR] Something already answers on port {PORT}.")
        log("        That is either a leftover bench instance or another copy")
        log("        of this script. Stop it first, or set BENCH_PORT:")
        log("            pkill -f bench.py")
        log(f"            pkill -f 'main.py.*--port {PORT}'")
        return 1

    global BASE_SEED
    BASE_SEED = args.seed

    n_seeds = force_seed(workflow, BASE_SEED)
    log(f"Found {n_seeds} seed widget(s); repetitions run at {BASE_SEED}+i")
    if n_seeds == 0:
        # Without one, every repetition resubmits a byte-identical prompt,
        # ComfyUI serves it from cache, and the whole run reports 0.00 s.
        log("[ERROR] No seed widget found, so repeated runs cannot be forced to")
        log("        execute - they would all be served from ComfyUI's cache.")
        log("        Add a sampler seed to the workflow and re-export it.")
        return 1

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
    # Before the numbers, not after: the machine is part of the result.
    for line in hardware():
        log(line)
    log("")
    log(f"{'config':<12} {'median':>9} {'min':>9} {'max':>9} {'spread':>8}"
        f"  {'peak VRAM':>10}  vs base")

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
        # Peak VRAM is the point of some changes and invisible to the clock:
        # the H3 VAE rework upstream moved 14.6 s to 13.9 s while freeing 3.5 GB.
        peaks = r.get("peaks") or []
        vram = f"{max(peaks) / 1024:8.1f} GB" if peaks else f"{'-':>11}"
        # Spread wider than the effect means the result is noise.
        log(f"{r['label']:<12} {med:8.2f}s {lo:8.2f}s {hi:8.2f}s "
            f"{(hi - lo) / med * 100:7.1f}% {vram} {delta}")

    for line in fit_steps(results):
        log(line)

    log("\nRead spread before believing a delta: if it exceeds the difference")
    log("between two configs, you measured noise, not an improvement.")
    log("Peak VRAM is sampled at 200 ms - a high-water mark of what was seen,")
    log("not a proven maximum. It IS comparable between pods: the same card and")
    log("the same weights use the same memory, whatever the host does.")
    log("Absolute times belong to THIS pod - two 5090s here differed by 20%.")
    log("Compare configs within one run; carry ratios between pods, not seconds.")
    log("--fast features can degrade quality - watch the video, not just the clock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
