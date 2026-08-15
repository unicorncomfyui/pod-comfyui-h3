#!/usr/bin/env python3
"""Render several prompts through one graph under identical conditions.

This is not bench.py. bench.py answers "which setting is faster" and its output
is a table of seconds. This answers "which prompt is better", and no script can
read that off a clock - the answer is in the video. What a script CAN do is
make the comparison honest: hold the seed, the references, the resolution, the
step count and the model fixed, vary one thing, and label the results so they
can be judged without knowing which is which.

Two days of prompt iteration here left no record of which wording produced
which render. That is the actual problem this solves. Every run writes a
manifest next to the videos: arm, seed, output file, and the prompt itself.

Unlike bench.py this uses the POD'S OWN ComfyUI rather than spawning its own.
Reloading 37 GB of weights between arms would cost about two minutes each and
change nothing about the comparison, and there is no command-line lever under
test here - only the text.

Usage:
    python /app/scripts/prompt_ab.py /workspace/comfyui-data/prompts
    python /app/scripts/prompt_ab.py prompts/ --reps 3
    python /app/scripts/prompt_ab.py prompts/ --set 137.image=karamazov.png \\
                                              --set 139.image=coffre.png

Each *.txt in the directory is one arm, named after the file. Write the
hand-written control as control.txt and it will sit in the same table as the
generated ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFYUI_HOME = os.environ.get("COMFYUI_HOME", "/app/comfyui")
DATA_DIR = os.environ.get("COMFYUI_DATA_DIR", "/workspace/comfyui-data")
PORT = int(os.environ.get("COMFYUI_PORT", "3000"))

DEFAULT_GRAPH = "bench/promptgen_api.json"

# Which widget receives the prompt. Named by class rather than by node id so a
# re-exported graph with renumbered nodes still works; --prompt-node overrides
# it when a graph carries more than one multiline string.
PROMPT_CLASS = "PrimitiveStringMultiline"


def log(msg: str = "") -> None:
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
        # /prompt refuses a graph with a 400 whose body names the node and the
        # input. Printing the bare status sends the reader to the logs for
        # something ComfyUI already explained.
        body = e.read().decode("utf-8", "replace")
        log(f"[ERROR] {path} -> {e.code}")
        try:
            parsed = json.loads(body)
            err = parsed.get("error", {})
            log(f"        {err.get('type', '?')}: "
                f"{err.get('message', body[:300])}")
            for d in parsed.get("node_errors", {}).values():
                for m in d.get("errors", []):
                    log(f"        {m.get('details', m)}")
        except json.JSONDecodeError:
            log(f"        {body[:400]}")
        raise


def resolve_graph(name: str) -> Path | None:
    """Accept a real path, or a name relative to the workflow directory.

    Two roots, because the graphs live in two places that are both legitimate:
    /app/workflows on a pod, and the repository checkout when the script is
    being edited. Falling back to the script's own parent means an unmodified
    command line works in either, which is the difference between testing a
    change and rebuilding an image to test a change.
    """
    p = Path(name)
    if p.is_file():
        return p
    for root in (Path(COMFYUI_HOME).parent / "workflows",
                 Path(__file__).resolve().parent.parent / "workflows"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def find_prompt_node(graph: dict, explicit: str | None) -> tuple[str, str]:
    """Return (node_id, field) for the widget that carries the prompt."""
    if explicit:
        node_id, _, field = explicit.partition(".")
        if node_id not in graph:
            raise SystemExit(f"[ERROR] No node '{node_id}' in this graph.")
        return node_id, (field or "value")

    hits = [nid for nid, n in graph.items()
            if n.get("class_type") == PROMPT_CLASS]
    if len(hits) == 1:
        return hits[0], "value"
    if not hits:
        raise SystemExit(
            f"[ERROR] No {PROMPT_CLASS} node in this graph. Name the widget "
            f"yourself with --prompt-node NODE_ID.field")
    raise SystemExit(
        f"[ERROR] {len(hits)} {PROMPT_CLASS} nodes ({', '.join(hits)}). "
        f"Say which one with --prompt-node NODE_ID.field")


def set_input(graph: dict, spec: str) -> int:
    """'137.image=x.png' or 'steps=8'. Never overwrites a link (a list)."""
    target_field, _, value = spec.partition("=")
    if not value:
        raise SystemExit(f"[ERROR] --set expects NODE.field=value (got {spec!r})")
    target, _, field = target_field.rpartition(".")
    n = 0
    for node_id, node in graph.items():
        if target and target not in (node_id, node.get("class_type")):
            continue
        inputs = node.get("inputs", {})
        if field in inputs and not isinstance(inputs[field], list):
            current = inputs[field]
            try:
                inputs[field] = type(current)(value) if current is not None \
                    else value
            except (TypeError, ValueError):
                inputs[field] = value
            n += 1
    return n


def force_seed(graph: dict, seed: int) -> int:
    n = 0
    for node in graph.values():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs and not isinstance(inputs[key], list):
                inputs[key] = seed
                n += 1
    return n


def label_outputs(graph: dict, arm: str, seed: int) -> None:
    """Name every file after the arm that produced it.

    Without this, a five-arm comparison lands as a pile of
    MiniMax_H3_000NN.mp4 and the only way to tell them apart is the
    modification time - which is exactly the record-keeping failure this
    script exists to fix.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", arm).strip("_") or "arm"
    for node in graph.values():
        inputs = node.get("inputs", {})
        if isinstance(inputs.get("filename_prefix"), str):
            inputs["filename_prefix"] = f"promptgen/{safe}/{safe}_seed{seed}"


def run_prompt(graph: dict) -> tuple[float | None, list[str]]:
    """Queue one prompt; return (seconds, output filenames)."""
    prompt_id = api("/prompt", {"prompt": graph})["prompt_id"]

    while True:
        time.sleep(1)
        entry = api(f"/history/{prompt_id}").get(prompt_id)
        if not entry:
            continue
        status = entry.get("status", {})
        if not status.get("completed") and status.get("status_str") != "error":
            continue

        if status.get("status_str") == "error":
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
            return None, []

        stamps = {
            m[0]: m[1].get("timestamp")
            for m in status.get("messages", [])
            if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict)
        }
        files: list[str] = []
        for out in (entry.get("outputs") or {}).values():
            for key in ("videos", "images", "gifs", "audio"):
                for item in out.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        sub = item.get("subfolder") or ""
                        files.append(f"{sub}/{item['filename']}".lstrip("/"))

        start, end = stamps.get("execution_start"), stamps.get("execution_success")
        return ((end - start) / 1000.0 if start and end else None), files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompts", help="directory of *.txt, one file per arm")
    p.add_argument("--graph", default=DEFAULT_GRAPH,
                   help=f"API-format workflow (default {DEFAULT_GRAPH})")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--reps", type=int, default=1,
                   help="seeds per arm; every arm gets the SAME seeds")
    p.add_argument("--prompt-node", help="NODE_ID.field carrying the prompt")
    p.add_argument("--set", action="append", dest="overrides", default=[],
                   help="NODE.field=value, repeatable - point the graph at "
                        "your own reference images")
    args = p.parse_args()

    prompt_dir = Path(args.prompts)
    if not prompt_dir.is_dir():
        log(f"[ERROR] Not a directory: {prompt_dir}")
        return 1
    arms = sorted(prompt_dir.glob("*.txt"))
    if not arms:
        log(f"[ERROR] No *.txt in {prompt_dir}.")
        log("        One file per arm; name the hand-written one control.txt.")
        return 1

    graph_path = resolve_graph(args.graph)
    if graph_path is None:
        log(f"[ERROR] Workflow not found: {args.graph}")
        return 1
    base = json.loads(graph_path.read_text(encoding="utf-8"))
    if not all(isinstance(v, dict) and "class_type" in v for v in base.values()):
        log("[ERROR] Not an API-format workflow. Re-export with Export (API).")
        return 1

    try:
        api("/system_stats")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        log(f"[ERROR] Nothing answers on port {PORT}.")
        log("        This script drives the pod's own ComfyUI - start it first,")
        log("        or set COMFYUI_PORT if it listens elsewhere.")
        return 1

    node_id, field = find_prompt_node(base, args.prompt_node)

    log(f"Graph        {graph_path}")
    log(f"Prompt into  node {node_id}.{field}")
    log(f"Arms         {', '.join(a.stem for a in arms)}")
    log(f"Seeds        {', '.join(str(args.seed + i) for i in range(args.reps))}")
    for spec in args.overrides:
        hits = set_input(base, spec)
        log(f"Override     {spec}  ({hits} widget(s))")
        if hits == 0:
            log("[WARN] That matched nothing - check the node id and field.")

    if not force_seed(dict(base), args.seed):
        log("[ERROR] No seed widget in this graph, so the arms cannot be held")
        log("        to the same noise - which is the only thing that makes")
        log("        this a comparison rather than two unrelated renders.")
        return 1

    records: list[dict] = []
    for arm_path in arms:
        arm = arm_path.stem
        text = arm_path.read_text(encoding="utf-8").strip()
        if not text:
            log(f"\n[WARN] {arm_path.name} is empty - skipped.")
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

        log(f"\n--- {arm}  ({len(text.split())} words, {digest}) ---")
        for i in range(args.reps):
            seed = args.seed + i
            graph = json.loads(json.dumps(base))  # deep copy per run
            graph[node_id]["inputs"][field] = text
            force_seed(graph, seed)
            label_outputs(graph, arm, seed)

            seconds, files = run_prompt(graph)
            if seconds is None and not files:
                records.append({"arm": arm, "seed": seed, "prompt_sha": digest,
                                "error": "failed"})
                break
            log(f"  seed {seed}: {seconds:7.2f} s   "
                + (files[0] if files else "(no output file reported)"))
            records.append({
                "arm": arm,
                "seed": seed,
                "prompt_sha": digest,
                "seconds": round(seconds, 2) if seconds else None,
                "outputs": files,
                "prompt": text,
            })

    out_dir = Path(DATA_DIR) / "output" / "promptgen"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = out_dir / f"run-{stamp}.json"
    manifest.write_text(json.dumps({
        "graph": str(graph_path),
        "seeds": [args.seed + i for i in range(args.reps)],
        "overrides": args.overrides,
        "runs": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log(f"\n{'=' * 62}")
    log(f"{'arm':<20} {'seed':>8} {'seconds':>9}  output")
    for r in records:
        if r.get("error"):
            log(f"{r['arm']:<20} {r['seed']:>8} {'FAILED':>9}")
            continue
        outs = r.get("outputs") or []
        log(f"{r['arm']:<20} {r['seed']:>8} {r.get('seconds') or 0:9.2f}  "
            f"{outs[0] if outs else '-'}")

    log(f"\nManifest  {manifest}")
    log(f"Videos    {out_dir}/<arm>/")
    log("")
    log("The seconds are not the result. Prompts of similar length cost the")
    log("same to render - what changed is what is IN the frame. Watch the")
    log("videos with the arm names hidden, pick a winner, then read the")
    log("manifest. Judging while knowing which arm you are watching is how")
    log("a preference for the tool you just installed gets recorded as a")
    log("measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
