#!/usr/bin/env python3
"""Run generation jobs against the pod's ComfyUI, one at a time.

This is the local-first half of the pipeline: jobs arrive as JSON files in an
inbox directory and results land in an outbox. Everything the eventual queue
version needs is already here - template injection, image upload, /history
polling, the result manifest - and only the three seams marked SEAM below
change when the queue becomes real. That ordering is deliberate: the business
logic is what breaks, and it can be debugged on the pod with nothing created
on AWS.

    python scripts/worker.py --job examples/job.example.json --dry-run
    python scripts/worker.py --once                  drain the inbox and exit
    python scripts/worker.py                         poll forever

A job is a JSON object; every field but `prompt` is optional and falls back to
whatever the template already holds:

    {
      "id":           "mouse-001",
      "prompt":       "Editorial tech product film. ...",
      "image":        "in/mouse.png"  |  "https://...",
      "duration":     5.0,
      "steps":        12,
      "seed":         null,
      "aspect_ratio": "2:3 (Portrait Photo)",
      "megapixels":   0.4
    }

Concurrency is one, deliberately. A single 5090 holds one H3 workflow
resident; a second concurrent prompt does not overlap with the first, it
evicts it - and on a small pod it triggers the OOM killer instead.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:3000").rstrip("/")
POLL_SECONDS = int(os.environ.get("WORKER_POLL_SECONDS", "5"))

# How a logical job field finds its widget in the graph.
#
# Matching on the widget name alone is not enough: the reference workflow
# carries `megapixels` on both the resolution selector and an image rescale
# node, and `value` on any primitive. So a binding is (widget name, class
# filter), and a selector matching anything other than exactly one node is an
# error - a template edited underneath us must stop the worker, not get
# written to in an arbitrary place.
#
# Node IDs are deliberately absent. A workflow exported out of a subgraph
# numbers its nodes "105:104"; re-exporting renumbers them. Names and classes
# survive that, IDs do not.
BINDINGS: dict[str, dict] = {
    "prompt":        {"input": "prompt", "class": "MiniMaxH3"},
    "image":         {"input": "image", "class": "LoadImage"},
    "duration":      {"input": "value", "class": "PrimitiveFloat"},
    "steps":         {"input": "steps", "class": "BasicScheduler"},
    "seed":          {"input": "noise_seed", "class": "RandomNoise"},
    "aspect_ratio":  {"input": "aspect_ratio", "class": "ResolutionSelector"},
    "megapixels":    {"input": "megapixels", "class": "ResolutionSelector"},
    "output_prefix": {"input": "filename_prefix", "class": "SaveVideo"},
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ComfyUI groups a node's outputs under a key that describes the node, not the
# file: SaveVideo files arrive under "images", so a consumer trusting that key
# would treat an mp4 as a still. The extension is the honest answer, and the
# original key is kept as comfy_key so nothing is lost.
_KIND_BY_SUFFIX = {
    ".mp4": "video", ".webm": "video", ".mkv": "video", ".mov": "video",
    ".flac": "audio", ".wav": "audio", ".mp3": "audio", ".ogg": "audio",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
}


def media_kind(filename: str, comfy_key: str) -> str:
    return _KIND_BY_SUFFIX.get(Path(filename).suffix.lower(), comfy_key)


# ---------------------------------------------------------------------------
# ComfyUI HTTP
# ---------------------------------------------------------------------------
def explain_http_error(body: str) -> str:
    """Turn ComfyUI's rejection body into something a human can act on.

    /prompt answers a refused graph with 400 and a JSON body naming the node
    and the field. urllib puts that body on the exception rather than in the
    message, so the default rendering is a bare "HTTP Error 400: Bad Request"
    - the one message that says nothing about which widget is wrong.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body[:800]

    lines = []
    err = data.get("error") or {}
    if err:
        detail = err.get("details") or ""
        lines.append(f"{err.get('type', 'error')}: {err.get('message', '')} {detail}".strip())
    for nid, node_err in (data.get("node_errors") or {}).items():
        klass = node_err.get("class_type", "?")
        for d in node_err.get("errors", []):
            lines.append(f"  node {nid} ({klass}): {d.get('message')} - {d.get('details')}")
    return "\n".join(lines) or body[:800]


def api(path: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{COMFY_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"HTTP {e.code} from {path}\n{explain_http_error(body)}"
        ) from None


def upload_image(path: Path) -> str:
    """Push an image into ComfyUI's input directory, return its LoadImage name.

    Hand-rolled multipart rather than `requests`, so this file runs on a bare
    Python - including a laptop driving the pod through an SSH tunnel.
    """
    boundary = f"----worker{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode()
        + f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode()
        + b"Content-Type: application/octet-stream\r\n\r\n"
        + path.read_bytes() + b"\r\n",
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"{COMFY_URL}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        got = json.loads(r.read())

    # LoadImage addresses a nested file as "subfolder/name".
    sub = got.get("subfolder") or ""
    return f"{sub}/{got['name']}" if sub else got["name"]


def fetch_output(item: dict, dest: Path) -> Path:
    """Pull one produced file out of ComfyUI by its history entry.

    Goes through /view rather than reading the output directory, so the worker
    does not have to sit on the same filesystem as ComfyUI.
    """
    query = urllib.parse.urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{COMFY_URL}/view?{query}", timeout=600) as r:
        dest.write_bytes(r.read())
    return dest


def submit_and_wait(graph: dict, timeout: int = 3600) -> tuple[float, list[dict]]:
    """Queue a prompt, block until it finishes, return (seconds, output items).

    The duration comes from ComfyUI's own execution_start/execution_success
    stamps rather than wall clock, so a queue wait behind another job is not
    billed to this one.
    """
    prompt_id = api("/prompt", {"prompt": graph})["prompt_id"]
    log(f"  queued {prompt_id}")
    deadline = time.time() + timeout

    while time.time() < deadline:
        time.sleep(2)
        entry = api(f"/history/{prompt_id}").get(prompt_id)
        if not entry:
            continue

        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI reported an execution error ({prompt_id})")
        if not status.get("completed"):
            continue

        stamps = {
            m[0]: m[1].get("timestamp")
            for m in status.get("messages", [])
            if isinstance(m, list) and len(m) == 2 and isinstance(m[1], dict)
        }
        start, end = stamps.get("execution_start"), stamps.get("execution_success")
        elapsed = (end - start) / 1000.0 if start and end else -1.0

        # Outputs are grouped per node then per history key; flatten, because a
        # caller only cares which files came out.
        items = []
        for node_out in entry.get("outputs", {}).values():
            for key, files in node_out.items():
                if not isinstance(files, list):
                    continue
                items += [{**f, "kind": media_kind(f["filename"], key),
                           "comfy_key": key}
                          for f in files
                          if isinstance(f, dict) and "filename" in f]
        return elapsed, items

    raise TimeoutError(f"{prompt_id} did not finish within {timeout}s")


# ---------------------------------------------------------------------------
# Template injection
# ---------------------------------------------------------------------------
def find_node(graph: dict, binding: dict) -> tuple[str, dict]:
    """Locate the single node a binding refers to, or explain why it cannot."""
    name, klass = binding["input"], binding.get("class", "")
    hits = [
        (nid, node) for nid, node in graph.items()
        if klass in node.get("class_type", "")
        and name in node.get("inputs", {})
        # A list value is a link to another node, not a widget. Writing a
        # literal over one would silently sever an edge.
        and not isinstance(node["inputs"][name], list)
    ]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"no node of class ~{klass!r} exposes widget {name!r}")
    raise LookupError(
        f"{name!r} on ~{klass!r} is ambiguous - matched "
        + ", ".join(nid for nid, _ in hits)
    )


def coerce(current, value):
    """Match the widget's type without silently truncating.

    The template holds JSON literals, so a FLOAT widget that happens to sit at
    5 reads back as an int. Coercing blindly to type(current) would turn a
    requested duration of 7.5 into 7 - a quietly wrong video rather than an
    error. Narrow to int only when the value really is whole; otherwise send a
    float and let ComfyUI's validation reject it loudly if the widget is INT.
    """
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    return int(value) if float(value).is_integer() else float(value)


def inject(template: dict, job: dict, comfy_image: str | None) -> tuple[dict, dict]:
    """Apply a job to a copy of the template. Returns (graph, applied values)."""
    graph = json.loads(json.dumps(template))  # deep copy; the template is reused
    applied: dict = {}

    values = {k: v for k, v in job.items() if k in BINDINGS and v is not None}
    if comfy_image:
        values["image"] = comfy_image
    values.setdefault("output_prefix", f"video/{job['id']}")

    # A missing seed is drawn here and recorded in the result, so a run can be
    # reproduced later. Leaving it to the template would reuse one fixed value
    # and, worse, hit ComfyUI's node cache: a byte-identical graph executes
    # nothing at all and returns the previous video in zero seconds.
    if "seed" not in values:
        values["seed"] = random.getrandbits(48)

    for field, value in values.items():
        key = BINDINGS[field]["input"]
        nid, node = find_node(graph, BINDINGS[field])
        node["inputs"][key] = coerce(node["inputs"][key], value)
        applied[field] = node["inputs"][key]

    return graph, applied


def stage_image(ref: str, workdir: Path) -> str:
    """Get the job's image into ComfyUI, whatever form the reference takes.

    SEAM: an object-store key becomes one more branch here. The presigned-URL
    case is already the http branch, so the queue version mostly stops at
    generating that URL rather than teaching the worker about buckets.
    """
    if ref.startswith(("http://", "https://")):
        workdir.mkdir(parents=True, exist_ok=True)
        local = workdir / (Path(urllib.parse.urlparse(ref).path).name or "input.png")
        log(f"  fetching {ref}")
        with urllib.request.urlopen(ref, timeout=300) as r:
            local.write_bytes(r.read())
    else:
        local = Path(ref)
        if not local.is_file():
            raise FileNotFoundError(f"image not found: {ref}")

    name = upload_image(local)
    log(f"  uploaded as {name}")
    return name


# ---------------------------------------------------------------------------
# Job source.
#
# SEAM: FileSource stands in for the queue. `claim` corresponds to receiving a
# message, `done` to deleting it, `fail` to letting the visibility timeout
# expire into a dead-letter queue. Moving the file between directories is what
# makes a crash mid-generation visible afterwards instead of silently
# re-running - the same property the queue gives you, in twenty lines.
# ---------------------------------------------------------------------------
class FileSource:
    def __init__(self, root: Path):
        self.inbox = root / "inbox"
        self.running = root / "running"
        self.done_dir = root / "done"
        self.failed = root / "failed"
        for d in (self.inbox, self.running, self.done_dir, self.failed):
            d.mkdir(parents=True, exist_ok=True)

    def claim(self) -> tuple[dict, Path] | None:
        for path in sorted(self.inbox.glob("*.json")):
            held = self.running / path.name
            try:
                path.rename(held)      # atomic: two workers cannot both win it
            except OSError:
                continue
            job = json.loads(held.read_text(encoding="utf-8"))
            job.setdefault("id", held.stem)
            return job, held
        return None

    def done(self, held: Path) -> None:
        held.replace(self.done_dir / held.name)

    def fail(self, held: Path, err: Exception) -> None:
        held.replace(self.failed / held.name)
        (self.failed / f"{held.stem}.error.txt").write_text(
            f"{type(err).__name__}: {err}\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
def process(job: dict, template: dict, outbox: Path, dry_run: bool) -> dict:
    job_id = job["id"]
    log(f"[JOB] {job_id}")
    out_dir = outbox / job_id

    comfy_image = None
    if job.get("image"):
        if dry_run:
            # Do not upload on a dry run - the point is to inspect the graph,
            # which must stay possible with no ComfyUI listening at all.
            comfy_image = Path(urllib.parse.urlparse(job["image"]).path).name
            log(f"  (dry-run) image would upload as {comfy_image}")
        else:
            comfy_image = stage_image(job["image"], out_dir / "_in")

    graph, applied = inject(template, job, comfy_image)

    if dry_run:
        for field, value in applied.items():
            shown = value if not isinstance(value, str) or len(value) < 60 else value[:57] + "..."
            log(f"  {field:14s} = {shown}")
        (out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "graph.dry-run.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"  graph written to {out_dir / 'graph.dry-run.json'} (not queued)")
        return {"id": job_id, "dry_run": True, "applied": applied}

    elapsed, items = submit_and_wait(graph)
    log(f"  done in {elapsed:.2f}s, {len(items)} file(s)")

    # SEAM: an object-store upload replaces this loop. The manifest below is
    # what the orchestrator reads back, so keep its shape when the destination
    # changes - only the values under "path" become URIs.
    saved = []
    for item in items:
        dest = fetch_output(item, out_dir / item["filename"])
        size = dest.stat().st_size
        saved.append({"kind": item["kind"], "path": str(dest), "bytes": size})
        log(f"  saved {dest.name} ({size / 1e6:.1f} MB)")

    result = {
        "id": job_id,
        "status": "ok",
        "seconds": round(elapsed, 2),
        "applied": applied,
        "outputs": saved,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def wait_for_comfy(timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            api("/system_stats", timeout=5)
            return
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(3)
    raise RuntimeError(f"ComfyUI at {COMFY_URL} never answered")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run ComfyUI generation jobs from a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="workflows/minimax_h3_i2v_api.json",
                    help="workflow exported with Export (API), not the plain save")
    ap.add_argument("--root", default="jobs",
                    help="inbox/running/done/failed live under here")
    ap.add_argument("--outbox", default="jobs/out")
    ap.add_argument("--job", help="run this one file and exit, ignoring the inbox")
    ap.add_argument("--once", action="store_true", help="drain the inbox, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the injected graph instead of queueing it")
    args = ap.parse_args()

    template_path = Path(args.template)
    if not template_path.is_file():
        log(f"[ERROR] Template not found: {template_path}")
        log("        Export it from ComfyUI with Export (API), not the plain save.")
        return 1
    template = json.loads(template_path.read_text(encoding="utf-8"))

    # Resolve every binding against the template at startup, so a graph that
    # has drifted fails here rather than on the first job at 3am. A binding no
    # job ever sets is harmless, so report and continue instead of aborting.
    log(f"Template: {template_path} ({len(template)} nodes)")
    for field, binding in BINDINGS.items():
        try:
            nid, _ = find_node(template, binding)
            log(f"[OK]   {field:14s} -> node {nid}")
        except LookupError as e:
            log(f"[WARN] {field:14s} unbindable: {e}")

    if not args.dry_run:
        wait_for_comfy()

    outbox = Path(args.outbox)

    if args.job:
        job_path = Path(args.job)
        if not job_path.is_file():
            log(f"[ERROR] Job file not found: {job_path}")
            log('        Minimal job: {"id": "t1", "prompt": "..."}')
            log("        Every other field falls back to the template.")
            return 1
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log(f"[ERROR] {job_path} is not valid JSON: {e}")
            return 1
        job.setdefault("id", job_path.stem)
        try:
            process(job, template, outbox, args.dry_run)
        except Exception as e:  # noqa: BLE001 - single job: report, exit non-zero
            log(f"[ERROR] {job['id']}: {type(e).__name__}: {e}")
            return 1
        return 0

    source = FileSource(Path(args.root))
    log(f"Watching {source.inbox} (Ctrl-C to stop)")

    while True:
        claimed = source.claim()
        if claimed is None:
            if args.once:
                log("Inbox empty - done.")
                return 0
            time.sleep(POLL_SECONDS)
            continue

        job, held = claimed
        try:
            process(job, template, outbox, args.dry_run)
            source.done(held)
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the loop
            log(f"[ERROR] {job['id']}: {type(e).__name__}: {e}")
            source.fail(held, e)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
