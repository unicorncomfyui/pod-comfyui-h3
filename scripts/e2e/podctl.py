#!/usr/bin/env python3
"""Create, inspect and destroy pods from the command line.

The first piece of the end-to-end runner, and deliberately the smallest one:
before anything can be automated, the three calls it rests on have to be known
to work - read the catalogue, start a pod, stop it again.

WHY THIS IS NOT IN THE IMAGE
This drives RunPod from outside. It has no business inside a container that
RunPod is running, and it never enters one: the Dockerfile copies named script
files, not scripts/ as a directory.

WHY THE STANDARD LIBRARY ONLY
It has to run on a laptop, in a CI job and in a pod shell without anyone being
asked to install something first. urllib is enough for six HTTP calls.

WHY v2
REST v1 and the GraphQL API are both announced as heading for deprecation. v2
is in public beta and its own documentation says the endpoints may still move,
so the spec version is checked on every run rather than discovered by a
confusing 404 six months from now.

Usage:
    export RUNPOD_API_KEY=...

    python scripts/e2e/podctl.py catalog --min-memory 32
    python scripts/e2e/podctl.py up --gpu "NVIDIA GeForce RTX 5090" --dry-run
    python scripts/e2e/podctl.py up --gpu "NVIDIA GeForce RTX 5090"
    python scripts/e2e/podctl.py ls
    python scripts/e2e/podctl.py logs <pod-id>
    python scripts/e2e/podctl.py down <pod-id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.io/v2")

# The beta is expected to move within 2.x and not outside it. A mismatch is a
# warning rather than a refusal: being unable to destroy a running pod because
# a minor version moved would be worse than the drift it protects against.
SPEC_MAJOR = "2"

# urllib announces itself as "Python-urllib/3.x", and the edge in front of the
# API rejects that signature with a Cloudflare 403 before RunPod ever sees the
# request. It looks exactly like a permissions failure and it is not: the same
# call succeeds with any other agent. Naming the tool also means a support
# request can be traced to it.
USER_AGENT = "podctl/1.0 (+https://github.com/unicorncomfyui/pod-comfyui-h3)"

# DataCenterRegion, verbatim from the spec, plus ALL to opt out of filtering.
REGIONS = ["ALL", "EUROPE", "NORTH_AMERICA", "SOUTH_AMERICA", "ASIA",
           "MIDDLE_EAST", "AFRICA", "OCEANIA", "ANTARCTICA", "UNKNOWN"]

IMAGE = os.environ.get("E2E_IMAGE", "vlop12ui/pod-comfyui-h3:cu130-develop")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def note(msg: str = "") -> None:
    """Commentary, on stderr.

    --dry-run exists to be piped somewhere - jq, a diff, a request builder -
    and a status line interleaved with the JSON breaks that. Anything that is
    not the payload goes to the other stream.
    """
    print(msg, file=sys.stderr, flush=True)


def redact(obj):
    """Blank every env value before a pod object is ever printed.

    The API echoes back the environment it was given, so anything passed to a
    pod comes home in the response - and on a public repository the CI log that
    prints it is world-readable. HF_TOKEN is the concrete case: it is a
    documented variable of this image, it would be handed to the pod as env,
    and it would then appear in full in a log nobody thought of as an output.

    GitHub masks the exact string of a registered secret, so a value that
    arrives back verbatim would be caught - but only if it was registered as a
    secret, only exactly, and not once JSON escaping has touched it. Not
    printing it at all is the version that does not depend on any of that.
    """
    if isinstance(obj, dict):
        return {k: ("***" if k == "env" and isinstance(v, dict)
                    else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


def api(method: str, path: str, body: dict | None = None,
        query: dict | None = None) -> dict | list:
    url = f"{API}{path}"
    if query:
        # Comma-separated arrays, per the spec's style: form / explode false.
        flat = {k: (",".join(v) if isinstance(v, list) else v)
                for k, v in query.items() if v is not None}
        url += "?" + urllib.parse.urlencode(flat)

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("[ERROR] RUNPOD_API_KEY is not set.")

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        log(f"[ERROR] {method} {path} -> {e.code}")
        # A 403 carrying Cloudflare's 1010 never reached RunPod, so it says
        # nothing about the key. Left unexplained it sends you back to the
        # permissions screen, which is the one place the answer is not.
        if e.code == 403 and "1010" in detail:
            log("        Blocked by the edge on the client signature, not by "
                "RunPod.")
            log("        This is not a permissions problem - the request never "
                "arrived.")
        else:
            try:
                log("        " + json.dumps(json.loads(detail))[:500])
            except json.JSONDecodeError:
                log("        " + detail[:500])
        raise


def check_spec() -> None:
    """Read the served spec version once, so drift is announced not guessed."""
    try:
        req = urllib.request.Request(f"{API}/openapi.json",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            version = str(json.loads(r.read()).get("info", {}).get("version", ""))
    except Exception:  # noqa: BLE001 - never let a check block a teardown
        return
    if version and not version.startswith(SPEC_MAJOR + "."):
        log(f"[WARN] The API now serves spec {version}; this script was written "
            f"against {SPEC_MAJOR}.x. Check the release notes before trusting "
            f"anything below.")


# ---------------------------------------------------------------------------
def datacenter_index() -> dict:
    """id -> data centre record.

    Fetched separately because the GPU catalogue carries only an id and a stock
    level per data centre. Everything needed to CHOOSE one lives here: its
    continental region, and which network volume tiers it offers - the
    high-performance tier is not everywhere, and a zone without it cannot host
    the fast volume whatever its GPU stock says.

    The GPU endpoint does have a countryCodes filter, but it takes ISO country
    codes; asking for "Europe" through it would mean hard-coding a list of
    countries that goes stale the day Runpod opens a zone in one more. The
    region enum is theirs to maintain, so it is theirs that gets used.
    """
    body = api("GET", "/catalog/datacenters")
    if isinstance(body, dict) and "dataCenters" in body:
        rows = body["dataCenters"]
    elif isinstance(body, list):
        rows = body
    else:
        log("[WARN] Could not read the data centre catalogue; region filtering "
            "and volume tiers are unavailable this run.")
        return {}
    return {d.get("id"): d for d in rows if d.get("id")}


def cmd_datacenters(args) -> int:
    index = datacenter_index()
    rows = [d for d in index.values()
            if args.region == "ALL" or d.get("region") == args.region]
    rows.sort(key=lambda d: str(d.get("id")))
    log(f"{'id':<12}{'region':<16}{'volumes':<28}{'global net':<12}name")
    for d in rows:
        tiers = ",".join(d.get("networkVolumeTypes") or []) or "-"
        log(f"{str(d.get('id','')):<12}{str(d.get('region','')):<16}"
            f"{tiers:<28}{'yes' if d.get('globalNetwork') else 'no':<12}"
            f"{d.get('name','')}")
    log(f"\n{len(rows)} data centre(s). HIGH_PERFORMANCE is the tier that "
        f"claims 3x throughput; a zone without it cannot host one.")
    return 0


def cmd_catalog(args) -> int:
    # Per the spec: `product` is an array and is REQUIRED with
    # include=AVAILABILITY, `cloud` scopes the availability and lowest-price
    # figures, and both are valid only alongside that include. `cloud` defaults
    # to SECURE here to match what `up` actually creates - availability
    # measured across a cloud you will not deploy into describes a pod you
    # cannot get.
    body = api("GET", "/catalog/gpus", query={
        "include": ["AVAILABILITY"],
        "product": ["POD"],
        "cloud": args.cloud,
        "minCudaVersion": args.min_cuda,
    })

    # Named key, and a loud failure if it is not there. Guessing at a list of
    # plausible key names is how this printed "nothing matched" for a response
    # that was full of GPUs: an empty result and a shape mismatch must not look
    # the same, because one means "look elsewhere" and the other means
    # "the reader is wrong".
    if isinstance(body, dict) and "gpus" in body:
        gpus = body["gpus"]
    elif isinstance(body, list):
        gpus = body
    else:
        log("[ERROR] Unexpected response shape - no 'gpus' key.")
        log(f"        keys: {sorted(body) if isinstance(body, dict) else type(body).__name__}")
        log("        The spec may have moved; check /v2/openapi.json.")
        return 1

    rows = [g for g in gpus
            if (g.get("memory") or 0) >= args.min_memory
            and (args.name.lower() in str(g.get("id", "")).lower()
                 if args.name else True)]
    rows.sort(key=lambda g: (g.get("price") or {}).get("secure") or 1e9)

    if not rows:
        log(f"(no GPU matched name~'{args.name}' with >={args.min_memory}G "
            f"and CUDA >={args.min_cuda})")
        return 0

    # Always, not only when filtering: the volume tiers are wanted either way.
    index = datacenter_index()

    for g in rows:
        price = g.get("price") or {}
        rate = price.get("secure" if args.cloud == "SECURE" else "community")
        log(f"{str(g.get('id','')):<34}{g.get('memory') or 0:>4}G"
            f"   {args.cloud.lower()} {rate if rate is not None else '-'} $/h"
            f"   overall {g.get('availability') or '?'}")

        # Per-data-centre stock, which is the reason to ask for AVAILABILITY at
        # all: choosing a zone is the decision this output informs, and the
        # overall figure hides it. NONE is dropped - a zone with no stock is
        # not a candidate. The volume tiers are pulled in alongside because the
        # zone has to satisfy both constraints at once, and discovering the
        # second one after committing to the first is how a volume ends up in
        # the wrong place. A volume cannot be moved between zones.
        live = []
        for d in (g.get("dataCenters") or []):
            if d.get("availability") == "NONE":
                continue
            if args.region != "ALL":
                meta = index.get(d.get("id")) or {}
                if meta.get("region") != args.region:
                    continue
            live.append(d)

        if not live:
            log(f"      (no {args.region.lower().replace('_', ' ')} data centre "
                f"reporting stock right now)")
            continue
        for d in live:
            meta = index.get(d.get("id"))
            # "none" and "?" are different answers. networkVolumeTypes is a
            # required field of a data centre record, so an empty list means
            # this zone offers no network volumes at all - which disqualifies
            # it for anything that has to persist. Only a zone missing from
            # the catalogue is genuinely unknown.
            if meta is None:
                tiers = "?"
            else:
                tiers = ",".join(meta.get("networkVolumeTypes") or []) or "none"
            log(f"      {str(d.get('id','')):<10} {str(d.get('availability')):<7}"
                f" volumes: {tiers}")

    log(f"\n{len(rows)} type(s), {args.cloud} cloud, region {args.region}. "
        f"Availability is this moment, not a reservation.")
    return 0


def model_set_size(env: dict) -> tuple[float, str] | None:
    """GB the requested model sets will occupy, from the repository manifest.

    Read here rather than trusted to the pod, because the pod's own pre-flight
    check runs after it has been created and started billing, and reports into
    a log somebody then has to go and read. The manifest is right beside this
    script; asking it costs nothing and turns "the disk filled up" into a
    refusal that names the number.

    Returns None whenever the answer would be a guess - an unreadable
    manifest, an unknown set name. A wrong figure here would be worse than
    none: it would refuse pods that are fine.
    """
    if str(env.get("DOWNLOAD_MODELS", "true")).lower() == "false":
        return 0.0, "downloads disabled"

    manifest = Path(__file__).resolve().parent.parent.parent / "models" / "manifest.json"
    try:
        sets = json.loads(manifest.read_text(encoding="utf-8"))["sets"]
    except Exception:  # noqa: BLE001 - never block a deploy on bookkeeping
        return None

    names = env.get("MODEL_SETS")
    if not names:
        # Fall back to what the image itself ships, so the check describes the
        # pod actually being created rather than a hypothetical one.
        try:
            dockerfile = manifest.parent.parent / "Dockerfile"
            for line in dockerfile.read_text(encoding="utf-8").splitlines():
                if "MODEL_SETS=" in line:
                    names = line.split("MODEL_SETS=", 1)[1].strip().strip('"\\')
                    break
        except Exception:  # noqa: BLE001
            return None
    if not names or names in ("default", "all", "none"):
        return None

    # Deduplicated by source path, because the sets share files: every H3 set
    # names the same text encoder and the same two VAEs, and they land in one
    # place on disk. Counting them per set would inflate the floor by 21 GB
    # and refuse pods that are perfectly sized.
    files, seen = {}, []
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        if name not in sets:
            return None
        seen.append(name)
        for f in sets[name].get("files", []):
            files[(f.get("repo"), f.get("path"))] = float(f.get("size_gb") or 0)
    return sum(files.values()), f"{len(seen)} set(s), {len(files)} file(s)"


def build_request(args) -> dict:
    body: dict = {
        "name": args.name,
        "gpu": {"id": args.gpu, "count": args.count},
        "env": {},
    }

    # A comparison, NOT the exact-match allowedCudaVersions. The spec is
    # explicit that naming a version no machine reports yields a capacity
    # error rather than a fallback, and this image only needs a floor: its own
    # NVIDIA_REQUIRE_CUDA says cuda>=13.0. Note that sending either CUDA field
    # replaces a template's constraint entirely - which is wanted, the floor
    # belongs to the image and not to whoever last edited the template.
    if args.min_cuda:
        body["minCudaVersion"] = args.min_cuda

    # With a template, silence is meaningful. The spec says explicit body
    # fields override the template's, so sending argparse defaults would
    # quietly replace the image, the disk, the ports and the mount that the
    # template exists to carry. Only what was actually typed is sent; the
    # built-in defaults apply solely when there is no template to defer to.
    # `env` is the exception the spec carves out: it merges per key with body
    # values winning, so passing --env adds to the template rather than
    # replacing it.
    if args.template:
        body["templateId"] = args.template
        if args.image:
            body["image"] = args.image
        if args.disk is not None:
            body["disk"] = args.disk
        if args.ports:
            body["ports"] = args.ports
    else:
        body["image"] = args.image or IMAGE
        body["disk"] = args.disk if args.disk is not None else 30
        body["ports"] = args.ports or ["3000/http", "8080/http"]

    if args.datacenter:
        body["dataCenterIds"] = args.datacenter
    if args.volume:
        # Mount kind is fixed at create time and the volume must live in the
        # same data centre as the pod - which is why --datacenter and --volume
        # travel together or not at all.
        body["mounts"] = {"network": [{"volumeId": args.volume,
                                       "path": "/workspace"}]}
    else:
        # Host-local persistent storage: RunPod's "volume disk". Faster than a
        # network volume and it needs no data centre pin, so the scheduler
        # keeps the whole fleet to choose from - which matters when healthy
        # hosts are the scarce thing. The trade is that it is pinned to one
        # machine and dies with it, so the next pod re-downloads everything.
        # Right for a smoke test, wrong for anything you cannot recreate.
        size = args.workspace
        if size is None and not args.template:
            size = 100
        if size is not None:
            body["mounts"] = {"persistent": {"size": size, "path": "/workspace"}}
    for pair in args.env:
        k, _, v = pair.partition("=")
        body["env"][k] = v
    return body


def cmd_up(args) -> int:
    """Draw machines until one satisfies --min-ram, or the attempts run out.

    Retrying is the only lever there is. RAM cannot be requested and cannot be
    read before boot, so the shape of the machine is discovered by taking one -
    and a rejected attempt costs a minute of a pod, not a run. Without the
    loop, --min-ram is a way to fail rather than a way to get what you asked
    for, and you end up typing `up` in a while loop by hand.
    """
    for attempt in range(1, args.attempts + 1):
        if args.attempts > 1:
            log(f"\n=== attempt {attempt}/{args.attempts} ===")
        rc = _up_once(args)
        if rc != "retry":
            return rc
        if attempt == args.attempts:
            log(f"\n[ERROR] {args.attempts} machine(s) drawn, none with "
                f"{args.min_ram} GB. That size may not exist in these data "
                f"centres - the console's RAM/GPU filter will tell you where "
                f"it does.")
            return 1
    return 1


def _up_once(args):
    body = build_request(args)

    # Disk pre-flight, before the pod exists. The image downloads its weights
    # onto /workspace at first boot, and without a mount that is the 30 GB
    # container disk - which the default model set overruns by more than
    # twice. The failure that follows reads like a broken download rather than
    # a pod that was never given room.
    need = model_set_size(body.get("env") or {})
    have = (body.get("mounts") or {}).get("persistent", {}).get("size")
    if args.template and not {"MODEL_SETS", "DOWNLOAD_MODELS"} & set(body["env"]):
        # The template carries its own env and its own mount, and neither is
        # visible from here - the API resolves them at create time. Guessing
        # would produce a confident wrong number, so the check says it is
        # standing down rather than implying the sizing was verified.
        note("Models     not checked - the template supplies MODEL_SETS and "
             "the mount, and neither is readable from here.")
    elif need is not None and have is not None:
        weights, how = need
        # Headroom for the Triton cache, outputs and the ComfyUI user
        # directory - all of which live on the same mount.
        floor = weights + 20
        note(f"Models     {weights:.2f} GB ({how}); /workspace {have} GB")
        if have < floor:
            note(f"[ERROR] {have} GB will not hold {weights:.2f} GB of weights "
                 f"plus room to work.")
            note(f"        Use --workspace {int(floor + 0.5)} or more, or pass "
                 f"--env DOWNLOAD_MODELS=false for a smoke test, or narrow "
                 f"--env MODEL_SETS=...")
            return 2
    if args.dry_run:
        # Redacted too. A dry run is the output most likely to be pasted into
        # an issue or a chat to ask "does this look right", and the shape is
        # what that question is about - the keys are still visible, only the
        # values are not.
        log(json.dumps(redact(body), indent=2))
        return 0

    pod = api("POST", "/pods", body=body)
    pod_id = pod.get("id")
    if not pod_id:
        log("[ERROR] The API accepted the request but returned no pod id.")
        log("        " + json.dumps(redact(pod))[:400])
        return 1

    # Printed before the wait, and on its own line, because everything after
    # this point can fail while the pod keeps billing. This id is how it gets
    # stopped, so it must survive a scrollback nobody read to the end.
    log(f"\n  pod {pod_id}")
    log(f"  stop it with:  python {sys.argv[0]} down {pod_id}\n")

    deadline = time.time() + args.timeout
    last = ""
    while time.time() < deadline:
        state = api("GET", f"/pods/{pod_id}")
        status = str(state.get("status") or "")
        if status != last:
            log(f"  {status}")
            last = status
        if status == "RUNNING":
            log(f"\n[OK] Running after {int(time.time() - (deadline - args.timeout))} s.")
            describe(state)
            if not args.min_ram:
                return 0
            # The one check that cannot be a filter. Asked for after the fact,
            # so a machine that is too small costs a boot rather than a run -
            # which is the cheap end of a mistake that would otherwise surface
            # as an offloader thrashing mid-generation.
            gib = wait_for_ram(pod_id)
            ram = gib_to_gb(gib) if gib is not None else None
            if ram is None:
                log("[WARN] Could not read the pod's RAM from its log within "
                    "3 min. Left running - check it yourself.")
                return 0
            if ram >= args.min_ram:
                log(f"[OK] {ram} GB of RAM ({gib} GiB as the cgroup reports "
                    f"it), at or above the {args.min_ram} GB asked for.")
                return 0
            log(f"[ERROR] {ram} GB of RAM ({gib} GiB in the log), below the "
                f"{args.min_ram} GB asked for. The API cannot request memory, "
                f"so this is the only place it can be caught.")
            if args.keep:
                log("       Left running as asked.")
                return 1
            try:
                api("DELETE", f"/pods/{pod_id}")
                log("[OK] Terminated.")
            except urllib.error.HTTPError:
                log(f"[WARN] Could not terminate {pod_id}. STOP IT BY HAND.")
            return "retry"
        if status in ("ERROR", "EXITED", "TERMINATED"):
            log(f"\n[ERROR] Pod reached {status} without running.")
            break
        time.sleep(5)
    else:
        log(f"\n[ERROR] Still {last or 'unknown'} after {args.timeout} s.")

    # A pod that never became useful still costs money. Tear it down unless
    # asked not to - the one case for keeping it is reading the logs of a
    # container that failed to start, and that is what --keep is for.
    if args.keep:
        log(f"       Left running as asked. logs: python {sys.argv[0]} "
            f"logs {pod_id}")
        return 1
    log("       Terminating it; pass --keep to inspect it instead.")
    try:
        api("DELETE", f"/pods/{pod_id}")
        log("[OK] Terminated.")
    except urllib.error.HTTPError:
        log(f"[WARN] Could not terminate {pod_id}. STOP IT BY HAND.")
    return 1


def cmd_down(args) -> int:
    api("DELETE", f"/pods/{args.pod_id}")
    log(f"[OK] {args.pod_id} terminated.")
    return 0


def cmd_ls(args) -> int:
    pods = api("GET", "/pods")
    if isinstance(pods, dict):
        pods = pods.get("pods") or pods.get("data") or []
    if not pods:
        log("No pods. Nothing is billing.")
        return 0
    log(f"{'id':<22}{'status':<14}{'gpu':<30}name")
    for p in pods:
        gpu = (p.get("gpu") or {}).get("id") or ""
        log(f"{str(p.get('id','')):<22}{str(p.get('status','')):<14}"
            f"{str(gpu)[:29]:<30}{p.get('name','')}")
    log(f"\n{len(pods)} pod(s). Anything RUNNING here is being charged for.")
    return 0


def describe(pod: dict) -> None:
    """Say what was actually created, not what was asked for.

    A template is resolved server-side, so the request body says nothing about
    the disk or the mount the pod ends up with - and a template that carries no
    persistent mount produces a pod with only its container disk, silently. The
    first sign of that is otherwise a download failing at 10 GB, an hour later,
    with nothing on screen having suggested it.
    """
    mounts = pod.get("mounts") or {}
    persistent = (mounts.get("persistent") or {}).get("size")
    network = [m.get("volumeId") for m in (mounts.get("network") or [])]
    disk = pod.get("disk")

    log(f"  data centre  {pod.get('dataCenterId') or '?'}"
        f"   cuda {pod.get('cudaVersion') or '?'}"
        f"   {pod.get('cost') or 0} $/h")
    if network:
        log(f"  storage      network volume {', '.join(str(v) for v in network)}"
            f"   + {disk} GB container disk")
    elif persistent:
        log(f"  storage      {persistent} GB at /workspace"
            f"   + {disk} GB container disk")
    else:
        log(f"  storage      NO PERSISTENT MOUNT - {disk} GB container disk only")
        # 40 GB is below the smallest useful model set, so at that point the
        # download cannot succeed whatever else is configured.
        if isinstance(disk, int) and disk < 40:
            log("")
            log("[WARN] Nothing is mounted at /workspace and the container disk")
            log(f"       is {disk} GB. If this pod downloads models it will run")
            log("       out of room. A template only supplies a mount if one was")
            log("       saved into it - pass --workspace GB to attach one.")


def stream_logs(pod_id: str, tail: int = 500, idle: int = 20,
                deadline: float | None = None):
    """Yield log lines from the pod's SSE stream.

    This endpoint answers text/event-stream, not JSON: it backfills `tail`
    lines and then stays open forever, streaming. Reading it like a document -
    one urlopen().read() - blocks until the pod dies, which is why the earlier
    version appeared to hang right after reporting the pod as running, and left
    the pod billing while it waited.

    So it is consumed as a stream, and stopped by silence: iterating raises a
    timeout once `idle` seconds pass with nothing arriving, which for a
    backfill means the history has been delivered. `deadline` bounds the whole
    thing for callers that are waiting for one specific line to appear.
    """
    url = f"{API}/pods/{pod_id}/logs?tail={tail}"
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("[ERROR] RUNPOD_API_KEY is not set.")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream",
        "User-Agent": USER_AGENT,
    })
    try:
        with urllib.request.urlopen(req, timeout=idle) as r:
            for raw in r:
                if deadline and time.time() > deadline:
                    return
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue  # event:, id:, retry: and the blank separators
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                text = event.get("line")
                if isinstance(text, str):
                    yield text
    except (TimeoutError, urllib.error.URLError, OSError):
        # Silence on the socket is the normal end of a backfill, not a fault.
        return


# init.sh prints exactly one of these two lines, and prefers the cgroup because
# /proc/meminfo inside a container reports the HOST's memory: a 92 GB pod on a
# big machine announces 386 GB. The cgroup limit is the allocation.
RAM_LINE = re.compile(
    r"(?:memory limit read from cgroup|reporting host RAM):\s*(\d+)\s*GB")


def gib_to_gb(gib: int) -> int:
    """Restate a GiB figure in the decimal GB the machine was sold as.

    init.sh divides the cgroup limit by 1024 three times and labels the result
    "GB", so a pod Runpod advertises as 92 GB reports 86. Comparing a threshold
    the operator typed - who is thinking of the console and the invoice -
    against that number rejects machines that are exactly the right size, which
    is what four wasted attempts in a row looked like.

    bench.py already gets this right and says so in a comment; this is the same
    correction applied where the decision is made.
    """
    return round(gib * 1024 ** 3 / 1e9)


def wait_for_ram(pod_id: str, timeout: int = 180) -> int | None:
    """Host RAM in GB, read out of the pod's own boot log.

    There is no other way to get it. REST v2 has no memory or vCPU field on
    CreatePodRequest for GPU pods and none on the Pod object either - RAM comes
    bundled with whatever machine the scheduler picked, and the API neither
    accepts a floor nor reports the result. The pod knows, because init.sh was
    written to find out; asking it after the fact is the only route.
    """
    deadline = time.time() + timeout
    log(f"  reading the pod's log for its memory limit "
        f"(up to {timeout // 60} min)...")
    for line in stream_logs(pod_id, tail=1000, idle=30, deadline=deadline):
        m = RAM_LINE.search(line)
        if m:
            return int(m.group(1))
    return None


def cmd_templates(args) -> int:
    body = api("GET", "/catalog/templates" if args.public else "/templates")
    rows = body.get("templates") if isinstance(body, dict) else body
    if not rows:
        log("No templates." + ("" if args.public else
            " Save one from the console, or pass --public for the catalogue."))
        return 0
    log(f"{'id':<26}{'name':<34}image")
    for t in rows:
        log(f"{str(t.get('id','')):<26}{str(t.get('name',''))[:33]:<34}"
            f"{t.get('image') or t.get('imageName') or ''}")
    log(f"\n{len(rows)} template(s). A template is read ONCE at create time - "
        f"the pod keeps no link to it, so later edits change nothing.")
    return 0


def cmd_logs(args) -> int:
    n = 0
    deadline = None if args.follow else time.time() + args.timeout
    for line in stream_logs(args.pod_id, tail=args.tail,
                            idle=args.timeout if args.follow else 15,
                            deadline=deadline):
        log(line)
        n += 1
    if n == 0:
        log("(nothing yet - the container may not have started writing)")
    elif not args.follow:
        log(f"\n-- {n} line(s). The stream stays open; --follow to keep reading.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # The defaults are this project's target, not neutral ones: an RTX 5090 in
    # Europe on the secure cloud. Widen with --name "" --region ALL when the
    # question is what else exists.
    c = sub.add_parser("catalog", help="GPU types, prices and availability")
    c.add_argument("--min-memory", type=int, default=32, help="VRAM floor in GB")
    c.add_argument("--min-cuda", default="13.0")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"],
                   help="which cloud the availability figures describe. "
                        "SECURE matches what `up` creates by default")
    c.add_argument("--region", default="EUROPE", choices=REGIONS,
                   help="continental region, or ALL")
    c.add_argument("--name", default="5090",
                   help="substring filter on the GPU id; pass '' for every type")
    c.set_defaults(func=cmd_catalog)

    dc = sub.add_parser("datacenters",
                        help="zones, their volume tiers and global networking")
    dc.add_argument("--region", default="EUROPE", choices=REGIONS)
    dc.set_defaults(func=cmd_datacenters)

    u = sub.add_parser("up", help="create a pod and wait for it to run")
    u.add_argument("--gpu", required=True, help="GPU type id, from `catalog`")
    u.add_argument("--count", type=int, default=1)
    u.add_argument("--template", metavar="ID",
                   help="base the pod on a pod template (yours or a public "
                        "catalog one). It supplies image, disk, ports, env "
                        "and the mount; anything given here overrides it, "
                        "except --env which merges. List them with "
                        "`podctl templates`")
    u.add_argument("--image", default=None,
                   help=f"defaults to {IMAGE} when no --template is given")
    u.add_argument("--name", default=f"e2e-{int(time.time())}")
    u.add_argument("--disk", type=int, default=None,
                   help="container disk in GB (default 30 without a template)")
    u.add_argument("--ports", action="append", default=[],
                   metavar="PORT/PROTO", help="repeatable, e.g. 3000/http")
    u.add_argument("--min-cuda", default="13.0")
    u.add_argument("--datacenter", action="append", default=[],
                   help="pin placement; repeatable. Required with --volume")
    u.add_argument("--volume", help="network volume id, same data centre")
    # 100 GB, which is what the README has always told operators to give this
    # image: the default model set is 68.27 GB and the rest is outputs, the
    # Triton cache and the user directory. Not a round number picked for
    # looks - the pre-flight above computes the real floor from the manifest.
    u.add_argument("--workspace", type=int, default=None, metavar="GB",
                   help="host-local persistent disk at /workspace, in GB "
                        "(default 100 without a template; left to the "
                        "template otherwise). No data centre pin needed, but "
                        "it dies with the host. Ignored when --volume is given")
    u.add_argument("--env", action="append", default=[], help="KEY=value")
    u.add_argument("--min-ram", type=int, default=0, metavar="GB",
                   help="reject the machine if it has less host RAM than this, in the decimal GB Runpod advertises - 92 means the 92 GB offer, not the 86 GiB its cgroup reports. Checked AFTER boot from the pod's own log: the API has no memory filter for GPU pods and does not report it either")
    u.add_argument("--attempts", type=int, default=1, metavar="N",
                   help="draw up to N machines until one meets --min-ram, terminating each that does not. Only useful with --min-ram")
    u.add_argument("--timeout", type=int, default=600)
    u.add_argument("--keep", action="store_true",
                   help="do not terminate a pod that failed to start")
    u.add_argument("--dry-run", action="store_true",
                   help="print the request body and send nothing")
    u.set_defaults(func=cmd_up)

    d = sub.add_parser("down", help="terminate a pod")
    d.add_argument("pod_id")
    d.set_defaults(func=cmd_down)

    sub.add_parser("ls", help="every pod on the account").set_defaults(func=cmd_ls)

    tp = sub.add_parser("templates", help="your pod templates, and their ids")
    tp.add_argument("--public", action="store_true",
                    help="the public catalogue instead of your own")
    tp.set_defaults(func=cmd_templates)

    g = sub.add_parser("logs", help="container logs, without a shell")
    g.add_argument("pod_id")
    g.add_argument("--tail", type=int, default=500,
                   help="historical lines to backfill (max 5000)")
    g.add_argument("--follow", action="store_true",
                   help="keep the stream open instead of stopping once the "
                        "backfill has been delivered")
    g.add_argument("--timeout", type=int, default=30,
                   help="seconds of silence before giving up")
    g.set_defaults(func=cmd_logs)

    args = p.parse_args()
    if args.cmd != "down":
        check_spec()
    if getattr(args, "volume", None) and not args.datacenter:
        log("[ERROR] --volume needs --datacenter: a network volume only "
            "attaches to a pod in its own data centre, and without the pin "
            "the scheduler will place the pod somewhere it cannot mount.")
        return 2
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError:
        # api() has already printed the status, the path and whatever the body
        # explained. A traceback on top of that adds twenty lines of urllib
        # internals and buries the one line that matters.
        sys.exit(1)
    except KeyboardInterrupt:
        log("\n[WARN] Interrupted. If a pod was created, check `ls` - it is "
            "still billing.")
        sys.exit(130)
