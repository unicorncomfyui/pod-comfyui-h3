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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

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

IMAGE = os.environ.get("E2E_IMAGE", "vlop12ui/pod-comfyui-h3:cu130-develop")


def log(msg: str = "") -> None:
    print(msg, flush=True)


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
        log("(nothing matched - loosen --min-memory or --min-cuda)")
        return 0

    for g in rows:
        price = g.get("price") or {}
        secure = price.get("secure")
        community = price.get("community")
        log(f"{str(g.get('id','')):<34}{g.get('memory') or 0:>4}G"
            f"   secure {secure if secure is not None else '-':>6}"
            f"   community {community if community is not None else '-':>6}"
            f"   {g.get('availability') or '?'}")
        # Per-data-centre stock, and the reason to ask for AVAILABILITY at all:
        # choosing a zone is the decision this output exists to inform, and the
        # overall figure hides it. NONE entries are dropped - a data centre
        # with no stock is not a candidate.
        live = [d for d in (g.get("dataCenters") or [])
                if d.get("availability") != "NONE"]
        if live:
            log("      " + "   ".join(
                f"{d.get('id')} {d.get('availability')}" for d in live))
        else:
            log("      (no data centre reporting stock right now)")

    log(f"\n{len(rows)} type(s). Availability is this moment, not a "
        f"reservation - HIGH now can be NONE in an hour.")
    return 0


def build_request(args) -> dict:
    body: dict = {
        "name": args.name,
        "image": args.image,
        "gpu": {"id": args.gpu, "count": args.count},
        "disk": args.disk,
        # A comparison, NOT the exact-match allowedCudaVersions. The spec is
        # explicit that naming a version no machine reports yields a capacity
        # error rather than a fallback, and this image only needs a floor:
        # its own NVIDIA_REQUIRE_CUDA says cuda>=13.0.
        "minCudaVersion": args.min_cuda,
        "ports": ["3000/http", "8080/http"],
        "env": {},
    }
    if args.datacenter:
        body["dataCenterIds"] = args.datacenter
    if args.volume:
        # Mount kind is fixed at create time and the volume must live in the
        # same data centre as the pod - which is why --datacenter and --volume
        # travel together or not at all.
        body["mounts"] = {"network": [{"volumeId": args.volume,
                                       "path": "/workspace"}]}
    elif args.workspace:
        # Host-local persistent storage: RunPod's "volume disk". Faster than a
        # network volume and it needs no data centre pin, so the scheduler
        # keeps the whole fleet to choose from - which matters when healthy
        # hosts are the scarce thing. The trade is that it is pinned to one
        # machine and dies with it, so the next pod re-downloads everything.
        # Right for a smoke test, wrong for anything you cannot recreate.
        body["mounts"] = {"persistent": {"size": args.workspace,
                                         "path": "/workspace"}}
    for pair in args.env:
        k, _, v = pair.partition("=")
        body["env"][k] = v
    return body


def cmd_up(args) -> int:
    body = build_request(args)
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
            return 0
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


def cmd_logs(args) -> int:
    out = api("GET", f"/pods/{args.pod_id}/logs")
    if isinstance(out, dict):
        out = out.get("logs") or out.get("data") or out
    log(out if isinstance(out, str) else json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="GPU types, prices and availability")
    c.add_argument("--min-memory", type=int, default=0, help="VRAM floor in GB")
    c.add_argument("--min-cuda", default="13.0")
    c.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"],
                   help="which cloud the availability figures describe. "
                        "SECURE matches what `up` creates by default")
    c.add_argument("--name", default="", help="substring filter on the id")
    c.set_defaults(func=cmd_catalog)

    u = sub.add_parser("up", help="create a pod and wait for it to run")
    u.add_argument("--gpu", required=True, help="GPU type id, from `catalog`")
    u.add_argument("--count", type=int, default=1)
    u.add_argument("--image", default=IMAGE)
    u.add_argument("--name", default=f"e2e-{int(time.time())}")
    u.add_argument("--disk", type=int, default=30, help="container disk, GB")
    u.add_argument("--min-cuda", default="13.0")
    u.add_argument("--datacenter", action="append", default=[],
                   help="pin placement; repeatable. Required with --volume")
    u.add_argument("--volume", help="network volume id, same data centre")
    u.add_argument("--workspace", type=int, metavar="GB",
                   help="host-local persistent disk at /workspace, in GB. "
                        "No data centre pin needed, but it dies with the "
                        "host. Ignored when --volume is given")
    u.add_argument("--env", action="append", default=[], help="KEY=value")
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

    g = sub.add_parser("logs", help="container logs, without a shell")
    g.add_argument("pod_id")
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
