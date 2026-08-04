#!/usr/bin/env python3
"""Download the model sets declared in models/manifest.json.

Driven entirely by the manifest plus two environment variables, so adding a
model never requires touching init.sh or the Dockerfile:

    MODEL_SETS      comma-separated set names, or "default" / "none" / "all"
    COMFYUI_DIR     ComfyUI root; files land in <COMFYUI_DIR>/models/<dest>/

Downloads go through huggingface_hub, which gives resumable, chunk-deduplicated
transfers via Xet - it matters when a full H3 install is 63 GB. Files already
present are skipped, so re-running on an existing network volume is cheap.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

MANIFEST = Path(os.environ.get("MODEL_MANIFEST", "/app/models/manifest.json"))
# 15% headroom over the raw weight total: HF stages a partial file next to the
# final one, and the pod still needs room for outputs.
DISK_HEADROOM = 1.15


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_sets(manifest: dict, requested: str) -> list[str]:
    """Turn the MODEL_SETS value into a concrete, validated list of set names."""
    available = manifest["sets"]
    requested = requested.strip()

    if requested.lower() in ("none", ""):
        return []
    if requested.lower() == "all":
        return list(available)
    if requested.lower() == "default":
        return [name for name, s in available.items() if s.get("default")]

    names, unknown = [], []
    for raw in requested.split(","):
        name = raw.strip()
        if not name:
            continue
        if name in available:
            names.append(name)
        else:
            unknown.append(name)

    if unknown:
        log(f"[ERROR] Unknown model set(s): {', '.join(unknown)}")
        log(f"        Available: {', '.join(available)}")
        sys.exit(1)
    return names


def plan(manifest: dict, set_names: list[str], models_root: Path) -> list[dict]:
    """Build the deduplicated download list, skipping files already on disk.

    Sets deliberately overlap (fl2va and ref2va share the text encoder and both
    VAEs), so dedupe by destination path or the shared 15.7 GB encoder would be
    fetched twice.
    """
    seen: set[Path] = set()
    todo: list[dict] = []

    for name in set_names:
        for entry in manifest["sets"][name]["files"]:
            target = models_root / entry["dest"] / Path(entry["path"]).name
            if target in seen:
                continue
            seen.add(target)

            if target.exists() and target.stat().st_size > 0:
                log(f"[OK]   present  {entry['dest']}/{target.name}")
                continue
            todo.append({**entry, "target": target})

    return todo


def check_disk(todo: list[dict], models_root: Path) -> None:
    needed = sum(e.get("size_gb", 0) for e in todo) * DISK_HEADROOM
    free = shutil.disk_usage(models_root).free / 1e9
    log(f"       need ~{needed:.1f} GB (incl. headroom), {free:.1f} GB free")
    if free < needed:
        log(f"[ERROR] Not enough free space on {models_root}.")
        log("        Attach a larger network volume, or trim MODEL_SETS.")
        sys.exit(1)


def download(todo: list[dict], models_root: Path) -> int:
    from huggingface_hub import hf_hub_download

    failures = 0
    for i, entry in enumerate(todo, 1):
        target: Path = entry["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"[{i}/{len(todo)}] {target.name} ({entry.get('size_gb', 0):.2f} GB)")

        try:
            # local_dir makes hf_hub_download write straight into the volume.
            # Going through the shared HF cache instead would keep a second
            # full copy of every file - 127 GB for a 63 GB install.
            got = Path(
                hf_hub_download(
                    repo_id=entry["repo"],
                    filename=entry["path"],
                    local_dir=str(models_root),
                    token=os.environ.get("HF_TOKEN") or None,
                )
            )
            # The manifest's `dest` need not match the directory part of
            # `path`; when it differs, rename (same filesystem, so no copy).
            if got != target:
                got.replace(target)
            log(f"[OK]   {target.name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            log(f"[ERROR] {target.name}: {exc}")
            target.unlink(missing_ok=True)

    # Staging area for interrupted transfers; safe to drop once we are done.
    shutil.rmtree(models_root / ".cache" / "huggingface", ignore_errors=True)
    return failures


def main() -> int:
    if not MANIFEST.exists():
        log(f"[ERROR] Manifest not found: {MANIFEST}")
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    comfyui_dir = os.environ.get("COMFYUI_DIR")
    if not comfyui_dir:
        log("[ERROR] COMFYUI_DIR is not set.")
        return 1

    models_root = Path(comfyui_dir) / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # `--list <set>` prints the on-disk path of every file in a set, one per
    # line. init.sh uses it to warm the page cache without duplicating the
    # manifest logic in shell.
    if len(sys.argv) > 2 and sys.argv[1] == "--list":
        listed: set[Path] = set()
        for name in resolve_sets(manifest, sys.argv[2]):
            for entry in manifest["sets"][name]["files"]:
                target = models_root / entry["dest"] / Path(entry["path"]).name
                # Sets share the text encoder and both VAEs - emit each once.
                if target.exists() and target not in listed:
                    listed.add(target)
                    print(target)
        return 0

    set_names = resolve_sets(manifest, os.environ.get("MODEL_SETS", "default"))
    if not set_names:
        log("[SKIP] No model set selected (MODEL_SETS=none)")
        return 0

    log(f"       Model sets: {', '.join(set_names)}")
    todo = plan(manifest, set_names, models_root)

    if not todo:
        log("[OK]   All models already present")
        return 0

    check_disk(todo, models_root)
    failures = download(todo, models_root)

    if failures:
        log(f"[WARN] {failures} file(s) failed to download")
        # Non-fatal: the pod still boots, ComfyUI just will not list the model.
        return 0

    log("[OK]   All models downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
