# RunPod ComfyUI Pod — MiniMax H3 / RTX 5090

**English** | **[Français](README.fr.md)**

![CUDA](https://img.shields.io/badge/CUDA-13.3%20%7C%2012.9-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-red) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![ComfyUI](https://img.shields.io/badge/ComfyUI-v0.30.0-purple) ![Model](https://img.shields.io/badge/MiniMax-H3-orange)

Persistent RunPod Pod running **ComfyUI** + **VSCode (code-server)**, built around
**MiniMax H3** video generation on the **RTX 5090** (Blackwell, sm_120).

MiniMax H3 generates up to **2K, 24 fps, 4–15 s clips with native stereo audio**
(dialogue, effects and room tone produced in the same pass) from text, images,
video or audio references.

## What makes this pod work

H3's usable weights total **42.5 GB** (T2V/I2V) to **63.4 GB** (with R2V) against
32 GB of VRAM. Three things close that gap, all of them new since ComfyUI 0.27–0.30:

| Piece | Role |
|---|---|
| **comfy-aimdo** | Dynamic VRAM allocator. Faults model weights in on demand and offloads them under pressure. Enabled by default on NVIDIA. |
| **comfy-kitchen** | ComfyUI's kernel library: NVFP4, int8-convrot, AWQ w4a16, fused RoPE/AdaLN. |
| **Pruned int8-convrot weights** | ~40% of parameters (modulation) replaced by a lookup table, the rest quantised to int8. |

The text encoder ships in **NVFP4**, which needs Blackwell tensor cores — hence
the RTX 5090 target rather than a cheaper card.

Both packages ship as **prebuilt wheels**, so nothing is compiled when the image
is built. SageAttention, which the previous generation of this pod built from
source, is gone: its upstream has been idle since January 2026 and H3's int8
layers are not FP16/BF16 anyway.

The image does still carry `gcc`, on purpose. Triton compiles its kernels
just-in-time *at runtime* — torch 2.13 routes `torch._native` ops through it,
and the H3 text encoder's RoPE hits one — so a compiler must be present or
generation dies with `Failed to find C compiler`. Build-time and run-time
compilation are separate questions; only the first one was eliminated.

## Quick start

### 1. Deploy on RunPod

Target pod profile — the defaults in this repo are tuned for it:

| | |
|---|---|
| GPU | RTX 5090, 32 GB VRAM |
| Host RAM | **92 GB** |
| vCPU | 16 |
| Price | $0.99/hr |

92 GB of RAM is the number that matters: a single H3 workflow needs ~42.5 GB
resident, so it fits outright with margin to spare. This is why `FAST_DISK`
defaults to `false` (see [Host RAM](#host-ram-is-the-real-constraint)).

1. GPU: **RTX 5090** (32 GB VRAM / 92 GB RAM / 16 vCPU)
2. Container disk: 30 GB
3. **100 GB of persistent storage at `/workspace`** — the weights live there,
   not in the image. See the trade-off below.
4. Image: `vlop12ui/pod-comfyui-h3:latest`
5. **Allowed CUDA versions: 13.0, 13.1, 13.2, 13.3 only.** Untick 12.x.

> This one is not optional. Leaving 12.8/12.9 ticked lets RunPod schedule the
> pod on a machine whose driver cannot run a CUDA 13 image; torch then reports
> `CUDA available: False` and nothing works. Switching to `:cu129` does not
> rescue it either — a driver capped at CUDA 12.8 is a 570.x, and cu129 wants
> 575+. Verified the hard way on 2026-08-04.

#### Volume disk or network volume?

Both mount at `/workspace` and this image works with either. RunPod rates volume
disk as *fast (local)* and network volume as *variable (network)*:

| | Volume disk | Network volume |
|---|---|---|
| Speed | Local — faster weight loading | Network — variable |
| Persistence | Until the pod is **deleted** | Independent of any pod |
| Shareable across pods | No | Yes |
| Re-download 63 GB on pod deletion | Yes | No |

Pick **volume disk** for a single long-lived pod: loading 42 GB of weights each
generation session is I/O-bound and local storage wins. Pick **network volume**
if you spin pods up and down, or run several against the same weights.

On a **network volume**, set `PREWARM_SET=minimax-h3-fl2va`. Reads are
network-bound there, so those 42 GB would otherwise stream in as unpredictable
stalls during the first generation; prewarming turns that into a one-off,
visible cost at boot. Name a single set — prewarming more than fits in RAM just
evicts itself, and `init.sh` skips the step if RAM is short.

`FAST_DISK=true` only ever makes sense on a volume disk, and even then only if
host RAM is short.

**Sizing**: 63.45 GB for both H3 variants leaves ~36 GB on a 100 GB volume for
outputs and inputs. Go to 150 GB if you generate heavily or keep raw footage.

> If no CUDA 13 machine is available, use `:cu129` instead. It is the same
> PyTorch on an older CUDA runtime and runs on 12.9 hosts.

### 2. Access

- **ComfyUI**: `https://<pod-id>-3000.proxy.runpod.net`
- **VSCode**: `https://<pod-id>-8080.proxy.runpod.net`

First boot downloads 42–63 GB of weights. That happens **in the background**,
after code-server and ComfyUI are already listening — so you get a shell and a
UI immediately rather than waiting 15+ minutes blind. Follow it with
`tail -f /var/log/models.log`. H3 appears in the loaders as each file lands,
on the next UI refresh.

If the host driver is too old for the image's CUDA runtime, the pod stops at
diagnostics with a `[FATAL]` block instead of booting and failing obscurely
later. That is almost always the *Allowed CUDA versions* filter in the template
being too permissive — see below.

### 3. Generate

ComfyUI 0.30 ships the H3 templates: *Template Library → MiniMax H3 T2V / I2V / R2V*.

## Image tags

Every tag carries its CUDA target, so the two build legs can never overwrite
each other.

| Tag | Meaning |
|---|---|
| `latest` | Newest `cu130` build from `main` |
| `cu130` / `cu129` | Newest build of that target |
| `cu130-main`, `cu129-develop` | Newest build of a target on a branch |
| `cu130-main-<sha>` | Exact commit — use this for reproducibility |
| `cu130-<date>-<sha>` | Chronologically sortable |

## Stack

| Component | Version | Note |
|---|---|---|
| Base image | `nvidia/cuda:13.3.1-cudnn-runtime-ubuntu24.04` | `-runtime`, not `-devel`: nothing is compiled |
| PyTorch | 2.13.0+cu130 | ComfyUI's recommended stable index |
| Python | 3.13 | in a venv at `/opt/venv` |
| ComfyUI | v0.30.0 | from `Comfy-Org/ComfyUI` (the repo moved orgs) |
| comfy-kitchen | pinned by ComfyUI | NVFP4 / int8-convrot kernels |
| comfy-aimdo | pinned by ComfyUI | dynamic VRAM offloader |
| code-server | 4.131.0 | VSCode in the browser |
| Driver required | 580+ (cu130), 575+ (cu129) | provided by the RunPod host |

Custom nodes are limited to a video-oriented set: ComfyUI-Manager,
VideoHelperSuite, KJNodes, rgthree, Frame-Interpolation, cg-use-everywhere.
The image-era suites (WAS, Impact Pack, Comfyroll, RES4LYF…) were dropped — they
are heavy and break on core version bumps. Easy-Use was dropped for a specific
reason: it depends on `clip_interrogator` 0.6.0, last released in March 2023,
which is not a safe bet against transformers 5.x.

## Volume layout

ComfyUI itself stays **in the image** and is never copied to the volume. Only
mutable data persists. Updating ComfyUI is therefore just pulling a newer tag,
rather than being permanently shadowed by a stale copy on the volume — which is
what the previous pod design did.

```
/workspace/comfyui-data/
├── models/
│   ├── diffusion_models/   # minimax_h3_*_pruned_int8_convrot.safetensors
│   ├── text_encoders/      # qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── vae/                # video (fp16) + audio (fp32) VAEs
│   └── loras/ upscale_models/ checkpoints/
├── custom_nodes/           # your own nodes, survive image updates
├── input/  output/  user/
└── extra_model_paths.yaml  # regenerated on every boot
```

## Configuration

All behaviour is driven by environment variables — see [.env.example](.env.example).
The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SETS` | `minimax-h3-fl2va,minimax-h3-ref2va` | Which sets from `models/manifest.json` to download |
| `DOWNLOAD_MODELS` | `true` | Set `false` to boot without fetching weights |
| `FAST_DISK` | `false` | Trade host RAM for disk when offloading — only worth it on a RAM-starved pod |
| `PREWARM_SET` | — | Read one set into the page cache at boot. Recommended on a network volume |
| `VRAM_HEADROOM` | — | Extra GB kept free; raise on OOM mid-sampling |
| `COMFYUI_EXTRA_ARGS` | — | Appended verbatim to the ComfyUI command line |

### Adding a model

Add an entry to [models/manifest.json](models/manifest.json) and reference it in
`MODEL_SETS`. No change to `init.sh` or the `Dockerfile`:

```json
"my-model": {
  "description": "...",
  "default": false,
  "files": [
    { "repo": "org/repo", "path": "diffusion_models/x.safetensors",
      "dest": "diffusion_models", "size_gb": 12.3 }
  ]
}
```

Sets may overlap — shared files are downloaded once. The downloader skips files
already present, so re-running on an existing volume is cheap.

## Host RAM is the real constraint

On a 32 GB card the bottleneck is **system RAM**, not VRAM: the offloader streams
weights through it.

The figure that matters is **~42.5 GB** — one diffusion model plus the text
encoder and both VAEs. Not the 63.4 GB a full `fl2va` + `ref2va` install
occupies on disk: `fl2va` and `ref2va` are alternatives, never resident together.

| Host RAM | Verdict |
|---|---|
| 92 GB (target pod) | Comfortable — weights fit in RAM with margin |
| 48–80 GB | Workable, little margin |
| < 48 GB | Expect swapping or OOM |

`init.sh` reports which bracket the pod falls in at boot.

**On `FAST_DISK`**: it makes the offloader use disk instead of host RAM. That
only pays off with fast *local* NVMe. On RunPod the weights sit on a **network
volume**, so enabling it on a 92 GB pod is a pessimisation twice over — hence
the `false` default. `init.sh` warns if you set it anyway on a large-RAM host.

If you do hit host-memory exhaustion, in order:

1. `COMFYUI_EXTRA_ARGS=--disable-pinned-memory`
2. `COMFYUI_EXTRA_ARGS=--disable-pinned-memory --cache-none`
3. `FAST_DISK=true` — last resort, trades speed for survival
4. Lower resolution and duration, then change one variable at a time

Do **not** add `--lowvram`: it disables dynamic VRAM, which is the mechanism
making H3 viable here.

## Building

Builds run on GitHub Actions — see [BUILD.md](BUILD.md). Local builds are
supported for smoke tests via `docker compose up --build`, but are not the
publishing path.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **403 on port 3000, while 8080 works** | ComfyUI rejects `Sec-Fetch-Site: cross-site`, which is what clicking the dashboard link sends | Retype the URL in the address bar, or keep `ENABLE_CORS=true` (default) |
| `Failed to find C compiler` mid-generation | Triton JIT-compiles kernels at runtime and found no `gcc` | Fixed in the image; if you stripped it, reinstall `gcc libc6-dev` |
| H3 templates missing | ComfyUI < 0.30.0 | Pull a newer image tag |
| Model absent from loader | Download incomplete | Check the pod log; re-run with `DOWNLOAD_MODELS=true` |
| `CUDA error` / driver mismatch at boot | cu130 image on a 12.x host | Redeploy with the CUDA filter, or use `:cu129` |
| R2V fails, T2V works | Wrong diffusion model selected | Pick `minimax_h3_ref2va_*`, and add `minimax-h3-ref2va` to `MODEL_SETS` |
| Video generated without audio | Audio VAE not wired | Both VAE decodes must feed the `CreateVideo` node |
| Clip slightly longer than requested | H3 frame-grid alignment (17k+5) | Expected: 5 s → 124 frames ≈ 5.17 s at 24 fps |

## License

AGPL-3.0 (inherited from ComfyUI). MiniMax H3 weights are covered by the
MiniMax Community License — check its terms before commercial use.
