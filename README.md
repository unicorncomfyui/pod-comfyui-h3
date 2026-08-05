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
├── .triton/                # Triton's JIT kernel cache - see below
├── input/  output/  user/
└── extra_model_paths.yaml  # regenerated on every boot
```

`.triton` matters more than its size suggests. Triton compiles kernels on first
use. Measured on an RTX 5090: the first sampling step took **37.17 s** on a cold
cache against **2.59 s** once warm — roughly 35 s of one-off compilation.
Keeping the cache on the volume pays that once rather than on every pod restart.

## Configuration

All behaviour is driven by environment variables — see [.env.example](.env.example).
The ones that matter:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SETS` | `minimax-h3-fl2va,minimax-h3-ref2va` | Which sets from `models/manifest.json` to download |
| `DOWNLOAD_MODELS` | `true` | Set `false` to boot without fetching weights |
| `FAST_DISK` | `false` | Trade host RAM for disk when offloading — only worth it on a RAM-starved pod |
| `PREWARM_SET` | — | Read one set into the page cache at boot. Recommended on a network volume |
| `CACHE_LRU` | — | Keep N node results; skips re-encoding an unchanged prompt |
| `FAST_MODE` | — | ComfyUI `--fast` features, or `all` |
| `ASYNC_OFFLOAD_STREAMS` | 2 | Weight-offload streams |
| `ENABLE_SSH` | `false` | Start sshd on port 22 |
| `PUBLIC_KEY` | — | SSH public key for root. Preferred over a password |
| `SSH_PASSWORD` | — | Root password for SSH. Also flips `PermitRootLogin`, which Ubuntu otherwise leaves at `prohibit-password` |

### SSH

You most likely do not need it: code-server on 8080 already gives you a
terminal, and RunPod ships its own web terminal on top. Leave `ENABLE_SSH` at
`false` unless something actually requires port 22.

If you do enable it, set `PUBLIC_KEY` or `SSH_PASSWORD`. With neither, sshd
listens but no login can succeed — root ships with a locked password and the
image carries no `authorized_keys`.

Host keys live on the network volume at `$COMFYUI_DATA_DIR/ssh`, generated on
first boot. They are therefore specific to your deployment and stable across
restarts, so the fingerprint your client pins stays valid.

### Tuning generation speed

On a measured run, 12 steps took 43.26 s of which ~30.8 s was sampling — the
other **~12.5 s was overhead**, and the log shows why: the 15 GB text encoder is
re-staged on *every* prompt, even when the text is unchanged.

Attack the two halves separately:

| Overhead (~29%) | Sampling (~71%) |
|---|---|
| `CACHE_LRU=10` — reuse the conditioning | `FAST_MODE=fp16_accumulation` |
| `PREWARM_SET` — first-load I/O | `FAST_MODE=cublas_ops` |
| `ASYNC_OFFLOAD_STREAMS=4` — ~40 GB crosses PCIe per run | `FAST_MODE=autotune` |

Which half dominates depends entirely on the workflow. Two runs on this pod,
identical models and step count, differed 3× per step — so on a light workflow
overhead was 29% of the total, and on a heavy one about 11%. Tune the half that
is actually large for *your* workflow.

**Before tuning anything, settle whether this is compute-bound at all.** A
third-party accelerator skips 30–35% of transformer evaluations for 2.6% of
wall-clock, which points at the ~40.5 GB crossing PCIe per run rather than at
the maths. One bench run decides it:

```bash
python /app/scripts/bench.py wf.json --config fp16 --config offload4 --config offload8
```

If `offload*` moves and `fp16` does not, `--fast` is the wrong place to spend
effort — and the real lever becomes a card that does not need to offload at all.
See [VERSIONS.md §11](VERSIONS.md) for the full survey and the dead ends.

`--fast` features are labelled untested and potentially quality-deteriorating
upstream. Judge the output, not only the clock.

### Benchmarking properly

Eyeballing two generations proves nothing: per-step time on this pod has been
measured at 0.85 s, 2.57 s and 8.07 s, a 9.5× spread driven purely by workflow
settings. [`scripts/bench.py`](scripts/bench.py) removes that variance.

```bash
# in the pod, from the VSCode terminal
python /app/scripts/bench.py my_workflow_api.json --reps 3
python /app/scripts/bench.py my_workflow_api.json --config fp16 --config lru
python /app/scripts/bench.py --list          # available configurations
```

Export the workflow with **Export (API)** — the plain save format is rejected
by `/prompt`, and the script says so rather than failing obscurely.

What it does, and why:

- **Spawns its own ComfyUI** per configuration on port 3111. The levers are CLI
  arguments, so comparing them requires a restart, and `start.sh`'s watchdog
  would restart the pod's instance with its original arguments.
- **Pins every seed widget**, so runs are actually comparable.
- **Times from ComfyUI's own history** (`execution_start` → `execution_success`),
  not wall-clock around the HTTP call, keeping queue and network out of it.
- **Discards the first run** of each configuration — it pays model staging and
  any cold Triton kernels.
- **Disables custom nodes and previews**, so third-party code and preview
  encoding stay out of the numbers.
- **Reports spread alongside the median.** If spread exceeds the difference
  between two configurations, you measured noise.

Each configuration reloads the models, so a full sweep is slow. Start with
`--config fp16 --config lru` rather than all nine.

### Finding your resolution / steps sweet spot

There is no published curve for H3 — the model is days old. But part of the
question has a structural answer.

**H3's native canvas is 768 px on the short edge, capped at 768×1344, rounded
to a multiple of 32** — about 1.0 megapixel at 16:9. That is where the model
learned. Above it, upstream is blunt: extra pixels *"may add pixels without
adding equivalent learned detail"*. The advertised 2K comes from in-model
regeneration, not from a bigger canvas.

So the ceiling worth paying for is **1344×768**. Beyond that, generate at
native and use a real upscaler — `4x-UltraSharp` ships in the image.

For steps, the templates offer two presets: **12** (speed) and **20** (quality).

And the budget that actually governs cost is **pixels × frames**, not resolution
alone. A duration change moves it as much as a resolution change — which is
exactly what produced 2.57 s and 8.07 s per step on identical model staging here.

To measure your own curve:

```bash
python /app/scripts/bench.py wf_api.json --sweep steps=8,12,16,20
python /app/scripts/bench.py wf_api.json --sweep megapixels=0.25,0.5,0.75,1.0
python /app/scripts/bench.py wf_api.json --sweep steps=12,20 --sweep megapixels=0.5,1.0
```

`--sweep` sets every widget of that name across the workflow and runs the full
cartesian product. Unlike `--config`, it reuses a **single** ComfyUI instance —
changing a widget does not change the command line, so reloading 40 GB of models
per point would cost minutes and prove nothing.

Time is all this measures. Quality is yours to judge: keep the seed fixed, watch
the outputs side by side, and find where more steps stop being visible.
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

AGPL-3.0 (inherited from ComfyUI).

**MiniMax H3 weights are a separate matter, and possibly a restrictive one.**
They ship under the MiniMax Community License. A
[HuggingFace discussion](https://huggingface.co/Comfy-Org/MiniMax-H3/discussions/11)
reports that the licence grants no rights to users in the **EU, US, UK and South
Korea** owing to ongoing litigation — and that this is what has prevented
distilled speed-up variants from being published.

That is a user report, not something verified against the licence text here.
Read the licence yourself before relying on H3 output commercially.
