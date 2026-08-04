# Version & Compatibility Reference

State of the stack as of **2026-08-04**. Figures below are either pinned in this
repository or read from upstream; no benchmark numbers are quoted unless the
source is named.

---

## 1. Pinned versions

Everything is set by build args in [`Dockerfile`](Dockerfile) — change them
there, not in the CI workflow.

| Build arg | Default | Alternatives |
|---|---|---|
| `CUDA_BASE` | `13.3.1-cudnn-runtime-ubuntu24.04` | `12.9.2-cudnn-runtime-ubuntu24.04` |
| `TORCH_INDEX` | `cu130` | `cu129`, `cu132`, `cu126` |
| `TORCH_VERSION` | `2.13.0` | any version present on the chosen index |
| `PYTHON_VERSION` | `3.13` | 3.11 – 3.15 |
| `COMFYUI_VERSION` | `v0.30.0` | any tag on `Comfy-Org/ComfyUI` |
| `CODE_SERVER_VERSION` | `4.131.0` | — |

### torch 2.13.0 availability per CUDA index

Checked against `download.pytorch.org/whl/<index>/torch/`:

| Index | 2.13.0 | Python wheels | Notes |
|---|:---:|---|---|
| `cu126` | yes | cp310–cp315 | legacy |
| `cu128` | **no** | — | stops at 2.11.0 — do not target |
| `cu129` | yes | cp310–cp315 | fallback target |
| `cu130` | yes | cp310–cp315 | **primary**, ComfyUI's recommended stable index |
| `cu132` | yes | cp310–cp315 | ComfyUI's nightly index; viable, untested here |

`cu128` is a dead end for this project: it never received a torch 2.13 build.

---

## 2. Driver requirements

| Image target | Minimum host driver | RunPod availability |
|---|---|---|
| `cu130` | 580+ | machines advertise CUDA up to 13.3 |
| `cu129` | 575+ | the majority of the fleet |

RunPod's *Additional filters → CUDA Versions* dropdown spans 12.5 → 13.3, i.e.
the fleet is heterogeneous. This is the reason the `cu129` leg exists — not a
performance choice.

The driver is supplied by the host, never by the container. `init.sh` compares
the detected driver against `BUILD_TORCH_INDEX` at boot and prints an explicit
error on a mismatch rather than letting CUDA fail obscurely later.

---

## 3. MiniMax H3

Released 2026-07-31 (API), weights published 2026-08-03, native ComfyUI support
merged the same day in `Comfy-Org/ComfyUI` PR #15224 — hence the hard floor of
ComfyUI **0.30.0**.

**Output**: up to 2K (1440 px short edge), 24 fps, 4–15 s, native stereo audio.
Native canvas is 768 px short edge; 2K comes from in-model regeneration.
**Input**: ≤9 reference images, ≤3 reference videos, ≤3 reference audio clips,
12 files and 64 MB total, ≤7000-character prompt.

### Weights — [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)

| File | Size | Used by |
|---|---:|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97 GB | T2V, I2V |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97 GB | R2V |
| `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 20.96 GB | alternative quantisation |
| `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` | 34.04 GB | unpruned |
| `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | 66.28 GB | full precision |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69 GB | all sets |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14 GB | `-hq` set |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | all sets |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | all sets |

Totals: **42.48 GB** for T2V/I2V, **63.45 GB** adding R2V (the text encoder and
both VAEs are shared, and the downloader deduplicates them).

The full repository is 385 GB. Only the pruned variants are within reach of a
single 32 GB card.

### Target pod profile

| | RunPod RTX 5090 |
|---|---|
| VRAM | 32 GB GDDR7 |
| Host RAM | 92 GB |
| vCPU | 16 |
| Price | $0.99/hr |
| Max GPUs/pod | 8 |

Memory budget for one workflow — `fl2va` and `ref2va` are alternatives and are
never resident together, so the peak is a single diffusion model:

| Component | Size |
|---|---:|
| Diffusion model (pruned int8-convrot) | 20.97 GB |
| Text encoder (NVFP4 AWQ) | 15.69 GB |
| Video VAE (fp16) | 5.21 GB |
| Audio VAE (fp32) | 0.61 GB |
| **Peak resident** | **42.48 GB** |

Against 92 GB of host RAM that leaves roughly 50 GB for the OS, ComfyUI, pinned
transfer buffers and 2K video decode. Comfortable — which is why `FAST_DISK`
defaults to off: offloading to a *network* volume would be slower than the RAM
that is already available.

The 63.45 GB figure quoted elsewhere is the **disk** footprint of installing
both variants, not a RAM requirement.

The `minimax-h3-hq` set (67 GB of weights, unpruned) is the case where the 8-GPU
ceiling becomes interesting; it is not viable on a single card.

### Why RTX 5090 specifically

The text encoder is **NVFP4**. FP4 is a Blackwell tensor-core format; on Ada or
Ampere it falls back to an emulated path. This is an architectural requirement,
not a preference — `init.sh` warns when compute capability is not 12.0.

---

## 4. The runtime stack that replaced SageAttention

ComfyUI 0.27–0.30 moved kernels and offloading in-tree. Both are ordinary
dependencies in ComfyUI's `requirements.txt`, so they are pinned by the ComfyUI
tag rather than by this repository.

### `comfy-kitchen` — [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)

Kernel library with eager/cuda/triton/hip backends. Relevant to H3:
`quantize_nvfp4`, `dequantize_nvfp4`, `scaled_mm_nvfp4`, `quantize_int8_convrot_weight`,
`dequantize_int8_convrot_weight_dtype`, `int8_linear`, `gemv_awq_w4a16`, plus
fused RoPE and AdaLN variants.

### `comfy-aimdo` — [Comfy-Org/comfy-aimdo](https://github.com/Comfy-Org/comfy-aimdo)

A PyTorch VRAM allocator doing on-demand weight offloading. Models get a Virtual
Base Address Register costing only address space; tensors are faulted in when a
layer needs them and evicted under pressure, by priority. Requires PyTorch 2.8+
and CUDA 12.8+, NVIDIA only.

This is the mechanism that makes 42–63 GB of weights run on 32 GB of VRAM.

### SageAttention: removed

`thu-ml/SageAttention` has had no commit since 2026-01-17 and PyPI still serves
1.0.6. ComfyUI's own guidance notes that some H3 layers are not FP16/BF16, so
they fall back to standard PyTorch anyway. Compiling it cost ~20 minutes of
build time for no benefit on this model, so it is gone — which is also what
allows the image to sit on `-runtime` instead of `-devel`.

`--use-sage-attention` still exists in ComfyUI if you install it yourself.

---

## 5. ComfyUI memory flags

From `comfy/cli_args.py` at v0.30.0. Dynamic VRAM is on by default on NVIDIA
unless `--highvram`, `--gpu-only`, `--novram` or `--cpu` is passed.

| Flag | Effect |
|---|---|
| `--enable-dynamic-vram` | force on where it is not default |
| `--disable-dynamic-vram` | revert to estimate-based loading |
| `--vram-headroom N` | GB kept completely free above the default |
| `--fast-disk` | prefer disk-backed offload over unpinned RAM |
| `--async-offload [N]` | async weight offload, N streams (default 2, on for NVIDIA) |
| `--reserve-vram N` | GB reserved for the OS |
| `--disable-pinned-memory` | relieves host-RAM pressure, costs transfer speed |
| `--cache-none` | re-executes every node each run; lowest RAM |
| `--lowvram` | **no-op when dynamic VRAM is on** — do not use it here |

`--lowvram` is worth calling out: its own help text says it does nothing when
dynamic VRAM is enabled, and selecting it can only disable the better path.

---

## 6. GPU reference

| GPU | VRAM | Arch | CC | `TORCH_CUDA_ARCH_LIST` | H3 viable |
|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell | 96 GB | Blackwell | sm_120 | `12.0` | yes, comfortably |
| B200 | 180 GB | Blackwell (DC) | sm_120 | `12.0` | yes |
| **RTX 5090** | **32 GB** | **Blackwell** | **sm_120** | **`12.0`** | **yes, with offloading — the target** |
| H100 SXM | 80 GB | Hopper | sm_90 | `9.0` | yes, but no native NVFP4 |
| RTX 4090 | 24 GB | Ada | sm_89 | `8.9` | degraded, NVFP4 emulated |
| RTX 3090 | 24 GB | Ampere | sm_86 | `8.6` | not recommended |

---

## 7. Migration from `pod-comfyui-vscode`

| | Old (Dec 2025) | New |
|---|---|---|
| Base | `cuda:12.9.0-devel-ubuntu24.04` | `cuda:13.3.1-cudnn-runtime-ubuntu24.04` |
| Python | 3.11 | 3.13, in a venv |
| PyTorch | 2.8.0+cu129 | 2.13.0+cu130 |
| ComfyUI | commit `36357bb`, `comfyanonymous/*` | `v0.30.0`, `Comfy-Org/*` |
| Attention | SageAttention v2.2.0, compiled | comfy-kitchen, prebuilt |
| Offloading | none | comfy-aimdo |
| Model | Z-Image-Turbo (image) | MiniMax H3 (video + audio) |
| Custom nodes | 20 | 7, video-oriented |
| ComfyUI location | copied to the volume on first boot | stays in the image |
| Model selection | hardcoded in `init.sh` | `models/manifest.json` + `MODEL_SETS` |
| Build | local Docker on Windows | GitHub Actions, cu130/cu129 matrix |
| Registry origin | GitLab | GitHub |

The ComfyUI-location change is the one with lasting consequences: the old design
copied ComfyUI onto the network volume at first boot and then always preferred
that copy, so a volume permanently pinned whichever ComfyUI version created it
and image updates had no effect. Now only mutable data persists.

---

## 8. Verification commands

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import comfy_kitchen, comfy_aimdo; print('runtime stack OK')"
cat /usr/local/cuda/version.json
```

---

## 9. Build status

First green build: 2026-08-04, commit `d808b36`, branch `develop`.

| Target | Build | Image (compressed) |
|---|---|---|
| `cu130` | 11.8 min | 6.27 GB |
| `cu129` | 15.2 min | 8.46 GB |

Proven by that build: pip resolves torch 2.13.0 on the pinned CUDA index without
being clobbered by ComfyUI's unpinned `torch` requirement; ComfyUI 0.30.0
installs against transformers 5.x; the six custom nodes install under Python
3.13; nothing needs a compiler, so the `-runtime` base holds; Trivy reports no
CRITICAL/HIGH findings.

**Not proven**: nothing has run on a GPU yet. Whether comfy-kitchen's NVFP4 path
engages on real sm_120 hardware, whether comfy-aimdo keeps a 42.5 GB working set
inside 32 GB of VRAM, and end-to-end H3 generation all remain untested until a
pod is deployed.

---

**Last updated**: 2026-08-04
**Primary target**: `cu130` — RTX 5090, driver 580+
