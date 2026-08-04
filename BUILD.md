# Build & CI

Images are built and published exclusively by **GitHub Actions**. Local builds
exist only for smoke-testing a change before pushing.

## One-time setup

### 1. Create the GitHub repository

```bash
gh repo create unicorncomfyui/pod-comfyui-h3 --private --source=. --remote=origin
git push -u origin main-h3
```

Or, without `gh`: create the repository in the web UI, then

```bash
git remote add origin https://github.com/unicorncomfyui/pod-comfyui-h3.git
git push -u origin main-h3
```

### 2. Add the Docker Hub secrets

*Settings → Secrets and variables → Actions*:

| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub **access token**, not the account password |

Without these, `validate.yml` still runs but `docker-build.yml` fails at login.

### 3. Enable code scanning

*Settings → Code security → Code scanning* must be on for the Trivy SARIF
upload to succeed. On a private repository this requires GitHub Advanced
Security; if you do not have it, drop the two Trivy steps from
`docker-build.yml`.

## Workflows

| Workflow | Trigger | Duration | Does |
|---|---|---|---|
| `validate.yml` | every push and PR | ~30 s | hadolint, shellcheck, manifest schema, compose syntax |
| `docker-build.yml` | push to `main`/`develop`, manual | ~20–30 min per target | builds and publishes the CUDA matrix |

`validate.yml` is the cheap gate: a shell typo caught there saves a 20-minute
build.

Pull requests build the image but never publish it — the Docker Hub login and
push steps are skipped.

## The build matrix

Both targets come from the same `Dockerfile`, parameterised by two build args:

| Target | `CUDA_BASE` | `TORCH_INDEX` | Host driver |
|---|---|---|---|
| `cu130` (primary) | `13.3.1-cudnn-runtime-ubuntu24.04` | `cu130` | 580+ |
| `cu129` (fallback) | `12.9.2-cudnn-runtime-ubuntu24.04` | `cu129` | 575+ |

Only `cu130` claims the `latest` tag, and only from `main`.

### Adding a target

Append to the `matrix.include` list in `docker-build.yml`:

```yaml
- target: cu132
  cuda_base: 13.2.1-cudnn-runtime-ubuntu24.04
  torch_index: cu132
  primary: false
```

Nothing else changes: tags, cache scope and the Trivy scan all derive from
`target`.

### Building one target only

*Actions → Build and Push Docker Image → Run workflow*, then pick the target
from the dropdown. Useful when only the fallback needs a rebuild.

## Caching

Each matrix leg uses its own GitHub Actions cache scope
(`cache-to: type=gha,scope=<target>`). Sharing one scope would make the two legs
evict each other on every run, and neither would ever hit.

GitHub caps Actions cache at 10 GB per repository; expect partial hits on the
heavy PyTorch layer. A cold build is ~25–30 min, a warm one closer to 10.

## Disk space on the runner

`ubuntu-latest` starts with roughly 14 GB free, which is not enough. The
workflow's first step removes the preinstalled .NET, Android, GHC, Swift and
CodeQL toolchains to recover ~25 GB.

The published image stays manageable because **no model weights are baked in** —
the 42–63 GB of MiniMax H3 is downloaded to the network volume at first boot.
Baking them in would blow past both the runner disk and the Docker Hub layer
limits.

## Local smoke test

```bash
# cu130 (default)
docker compose up --build

# cu129
CUDA_BASE=12.9.2-cudnn-runtime-ubuntu24.04 TORCH_INDEX=cu129 docker compose up --build
```

`DOWNLOAD_MODELS` defaults to `false` in `docker-compose.yml`, so a local run
does not pull 63 GB. Set it to `true` only when deliberately testing the
downloader.

To lint exactly as CI does, before pushing:

```bash
docker run --rm -i hadolint/hadolint < Dockerfile
shellcheck --severity=error init.sh start.sh
python -m py_compile scripts/download_models.py
```

## Branches

- `develop` — publishes `cu130-develop`, `cu129-develop`
- `main` — publishes `latest`, `cu130-main`, `cu129-main`

Deploy pods from an immutable `cu130-main-<sha>` tag rather than `latest`, so a
rebuild cannot change what a running template resolves to.
