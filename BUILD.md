# Local Docker Build Guide

Build Docker images locally on your RTX 3080 instead of using GitLab CI/CD. This is faster, more reliable, and automatically cleans up disk space.

## Prerequisites

- Docker installed and running
- Git installed (to get commit SHA)
- 200GB+ disk space on D: drive
- Docker Hub credentials set as environment variables

## Setup Environment Variables

Set your Docker Hub credentials:

```bash
# Windows PowerShell
$env:DOCKER_USERNAME="vlop12ui"
$env:DOCKER_PASSWORD="your-docker-hub-token"

# Windows CMD
set DOCKER_USERNAME=vlop12ui
set DOCKER_PASSWORD=your-docker-hub-token

# Linux/macOS
export DOCKER_USERNAME="vlop12ui"
export DOCKER_PASSWORD="your-docker-hub-token"
```

## Usage

### Build and Push Develop Branch

```bash
# Default: builds develop branch
bash build-and-push.sh

# Or explicitly
bash build-and-push.sh develop
```

This will:
1. Build image with tags: `develop`, `develop-{SHA}`, `{DATE}-{SHA}`
2. Push all tags to Docker Hub
3. Delete local images to free disk space
4. Show disk space before/after

### Build and Push Main Branch

```bash
bash build-and-push.sh main
```

This will:
1. Build image with tags: `latest`, `main`, `main-{SHA}`, `{DATE}-{SHA}`
2. Push all tags to Docker Hub
3. Delete local images to free disk space

## What Gets Created

For **develop** branch:
- `vlop12ui/pod-comfyui-vscode:develop`
- `vlop12ui/pod-comfyui-vscode:develop-abc1234` (commit SHA)
- `vlop12ui/pod-comfyui-vscode:20251220-abc1234` (date + SHA)

For **main** branch:
- `vlop12ui/pod-comfyui-vscode:latest`
- `vlop12ui/pod-comfyui-vscode:main`
- `vlop12ui/pod-comfyui-vscode:main-abc1234`
- `vlop12ui/pod-comfyui-vscode:20251220-abc1234`

## Disk Space Management

The script automatically:
- Shows disk space before build
- Builds image (~10-14GB optimized, ~25-35GB during build)
- Pushes to Docker Hub
- **Deletes all local images** for this build
- Cleans up dangling images
- Shows disk space after cleanup

**After build completes: 0GB local disk usage**

Images are only stored on Docker Hub, not locally.

## Build Time Estimates

On RTX 3080 with 200GB disk:
- First build: ~20-30 minutes (downloads base images, PyTorch nightly)
- Subsequent builds: ~10-15 minutes (Docker layer cache)
- Push to Docker Hub: ~8-12 minutes (larger image than rtx3000)
- **Total: ~25-40 minutes** (vs 60+ minutes or failures on GitLab)

## Advantages Over GitLab CI/CD

1. **Faster**: 2-3x faster with local Docker cache
2. **More reliable**: No runner failures, no disk space errors
3. **More disk space**: 200GB vs 10-12GB on shared runners
4. **GPU acceleration**: Helps with CUDA 12.8.1 compilation
5. **Zero local storage**: Auto-cleanup after push
6. **Full control**: See real-time progress, debug issues

## Troubleshooting

### Error: Docker login failed
Make sure `DOCKER_USERNAME` and `DOCKER_PASSWORD` are set correctly.

### Error: No space left on device
- Free up space on D: drive (need ~35GB during build)
- Run `docker system prune -af --volumes` to clean Docker cache

### Error: git not found
Install Git for Windows or make sure it's in PATH.

### Build is slow
- First build downloads large base images (CUDA 12.8.1)
- PyTorch nightly is larger than stable
- Subsequent builds will be much faster due to layer cache
- Consider building on SSD instead of HDD

## Workflow Integration

### Git Flow Workflow

```bash
# 1. Make changes on develop branch
git checkout develop
# ... make changes ...

# 2. Commit and push to GitLab
git add .
git commit -m "feat: add new feature"
git push origin develop

# 3. Build and push Docker image
bash build-and-push.sh develop

# 4. When ready for production
git checkout main
git merge develop
git push origin main
bash build-and-push.sh main
```

### Quick One-Liner

```bash
# Commit, push git, build and push Docker - all in one
git add . && git commit -m "update" && git push && bash build-and-push.sh
```

## Manual Build (Without Script)

If you prefer manual control:

```bash
# Build
docker build -t vlop12ui/pod-comfyui-vscode:develop .

# Push
docker push vlop12ui/pod-comfyui-vscode:develop

# Clean up
docker image rm vlop12ui/pod-comfyui-vscode:develop
docker image prune -f
```
