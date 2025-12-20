#!/usr/bin/env python3
"""
Docker Build and Push Script
Builds Docker image, pushes to Docker Hub, then cleans up local image
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        return False

    print(f"Loading credentials from {env_path}...")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    return True


def run_command(cmd, check=True, input_data=None, show_progress=False):
    """Run a shell command and return output"""
    print(f"\n> {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    try:
        if show_progress:
            # For commands that show progress (docker build, docker push)
            # Don't capture output, let it stream to terminal
            if input_data:
                result = subprocess.run(
                    cmd,
                    shell=isinstance(cmd, str),
                    check=check,
                    input=input_data,
                    text=True
                )
            else:
                result = subprocess.run(
                    cmd,
                    shell=isinstance(cmd, str),
                    check=check
                )
            return result
        else:
            # For other commands, capture and print output
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                check=check,
                capture_output=True,
                text=True,
                input=input_data
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode == 0:
                print(result.stderr)

            return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print(e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr)
        raise


def get_commit_sha():
    """Get short git commit SHA"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "local"


def cleanup_wsl_disk():
    """Compact WSL2 docker disk to reclaim space"""
    if os.name != 'nt':  # Only on Windows
        return

    print("\n" + "=" * 50)
    print("Compacting WSL2 Docker disk...")
    print("=" * 50)

    try:
        # Shutdown WSL to release the VHDX
        print("\nShutting down WSL...")
        run_command(['wsl', '--shutdown'], check=False)

        import time
        time.sleep(3)  # Wait for shutdown

        # Find the docker VHDX path
        username = os.getenv('USERNAME')
        vhdx_path = f"C:\\Users\\{username}\\AppData\\Local\\Docker\\wsl\\data\\ext4.vhdx"

        if not os.path.exists(vhdx_path):
            print(f"Docker VHDX not found at {vhdx_path}")
            return

        print(f"\nCompacting {vhdx_path}...")
        print("This may take 1-2 minutes...")

        # Try Optimize-VHD first (requires Hyper-V)
        try:
            run_command([
                'powershell', '-Command',
                f'Optimize-VHD -Path "{vhdx_path}" -Mode Full'
            ], check=True)
            print("✅ VHDX compacted successfully with Optimize-VHD")
        except:
            # Fallback to diskpart (works without Hyper-V)
            print("Optimize-VHD failed, trying diskpart...")

            diskpart_script = f"""select vdisk file="{vhdx_path}"
compact vdisk
exit
"""
            result = run_command(
                ['diskpart'],
                input_data=diskpart_script,
                check=False
            )

            if result.returncode == 0:
                print("✅ VHDX compacted successfully with diskpart")
            else:
                print("⚠️  Could not compact VHDX automatically")
                print(f"   Run manually: diskpart > select vdisk file=\"{vhdx_path}\" > compact vdisk")

    except Exception as e:
        print(f"⚠️  WSL disk compaction failed: {e}")
        print("   You can manually compact later if needed")


def main():
    # Configuration
    IMAGE_NAME = "vlop12ui/pod-comfyui-vscode"

    # Check for --local flag
    local_mode = "--local" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--local"]
    BRANCH = args[0] if args else "develop"

    # Load .env file
    if not load_env_file():
        print("WARNING: .env file not found")

    # Check credentials
    DOCKER_USERNAME = os.getenv('DOCKER_USERNAME')
    DOCKER_PASSWORD = os.getenv('DOCKER_PASSWORD')

    if not DOCKER_USERNAME or not DOCKER_PASSWORD:
        print("ERROR: DOCKER_USERNAME and DOCKER_PASSWORD must be set!")
        print("")
        print("Either:")
        print("1. Create a .env file with your credentials (copy .env.example)")
        print("2. Set environment variables in System Properties")
        print("3. Set them before running this script:")
        print('   set DOCKER_USERNAME=vlop12ui')
        print('   set DOCKER_PASSWORD=your-token')
        sys.exit(1)

    # Get commit info
    COMMIT_SHA = get_commit_sha()
    BUILD_DATE = datetime.now().strftime("%Y%m%d")

    # Determine tags based on branch
    # Simple strategy: only branch-SHA tag for traceability
    TAGS = [f"{IMAGE_NAME}:{BRANCH}-{COMMIT_SHA}"]

    # Print header
    print("=" * 50)
    print(f"Building Docker image: {IMAGE_NAME}")
    print(f"Branch: {BRANCH}")
    print(f"Commit: {COMMIT_SHA}")
    print(f"Tag: {TAGS[0].split(':')[1]}")
    print("=" * 50)

    # Clean up Docker before build to free space
    print("\n" + "=" * 50)
    print("Cleaning up Docker before build...")
    print("=" * 50)
    run_command(['docker', 'system', 'prune', '-af', '--volumes'], check=False)

    # Check disk space before build
    print("\nDisk space before build:")
    try:
        if os.name == 'nt':  # Windows
            run_command('wmic logicaldisk get caption,freespace,size', check=False)
        else:  # Linux/macOS
            run_command('df -h', check=False)
    except:
        pass

    # Build the image
    print("\n" + "=" * 50)
    print("Building image...")
    print("=" * 50)

    build_cmd = ['docker', 'build']
    for tag in TAGS:
        build_cmd.extend(['-t', tag])
    build_cmd.append('.')

    run_command(build_cmd, show_progress=True)

    print("\n✅ Build completed successfully!")

    # Show image size
    print("\nImage size:")
    run_command(f'docker images {IMAGE_NAME} --format "table {{{{.Repository}}}}:{{{{.Tag}}}}\t{{{{.Size}}}}"', check=False)

    # Login to Docker Hub
    print("\n" + "=" * 50)
    print("Logging in to Docker Hub...")
    print("=" * 50)

    run_command(
        ['docker', 'login', '-u', DOCKER_USERNAME, '--password-stdin'],
        input_data=DOCKER_PASSWORD
    )

    # Push all tags
    print("\n" + "=" * 50)
    print("Pushing images to Docker Hub...")
    print("=" * 50)

    for tag in TAGS:
        print(f"\nPushing: {tag}")
        run_command(['docker', 'push', tag], show_progress=True)

    print("\n✅ All tags pushed successfully!")

    # Logout from Docker Hub
    run_command(['docker', 'logout'], check=False)

    # Clean up: Remove all local images for this build
    print("\n" + "=" * 50)
    print("Cleaning up local images to save disk space...")
    print("=" * 50)

    for tag in TAGS:
        run_command(['docker', 'image', 'rm', tag], check=False)

    # Also clean dangling images from build
    run_command(['docker', 'image', 'prune', '-f'], check=False)

    # Compact WSL disk to reclaim space (Windows only)
    if os.name == 'nt':
        cleanup_wsl_disk()

    # Show disk space after cleanup
    print("\nDisk space after cleanup:")
    try:
        if os.name == 'nt':  # Windows
            run_command('wmic logicaldisk get caption,freespace,size', check=False)
        else:
            run_command('df -h', check=False)
    except:
        pass

    # Final summary
    print("\n" + "=" * 50)
    print("✅ Build, push, and cleanup completed!")
    print(f"📦 Image: {IMAGE_NAME}")
    print(f"🏷️  Tag: {TAGS[0].split(':')[1]}")
    print("💾 Local disk space freed")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)
