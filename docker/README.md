# Docker Utilities

This directory contains utilities for building and managing Docker images, especially on EC2 instances with limited root filesystem space.

## General Docker Management Scripts

### `setup_docker_on_ec2.sh`
Configures Docker to use `/data` volume for storage instead of the default `/var/lib/docker`. This is essential for EC2 instances with limited root filesystem space.

```bash
sudo ./setup_docker_on_ec2.sh
```

This script:
- Stops Docker service
- Moves Docker data to `/data/docker`
- Updates Docker daemon configuration
- Restarts Docker with new storage location

### `fix_docker_space.sh`
Quick fixes for Docker space issues without moving the entire Docker installation.

```bash
./fix_docker_space.sh
```

This script:
- Cleans up Docker cache and unused images
- Shows current disk usage
- Sets up temporary directories on `/data`
- Provides recommendations for permanent fixes

### `build_with_cache.sh`
Generic Docker build script that uses `/data` for caching to avoid space issues.

```bash
# Build any Docker image with cache on /data
./build_with_cache.sh <dockerfile_dir> <image_name>

# Examples:
./build_with_cache.sh scgpt/ scgpt-embeddings
./build_with_cache.sh training/ training-image
```

Features:
- Uses Docker BuildKit with cache on `/data`
- Sets TMPDIR to `/data/tmp` to avoid space issues
- Works with any Dockerfile
- Falls back to regular build if buildx is not available

## Project-Specific Docker Images

### `scgpt/`
Docker image for scGPT embedding generation. See [scgpt/README.md](scgpt/README.md) for details.

### `training/` (if exists)
Docker image for model training. See specific README in that directory.

## Common Issues and Solutions

### "No space left on device" during Docker build

**Option 1: Quick Fix**
```bash
# Clean up and build with cache on /data
./fix_docker_space.sh
./build_with_cache.sh <dir> <image>
```

**Option 2: Permanent Fix**
```bash
# Move Docker storage to /data
sudo ./setup_docker_on_ec2.sh
```

### Docker daemon not starting after configuration change

```bash
# Check Docker daemon logs
sudo journalctl -u docker.service

# Reset to default configuration
sudo rm /etc/docker/daemon.json
sudo systemctl restart docker
```

### BuildKit not available

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Or install docker-buildx-plugin
sudo apt-get update
sudo apt-get install docker-buildx-plugin
```

## Best Practices

1. **On EC2 with limited root space**: Always run `setup_docker_on_ec2.sh` first
2. **For large builds**: Use `build_with_cache.sh` instead of direct `docker build`
3. **Regular cleanup**: Run `docker system prune -af` periodically
4. **Monitor space**: Check with `df -h` before and after builds

## Directory Structure

```
docker/
├── README.md                    # This file
├── setup_docker_on_ec2.sh      # Move Docker to /data
├── fix_docker_space.sh          # Quick space fixes
├── build_with_cache.sh          # Generic build with /data cache
└── scgpt/                       # scGPT-specific Docker files
    ├── Dockerfile
    ├── build.sh                 # scGPT build script
    └── README.md                # scGPT-specific docs
```