#!/bin/bash
# Generic Docker build script with custom cache directory on /data
# Usage: ./build_with_cache.sh <dockerfile_dir> <image_name>

set -e

# Parse arguments
DOCKERFILE_DIR=${1:-.}
IMAGE_NAME=${2:-myimage}

echo "Building Docker image '$IMAGE_NAME' from $DOCKERFILE_DIR with cache on /data..."

# Set Docker cache directory
export DOCKER_BUILDKIT=1
export BUILDKIT_CACHE_DIR=/data/docker-buildkit-cache

# Create cache directories
echo "Setting up cache directories..."
sudo mkdir -p $BUILDKIT_CACHE_DIR /data/docker-cache /data/tmp
sudo chmod 777 $BUILDKIT_CACHE_DIR /data/docker-cache /data/tmp

# Set temp directory to /data to avoid space issues
export TMPDIR=/data/tmp
export DOCKER_TMPDIR=/data/docker-tmp

# Build Docker image with BuildKit and custom cache
echo "Building Docker image with cache on /data..."
cd "$DOCKERFILE_DIR"

# Check if docker buildx is available
if docker buildx version &>/dev/null; then
    echo "Using docker buildx with cache..."
    docker buildx build \
        --cache-from type=local,src=/data/docker-cache \
        --cache-to type=local,dest=/data/docker-cache,mode=max \
        -t "$IMAGE_NAME" \
        .
else
    echo "Using regular docker build (buildx not available)..."
    docker build -t "$IMAGE_NAME" .
fi

echo ""
echo "Build complete!"
echo "Image: $IMAGE_NAME"
echo "Cache stored in: /data/docker-cache"
echo ""

# Show image info
docker images | grep "$IMAGE_NAME" | head -1