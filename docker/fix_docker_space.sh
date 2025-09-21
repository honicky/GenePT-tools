#!/bin/bash
# Quick fix for Docker space issues on EC2

set -e

echo "=== Fixing Docker Space Issues ==="
echo

# 1. Clean up Docker system
echo "1. Cleaning Docker cache and unused images..."
docker system prune -af --volumes || true
docker builder prune -af || true

echo
echo "2. Current disk usage:"
df -h

echo
echo "3. Docker storage location:"
docker info | grep "Docker Root Dir"

# 2. If still not enough space, move Docker to /data
echo
echo "4. Checking if we need to move Docker to /data..."
ROOT_SPACE=$(df / | awk 'NR==2 {print $4}' | sed 's/G//')
DATA_SPACE=$(df /data | awk 'NR==2 {print $4}' | sed 's/G//')

echo "   Root filesystem available: ${ROOT_SPACE}G"
echo "   /data available: ${DATA_SPACE}G"

if (( $(echo "$ROOT_SPACE < 10" | bc -l) )); then
    echo "   ⚠️  Less than 10GB on root filesystem!"
    echo "   Recommend moving Docker to /data with:"
    echo "   sudo ./setup_docker_on_ec2.sh"
fi

# 3. Alternative: Build with TMPDIR on /data
echo
echo "5. Quick workaround - build with temp directory on /data:"
echo
echo "   export TMPDIR=/data/tmp"
echo "   mkdir -p \$TMPDIR"
echo "   export DOCKER_TMPDIR=/data/docker-tmp"
echo "   mkdir -p \$DOCKER_TMPDIR"
echo "   "
echo "   # Then build normally:"
echo "   ./build.sh"
echo
echo "Or use the build_with_cache.sh script which uses /data for caching"

# 4. Set up temporary directories
echo
echo "6. Setting up temporary directories on /data..."
mkdir -p /data/tmp /data/docker-tmp /data/docker-cache
chmod 777 /data/tmp /data/docker-tmp /data/docker-cache

export TMPDIR=/data/tmp
export DOCKER_TMPDIR=/data/docker-tmp
echo "   TMPDIR set to: $TMPDIR"
echo "   DOCKER_TMPDIR set to: $DOCKER_TMPDIR"

echo
echo "=== Fixes applied ==="
echo
echo "Now you can build with:"
echo "  ./build_with_cache.sh"
echo
echo "Or if you need to permanently move Docker storage:"
echo "  sudo ./setup_docker_on_ec2.sh"