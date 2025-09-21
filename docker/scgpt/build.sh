#!/bin/bash
# Build script for scGPT Docker image

set -e

echo "Building scGPT Docker image..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Copy necessary scripts to docker build context
echo "Copying scripts to build context..."
cp "$REPO_ROOT/scripts/generate_scgpt_embeddings.py" "$SCRIPT_DIR/"
cp "$REPO_ROOT/scripts/scgpt_wrapper.py" "$SCRIPT_DIR/"

# Check if we should use the cache build script (if /data exists and has space)
if [ -d "/data" ] && [ $(df /data | awk 'NR==2 {print int($4/1024/1024)}') -gt 10 ]; then
    echo "Using build_with_cache.sh for building on /data..."
    "$REPO_ROOT/docker/build_with_cache.sh" "$SCRIPT_DIR" "scgpt-embeddings"
else
    echo "Building Docker image with standard docker build..."
    cd "$SCRIPT_DIR"
    docker build -t scgpt-embeddings .
fi

# Clean up copied files
echo "Cleaning up..."
rm -f generate_scgpt_embeddings.py scgpt_wrapper.py

echo "Build complete!"
echo ""
echo "To tag for ECR:"
echo "  docker tag scgpt-embeddings:latest 971422677163.dkr.ecr.us-west-2.amazonaws.com/scgpt-embeddings:latest"
echo ""
echo "To push to ECR:"
echo "  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 971422677163.dkr.ecr.us-west-2.amazonaws.com"
echo "  docker push 971422677163.dkr.ecr.us-west-2.amazonaws.com/scgpt-embeddings:latest"