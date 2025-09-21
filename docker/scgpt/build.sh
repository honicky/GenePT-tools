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

# Build Docker image
echo "Building Docker image..."
cd "$SCRIPT_DIR"
docker build -t scgpt-embeddings .

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