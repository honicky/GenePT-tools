#!/bin/bash
# Script to rebuild and push the Docker image with updated dependencies

set -e  # Exit on error

# Configuration
AWS_ACCOUNT_ID="971422677163"
AWS_REGION="us-west-2"
ECR_REPOSITORY="miratyper-training"
IMAGE_TAG="latest"

echo "========================================="
echo "Rebuilding Docker Image with Optuna"
echo "========================================="

# Navigate to project root
cd /Users/rj/personal/GenePT-tools

# Build the Docker image for AMD64 architecture (required for AWS)
echo "Building Docker image..."
docker build \
    --platform linux/amd64 \
    -f aws_batch/training/Dockerfile \
    -t ${ECR_REPOSITORY}:${IMAGE_TAG} \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Authenticate Docker to ECR
echo "Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} --profile memverge | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

if [ $? -ne 0 ]; then
    echo "❌ ECR authentication failed"
    exit 1
fi

# Tag the image for ECR
echo "Tagging image for ECR..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

if [ $? -ne 0 ]; then
    echo "❌ Failed to push image to ECR"
    exit 1
fi

echo "✅ Image pushed successfully to ECR"
echo ""
echo "Image URI: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
echo ""
echo "You can now submit a job using:"
echo "  ./submit_memverge_test.sh"