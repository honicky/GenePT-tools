#!/bin/bash
# Script to run scGPT embedding generation on EC2 instance

set -e

echo "=== scGPT Embedding Generation on EC2 ==="
echo

# Configuration
ECR_REGISTRY="971422677163.dkr.ecr.us-west-2.amazonaws.com"
IMAGE_NAME="scgpt-embeddings"
IMAGE_TAG="latest"
CONTAINER_NAME="scgpt-embed-run"

# S3 paths
MODEL_PATH="s3://pythiomicsdata/models/scgpt/whole_human"
OUTPUT_BUCKET="pythiomicsdata"
OUTPUT_PREFIX="cellxgene_v2/scgpt_embeddings_test"

# Local directories on EC2
DATA_DIR="/data/scgpt"
MODEL_DIR="/data/scgpt/model"
OUTPUT_DIR="/data/scgpt/output"

echo "1. Setting up directories..."
sudo mkdir -p $DATA_DIR $MODEL_DIR $OUTPUT_DIR
sudo chmod 777 $DATA_DIR $MODEL_DIR $OUTPUT_DIR

echo "2. Authenticating with ECR..."
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REGISTRY

echo "3. Pulling Docker image..."
docker pull $ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG

echo "4. Creating test file list..."
cat > $DATA_DIR/test_files.json << 'EOF'
[
  "s3://pythiomicsdata/cellxgene_v2/h5ad/00476f9f-ebc1-4b72-b541-32f912ce36ea.h5ad",
  "s3://pythiomicsdata/cellxgene_v2/h5ad/0087cde2-967d-4f7c-8e6e-40e4c9ad1891.h5ad",
  "s3://pythiomicsdata/cellxgene_v2/h5ad/00e5dedd-b9b7-43be-8c28-b0e5c6414a62.h5ad"
]
EOF

echo "5. Downloading model from S3 (if not cached)..."
if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
    echo "   Downloading model files..."
    aws s3 sync $MODEL_PATH/ $MODEL_DIR/ --exclude "*" --include "*.pt" --include "*.json"
else
    echo "   Model already cached"
fi

echo "6. Running Docker container..."
docker run --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size=8g \
    -v $DATA_DIR:/data \
    -v $MODEL_DIR:/models \
    -v $OUTPUT_DIR:/output \
    -v ~/.aws:/root/.aws:ro \
    -e AWS_DEFAULT_REGION=us-west-2 \
    -e CUDA_VISIBLE_DEVICES=0 \
    $ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG \
    python /app/scgpt_wrapper.py \
        --input-list /data/test_files.json \
        --model-path /models \
        --output-bucket $OUTPUT_BUCKET \
        --output-prefix $OUTPUT_PREFIX \
        --batch-size 256 \
        --device cuda

echo "7. Checking output..."
echo "   Local output files:"
ls -la $OUTPUT_DIR/

echo "   S3 output location:"
aws s3 ls s3://$OUTPUT_BUCKET/$OUTPUT_PREFIX/ --recursive

echo
echo "=== Processing complete ==="