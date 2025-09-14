# PyTorch base image with CUDA support for A10G GPUs
# Use --platform to ensure AMD64 for AWS deployment
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements for AWS Batch
COPY aws_batch/training/requirements_batch.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_batch.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables for data paths (will be mounted from EBS)
ENV DATA_ROOT=/data/GenePT-Tools/data
ENV CHECKPOINT_DIR=/scratch/checkpoints
ENV OUTPUT_DIR=/scratch/outputs
ENV WANDB_DIR=/scratch/wandb
ENV PYTHONUNBUFFERED=1

# Verify GPU support
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Default command (can be overridden at runtime)
ENTRYPOINT ["python", "scripts/train_cellxgene_mlp.py"]