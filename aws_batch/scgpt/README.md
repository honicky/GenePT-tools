# scGPT Embedding Generation Pipeline

This pipeline generates scGPT embeddings for single-cell RNA-seq data stored in H5AD format on S3.

## Overview

scGPT is a foundation model for single-cell biology that generates 512-dimensional embeddings from gene expression data. This pipeline:

1. Downloads H5AD files from S3
2. Preprocesses the expression data
3. Generates embeddings using pre-trained scGPT model
4. Saves embeddings in both Parquet and PyTorch tensor formats
5. Uploads results back to S3

## Prerequisites

1. AWS Batch compute environment with GPU support (p3 or g4 instances)
2. Pre-trained scGPT model uploaded to S3
3. H5AD files accessible from S3
4. ECR repository for Docker image

## Setup

### 1. Build and Push Docker Image

```bash
# Build the Docker image
cd docker/scgpt
docker build -t scgpt-embeddings .

# Tag for ECR
docker tag scgpt-embeddings:latest 971422677163.dkr.ecr.us-west-2.amazonaws.com/scgpt-embeddings:latest

# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 971422677163.dkr.ecr.us-west-2.amazonaws.com
docker push 971422677163.dkr.ecr.us-west-2.amazonaws.com/scgpt-embeddings:latest
```

### 2. Upload Pre-trained Model to S3

```bash
# Download scGPT model (if not already available)
# Visit https://github.com/bowang-lab/scGPT for model links

# Upload to S3
aws s3 cp --recursive ./scgpt_whole_human_model s3://pythiomicsdata/models/scgpt/whole_human/
```

### 3. Create Job Definition

```bash
cd aws_batch/scgpt
aws batch register-job-definition --cli-input-json file://job_definition.json
```

### 4. Prepare File List

Create a JSON file with S3 paths to H5AD files:

```json
[
  "s3://pythiomicsdata/cellxgene_v2/h5ad/file1.h5ad",
  "s3://pythiomicsdata/cellxgene_v2/h5ad/file2.h5ad",
  ...
]
```

## Usage

### Submit Jobs to AWS Batch

```bash
python aws_batch/scgpt/submit_job.py \
  --file-list h5ad_files.json \
  --job-queue gpu-queue \
  --files-per-job 10 \
  --output-bucket pythiomicsdata \
  --output-prefix cellxgene_v2/scgpt_embeddings_v2
```

### Monitor Jobs

```bash
# List running jobs
aws batch list-jobs --job-queue gpu-queue --job-status RUNNING

# Get job details
aws batch describe-jobs --jobs <job-id>

# Check CloudWatch logs
aws logs tail /aws/batch/scgpt-embeddings --follow
```

### Process Files Locally

For testing or small datasets:

```bash
python scripts/generate_scgpt_embeddings.py \
  --input-dir /path/to/h5ad/files \
  --model-path /path/to/scgpt/model \
  --output-dir /path/to/output \
  --batch-size 256 \
  --device cuda
```

## Output Format

### Parquet Files
- `{filename}_scgpt_embeddings.parquet`
- Columns:
  - `cell_id`: Unique cell identifier
  - `scgpt_embedding`: 512-dim array
  - Additional metadata columns (cell_type, tissue, etc.)

### PyTorch Tensor Files
- `{filename}_scgpt_embeddings.pt`
- Contains:
  - `embeddings`: FloatTensor of shape (n_cells, 512)
  - `cell_ids`: List of cell identifiers
  - `metadata`: Dictionary with processing info

## Configuration

### Job Definition Parameters

- `vcpus`: Number of vCPUs (default: 8)
- `memory`: Memory in MB (default: 32768)
- `GPU`: Number of GPUs (default: 1)
- `sharedMemorySize`: Shared memory for PyTorch (default: 8192)

### Processing Parameters

- `batch_size`: Number of cells per batch (default: 256)
- `max_genes`: Maximum genes to use (default: 3000)
- `device`: Computation device (cuda/cpu)

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` parameter
- Use instances with more GPU memory
- Process fewer files per job

### Model Loading Issues
- Verify model files are complete on S3
- Check vocabulary file exists (`vocab.json`)
- Ensure model checkpoint is compatible

### Slow Processing
- Increase `batch_size` if memory allows
- Use more powerful GPU instances (p3.2xlarge)
- Process files in parallel with multiple jobs

## Cost Optimization

1. Use Spot instances for Batch compute environment
2. Process multiple small files per job to reduce overhead
3. Store intermediate results on EFS for faster access
4. Use g4dn instances for cost-effective GPU compute

## Model Variants

Available pre-trained models:
- `whole_human`: 33M cells, general purpose
- `brain`: 13.2M brain cells
- `blood_bone_marrow`: 10.3M blood/bone marrow cells

Select model based on your dataset characteristics.