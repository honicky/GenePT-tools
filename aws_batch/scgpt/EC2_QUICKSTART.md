# Running scGPT on EC2 - Quick Start Guide

## Prerequisites

- EC2 instance with GPU (g4dn.xlarge or larger)
- Docker and nvidia-docker installed
- AWS CLI configured with appropriate credentials
- At least 50GB of free disk space

## Step 1: Connect to EC2 Instance

```bash
ssh -i your-key.pem ec2-user@your-ec2-instance.com
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/honicky/GenePT-tools.git
cd GenePT-tools
git checkout feature/scgpt-embedding-generation
```

## Step 3: Build Docker Image (First Time Only)

```bash
# Build the Docker image using the build script
chmod +x docker/scgpt/build.sh
./docker/scgpt/build.sh

# Or manually copy files and build
cp scripts/generate_scgpt_embeddings.py docker/scgpt/
cp scripts/scgpt_wrapper.py docker/scgpt/
cd docker/scgpt
docker build -t scgpt-embeddings .
rm generate_scgpt_embeddings.py scgpt_wrapper.py

# Tag for ECR (optional, if you want to push to registry)
docker tag scgpt-embeddings:latest 971422677163.dkr.ecr.us-west-2.amazonaws.com/scgpt-embeddings:latest
```

## Step 4: Prepare Test Data

Create a small file list for testing:

```bash
# Create data directory
mkdir -p /data/scgpt

# Create a test file list with 3 H5AD files
cat > /data/scgpt/test_files.json << 'EOF'
[
  "s3://pythiomicsdata/cellxgene_v2/h5ad/file1.h5ad",
  "s3://pythiomicsdata/cellxgene_v2/h5ad/file2.h5ad",
  "s3://pythiomicsdata/cellxgene_v2/h5ad/file3.h5ad"
]
EOF
```

## Step 5: Run with Docker (Simple Method)

```bash
# Make the script executable
chmod +x aws_batch/scgpt/run_on_ec2.sh

# Run the script
./aws_batch/scgpt/run_on_ec2.sh
```

## Step 6: Run with Docker (Manual Method)

If you prefer to run manually or need custom settings:

```bash
# Set up directories
mkdir -p /data/scgpt/model /data/scgpt/output
chmod 777 /data/scgpt/*

# Download model (one time)
aws s3 sync s3://pythiomicsdata/models/scgpt/whole_human/ /data/scgpt/model/

# Run container
docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v /data/scgpt:/data \
    -v ~/.aws:/root/.aws:ro \
    -e AWS_DEFAULT_REGION=us-west-2 \
    scgpt-embeddings \
    python /app/scgpt_wrapper.py \
        --input-list /data/test_files.json \
        --model-path /data/model \
        --output-bucket pythiomicsdata \
        --output-prefix cellxgene_v2/scgpt_embeddings_test \
        --batch-size 256 \
        --device cuda
```

## Step 7: Monitor Progress

```bash
# Watch Docker logs in real-time
docker logs -f <container-id>

# Check GPU usage
nvidia-smi -l 1

# Check output files
ls -la /data/scgpt/output/
```

## Step 8: Verify Results

```bash
# Check S3 output
aws s3 ls s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_test/ --recursive

# Download a sample output file
aws s3 cp s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_test/file1_scgpt_embeddings.parquet .

# Check with Python
python -c "
import pandas as pd
df = pd.read_parquet('file1_scgpt_embeddings.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Embedding dim: {len(df.iloc[0][\"scgpt_embedding\"])}')
"
```

## Processing Larger File Lists

For processing more files, create a larger JSON list:

```bash
# Example: Get first 10 H5AD files from S3
aws s3 ls s3://pythiomicsdata/cellxgene_v2/h5ad/ \
    | grep '\.h5ad$' \
    | head -10 \
    | awk '{print "\"s3://pythiomicsdata/cellxgene_v2/h5ad/"$4"\","}' \
    | sed '$ s/,$//' \
    | (echo '['; cat; echo ']') \
    > /data/scgpt/batch_files.json

# Run with the larger list
docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v /data/scgpt:/data \
    -v ~/.aws:/root/.aws:ro \
    -e AWS_DEFAULT_REGION=us-west-2 \
    scgpt-embeddings \
    python /app/scgpt_wrapper.py \
        --input-list /data/batch_files.json \
        --model-path /data/model \
        --output-bucket pythiomicsdata \
        --output-prefix cellxgene_v2/scgpt_embeddings_batch \
        --batch-size 256 \
        --device cuda
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 128  # or even 64
```

### CUDA Not Available
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi

# If not working, install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Model Not Found
```bash
# Verify model files
ls -la /data/scgpt/model/
# Should contain: best_model.pt, vocab.json, config.json

# Re-download if needed
aws s3 sync s3://pythiomicsdata/models/scgpt/whole_human/ /data/scgpt/model/ --delete
```

### Permission Denied
```bash
# Fix permissions
sudo chown -R $USER:$USER /data/scgpt
chmod -R 755 /data/scgpt
```

## Performance Tips

1. **Use Local NVMe Storage**: Copy frequently accessed files to instance storage
   ```bash
   sudo mkdir -p /mnt/nvme/scgpt
   sudo chown $USER:$USER /mnt/nvme/scgpt
   ln -s /mnt/nvme/scgpt /data/scgpt
   ```

2. **Process in Parallel**: Run multiple containers with different file lists
   ```bash
   # Split file list
   split -l 5 all_files.json chunk_
   
   # Run multiple containers
   for chunk in chunk_*; do
     docker run -d --gpus all ... --input-list /data/$chunk ...
   done
   ```

3. **Cache Model in Memory**: Keep container running between batches
   ```bash
   docker run -it --gpus all ... /bin/bash
   # Then run multiple batches from inside container
   ```

## Expected Processing Times

- Model loading: ~30 seconds
- Per H5AD file (10k cells): ~2-3 minutes
- Per H5AD file (100k cells): ~15-20 minutes
- Batch of 10 files: ~30-60 minutes

## Cost Estimates (g4dn.xlarge)

- Instance: ~$0.526/hour
- Processing 100 files: ~10 hours = ~$5.26
- Storage (EBS): ~$0.10/GB/month