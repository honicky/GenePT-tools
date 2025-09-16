# AWS Batch GPU Training with MemVerge

This guide covers setting up and using AWS Batch for GPU-accelerated training with MemVerge checkpoint/restore capabilities.

## Architecture Overview

- **GPU Compute**: g5 instances with NVIDIA A10G GPUs
- **Data Strategy**: EBS snapshot with pre-populated training data (zero download time)
- **Model Tracking**: WandB for metrics and model artifacts
- **Cost Optimization**: Spot instances with checkpoint/restore via MemVerge

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Creating and Submitting Jobs](#creating-and-submitting-jobs)
- [Monitoring Jobs](#monitoring-jobs)
- [Setting Up New MemVerge Queues](#setting-up-new-memverge-queues)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- AWS CLI v2 installed and configured
- Docker installed for building container images
- Python 3.10+ for local testing
- jq for JSON processing (optional but helpful)

### AWS Permissions
Your AWS user/role needs permissions for:
- **AWS Batch**: Create/manage compute environments, job queues, job definitions
- **EC2**: Launch instances, create/manage launch templates, EBS snapshots
- **ECR**: Push/pull Docker images
- **IAM**: Create/manage roles and instance profiles
- **CloudWatch Logs**: Read log streams
- **Secrets Manager**: Store/retrieve secrets (for WandB API keys)
- **S3**: Read configuration files and training data

### Required IAM Roles

1. **Batch Service Role** (`aws-batch-service-role`):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "batch.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```
Attach policy: `arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole`

2. **ECS Task Execution Role** (`genept-training-execution-role`):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```
Attach policy: `arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy`

3. **ECS Task Role** (`genept-training-job-role`):
Custom policy for S3, Secrets Manager, and CloudWatch access.

## Environment Setup

### 1. Create `.env` File
```bash
cd aws_batch/training
cp .env.example .env
```

Edit `.env` with your values:
```bash
# AWS Configuration
AWS_ACCOUNT_ID=971422677163
AWS_REGION=us-west-2
AWS_PROFILE=memverge

# ECR Repository
ECR_REPOSITORY=miratyper-training

# S3 Buckets
CONFIG_BUCKET=miratyper-training-configs
OUTPUT_BUCKET=miratyper-training-outputs
DATA_BUCKET=pythiomicsdata

# EBS Snapshot with training data
DATA_SNAPSHOT_ID=snap-09a79b0190b9df72e

# Secrets
WANDB_SECRET_NAME=wandb-api-key

# Compute Environment
INSTANCE_TYPES=g5.2xlarge,g5.4xlarge,g5.8xlarge
MAX_VCPUS=256
```

### 2. Configure AWS CLI
```bash
aws configure --profile memverge
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Default region: us-west-2
# Default output format: json
```

### 3. Login to ECR
```bash
aws ecr get-login-password --region us-west-2 --profile memverge | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com
```

## Data Preparation

### Creating/Updating EBS Snapshots

1. **Launch a temporary EC2 instance** with sufficient storage:
```bash
aws ec2 run-instances \
  --image-id ami-0c94755bb95c71c0a \
  --instance-type t3.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvdb",
    "Ebs": {"VolumeSize": 600, "VolumeType": "gp3"}
  }]' \
  --profile memverge
```

2. **Mount and prepare the volume**:
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@instance-ip

# Format and mount the volume
sudo mkfs -t ext4 /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown ec2-user:ec2-user /data

# Create directory structure
mkdir -p /data/GenePT-tools/data/cellxgene_embeddings
```

3. **Copy training data** to the volume:
```bash
# Example: Copy from S3
aws s3 sync s3://your-data-source/ /data/GenePT-tools/data/ --profile your-profile

# Or copy from local
rsync -avz local-data/ ec2-user@instance-ip:/data/GenePT-tools/data/
```

4. **Create snapshot**:
```bash
# Get volume ID
VOLUME_ID=$(aws ec2 describe-instances \
  --instance-ids i-xxxxxxxxx \
  --query "Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName=='/dev/xvdb'].Ebs.VolumeId" \
  --output text \
  --profile memverge)

# Create snapshot
aws ec2 create-snapshot \
  --volume-id $VOLUME_ID \
  --description "GenePT training data $(date +%Y%m%d)" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=genept-training-data}]' \
  --profile memverge
```

5. **Update launch template** with new snapshot ID (see MemVerge setup section).

## Creating and Submitting Jobs

### 1. Build and Push Docker Image
```bash
# Build the Docker image
docker build --platform linux/amd64 -f aws_batch/training/Dockerfile -t miratyper-training .

# Tag for ECR
docker tag miratyper-training:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/miratyper-training:latest

# Push to ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/miratyper-training:latest
```

### 2. Register Job Definition
```bash
# For hyperparameter tuning jobs
aws batch register-job-definition \
  --cli-input-json file://aws_batch/training/job_definition_tuning.json \
  --region us-west-2 \
  --profile memverge
```

### 3. Submit a Job
```bash
# Basic submission
aws batch submit-job \
  --job-name "tuning-$(date +%Y%m%d-%H%M%S)" \
  --job-queue "miratyper-memverge-queue" \
  --job-definition "genept-tuning-job" \
  --parameters tuning_config=s3://miratyper-training-configs/tuning_config.yaml,wandb_project=my-project \
  --region us-west-2 \
  --profile memverge

# With custom parameters
aws batch submit-job \
  --job-name "custom-tuning-$(date +%Y%m%d-%H%M%S)" \
  --job-queue "miratyper-memverge-queue" \
  --job-definition "genept-tuning-job" \
  --parameters tuning_config=s3://miratyper-training-configs/custom_config.yaml,wandb_project=custom-project \
  --container-overrides '{"vcpus":8,"memory":61440}' \
  --region us-west-2 \
  --profile memverge
```

### 4. Job Definition Structure

Key components in `job_definition_tuning.json`:
```json
{
  "jobDefinitionName": "genept-tuning-job",
  "type": "container",
  "containerProperties": {
    "image": "${AWS_ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/miratyper-training:latest",
    "vcpus": 8,
    "memory": 61440,
    "linuxParameters": {
      "sharedMemorySize": 8192  // Important: 8GB for PyTorch DataLoader workers
    },
    "jobRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/genept-training-job-role",
    "volumes": [
      {"name": "data", "host": {"sourcePath": "/data"}},
      {"name": "scratch", "host": {"sourcePath": "/scratch"}}
    ],
    "mountPoints": [
      {"sourceVolume": "data", "containerPath": "/data", "readOnly": true},
      {"sourceVolume": "scratch", "containerPath": "/scratch", "readOnly": false}
    ],
    "command": [
      "--tuning-config", "Ref::tuning_config",
      "--wandb-project", "Ref::wandb_project",
      "--local-data-dir", "/data/GenePT-tools/data/cellxgene_embeddings/training_v1_shuffled",
      "--checkpoint-dir", "/tmp/checkpoints"
    ]
  }
}
```

**Important Notes:**
- **Shared Memory**: The `sharedMemorySize` parameter allocates memory for `/dev/shm`, crucial for PyTorch multi-worker data loading
- **Default**: Without this setting, containers only get 64MB of shared memory, causing "Bus error" crashes
- **Sizing**: Use 2-4GB per DataLoader worker (e.g., 8GB for 4 workers)

## Monitoring Jobs

### 1. Check Job Status
```bash
# Get job status
aws batch describe-jobs \
  --jobs JOB_ID \
  --region us-west-2 \
  --profile memverge \
  --query "jobs[0].{Status:status,StartedAt:startedAt,StatusReason:statusReason}"
```

### 2. View Logs via AWS Console
- Navigate to: https://us-west-2.console.aws.amazon.com/batch/
- Click on your job ID
- Select the "Logs" tab

### 3. Stream Logs via CLI
```bash
# Get log stream name
LOG_STREAM=$(aws batch describe-jobs \
  --jobs JOB_ID \
  --region us-west-2 \
  --profile memverge \
  --query "jobs[0].container.logStreamName" \
  --output text)

# Tail logs
aws logs tail /aws/batch/job --follow \
  --log-stream-names $LOG_STREAM \
  --profile memverge \
  --region us-west-2
```

### 4. Use Monitoring Script
```bash
./scripts/monitor_job.sh JOB_ID
```

### 5. WandB Dashboard
- Visit https://wandb.ai
- Navigate to your project (e.g., `memverge-tuning-test`)
- View real-time training metrics and hyperparameter comparisons

## Setting Up New MemVerge Queues

### 1. Create Launch Template
```bash
# Create launch template with user data for MemVerge
aws ec2 create-launch-template \
  --launch-template-name "genept-memverge-template" \
  --launch-template-data '{
    "ImageId": "ami-0c94755bb95c71c0a",
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/xvda",
        "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
      },
      {
        "DeviceName": "/dev/xvdb",
        "Ebs": {"SnapshotId": "snap-09a79b0190b9df72e", "VolumeType": "gp3"}
      }
    ],
    "UserData": "'"$(base64 -w 0 user_data.sh)"'"
  }' \
  --region us-west-2 \
  --profile memverge
```

### 2. Create Compute Environment
```bash
aws batch create-compute-environment \
  --compute-environment-name "my-memverge-compute" \
  --type MANAGED \
  --state ENABLED \
  --service-role "arn:aws:iam::${AWS_ACCOUNT_ID}:role/aws-batch-service-role" \
  --compute-resources '{
    "type": "EC2",
    "minvCpus": 0,
    "maxvCpus": 256,
    "desiredvCpus": 0,
    "instanceTypes": ["g5.2xlarge", "g5.4xlarge", "g5.8xlarge"],
    "subnets": ["subnet-xxxxxx", "subnet-yyyyyy"],
    "securityGroupIds": ["sg-xxxxxx"],
    "instanceRole": "arn:aws:iam::${AWS_ACCOUNT_ID}:instance-profile/ecsInstanceRole",
    "launchTemplate": {
      "launchTemplateName": "genept-memverge-template",
      "version": "$Latest"
    },
    "tags": {
      "Name": "memverge-instance",
      "Environment": "training"
    }
  }' \
  --region us-west-2 \
  --profile memverge
```

### 3. Create Job Queue
```bash
aws batch create-job-queue \
  --job-queue-name "my-memverge-queue" \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=my-memverge-compute \
  --region us-west-2 \
  --profile memverge
```

### 4. Instance Type Selection

| Instance Type | GPUs | GPU Memory | vCPUs | RAM | Use Case |
|--------------|------|------------|-------|-----|----------|
| g5.2xlarge | 1x A10G | 24 GB | 8 | 32 GB | Small models, testing |
| g5.4xlarge | 1x A10G | 24 GB | 16 | 64 GB | Medium training jobs |
| g5.8xlarge | 1x A10G | 24 GB | 32 | 128 GB | Large batch processing |
| g5.12xlarge | 4x A10G | 96 GB | 48 | 192 GB | Multi-GPU training |

### 5. Customize for Specific Workloads

#### For Memory-Intensive Tasks:
- Use larger instance types (g5.8xlarge+)
- Increase container memory limits
- Mount additional EBS volumes

#### For Checkpoint/Restore:
- Configure frequent checkpointing in training config
- Ensure `/scratch` is writable for checkpoint storage
- Set appropriate checkpoint intervals (e.g., every 50 batches)

## Troubleshooting

### Common Errors and Fixes

#### 1. S3 Access Denied
**Error**: `AccessDenied when calling the ListObjectsV2 operation`

**Cause**: Cross-account S3 bucket access or missing permissions

**Fix**:
- If bucket is in different account, copy data to your account's bucket
- Or update IAM role with explicit S3 permissions:
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::bucket-name",
    "arn:aws:s3:::bucket-name/*"
  ]
}
```

#### 2. Read-only Filesystem
**Error**: `OSError: [Errno 30] Read-only file system: '/data'`

**Cause**: Trying to write to read-only mounted volume

**Fix**:
- Use `/tmp` or `/scratch` for writable storage
- Update mount points in job definition:
```json
"mountPoints": [
  {"sourceVolume": "data", "containerPath": "/data", "readOnly": true},
  {"sourceVolume": "scratch", "containerPath": "/scratch", "readOnly": false}
]
```

#### 3. Profile Not Found
**Error**: `ProfileNotFound: The config profile (xcellerate) could not be found`

**Cause**: Hardcoded AWS profile in code

**Fix**:
- Pass `--aws-profile none` to use ECS task role credentials
- Or set profile to empty string in config

#### 4. Invalid Compute Environment
**Error**: `CLIENT_ERROR - subnet-xxxxx does not exist`

**Cause**: Incorrect subnet IDs or missing permissions

**Fix**:
```bash
# List available subnets
aws ec2 describe-subnets --region us-west-2 --profile memverge

# List security groups
aws ec2 describe-security-groups --region us-west-2 --profile memverge
```

#### 5. Missing Instance Profile
**Error**: `Instance profile arn:aws:iam::xxx:instance-profile/xxx does not exist`

**Fix**:
```bash
# List available instance profiles
aws iam list-instance-profiles --profile memverge

# Use the Batch-managed instance profile
```

#### 6. DataLoader Shared Memory Error
**Error**: `DataLoader worker (pid X) is killed by signal: Bus error`

**Cause**: PyTorch DataLoader workers run out of shared memory in container

**Fix**:
Add `linuxParameters` to job definition to increase shared memory:
```json
"containerProperties": {
  "linuxParameters": {
    "sharedMemorySize": 8192  // 8GB of shared memory
  },
  // ... rest of container properties
}
```

Or use container overrides when submitting:
```bash
aws batch submit-job \
  --job-name "my-job" \
  --job-queue "queue-name" \
  --job-definition "job-def" \
  --container-overrides '{"linuxParameters":{"sharedMemorySize":16384}}' \
  --region us-west-2 \
  --profile memverge
```

Alternative: Set `num_workers: 0` in config to disable multiprocessing (slower but avoids shared memory issues)

### Debugging Tools

#### 1. Filesystem Debug Script
Use to explore container filesystem and find data:
```bash
# Create debug job definition
aws batch register-job-definition \
  --cli-input-json file://job_definition_debug.json \
  --region us-west-2 \
  --profile memverge

# Submit debug job
aws batch submit-job \
  --job-name "debug-filesystem" \
  --job-queue "miratyper-memverge-queue" \
  --job-definition "genept-debug-filesystem" \
  --region us-west-2 \
  --profile memverge
```

#### 2. Check Volume Mounts
Verify volumes are correctly mounted:
```bash
# In debug container or via job overrides
df -h
mount | grep -E "(data|scratch)"
ls -la /data /scratch
```

#### 3. Verify Data Paths
Common data locations to check:
```bash
# Check for training data (case sensitive!)
ls -la /data/GenePT-tools/data/cellxgene_embeddings/
ls -la /data/GenePT-tools/data/  # Note: lowercase 't' in tools

# Check for test data
ls -la /data/GenePT-tools/data/cellxgene_embeddings/test_v1/

# Find any .pt or .parquet files
find /data -name "*.pt" -o -name "*.parquet" | head -20
```

#### 4. Test S3 Access
Verify the container can access S3:
```bash
# Test with AWS CLI in container
aws s3 ls s3://miratyper-training-configs/ --region us-west-2

# Test with boto3
python -c "import boto3; s3 = boto3.client('s3'); print(s3.list_buckets())"
```

## Tips

1. **Always use lowercase paths**: Linux is case-sensitive (`GenePT-tools` not `GenePT-Tools`)
2. **Check instance status**: Ensure compute environment is VALID before submitting jobs
3. **Monitor costs**: Use spot instances when possible for cost savings
4. **Checkpoint frequently**: For resilience against spot interruptions
5. **Use debug jobs**: When in doubt, submit a debug job to explore the environment