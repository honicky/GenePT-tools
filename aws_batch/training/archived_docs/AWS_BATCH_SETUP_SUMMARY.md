# AWS Batch GPU Training Setup Summary

## What We Built

We successfully set up a complete AWS Batch infrastructure for GPU-accelerated machine learning training with the following components:

### Core Infrastructure

1. **Compute Environment**: `miratyper-gpu-training-spot`
   - Uses NVIDIA A10G GPU instances (g5.xlarge, g5.2xlarge, g5.4xlarge)
   - Configured for spot instances with 80% bid percentage (60% cost savings)
   - Auto-scales from 0 to 256 vCPUs based on job demand
   - Automatically provisions instances when jobs are submitted, scales to zero when idle

2. **Job Queue**: `miratyper-training-queue`
   - Connected to the GPU compute environment
   - Manages job scheduling and resource allocation
   - Priority-based job execution

3. **Job Definition**: `genept-training-job`
   - Specifies container configuration with PyTorch 2.0.1 + CUDA 11.7
   - Allocates 1 GPU, 8 vCPUs, and 30GB memory per job
   - Includes retry strategy (2 attempts) and 12-hour timeout
   - Parameterized for flexible training configurations

### Data and Storage

1. **EBS Snapshot Integration**: 
   - Pre-populated snapshot `snap-0b1c573caa4318e2f` containing 200GB+ training data
   - Automatically attached to instances at `/data` (read-only)
   - Zero data transfer time - data is immediately available

2. **S3 Buckets**:
   - `miratyper-training-configs`: Stores training configuration files
   - `miratyper-training-outputs`: Stores model checkpoints and results

3. **Container Registry**:
   - ECR repository: `971422677163.dkr.ecr.us-west-2.amazonaws.com/miratyper-training`
   - Contains custom Docker image with all ML dependencies

### Monitoring and Tracking

1. **CloudWatch Logs**: `/aws/batch/miratyper-training`
   - Real-time training progress monitoring
   - 30-day retention policy

2. **WandB Integration**:
   - API key stored securely in AWS Secrets Manager
   - Automatic experiment tracking and metrics visualization

### Security Configuration

1. **IAM Roles**:
   - `genept-training-execution-role`: Allows ECS to pull images and write logs
   - `genept-training-job-role`: Grants containers access to S3 and Secrets Manager
   - `aws-batch-service-role`: Enables Batch to manage compute resources
   - `aws-batch-spot-fleet-role`: Allows spot fleet management

2. **Launch Template**: `genept-training-gpu-template`
   - Configures GPU instances with ECS-optimized Amazon Linux 2023 AMI
   - Attaches EBS volumes (100GB root + data snapshot)
   - Sets up scratch directories for temporary files

## How It Works

1. **Job Submission**: 
   - Submit a job to the queue with training parameters
   - Batch automatically provisions a spot GPU instance if none available
   - Instance boots with pre-attached training data from EBS snapshot

2. **Execution**:
   - Container starts with mounted data volume at `/data`
   - Training script runs with GPU acceleration
   - Progress logged to CloudWatch and WandB
   - Checkpoints saved to scratch space and optionally to S3

3. **Completion**:
   - Results uploaded to S3 output bucket
   - Instance automatically terminated if no pending jobs
   - Logs preserved in CloudWatch for analysis

## Key Design Decisions

1. **Spot Instances**: Chose spot over on-demand for 60% cost reduction, with retry logic for interruption handling

2. **EBS Snapshots**: Instead of downloading data each time, instances start with data pre-attached, saving hours of transfer time

3. **Auto-scaling to Zero**: Compute environment scales down completely when idle, eliminating standby costs

4. **WandB over S3**: Using WandB for model tracking instead of complex S3 versioning, providing better experiment management

5. **Multiple Instance Types**: Allows Batch to optimize resource allocation based on availability and cost

## Resources Created

| Resource Type | Name/ID | Purpose |
|--------------|---------|---------|
| ECR Repository | miratyper-training | Container image storage |
| Compute Environment | miratyper-gpu-training-spot | GPU instance management |
| Job Queue | miratyper-training-queue | Job scheduling |
| Job Definition | genept-training-job:1 | Container configuration |
| S3 Bucket | miratyper-training-configs | Configuration storage |
| S3 Bucket | miratyper-training-outputs | Output storage |
| Launch Template | genept-training-gpu-template | Instance configuration |
| CloudWatch Log Group | /aws/batch/miratyper-training | Training logs |
| Secret | wandb_api_key | WandB authentication |
| IAM Roles | 4 roles created | Service permissions |

## Current Status

✅ All components successfully deployed and verified:
- Compute Environment: **VALID**
- Job Queue: **VALID** 
- Job Definition: **ACTIVE** (revision 1)
- Docker Image: Pushed to ECR (digest: sha256:94faf4750b19b51...)
- Infrastructure ready for training job submission

## Usage

Submit a training job:
```bash
aws batch submit-job \
    --job-name experiment-001 \
    --job-queue miratyper-training-queue \
    --job-definition genept-training-job \
    --parameters epochs=100,batch_size=1024 \
    --region us-west-2 \
    --profile memverge
```

Or using the Python script:
```bash
python submit_job.py --job-name experiment-001 --epochs 100
```

The system will automatically:
1. Provision a GPU spot instance
2. Attach the training data
3. Run the training container
4. Save results to S3 and WandB
5. Terminate the instance when complete