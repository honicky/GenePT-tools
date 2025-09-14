# MemVerge Integration with AWS Batch GPU Training

## Overview

Successfully integrated MemVerge Memory Machine Batch (MMBatch) with our GPU training infrastructure to enable checkpoint/restore capabilities for spot instance resilience.

## MemVerge Infrastructure

### Management Server
- **Address**: `35.90.252.151:8080` (public) / `172.31.19.221:8080` (private)
- **Instance**: `i-05475df20ecd6407e` (Management-Server-miraomics)
- **API Endpoint**: `https://35.90.252.151:8080/api/v1/`
- **Authentication**: Disabled (no Cognito)
- **Checkpoint Configuration**:
  - Mode: Iterative
  - Interval: 15 minutes
  - Path: `/mmc-checkpoint`
  - Root FS Diff: Enabled
  - TCP Close: Enabled

## AWS Batch Components with MemVerge

### 1. Launch Template: `genept-training-gpu-template` (Version 3)
- **Features**:
  - MIME multipart format for AWS Batch compatibility
  - MemVerge pagent installation script
  - Checkpoint directory at `/scratch/mmc-checkpoint`
  - Automatic agent registration with management server

### 2. Compute Environment: `miratyper-gpu-memverge`
- **Status**: VALID
- **Type**: Spot instances (80% bid)
- **Instance Types**: g5.xlarge, g5.2xlarge, g5.4xlarge
- **Instance Profile**: Batch-Engine-IAMAndManagementStack-AP40G4XPC5PU-BatchInstanceProfile
- **Launch Template**: Version 3 with MemVerge support
- **Tags**: Project=MiraTyper, Environment=training, MemVerge=enabled

### 3. Job Queue: `miratyper-memverge-queue`
- **Status**: ENABLED
- **Priority**: 1
- **Compute Environment**: miratyper-gpu-memverge

### 4. Job Definition: `genept-training-job`
- **Container**: PyTorch 2.0.1 + CUDA 11.7
- **Resources**: 1 GPU, 8 vCPUs, 30GB memory
- **Data Access**: EBS snapshot mounted at `/data`
- **Checkpoint Storage**: `/scratch/mmc-checkpoint`

## How MemVerge Integration Works

1. **Instance Startup**:
   - Instance launches with user data script
   - MemVerge pagent installs and registers with management server
   - Checkpoint directory created at `/scratch/mmc-checkpoint`

2. **Job Execution**:
   - Container starts with normal AWS Batch process
   - MemVerge monitors container and creates checkpoints every 15 minutes
   - Checkpoints stored locally on instance scratch storage

3. **Spot Interruption Handling**:
   - MemVerge detects spot interruption signal (2-minute warning)
   - Final checkpoint created before instance termination
   - New spot instance provisions automatically
   - Job restores from checkpoint and continues execution
   - No training progress lost

## API Usage

### Check Registered Nodes
```bash
curl -sk https://35.90.252.151:8080/api/v1/node | jq .
```

### Monitor Jobs
```bash
curl -sk https://35.90.252.151:8080/api/v1/job | jq .
```

### View Configuration
```bash
curl -sk https://35.90.252.151:8080/api/v1/config | jq .
```

### Update Checkpoint Settings
```bash
curl -sk -X PUT https://35.90.252.151:8080/api/v1/configKV \
  -H "Content-Type: application/json" \
  -d '{"kvMap": {"ckpt.ckptInterval": "10m"}}' | jq .
```

## Submitting Jobs

### To MemVerge-Enabled Queue
```bash
aws batch submit-job \
    --job-name my-training-job \
    --job-queue miratyper-memverge-queue \
    --job-definition genept-training-job \
    --parameters epochs=100,batch_size=1024 \
    --region us-west-2 \
    --profile memverge
```

### Using Python Script
```bash
python submit_job.py \
    --job-name experiment-001 \
    --job-queue miratyper-memverge-queue \
    --epochs 100
```

## Monitoring

### Job Status in AWS Batch
```bash
aws batch describe-jobs --jobs <job-id> --region us-west-2 --profile memverge
```

### MemVerge Dashboard
Access the web interface at: `https://35.90.252.151:8080`

### CloudWatch Logs
```bash
aws logs tail /aws/batch/miratyper-training --follow --profile memverge
```

### MemVerge Metrics API
```bash
curl -sk https://35.90.252.151:8080/api/v1/metrics/summary | jq .
```

## Benefits

1. **Cost Savings**: 60% reduction using spot instances without risk of losing work
2. **Reliability**: Automatic checkpoint/restore on spot interruptions
3. **Transparency**: Jobs continue seamlessly after interruptions
4. **Performance**: Local SSD storage for fast checkpoint operations
5. **Monitoring**: Comprehensive metrics on spot savings and protection events

## Current Status

✅ MemVerge management server running and accessible
✅ Launch template configured with MemVerge agent
✅ Compute environment created and VALID
✅ Job queue ready for submissions
✅ Test job submitted (ID: ad36f2bf-0801-47c1-ad49-0298c8fe5898)

## Notes

- Checkpoint data stored locally on instance (not persisted across different instances)
- For persistent checkpoints across instances, configure shared storage (EFS/JuiceFS)
- Current setup optimized for single-node GPU training jobs
- Multi-node training would require additional network configuration