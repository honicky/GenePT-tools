# GPU Training Infrastructure Specification for GenePT

## Executive Summary
This specification defines a GPU-enabled AWS Batch infrastructure optimized for PyTorch-based ML training with pre-populated EBS data volumes and cost-effective spot instances.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Batch Job Queue                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             GPU Compute Environment (Spot)                  │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │   g5.xlarge   │  g5.2xlarge  │  g5.4xlarge         │     │
│  │   1x A10G GPU │  1x A10G GPU │  1x A10G GPU        │     │
│  │   4 vCPUs     │  8 vCPUs     │  16 vCPUs           │     │
│  │   16 GB RAM   │  32 GB RAM   │  64 GB RAM          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Amazon Linux 2023 GPU-Optimized AMI        │     │
│  │         PyTorch 2.0+ | CUDA 11.8 | cuDNN 8         │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                              │
┌───────▼────────┐           ┌─────────▼──────────┐
│  EBS Snapshot  │           │   Instance Store   │
│  (Training Data)│           │   (Scratch/Temp)   │
│  /data (RO)    │           │   /scratch (RW)    │
└────────────────┘           └────────────────────┘
```

## Component Specifications

### 1. Compute Environment

#### Instance Types (G5 Family)
```yaml
Instance Types:
  - g5.xlarge:
      GPU: 1x NVIDIA A10G (24GB VRAM)
      vCPUs: 4
      Memory: 16 GB
      Network: Up to 10 Gbps
      Storage: 250 GB NVMe SSD
      Use Case: Small models, testing
      Spot Price: ~$0.42/hour (vs $1.006 on-demand)
  
  - g5.2xlarge:
      GPU: 1x NVIDIA A10G (24GB VRAM)
      vCPUs: 8
      Memory: 32 GB
      Network: Up to 10 Gbps
      Storage: 450 GB NVMe SSD
      Use Case: Standard training jobs
      Spot Price: ~$0.50/hour (vs $1.212 on-demand)
  
  - g5.4xlarge:
      GPU: 1x NVIDIA A10G (24GB VRAM)
      vCPUs: 16
      Memory: 64 GB
      Network: Up to 25 Gbps
      Storage: 600 GB NVMe SSD
      Use Case: Large batch sizes, memory-intensive
      Spot Price: ~$0.65/hour (vs $1.624 on-demand)
  
  - g5.8xlarge:
      GPU: 1x NVIDIA A10G (24GB VRAM)
      vCPUs: 32
      Memory: 128 GB
      Network: 25 Gbps
      Storage: 900 GB NVMe SSD
      Use Case: Very large models
      Spot Price: ~$1.00/hour (vs $2.448 on-demand)
```

#### Compute Environment Configuration
```json
{
  "computeEnvironmentName": "genept-gpu-training-spot",
  "type": "MANAGED",
  "state": "ENABLED",
  "serviceRole": "arn:aws:iam::ACCOUNT_ID:role/aws-batch-service-role",
  "computeResources": {
    "type": "SPOT",
    "bidPercentage": 80,
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-batch-spot-fleet-role",
    "minvCpus": 0,
    "maxvCpus": 256,
    "desiredvCpus": 0,
    "instanceTypes": [
      "g5.xlarge",
      "g5.2xlarge",
      "g5.4xlarge",
      "g5.8xlarge"
    ],
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "instanceRole": "arn:aws:iam::ACCOUNT_ID:instance-profile/ecsInstanceRole",
    "launchTemplate": {
      "launchTemplateName": "genept-gpu-training-template",
      "version": "$Latest"
    },
    "tags": {
      "Project": "GenePT",
      "Environment": "Training",
      "Type": "GPU-Spot"
    }
  }
}
```

### 2. AMI Specification

#### Amazon Linux 2023 GPU-Optimized
```yaml
AMI Selection:
  Base: Amazon Linux 2023 ECS GPU-Optimized
  AMI Path: /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended
  
Pre-installed Software:
  - NVIDIA Driver: 535.104.12 or later
  - CUDA Toolkit: 11.8 or 12.1
  - cuDNN: 8.9.x
  - Docker: 24.x with NVIDIA Container Toolkit
  - ECS Agent: Latest with GPU support
  - Python: 3.11
  
PyTorch Installation (in container):
  - PyTorch: 2.0.1 or later
  - torchvision: 0.15.2
  - CUDA variant: cu118 or cu121
  
Key Features:
  - GPU device auto-detection
  - NVIDIA GPU metrics in CloudWatch
  - Optimized kernel for ML workloads
  - Enhanced networking (ENA enabled)
```

### 3. Storage Configuration

#### EBS Snapshot Attachment
```yaml
Data Volume:
  Snapshot ID: snap-0b1c573caa4318e2f
  Mount Point: /data
  Access: Read-Only
  Type: gp3
  Size: Determined by snapshot
  
Structure:
  /data/
  └── GenePT-Tools/
      └── data/
          ├── cellxgene_v2/
          │   ├── training_v1_shuffled/
          │   │   ├── train_data_0000.pt
          │   │   ├── train_data_0001.pt
          │   │   └── ... (sharded training files)
          │   └── validation/
          │       ├── val_5k.h5ad
          │       └── val_120k.h5ad
          ├── ontology/
          │   └── cl.owl
          └── cell_types.csv
```

#### Instance Store Configuration
```yaml
Scratch Volume:
  Type: NVMe Instance Store
  Mount Point: /scratch
  Access: Read-Write
  
Directory Structure:
  /scratch/
  ├── checkpoints/     # Model checkpoints
  ├── tensorboard/     # TensorBoard logs
  ├── tmp/            # Temporary files
  └── outputs/        # Final models
  
Performance:
  - g5.xlarge: 250 GB, ~1.9 GB/s read
  - g5.2xlarge: 450 GB, ~2.3 GB/s read
  - g5.4xlarge: 600 GB, ~4.6 GB/s read
```

### 4. Launch Template

```json
{
  "LaunchTemplateName": "genept-gpu-training-template",
  "LaunchTemplateData": {
    "ImageIdOverride": "{{resolve:ssm:/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended:image_id}}",
    "InstanceType": "g5.2xlarge",
    "IamInstanceProfile": {
      "Arn": "arn:aws:iam::ACCOUNT_ID:instance-profile/ecsInstanceRole"
    },
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/xvda",
        "Ebs": {
          "VolumeSize": 100,
          "VolumeType": "gp3",
          "Iops": 3000,
          "Throughput": 125,
          "DeleteOnTermination": true,
          "Encrypted": true
        }
      },
      {
        "DeviceName": "/dev/xvdb",
        "Ebs": {
          "SnapshotId": "snap-0b1c573caa4318e2f",
          "VolumeType": "gp3",
          "DeleteOnTermination": true,
          "Encrypted": false
        }
      }
    ],
    "UserData": "{{BASE64_ENCODED_SCRIPT}}",
    "TagSpecifications": [
      {
        "ResourceType": "instance",
        "Tags": [
          {"Key": "Name", "Value": "GenePT-GPU-Training"},
          {"Key": "Project", "Value": "GenePT"},
          {"Key": "ManagedBy", "Value": "Batch"}
        ]
      }
    ],
    "MetadataOptions": {
      "HttpTokens": "required",
      "HttpPutResponseHopLimit": 2
    },
    "Monitoring": {
      "Enabled": true
    }
  }
}
```

#### User Data Script
```bash
#!/bin/bash
set -x

# Mount data volume
mkdir -p /data
mount -o ro /dev/xvdb /data
echo "/dev/xvdb /data ext4 ro,defaults 0 0" >> /etc/fstab

# Setup instance store for scratch
INSTANCE_STORE="/dev/nvme1n1"
if [ -b "$INSTANCE_STORE" ]; then
    mkfs.ext4 $INSTANCE_STORE
    mkdir -p /scratch
    mount $INSTANCE_STORE /scratch
    echo "$INSTANCE_STORE /scratch ext4 defaults,noatime 0 0" >> /etc/fstab
    
    # Create scratch subdirectories
    mkdir -p /scratch/{checkpoints,tensorboard,tmp,outputs}
    chmod 777 /scratch/*
fi

# Configure ECS
cat >> /etc/ecs/ecs.config <<EOF
ECS_CLUSTER=genept-gpu-training-cluster
ECS_ENABLE_GPU_SUPPORT=true
ECS_NVIDIA_RUNTIME=nvidia
ECS_CONTAINER_START_TIMEOUT=10m
ECS_CONTAINER_CREATE_TIMEOUT=10m
ECS_ENABLE_SPOT_INSTANCE_DRAINING=true
EOF

# Install CloudWatch GPU metrics
yum install -y amazon-cloudwatch-agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/custom_gpu_metrics.json <<EOF
{
  "metrics": {
    "namespace": "GenePT/GPU",
    "metrics_collected": {
      "nvidia_smi": {
        "measurement": [
          {"name": "utilization_gpu", "unit": "Percent"},
          {"name": "utilization_memory", "unit": "Percent"},
          {"name": "temperature_gpu", "unit": "None"},
          {"name": "power_draw", "unit": "Watts"}
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF

# Start GPU monitoring
nvidia-smi dmon -s pucvmet -d 60 -o TD > /var/log/gpu-metrics.log 2>&1 &

# Optimize for PyTorch
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> /etc/profile
echo "export CUDA_LAUNCH_BLOCKING=0" >> /etc/profile
echo "export CUDNN_BENCHMARK=1" >> /etc/profile

# System optimizations
sysctl -w vm.max_map_count=262144
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728

echo "GPU instance initialization complete"
```

### 5. Container Configuration

#### Docker Image
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install project code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV DATA_ROOT=/data/GenePT-Tools/data
ENV CHECKPOINT_DIR=/scratch/checkpoints
ENV OUTPUT_DIR=/scratch/outputs

# Health check for GPU
RUN python -c "import torch; assert torch.cuda.is_available()"

ENTRYPOINT ["python", "scripts/train_cellxgene_mlp.py"]
```

### 6. Job Definition

```json
{
  "jobDefinitionName": "genept-gpu-training-job",
  "type": "container",
  "platformCapabilities": ["EC2"],
  "containerProperties": {
    "image": "YOUR_ECR_URI:latest",
    "vcpus": 8,
    "memory": 30720,
    "jobRoleArn": "arn:aws:iam::ACCOUNT_ID:role/batch-job-role",
    "resourceRequirements": [
      {
        "type": "GPU",
        "value": "1"
      }
    ],
    "volumes": [
      {
        "name": "data-volume",
        "host": {"sourcePath": "/data"}
      },
      {
        "name": "scratch-volume",
        "host": {"sourcePath": "/scratch"}
      }
    ],
    "mountPoints": [
      {
        "sourceVolume": "data-volume",
        "containerPath": "/data",
        "readOnly": true
      },
      {
        "sourceVolume": "scratch-volume",
        "containerPath": "/scratch",
        "readOnly": false
      }
    ],
    "environment": [
      {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
      {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "max_split_size_mb:512"},
      {"name": "DATA_DIR", "value": "/data/GenePT-Tools/data/cellxgene_v2"},
      {"name": "CHECKPOINT_DIR", "value": "/scratch/checkpoints"},
      {"name": "OUTPUT_DIR", "value": "/scratch/outputs"},
      {"name": "WANDB_DIR", "value": "/scratch/wandb"}
    ],
    "ulimits": [
      {
        "name": "memlock",
        "hardLimit": -1,
        "softLimit": -1
      },
      {
        "name": "stack",
        "hardLimit": 67108864,
        "softLimit": 67108864
      }
    ],
    "linuxParameters": {
      "sharedMemorySize": 8192,
      "devices": [
        {
          "hostPath": "/dev/nvidia0",
          "containerPath": "/dev/nvidia0",
          "permissions": ["read", "write", "mknod"]
        }
      ]
    },
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/aws/batch/genept-gpu-training",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "training"
      }
    }
  },
  "timeout": {
    "attemptDurationSeconds": 172800
  },
  "retryStrategy": {
    "attempts": 3,
    "evaluateOnExit": [
      {
        "action": "RETRY",
        "onStatusReason": "Task failed*",
        "onExitCode": "1"
      },
      {
        "action": "EXIT",
        "onReason": "ResourcesNotAvailable"
      }
    ]
  }
}
```

### 7. Monitoring & Metrics

#### CloudWatch Metrics
```yaml
GPU Metrics:
  - GPU Utilization (%)
  - GPU Memory Usage (%)
  - GPU Temperature (°C)
  - Power Draw (W)
  
Training Metrics:
  - Batch Processing Time
  - Epoch Duration
  - Loss Values
  - Validation Accuracy
  
System Metrics:
  - CPU Utilization
  - Memory Usage
  - Disk I/O
  - Network Throughput
  
Cost Metrics:
  - Spot Instance Savings
  - Job Duration
  - Cost per Epoch
```

### 8. Cost Optimization

#### Spot Instance Strategy
```yaml
Bidding Strategy:
  Type: SPOT_CAPACITY_OPTIMIZED
  Bid Percentage: 80%
  
Interruption Handling:
  - 2-minute warning via EC2 metadata
  - Automatic checkpoint on interruption
  - Resume from last checkpoint
  
Cost Comparison (us-east-1):
  g5.xlarge:  $0.42/hr spot vs $1.006/hr on-demand (58% savings)
  g5.2xlarge: $0.50/hr spot vs $1.212/hr on-demand (59% savings)
  g5.4xlarge: $0.65/hr spot vs $1.624/hr on-demand (60% savings)
  
Monthly Estimates (24/7 usage):
  g5.2xlarge spot: ~$360/month
  g5.2xlarge on-demand: ~$872/month
  Savings: ~$512/month
```

### 9. Security Configuration

#### IAM Roles
```yaml
Instance Profile Role:
  - ECS container management
  - CloudWatch logs/metrics
  - EC2 describe operations
  
Job Role:
  - S3 read (configs)
  - S3 write (outputs)
  - Secrets Manager (API keys)
  - CloudWatch metrics
  
No permissions for:
  - Data volume modification
  - Network changes
  - IAM operations
```

### 10. Operational Procedures

#### Job Submission
```bash
# Submit training job
aws batch submit-job \
  --job-name "genept-training-$(date +%Y%m%d-%H%M%S)" \
  --job-queue genept-gpu-queue \
  --job-definition genept-gpu-training-job \
  --container-overrides '{
    "environment": [
      {"name": "EPOCHS", "value": "100"},
      {"name": "BATCH_SIZE", "value": "512"},
      {"name": "LEARNING_RATE", "value": "0.001"}
    ]
  }'
```

#### Monitoring
```bash
# Watch GPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name GPUUtilization \
  --dimensions Name=ClusterName,Value=genept-gpu-training-cluster \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 300 \
  --statistics Average
```

#### Checkpoint Recovery
```python
# In training script
checkpoint_dir = "/scratch/checkpoints"
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_state_dict(torch.load(latest_checkpoint))
    print(f"Resumed from {latest_checkpoint}")
```

## Summary

This infrastructure provides:
- **High-performance GPU compute** with NVIDIA A10G GPUs
- **Cost-effective spot instances** with 60% savings
- **Fast local data access** via EBS snapshots
- **Optimized for PyTorch** with CUDA 11.8+
- **Automatic recovery** from spot interruptions
- **Comprehensive monitoring** of GPU and training metrics

The architecture balances performance, cost, and reliability for production ML training workloads.