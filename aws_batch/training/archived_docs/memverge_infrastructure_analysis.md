# MemVerge Infrastructure Analysis

## Compute Environment: miraomics-spot-MM-Batch-ComputeEnvironment

### Overview
- **Type**: SPOT (cost-optimized)
- **State**: ENABLED
- **Status**: VALID (Healthy)
- **Region**: us-west-2
- **ECS Cluster**: `miraomics-spot-MM-Batch-ComputeEnvironment_Batch_297d06db-714b-362a-82b3-741e9088a489`

### Compute Resources
- **Instance Types**: m6i family (CPU-only, no GPU)
  - m6i.large to m6i.16xlarge
  - Intel Ice Lake processors
- **vCPUs**: 0 min, 256 max, 0 desired
- **Spot Fleet**: Enabled for cost savings
- **Launch Template**: MemVergeBatchLaunchTemplate-miraomics

### AMI Details
- **AMI ID**: ami-0d3bb50d3c35f67d4
- **Name**: al2023-ami-ecs-hvm-2023.0.20241003-kernel-6.1-x86_64
- **Type**: Amazon Linux 2023 ECS-optimized (non-GPU)
- **Architecture**: x86_64
- **Root Volume**: 30GB gp3

### MemVerge Features
The infrastructure includes Memory Machine Cloud (MMC) with:

1. **JuiceFS Distributed Storage**:
   - Checkpoint storage: Redis-backed at `mm-engine-checkpoint-miraomics.p7h3a4.clustercfg.memorydb.us-west-2.amazonaws.com`
   - Scratch storage: Redis-backed at `mm-engine-scratch-miraomics.p7h3a4.clustercfg.memorydb.us-west-2.amazonaws.com`

2. **Mount Points**:
   - `/mmc-checkpoint` → Persistent checkpoint storage
   - `/mmc-scratch` → Distributed scratch space
   - `/mnt/jfs/scratch/temp` → Temporary files
   - `/mnt/jfs/scratch/work` → Work directory
   - `/mnt/jfs/scratch/out` → Output directory

3. **Checkpoint Configuration** (from job definition):
   - Interval: 2 minutes
   - Mode: Iterative
   - Path: `/mmc-checkpoint`

### Job Queue
- **Name**: miraomics-spot-MM-Batch-JobQueue
- **State**: ENABLED
- **Priority**: 1

### Test Job Definition: jd-test
- Simple counter script for testing
- Uses JuiceFS checkpoint volume
- 1 vCPU, 2GB memory
- 5 retry attempts

## Key Observations

### Limitations for ML Training
1. **No GPU Support**: Current setup uses CPU-only instances (m6i family)
2. **No Data Volume**: No EBS snapshot attachment for training data
3. **Test-only Job Definition**: Only a simple test job exists

### Advantages
1. **Checkpoint Persistence**: Redis-backed storage survives spot interruptions
2. **Cost Optimization**: Spot instances reduce costs by up to 70%
3. **Distributed Storage**: JuiceFS enables shared data across jobs
4. **Auto-recovery**: MMC handles checkpoint/resume automatically

## Required Changes for ML Training

To use this infrastructure for GenePT training:

### 1. Create GPU Compute Environment
```bash
# Need to create new compute environment with:
- GPU instance types (g5.2xlarge, p3.2xlarge)
- GPU-optimized ECS AMI
- Same MemVerge launch template (modified for GPU)
```

### 2. Modify Launch Template
- Add EBS snapshot attachment for training data
- Update to GPU-compatible AMI
- Increase root volume size if needed

### 3. Create Training Job Definition
```json
{
  "containerProperties": {
    "image": "YOUR_TRAINING_IMAGE",
    "vcpus": 4,
    "memory": 61440,
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ],
    "volumes": [
      {
        "name": "juicefs-checkpoint",
        "host": {"sourcePath": "/mnt/mmc-checkpoint/checkpoint"}
      },
      {
        "name": "training-data",
        "host": {"sourcePath": "/data"}
      }
    ],
    "mountPoints": [
      {
        "sourceVolume": "juicefs-checkpoint",
        "containerPath": "/checkpoint"
      },
      {
        "sourceVolume": "training-data",
        "containerPath": "/data",
        "readOnly": true
      }
    ],
    "environment": [
      {"name": "MMC_CHECKPOINT_INTERVAL", "value": "10m"},
      {"name": "CHECKPOINT_DIR", "value": "/checkpoint"},
      {"name": "DATA_DIR", "value": "/data"}
    ]
  }
}
```

### 4. Data Strategy Options

**Option A: Use S3 with JuiceFS caching**
- Store training data in S3
- JuiceFS automatically caches frequently accessed data
- No EBS snapshot needed

**Option B: Attach EBS snapshot**
- Modify launch template to attach snap-0b1c573caa4318e2f
- Mount at `/data` alongside JuiceFS mounts

**Option C: Pre-load data to JuiceFS**
- Copy training data to JuiceFS scratch volume once
- All jobs access shared data
- Best for frequently reused datasets

## Recommendation

The MemVerge infrastructure provides excellent checkpoint/resume capabilities but needs modifications for GPU ML training:

1. **Short term**: Create a new GPU compute environment alongside the existing CPU one
2. **Use JuiceFS for checkpoints**: Leverage existing Redis infrastructure
3. **Store data in S3**: Use JuiceFS caching instead of EBS snapshots
4. **Test incrementally**: Start with small jobs to validate the setup

This approach combines MemVerge's fault tolerance with GPU compute for training.