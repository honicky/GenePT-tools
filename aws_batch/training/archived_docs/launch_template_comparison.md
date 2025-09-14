# Launch Template Comparison

## MemVerge Launch Template (Existing)
**Name**: MemVergeBatchLaunchTemplate-miraomics  
**Region**: us-west-2  
**Created**: 2025-08-17  

### Key Features:
- **Instance Type**: Not specified in template (flexible)
- **AMI**: ami-0d3bb50d3c35f67d4
- **Storage**: 
  - Root volume: 30GB gp3
- **Special Setup**:
  - JuiceFS distributed filesystem for checkpoint and scratch
  - MemVerge Memory Machine Cloud (MMC) integration
  - Redis-backed storage for checkpoint persistence
  - Miniconda environment
  - Extended ECS timeouts (16m start, 15m create)

### Mount Points:
- `/mmc-checkpoint` → JuiceFS checkpoint storage (Redis-backed)
- `/mmc-scratch` → JuiceFS scratch storage (Redis-backed)
- `/mnt/jfs/scratch/temp` - Temp directory
- `/mnt/jfs/scratch/work` - Work directory
- `/mnt/jfs/scratch/out` - Output directory

## GenePT Training Template (New)
**Name**: genept-training-gpu-template  
**Region**: us-east-1 (to be configured)  
**Status**: Not yet created  

### Key Features:
- **Instance Type**: g5.2xlarge (modified from p3.2xlarge)
- **AMI**: ECS-optimized GPU AMI (placeholder)
- **Storage**:
  - Root volume: 100GB gp3
  - Data volume: EBS snapshot snap-0b1c573caa4318e2f
- **Simple Setup**:
  - Direct EBS snapshot mount for training data
  - Local scratch directories
  - Standard ECS configuration

### Mount Points:
- `/data` → EBS snapshot with training data (read-only)
- `/scratch/checkpoints` → Local instance storage for checkpoints
- `/scratch/outputs` → Local instance storage for outputs

## Key Differences

| Feature | MemVerge Template | GenePT Template |
|---------|------------------|-----------------|
| **Storage Strategy** | Distributed JuiceFS | Local EBS + snapshot |
| **Checkpoint Persistence** | Redis-backed, survives instance termination | Local, lost on termination |
| **Data Source** | Not specified | Pre-populated EBS snapshot |
| **GPU Type** | Flexible | g5.2xlarge (A10G GPU) |
| **Complexity** | High (MMC, JuiceFS, Redis) | Low (standard AWS) |
| **Cost** | Higher (Redis clusters, distributed FS) | Lower (spot instances, local storage) |
| **Resume Capability** | Built-in via MMC | Manual via S3 sync |

## Recommendations

### Option 1: Use MemVerge Infrastructure
**Pros:**
- Already set up and tested
- Checkpoint persistence across job failures
- Distributed scratch space shared across jobs
- Memory Machine Cloud optimizations

**Cons:**
- More complex to manage
- Higher operational costs
- Requires MemVerge-specific configurations

### Option 2: Continue with Simple GenePT Template  
**Pros:**
- Simpler architecture
- Lower costs with spot instances
- Direct control over resources
- EBS snapshot already prepared

**Cons:**
- No built-in checkpoint persistence
- Need to implement S3 sync for checkpoints
- Lost work on spot termination

### Option 3: Hybrid Approach
Modify the GenePT template to use some MemVerge features:
1. Keep EBS snapshot for training data
2. Use JuiceFS for checkpoint storage only
3. Use local instance storage for scratch
4. This provides checkpoint persistence without full MMC complexity

## Next Steps

To proceed with Phase 1 using the existing MemVerge infrastructure:

1. **Adapt job definition** to work with MemVerge mount points:
   - Update data paths to use MemVerge conventions
   - Configure checkpoint directory to `/mmc-checkpoint`
   - Use `/mmc-scratch` for temporary files

2. **Test with MemVerge compute environment**:
   ```bash
   aws batch describe-compute-environments --profile memverge --region us-west-2
   ```

3. **Submit test job** to validate the setup works with your training code

To proceed with the simple GenePT template:

1. Continue with the setup script in us-east-1
2. Create new compute environment with GPU support  
3. Use S3 for checkpoint backup if needed