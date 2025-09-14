# AWS Batch Training Specification

## Overview
This specification defines the strategy for running `scripts/train_cellxgene_mlp.py` on AWS Batch with pre-populated data volumes and parameterized job submission.

## Design Goals
- **Zero data transfer**: Use EBS snapshots to eliminate download time
- **Simple parameterization**: Pass training parameters without container rebuilds
- **Persistent outputs**: Store results in S3 beyond WandB artifacts
- **Cost optimization**: Leverage spot instances and ephemeral compute
- **Resume capability**: Support checkpoint recovery for interrupted jobs

## Architecture

### Data Strategy
**EBS Snapshot Volume**
- Pre-populate an EBS volume with all training/validation data
- Create snapshot for instant volume creation across batch jobs
- Mount as `/data` in container filesystem
- Contents include PT files, validation data, cell type mappings, and ontology cache

### Container Strategy
**Minimal Docker Image**
- Base: PyTorch with CUDA support
- Include only application code and dependencies
- Parameters passed at runtime via command line
- No embedded data or configuration

### Job Submission Strategy
**Parameter Injection**
- All training parameters passed as command-line arguments
- Configuration files uploaded to S3 and downloaded at job start
- Environment variables for secrets (WandB API key)
- Job arrays for parallel hyperparameter searches

## Data Volume Design

### Snapshot Contents Structure
```
/data/GenePT-Tools/data/
├── cellxgene_v2/
│   ├── training_v1_shuffled/    # PT format training files
│   └── validation/               # Validation datasets
├── ontology/                     # Cell Ontology cache
└── cell_types.csv               # Cell type to code mappings
```

### Volume Attachment
- Attach snapshot-based volume to Batch compute instances
- Mount automatically via launch template user data
- Read-only access sufficient for training data
- Ephemeral scratch space on instance storage for checkpoints

## Output Management

### Dual Storage Strategy

**Primary (WandB)**
- Model checkpoints as artifacts
- Real-time metrics and logging
- Hyperparameter tracking
- Best model preservation

**Secondary (S3)**
- Final model weights
- Training completion metrics
- Configuration snapshots
- Failed job checkpoints for recovery

### S3 Organization
```
s3://genept-training-outputs/
├── runs/{job_id}/
│   ├── final_model.pt
│   ├── final_metrics.json
│   └── checkpoints/           # For resume capability
└── configs/{job_id}.yaml      # Job configuration snapshot
```

## Parameter Management

### Configuration Hierarchy
1. **Base defaults**: Hardcoded in container for data paths
2. **Config file**: YAML with experiment-specific settings
3. **Command line**: Override any parameter at submission
4. **Environment**: Secrets and AWS metadata

### Parameter Types

**Fixed (in container)**
- Data volume mount points
- Output directories structure
- AWS service endpoints

**Configurable (at runtime)**
- Model architecture (layers, dimensions, dropout)
- Training hyperparameters (learning rate, batch size)
- Evaluation frequency
- WandB project/run naming
- Checkpoint intervals

## Job Execution Flow

### Standard Training
1. Batch launches container on GPU instance
2. EBS volume attached from snapshot
3. Parameters parsed from command line
4. Training runs with periodic S3 checkpoint sync
5. Final outputs uploaded to S3
6. WandB artifacts created throughout

### Hyperparameter Tuning
1. Tuning config uploaded to S3
2. Batch job downloads config at start
3. Optuna manages trial distribution
4. Each trial logs to WandB with unique run name
5. Best parameters saved to S3

### Resume from Failure
1. Hyperparameter search is resumed from stopping point
2. Model training can be restarted for simplicity

## Resource Allocation

### Compute Environment
- **GPU instances**: p3.2xlarge for single GPU training
- **Spot pricing**: 70% cost reduction for non-critical runs
- **Auto-scaling**: 0-10 instances based on queue depth

### Storage
- **EBS snapshots**: GP3 volumes for consistent performance
- **Instance storage**: NVMe SSDs for temporary checkpoints
- **S3 lifecycle**: Archive completed runs after 30 days

## Security Model

### IAM Roles
**Batch Job Role**
- Read from data snapshot volumes
- Access secrets for API keys
- No cross-account permissions

**Compute Environment Role**
- Attach EBS volumes from snapshots
- Create CloudWatch logs
- Register/deregister from ECS

### Secrets Management
- WandB API key in AWS Secrets Manager
- Rotation policy every 90 days
- Injected as environment variables
- Never logged or persisted

## Monitoring Strategy

### Job Metrics
- Execution time per epoch
- GPU utilization percentage
- Memory usage patterns
- S3 transfer volumes

### Failure Detection
- CloudWatch alarms for stuck jobs
- Automatic retry with exponential backoff
- Dead letter queue for persistent failures

## Cost Optimization

### Strategies
1. **Spot instances**: Primary compute with on-demand fallback
2. **Right-sizing**: Match instance type to model requirements
3. **Data reuse**: Snapshot eliminates repeated downloads
4. **Cleanup policies**: Automatic deletion of temporary files


## Implementation Phases

### Phase 1: Basic Infrastructure
- Create data snapshot from existing EBS volume
- Build minimal Docker container
- Set up Batch compute environment and job queue
- Test single job submission

### Phase 2: Parameterization
- Implement configuration file support
- Add command-line parameter mapping
- Create submission scripts
- Test various configurations

### Phase 3: Production Features
- Add checkpoint/resume capability
- Set up monitoring and alerting

## Success Criteria
- Parameters easily modified without code changes
- Failed jobs automatically recoverable
- Results consistently stored and accessible
