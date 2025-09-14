# AWS Batch Training Pipeline for GenePT

This directory contains the infrastructure for running GenePT training jobs on AWS Batch with GPU support and pre-populated data volumes.

## Architecture Overview

- **GPU Compute**: p3.2xlarge instances with NVIDIA V100 GPUs
- **Data Strategy**: EBS snapshot with pre-populated training data (zero download time)
- **Model Tracking**: WandB for metrics and model artifacts
- **Cost Optimization**: Spot instances with 70% discount

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Docker installed locally
3. EBS snapshot with training data (snap-0b1c573caa4318e2f)
4. WandB account and API key

## Phase 1: Basic Infrastructure Setup

### Step 1: Configure AWS Account

Export your AWS account ID:
```bash
export AWS_ACCOUNT_ID=YOUR_ACCOUNT_ID
export AWS_DEFAULT_REGION=us-east-1
```

### Step 2: Run Setup Script

```bash
cd aws_batch/training
chmod +x setup_infrastructure.sh
./setup_infrastructure.sh
```

The script will:
1. Create ECR repository and push Docker image
2. Set up IAM roles with necessary permissions
3. Store WandB API key in Secrets Manager
4. Create launch template with data volume attachment
5. Configure GPU compute environment with spot instances
6. Create job queue and job definition
7. Set up CloudWatch logging

### Step 3: Verify Setup

Check that resources were created:
```bash
# Check compute environment
aws batch describe-compute-environments --compute-environments genept-training-gpu-env

# Check job queue
aws batch describe-job-queues --job-queues genept-training-queue

# Check job definition
aws batch describe-job-definitions --job-definition-name genept-training-job --status ACTIVE
```

## Submitting Training Jobs

### Basic Job Submission

```bash
python submit_job.py \
    --job-name experiment-001 \
    --epochs 100 \
    --batch-size 1024 \
    --learning-rate 0.001
```

### With Custom Configuration

```bash
python submit_job.py \
    --job-name experiment-002 \
    --config configs/my_config.yaml \
    --wandb-project genept-experiments
```

### Submit Without Monitoring

```bash
python submit_job.py \
    --job-name experiment-003 \
    --no-monitor \
    --epochs 50
```

## Data Volume Structure

The EBS snapshot contains pre-populated data mounted at `/data`:
```
/data/GenePT-Tools/data/
├── cellxgene_v2/
│   ├── training_v1_shuffled/    # Training data in PT format
│   └── validation/               # Validation datasets
├── ontology/                     # Cell Ontology cache
└── cell_types.csv               # Cell type mappings
```

## Configuration Management

### Default Configuration

A default configuration is stored at `s3://genept-training-configs/default_config.yaml`:
```yaml
model:
  input_dim: 3072
  hidden_dims: [2048, 512, 256]
  num_classes: 107
  dropout: 0.2
  
training:
  batch_size: 512
  epochs: 50
  learning_rate: 0.001
```

### Custom Configurations

Upload custom configs to S3:
```bash
aws s3 cp my_config.yaml s3://genept-training-configs/
```

Or pass them directly when submitting:
```bash
python submit_job.py --job-name test --config my_config.yaml
```

## Monitoring Jobs

### Real-time Monitoring

The submit script monitors jobs by default:
```bash
python submit_job.py --job-name my-job
# Shows live status and logs
```

### Check Job Status

```bash
aws batch describe-jobs --jobs JOB_ID
```

### View Logs

```bash
# Stream logs
aws logs tail /aws/batch/genept-training --follow

# Get specific job logs
aws logs get-log-events \
    --log-group-name /aws/batch/genept-training \
    --log-stream-name training/genept-training-job/JOB_ID
```

### WandB Dashboard

Training metrics and model artifacts are automatically logged to WandB:
- Project: `genept-training` (or custom via --wandb-project)
- View at: https://wandb.ai/YOUR_USERNAME/genept-training

## Cost Optimization

The setup uses several cost optimization strategies:

1. **Spot Instances**: 70% discount on GPU instances
2. **Auto-scaling**: Scales to 0 when no jobs are running
3. **Data Snapshot**: Eliminates repeated data transfer costs
4. **Ephemeral Storage**: Uses instance storage for temporary files

Estimated costs:
- p3.2xlarge spot: ~$0.92/hour (vs $3.06/hour on-demand)
- EBS snapshot storage: ~$0.05/GB/month
- S3 storage: ~$0.023/GB/month

## Troubleshooting

### Job Stuck in RUNNABLE

Check compute environment capacity:
```bash
aws batch describe-compute-environments --compute-environments genept-training-gpu-env
```

### Out of Memory Errors

Increase memory in job definition or use larger instance type.

### Permission Errors

Verify IAM roles:
```bash
aws iam get-role --role-name genept-training-job-role
aws iam get-role-policy --role-name genept-training-job-role --policy-name genept-training-job-policy
```

### Data Volume Not Mounting

Check launch template and user data script:
```bash
aws ec2 describe-launch-template-versions --launch-template-name genept-training-gpu-template
```

## Next Steps (Phase 2 & 3)

### Phase 2: Parameterization
- [ ] Add support for hyperparameter tuning configs
- [ ] Implement job arrays for parallel experiments
- [ ] Add resume from checkpoint capability

### Phase 3: Production Features
- [ ] Set up monitoring dashboards
- [ ] Add automatic retry logic
- [ ] Implement cost tracking
- [ ] Add model registry integration

## Security Considerations

1. **Secrets**: WandB API key stored in AWS Secrets Manager
2. **Network**: Consider using VPC endpoints for S3 access
3. **IAM**: Roles follow least privilege principle
4. **Encryption**: Enable S3 bucket encryption for outputs

## Support

For issues or questions:
1. Check CloudWatch logs for error details
2. Verify all IAM permissions are correctly configured
3. Ensure the data snapshot is accessible in your region