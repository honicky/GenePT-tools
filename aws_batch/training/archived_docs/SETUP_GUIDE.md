# AWS Batch GPU Training Infrastructure Setup Guide

## Overview

This guide documents the complete setup of a production-ready AWS Batch infrastructure for GPU-accelerated machine learning training. The system is optimized for cost-efficiency using spot instances and provides seamless integration with existing training data through EBS snapshots.

## Architecture Components

### 1. **Compute Infrastructure**
- **Instance Types**: NVIDIA A10G GPU instances (g5.xlarge, g5.2xlarge, g5.4xlarge)
- **Pricing Strategy**: Spot instances with 80% bid percentage for ~60% cost savings
- **Scaling**: Auto-scaling from 0 to 256 vCPUs based on job queue demand
- **Data Access**: Pre-populated EBS volumes from snapshot `snap-0b1c573caa4318e2f` (zero download time)

### 2. **Container Environment**
- **Base Image**: PyTorch 2.0.1 with CUDA 11.7 support
- **Platform**: Linux/AMD64 (compatible with Apple Silicon development)
- **Dependencies**: Transformers, WandB, scikit-learn, and bioinformatics libraries
- **Registry**: Amazon ECR for secure container storage

### 3. **Data Management**
- **Training Data**: 200GB+ pre-loaded via EBS snapshot (mounts at `/data`)
- **Configuration Storage**: S3 bucket for training configs
- **Output Storage**: S3 bucket for model checkpoints and results
- **Scratch Space**: Local NVMe for temporary files and checkpoints

### 4. **Monitoring & Tracking**
- **Experiment Tracking**: Weights & Biases (WandB) integration
- **Logging**: CloudWatch Logs for real-time training progress
- **Metrics**: GPU utilization and training metrics via WandB

### 5. **Security & Access**
- **IAM Roles**: Separate execution and job roles with minimal permissions
- **Secrets Management**: AWS Secrets Manager for WandB API keys
- **Network**: Default VPC with security groups

## Setup Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured with profile `memverge`
3. **Docker** installed and running
4. **EBS Snapshot** `snap-0b1c573caa4318e2f` containing training data
5. **WandB Account** (optional but recommended)

## Configuration Files

### `.env` Configuration
```bash
# AWS Configuration
AWS_ACCOUNT_ID=971422677163
AWS_DEFAULT_REGION=us-west-2

# Repository and Data
ECR_REPOSITORY_NAME=miratyper-training
DATA_SNAPSHOT_ID=snap-0b1c573caa4318e2f

# Compute Environment
MAX_VCPUS=256
MIN_VCPUS=0
DESIRED_VCPUS=0
SPOT_BID_PERCENTAGE=80
INSTANCE_TYPES=g5.xlarge,g5.2xlarge,g5.4xlarge

# S3 Buckets
CONFIG_BUCKET=miratyper-training-configs
OUTPUT_BUCKET=miratyper-training-outputs

# Resource Names
COMPUTE_ENV_NAME=miratyper-gpu-training-spot
JOB_QUEUE_NAME=miratyper-training-queue
LOG_GROUP_NAME=/aws/batch/miratyper-training

# Optional
WANDB_API_KEY=your_wandb_api_key_here
LOG_RETENTION_DAYS=30
```

## Complete Setup Script

Here's the exact sequence of AWS commands used to successfully set up the infrastructure:

```bash
#!/bin/bash
# AWS Batch GPU Training Infrastructure Setup - Reproducible Script

set -e

# Configuration
export AWS_PROFILE=memverge
export AWS_REGION=us-west-2
export ACCOUNT_ID=971422677163
export ECR_REPO=miratyper-training
export SNAPSHOT_ID=snap-0b1c573caa4318e2f

# ============================================
# Step 1: Create ECR Repository
# ============================================
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name ${ECR_REPO} \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --image-scanning-configuration scanOnPush=true \
    || echo "Repository already exists"

# Get ECR URI
ECR_URI=$(aws ecr describe-repositories \
    --repository-names ${ECR_REPO} \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'repositories[0].repositoryUri' \
    --output text)

# ============================================
# Step 2: Build and Push Docker Image
# ============================================
echo "Building Docker image..."

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | \
    docker login --username AWS --password-stdin ${ECR_URI%/*}

# Build image (from project root directory)
docker build --platform linux/amd64 \
    -f aws_batch/training/Dockerfile \
    -t ${ECR_REPO} .

# Tag and push
docker tag ${ECR_REPO}:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest

# ============================================
# Step 3: Create IAM Roles
# ============================================
echo "Creating IAM roles..."

# Create ECS task execution role
aws iam create-role \
    --role-name genept-training-execution-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile ${AWS_PROFILE} \
    || echo "Execution role exists"

aws iam attach-role-policy \
    --role-name genept-training-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    --profile ${AWS_PROFILE}

# Create job role
aws iam create-role \
    --role-name genept-training-job-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile ${AWS_PROFILE} \
    || echo "Job role exists"

# Add job role policy
aws iam put-role-policy \
    --role-name genept-training-job-role \
    --policy-name genept-training-job-policy \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::miratyper-training-*/*",
                    "arn:aws:s3:::miratyper-training-*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": "arn:aws:secretsmanager:*:971422677163:secret:wandb_api_key*"
            }
        ]
    }' \
    --profile ${AWS_PROFILE}

# Create Batch service roles
aws iam create-role \
    --role-name aws-batch-service-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "batch.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile ${AWS_PROFILE} \
    || echo "Batch service role exists"

aws iam attach-role-policy \
    --role-name aws-batch-service-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole \
    --profile ${AWS_PROFILE}

# Create spot fleet role
aws iam create-role \
    --role-name aws-batch-spot-fleet-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "spotfleet.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' \
    --profile ${AWS_PROFILE} \
    || echo "Spot fleet role exists"

aws iam attach-role-policy \
    --role-name aws-batch-spot-fleet-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole \
    --profile ${AWS_PROFILE}

# ============================================
# Step 4: Create S3 Buckets
# ============================================
echo "Creating S3 buckets..."
aws s3 mb s3://miratyper-training-configs --region ${AWS_REGION} --profile ${AWS_PROFILE} || echo "Bucket exists"
aws s3 mb s3://miratyper-training-outputs --region ${AWS_REGION} --profile ${AWS_PROFILE} || echo "Bucket exists"

# ============================================
# Step 5: Store WandB API Key (Optional)
# ============================================
if [ ! -z "${WANDB_API_KEY}" ]; then
    echo "Storing WandB API key..."
    aws secretsmanager create-secret \
        --name wandb_api_key \
        --secret-string "${WANDB_API_KEY}" \
        --region ${AWS_REGION} \
        --profile ${AWS_PROFILE} \
        || echo "Secret exists"
fi

# ============================================
# Step 6: Create Launch Template
# ============================================
echo "Creating launch template..."

# Get latest ECS GPU-optimized AMI
AMI_ID=$(aws ssm get-parameter \
    --name /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'Parameter.Value' \
    --output text)

# Create user data script
cat > /tmp/user_data.sh << 'EOF'
#!/bin/bash
echo ECS_CLUSTER=miratyper-gpu-training-spot >> /etc/ecs/ecs.config
echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config

# Mount data volume
mkdir -p /data
mount /dev/xvdb /data

# Create scratch directories
mkdir -p /scratch/checkpoints
mkdir -p /scratch/outputs
mkdir -p /scratch/wandb
chmod 777 /scratch/*
EOF

USER_DATA=$(base64 -i /tmp/user_data.sh)

# Create launch template
aws ec2 create-launch-template \
    --launch-template-name genept-training-gpu-template \
    --launch-template-data "{
        \"ImageId\": \"${AMI_ID}\",
        \"InstanceType\": \"g5.2xlarge\",
        \"IamInstanceProfile\": {
            \"Arn\": \"arn:aws:iam::${ACCOUNT_ID}:instance-profile/ecsInstanceRole\"
        },
        \"BlockDeviceMappings\": [
            {
                \"DeviceName\": \"/dev/xvda\",
                \"Ebs\": {
                    \"VolumeSize\": 100,
                    \"VolumeType\": \"gp3\",
                    \"DeleteOnTermination\": true
                }
            },
            {
                \"DeviceName\": \"/dev/xvdb\",
                \"Ebs\": {
                    \"SnapshotId\": \"${SNAPSHOT_ID}\",
                    \"VolumeType\": \"gp3\",
                    \"DeleteOnTermination\": true
                }
            }
        ],
        \"UserData\": \"${USER_DATA}\",
        \"TagSpecifications\": [{
            \"ResourceType\": \"instance\",
            \"Tags\": [
                {\"Key\": \"Project\", \"Value\": \"MiraTyper\"},
                {\"Key\": \"Environment\", \"Value\": \"training\"}
            ]
        }]
    }" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    || echo "Launch template exists"

# ============================================
# Step 7: Create Compute Environment
# ============================================
echo "Creating compute environment..."

# Get VPC information
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=is-default,Values=true" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'Vpcs[0].VpcId' \
    --output text)

SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'Subnets[0:2].SubnetId' \
    --output json)

SECURITY_GROUP=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" "Name=group-name,Values=default" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

aws batch create-compute-environment \
    --compute-environment-name miratyper-gpu-training-spot \
    --type MANAGED \
    --state ENABLED \
    --service-role arn:aws:iam::${ACCOUNT_ID}:role/aws-batch-service-role \
    --compute-resources "{
        \"type\": \"SPOT\",
        \"bidPercentage\": 80,
        \"spotIamFleetRole\": \"arn:aws:iam::${ACCOUNT_ID}:role/aws-batch-spot-fleet-role\",
        \"minvCpus\": 0,
        \"maxvCpus\": 256,
        \"desiredvCpus\": 0,
        \"instanceTypes\": [\"g5.xlarge\", \"g5.2xlarge\", \"g5.4xlarge\"],
        \"allocationStrategy\": \"SPOT_CAPACITY_OPTIMIZED\",
        \"subnets\": ${SUBNETS},
        \"securityGroupIds\": [\"${SECURITY_GROUP}\"],
        \"instanceRole\": \"arn:aws:iam::${ACCOUNT_ID}:instance-profile/ecsInstanceRole\",
        \"launchTemplate\": {
            \"launchTemplateName\": \"genept-training-gpu-template\",
            \"version\": \"\$Latest\"
        },
        \"tags\": {
            \"Project\": \"MiraTyper\",
            \"Environment\": \"training\"
        }
    }" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    || echo "Compute environment exists"

# Wait for compute environment to be valid
echo "Waiting for compute environment to become valid..."
while true; do
    STATUS=$(aws batch describe-compute-environments \
        --compute-environments miratyper-gpu-training-spot \
        --region ${AWS_REGION} \
        --profile ${AWS_PROFILE} \
        --query 'computeEnvironments[0].status' \
        --output text)
    
    if [ "${STATUS}" == "VALID" ]; then
        echo "Compute environment is ready!"
        break
    fi
    echo "Current status: ${STATUS}. Waiting..."
    sleep 10
done

# ============================================
# Step 8: Create Job Queue
# ============================================
echo "Creating job queue..."
aws batch create-job-queue \
    --job-queue-name miratyper-training-queue \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=miratyper-gpu-training-spot \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    || echo "Job queue exists"

# ============================================
# Step 9: Register Job Definition
# ============================================
echo "Registering job definition..."
aws batch register-job-definition \
    --job-definition-name genept-training-job \
    --type container \
    --platform-capabilities EC2 \
    --parameters '{
        "config_file": "s3://miratyper-training-configs/default_config.yaml",
        "wandb_project": "miratyper-training",
        "epochs": "50",
        "batch_size": "512",
        "learning_rate": "0.001"
    }' \
    --container-properties "{
        \"image\": \"${ECR_URI}:latest\",
        \"vcpus\": 8,
        \"memory\": 30720,
        \"resourceRequirements\": [{
            \"type\": \"GPU\",
            \"value\": \"1\"
        }],
        \"jobRoleArn\": \"arn:aws:iam::${ACCOUNT_ID}:role/genept-training-job-role\",
        \"executionRoleArn\": \"arn:aws:iam::${ACCOUNT_ID}:role/genept-training-execution-role\",
        \"volumes\": [
            {\"name\": \"data\", \"host\": {\"sourcePath\": \"/data\"}},
            {\"name\": \"scratch\", \"host\": {\"sourcePath\": \"/scratch\"}}
        ],
        \"mountPoints\": [
            {\"sourceVolume\": \"data\", \"containerPath\": \"/data\", \"readOnly\": true},
            {\"sourceVolume\": \"scratch\", \"containerPath\": \"/scratch\", \"readOnly\": false}
        ],
        \"environment\": [
            {\"name\": \"DATA_ROOT\", \"value\": \"/data/GenePT-Tools/data\"},
            {\"name\": \"CHECKPOINT_DIR\", \"value\": \"/scratch/checkpoints\"},
            {\"name\": \"OUTPUT_DIR\", \"value\": \"/scratch/outputs\"},
            {\"name\": \"CUDA_VISIBLE_DEVICES\", \"value\": \"0\"},
            {\"name\": \"PYTORCH_CUDA_ALLOC_CONF\", \"value\": \"max_split_size_mb:512\"},
            {\"name\": \"WANDB_DIR\", \"value\": \"/scratch/wandb\"}
        ],
        \"secrets\": [{
            \"name\": \"WANDB_API_KEY\",
            \"valueFrom\": \"arn:aws:secretsmanager:${AWS_REGION}:${ACCOUNT_ID}:secret:wandb_api_key\"
        }],
        \"command\": [
            \"--config\", \"Ref::config_file\",
            \"--wandb-project\", \"Ref::wandb_project\",
            \"--epochs\", \"Ref::epochs\",
            \"--batch-size\", \"Ref::batch_size\",
            \"--learning-rate\", \"Ref::learning_rate\",
            \"--data-dir\", \"/data/GenePT-Tools/data/cellxgene_v2\",
            \"--checkpoint-dir\", \"/scratch/checkpoints\"
        ],
        \"logConfiguration\": {
            \"logDriver\": \"awslogs\",
            \"options\": {
                \"awslogs-group\": \"/aws/batch/miratyper-training\",
                \"awslogs-region\": \"${AWS_REGION}\",
                \"awslogs-stream-prefix\": \"training\"
            }
        }
    }" \
    --timeout "{\"attemptDurationSeconds\": 43200}" \
    --retry-strategy "{\"attempts\": 2}" \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE}

# ============================================
# Step 10: Create CloudWatch Log Group
# ============================================
echo "Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name /aws/batch/miratyper-training \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    || echo "Log group exists"

# Set retention policy
aws logs put-retention-policy \
    --log-group-name /aws/batch/miratyper-training \
    --retention-in-days 30 \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE}

# ============================================
# Step 11: Upload Default Configuration
# ============================================
echo "Creating default training configuration..."
cat > /tmp/default_config.yaml << 'EOF'
# Default training configuration
model:
  input_dim: 3072
  hidden_dims: [2048, 512, 256]
  num_classes: 107
  dropout: 0.2
  
training:
  batch_size: 512
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
  
validation:
  val_interval: 1000
  use_5k: true
  
checkpoint:
  checkpoint_every_n_batches: 5000
  
wandb:
  project: miratyper-training
  enabled: true
EOF

aws s3 cp /tmp/default_config.yaml \
    s3://miratyper-training-configs/default_config.yaml \
    --profile ${AWS_PROFILE}

# ============================================
# Verification
# ============================================
echo ""
echo "✅ Setup Complete! Verifying resources..."
echo "==========================================="

# Verify compute environment
CE_STATUS=$(aws batch describe-compute-environments \
    --compute-environments miratyper-gpu-training-spot \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'computeEnvironments[0].status' \
    --output text)
echo "✓ Compute Environment: miratyper-gpu-training-spot (${CE_STATUS})"

# Verify job queue
JQ_STATUS=$(aws batch describe-job-queues \
    --job-queues miratyper-training-queue \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'jobQueues[0].status' \
    --output text)
echo "✓ Job Queue: miratyper-training-queue (${JQ_STATUS})"

# Verify job definition
JD_REVISION=$(aws batch describe-job-definitions \
    --job-definition-name genept-training-job \
    --status ACTIVE \
    --region ${AWS_REGION} \
    --profile ${AWS_PROFILE} \
    --query 'jobDefinitions[0].revision' \
    --output text)
echo "✓ Job Definition: genept-training-job:${JD_REVISION}"

echo "✓ ECR Repository: ${ECR_URI}"
echo "✓ Config Bucket: s3://miratyper-training-configs"
echo "✓ Output Bucket: s3://miratyper-training-outputs"
echo ""
echo "Ready to submit training jobs!"
```

## Submitting Training Jobs

Once the infrastructure is set up, you can submit training jobs using either the AWS CLI or the provided Python script:

### Using AWS CLI
```bash
aws batch submit-job \
    --job-name my-training-job \
    --job-queue miratyper-training-queue \
    --job-definition genept-training-job \
    --parameters epochs=100,batch_size=1024 \
    --region us-west-2 \
    --profile memverge
```

### Using Python Script
```bash
python submit_job.py --job-name experiment-001 --epochs 100 --batch-size 1024
```

The Python script provides additional features:
- Automatic config file upload to S3
- Real-time job monitoring with log streaming
- Resource override options (memory, vCPUs, GPUs)
- Integration with .env configuration

## Monitoring & Debugging

### View Job Status
```bash
aws batch describe-jobs --jobs <job-id> --profile memverge
```

### Stream Logs
```bash
aws logs tail /aws/batch/miratyper-training --follow --profile memverge
```

### AWS Console
Monitor jobs in the AWS Batch console:
```
https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2#jobs
```

### WandB Dashboard
Track experiments and metrics at:
```
https://wandb.ai/<your-entity>/miratyper-training
```

## Cost Optimization

The infrastructure is designed for maximum cost efficiency:

1. **Spot Instances**: ~60% savings over on-demand pricing
2. **Auto-scaling**: Scales to zero when no jobs are running
3. **Efficient Data Loading**: EBS snapshots eliminate download time
4. **Resource Right-sizing**: Multiple instance types for optimal allocation
5. **Automatic Retry**: Handles spot interruptions gracefully

## Troubleshooting

### Common Issues

1. **Compute Environment Invalid**
   - Check IAM roles and permissions
   - Verify ecsInstanceRole exists
   - Ensure launch template is valid

2. **Job Fails to Start**
   - Check CloudWatch logs for errors
   - Verify Docker image exists in ECR
   - Ensure job definition parameters are correct

3. **GPU Not Available**
   - Verify instance type supports GPUs
   - Check CUDA version compatibility
   - Ensure ECS GPU support is enabled

4. **Data Not Accessible**
   - Verify EBS snapshot ID is correct
   - Check mount points in user data script
   - Ensure IAM role has necessary permissions

## Clean Up

To remove all resources when no longer needed:

```bash
# Delete job queue
aws batch update-job-queue \
    --job-queue miratyper-training-queue \
    --state DISABLED \
    --profile memverge

aws batch delete-job-queue \
    --job-queue miratyper-training-queue \
    --profile memverge

# Delete compute environment
aws batch update-compute-environment \
    --compute-environment miratyper-gpu-training-spot \
    --state DISABLED \
    --profile memverge

aws batch delete-compute-environment \
    --compute-environment miratyper-gpu-training-spot \
    --profile memverge

# Delete S3 buckets (after backing up data)
aws s3 rb s3://miratyper-training-configs --force --profile memverge
aws s3 rb s3://miratyper-training-outputs --force --profile memverge

# Delete ECR repository
aws ecr delete-repository \
    --repository-name miratyper-training \
    --force \
    --profile memverge

# Delete IAM roles
aws iam delete-role --role-name genept-training-execution-role --profile memverge
aws iam delete-role --role-name genept-training-job-role --profile memverge
aws iam delete-role --role-name aws-batch-service-role --profile memverge
aws iam delete-role --role-name aws-batch-spot-fleet-role --profile memverge

# Delete launch template
aws ec2 delete-launch-template \
    --launch-template-name genept-training-gpu-template \
    --profile memverge

# Delete CloudWatch log group
aws logs delete-log-group \
    --log-group-name /aws/batch/miratyper-training \
    --profile memverge

# Delete secrets
aws secretsmanager delete-secret \
    --secret-id wandb_api_key \
    --force-delete-without-recovery \
    --profile memverge
```

## Summary

This infrastructure provides a production-ready, cost-optimized solution for GPU-accelerated ML training on AWS. The combination of spot instances, pre-populated EBS volumes, and automated scaling ensures both performance and cost-efficiency. The integration with WandB enables comprehensive experiment tracking, while CloudWatch provides real-time monitoring of training progress.