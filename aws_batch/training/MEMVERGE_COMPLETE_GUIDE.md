# MemVerge AWS Batch GPU Training - Complete Setup Guide

## Table of Contents
1. [Infrastructure Overview](#infrastructure-overview)
2. [Setting Up MemVerge-Enabled Job Queues](#setting-up-memverge-enabled-job-queues)
3. [Submitting Jobs to MemVerge Queues](#submitting-jobs-to-memverge-queues)
4. [Resource Locations and Definitions](#resource-locations-and-definitions)
5. [Troubleshooting and Monitoring](#troubleshooting-and-monitoring)

---

## Infrastructure Overview

### MemVerge Management Server
- **Public IP**: `35.90.252.151:8080`
- **Private IP**: `172.31.19.221:8080`
- **Instance ID**: `i-05475df20ecd6407e` (Management-Server-miraomics)
- **Web Dashboard**: https://35.90.252.151:8080
- **API Base URL**: `https://35.90.252.151:8080/api/v1/`

### AWS Resources
- **Region**: us-west-2
- **AWS Profile**: memverge
- **Account ID**: 971422677163

---

## Setting Up MemVerge-Enabled Job Queues

### Step 1: Create Launch Template with MemVerge Agent

Create a launch template that includes MemVerge agent installation in the user data:

```bash
# Key components needed in user_data_mime.txt:
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="
MIME-Version: 1.0

--==MYBOUNDARY==
Content-Type: text/x-shellscript; charset="us-ascii"

#!/bin/bash
# MemVerge agent installation
MMAB_SERVER_ADDRESS="172.31.19.221"
MMAB_SERVER_PORT="8080"
curl -k https://${MMAB_SERVER_ADDRESS}:${MMAB_SERVER_PORT}/api/v1/scripts/install-pagent | bash

# Configure ECS cluster (must match Batch-generated name)
echo "ECS_CLUSTER=<compute-env-name>_Batch_<uuid>" >> /etc/ecs/ecs.config
```

**File Location**: `/Users/rj/personal/GenePT-tools/aws_batch/training/user_data_mime.txt`

Create the launch template:
```bash
aws ec2 create-launch-template \
    --launch-template-name genept-training-gpu-template \
    --version-description "GPU instances with MemVerge support" \
    --launch-template-data '{
        "UserData": "'$(base64 < user_data_mime.txt)'",
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvdb",
                "Ebs": {"SnapshotId": "snap-0e0e93bb43c604c48", "VolumeSize": 500}
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "MemVerge", "Value": "enabled"}]}
        ]
    }' \
    --region us-west-2 --profile memverge
```

### Step 2: Create Compute Environment with MemVerge

```bash
aws batch create-compute-environment \
    --compute-environment-name miratyper-gpu-memverge \
    --type MANAGED \
    --state ENABLED \
    --service-role arn:aws:iam::971422677163:role/aws-batch-service-role \
    --compute-resources '{
        "type": "SPOT",
        "bidPercentage": 80,
        "spotIamFleetRole": "arn:aws:iam::971422677163:role/aws-batch-spot-fleet-role",
        "minvCpus": 0,
        "maxvCpus": 256,
        "desiredvCpus": 0,
        "instanceTypes": ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"],
        "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
        "subnets": ["subnet-xxx", "subnet-yyy"],
        "securityGroupIds": ["sg-xxx"],
        "instanceRole": "arn:aws:iam::971422677163:instance-profile/Batch-Engine-IAMAndManagementStack-AP40G4XPC5PU-BatchInstanceProfile-lA39EGvF2Fvo",
        "launchTemplate": {
            "launchTemplateName": "genept-training-gpu-template",
            "version": "$Latest"
        },
        "tags": {
            "Project": "MiraTyper",
            "Environment": "training",
            "MemVerge": "enabled"
        }
    }' \
    --region us-west-2 --profile memverge
```

**Important**: After creation, get the actual ECS cluster name and update the user data:
```bash
aws batch describe-compute-environments \
    --compute-environments miratyper-gpu-memverge \
    --region us-west-2 --profile memverge \
    | jq -r '.computeEnvironments[0].ecsClusterArn'
# Extract cluster name like: miratyper-gpu-memverge_Batch_7ecfc17d-38b4-3977-880f-43dbe0789807
```

### Step 3: Create Job Queue

```bash
aws batch create-job-queue \
    --job-queue-name miratyper-memverge-queue \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=miratyper-gpu-memverge \
    --region us-west-2 --profile memverge
```

### Step 4: Create IAM Roles

**Execution Role** (for ECS to pull images and write logs):
```bash
aws iam create-role \
    --role-name genept-training-execution-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' --profile memverge

# Attach policies
aws iam attach-role-policy \
    --role-name genept-training-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    --profile memverge

aws iam attach-role-policy \
    --role-name genept-training-execution-role \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite \
    --profile memverge
```

**Job Role** (for container to access AWS resources):
```bash
aws iam create-role \
    --role-name genept-training-job-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }' --profile memverge

# Create and attach custom policy (see job-role-policy.json)
aws iam put-role-policy \
    --role-name genept-training-job-role \
    --policy-name genept-training-policy \
    --policy-document file://job-role-policy.json \
    --profile memverge
```

---

## Submitting Jobs to MemVerge Queues

### Method 1: AWS CLI Direct Submission

```bash
aws batch submit-job \
    --job-name "my-training-job" \
    --job-queue miratyper-memverge-queue \
    --job-definition genept-training-job \
    --region us-west-2 \
    --profile memverge \
    --parameters epochs=100,batch_size=512 \
    --container-overrides '{
        "environment": [
            {"name": "WANDB_PROJECT", "value": "genept-training"},
            {"name": "WANDB_RUN_NAME", "value": "experiment-001"}
        ]
    }'
```

### Method 2: Python Script (submit_job.py)

```bash
python submit_job.py \
    --job-name experiment-001 \
    --job-queue miratyper-memverge-queue \
    --epochs 100 \
    --batch-size 512 \
    --learning-rate 0.001
```

**File Location**: `aws_batch/training/submit_job.py`

### Method 3: Job Definition with Parameters

Create job definition:
```bash
aws batch register-job-definition \
    --cli-input-json file://job_definition.json \
    --region us-west-2 \
    --profile memverge
```

Submit with parameters:
```bash
aws batch submit-job \
    --job-name my-job \
    --job-queue miratyper-memverge-queue \
    --job-definition genept-training-job:latest \
    --parameters config_file=s3://bucket/config.yaml,epochs=100 \
    --region us-west-2 --profile memverge
```

---

## Resource Locations and Definitions

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `user_data_mime.txt` | `/aws_batch/training/user_data_mime.txt` | Launch template user data with MemVerge agent |
| `job_definition.json` | `/aws_batch/training/job_definition.json` | Job definition template |
| `compute_environment_gpu.json` | `/aws_batch/training/compute_environment_gpu.json` | Compute environment configuration |
| `job-role-policy.json` | `/aws_batch/training/job-role-policy.json` | IAM policy for job role |
| `submit_job.py` | `/aws_batch/training/submit_job.py` | Python script for job submission |
| `recreate_compute_env.sh` | `/aws_batch/training/recreate_compute_env.sh` | Script to recreate compute environment |

### AWS Resources

#### Active Resources
- **Job Queue**: `miratyper-memverge-queue`
- **Compute Environment**: `miratyper-gpu-memverge`
- **Launch Template**: `genept-training-gpu-template` (Version 4)
- **EBS Snapshot**: `snap-0e0e93bb43c604c48` (500GB with training data)
- **Container Image**: `971422677163.dkr.ecr.us-west-2.amazonaws.com/miratyper-training:latest`

#### IAM Roles
- **Execution Role**: `arn:aws:iam::971422677163:role/genept-training-execution-role`
- **Job Role**: `arn:aws:iam::971422677163:role/genept-training-job-role`
- **Service Role**: `arn:aws:iam::971422677163:role/aws-batch-service-role`
- **Spot Fleet Role**: `arn:aws:iam::971422677163:role/aws-batch-spot-fleet-role`
- **Instance Profile**: `arn:aws:iam::971422677163:instance-profile/Batch-Engine-IAMAndManagementStack-AP40G4XPC5PU-BatchInstanceProfile-lA39EGvF2Fvo`

#### Secrets
- **WandB API Key**: `arn:aws:secretsmanager:us-west-2:971422677163:secret:wandb-api-key-WjZfMv`

#### CloudWatch Logs
- **Log Group**: `/aws/batch/miratyper-training`
- **Job Logs**: `/aws/batch/job`

### MemVerge API Endpoints

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `/api/v1/node` | View registered compute nodes | `curl -sk https://35.90.252.151:8080/api/v1/node \| jq .` |
| `/api/v1/job` | View running/completed jobs | `curl -sk https://35.90.252.151:8080/api/v1/job \| jq .` |
| `/api/v1/config` | View checkpoint configuration | `curl -sk https://35.90.252.151:8080/api/v1/config \| jq .` |
| `/api/v1/metrics/summary` | View cost savings metrics | `curl -sk https://35.90.252.151:8080/api/v1/metrics/summary \| jq .` |

### GPU Instance Types

| Type | GPUs | vCPUs | Memory | GPU Memory |
|------|------|-------|--------|------------|
| g5.xlarge | 1x A10G | 4 | 16 GB | 24 GB |
| g5.2xlarge | 1x A10G | 8 | 32 GB | 24 GB |
| g5.4xlarge | 1x A10G | 16 | 64 GB | 24 GB |
| g5.8xlarge | 1x A10G | 32 | 128 GB | 24 GB |

---

## Troubleshooting and Monitoring

### Check Job Status

```bash
# AWS Batch job status
aws batch describe-jobs --jobs <job-id> --region us-west-2 --profile memverge | jq '.jobs[0] | {status, statusReason}'

# MemVerge job status
curl -sk https://35.90.252.151:8080/api/v1/job | jq '.[] | select(.id=="<job-id>")'
```

### View Logs

```bash
# CloudWatch logs
aws logs tail /aws/batch/miratyper-training --follow --profile memverge

# Filter for specific job
aws logs filter-log-events \
    --log-group-name "/aws/batch/job" \
    --filter-pattern "<job-id>" \
    --region us-west-2 --profile memverge
```

### Monitor MemVerge

```bash
# Check registered nodes
curl -sk https://35.90.252.151:8080/api/v1/node | jq '.[] | {id, status, instanceType}'

# View checkpoint status
curl -sk https://35.90.252.151:8080/api/v1/job | jq '.[] | {id, checkpointAttempts, checkpointSucceeded, status}'
```

### Common Issues and Solutions

1. **Job fails with "ECS unable to assume role"**
   - Ensure execution role exists: `genept-training-execution-role`
   - Has AmazonECSTaskExecutionRolePolicy attached

2. **No GPU detected in container**
   - Add GPU resource requirement in job definition
   - Ensure using GPU-enabled instance types (g5.*)

3. **MemVerge node not registering**
   - Check user data includes MemVerge agent installation
   - Verify ECS_CLUSTER name matches actual Batch cluster
   - Check connectivity to MemVerge server (172.31.19.221:8080)

4. **Jobs not appearing in MemVerge UI**
   - Jobs only appear when containers are RUNNING
   - Check if instance has registered: `curl -sk https://35.90.252.151:8080/api/v1/node`

### Test Job Definitions

**Simple GPU Test**:
```bash
aws batch submit-job \
    --job-name gpu-test \
    --job-queue miratyper-memverge-queue \
    --job-definition test-gpu-env \
    --region us-west-2 --profile memverge
```

**Debug Environment Test**:
```bash
aws batch submit-job \
    --job-name debug-test \
    --job-queue miratyper-memverge-queue \
    --job-definition test-debug-job:2 \
    --region us-west-2 --profile memverge
```

---

## Key Configurations

### MemVerge Checkpoint Settings
- **Mode**: Iterative
- **Interval**: 15 minutes
- **Path**: `/mmc-checkpoint` (local scratch)
- **Root FS Diff**: Enabled
- **TCP Close**: Enabled

### Spot Instance Configuration
- **Bid Percentage**: 80%
- **Allocation Strategy**: SPOT_CAPACITY_OPTIMIZED
- **Cost Savings**: ~60% from on-demand pricing
- **Protection**: Automatic checkpoint/restore on interruption

### Data Access
- **Training Data**: Mounted via EBS snapshot at `/data`
- **Scratch Space**: Instance store at `/scratch`
- **Checkpoint Storage**: `/scratch/mmc-checkpoint`
- **Model Outputs**: `/scratch/outputs`

---

## Documentation References

- **MemVerge Docs**: Available in `/aws_batch/training/MEMVERGE_SUMMARY.md`
- **Integration Status**: `/aws_batch/training/MEMVERGE_FINAL_STATUS.md`
- **Integration Summary**: `/aws_batch/training/MEMVERGE_INTEGRATION_SUMMARY.md`
- **AWS Batch Docs**: https://docs.aws.amazon.com/batch/
- **MemVerge Dashboard**: https://35.90.252.151:8080

---

Last Updated: 2025-09-13
Status: ✅ Fully Operational with MemVerge Integration