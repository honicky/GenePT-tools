#!/bin/bash
# MemVerge Test Job Submission Script
# Tests hyperparameter tuning with checkpoint/restore on spot instances

# Configuration
JOB_NAME="memverge-tuning-test-$(date +%Y%m%d-%H%M%S)"
CONFIG_S3="s3://miratyper-training-configs/tuning_config_memverge_test.yaml"
WANDB_PROJECT="memverge-tuning-test"

echo "========================================="
echo "MemVerge Hyperparameter Tuning Test Job"
echo "========================================="
echo "Job Name: $JOB_NAME"
echo "Config: $CONFIG_S3"
echo "WandB Project: $WANDB_PROJECT"
echo ""

# Submit job using AWS CLI
echo "Submitting job to AWS Batch..."
JOB_ID=$(aws batch submit-job \
    --job-name "$JOB_NAME" \
    --job-queue miratyper-memverge-queue \
    --job-definition genept-training-job \
    --region us-west-2 \
    --profile memverge \
    --parameters config_file="$CONFIG_S3",wandb_project="$WANDB_PROJECT" \
    --container-overrides '{
        "environment": [
            {"name": "ENABLE_MEMVERGE_CHECKPOINT", "value": "true"},
            {"name": "CHECKPOINT_INTERVAL", "value": "300"},
            {"name": "TUNING_MODE", "value": "true"}
        ],
        "resourceRequirements": [
            {"type": "GPU", "value": "1"},
            {"type": "MEMORY", "value": "32768"},
            {"type": "VCPU", "value": "8"}
        ]
    }' \
    --output json | jq -r '.jobId')

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job"
    exit 1
fi

echo "✅ Job submitted successfully!"
echo "Job ID: $JOB_ID"
echo ""

# Provide monitoring commands
echo "========================================="
echo "Monitoring Commands:"
echo "========================================="
echo ""
echo "1. Check AWS Batch job status:"
echo "   aws batch describe-jobs --jobs $JOB_ID --region us-west-2 --profile memverge | jq '.jobs[0] | {status, statusReason}'"
echo ""
echo "2. Monitor with Python script:"
echo "   python submit_job.py --monitor-only --job-id $JOB_ID"
echo ""
echo "3. View CloudWatch logs (once running):"
echo "   aws logs tail /aws/batch/job --follow --profile memverge --region us-west-2"
echo ""
echo "4. Check MemVerge status:"
echo "   curl -sk https://35.90.252.151:8080/api/v1/job | jq '.[] | select(.batchJobIds[] == \"$JOB_ID\")'"
echo ""
echo "5. View MemVerge dashboard:"
echo "   https://35.90.252.151:8080"
echo ""
echo "6. Monitor WandB project:"
echo "   https://wandb.ai/YOUR_USERNAME/$WANDB_PROJECT"
echo ""

# Optional: Start monitoring immediately
read -p "Start monitoring now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting job monitor..."
    while true; do
        STATUS=$(aws batch describe-jobs --jobs "$JOB_ID" --region us-west-2 --profile memverge --output json | jq -r '.jobs[0].status')
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$TIMESTAMP] Status: $STATUS"
        
        if [[ "$STATUS" == "SUCCEEDED" ]] || [[ "$STATUS" == "FAILED" ]]; then
            echo "Job completed with status: $STATUS"
            if [[ "$STATUS" == "FAILED" ]]; then
                REASON=$(aws batch describe-jobs --jobs "$JOB_ID" --region us-west-2 --profile memverge --output json | jq -r '.jobs[0].statusReason')
                echo "Failure reason: $REASON"
            fi
            break
        fi
        
        sleep 30
    done
fi