#!/bin/bash
set -e

# Wait for compute environment to be ready for deletion
echo "Waiting for compute environment to be ready for deletion..."
while true; do
    STATUS=$(aws batch describe-compute-environments \
        --compute-environments miratyper-gpu-training-spot \
        --region us-west-2 \
        --profile memverge \
        --query 'computeEnvironments[0].status' \
        --output text 2>/dev/null || echo "DELETED")
    
    if [ "$STATUS" == "DELETED" ] || [ "$STATUS" == "None" ]; then
        echo "Compute environment deleted"
        break
    elif [ "$STATUS" == "UPDATING" ]; then
        echo "Still updating, waiting..."
        sleep 10
    else
        echo "Status: $STATUS - attempting deletion..."
        aws batch delete-compute-environment \
            --compute-environment miratyper-gpu-training-spot \
            --region us-west-2 \
            --profile memverge 2>/dev/null || true
        sleep 10
    fi
done

# Get VPC information
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=is-default,Values=true" \
    --region us-west-2 \
    --profile memverge \
    --query 'Vpcs[0].VpcId' \
    --output text)

SUBNET1=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" \
    --region us-west-2 \
    --profile memverge \
    --query 'Subnets[0].SubnetId' \
    --output text)

SUBNET2=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" \
    --region us-west-2 \
    --profile memverge \
    --query 'Subnets[1].SubnetId' \
    --output text)

SECURITY_GROUP=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=${DEFAULT_VPC}" "Name=group-name,Values=default" \
    --region us-west-2 \
    --profile memverge \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "Creating new compute environment with MemVerge support..."

# Create new compute environment with MemVerge-enabled launch template
aws batch create-compute-environment \
    --compute-environment-name miratyper-gpu-training-spot \
    --type MANAGED \
    --state ENABLED \
    --service-role arn:aws:iam::971422677163:role/aws-batch-service-role \
    --compute-resources "{
        \"type\": \"SPOT\",
        \"bidPercentage\": 80,
        \"spotIamFleetRole\": \"arn:aws:iam::971422677163:role/aws-batch-spot-fleet-role\",
        \"minvCpus\": 0,
        \"maxvCpus\": 256,
        \"desiredvCpus\": 0,
        \"instanceTypes\": [\"g5.xlarge\", \"g5.2xlarge\", \"g5.4xlarge\"],
        \"allocationStrategy\": \"SPOT_CAPACITY_OPTIMIZED\",
        \"subnets\": [\"${SUBNET1}\", \"${SUBNET2}\"],
        \"securityGroupIds\": [\"${SECURITY_GROUP}\"],
        \"instanceRole\": \"arn:aws:iam::971422677163:instance-profile/ecsInstanceRole\",
        \"launchTemplate\": {
            \"launchTemplateName\": \"genept-training-gpu-template\",
            \"version\": \"3\"
        },
        \"tags\": {
            \"Project\": \"MiraTyper\",
            \"Environment\": \"training\",
            \"MemVerge\": \"enabled\"
        }
    }" \
    --region us-west-2 \
    --profile memverge

echo "Compute environment created with MemVerge support!"
echo ""
echo "Next steps:"
echo "1. Wait for compute environment to become VALID"
echo "2. The job queue should automatically reconnect"
echo "3. Submit a test job to verify MemVerge integration"