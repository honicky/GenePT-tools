#!/bin/bash
# Script to monitor AWS Batch job logs

JOB_ID="${1:-32072e1b-a715-435d-a161-95aafcb99518}"
PROFILE="${2:-memverge}"
REGION="${3:-us-west-2}"

echo "Monitoring job: $JOB_ID"
echo "----------------------------------------"

# Get job status and log stream
JOB_INFO=$(aws batch describe-jobs \
  --jobs "$JOB_ID" \
  --region "$REGION" \
  --profile "$PROFILE" \
  --query "jobs[0].[status,container.logStreamName]" \
  --output json)

STATUS=$(echo "$JOB_INFO" | jq -r '.[0]')
LOG_STREAM=$(echo "$JOB_INFO" | jq -r '.[1]')

echo "Status: $STATUS"
echo "Log stream: $LOG_STREAM"
echo "----------------------------------------"

if [ "$LOG_STREAM" != "null" ] && [ -n "$LOG_STREAM" ]; then
    echo "Fetching recent logs..."
    
    # Get the last 50 log events
    aws logs filter-log-events \
        --log-group-name /aws/batch/job \
        --log-stream-names "$LOG_STREAM" \
        --profile "$PROFILE" \
        --region "$REGION" \
        --query "events[].message" \
        --output text | tail -50
    
    echo ""
    echo "----------------------------------------"
    echo "To follow logs in real-time, run:"
    echo "aws logs tail /aws/batch/job --follow --log-stream-names '$LOG_STREAM' --profile $PROFILE --region $REGION"
else
    echo "No log stream available yet. Job may be pending or starting."
fi

# Show job runtime if running
if [ "$STATUS" == "RUNNING" ]; then
    STARTED_AT=$(aws batch describe-jobs \
        --jobs "$JOB_ID" \
        --region "$REGION" \
        --profile "$PROFILE" \
        --query "jobs[0].startedAt" \
        --output text)
    
    if [ -n "$STARTED_AT" ]; then
        python3 -c "
import time
started = $STARTED_AT
now = int(time.time() * 1000)
mins = (now - started) / 60000
print(f'\nJob has been running for {mins:.1f} minutes')
"
    fi
fi