# Amazon Batch Data Sharing Strategy

## Recommended Approach: Hybrid S3 + EFS

### Architecture
1. **Long-term Storage**: S3 bucket for shuffled batches
2. **Training Cache**: EFS for active training runs
3. **Local Cache**: Container /tmp for current batch

### Implementation Steps

#### 1. Upload to S3
```bash
# Upload shuffled data to S3
aws s3 sync data/cellxgene_embeddings/training_v1_shuffled/ \
    s3://your-bucket/cellxgene/training_v1_shuffled/ \
    --storage-class INTELLIGENT_TIERING

# Verify upload
aws s3 ls s3://your-bucket/cellxgene/training_v1_shuffled/ --summarize
```

#### 2. Create EFS File System
```bash
# Create EFS
aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --tags "Key=Name,Value=cellxgene-training"

# Create mount targets in your VPC subnets
aws efs create-mount-target \
    --file-system-id fs-xxxxxx \
    --subnet-id subnet-xxxxxx \
    --security-groups sg-xxxxxx
```

#### 3. Configure Batch Compute Environment
```json
{
  "computeEnvironmentName": "training-env",
  "type": "MANAGED",
  "computeResources": {
    "type": "EC2",
    "launchTemplate": {
      "launchTemplateId": "lt-xxxxxx",
      "version": "$Latest"
    }
  }
}
```

#### 4. Launch Template User Data
```bash
#!/bin/bash
# Mount EFS on EC2 instances
yum install -y amazon-efs-utils
mkdir -p /mnt/efs
mount -t efs -o tls fs-xxxxxx:/ /mnt/efs

# Pre-cache popular batches from S3 to EFS (optional)
aws s3 sync s3://your-bucket/cellxgene/training_v1_shuffled/ \
    /mnt/efs/training_v1_shuffled/ \
    --exclude "*" --include "batch_00*.parquet"
```

#### 5. Container Training Script
```python
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import boto3
import hashlib

class BatchDataLoader:
    def __init__(self, 
                 s3_bucket='your-bucket',
                 s3_prefix='cellxgene/training_v1_shuffled',
                 efs_cache='/mnt/efs/training_v1_shuffled',
                 local_cache='/tmp/batch_cache'):
        
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.efs_cache = Path(efs_cache)
        self.local_cache = Path(local_cache)
        self.s3_client = boto3.client('s3')
        
        # Create cache directories
        self.efs_cache.mkdir(parents=True, exist_ok=True)
        self.local_cache.mkdir(parents=True, exist_ok=True)
    
    def get_batch(self, batch_id):
        """Load batch with hierarchical caching."""
        batch_file = f"batch_{batch_id:04d}.parquet"
        
        # Check local cache first (fastest)
        local_path = self.local_cache / batch_file
        if local_path.exists():
            return pq.read_table(local_path)
        
        # Check EFS cache (fast)
        efs_path = self.efs_cache / batch_file
        if efs_path.exists():
            # Copy to local cache for next access
            table = pq.read_table(efs_path)
            pq.write_table(table, local_path)
            return table
        
        # Download from S3 (slower)
        s3_key = f"{self.s3_prefix}/{batch_file}"
        self.s3_client.download_file(
            self.s3_bucket, s3_key, str(efs_path)
        )
        
        # Also cache locally
        table = pq.read_table(efs_path)
        pq.write_table(table, local_path)
        return table
    
    def get_batch_iterator(self, batch_ids, shuffle=True):
        """Iterator for training loops."""
        if shuffle:
            import random
            batch_ids = batch_ids.copy()
            random.shuffle(batch_ids)
        
        for batch_id in batch_ids:
            yield self.get_batch(batch_id).to_pandas()
```

#### 6. Batch Job Definition
```json
{
  "jobDefinitionName": "cellxgene-training",
  "type": "container",
  "containerProperties": {
    "image": "your-ecr-repo/training:latest",
    "vcpus": 4,
    "memory": 16384,
    "jobRoleArn": "arn:aws:iam::xxxx:role/BatchJobRole",
    "environment": [
      {"name": "S3_BUCKET", "value": "your-bucket"},
      {"name": "EFS_MOUNT", "value": "/mnt/efs"}
    ],
    "mountPoints": [
      {
        "sourceVolume": "efs-volume",
        "containerPath": "/mnt/efs"
      }
    ],
    "volumes": [
      {
        "name": "efs-volume",
        "host": {"sourcePath": "/mnt/efs"}
      }
    ]
  }
}
```

## Cost Optimization Tips

1. **Use S3 Intelligent Tiering**: Automatically moves less-accessed batches to cheaper storage
2. **EFS Lifecycle Management**: Auto-move infrequent files to EFS IA (cheaper tier)
3. **Spot Instances**: Use Spot for Batch compute environment (up to 90% savings)
4. **Regional Data Transfer**: Keep S3, EFS, and Batch in same region to avoid transfer costs

## Monitoring
```python
# Add CloudWatch metrics for data loading
import time
import boto3

cloudwatch = boto3.client('cloudwatch')

def timed_batch_load(batch_id):
    start = time.time()
    data = loader.get_batch(batch_id)
    duration = time.time() - start
    
    cloudwatch.put_metric_data(
        Namespace='CellXGene/Training',
        MetricData=[
            {
                'MetricName': 'BatchLoadTime',
                'Value': duration,
                'Unit': 'Seconds',
                'Dimensions': [
                    {'Name': 'BatchID', 'Value': str(batch_id)}
                ]
            }
        ]
    )
    return data
```

## Estimated Costs (Monthly)
- S3 Storage (88GB): ~$2
- S3 Data Transfer to same-region EC2: $0
- EFS Storage (88GB, standard): ~$26
- EFS Storage (88GB, with IA for 80%): ~$8
- Batch Compute: Depends on instance types and hours

## Alternative: Direct S3 Streaming
For simpler setup without EFS:

```python
import s3fs
import pyarrow.parquet as pq

fs = s3fs.S3FileSystem()

def stream_batch_from_s3(batch_id):
    """Direct streaming from S3 without local download."""
    path = f"s3://your-bucket/cellxgene/training_v1_shuffled/batch_{batch_id:04d}.parquet"
    with fs.open(path, 'rb') as f:
        return pq.read_table(f).to_pandas()
```

This approach works well if:
- You process batches sequentially
- Network bandwidth is sufficient
- You want minimal infrastructure complexity