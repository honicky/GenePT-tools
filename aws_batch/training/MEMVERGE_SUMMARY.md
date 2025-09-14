# MemVerge Memory Machine Batch (MMBatch) User Guide
## Comprehensive Documentation for AWS Infrastructure Development

---

## Overview

Memory Machine Batch (MMBatch) is an Application Program Interface used within Cloud Service Provider's batch services that provides checkpoint and restore functionality when using Spot instances. It captures the entire running state of a Batch Job into a consistent image and restores the Job on a new Compute Instance without losing any work progress. This plugin ensures high quality service at the Batch level by using low-cost, but unreliable Spot-based Compute Instances.

**Key Capability: SpotSurfing** - The ability to more reliably use spot instances through checkpoint and restore functionality, available for select CPU Instances and GPU Instances.

---

## Architecture

MMBatch consists of several key components:

### Core Components

1. **MMBatch Management Server**
   - Functions as both an API server and a central point for metrics collection
   - Global checkpoint configuration management
   - Interactive single-page application (SPA) for metrics display
   - Provides reporting for spot reclaim protection visibility and estimated time savings

2. **Memory Machine AWS Batch Engine (MMAB)**
   - Allows central configuration and visibility of distributed Memory Machine Engines
   - Runs as a service on management instances
   - Provides API endpoints for configuration and monitoring

3. **Agent Components (pagent)**
   - Deployed on worker nodes
   - Manages checkpoint and restore operations
   - Communicates with the management server
   - Reports job status and handles spot instance interruptions

---

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- Access to AWS Batch service
- Understanding of AWS EC2, VPC, and storage services
- Basic knowledge of containerized applications

### Installation Overview

The installation process involves:

1. **Management Server Setup**
   - Deploy MMBatch Management Server instance
   - Configure API endpoints and monitoring
   - Set up authentication (optional: AWS Cognito integration)

2. **Worker Node Configuration**
   - Install Memory Machine Engine on AWS Batch compute instances
   - Modify EC2 Launch Templates to include MMBatch components
   - Configure checkpoint storage locations

3. **AWS Batch Integration**
   - Update Compute Environments to use modified Launch Templates
   - Configure Job Queues to use MMBatch-enabled environments
   - Set up appropriate IAM roles and permissions

---

## Management Server Features

### Reporting and Metrics

The Management Server provides comprehensive reporting including:

- **Total Jobs**: Number of AWS Batch jobs with MMBatch enabled
- **Spot Protections**: Total spot reclaim protections provided
- **Total Job CPU Hours**: CPU hours requested by all MMBatch jobs
- **Total EC2 Instance Cost**: Estimated total cost for all jobs
- **EC2 Spot Savings**: Cost savings from restoring preempted spot instances
- **EC2 On-demand Savings**: Estimated savings from using spot vs on-demand instances

### Configuration Options

**Checkpoint Configuration:**
- Enable/disable checkpointing for spot reclaim protection
- Configure interval between checkpoints
- Set checkpoint image path (default: `/mmc-checkpoint`)
- Configure checkpoint mode (iterative)
- Enable root filesystem differential checkpoints
- Set diagnosis mode and checkpoint on SIGTERM

**Job EBS Volume Management:**
- Enable/disable managed EBS feature
- Configure EBS volume type, size, mount path
- Set custom tags for volumes

**Logging Configuration:**
- Configure log levels and file sizes
- Set log file rotation policies
- Access comprehensive log bundles for troubleshooting

**Server Configuration:**
- Configure port access to MMBatch Management Server
- Set certificate and private key file paths
- Enable HTTPS access

**Authentication:**
- AWS Cognito integration for sign-in
- API token management
- User access control

---

## Jobs Management

### Job Status Overview

The Jobs view provides monitoring and filtering capabilities for all batch jobs:

**Job Statuses:**
- Creating: Job is being created
- Created: Job has been created
- Running: Job is currently running
- Failed: Job has failed
- Succeeded: Job completed successfully
- Restoring: Job is restoring from checkpoint
- Restore Succeeded: Restore completed, job continues
- Restore Failed: Restore operation failed
- Stopped: Job has stopped
- Checkpointing: Checkpoint being generated
- Checkpoint Succeeded: Checkpoint completed successfully
- Checkpoint Failed: Checkpoint operation failed
- Volume Unready: Managed EBS volume not available at job start
- Restore Volume Unready: EBS volume not found during restore

### Monitoring Features

- **Export to CSV**: Download job data for offline analysis
- **Real-time Refresh**: Update job table with latest data
- **Advanced Filtering**: Filter by queue, status, creation date
- **Search Functionality**: Find jobs by ID, name, or other attributes
- **Event Tracking**: View job events with timestamps
- **Log Access**: Access and download comprehensive log bundles

### Log Bundle Components

| File Name | Level | Purpose | Troubleshooting Use |
|-----------|-------|---------|-------------------|
| (server)-access.log | Server | Access Log | Track requests to MMBatch server |
| (server).log | Server | Server Log | All MMBatch server activity |
| pagent.log | Node | Agent Log | Node-to-server communication, job status |
| mmrunc.log | Node | Container File | Job container events |
| output.log | Job | Container Output | Last 100 lines of container output |
| restore.log | Job | Restore Log | Restore operation details |
| ...dump.log | Job | Dump File | Checkpoint operation details |

---

## Storage Configuration

### Supported Storage Types

MMBatch supports various storage systems for checkpoint data:

#### AWS EFS Configuration

```bash
# Create mount point and mount EFS
mkdir -p /mmc-checkpoint
mount -t efs ${BatchEFSFileSystem}:/ /mmc-checkpoint
echo "${BatchEFSFileSystem}:/ /mmc-checkpoint efs defaults,_netdev 0 0" >> /etc/fstab
chown ec2-user:ec2-user /mmc-checkpoint
```

#### JuiceFS with AWS S3 Configuration

```bash
mkdir -p /mmc-checkpoint
chmod 777 /mmc-checkpoint
curl -sSL https://d.juicefs.com/install | sh -

# Format and mount JuiceFS
/usr/local/bin/juicefs format \
    --storage s3 \
    --bucket https://${JuiceFSS3BucketName}.s3.${AWS::Region}.amazonaws.com \
    --trash-days=0 \
    "rediss://${RedisClusterEndpoint}:6379/1" \
    juicefs-metadata

nohup /usr/local/bin/juicefs mount \
    "rediss://${RedisClusterEndpoint}:6379/1" \
    --cache-dir /mnt/jfs_cache \
    --cache-size 102400 \
    /mnt/jfs > /tmp/juicefs-mount.log 2>&1 &

# Wait for mount and create checkpoint directory
MOUNTPOINT=/mnt/jfs
CHECKPOINT_DIR=$MOUNTPOINT/mmc-checkpoint
mkdir -p $CHECKPOINT_DIR
chmod 777 $CHECKPOINT_DIR

# Create symlink
ln -sf $CHECKPOINT_DIR /mmc-checkpoint
```

### Storage Requirements

- Checkpoint storage must be accessible from all compute nodes
- Sufficient space for application state snapshots
- High-performance I/O for checkpoint/restore operations
- Persistent storage that survives instance termination

---

## Compute Instance Configuration

### CPU Architecture Compatibility

**Critical Requirement**: Checkpoints created on one CPU architecture must be restored on compatible architecture.

**Compatible Instance Type Groups:**

**x86_64 Intel/AMD Group:**
- c5, c5n, c5d, c6i, c6id, c6in, c6a, c6ad
- m5, m5n, m5d, m5a, m5ad, m5dn, m5zn, m6i, m6id, m6in, m6a, m6ad
- r5, r5n, r5d, r5a, r5ad, r5dn, r6i, r6id, r6in, r6a, r6ad
- t3, t3a
- And other x86_64 compatible instances

**ARM64 Graviton Group:**
- c6g, c6gn, c6gd, c7g, c7gn, c7gd
- m6g, m6gd, m7g, m7gd
- r6g, r6gd, r7g, r7gd
- t4g
- And other ARM64 compatible instances

### Launch Template Configuration

When modifying EC2 Launch Templates for MMBatch integration:

```bash
# Install MMBatch components
curl -k https://$MMAB_SERVER_ADDRESS/api/v1/scripts/install-pagent | bash
```

Key considerations:
- Add MMBatch installation to Launch Template user data
- Ensure security group allows HTTPS access to MMBatch Management Server
- Configure checkpoint storage mount points
- Set appropriate IAM roles for checkpoint storage access

---

## Working with Applications

### Cromwell Integration

MMBatch supports Cromwell workflow engine integration with AWS Batch:

#### Prerequisites
- Cromwell deployment with AWS Batch backend
- MMBatch Management Server running and accessible
- Proper security group configuration

#### Configuration Steps

1. **Enable Checkpointing for All Jobs:**
```bash
curl -sk -X PUT http://localhost:8080/api/v1/ckptConfig \
  -H "Content-Type: application/json" \
  -d '{
    "ckptMode":"iterative",
    "ckptImagePath":"/mmc-checkpoint",
    "ckptInterval":120000000000,
    "rootFSDiff":true,
    "diagnosisMode":true,
    "ckptOnSigTerm":true
  }' | jq .
```

2. **CloudFormation Template Setup:**
Use CloudFormation templates for automated deployment:
- AWS Batch queue with spot and on-demand compute environments
- Security groups for MMBatch server communication
- IAM roles with appropriate permissions

3. **Launch Template Modification:**
- Create new version of existing Launch Template
- Add MMBatch installation script to user data
- Set default version and apply to Compute Environment

4. **Job Queue Update:**
- Clone existing Compute Environment with new Launch Template
- Update Job Queue to use new Compute Environment
- Verify ARN configuration for job submissions

#### Example Workflow

```wdl
workflow helloWorld {
  String name
  call sayHello { input: name=name }
}

task sayHello {
  String name
  command {
    for i in $(seq 1 90); do
      printf "[cromwell-say-hello] Iteration $i: hello to ${name} on $(date)\n"
      sleep 10
    done
  }
  output {
    String out = read_string(stdout())
  }
  runtime {
    docker: "archlinux:latest"
    maxRetries: 3
  }
}
```

### Spot Instance Handling

When spot instances are interrupted:
1. MMBatch detects interruption signal
2. Triggers final checkpoint for all containers
3. New instance starts automatically
4. Job restores from checkpoint and continues
5. No work progress is lost

---

## Advanced Configuration

### API Configuration

**Checkpoint Configuration via API:**
```json
{
  "ckptMode": "iterative",
  "ckptImagePath": "/mmc-checkpoint", 
  "ckptInterval": 120000000000,
  "rootFSDiff": true,
  "diagnosisMode": true,
  "ckptOnSigTerm": true
}
```

**Configuration Parameters:**
- `ckptMode`: Checkpoint mode (iterative recommended)
- `ckptImagePath`: Path for storing checkpoint images
- `ckptInterval`: Interval between checkpoints (nanoseconds)
- `rootFSDiff`: Enable root filesystem differential checkpoints
- `diagnosisMode`: Enable diagnostic logging
- `ckptOnSigTerm`: Checkpoint on termination signal

### AWS Cognito Integration

For enhanced security and user management:
- Configure AWS Cognito user pools
- Set up authentication tokens
- Integrate with MMBatch Management Server
- Enable single sign-on capabilities

### Managed EBS Features

- Automatic EBS volume creation and attachment
- Volume type configuration (gp2, gp3, io1, io2)
- Custom volume sizes based on workload requirements
- Automatic volume cleanup after job completion
- Custom tagging for cost management

---

## CloudFormation Integration

### Infrastructure as Code

MMBatch supports infrastructure deployment through CloudFormation templates:

#### Key Components for CloudFormation

**Required Resources:**
- AWS Batch Compute Environment with spot instances
- Job Queues connecting to compute environments  
- IAM roles for batch jobs and instance profiles
- Security groups allowing MMBatch server communication
- VPC and networking components
- Storage resources (EFS, S3 buckets for JuiceFS)

**Example Resource Structure:**
```yaml
# Compute Environment with MMBatch-enabled Launch Template
BatchComputeEnvironment:
  Type: AWS::Batch::ComputeEnvironment
  Properties:
    Type: MANAGED
    ServiceRole: !Ref BatchServiceRole
    ComputeResources:
      Type: EC2
      AllocationStrategy: SPOT_CAPACITY_OPTIMIZED
      InstanceTypes: [c5.large, c5.xlarge, c5.2xlarge]
      LaunchTemplate:
        LaunchTemplateId: !Ref MMBatchLaunchTemplate
        Version: !GetAtt MMBatchLaunchTemplate.LatestVersionNumber

# Launch Template with MMBatch installation
MMBatchLaunchTemplate:
  Type: AWS::EC2::LaunchTemplate
  Properties:
    LaunchTemplateName: !Sub "${AWS::StackName}-mmbatch-template"
    LaunchTemplateData:
      UserData: !Base64
        Fn::Sub: |
          #!/bin/bash
          # Standard AWS Batch initialization
          # ... existing user data ...
          
          # Install MMBatch components
          curl -k https://${MMBatchServerAddress}/api/v1/scripts/install-pagent | bash
```

### Storage CloudFormation Examples

**EFS Configuration:**
```yaml
BatchEFSFileSystem:
  Type: AWS::EFS::FileSystem
  Properties:
    CreationToken: !Sub "${AWS::StackName}-batch-efs"
    PerformanceMode: generalPurpose
    ThroughputMode: provisioned
    ProvisionedThroughputInMibps: 100

BatchEFSMountTarget:
  Type: AWS::EFS::MountTarget  
  Properties:
    FileSystemId: !Ref BatchEFSFileSystem
    SubnetId: !Ref PrivateSubnet
    SecurityGroups: [!Ref EFSSecurityGroup]
```

**JuiceFS with S3 and Redis:**
```yaml
JuiceFSS3Bucket:
  Type: AWS::S3::Bucket
  Properties:
    BucketName: !Sub "${AWS::StackName}-juicefs-storage"

RedisCluster:
  Type: AWS::ElastiCache::ReplicationGroup  
  Properties:
    ReplicationGroupDescription: "Redis cluster for JuiceFS metadata"
    NumCacheClusters: 1
    Engine: redis
    CacheNodeType: cache.t3.micro
```

---

## Troubleshooting

### Common Issues and Solutions

#### Checkpoint Failures
- Verify checkpoint storage accessibility from compute nodes
- Check storage space availability  
- Confirm proper mount point configuration
- Review pagent.log and dump.log files

#### Restore Failures  
- Ensure CPU architecture compatibility between checkpoint and restore instances
- Verify checkpoint data integrity
- Check restore.log for detailed error information
- Confirm network connectivity to checkpoint storage

#### Performance Issues
- Monitor checkpoint interval settings
- Evaluate storage performance (IOPS, throughput)
- Review instance types for compute requirements
- Check for network bottlenecks in storage access

#### Authentication Problems
- Verify AWS Cognito configuration if enabled
- Check API token validity and refresh requirements  
- Confirm IAM role permissions for storage and compute access
- Review security group rules for MMBatch server communication

### Log Analysis

**Event-Driven Reporting Characteristics:**
- Real-time event capture as discrete operations occur
- Granular detail with precise sequence recording
- High fidelity reflecting exact operation order
- Scalable for high-volume data streams

**Potential Challenges:**
- Data inconsistency with other application layers
- Processing delays in dependent systems
- Complex system maintenance requirements  
- High data volume impacting storage costs

### Best Practices

1. **Architecture Planning:**
   - Group compatible instance types in compute environments
   - Plan checkpoint storage capacity based on application memory usage
   - Design for network connectivity between all components

2. **Security Configuration:**
   - Use IAM roles with least privilege principles
   - Secure checkpoint storage with appropriate access controls  
   - Enable encryption for sensitive checkpoint data
   - Monitor API access and usage patterns

3. **Performance Optimization:**
   - Tune checkpoint intervals based on job characteristics
   - Use high-performance storage for checkpoint operations
   - Monitor spot instance pricing and availability patterns
   - Implement proper resource tagging for cost management

4. **Monitoring and Alerting:**
   - Set up CloudWatch monitoring for MMBatch metrics
   - Configure alerts for checkpoint/restore failures
   - Monitor cost savings and optimization opportunities
   - Track job completion rates and performance trends

---

## API Reference

### Complete API Documentation

#### Configuration Management

**GET /api/v1/config**
Retrieve the complete server and application configuration.

*Sample Response:*
```json
{
  "addr": "https://0.0.0.0:8081",
  "id": "f3c3be48-0f71-4a0c-b4e7-8fd36f37061c",
  "staticFolder": "mmabWeb",
  "security": {
    "certFile": "/home/ec2-user/.memverge/mmab/conf/server.crt",
    "keyFile": "/home/ec2-user/.memverge/mmab/conf/server.pem",
    "cognito": {
      "enabled": false,
      "userPoolID": "",
      "identityPoolID": "",
      "clientID": "",
      "adminGroups": ["admin"]
    }
  },
  "ckpt": {
    "ckptMode": "iterative",
    "ckptImagePath": "/mmc-checkpoint",
    "ckptInterval": "1h0m0s",
    "ckptFiles": [],
    "IRMapScanPaths": [],
    "ckptOnSigTerm": false,
    "diagnosisMode": true,
    "rootFSDiff": false,
    "cloudWatchMode": false,
    "tcpClose": false
  },
  "node": {
    "heartbeat": "30s",
    "ttl": "5m0s",
    "maxPerLogSizeMB": 2,
    "maxNodeLogTotalMB": 1024,
    "cleanLogInterval": "12h0s"
  },
  "job": {
    "ebsPerJob": true,
    "customTags": {
      "owner": "cedric",
      "team": "engineer"
    },
    "ebsMountPath": "/mnt/mmab",
    "diskType": "gp3",
    "diskSizeGB": 100,
    "storCleanIntvl": "24h0m0s",
    "retentionPolicy": "time",
    "retentionInterval": "1h0m0s",
    "successTTL": "72h0m0s",
    "failureTTL": "168h0m0s"
  }
}
```

**PUT /api/v1/configKV**
Change specific configuration values using key-value pairs.

*Request Format:*
```json
{"kvMap": {"<key>":"<value>"}}
```

*Examples:*
```json
{"kvMap": {"node.ttl": "5m"}}
{"kvMap": {"security.cognito.adminGroups": "admin,root"}}
{"kvMap": {"job.customTags": "owner:name,team:engineer"}}
```

**Configuration Properties:**
- `addr` - Server address
- `security.certFile` - Certificate file location
- `security.keyFile` - Private key file location
- `security.cognito.enabled` - Enable Cognito authentication
- `security.cognito.userPoolID` - Cognito user pool ID
- `security.cognito.identityPoolID` - Cognito identity pool ID
- `security.cognito.clientID` - SPA client ID
- `security.cognito.adminGroups` - Administrator groups
- `ckpt.ckptMode` - Checkpoint mode (iterative/none)
- `ckpt.ckptImagePath` - Checkpoint image storage path
- `ckpt.ckptInterval` - Interval between checkpoints
- `ckpt.rootFSDiff` - Include root filesystem in checkpoint
- `ckpt.tcpClose` - Close TCP connections during checkpoint
- `node.heartbeat` - Heartbeat interval
- `node.ttl` - Node time-to-live
- `job.ebsPerJob` - Create EBS volume per job
- `job.ebsMountPath` - EBS mount path
- `job.diskType` - EBS volume type
- `job.diskSizeGB` - EBS volume size
- `job.retentionPolicy` - Job retention policy
- `job.successTTL` - Success job retention time
- `job.failureTTL` - Failed job retention time

#### Logging Configuration

**GET /api/v1/log/config**
Retrieve server log configuration.

*Sample Response:*
```json
{
  "level": "info",
  "maxSizeMB": 10,
  "maxBackups": 10
}
```

**PUT /api/v1/log/config**
Update server log configuration.

*Request Body:*
```json
{
  "level": "debug"
}
```

#### Node Management

**GET /api/v1/node**
List all running worker nodes.

*Sample Response:*
```json
[
  {
    "id": "0e0c7b08-32cc-4737-a2ea-9e2a2bfc7fd1",
    "ips": ["172.31.1.234"],
    "hostName": "ip-172-31-1-234.us-west-1.compute.internal",
    "cloud": "aws",
    "arch": "x86_64",
    "cores": 2,
    "cpuModel": "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
    "cpuVendor": "GenuineIntel",
    "memoryInMB": 7638,
    "instance": {
      "zone": "us-west-1a",
      "instanceId": "i-044f3323bfd228fd6",
      "instanceType": "m6i.large",
      "region": "us-west-1",
      "createTime": "2025-03-11T23:05:22Z",
      "payType": "Spot"
    },
    "lastHeartbeat": "2025-03-11T23:12:14.07346093Z"
  }
]
```

**GET /api/v1/nodes/{nodeID}/files**
List available log files for a specific node.

*Sample Response:*
```json
[
  "/nodes/<nodeID>/var/log/cloud-init-output.log",
  "/nodes/<nodeID>/var/log/memverge/mmrunc.log",
  "/nodes/<nodeID>/var/log/memverge/pagent.log"
]
```

**GET /nodes/{nodeID}/{filePath}**
Retrieve the content of a specific node log file.

*Example:*
```
GET /nodes/<nodeID>/var/log/memverge/mmrunc.log
```

#### Log File Access

**GET /mmab.log**
Retrieve the latest MMBatch server log.

**GET /mmab-access.log**
Retrieve the server access log.

*Note: Log file paths change if log file names are modified in configuration.*

#### Job Management

**GET /api/v1/job**
List all jobs with complete details and event history.

*Sample Response:*
```json
[
  {
    "id": "8252d813-e5fb-4dad-b397-cb518fd0fc41",
    "queueName": "jacky-test",
    "createdAt": "2025-05-15T05:29:06.354967467Z",
    "updatedAt": "2025-05-15T06:35:48.011660015Z",
    "status": "Succeeded",
    "nodeOid": "i-03d7eb4de55c4355f",
    "containerId": "6fc76e79d59123f6f35a7c9d7507885664d6983a64ebe231fa1391505a520c40",
    "spotProtCount": 1,
    "batchJobIds": ["8252d813-e5fb-4dad-b397-cb518fd0fc41"],
    "volumeIds": ["vol-022893f19655019e0"],
    "events": [
      {
        "timestamp": "2025-05-15T05:29:06.354967467Z",
        "eventType": "Job-Creating",
        "nodeOid": "i-0bf682f18f3cb557e",
        "containerId": "9098f44bfd84597ce82f13974ab8400b894a3595fdb51ec0ff5c835e4bf1fedf",
        "batchJobId": "8252d813-e5fb-4dad-b397-cb518fd0fc41"
      },
      {
        "timestamp": "2025-05-15T05:29:12.49788509Z",
        "eventType": "Volume-Created",
        "volumnId": "vol-022893f19655019e0"
      },
      {
        "timestamp": "2025-05-15T05:29:19.49788509Z",
        "eventType": "Volume-Attached",
        "volumnId": "vol-022893f19655019e0"
      },
      {
        "timestamp": "2025-05-15T05:29:20.49788509Z",
        "eventType": "Job-Created"
      },
      {
        "timestamp": "2025-05-15T05:29:20.675862549Z",
        "eventType": "Job-Running"
      },
      {
        "timestamp": "2025-05-15T05:55:06.194790675Z",
        "eventType": "Job-Checkpointing"
      },
      {
        "timestamp": "2025-05-15T05:55:06.894972615Z",
        "eventType": "Job-CheckpointSucceeded"
      },
      {
        "timestamp": "2025-05-15T06:00:22.278832037Z",
        "eventType": "Job-Restoring",
        "nodeOid": "i-03d7eb4de55c4355f",
        "containerId": "6fc76e79d59123f6f35a7c9d7507885664d6983a64ebe231fa1391505a520c40"
      },
      {
        "timestamp": "2025-05-15T06:01:26.010124985Z",
        "eventType": "Job-RestoreSucceeded"
      },
      {
        "timestamp": "2025-05-15T06:01:26.083797074Z",
        "eventType": "Job-Running"
      },
      {
        "timestamp": "2025-05-15T06:35:48.011660015Z",
        "eventType": "Job-Succeeded"
      }
    ]
  }
]
```

**Job Event Types:**
- `Job-Creating` - Job initialization
- `Job-Created` - Job successfully created
- `Job-Running` - Job actively running
- `Job-Checkpointing` - Checkpoint operation in progress
- `Job-CheckpointSucceeded` - Checkpoint completed successfully
- `Job-Restoring` - Restore operation in progress
- `Job-RestoreSucceeded` - Restore completed successfully
- `Job-Succeeded` - Job completed successfully
- `Volume-Created` - EBS volume created
- `Volume-Attached` - EBS volume attached to instance

#### Metrics and Monitoring

**GET /api/v1/metric**
Retrieve list of all available system metrics.

*Sample Response:*
```json
[
  {
    "name": "Total runtime of jobs",
    "id": "metricDef-runtime.system-total",
    "definition": {
      "id": "metricDef-runtime",
      "description": "Total runtime of jobs",
      "labels": ["duration"]
    },
    "object": {
      "id": "system-total",
      "type": "system-total",
      "name": "system-total"
    },
    "levels": [
      {
        "interval": "1m0s",
        "retention": "168h0m0s"
      },
      {
        "interval": "24h0m0s",
        "retention": "18000h0m0s"
      },
      {
        "interval": "30m0s",
        "retention": "2160h0m0s"
      }
    ]
  }
]
```

**GET /api/v1/metricValue/{ObjectID}/{MetricDefinitionID}**
Retrieve specific metric values with optional time range and interval filtering.

*Query Parameters:*
- `interval` (string, optional) - Metric value interval
- `start` (time.Time, optional) - Start of time range
- `end` (time.Time, optional) - End of time range

*Example Request:*
```
GET /api/v1/metricValue/system-total/metricDef-volumeAttachTime?interval=1m&end=2025-03-15T00:00:00Z
```

*Sample Response:*
```json
{
  "id": "metricDef-volumeAttachTime.system-total",
  "points": [
    {
      "time": "2025-03-13T04:51:00Z",
      "value": 6.741566292
    },
    {
      "time": "2025-03-13T05:40:00Z",
      "value": 6.554704421
    }
  ],
  "metaData": {}
}
```

**GET /api/v1/metrics/summary**
Retrieve summary of metrics within a specified time range.

*Query Parameters:*
- `start` (time.Time, optional) - Start of time range
- `end` (time.Time, optional) - End of time range

*Example Request:*
```
GET /api/v1/metrics/summary?start=2025-01-01T00:00:00Z&end=2025-01-07T23:59:59Z
```

*Sample Response:*
```json
{
  "items": {
    "jobSubmitted": [
      {
        "id": "jobSubmitted",
        "point": {
          "time": "2025-01-01T12:00:00Z",
          "value": 150.0
        },
        "metaData": {
          "queueName": "queue1"
        }
      }
    ],
    "runtime": [
      {
        "id": "runtime",
        "point": {
          "time": "2025-01-02T12:00:00Z",
          "value": 300.5
        },
        "metaData": {
          "queueName": "queue1"
        }
      }
    ],
    "spotProtection": [
      {
        "id": "spotProtection",
        "point": {
          "time": "2025-01-02T12:00:00Z",
          "value": 3
        },
        "metaData": {
          "queueName": "queue1"
        }
      }
    ],
    "timeSaved": [
      {
        "id": "timeSaved",
        "point": {
          "time": "2025-01-02T12:00:00Z",
          "value": 120.5
        },
        "metaData": {
          "queueName": "queue1"
        }
      }
    ]
  }
}
```

#### Authentication Requirements

**When Cognito is Enabled:**
- Token-based authentication required for all API calls
- Tokens must be refreshed periodically for extended sessions
- Integration with AWS Cognito user pools
- Admin group membership determines access levels

**API Access Patterns:**
- Use HTTPS for all API communications
- Include proper Content-Type headers for PUT/POST requests
- Handle authentication tokens in request headers
- Implement proper error handling for API responses

---

## Conclusion

MMBatch provides a robust solution for leveraging AWS Spot instances in batch computing workloads while maintaining high reliability through checkpoint and restore capabilities. The combination of cost savings from spot instances and reliability from MMBatch's SpotSurfing technology makes it an ideal choice for large-scale, cost-sensitive computational workloads.

Key benefits include:
- Significant cost reduction through spot instance utilization
- No loss of computational work due to spot interruptions  
- Comprehensive monitoring and reporting capabilities
- Integration with popular workflow managers like Cromwell
- Support for various storage backends and compute architectures
- Infrastructure as Code support through CloudFormation

For AWS infrastructure development, MMBatch enables the creation of resilient, cost-effective batch computing environments that can handle interruptions gracefully while providing detailed visibility into operations and cost savings.

---

*This document represents a comprehensive compilation of the MemVerge MMBatch User Guide based on the latest available documentation. For the most current information and updates, refer to the official MemVerge documentation at https://docs.memverge.com/MMBatch/latest/*