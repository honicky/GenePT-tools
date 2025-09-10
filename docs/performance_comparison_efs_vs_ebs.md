# Performance Comparison: EFS vs EBS Snapshots for Training Data

## Test Scenario: 88GB Training Data (376 batches × ~239MB each)

## Performance Metrics

### EBS from Snapshot

#### Setup Time
- **Snapshot creation**: 10-30 minutes (one-time)
- **Volume creation from snapshot**: 2-5 minutes per instance
- **Volume attachment**: < 1 minute
- **Total setup per job**: ~3-6 minutes

#### Read Performance
```
Type        | IOPS   | Throughput | Latency  | Cost/Month
------------|--------|------------|----------|------------
gp3         | 3,000  | 125 MB/s   | 1-2 ms   | $7.04 (88GB)
gp3 (tuned) | 16,000 | 1,000 MB/s | 1-2 ms   | $28.16
io2         | 64,000 | 1,000 MB/s | < 1 ms   | $92.40
```

#### Batch Loading Times
```python
# Expected performance for 239MB batch file
gp3 (base):   1.91 seconds per batch
gp3 (tuned):  0.24 seconds per batch  
io2:          0.24 seconds per batch
```

### EFS Performance

#### Setup Time
- **File system creation**: One-time setup
- **Mount on instance**: < 10 seconds
- **Total setup per job**: Essentially instant

#### Read Performance
```
Mode              | Throughput | Latency   | Cost/Month
------------------|------------|-----------|------------
Bursting          | 100 MB/s*  | 3-5 ms    | $26.40 (88GB)
Provisioned (100) | 100 MB/s   | 3-5 ms    | $32.40
Provisioned (500) | 500 MB/s   | 3-5 ms    | $56.40
Max Bursting      | 264 MB/s** | 3-5 ms    | $26.40

* Baseline throughput for 88GB storage
** Burst throughput (limited credits)
```

#### Batch Loading Times
```python
# Expected performance for 239MB batch file
EFS Baseline:  2.39 seconds per batch
EFS Burst:     0.90 seconds per batch
EFS Prov 500:  0.48 seconds per batch
```

## Detailed Performance Analysis

### Sequential Training (Processing All 376 Batches)

```python
# Total time to read all batches sequentially

EBS gp3 (base):
  376 × 1.91s = 718 seconds (12 minutes)
  
EBS gp3 (tuned):
  376 × 0.24s = 90 seconds (1.5 minutes)
  
EFS (baseline):
  376 × 2.39s = 898 seconds (15 minutes)
  
EFS (burst):
  376 × 0.90s = 338 seconds (5.6 minutes)
```

### Random Access Pattern

```python
# Loading random batches (typical training pattern)
# Assuming 10 epochs, random batch order each time

                 | First Load | Cached Load | 10 Epochs Total
-----------------|------------|-------------|------------------
EBS gp3          | 1.91s      | 1.91s       | 2.0 hours
EBS gp3 (tuned)  | 0.24s      | 0.24s       | 15 minutes
EFS (baseline)   | 2.39s      | 2.39s       | 2.5 hours  
EFS (with cache) | 2.39s      | 0.10s*      | 25 minutes

* Linux page cache makes subsequent reads much faster
```

### Concurrent Training Jobs

```python
# Multiple containers reading different batches simultaneously

Scenario: 10 concurrent training jobs
```

#### EBS Snapshot Approach
- Need 10 separate volumes (10 × 88GB = 880GB total)
- Each gets full throughput independently
- Cost: 10 × $7.04 = $70.40/month
- **Performance: No degradation**

#### EFS Approach  
- Single file system shared by all
- Throughput shared (but can burst)
- Cost: $26.40/month (same filesystem)
- **Performance: Depends on access pattern**

```python
# EFS concurrent performance
Total Available: 100-264 MB/s (burst)
Per Job: 10-26 MB/s average
Batch load time: 9-24 seconds per batch

# With provisioned throughput (500 MB/s)
Per Job: 50 MB/s average  
Batch load time: 4.8 seconds per batch
```

## Memory/Cache Optimization

### EBS Page Cache Behavior
```python
# First epoch - cold cache
for batch in range(376):
    data = read_batch(batch)  # 1.91s from disk
    train(data)

# Second epoch - warm cache (if enough RAM)
for batch in range(376):
    data = read_batch(batch)  # ~0.01s from RAM cache
    train(data)
```

### EFS Cache Behavior
```python
# EFS has built-in caching at multiple levels:
# 1. Client-side: Linux page cache (same as EBS)
# 2. EFS-side: Distributed cache across AZs

# First read from EFS
data = read_batch(0)  # 2.39s

# Second read (cached)
data = read_batch(0)  # 0.10-0.50s depending on cache hit
```

## Cost-Performance Trade-offs

### Small Scale (1-2 concurrent jobs)
**Winner: EBS gp3 (base)**
- Cost: $7-14/month
- Performance: 1.91s per batch
- Setup: 3-6 minutes

### Medium Scale (3-10 concurrent jobs)
**Winner: EFS (bursting)**
- Cost: $26.40/month (shared)
- Performance: 0.90-2.39s per batch
- Setup: Instant

### Large Scale (10+ concurrent jobs)
**Winner: EFS with provisioned throughput**
- Cost: $56.40/month for 500 MB/s
- Performance: Consistent 0.48s per batch
- Setup: Instant

### High Performance Single Job
**Winner: EBS gp3 (tuned) or io2**
- Cost: $28-92/month
- Performance: 0.24s per batch
- Setup: 3-6 minutes

## Recommendation Matrix

| Scenario | Best Option | Why |
|----------|------------|-----|
| Single experiments | EBS gp3 | Cheapest, good performance |
| Parameter sweeps | EFS | Share data, no duplication |
| Production training | EBS gp3 (tuned) | Predictable performance |
| Development/testing | EFS | Flexibility, instant setup |
| Budget-constrained | EFS (bursting) | Best $/performance ratio |
| Distributed training | EFS or FSx | Designed for parallel access |

## Practical Hybrid Approach

```python
class HybridDataLoader:
    """Use EBS for primary, EFS for overflow/sharing"""
    
    def __init__(self):
        self.primary = "/mnt/ebs/data"   # Fast EBS
        self.shared = "/mnt/efs/data"    # Shared EFS
        self.cache = "/dev/shm"          # RAM disk
        
    def get_batch(self, batch_id):
        # Try RAM first (fastest)
        if exists(f"{self.cache}/batch_{batch_id}.parquet"):
            return read(self.cache)  # ~0.01s
            
        # Try local EBS (fast)
        if exists(f"{self.primary}/batch_{batch_id}.parquet"):
            return read(self.primary)  # ~0.24-1.91s
            
        # Fall back to EFS (shared)
        return read(self.shared)  # ~0.90-2.39s
```

## Key Insights

1. **EBS wins for single-instance performance** (especially with tuned IOPS)
2. **EFS wins for multi-instance sharing** (no data duplication)
3. **Both benefit from Linux page cache** (subsequent reads are fast)
4. **EFS burst credits** can provide temporary high performance
5. **Setup time matters** for short jobs (EFS is instant)

## Testing Script

```bash
#!/bin/bash
# Performance test script

# Test EBS
echo "Testing EBS performance..."
time for i in {0..10}; do
    dd if=/mnt/ebs/batch_$(printf "%04d" $i).parquet \
       of=/dev/null bs=1M 2>/dev/null
done

# Test EFS  
echo "Testing EFS performance..."
time for i in {0..10}; do
    dd if=/mnt/efs/batch_$(printf "%04d" $i).parquet \
       of=/dev/null bs=1M 2>/dev/null
done

# Clear cache and retest
echo 3 > /proc/sys/vm/drop_caches
```