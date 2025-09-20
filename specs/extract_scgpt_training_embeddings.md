# Specification: Extract scGPT Embeddings for Shuffled Training Set

## Overview
Create a script that extracts scGPT embeddings for only the cells present in the shuffled training set, maintaining the same shuffled order and file structure. This will be executed on a dedicated EC2 instance, and the resulting data will be snapshotted as an EBS volume for use in AWS Batch training jobs.

## Background
- The shuffled training data in `s3://pythiomicsdata/cellxgene_v2/training_v1_shuffled/` contains cells from multiple origin files, shuffled for training
- The full scGPT embeddings are in `s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1/` organized by origin file
- We need to extract only the training cells' scGPT embeddings and maintain the shuffled order

## Input Data Structure

### Source 1: Shuffled Training Data
- **Location**: `s3://pythiomicsdata/cellxgene_v2/training_v1_suffled/` (not the misspelling suffled not shuffled, but the local data  directory is correctly spelled)
- **Format**: Parquet files (batch_0000.parquet, batch_0001.parquet, ...)
- **Schema**:
  ```
  - cell_id: string (unique identifier)
  - origin_file: string (source h5ad filename)
  - embedding: array<float32> (GenePT embedding, 3072 dims)
  - cell_type: string
  - cell_type_code: int32
  - [other metadata columns]
  ```

### Source 2: scGPT Embeddings
- **Location**: `s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1/`
- **Format**: Parquet files organized by origin file
- **File naming**: `{origin_file_stem}_scgpt.parquet`
- **Schema**:
  ```
  - cell_id: string (matches cell_id in training data)
  - scgpt_embedding: array<float32> (512 dims)
  ```

## Output Data Structure

### S3 Output
- **Location**: `s3://pythiomicsdata/cellxgene_v2/training_v1_scgpt_shuffled/` (note the correct spelling)
- **Format**: Parquet files matching training batch structure
- **File naming**: `batch_XXXX.parquet` (same as training files)
- **Schema**:
  ```
  - cell_id: string
  - origin_file: string  
  - scgpt_embedding: array<float32> (512 dims)
  - cell_type: string
  - cell_type_code: int32
  ```

### Local Output (EC2 Instance)
- **Location**: `/data/GenePT-tools/data/cellxgene_embeddings/training_v1_scgpt_shuffled/`
- **Format**: Same as S3 output
- **EBS Volume**: Will be snapshotted after processing for AWS Batch training

## Processing Logic

### Step 1: Inventory Training Data
1. Use the local copy of 'data/cellxgene_embeddings/training_v1_shuffled/*.parquet'
2. For each batch file:
   - Load the parquet file
   - Extract unique (origin_file, cell_id) pairs
   - Build a mapping: `origin_file -> Set[cell_id]`

### Step 2: Load scGPT Embeddings
1. For each unique origin_file found in Step 1:
   - Construct scGPT file path: `scgpt_embeddings_v1/{origin_file_stem}_scgpt.parquet`
   - Load the scGPT embeddings parquet
   - Create lookup dict: `cell_id -> scgpt_embedding`
   - Cache in memory (with LRU if needed for memory management)

### Step 3: Create Shuffled scGPT Batches
1. For each training batch file:
   - Load the training batch
   - For each row:
     - Look up scgpt_embedding using (origin_file, cell_id)
     - Create output row with:
       - cell_id
       - origin_file
       - scgpt_embedding
       - cell_type
       - cell_type_code
   - Write batch to both S3 and local filesystem

### Step 4: Validation
1. Verify row counts match between input and output batches
2. Verify all cell_ids were found in scGPT embeddings
3. Log any missing cells (should be none if data is complete)
4. Verify embedding dimensions (512 for scGPT)

## EC2 Execution Environment

### Instance Configuration
- **Instance Type**: `c5.4xlarge` or `c5.9xlarge`
  - 16-36 vCPUs for parallel processing
  - 32-72 GB RAM for caching embeddings
  - Up to 10 Gbps network for S3 transfers
- **EBS Volume**: 
  - Size: 500 GB (to accommodate all training data variants)
  - Type: `gp3` with 10,000 IOPS for fast I/O
  - Mount point: `/data`
- **AMI**: Use existing GenePT training AMI with dependencies pre-installed
- **IAM Role**: Requires S3 read access to `pythiomicsdata` bucket

### Execution Steps
1. **Launch EC2 Instance**:
   ```bash
   aws ec2 run-instances \
     --image-id ami-xxxxx \  # GenePT training AMI
     --instance-type c5.4xlarge \
     --block-device-mappings file://ebs-mapping.json \
     --iam-instance-profile Name=genept-processing \
     --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=scgpt-extraction}]'
   ```

2. **Mount and Prepare EBS Volume**:
   ```bash
   # Mount the data volume
   sudo mkfs -t xfs /dev/nvme1n1  # Only if new volume
   sudo mount /dev/nvme1n1 /data
   
   # Create directory structure
   sudo mkdir -p /data/GenePT-tools/data/cellxgene_embeddings/
   sudo chown -R ec2-user:ec2-user /data/GenePT-tools
   ```

3. **Copy Existing Training Data** (if not already on volume):
   ```bash
   aws s3 sync s3://pythiomicsdata/cellxgene_v2/training_v1_shuffled/ \
     /data/GenePT-tools/data/cellxgene_embeddings/training_v1_shuffled/ \
     --profile xcellerate
   ```

4. **Run Extraction Script**:
   ```bash
   python scripts/extract_scgpt_training_embeddings.py \
     --local-mode \  # Use local training data
     --output-format both \  # Create both parquet and PT formats
     --max-workers 16
   ```

5. **Create EBS Snapshot**:
   ```bash
   # Get volume ID
   VOLUME_ID=$(aws ec2 describe-volumes \
     --filters "Name=attachment.instance-id,Values=$(ec2-metadata --instance-id | cut -d " " -f 2)" \
     --query "Volumes[?Attachments[0].Device=='/dev/xvdf'].VolumeId" \
     --output text)
   
   # Create snapshot
   aws ec2 create-snapshot \
     --volume-id $VOLUME_ID \
     --description "GenePT training data with scGPT embeddings - $(date +%Y%m%d)" \
     --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=genept-scgpt-training-data},{Key=Version,Value=v1}]'
   ```

### Data Layout on EBS Volume
```
/data/
└── GenePT-tools/
    └── data/
        └── cellxgene_embeddings/
            ├── training_v1_shuffled/           # Original GenePT training
            │   ├── batch_0000.parquet
            │   └── ...
            ├── training_v1_shuffled_pt/        # PT format GenePT
            │   ├── batch_0000.pt
            │   └── ...
            ├── training_v1_scgpt_shuffled/     # New scGPT embeddings
            │   ├── parquet/
            │   │   ├── batch_0000.parquet
            │   │   └── ...
            │   ├── pt/
            │   │   ├── batch_0000.pt
            │   │   └── ...
            │   └── manifest.json
            ├── test_v1/                        # Test data
            └── scgpt_embeddings_v1/            # Source scGPT embeddings (cached)
```

### AWS Batch Integration
After snapshot creation:
1. Update Launch Template to use new snapshot:
   ```json
   {
     "BlockDeviceMappings": [
       {
         "DeviceName": "/dev/xvdf",
         "Ebs": {
           "SnapshotId": "snap-xxxxx",  # New snapshot with scGPT data
           "VolumeSize": 500,
           "VolumeType": "gp3",
           "Iops": 10000
         }
       }
     ]
   }
   ```

2. Update Compute Environment to use new Launch Template version

3. Training jobs can now access scGPT embeddings at:
   - `/data/GenePT-tools/data/cellxgene_embeddings/training_v1_scgpt_shuffled/`

## Implementation Requirements

### Dependencies
- `pandas` or `pyarrow` for parquet operations
- `boto3` for S3 operations
- `tqdm` for progress bars
- `numpy` for array operations
- Optional: `dask` or `ray` for parallel processing

### Configuration
```python
CONFIG = {
    "s3_profile": "xcellerate",  # or None for default
    "s3_region": "us-west-2",
    
    "training_s3_path": "s3://pythiomicsdata/cellxgene_v2/training_v1_shuffled/",
    "scgpt_s3_path": "s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1/",
    "output_s3_path": "s3://pythiomicsdata/cellxgene_v2/training_v1_scgpt_shuffled/",
    
    "local_data_dir": "/data/GenePT-tools/data/cellxgene_embeddings/",
    "training_local_dir": "training_v1_shuffled/",
    "scgpt_local_dir": "scgpt_embeddings_v1/",
    "output_local_dir": "training_v1_scgpt_shuffled/",
    
    "batch_pattern": "batch_*.parquet",
    "max_workers": 4,  # For parallel processing
    "cache_size_gb": 16,  # Memory cache for embeddings
}
```

### Error Handling
1. **Missing scGPT files**: Log error and skip origin file
2. **Missing cell_ids**: Log warning with count and details
3. **S3 errors**: Implement retry logic with exponential backoff
4. **Memory errors**: Implement batched processing if needed
5. **Data validation errors**: Stop processing and report

### Performance Considerations
1. **Memory optimization**:
   - Use memory mapping for large files if available
   - Process in chunks if memory is limited
   - Use LRU cache for scGPT embeddings lookup

2. **I/O optimization**:
   - Download files in parallel
   - Use multiprocessing for batch creation
   - Write to local first, then upload to S3 in parallel

3. **Expected performance**:
   - ~300 training batch files
   - ~50-100 unique origin files
   - Total data size: ~50-100 GB
   - Expected runtime: 1-2 hours on c5.4xlarge
   - EBS snapshot creation: ~30 minutes for 500GB volume
   - Network transfer from S3: ~1 GB/min with 10 Gbps network

## Script Interface

### Command Line Usage
```bash
python scripts/extract_scgpt_training_embeddings.py \
    --training-dir s3://pythiomicsdata/cellxgene_v2/training_v1_shuffled/ \
    --scgpt-dir s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1/ \
    --output-dir s3://pythiomicsdata/cellxgene_v2/training_v1_scgpt_shuffled/ \
    --local-output-dir /data/GenePT-tools/data/cellxgene_embeddings/training_v1_scgpt_shuffled/ \
    --aws-profile xcellerate \
    --max-workers 4 \
    --cache-gb 16 \
    --validate
```

### Python API
```python
from scripts.extract_scgpt_training_embeddings import extract_scgpt_embeddings

extract_scgpt_embeddings(
    training_dir="s3://pythiomicsdata/cellxgene_v2/training_v1_shuffled/",
    scgpt_dir="s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1/",
    output_dir="s3://pythiomicsdata/cellxgene_v2/training_v1_scgpt_shuffled/",
    local_output_dir="/data/GenePT-tools/data/cellxgene_embeddings/training_v1_scgpt_shuffled/",
    aws_profile="xcellerate",
    max_workers=4,
    validate=True
)
```

## Validation Checks

### Pre-processing
1. Verify training directory exists and contains parquet files
2. Verify scGPT directory exists
3. Check write permissions for output directories
4. Estimate memory requirements and warn if insufficient

### Post-processing
1. Count total cells processed vs expected
2. Verify all batches were created
3. Check embedding dimensions are correct (512)
4. Sample random cells and verify embeddings match source
5. Generate summary statistics:
   - Total cells processed
   - Number of batches created
   - Number of unique origin files
   - Any missing cells or files
   - Total data size

## Success Criteria
1. All cells from training set have corresponding scGPT embeddings
2. Output batch files maintain exact same order as input
3. No data corruption (validate with checksums)
4. Both S3 and local copies are identical
5. Process completes within 2 hours on c5.4xlarge instance
6. Memory usage stays under 32GB
7. EBS snapshot successfully created and tagged
8. Snapshot can be mounted and data verified on test instance
9. AWS Batch jobs can successfully use snapshot for training

## Testing Strategy
1. **Unit tests**: Test individual functions with mock data
2. **Integration test**: Process 2-3 batch files end-to-end
3. **Recovery test**: Verify resume capability after interruption

## Future Enhancements

### 1. Incremental Updates (New Batches Only)

#### Implementation Design
```python
class IncrementalProcessor:
    def __init__(self, manifest_file="processed_batches.json"):
        self.manifest_file = manifest_file
        self.processed_batches = self.load_manifest()
    
    def load_manifest(self):
        """Load list of already processed batch files with checksums"""
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_unprocessed_batches(self, all_batches):
        """Return only batches not in manifest or with changed checksums"""
        unprocessed = []
        for batch_file in all_batches:
            checksum = self.compute_checksum(batch_file)
            if batch_file not in self.processed_batches or \
               self.processed_batches[batch_file]['checksum'] != checksum:
                unprocessed.append(batch_file)
        return unprocessed
    
    def update_manifest(self, batch_file, checksum, timestamp):
        """Update manifest after successful processing"""
        self.processed_batches[batch_file] = {
            'checksum': checksum,
            'processed_at': timestamp,
            'output_file': f"batch_{batch_file.split('_')[1]}"
        }
        self.save_manifest()
```

#### Key Features
- **Manifest tracking**: JSON file tracking processed batches with checksums
- **Change detection**: MD5/SHA256 checksums to detect modified input files
- **Selective processing**: Only process new or modified batches
- **Resume capability**: Can restart from interruption point
- **Validation mode**: Option to force reprocessing for validation

#### Command Line Extension
```bash
python scripts/extract_scgpt_training_embeddings.py \
    --incremental \
    --manifest processed_batches.json \
    --force-reprocess batch_0150.parquet,batch_0151.parquet
```

### 2. PyTorch Tensor (.pt) Format Output

#### Implementation Design
```python
def create_pt_format(parquet_batch, output_path):
    """Convert parquet batch to PyTorch tensor format for fast loading"""
    
    # Load parquet data
    df = pd.read_parquet(parquet_batch)
    
    # Create tensor dictionary with fixed-size arrays
    tensor_dict = {
        'embeddings': torch.tensor(
            np.vstack(df['scgpt_embedding'].values), 
            dtype=torch.float32
        ),  # Shape: (n_cells, 512)
        
        'cell_type_codes': torch.tensor(
            df['cell_type_code'].values, 
            dtype=torch.long
        ),  # Shape: (n_cells,)
        
        'metadata': {
            'cell_ids': df['cell_id'].tolist(),
            'origin_files': df['origin_file'].tolist(),
            'cell_types': df['cell_type'].tolist(),
            'n_cells': len(df),
            'embedding_dim': 512,
            'created_at': datetime.now().isoformat()
        }
    }
    
    # Save with compression for space efficiency
    torch.save(
        tensor_dict, 
        output_path,
        pickle_protocol=4,  # Latest protocol for efficiency
        _use_new_zipfile_serialization=True  # Faster loading
    )
    
    # Create index file for fast random access
    create_index_file(output_path, df)
```

#### Parallel Format Creation
```python
def create_dual_format_output(batch_data, batch_number):
    """Create both parquet and PT format outputs"""
    
    # Original parquet output
    parquet_path = f"batch_{batch_number:04d}.parquet"
    batch_data.to_parquet(parquet_path)
    
    # PT format for fast loading
    pt_path = f"batch_{batch_number:04d}.pt"
    create_pt_format(parquet_path, pt_path)
    
    # Create metadata sidecar
    meta_path = f"batch_{batch_number:04d}_meta.json"
    save_metadata(batch_data, meta_path)
    
    return parquet_path, pt_path, meta_path
```

#### Directory Structure
```
training_v1_scgpt_shuffled/
├── parquet/
│   ├── batch_0000.parquet
│   ├── batch_0001.parquet
│   └── ...
├── pt/
│   ├── batch_0000.pt
│   ├── batch_0001.pt
│   └── ...
├── metadata/
│   ├── batch_0000_meta.json
│   ├── batch_0001_meta.json
│   └── ...
└── manifest.json
```

#### Performance Benefits
- **10x faster loading**: Direct tensor loading vs parquet parsing
- **Memory efficiency**: Pre-allocated contiguous tensors
- **GPU-ready**: Tensors can be directly moved to GPU
- **Batch compatibility**: Works seamlessly with PyTorch DataLoader

#### Usage in Training
```python
class FastScGPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_dir):
        self.pt_files = sorted(glob.glob(f"{pt_dir}/batch_*.pt"))
        self.current_batch = None
        self.current_batch_idx = -1
    
    def load_batch(self, batch_idx):
        if batch_idx != self.current_batch_idx:
            self.current_batch = torch.load(
                self.pt_files[batch_idx],
                weights_only=True
            )
            self.current_batch_idx = batch_idx
    
    def __getitem__(self, idx):
        batch_idx = idx // BATCH_SIZE
        within_batch_idx = idx % BATCH_SIZE
        self.load_batch(batch_idx)
        return (
            self.current_batch['embeddings'][within_batch_idx],
            self.current_batch['cell_type_codes'][within_batch_idx]
        )
```

#### Command Line Extension
```bash
python scripts/extract_scgpt_training_embeddings.py \
    --output-format both \  # Options: parquet, pt, both
    --pt-compression \
    --create-index \
    --validate-tensors
