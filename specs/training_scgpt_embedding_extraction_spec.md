# Training Set scGPT Embedding Extraction Specification

## Overview

Extract scGPT embeddings corresponding to cells in the CellXGene v2 training datasets. This creates aligned training datasets where each row from the original training set has its corresponding scGPT embedding features.

## Objectives

1. Read training dataset metadata from S3 to identify required cells
2. Extract matching scGPT embeddings from chunked local files
3. Produce aligned parquet files ready for model training
4. Ensure efficient processing using DuckDB for large-scale joins

## Data Sources

### Input Training Data (S3)
- **Location**: `s3://pythiomicsdata/cellxgene_v2/training_v1/*.parquet`
- **AWS Profile**: `xcellerate`
- **Key Columns**:
  - `observation_joinid`: Unique cell identifier for joining
  - `filename`: Original dataset UUID (used to locate scGPT chunks)
  - Additional features and labels for training

### Input scGPT Embeddings (Local)
- **Location**: `/mnt/scratch/cellxgene_v2_scgpt_chunks/`
- **Format**: Chunked parquet files per dataset
- **Naming Convention**: `{dataset_uuid}_chunk_{NNNN}_of_{TTTT}_scgpt.parquet`
  - Example: `518d9049-2a76-44f8-8abc-1e2b59ab5ba1_chunk_0000_of_0002_scgpt.parquet`
- **Key Columns**:
  - `observation_joinid`: Cell identifier (join key)
  - `embedding_0`, `embedding_1`, ..., `embedding_511`: scGPT embedding features (512-dim)
  - Potentially other metadata columns

### Output (Local)
- **Location**: `/mnt/scratch/cellxgene_v2_training_v1_scgpt/`
- **Format**: Parquet files, one per input training file
- **Naming Convention**: Match input training filenames
  - If input is `training_batch_001.parquet`, output is `training_batch_001.parquet`
- **Content**: Combined rows from training data with corresponding scGPT embeddings

## Architecture

### Processing Strategy

The key challenge is efficiently joining large training datasets with chunked scGPT embeddings. We use DuckDB for its efficient handling of:
- Large parquet file reads without loading into memory
- Hash joins optimized for multi-GB datasets
- Direct S3 reads with credentials
- Parallel parquet writes

### High-Level Workflow

```
For each training file in S3:
  1. Scan training file to extract (observation_joinid, filename) pairs
  2. Identify unique dataset UUIDs (filenames) present in training file
  3. For each dataset UUID:
     - Find all corresponding scGPT chunk files locally
     - Load scGPT chunks into DuckDB virtual table
  4. Perform join: training_data ⟗ scgpt_embeddings ON observation_joinid
  5. Write joined results to output parquet file
  6. Validate row counts match
```

### DuckDB Implementation Approach

DuckDB is the optimal choice because:
1. **Native S3 Support**: Can read directly from S3 with AWS credentials
2. **Efficient Joins**: Hash joins on large datasets without memory overflow
3. **Parquet Optimization**: Columnar reads leverage parquet metadata for pruning
4. **Zero-Copy**: Minimizes memory usage by streaming data
5. **Pre-installed**: Already available on the remote machine

```python
import duckdb

def extract_embeddings_for_training_file(
    training_s3_path: str,
    scgpt_chunks_dir: str,
    output_path: str
):
    """
    Extract scGPT embeddings matching a training file.

    Args:
        training_s3_path: S3 URI like s3://bucket/path/file.parquet
        scgpt_chunks_dir: Local directory with scGPT chunks
        output_path: Where to write the output parquet
    """
    conn = duckdb.connect()

    # Configure S3 access
    conn.execute("""
        SET s3_region='us-west-2';
        SET s3_access_key_id='...';
        SET s3_secret_access_key='...';
    """)

    # Strategy: Read training file, extract unique dataset UUIDs,
    # then union all relevant scGPT chunks for efficient join

    # Step 1: Get unique dataset UUIDs from training file
    dataset_uuids = conn.execute(f"""
        SELECT DISTINCT filename
        FROM read_parquet('{training_s3_path}')
    """).fetchall()

    # Step 2: Build list of scGPT chunk files to read
    scgpt_files = []
    for uuid in dataset_uuids:
        # Find all chunks for this dataset
        chunks = glob.glob(f"{scgpt_chunks_dir}/{uuid}_chunk_*_scgpt.parquet")
        scgpt_files.extend(chunks)

    # Step 3: Perform join
    # Note: DuckDB's read_parquet accepts a list of files and unions them
    result = conn.execute(f"""
        SELECT
            t.*,
            e.embedding_0, e.embedding_1, ..., e.embedding_511
        FROM read_parquet('{training_s3_path}') t
        INNER JOIN read_parquet({scgpt_files}) e
        ON t.observation_joinid = e.observation_joinid
        """).fetch_arrow_table()

    # Step 4: Write output
    conn.execute(f"""
        COPY result TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
```

### Alternative: Python with Polars

If DuckDB version on remote machine lacks necessary features, use Polars for similar performance:

```python
import polars as pl

def extract_with_polars(training_s3_path, scgpt_chunks_dir, output_path):
    # Read training file (lazy - not loaded into memory yet)
    training = pl.scan_parquet(training_s3_path, storage_options={
        'aws_profile': 'xcellerate'
    })

    # Get unique dataset UUIDs
    dataset_uuids = training.select('filename').unique().collect()['filename'].to_list()

    # Scan all relevant scGPT chunks
    scgpt_files = []
    for uuid in dataset_uuids:
        scgpt_files.extend(glob.glob(f"{scgpt_chunks_dir}/{uuid}_chunk_*_scgpt.parquet"))

    scgpt_embeddings = pl.scan_parquet(scgpt_files)

    # Perform join (lazy execution)
    joined = training.join(
        scgpt_embeddings,
        on='observation_joinid',
        how='inner'
    )

    # Execute and write
    joined.sink_parquet(output_path, compression='zstd')
```

### Pandas Fallback (If Needed)

Only use if data fits in memory or for debugging small files:

```python
import pandas as pd
import pyarrow.parquet as pq

def extract_with_pandas(training_s3_path, scgpt_chunks_dir, output_path):
    # Read training file
    training_df = pd.read_parquet(
        training_s3_path,
        storage_options={'profile': 'xcellerate'}
    )

    # Find unique datasets
    dataset_uuids = training_df['filename'].unique()

    # Load and concatenate relevant scGPT chunks
    scgpt_dfs = []
    for uuid in dataset_uuids:
        chunk_files = glob.glob(f"{scgpt_chunks_dir}/{uuid}_chunk_*_scgpt.parquet")
        for chunk_file in chunk_files:
            scgpt_dfs.append(pd.read_parquet(chunk_file))

    scgpt_df = pd.concat(scgpt_dfs, ignore_index=True)

    # Join
    result_df = training_df.merge(
        scgpt_df,
        on='observation_joinid',
        how='inner'
    )

    # Write
    result_df.to_parquet(output_path, compression='zstd', index=False)
```

## Implementation Details

### Script Structure

Create `scripts/extract_training_scgpt_embeddings.py`:

```python
#!/usr/bin/env python3
"""
Extract scGPT embeddings for training datasets.

Usage:
    # Process all training files
    python scripts/extract_training_scgpt_embeddings.py

    # Process single file
    python scripts/extract_training_scgpt_embeddings.py --training-file training_001.parquet

    # Resume from checkpoint
    python scripts/extract_training_scgpt_embeddings.py --resume
"""

import argparse
import logging
from pathlib import Path
from typing import List
import duckdb
import boto3
import json

def setup_logging(log_dir: Path):
    """Configure logging to file and console."""
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'extraction.log'),
            logging.StreamHandler()
        ]
    )

def get_training_files(s3_bucket: str, s3_prefix: str, aws_profile: str) -> List[str]:
    """List all training parquet files in S3."""
    session = boto3.Session(profile_name=aws_profile)
    s3 = session.client('s3')

    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)

    files = []
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.parquet'):
            files.append(f"s3://{s3_bucket}/{obj['Key']}")

    return sorted(files)

def get_scgpt_chunks_for_dataset(scgpt_dir: Path, dataset_uuid: str) -> List[Path]:
    """Find all chunk files for a given dataset UUID."""
    pattern = f"{dataset_uuid}_chunk_*_scgpt.parquet"
    return sorted(scgpt_dir.glob(pattern))

def extract_embeddings_duckdb(
    training_s3_path: str,
    scgpt_chunks_dir: Path,
    output_path: Path,
    aws_profile: str
):
    """
    Extract scGPT embeddings using DuckDB for efficient joins.
    """
    logging.info(f"Processing {training_s3_path}")

    # Initialize DuckDB with S3 credentials
    conn = duckdb.connect()

    # Configure AWS credentials from profile
    session = boto3.Session(profile_name=aws_profile)
    credentials = session.get_credentials()

    conn.execute(f"""
        SET s3_region='{session.region_name or 'us-west-2'}';
        SET s3_access_key_id='{credentials.access_key}';
        SET s3_secret_access_key='{credentials.secret_key}';
    """)

    if credentials.token:
        conn.execute(f"SET s3_session_token='{credentials.token}';")

    # Get unique dataset UUIDs from training file
    logging.info("Identifying datasets in training file...")
    dataset_uuids = conn.execute(f"""
        SELECT DISTINCT filename
        FROM read_parquet('{training_s3_path}')
    """).fetchall()

    dataset_uuids = [row[0] for row in dataset_uuids]
    logging.info(f"Found {len(dataset_uuids)} unique datasets")

    # Collect all scGPT chunk files
    scgpt_files = []
    for uuid in dataset_uuids:
        chunks = get_scgpt_chunks_for_dataset(scgpt_chunks_dir, uuid)
        if not chunks:
            logging.warning(f"No scGPT chunks found for dataset {uuid}")
            continue
        scgpt_files.extend([str(f) for f in chunks])
        logging.info(f"  {uuid}: {len(chunks)} chunks")

    if not scgpt_files:
        raise ValueError("No scGPT chunks found for any dataset in training file")

    logging.info(f"Total scGPT chunk files: {len(scgpt_files)}")

    # Perform join
    logging.info("Performing join...")
    scgpt_files_str = "[" + ", ".join(f"'{f}'" for f in scgpt_files) + "]"

    # Get row counts for validation
    training_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{training_s3_path}')
    """).fetchone()[0]

    # Execute join and write
    conn.execute(f"""
        COPY (
            SELECT
                t.*,
                e.* EXCLUDE (observation_joinid)
            FROM read_parquet('{training_s3_path}') t
            INNER JOIN (
                SELECT * FROM read_parquet({scgpt_files_str})
            ) e
            ON t.observation_joinid = e.observation_joinid
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Validate output
    output_count = conn.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{output_path}')
    """).fetchone()[0]

    logging.info(f"Training rows: {training_count}")
    logging.info(f"Output rows: {output_count}")

    if output_count != training_count:
        logging.warning(
            f"Row count mismatch! Training: {training_count}, Output: {output_count}"
        )
        missing = training_count - output_count
        logging.warning(f"Missing {missing} rows ({missing/training_count*100:.2f}%)")
    else:
        logging.info("✓ Row counts match")

    conn.close()
    return output_count

def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load processing checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    """Save processing checkpoint."""
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Extract scGPT embeddings for training datasets"
    )
    parser.add_argument(
        "--s3-bucket",
        default="pythiomicsdata",
        help="S3 bucket containing training files"
    )
    parser.add_argument(
        "--s3-prefix",
        default="cellxgene_v2/training_v1/",
        help="S3 prefix for training files"
    )
    parser.add_argument(
        "--scgpt-chunks-dir",
        type=Path,
        default=Path("/mnt/scratch/cellxgene_v2_scgpt_chunks"),
        help="Directory containing scGPT chunk files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/scratch/cellxgene_v2_training_v1_scgpt"),
        help="Output directory for extracted embeddings"
    )
    parser.add_argument(
        "--aws-profile",
        default="xcellerate",
        help="AWS profile to use"
    )
    parser.add_argument(
        "--training-file",
        help="Process single training file instead of all"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/mnt/scratch/logs"),
        help="Directory for log files"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "extraction_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {"completed": [], "failed": []}

    # Get training files to process
    if args.training_file:
        training_files = [f"s3://{args.s3_bucket}/{args.s3_prefix}/{args.training_file}"]
    else:
        logging.info("Listing training files from S3...")
        training_files = get_training_files(args.s3_bucket, args.s3_prefix, args.aws_profile)
        logging.info(f"Found {len(training_files)} training files")

    # Process each training file
    total = len(training_files)
    for idx, training_s3_path in enumerate(training_files, 1):
        filename = training_s3_path.split('/')[-1]

        # Skip if already processed
        if filename in checkpoint["completed"]:
            logging.info(f"[{idx}/{total}] Skipping {filename} (already completed)")
            continue

        output_path = args.output_dir / filename

        try:
            logging.info(f"[{idx}/{total}] Processing {filename}")

            row_count = extract_embeddings_duckdb(
                training_s3_path,
                args.scgpt_chunks_dir,
                output_path,
                args.aws_profile
            )

            # Update checkpoint
            checkpoint["completed"].append(filename)
            save_checkpoint(checkpoint_path, checkpoint)

            logging.info(f"✓ Completed {filename} ({row_count:,} rows)")

        except Exception as e:
            logging.error(f"✗ Failed to process {filename}: {e}")
            checkpoint["failed"].append({"file": filename, "error": str(e)})
            save_checkpoint(checkpoint_path, checkpoint)
            continue

    # Summary
    logging.info("\n" + "="*80)
    logging.info("EXTRACTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total files: {total}")
    logging.info(f"Completed: {len(checkpoint['completed'])}")
    logging.info(f"Failed: {len(checkpoint['failed'])}")

    if checkpoint["failed"]:
        logging.info("\nFailed files:")
        for failure in checkpoint["failed"]:
            logging.info(f"  - {failure['file']}: {failure['error']}")

if __name__ == "__main__":
    main()
```

### Execution Environment

**Remote Machine**: `memverge-dataset-curation`
- SSH host configured in local SSH config
- Has `python3` and `duckdb` installed
- Sufficient storage in `/mnt/scratch`

**Deployment Approach**:
1. Copy script to remote machine via scp
2. Install minimal Python dependencies (boto3, duckdb Python binding if needed)
3. Run script in tmux/screen session for resilience
4. Monitor logs remotely

```bash
# Deploy script
scp scripts/extract_training_scgpt_embeddings.py memverge-dataset-curation:/tmp/

# SSH and run
ssh memverge-dataset-curation
cd /tmp
tmux new -s extraction

# Check dependencies
python3 -c "import duckdb, boto3"  # Install if missing

# Run extraction
python3 extract_training_scgpt_embeddings.py --resume

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t extraction
```

## Data Validation

### Pre-Processing Validation

Before running extraction:

1. **Verify S3 Access**:
   ```bash
   aws s3 ls s3://pythiomicsdata/cellxgene_v2/training_v1/ --profile xcellerate
   ```

2. **Check scGPT Chunk Integrity**:
   ```python
   import duckdb
   conn = duckdb.connect()

   # Sample a chunk file to check schema
   result = conn.execute("""
       SELECT * FROM '/mnt/scratch/cellxgene_v2_scgpt_chunks/51*.parquet'
       LIMIT 5
   """).fetchdf()

   print(result.columns.tolist())
   # Expected: ['observation_joinid', 'embedding_0', ..., 'embedding_511', ...]
   ```

3. **Verify observation_joinid Format Consistency**:
   - Ensure both training and scGPT files use same ID format
   - Check for potential issues: leading zeros, string vs int, UUID format

### Post-Processing Validation

After extraction:

1. **Row Count Validation**:
   ```sql
   -- For each output file, verify:
   -- output_rows == training_rows
   SELECT
       COUNT(*) as row_count,
       COUNT(DISTINCT observation_joinid) as unique_ids
   FROM read_parquet('output_file.parquet')
   ```

2. **Column Validation**:
   ```python
   # Check all embedding columns present
   import duckdb
   conn = duckdb.connect()

   cols = conn.execute("""
       SELECT column_name
       FROM (DESCRIBE SELECT * FROM 'output_file.parquet')
       WHERE column_name LIKE 'embedding_%'
   """).fetchall()

   assert len(cols) == 512, f"Expected 512 embedding columns, found {len(cols)}"
   ```

3. **Data Quality Checks**:
   ```sql
   -- Check for nulls in embeddings
   SELECT
       COUNT(*) as null_count
   FROM read_parquet('output_file.parquet')
   WHERE embedding_0 IS NULL

   -- Should be 0
   ```

4. **Sample Inspection**:
   ```python
   # Manually inspect a few rows
   sample = conn.execute("""
       SELECT * FROM read_parquet('output_file.parquet')
       LIMIT 10
   """).fetchdf()

   print(sample[['observation_joinid', 'embedding_0', 'embedding_1']].head())
   ```

## Error Handling

### Common Issues and Resolutions

1. **Missing scGPT Chunks for Dataset**
   - **Issue**: Training file references a dataset UUID not present in scGPT chunks
   - **Detection**: Warning logged during dataset UUID collection
   - **Resolution**:
     - Log affected rows
     - Continue processing other datasets
     - Report summary of missing datasets
   - **Acceptable**: If <5% of cells affected

2. **observation_joinid Mismatch**
   - **Issue**: Cell IDs don't match between training and scGPT files
   - **Detection**: Output row count < training row count
   - **Resolution**:
     - Investigate ID format differences
     - Check if scGPT chunks were generated from same data version
     - May require ID mapping table
   - **Not Acceptable**: Must resolve before proceeding

3. **Memory Issues**
   - **Issue**: DuckDB/Polars runs out of memory
   - **Resolution**:
     - Process training files one at a time (default behavior)
     - Use DuckDB's streaming execution (automatic)
     - Reduce number of chunks processed simultaneously
   - **Fallback**: Split training files into smaller batches

4. **S3 Access Errors**
   - **Issue**: Cannot read from S3 (credentials, network, permissions)
   - **Detection**: boto3 or DuckDB S3 read errors
   - **Resolution**:
     - Verify AWS profile configuration: `aws configure --profile xcellerate`
     - Check network connectivity to S3
     - Verify bucket permissions
   - **Alternative**: Download training files locally first

5. **Partial Chunk Files**
   - **Issue**: scGPT chunk file corrupted or incomplete
   - **Detection**: Parquet read errors
   - **Resolution**:
     - Skip corrupted chunk
     - Log missing data
     - Continue with remaining chunks
   - **Report**: List of corrupted files for regeneration

## Performance Optimization

### Expected Performance

**Benchmark Estimates** (need to measure on actual hardware):
- Small training file (1M rows): ~2-5 minutes
- Large training file (10M rows): ~15-30 minutes
- Total processing time for all files: ~4-12 hours

### Optimization Strategies

1. **Parallel Processing**:
   - Run multiple training files in parallel (if memory allows)
   - Use GNU parallel or separate tmux sessions
   ```bash
   # Process 4 files in parallel
   parallel -j 4 python3 extract_training_scgpt_embeddings.py --training-file {} \
       ::: file1.parquet file2.parquet file3.parquet file4.parquet
   ```

2. **DuckDB Tuning**:
   ```python
   conn.execute("SET threads=16;")  # Use multiple cores
   conn.execute("SET memory_limit='64GB';")  # Set memory budget
   ```

3. **Chunk Pre-Loading**:
   - If same datasets appear in multiple training files, cache chunk metadata
   - Build index of observation_joinid → chunk_file mapping

4. **Compression**:
   - Use ZSTD compression for output (good compression, fast decompression)
   - Consider trade-off: Snappy (faster write) vs ZSTD (smaller files)

## Output Specifications

### File Naming
```
/mnt/scratch/cellxgene_v2_training_v1_scgpt/
├── training_001.parquet
├── training_002.parquet
├── ...
└── extraction_checkpoint.json
```

### Schema
```
Original training columns:
- observation_joinid (string): Cell identifier
- filename (string): Dataset UUID
- cell_type (string): Cell type label
- donor_id (string): Donor identifier
- tissue (string): Tissue type
- [other training features...]

Added scGPT columns:
- embedding_0 (float): scGPT embedding dimension 0
- embedding_1 (float): scGPT embedding dimension 1
- ...
- embedding_511 (float): scGPT embedding dimension 511
- [possibly other scGPT metadata...]

Total columns: ~520-550 (depends on training file schema)
```

### File Properties
- **Format**: Apache Parquet
- **Compression**: ZSTD (level 3, balanced)
- **Typical Size**: 2-10GB per file (depends on rows and columns)
- **Row Groups**: Default (let DuckDB/Polars optimize)

## Monitoring and Logging

### Log Levels

- **INFO**: Normal progress (file processing, row counts, completion)
- **WARNING**: Non-fatal issues (missing datasets, row count mismatches <5%)
- **ERROR**: Fatal issues (file access errors, schema mismatches, crashes)

### Key Metrics to Log

For each training file:
1. Training file name and row count
2. Number of unique datasets referenced
3. Number of scGPT chunk files loaded
4. Output row count
5. Processing time
6. Row count match status

### Progress Tracking

```
[1/50] Processing training_001.parquet
  Training rows: 1,234,567
  Datasets: 23
  scGPT chunks: 45
  Join completed: 1,234,567 rows (100.0% match)
  Wrote: /mnt/scratch/.../training_001.parquet (4.2 GB)
  Time: 3m 42s
✓ Completed training_001.parquet

[2/50] Processing training_002.parquet
...
```

## Testing Strategy

### Unit Tests

Not applicable - this is a one-off data processing task. Focus on validation checks instead.

### Integration Test

**Dry Run on Small Sample**:
1. Select 1-2 small training files
2. Run extraction script
3. Manually validate:
   - Row counts match
   - Embeddings are present and non-null
   - observation_joinids are correct
   - File size is reasonable
4. Load output in Python/DuckDB and spot-check values

### Acceptance Criteria

- [ ] All training files processed successfully
- [ ] Row counts match (100% for all files, or >95% with documented exceptions)
- [ ] All embedding columns present (512 dimensions)
- [ ] No null values in embedding columns
- [ ] Output files readable by DuckDB/Polars/Pandas
- [ ] Total processing time <24 hours
- [ ] Clear logs documenting any issues

## Execution Plan

### Phase 1: Preparation
1. Copy script to remote machine
2. Install required Python packages (boto3, duckdb)
3. Verify S3 access with `aws s3 ls`
4. Verify scGPT chunks exist and are readable
5. Run dry run on 1 small training file

### Phase 2: Full Extraction
1. Start tmux session for resilience
2. Run extraction script with `--resume` flag
3. Monitor logs for errors
4. Validate output files as they're created

### Phase 3: Validation
1. Check extraction_checkpoint.json for failures
2. Run validation queries on all output files
3. Generate summary report
4. Investigate and resolve any failures

### Phase 4: Cleanup
1. Archive logs
2. Document any issues encountered
3. Update data documentation with new dataset location

## Dependencies

### Python Packages
- `duckdb` - For efficient parquet joins
- `boto3` - For S3 access
- `pyarrow` (optional) - For parquet operations

### System Requirements
- **Storage**: 500GB-1TB free space in `/mnt/scratch`
- **Memory**: 32GB+ recommended
- **CPU**: 8+ cores (for parallel DuckDB)

### Remote Machine Setup
```bash
# Install Python packages
pip3 install duckdb boto3 pyarrow

# Configure AWS credentials
aws configure --profile xcellerate
# (enter credentials when prompted)

# Verify AWS access
aws s3 ls s3://pythiomicsdata/cellxgene_v2/training_v1/ --profile xcellerate

# Create output directory
mkdir -p /mnt/scratch/cellxgene_v2_training_v1_scgpt
mkdir -p /mnt/scratch/logs
```

## Success Criteria

1. ✅ All training files have corresponding output files
2. ✅ Row counts match between input and output
3. ✅ All embedding columns present (512 dimensions)
4. ✅ No null values in critical columns
5. ✅ Processing completed in <24 hours
6. ✅ Clear audit trail in logs
7. ✅ Output files are usable for model training

## Future Enhancements

1. **Incremental Updates**: Support adding new training files without reprocessing all
2. **Distributed Processing**: Use Dask or Spark for parallel processing across machines
3. **Embedding Caching**: Pre-build scGPT embedding index for faster lookups
4. **Validation Dashboard**: Real-time monitoring of extraction progress
5. **Alternative Embedding Sources**: Support other foundation models (Transcriptformer, Geneformer)
