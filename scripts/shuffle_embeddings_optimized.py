#!/usr/bin/env python
"""
Optimized out-of-core two-pass shuffle of embeddings data from parquet files.
Uses streaming writes to buckets for better performance.

Based on https://blog.janestreet.com/how-to-shuffle-a-big-dataset/ 
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import argparse
from tqdm import tqdm
import tempfile
import shutil
import gc

def get_common_columns():
  """Define the common columns across all files."""
  # Non-numeric metadata columns
  metadata_cols = [
    'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id',
    'development_stage', 'development_stage_ontology_term_id', 'disease', 
    'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid',
    'organism', 'organism_ontology_term_id', 'self_reported_ethnicity',
    'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id',
    'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type'
  ]
  
  # Numeric embedding columns (0-3071)
  embedding_cols = [str(i) for i in range(3072)]
  
  return metadata_cols + embedding_cols

class BucketWriter:
  """Efficient streaming writer for bucket files."""
  
  def __init__(self, path, schema=None):
    self.path = Path(path)
    self.writer = None
    self.schema = schema
    self.row_count = 0
    self.buffer = []
  
  def write(self, df):
    """Write dataframe to bucket file."""
    if df.empty:
      return
    
    # Buffer dataframes instead of writing immediately
    self.buffer.append(df)
    self.row_count += len(df)
    
    # Write when buffer gets large enough
    if sum(len(d) for d in self.buffer) >= 10000:
      self._flush()
  
  def _flush(self):
    """Flush buffer to file."""
    if not self.buffer:
      return
    
    # Concatenate all buffered dataframes
    combined_df = pd.concat(self.buffer, ignore_index=True)
    
    # Ensure consistent data types for numeric columns
    # Convert all numeric columns to float64 (which maps to double in parquet)
    for col in combined_df.columns:
      if combined_df[col].dtype in ['float32', 'float64', 'float', 'double']:
        combined_df[col] = combined_df[col].astype('float64')
    
    # Convert to pyarrow table
    table = pa.Table.from_pandas(combined_df, preserve_index=False)
    
    if self.writer is None:
      self.schema = table.schema
      self.writer = pq.ParquetWriter(self.path, self.schema, compression='snappy')
    else:
      # Cast the table to match the existing schema to avoid type mismatches
      table = table.cast(self.schema)
    
    self.writer.write_table(table)
    self.buffer = []
  
  def close(self):
    """Close the writer."""
    self._flush()
    if self.writer is not None:
      self.writer.close()

def optimized_shuffle(input_dir, output_dir, num_buckets=20, batch_size=10000, 
                      random_seed=42, max_rows_in_memory=50000):
  """
  Perform optimized out-of-core two-pass shuffle.
  
  Args:
    input_dir: Directory containing input parquet files
    output_dir: Directory to save shuffled batches
    num_buckets: Number of temporary buckets for first pass
    batch_size: Number of rows per output batch file
    random_seed: Random seed for reproducibility
    max_rows_in_memory: Maximum rows to hold in memory before flushing
  """
  np.random.seed(random_seed)
  rng = np.random.RandomState(random_seed)
  
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  
  parquet_files = sorted(input_path.glob('*.parquet'))
  print(f"Found {len(parquet_files)} parquet files")
  
  common_columns = get_common_columns()
  print(f"Using {len(common_columns)} common columns")
  
  # Create temporary directory for buckets in the current working directory
  # Use a subdirectory to avoid filling up the root filesystem
  project_temp_dir = Path.cwd() / "temp"
  project_temp_dir.mkdir(exist_ok=True)
  temp_dir = Path(tempfile.mkdtemp(prefix='shuffle_buckets_', dir=project_temp_dir))
  print(f"Using temporary directory: {temp_dir}")
  
  try:
    # PASS 1: Distribute rows randomly into buckets
    print(f"\nPass 1: Distributing data into {num_buckets} buckets...")
    
    # Initialize bucket writers
    bucket_writers = []
    for i in range(num_buckets):
      bucket_path = temp_dir / f"bucket_{i:03d}.parquet"
      bucket_writers.append(BucketWriter(bucket_path))
    
    # Process input files
    total_rows = 0
    
    # Define consistent column order (common columns + source_file)
    final_columns = common_columns + ['source_file']
    
    for file_path in tqdm(parquet_files, desc="Processing files"):
      # Read file (using pyarrow for better performance)
      table = pq.read_table(file_path, columns=common_columns)
      df = table.to_pandas()
      
      # Create a new dataframe with consistent types to avoid fragmentation
      # First, convert numeric columns
      numeric_data = {}
      for col in common_columns:
        if col.isdigit():  # Embedding columns are numeric strings
          numeric_data[col] = df[col].astype('float64')
        else:
          numeric_data[col] = df[col]
      
      # Add source filename column to the dict
      numeric_data['source_file'] = file_path.stem
      
      # Create new dataframe from dict (avoids fragmentation)
      df = pd.DataFrame(numeric_data)
      
      # Ensure consistent column order
      df = df[final_columns]
      
      num_rows = len(df)
      total_rows += num_rows
      
      # Randomly assign rows to buckets
      bucket_assignments = rng.randint(0, num_buckets, size=num_rows)
      
      # Group by bucket and write
      for bucket_id in range(num_buckets):
        mask = bucket_assignments == bucket_id
        if mask.any():
          bucket_df = df[mask].copy()
          bucket_writers[bucket_id].write(bucket_df)
      
      del df, table
      gc.collect()
    
    # Close all bucket writers
    print("Finalizing buckets...")
    for writer in bucket_writers:
      writer.close()
    
    # Print bucket statistics
    print(f"\nBucket statistics:")
    bucket_row_counts = []
    for i, writer in enumerate(bucket_writers):
      count = writer.row_count
      bucket_row_counts.append(count)
      print(f"  Bucket {i:2d}: {count:,} rows")
    print(f"  Total: {total_rows:,} rows")
    
    del bucket_writers
    gc.collect()
    
    # PASS 2: Shuffle each bucket and write to output files
    print(f"\nPass 2: Shuffling buckets and writing output batches...")
    
    output_batch_idx = 0
    accumulated_rows = []
    accumulated_count = 0
    
    bucket_paths = sorted(temp_dir.glob("bucket_*.parquet"))
    
    for bucket_path in tqdm(bucket_paths, desc="Processing buckets"):
      # Read bucket
      bucket_df = pd.read_parquet(bucket_path)
      
      if len(bucket_df) == 0:
        continue
      
      # Shuffle the bucket
      bucket_df = bucket_df.sample(frac=1, random_state=rng).reset_index(drop=True)
      
      # Add to accumulator
      accumulated_rows.append(bucket_df)
      accumulated_count += len(bucket_df)
      
      # Write complete batches
      while accumulated_count >= batch_size:
        # Combine accumulated data
        combined = pd.concat(accumulated_rows, ignore_index=True)
        
        # Write batches
        num_complete_batches = accumulated_count // batch_size
        for i in range(num_complete_batches):
          start_idx = i * batch_size
          end_idx = start_idx + batch_size
          batch_df = combined.iloc[start_idx:end_idx]
          
          output_file = output_path / f"batch_{output_batch_idx:04d}.parquet"
          batch_df.to_parquet(output_file, compression='snappy', index=False)
          output_batch_idx += 1
        
        # Keep remainder
        remainder_start = num_complete_batches * batch_size
        if remainder_start < len(combined):
          accumulated_rows = [combined.iloc[remainder_start:].copy()]
          accumulated_count = len(accumulated_rows[0])
        else:
          accumulated_rows = []
          accumulated_count = 0
        
        del combined
        gc.collect()
      
      # Clean up bucket file
      bucket_path.unlink()
    
    # Write final partial batch if exists
    if accumulated_rows and accumulated_count > 0:
      combined = pd.concat(accumulated_rows, ignore_index=True)
      output_file = output_path / f"batch_{output_batch_idx:04d}.parquet"
      combined.to_parquet(output_file, compression='snappy', index=False)
      print(f"Wrote final partial batch with {len(combined)} rows")
      output_batch_idx += 1
    
    print(f"\nShuffle complete! Created {output_batch_idx} output batches in {output_path}")
    
    # Verify output
    print("\nVerifying output batches...")
    output_files = sorted(output_path.glob("batch_*.parquet"))
    total_output_rows = 0
    for f in output_files[:5]:  # Check first 5
      df = pd.read_parquet(f)
      total_output_rows += len(df)
      print(f"  {f.name}: {len(df):,} rows, {len(df.columns)} columns")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total input files: {len(parquet_files)}")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Number of buckets: {num_buckets}")
    print(f"  Target batch size: {batch_size:,}")
    print(f"  Output batches: {output_batch_idx}")
    
  finally:
    # Clean up temporary directory
    if temp_dir.exists():
      print(f"\nCleaning up temporary directory: {temp_dir}")
      shutil.rmtree(temp_dir)

def main():
  parser = argparse.ArgumentParser(
    description="Optimized out-of-core two-pass shuffle of embedding parquet files"
  )
  parser.add_argument(
    "--input-dir",
    type=str,
    default="data/cellxgene_embeddings/training_v1",
    help="Input directory containing parquet files"
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="data/cellxgene_embeddings/training_v1_shuffled",
    help="Output directory for shuffled batches"
  )
  parser.add_argument(
    "--num-buckets",
    type=int,
    default=20,
    help="Number of temporary buckets for first pass"
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=10000,
    help="Number of rows per output batch"
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
  )
  
  args = parser.parse_args()
  
  optimized_shuffle(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    num_buckets=args.num_buckets,
    batch_size=args.batch_size,
    random_seed=args.seed
  )

if __name__ == "__main__":
  main()