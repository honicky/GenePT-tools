#!/usr/bin/env python
"""
Optimized out-of-core two-pass shuffle that outputs PyTorch tensor format.
Combines shuffling with conversion to .pt for maximum efficiency.

Based on https://blog.janestreet.com/how-to-shuffle-a-big-dataset/
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import tempfile
import shutil
import gc


class PTBucketWriter:
  """Efficient streaming writer for bucket files in PyTorch format."""
  
  def __init__(self, path):
    self.path = Path(path)
    self.buffer_X = []
    self.buffer_y = []
    self.row_count = 0
  
  def write(self, X, y):
    """Write tensors to bucket buffer."""
    if len(X) == 0:
      return
    
    self.buffer_X.append(X)
    self.buffer_y.append(y)
    self.row_count += len(X)
    
    # Write when buffer gets large enough
    if self.row_count >= 50000:
      self._flush()
  
  def _flush(self):
    """Flush buffer to file."""
    if not self.buffer_X:
      return
    
    # Concatenate all buffered tensors
    combined_X = torch.cat(self.buffer_X, dim=0)
    combined_y = torch.cat(self.buffer_y, dim=0)
    
    # Save to file (append if exists)
    if self.path.exists():
      existing = torch.load(self.path, weights_only=True)
      combined_X = torch.cat([existing['X'], combined_X], dim=0)
      combined_y = torch.cat([existing['y'], combined_y], dim=0)
    
    torch.save({
      'X': combined_X,
      'y': combined_y,
      'n_samples': len(combined_X)
    }, self.path)
    
    self.buffer_X = []
    self.buffer_y = []
  
  def close(self):
    """Close the writer."""
    self._flush()


def optimized_shuffle_to_pt(
    input_dir, 
    output_dir, 
    cell_types_file=None,
    n_dims=500,
    num_buckets=20, 
    batch_size=10000,
    random_seed=42
):
  """
  Perform optimized shuffle and convert to PyTorch format.
  
  Args:
    input_dir: Directory containing input parquet files
    output_dir: Directory to save shuffled .pt batches
    cell_types_file: Optional CSV with valid cell types to filter
    n_dims: Number of embedding dimensions to use
    num_buckets: Number of temporary buckets for first pass
    batch_size: Number of rows per output batch file
    random_seed: Random seed for reproducibility
  """
  np.random.seed(random_seed)
  rng = np.random.RandomState(random_seed)
  torch.manual_seed(random_seed)
  
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  
  # Load valid cell types if provided
  if cell_types_file:
    cell_types_df = pd.read_csv(cell_types_file)
    valid_cell_types = set(cell_types_df['cell_type'].values)
    cell_type_to_code = {ct: i for i, ct in enumerate(valid_cell_types)}
    print(f"Filtering to {len(valid_cell_types)} valid cell types")
  else:
    valid_cell_types = None
    cell_type_to_code = {}
  
  parquet_files = sorted(input_path.glob('*.parquet'))
  print(f"Found {len(parquet_files)} parquet files")
  print(f"Using {n_dims} embedding dimensions")
  
  # Create temporary directory for buckets
  project_temp_dir = Path.cwd() / "temp"
  project_temp_dir.mkdir(exist_ok=True)
  temp_dir = Path(tempfile.mkdtemp(prefix='shuffle_pt_buckets_', dir=project_temp_dir))
  print(f"Using temporary directory: {temp_dir}")
  
  # Embedding columns to extract
  embedding_cols = [str(i) for i in range(n_dims)]
  
  try:
    # PASS 1: Distribute rows randomly into buckets
    print(f"\nPass 1: Distributing data into {num_buckets} buckets...")
    
    # Initialize bucket writers
    bucket_writers = []
    for i in range(num_buckets):
      bucket_path = temp_dir / f"bucket_{i:03d}.pt"
      bucket_writers.append(PTBucketWriter(bucket_path))
    
    # Process input files
    total_rows = 0
    filtered_rows = 0
    
    for file_path in tqdm(parquet_files, desc="Processing files"):
      # Read file
      df = pd.read_parquet(file_path)
      
      # Filter to valid cell types if specified
      if valid_cell_types:
        mask = df['cell_type'].isin(valid_cell_types)
        filtered_rows += (~mask).sum()
        df = df[mask]
      
      if len(df) == 0:
        continue
      
      # Extract embeddings and convert to tensors
      X = torch.from_numpy(df[embedding_cols].values.astype(np.float32))
      
      # Encode labels
      if valid_cell_types:
        y = torch.tensor([cell_type_to_code[ct] for ct in df['cell_type'].values], 
                        dtype=torch.long)
      else:
        # If no filtering, create codes on the fly
        unique_types = df['cell_type'].unique()
        for ct in unique_types:
          if ct not in cell_type_to_code:
            cell_type_to_code[ct] = len(cell_type_to_code)
        y = torch.tensor([cell_type_to_code[ct] for ct in df['cell_type'].values],
                        dtype=torch.long)
      
      num_rows = len(X)
      total_rows += num_rows
      
      # Randomly assign rows to buckets
      bucket_assignments = rng.randint(0, num_buckets, size=num_rows)
      
      # Group by bucket and write
      for bucket_id in range(num_buckets):
        mask = bucket_assignments == bucket_id
        if mask.any():
          bucket_X = X[mask]
          bucket_y = y[mask]
          bucket_writers[bucket_id].write(bucket_X, bucket_y)
      
      del df, X, y
      gc.collect()
    
    # Close all bucket writers
    print("Finalizing buckets...")
    for writer in bucket_writers:
      writer.close()
    
    # Print statistics
    print(f"\nBucket statistics:")
    for i, writer in enumerate(bucket_writers):
      print(f"  Bucket {i:2d}: {writer.row_count:,} rows")
    print(f"  Total rows: {total_rows:,}")
    if filtered_rows > 0:
      print(f"  Filtered out: {filtered_rows:,} rows")
    
    del bucket_writers
    gc.collect()
    
    # PASS 2: Shuffle each bucket and write to output files
    print(f"\nPass 2: Shuffling buckets and writing output batches...")
    
    output_batch_idx = 0
    accumulated_X = []
    accumulated_y = []
    accumulated_count = 0
    
    bucket_paths = sorted(temp_dir.glob("bucket_*.pt"))
    
    for bucket_path in tqdm(bucket_paths, desc="Processing buckets"):
      # Read bucket
      bucket_data = torch.load(bucket_path, weights_only=True)
      
      if bucket_data['n_samples'] == 0:
        continue
      
      X = bucket_data['X']
      y = bucket_data['y']
      
      # Shuffle the bucket
      shuffle_idx = torch.randperm(len(X))
      X = X[shuffle_idx]
      y = y[shuffle_idx]
      
      # Add to accumulator
      accumulated_X.append(X)
      accumulated_y.append(y)
      accumulated_count += len(X)
      
      # Write complete batches
      while accumulated_count >= batch_size:
        # Combine accumulated data
        combined_X = torch.cat(accumulated_X, dim=0)
        combined_y = torch.cat(accumulated_y, dim=0)
        
        # Write batches
        num_complete_batches = accumulated_count // batch_size
        for i in range(num_complete_batches):
          start_idx = i * batch_size
          end_idx = start_idx + batch_size
          
          batch_X = combined_X[start_idx:end_idx]
          batch_y = combined_y[start_idx:end_idx]
          
          output_file = output_path / f"batch_{output_batch_idx:04d}.pt"
          torch.save({
            'X': batch_X,
            'y': batch_y,
            'n_samples': len(batch_X)
          }, output_file)
          output_batch_idx += 1
        
        # Keep remainder
        remainder_start = num_complete_batches * batch_size
        if remainder_start < len(combined_X):
          accumulated_X = [combined_X[remainder_start:]]
          accumulated_y = [combined_y[remainder_start:]]
          accumulated_count = len(accumulated_X[0])
        else:
          accumulated_X = []
          accumulated_y = []
          accumulated_count = 0
        
        del combined_X, combined_y
        gc.collect()
      
      # Clean up bucket file
      bucket_path.unlink()
    
    # Write final partial batch if exists
    if accumulated_X and accumulated_count > 0:
      combined_X = torch.cat(accumulated_X, dim=0)
      combined_y = torch.cat(accumulated_y, dim=0)
      
      output_file = output_path / f"batch_{output_batch_idx:04d}.pt"
      torch.save({
        'X': combined_X,
        'y': combined_y,
        'n_samples': len(combined_X)
      }, output_file)
      print(f"Wrote final partial batch with {len(combined_X)} rows")
      output_batch_idx += 1
    
    # Save metadata
    metadata = {
      'cell_types': list(cell_type_to_code.keys()),
      'cell_type_codes': cell_type_to_code,
      'n_dims': n_dims,
      'n_files': output_batch_idx,
      'total_samples': total_rows - filtered_rows if filtered_rows > 0 else total_rows
    }
    torch.save(metadata, output_path / 'metadata.pt')
    
    print(f"\nShuffle complete! Created {output_batch_idx} output batches in {output_path}")
    
    # Verify output
    print("\nVerifying output batches...")
    output_files = sorted(output_path.glob("batch_*.pt"))
    total_output_rows = 0
    for f in output_files[:5]:  # Check first 5
      data = torch.load(f, weights_only=True)
      total_output_rows += data['n_samples']
      print(f"  {f.name}: {data['n_samples']:,} samples, X shape: {data['X'].shape}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total input files: {len(parquet_files)}")
    print(f"  Total rows processed: {total_rows:,}")
    if filtered_rows > 0:
      print(f"  Rows filtered out: {filtered_rows:,}")
    print(f"  Rows in output: {total_rows - filtered_rows:,}")
    print(f"  Number of buckets: {num_buckets}")
    print(f"  Target batch size: {batch_size:,}")
    print(f"  Output batches: {output_batch_idx}")
    print(f"  Output format: PyTorch tensors (.pt)")
    
  finally:
    # Clean up temporary directory
    if temp_dir.exists():
      print(f"\nCleaning up temporary directory: {temp_dir}")
      shutil.rmtree(temp_dir)


def main():
  parser = argparse.ArgumentParser(
    description="Optimized shuffle with PyTorch tensor output"
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
    default="data/cellxgene_embeddings/training_v1_shuffled_pt",
    help="Output directory for shuffled .pt batches"
  )
  parser.add_argument(
    "--cell-types-file",
    type=str,
    default="cell_types_filtered.csv",
    help="CSV file with valid cell types (optional)"
  )
  parser.add_argument(
    "--n-dims",
    type=int,
    default=500,
    help="Number of embedding dimensions to extract"
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
  
  optimized_shuffle_to_pt(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    cell_types_file=args.cell_types_file if args.cell_types_file else None,
    n_dims=args.n_dims,
    num_buckets=args.num_buckets,
    batch_size=args.batch_size,
    random_seed=args.seed
  )


if __name__ == "__main__":
  main()