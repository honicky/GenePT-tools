#!/usr/bin/env python3
"""Convert Parquet files to PyTorch tensor format for faster loading."""

import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


def convert_parquet_to_pt(
    input_dir: Path,
    output_dir: Path,
    cell_types_file: Path,
    n_dims: int = 500,
    verbose: bool = True
):
  """Convert Parquet files to PyTorch tensor format.
  
  Args:
    input_dir: Directory containing parquet files
    output_dir: Directory to save .pt files
    cell_types_file: CSV file with cell types to keep
    n_dims: Number of embedding dimensions
    verbose: Print progress
  """
  # Load cell types
  cell_types_df = pd.read_csv(cell_types_file)
  # IMPORTANT: Use list to preserve order, not set!
  valid_cell_types = cell_types_df['cell_type'].values.tolist()
  cell_type_to_code = {ct: i for i, ct in enumerate(valid_cell_types)}
  
  if verbose:
    print(f"Loaded {len(valid_cell_types)} valid cell types")
  
  # Create output directory
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Get all parquet files
  parquet_files = sorted(input_dir.glob("*.parquet"))
  
  if verbose:
    print(f"Found {len(parquet_files)} parquet files")
  
  # Process each file and track total samples
  total_samples = 0
  converted_files = 0
  
  for parquet_file in tqdm(parquet_files, desc="Converting files"):
    try:
      # Read parquet
      df = pd.read_parquet(parquet_file)
      
      # Filter to valid cell types
      valid_mask = df['cell_type'].isin(set(valid_cell_types))
      df_filtered = df[valid_mask]
      
      if len(df_filtered) == 0:
        if verbose:
          print(f"Skipping {parquet_file.name}: no valid samples")
        continue
      
      # Extract embeddings
      embedding_cols = [str(i) for i in range(n_dims)]
      X = df_filtered[embedding_cols].values.astype(np.float32)
      
      # Encode labels
      y = df_filtered['cell_type'].map(cell_type_to_code).values.astype(np.int64)
      
      # Convert to tensors
      X_tensor = torch.from_numpy(X)
      y_tensor = torch.from_numpy(y)
      
      # Save as dictionary
      output_file = output_dir / f"{parquet_file.stem}.pt"
      n_samples = len(X_tensor)
      torch.save({
        'X': X_tensor,
        'y': y_tensor,
        'n_samples': n_samples
      }, output_file)
      
      # Update totals
      total_samples += n_samples
      converted_files += 1
      
      if verbose and len(parquet_files) < 10:
        print(f"Saved {output_file.name}: {n_samples} samples")
        
    except Exception as e:
      print(f"Error processing {parquet_file}: {e}")
      continue
  
  # Save metadata with total sample count
  metadata = {
    'cell_types': valid_cell_types,  # Already a list with correct order
    'cell_type_codes': cell_type_to_code,
    'n_dims': n_dims,
    'n_files': converted_files,
    'total_samples': total_samples
  }
  torch.save(metadata, output_dir / 'metadata.pt')
  
  if verbose:
    print(f"\nConversion complete!")
    print(f"Output files: {converted_files} .pt files")
    print(f"Total samples: {total_samples:,}")
    print(f"Metadata saved to: {output_dir / 'metadata.pt'}")


def main():
  parser = argparse.ArgumentParser(description="Convert Parquet files to PyTorch tensor format")
  parser.add_argument(
    '--input-dir',
    type=Path,
    required=True,
    help='Input directory containing parquet files'
  )
  parser.add_argument(
    '--output-dir',
    type=Path,
    required=True,
    help='Output directory for .pt files'
  )
  parser.add_argument(
    '--cell-types-file',
    type=Path,
    default=Path('cell_types_filtered.csv'),
    help='CSV file with valid cell types (default: cell_types_filtered.csv)'
  )
  parser.add_argument(
    '--n-dims',
    type=int,
    default=500,
    help='Number of embedding dimensions (default: 500)'
  )
  
  args = parser.parse_args()
  
  # Convert data
  convert_parquet_to_pt(
    args.input_dir,
    args.output_dir,
    args.cell_types_file,
    args.n_dims
  )


if __name__ == '__main__':
  main()