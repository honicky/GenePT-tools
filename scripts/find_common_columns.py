#!/usr/bin/env python
"""Find common columns across all parquet files in training_v1 directory."""

import pandas as pd
from pathlib import Path

def find_common_columns():
  dir_path = Path('data/cellxgene_embeddings/training_v1')
  parquet_files = list(dir_path.glob('*.parquet'))
  
  print(f"Found {len(parquet_files)} parquet files")
  
  all_columns = []
  file_info = []
  
  for i, file_path in enumerate(parquet_files):
    if i % 20 == 0:
      print(f"Processing file {i+1}/{len(parquet_files)}...")
    
    df = pd.read_parquet(file_path, columns=['cell_type'])  # Read minimal to get column names
    columns = set(pd.read_parquet(file_path).columns)
    all_columns.append(columns)
    
    file_info.append({
      'file': file_path.name,
      'num_columns': len(columns),
      'shape': pd.read_parquet(file_path).shape
    })
  
  # Find common columns
  common_columns = set.intersection(*all_columns)
  
  print(f"\nCommon columns across all files ({len(common_columns)} columns):")
  
  # Separate numeric and non-numeric columns
  numeric_cols = sorted([col for col in common_columns if col.isdigit()], key=int)
  non_numeric_cols = sorted([col for col in common_columns if not col.isdigit()])
  
  print(f"\nNon-numeric columns ({len(non_numeric_cols)}):")
  for col in non_numeric_cols:
    print(f"  - {col}")
  
  print(f"\nNumeric columns: {numeric_cols[0]} to {numeric_cols[-1]} ({len(numeric_cols)} columns)")
  
  # Check for any file-specific columns
  all_unique_cols = set.union(*all_columns)
  file_specific = all_unique_cols - common_columns
  
  if file_specific:
    print(f"\nFile-specific columns (not in all files): {len(file_specific)} columns")
    print(f"Examples: {list(file_specific)[:10]}")
  
  # Show file statistics
  print(f"\nFile statistics:")
  total_rows = sum(info['shape'][0] for info in file_info)
  print(f"Total rows across all files: {total_rows:,}")
  print(f"Average rows per file: {total_rows/len(file_info):,.0f}")
  
  return common_columns, file_info

if __name__ == "__main__":
  common_cols, files = find_common_columns()