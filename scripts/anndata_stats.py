#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to access src
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

import argparse
import json
from typing import Any

from src.utils import get_anndata_file_info


def print_human_readable(info: dict[str, Any]) -> None:
  """
    Print the information in a human-readable format.
    """
  print(f"Dataset file size: {info['file_size_gb']:.2f} GB")
  print(f"\nMain HDF5 groups: {info['main_groups']}")

  if 'x_storage' in info:
    print(f"\nExpression matrix (X) storage:")
    print(f"  Format: {info['x_storage']['format']}")
    print(f"  Components: {info['x_storage']['components']}")

    if 'dtype' in info['x_storage']:
      print(f"  Data type: {info['x_storage']['dtype']}")

    if 'matrix_shape' in info['x_storage']:
      print(f"  Matrix shape: {info['x_storage']['matrix_shape']}")
      if info['x_storage']['format'] == 'CSR':
        print(f"    Rows correspond to observations (e.g., cells)")
        print(f"    Columns correspond to variables (e.g., genes)")
      elif info['x_storage']['format'] == 'CSC':
        print(f"    Rows correspond to variables (e.g., genes)")
        print(f"    Columns correspond to observations (e.g., cells)")

    if 'chunk_size' in info['x_storage']:
      print(f"  Chunk size: {info['x_storage']['chunk_size']}")

    if 'data_shape' in info['x_storage']:
      print(f"\n  Storage details:")
      print(f"    Data array shape: {info['x_storage']['data_shape']}")

      # Format-specific shape information
      if info['x_storage']['format'] in ['CSR', 'CSC']:
        print(f"    Indices array shape: {info['x_storage']['indices_shape']}")
        print(f"    Index pointer array shape: {info['x_storage']['indptr_shape']}")
      elif info['x_storage']['format'] == 'COO':
        print(f"    Row indices shape: {info['x_storage']['row_shape']}")
        print(f"    Column indices shape: {info['x_storage']['col_shape']}")

  print(f"\nStored embeddings: {info['embeddings'] or 'No embeddings found'}")
  print(
    f"Pairwise relationships: {info['pairwise_relationships'] or 'No pairwise relationships found'}"
  )
  print(
    f"Additional expression matrices: {info['expression_layers'] or 'No additional layers found'}"
  )

  if 'obs_contents' in info:
    print(f"\nContents of obs group: {info['obs_contents']}")
  if 'var_contents' in info:
    print(f"Contents of var group: {info['var_contents']}")
  if 'cell_type_count' in info:
    print(f"\nNumber of cell type categories: {info['cell_type_count']}")


def main():
  parser = argparse.ArgumentParser(description='Analyze H5 file structure and contents')
  parser.add_argument('file_path', help='Path to the H5 file')
  parser.add_argument('--json', action='store_true', help='Output in JSON format')
  args = parser.parse_args()

  try:
    info = get_anndata_file_info(args.file_path)

    if args.json:
      print(json.dumps(info, indent=2))
    else:
      print_human_readable(info)

  except FileNotFoundError:
    print(f"Error: File '{args.file_path}' not found")
  except Exception as e:
    print(f"Error analyzing file: {str(e)}")


if __name__ == "__main__":
  main()
