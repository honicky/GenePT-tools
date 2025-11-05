#!/usr/bin/env python3
"""
Reduce GenePT embedding dimensions in training batch files.

This script processes .pt batch files containing GenePT embeddings and reduces
the embedding dimensions from 3072 to 1536 (or any specified size). This reduces
file sizes and I/O bandwidth by ~50%.

Usage:
    python reduce_genept_dims.py --input-dir /path/to/input --output-dir /path/to/output --dims 1536

The script:
1. Reads each batch_*.pt file
2. Slices the 'X' tensor to keep only the first N dimensions
3. Saves to a new directory with the same filename
4. Shows progress and validates the conversion
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch


def reduce_dimensions(input_file: Path, output_file: Path, target_dims: int) -> tuple[int, int]:
    """
    Reduce GenePT dimensions in a single batch file.

    Args:
        input_file: Path to input .pt file
        output_file: Path to output .pt file
        target_dims: Number of dimensions to keep (e.g., 1536)

    Returns:
        Tuple of (original_size_mb, new_size_mb)
    """
    # Load the batch file
    data = torch.load(input_file, weights_only=False)

    # Validate structure
    if not isinstance(data, dict) or 'X' not in data:
        raise ValueError(f"Invalid batch file structure: {input_file}")

    X = data['X']
    if not isinstance(X, torch.Tensor):
        raise ValueError(f"'X' is not a tensor in {input_file}")

    original_dims = X.shape[1]
    if original_dims < target_dims:
        raise ValueError(f"File has {original_dims} dims, cannot reduce to {target_dims}")

    # Slice to keep only first target_dims dimensions
    data['X'] = X[:, :target_dims]

    # Save to output file
    torch.save(data, output_file)

    # Get file sizes for reporting
    original_size = input_file.stat().st_size / (1024 * 1024)  # MB
    new_size = output_file.stat().st_size / (1024 * 1024)  # MB

    return original_size, new_size


def main():
    parser = argparse.ArgumentParser(
        description='Reduce GenePT embedding dimensions in training batch files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing batch_*.pt files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for reduced-dimension files'
    )
    parser.add_argument(
        '--dims',
        type=int,
        default=1536,
        help='Number of dimensions to keep (default: 1536)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing files'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all batch files
    batch_files = sorted(input_dir.glob('batch_*.pt'))
    if not batch_files:
        print(f"Error: No batch_*.pt files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(batch_files)} batch files in {input_dir}")
    print(f"Will reduce dimensions from 3072 to {args.dims}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\nDRY RUN - No files will be processed")
        print(f"Would process {len(batch_files)} files")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    total_original_size = 0
    total_new_size = 0

    print(f"\nProcessing {len(batch_files)} files...")

    for input_file in tqdm(batch_files, desc="Processing batches"):
        output_file = output_dir / input_file.name

        try:
            original_size, new_size = reduce_dimensions(input_file, output_file, args.dims)
            total_original_size += original_size
            total_new_size += new_size
        except Exception as e:
            print(f"\nError processing {input_file.name}: {e}")
            sys.exit(1)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Files processed: {len(batch_files)}")
    print(f"Original total size: {total_original_size / 1024:.2f} GB")
    print(f"New total size: {total_new_size / 1024:.2f} GB")
    print(f"Space saved: {(total_original_size - total_new_size) / 1024:.2f} GB ({100 * (1 - total_new_size / total_original_size):.1f}%)")
    print(f"I/O bandwidth reduction: ~{100 * (1 - args.dims / 3072):.1f}%")

    # Validate one file to confirm structure
    print("\nValidating output...")
    test_file = output_dir / batch_files[0].name
    test_data = torch.load(test_file, weights_only=False)
    print(f"Sample output file: {test_file.name}")
    print(f"  X shape: {test_data['X'].shape}")
    print(f"  Expected: [10000, {args.dims}]")

    if test_data['X'].shape[1] == args.dims:
        print("  ✓ Validation passed!")
    else:
        print("  ✗ Validation failed - unexpected dimensions!")
        sys.exit(1)


if __name__ == '__main__':
    main()
