#!/usr/bin/env python
"""Test script for ComposableTrainingDataset.

This script validates that the composable embedding dataset loads correctly
and produces expected outputs before running full training.
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading.composable_dataset import ComposableTrainingDataset


def test_dataset(
    base_dir: Path,
    embedding_types: list,
    genept_dims: int = 1536,
    n_batches: int = 2,
    batch_size: int = 512
):
    """
    Test the composable dataset with specified configuration.

    Args:
        base_dir: Base directory containing embedding subdirectories
        embedding_types: List of embedding types to test
        genept_dims: Number of GenePT dimensions to use
        n_batches: Number of batch files to load (for quick testing)
        batch_size: Mini-batch size
    """
    print("="*80)
    print("Testing ComposableTrainingDataset")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Base directory: {base_dir}")
    print(f"  Embedding types: {embedding_types}")
    print(f"  GenePT dims: {genept_dims}")
    print(f"  Batch files: 0-{n_batches-1} ({n_batches} files)")
    print(f"  Mini-batch size: {batch_size}")
    print()

    # Create dataset
    print("Creating dataset...")
    try:
        dataset = ComposableTrainingDataset(
            base_dir=base_dir,
            embedding_types=embedding_types,
            batch_size=batch_size,
            start_batch_file=0,
            end_batch_file=n_batches,
            genept_dims=genept_dims,
            shuffle_files_per_epoch=False,  # Disable shuffling for testing
            shuffle_within_files=False,
            seed=42,
            verbose=True
        )
        print("\n✓ Dataset created successfully!")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dataset properties
    print(f"\nDataset properties:")
    print(f"  Total dimensions: {dataset.get_total_dims()}")
    print(f"  Number of classes: {dataset.n_classes}")
    print(f"  Estimated batches: {len(dataset)}")
    print(f"  Estimated samples: {dataset.total_samples:,}")

    # Create dataloader
    print(f"\nCreating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Dataset returns batches
        num_workers=0  # Single process for testing
    )

    # Iterate through first few mini-batches
    print(f"\nIterating through mini-batches...")
    n_test_batches = 5
    total_samples = 0
    embedding_dims = None
    label_range = [float('inf'), float('-inf')]

    try:
        for i, (X, y) in enumerate(dataloader):
            if i >= n_test_batches:
                break

            # Check dimensions
            if embedding_dims is None:
                embedding_dims = X.shape[1]
                print(f"\n  Batch shape: {X.shape} (samples × dims)")
                print(f"  Label shape: {y.shape}")
                print(f"  Embedding dtype: {X.dtype}")
                print(f"  Label dtype: {y.dtype}")

            # Verify dimensions match
            if X.shape[1] != embedding_dims:
                print(f"\n✗ ERROR: Inconsistent dimensions in batch {i}: {X.shape[1]} vs {embedding_dims}")
                return False

            # Track statistics
            total_samples += len(X)
            label_range[0] = min(label_range[0], y.min().item())
            label_range[1] = max(label_range[1], y.max().item())

            # Print batch info
            print(f"  Batch {i}: {len(X)} samples, labels [{y.min().item():.0f}, {y.max().item():.0f}]")

            # Check for NaNs or Infs
            if torch.isnan(X).any():
                print(f"\n✗ ERROR: NaN values detected in batch {i}")
                return False
            if torch.isinf(X).any():
                print(f"\n✗ ERROR: Inf values detected in batch {i}")
                return False

        print(f"\n✓ Successfully loaded {n_test_batches} mini-batches ({total_samples} samples)")
        print(f"  Label range: [{label_range[0]:.0f}, {label_range[1]:.0f}]")
        print(f"  Expected classes: {dataset.n_classes}")

        # Verify label range
        if label_range[1] >= dataset.n_classes:
            print(f"\n✗ WARNING: Label {label_range[1]} >= n_classes {dataset.n_classes}")

    except Exception as e:
        print(f"\n✗ ERROR: Failed during iteration: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test dimension calculation
    print(f"\nVerifying dimension calculation...")

    expected_dims = 0
    for emb_type in embedding_types:
        if emb_type == 'genept':
            dims = genept_dims if genept_dims is not None else 3072
        elif emb_type == 'scgpt':
            dims = 512
        elif emb_type == 'tissue':
            dims = 126
        else:
            print(f"✗ ERROR: Unknown embedding type: {emb_type}")
            return False
        expected_dims += dims
        print(f"  {emb_type}: {dims} dims")

    print(f"  Total expected: {expected_dims} dims")
    print(f"  Total actual: {embedding_dims} dims")

    if expected_dims != embedding_dims:
        print(f"\n✗ ERROR: Dimension mismatch!")
        return False

    print(f"\n✓ Dimensions match!")

    # Summary
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print(f"\nDataset is ready for training with:")
    print(f"  Embedding types: {' + '.join(embedding_types)}")
    print(f"  Input dimensions: {embedding_dims}")
    print(f"  Output classes: {dataset.n_classes}")
    print(f"  Estimated total samples: {dataset.total_samples:,}")
    print()

    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test composable embedding dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/mmc-scratch/scratch"),
        help="Base directory containing embedding subdirectories"
    )
    parser.add_argument(
        "--embedding-types",
        nargs="+",
        default=["genept"],
        help="Embedding types to test (e.g., genept tissue scgpt)"
    )
    parser.add_argument(
        "--genept-dims",
        type=int,
        default=1536,
        help="Number of GenePT dimensions (0 for all 3072)"
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=2,
        help="Number of batch files to load for testing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size"
    )

    args = parser.parse_args()

    # Handle genept_dims (0 means use all)
    genept_dims = args.genept_dims if args.genept_dims > 0 else None

    # Run test
    success = test_dataset(
        base_dir=args.base_dir,
        embedding_types=args.embedding_types,
        genept_dims=genept_dims,
        n_batches=args.n_batches,
        batch_size=args.batch_size
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
