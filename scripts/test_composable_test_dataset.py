#!/usr/bin/env python3
"""Test script for composable test dataset with parquet files."""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loading.composable_dataset import ComposableTrainingDataset


def test_composable_test_dataset():
    """Test loading composable test dataset with parquet files."""

    print("=" * 80)
    print("Testing Composable Test Dataset with Parquet Files")
    print("=" * 80)

    # Load cell types to get mapping
    print("\n1. Loading cell types...")
    cell_types_file = Path("/data/batch-jobs/cell_types.csv")
    df = pd.read_csv(cell_types_file)
    cell_types = df['cell_type'].tolist()
    cell_type_codes = pd.Series(df['code'].values, index=df['cell_type'].values)
    print(f"   Loaded {len(cell_types)} cell types")
    print(f"   Cell type codes (first 5): {cell_type_codes.head()}")

    # Create test dataset with GenePT + Tissue + Metadata
    print("\n2. Creating composable test dataset...")
    print("   Configuration:")
    print("   - Embedding types: genept, tissue, metadata")
    print("   - GenePT dims: 1536")
    print("   - Number of files: 2 (for testing)")
    print("   - Batch size: 1024")
    print("   - Shuffling: disabled (test mode)")

    dataset = ComposableTrainingDataset(
        base_dir=Path("/mmc-scratch/scratch/"),
        embedding_types=['genept', 'tissue', 'metadata'],
        batch_size=1024,
        genept_dims=1536,
        end_batch_file=2,  # Test with just 2 files
        # Test mode parameters
        is_test_mode=True,
        test_genept_suffix="_test_v1_scgpt",
        test_tissue_suffix="_test_v1_tissue",
        test_metadata_suffix="_test_v1",
        cell_type_codes=cell_type_codes,
        # Disable invalid tracking for test
        track_invalid_embeddings=False,
        verbose=True
    )

    print("\n3. Dataset initialized successfully!")
    print(f"   Total dimensions: {dataset.get_total_dims()}")
    print(f"   Number of classes: {dataset.n_classes}")
    print(f"   Files to process: {len(dataset.file_list)}")

    # Test iteration
    print("\n4. Testing iteration (first 3 batches)...")
    batch_count = 0
    total_samples = 0

    for X, y in dataset:
        batch_count += 1
        total_samples += len(X)

        print(f"\n   Batch {batch_count}:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - X dtype: {X.dtype}")
        print(f"   - y dtype: {y.dtype}")
        print(f"   - X min/max: {X.min():.4f} / {X.max():.4f}")
        print(f"   - y unique values: {len(torch.unique(y))} classes")
        print(f"   - y min/max: {y.min()} / {y.max()}")

        # Check for NaNs or invalid values
        if torch.isnan(X).any():
            print(f"   WARNING: X contains NaN values")
        if torch.isnan(y.float()).any():
            print(f"   WARNING: y contains NaN values")
        if (y < 0).any():
            print(f"   WARNING: y contains negative values (unmapped cell types)")

        if batch_count >= 3:
            print(f"\n   ... stopping after 3 batches for testing")
            break

    print(f"\n5. Summary:")
    print(f"   - Processed {batch_count} batches")
    print(f"   - Total samples: {total_samples}")
    print(f"   - Average batch size: {total_samples / batch_count:.1f}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    test_composable_test_dataset()
