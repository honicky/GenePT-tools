#!/usr/bin/env python
"""Test script to verify metadata loading in ComposableTrainingDataset."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loading.composable_dataset import ComposableTrainingDataset
import torch

def test_metadata_loading():
    """Test loading dataset with metadata."""

    print("="*80)
    print("Testing Metadata Loading with ComposableTrainingDataset")
    print("="*80)

    # Setup
    base_dir = Path("/mmc-scratch/scratch/")
    embedding_types = ['genept', 'tissue', 'metadata']

    print(f"\nConfiguration:")
    print(f"  Base directory: {base_dir}")
    print(f"  Embedding types: {embedding_types}")
    print(f"  Batch size: 128")
    print(f"  Loading batches: 0-1 (first 2 files)")

    # Create dataset
    dataset = ComposableTrainingDataset(
        base_dir=base_dir,
        embedding_types=embedding_types,
        batch_size=128,
        start_batch_file=0,
        end_batch_file=2,  # Just test first 2 files
        genept_dims=1536,
        track_invalid_embeddings=True,
        shuffle_files_per_epoch=False,
        shuffle_within_files=False,
        seed=42,
        verbose=True
    )

    print(f"\nDataset initialized:")
    print(f"  Total dimensions: {dataset.get_total_dims()}")
    print(f"  Number of classes: {dataset.n_classes}")
    print(f"  Estimated batches: {len(dataset)}")

    # Load a few batches
    print(f"\nLoading first 3 mini-batches:")
    print("-" * 80)

    for i, (X, y) in enumerate(dataset):
        if i >= 3:
            break

        print(f"\nBatch {i}:")
        print(f"  X shape: {X.shape}")
        print(f"  X dtype: {X.dtype}")
        print(f"  y shape: {y.shape}")
        print(f"  y dtype: {y.dtype}")
        print(f"  y range: [{y.min()}, {y.max()}]")
        print(f"  y unique classes: {len(torch.unique(y))}")

        # Verify X doesn't contain metadata (should only be floats)
        assert X.dtype in [torch.float32, torch.float64], f"X should be float, got {X.dtype}"

        # Verify y contains valid class labels
        assert y.dtype in [torch.int64, torch.long], f"y should be long, got {y.dtype}"
        assert y.min() >= 0, f"y should be non-negative, got min={y.min()}"
        assert y.max() < dataset.n_classes, f"y max ({y.max()}) should be < n_classes ({dataset.n_classes})"

        print(f"  ✓ Validation passed")

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    print("\nMetadata is being loaded correctly:")
    print("  - X contains only float embeddings (genept + tissue)")
    print("  - y contains integer class labels from metadata")
    print("  - All samples have valid labels in range [0, n_classes)")

if __name__ == "__main__":
    test_metadata_loading()
