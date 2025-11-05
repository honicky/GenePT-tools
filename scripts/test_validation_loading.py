#!/usr/bin/env python3
"""Test script for validation data loading with composable test dataset."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import MLPTrainer
from src.training.config import TrainingConfig


def test_validation_loading():
    """Test validation data loading with composable test dataset."""

    print("=" * 80)
    print("Testing Validation Data Loading with Composable Test Dataset")
    print("=" * 80)

    # Load cell counts and create filtering
    print("\n1. Loading cell counts...")
    cell_counts_file = Path("/data/batch-jobs/cell_counts.csv")
    df_counts = pd.read_csv(cell_counts_file)

    # Filter to cell types with >= 5000 samples
    threshold = 5000
    valid_types = df_counts[df_counts['cell_count'] >= threshold]['cell_type'].tolist()
    print(f"   Total cell types: {len(df_counts)}")
    print(f"   Cell types with >= {threshold} samples: {len(valid_types)}")

    # Create cell type codes mapping (sequential codes for valid types)
    cell_type_codes = pd.Series(range(len(valid_types)), index=valid_types)

    # Create code remapping (not needed since we're only using valid types)
    # But set to empty dict for compatibility
    code_remapping = {}

    print(f"   Valid cell types (classes): {len(valid_types)}")

    # Create minimal training config
    print("\n2. Creating training configuration...")
    config = TrainingConfig(
        # Composable dataset settings
        use_composable_dataset=True,
        base_data_dir=Path("/mmc-scratch/scratch/"),
        embedding_types=['genept', 'tissue', 'metadata'],
        genept_dims=1536,

        # Test data configuration
        test_genept_suffix="_test_v1_scgpt",
        test_tissue_suffix="_test_v1_tissue",
        test_metadata_suffix="_test_v1",

        # Cell type filtering
        cell_count_threshold=threshold,
        cell_counts_file=cell_counts_file,

        # Training parameters (minimal for test)
        epochs=1,
        batch_size=1024,
        learning_rate=1e-4,

        # Other settings
        device='cpu',  # Use CPU for testing
        verbose=True,
        track_invalid_embeddings=False,
        seed=42
    )

    print("   Configuration:")
    print(f"   - Embedding types: {config.embedding_types}")
    print(f"   - GenePT dims: {config.genept_dims}")
    print(f"   - Cell type threshold: {threshold}")
    print(f"   - Test suffixes: genept={config.test_genept_suffix}, tissue={config.test_tissue_suffix}, metadata={config.test_metadata_suffix}")

    # Create trainer (but don't initialize model to save time)
    print("\n3. Creating trainer...")
    trainer = MLPTrainer(config, valid_types, cell_type_codes)

    # Manually set code remapping (normally done in load_training_data)
    trainer.code_remapping = code_remapping
    trainer.num_classes = len(valid_types)

    print(f"   Number of classes: {trainer.num_classes}")

    # Test validation data loading
    print("\n4. Loading validation data...")
    try:
        trainer.load_validation_data()

        if trainer.X_val_120k is not None:
            print("\n   ✓ Validation data loaded successfully!")
            print(f"   - Full validation set shape: X={trainer.X_val_120k.shape}, y={trainer.y_val_120k.shape}")
            print(f"   - Full validation samples: {len(trainer.X_val_120k)}")
            print(f"   - Validation X dtype: {trainer.X_val_120k.dtype}")
            print(f"   - Validation y dtype: {trainer.y_val_120k.dtype}")
            print(f"   - Validation y range: {trainer.y_val_120k.min()} to {trainer.y_val_120k.max()}")
            print(f"   - Validation y unique classes: {len(np.unique(trainer.y_val_120k))}")

            if trainer.X_val_5k is not None:
                print(f"\n   - 5k subset shape: X={trainer.X_val_5k.shape}, y={trainer.y_val_5k.shape}")
                print(f"   - 5k subset samples: {len(trainer.X_val_5k)}")

            # Sanity checks
            print("\n5. Running sanity checks...")

            # Check for NaNs
            if np.isnan(trainer.X_val_120k).any():
                print("   ⚠ WARNING: Validation X contains NaN values")
            else:
                print("   ✓ No NaN values in validation X")

            # Check y is in valid range
            if (trainer.y_val_120k < 0).any():
                print(f"   ⚠ WARNING: Validation y contains negative values: {(trainer.y_val_120k < 0).sum()} samples")
            else:
                print("   ✓ All validation y values are non-negative")

            if (trainer.y_val_120k >= trainer.num_classes).any():
                print(f"   ⚠ WARNING: Validation y contains out-of-range values: {(trainer.y_val_120k >= trainer.num_classes).sum()} samples")
            else:
                print(f"   ✓ All validation y values are in range [0, {trainer.num_classes})")

            # Check dimensions match expected (metadata doesn't add a dimension in test mode)
            expected_dims = 1536 + 126  # genept + tissue (metadata provides labels, not features)
            if trainer.X_val_120k.shape[1] == expected_dims:
                print(f"   ✓ Validation X dimensions match expected: {expected_dims}")
            else:
                print(f"   ⚠ WARNING: Validation X dimensions {trainer.X_val_120k.shape[1]} != expected {expected_dims}")

            print("\n" + "=" * 80)
            print("Validation data loading test PASSED!")
            print("=" * 80)

        else:
            print("\n   ✗ Validation data is None (loading may have failed)")
            print("\n" + "=" * 80)
            print("Validation data loading test FAILED!")
            print("=" * 80)
            return False

    except Exception as e:
        print(f"\n   ✗ Error during validation loading: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("Validation data loading test FAILED!")
        print("=" * 80)
        return False

    return True


if __name__ == '__main__':
    success = test_validation_loading()
    sys.exit(0 if success else 1)
