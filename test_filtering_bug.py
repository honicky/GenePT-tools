#!/usr/bin/env python3
"""Test script to reproduce the filtering bug with cell_count_threshold=13209"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loading.composable_dataset import ComposableTrainingDataset

# Load cell types
cell_counts_file = Path("/data/batch-jobs/cell_counts.csv")
counts_df = pd.read_csv(cell_counts_file)

# Apply threshold
threshold = 13209
included_df = counts_df[counts_df['cell_count'] >= threshold]
print(f"Threshold: {threshold}")
print(f"Included cell types ({len(included_df)}):")
print(included_df[['cell_type', 'cell_count']])

# Create filtered codes
filtered_cell_types = included_df['cell_type'].tolist()
filtered_codes = pd.Series(range(len(filtered_cell_types)), index=filtered_cell_types)

print(f"\nFiltered codes:")
print(filtered_codes)

# Create test dataset
print(f"\n{'='*60}")
print("Creating test dataset...")
print(f"{'='*60}")

test_dataset = ComposableTrainingDataset(
    base_dir=Path("/localdata/training_data/"),
    embedding_types=["genept", "tissue", "metadata"],
    batch_size=1024,
    genept_dims=1536,
    code_remapping=None,  # We'll check what happens WITHOUT remapping first
    track_invalid_embeddings=True,
    seed=4201,
    is_test_mode=True,
    test_genept_suffix="_test_v1_scgpt",
    test_tissue_suffix="_test_v1_tissue",
    test_metadata_suffix="_test_v1",
    cell_type_codes=filtered_codes,  # Only filtered cell types
    verbose=True
)

print(f"\nTest dataset has {len(test_dataset.file_list)} files")

# Load first batch
print(f"\n{'='*60}")
print("Loading first batch...")
print(f"{'='*60}")

batch_iterator = iter(test_dataset)
try:
    X_batch, y_batch = next(batch_iterator)
    print(f"\nFirst batch loaded:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  y shape: {y_batch.shape}")
    print(f"  y unique values: {np.unique(y_batch.numpy())}")
    print(f"  y value counts:")
    unique, counts = np.unique(y_batch.numpy(), return_counts=True)
    for val, count in zip(unique, counts):
        print(f"    {val}: {count} samples")
except StopIteration:
    print("\n⚠️  No batches yielded - all samples filtered out!")
except Exception as e:
    print(f"\n❌ Error loading batch: {e}")
    import traceback
    traceback.print_exc()

# Now try loading ALL batches to see total samples
print(f"\n{'='*60}")
print("Loading ALL test batches...")
print(f"{'='*60}")

all_y = []
all_X_sizes = []

for X_batch, y_batch in test_dataset:
    all_y.append(y_batch.numpy())
    all_X_sizes.append(X_batch.shape[0])

if all_y:
    y_combined = np.concatenate(all_y)
    total_samples = len(y_combined)
    print(f"\nTotal samples loaded: {total_samples:,}")
    print(f"Batch sizes: {all_X_sizes}")

    # Count by label
    unique, counts = np.unique(y_combined, return_counts=True)
    print(f"\nLabel distribution:")
    for val, count in zip(unique, counts):
        print(f"  Label {val}: {count:,} samples")

    # Check if all are -1 (not in vocabulary)
    if len(unique) == 1 and unique[0] == -1:
        print(f"\n⚠️  BUG CONFIRMED: All {total_samples:,} samples have label -1 (not in filtered vocabulary)")
        print(f"     These will all be filtered out, leaving 0 samples for evaluation!")

    # Check what happens after filtering
    valid_mask = (y_combined >= 0)
    valid_samples = valid_mask.sum()
    print(f"\nAfter filtering (y >= 0):")
    print(f"  Valid samples: {valid_samples:,} ({100*valid_samples/total_samples:.1f}%)")
    print(f"  Filtered samples: {total_samples - valid_samples:,} ({100*(total_samples - valid_samples)/total_samples:.1f}%)")

else:
    print("\n⚠️  No samples loaded at all!")
