#!/usr/bin/env python3
"""Test what labels are actually in training batches after filtering"""

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loading.composable_dataset import ComposableTrainingDataset

# Load cell types and apply filtering (same as train_cellxgene_mlp.py)
cell_counts_file = Path("/data/batch-jobs/cell_counts.csv")
counts_df = pd.read_csv(cell_counts_file)

# Load ALL cell types first (original mapping)
cell_types = counts_df['cell_type'].tolist()
cell_type_codes = pd.Series(range(len(cell_types)), index=cell_types)

print(f"Original cell types: {len(cell_types)}")
print(f"Original codes range: 0 to {len(cell_types)-1}")

# Apply filtering
threshold = 13209
included_df = counts_df[counts_df['cell_count'] >= threshold]
excluded_df = counts_df[counts_df['cell_count'] < threshold]

filtered_cell_types = included_df['cell_type'].tolist()
filtered_codes = pd.Series(range(len(filtered_cell_types)), index=filtered_cell_types)

print(f"\nAfter filtering with threshold={threshold}:")
print(f"  Filtered cell types: {len(filtered_cell_types)}")
print(f"  Filtered codes: {filtered_codes.to_dict()}")

# Create code remapping (same as create_code_remapping function)
code_remapping = {}

# Map included types to new sequential codes (0, 1, 2, ..., N-1)
for cell_type in filtered_cell_types:
    if cell_type in cell_type_codes.index:
        original_code = cell_type_codes[cell_type]
        new_code = filtered_codes[cell_type]
        code_remapping[original_code] = new_code

# Map excluded types to -100 (marker for filtering)
for cell_type in excluded_df['cell_type']:
    if cell_type in cell_type_codes.index:
        original_code = cell_type_codes[cell_type]
        code_remapping[original_code] = -100

print(f"\nCode remapping created:")
print(f"  Total mappings: {len(code_remapping)}")
print(f"  Included mappings (->0): {sum(1 for v in code_remapping.values() if v == 0)}")
print(f"  Excluded mappings (->-100): {sum(1 for v in code_remapping.values() if v == -100)}")

# Show a few example mappings
print(f"\nExample mappings:")
for old_code, new_code in list(code_remapping.items())[:10]:
    ct = cell_types[old_code]
    print(f"  {old_code} -> {new_code}: {ct[:60]}")

# Create TRAINING dataset with code_remapping
print(f"\n{'='*60}")
print("Creating TRAINING dataset with code_remapping...")
print(f"{'='*60}")

train_dataset = ComposableTrainingDataset(
    base_dir=Path("/localdata/training_data/"),
    embedding_types=["genept", "tissue", "metadata"],
    batch_size=1024,
    start_batch_file=0,
    end_batch_file=2,  # Just load first 3 batches
    genept_dims=1536,
    code_remapping=code_remapping,  # ← This should remap excluded types to -100
    track_invalid_embeddings=True,
    shuffle_files_per_epoch=False,
    shuffle_within_files=False,
    seed=4201,
    verbose=True
)

print(f"\nLoading first 3 training batches...")

all_y = []
batch_count = 0

for X_batch, y_batch in train_dataset:
    batch_count += 1
    all_y.append(y_batch.numpy())
    print(f"  Batch {batch_count}: {X_batch.shape[0]} samples")
    unique, counts = np.unique(y_batch.numpy(), return_counts=True)
    print(f"    Labels: {dict(zip(unique.tolist(), counts.tolist()))}")

if all_y:
    y_combined = np.concatenate(all_y)
    unique, counts = np.unique(y_combined, return_counts=True)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA AFTER FILTERING:")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_combined):,}")
    print(f"Unique labels: {unique.tolist()}")
    print(f"Label counts: {dict(zip(unique.tolist(), counts.tolist()))}")

    if len(unique) == 1 and unique[0] == 0:
        print(f"\n✓ SUCCESS: Only label 0 present (single cell type)")
        print(f"   With this, training accuracy should be ~100%")
    else:
        print(f"\n⚠️  BUG FOUND: Multiple labels present after filtering!")
        print(f"   Expected: only label 0")
        print(f"   Got: {unique.tolist()}")
        print(f"   This means code_remapping is NOT working correctly!")
