# Composable Embeddings Training Guide

This guide explains how to use the new composable embeddings system for training MLP cell type classifiers with different embedding combinations.

## Quick Start

### 1. Test Dataset Loading

First, verify that the composable dataset works correctly:

```bash
# Test GenePT only (1536 dimensions)
python scripts/test_composable_dataset.py \
  --base-dir /mmc-scratch/scratch \
  --embedding-types genept \
  --genept-dims 1536 \
  --n-batches 2

# Test GenePT + Tissue
python scripts/test_composable_dataset.py \
  --embedding-types genept tissue \
  --n-batches 2

# Test GenePT + scGPT
python scripts/test_composable_dataset.py \
  --embedding-types genept scgpt \
  --n-batches 2

# Test all embeddings (GenePT + scGPT + Tissue)
python scripts/test_composable_dataset.py \
  --embedding-types genept scgpt tissue \
  --n-batches 2
```

### 2. Run Training with Composable Embeddings

#### GenePT Only (1536 dimensions)

```bash
python scripts/train_cellxgene_mlp.py \
  --use-composable-dataset \
  --base-data-dir /mmc-scratch/scratch \
  --embedding-types genept \
  --genept-dims 1536 \
  --cell-types-file /mmc-scratch/scratch/cellxgene_v2_metadata/cell_types.csv \
  --checkpoint-dir /tmp/checkpoints/genept_only \
  --epochs 2 \
  --end-batch-file 10 \
  --verbose
```

#### GenePT + Tissue (1662 dimensions)

```bash
python scripts/train_cellxgene_mlp.py \
  --use-composable-dataset \
  --base-data-dir /mmc-scratch/scratch \
  --embedding-types genept tissue \
  --genept-dims 1536 \
  --cell-types-file /mmc-scratch/scratch/cellxgene_v2_metadata/cell_types.csv \
  --checkpoint-dir /tmp/checkpoints/genept_tissue \
  --epochs 2 \
  --end-batch-file 10 \
  --verbose
```

#### GenePT + scGPT (2048 dimensions)

```bash
python scripts/train_cellxgene_mlp.py \
  --use-composable-dataset \
  --base-data-dir /mmc-scratch/scratch \
  --embedding-types genept scgpt \
  --genept-dims 1536 \
  --cell-types-file /mmc-scratch/scratch/cellxgene_v2_metadata/cell_types.csv \
  --checkpoint-dir /tmp/checkpoints/genept_scgpt \
  --epochs 2 \
  --end-batch-file 10 \
  --verbose
```

#### All Embeddings (2174 dimensions)

```bash
python scripts/train_cellxgene_mlp.py \
  --use-composable-dataset \
  --base-data-dir /mmc-scratch/scratch \
  --embedding-types genept scgpt tissue \
  --genept-dims 1536 \
  --cell-types-file /mmc-scratch/scratch/cellxgene_v2_metadata/cell_types.csv \
  --checkpoint-dir /tmp/checkpoints/all_embeddings \
  --epochs 2 \
  --end-batch-file 10 \
  --verbose
```

## Command Line Options

### Composable Dataset Options

- `--use-composable-dataset`: Enable the composable embedding system
- `--base-data-dir PATH`: Base directory containing embedding subdirectories (e.g., `/mmc-scratch/scratch/`)
- `--embedding-types TYPE [TYPE ...]`: List of embedding types to combine (e.g., `genept tissue scgpt`)
- `--genept-dims N`: Number of GenePT dimensions to use (default: 1536, use 0 for all 3072)

### Standard Training Options

- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Mini-batch size (default: 1024)
- `--learning-rate LR`: Learning rate (default: 4.366e-5)
- `--dropout RATE`: Dropout rate (default: 0.053)
- `--n-hidden-layers N`: Number of hidden layers (default: 3)
- `--n-dims N`: Hidden layer dimensions (default: 500)

### Subset Options (for quick testing)

- `--start-batch-file N`: Start from batch file N (default: 0)
- `--end-batch-file N`: End at batch file N (default: all files)
- `--max-steps-per-epoch N`: Limit training steps per epoch

### Logging Options

- `--wandb-project NAME`: Weights & Biases project name
- `--wandb-entity ENTITY`: Weights & Biases entity/team
- `--wandb-run-name NAME`: Custom run name
- `--checkpoint-dir PATH`: Directory for saving checkpoints

## Embedding Dimensions

The composable system automatically calculates input dimensions based on selected embeddings:

| Configuration | Embeddings | Total Dimensions |
|--------------|------------|------------------|
| GenePT Only | genept (1536) | 1,536 |
| GenePT + Tissue | genept (1536) + tissue (126) | 1,662 |
| GenePT + scGPT | genept (1536) + scgpt (512) | 2,048 |
| All Embeddings | genept (1536) + scgpt (512) + tissue (126) | 2,174 |

**Note**: GenePT has 3,072 total dimensions, but we use only the first 1,536 for faster training and reduced overfitting.

## Data Location

The composable embeddings are stored in JuiceFS at:

```
/mmc-scratch/scratch/
├── cellxgene_v2_training_v1_shuffled_genept/
│   ├── batch_0000.pt
│   ├── batch_0001.pt
│   ├── ...
│   ├── batch_0377.pt
│   └── metadata.pt
├── cellxgene_v2_training_v1_shuffled_scgpt/
│   ├── batch_0000.pt
│   ├── ...
│   └── metadata.pt
└── cellxgene_v2_training_v1_shuffled_tissue/
    ├── batch_0000.pt
    ├── ...
    └── metadata.pt
```

Each batch file contains:
- `X`: Embedding tensor (samples × dimensions)
- `row_hash`: Row identifiers for alignment
- `y`: Cell type labels (optional, in first embedding type)

## Test/Validation Data

The composable system supports test/validation data for hyperparameter tuning and model evaluation.

### Test Data Structure

Test data follows the same structure as training data, with key differences:

1. **No shuffling**: Test data is unshuffled (no need to shuffle since we're not training)
2. **File format**: Uses `.parquet` files instead of `.pt` files
3. **Directory naming**: Different suffix pattern (e.g., `_test_v1_scgpt` instead of `_shuffled_scgpt`)

### Test Data Locations

```
/mmc-scratch/scratch/
├── cellxgene_v2_test_v1/              # Original test parquet files
│   ├── <uuid>.parquet
│   └── ...
├── cellxgene_v2_test_v1_scgpt/        # GenePT embeddings (test)
│   ├── <uuid>.parquet
│   └── ...
└── cellxgene_v2_test_v1_tissue/       # Tissue embeddings (test)
    ├── <uuid>.parquet
    └── ...
```

### Configuring Test Data

When using composable datasets with test/validation, specify the test data suffixes:

```python
# In configuration or command line:
test_genept_suffix: "_test_v1_scgpt"
test_tissue_suffix: "_test_v1_tissue"
test_metadata_suffix: "_test_v1"
```

**Example in Optuna tuning config:**

```yaml
fixed_params:
  use_composable_dataset: true
  base_data_dir: "/mmc-scratch/scratch/"
  embedding_types: ["genept", "tissue", "metadata"]

  # Test data configuration (different suffixes)
  test_genept_suffix: "_test_v1_scgpt"
  test_tissue_suffix: "_test_v1_tissue"
  test_metadata_suffix: "_test_v1"
```

### Requirements for Test Data Support

To add test/validation support to the `ComposableTrainingDataset`:

1. **Same alignment mechanism**: Use `row_hash` for aligning embeddings across types
2. **Parquet file loading**: Support loading from `.parquet` files (contains same structure as `.pt` files)
3. **No shuffling**: Disable file and sample shuffling for test data
4. **Flexible suffixes**: Allow configurable directory suffixes for train vs test data

## Troubleshooting

### Dataset Loading Errors

If you encounter errors loading the dataset:

1. **Verify JuiceFS is mounted**: `ls /mmc-scratch/scratch/`
2. **Check embedding directories exist**:
   ```bash
   ls /mmc-scratch/scratch/cellxgene_v2_training_v1_shuffled_genept/
   ls /mmc-scratch/scratch/cellxgene_v2_training_v1_shuffled_tissue/
   ls /mmc-scratch/scratch/cellxgene_v2_training_v1_shuffled_scgpt/
   ```
3. **Run test script** to isolate the issue:
   ```bash
   python scripts/test_composable_dataset.py --n-batches 1
   ```

### Dimension Mismatches

If you see dimension mismatch errors:

- Verify `--genept-dims` matches your model's expected input size
- Check that all embedding types are spelled correctly
- Use the test script to verify actual loaded dimensions

### Memory Issues

If you run out of memory:

- Reduce `--batch-size` (e.g., from 1024 to 512)
- Use fewer batch files with `--end-batch-file` (e.g., `--end-batch-file 50`)
- Use fewer embedding types (start with GenePT only)

## Implementation Details

### ComposableTrainingDataset

The dataset class (`src/data_loading/composable_dataset.py`) handles:

- Loading embeddings from multiple types
- Aligning embeddings by row_hash
- Slicing GenePT to specified dimensions
- Batch-wise streaming for memory efficiency
- File and sample shuffling

### TrainingConfig Changes

New parameters added to `TrainingConfig`:

```python
use_composable_dataset: bool = False
base_data_dir: Optional[Path] = None
embedding_types: List[str] = ["genept"]
genept_dims: Optional[int] = 1536
```

### Trainer Integration

The `MLPTrainer` now checks `config.use_composable_dataset` and creates a `ComposableTrainingDataset` instead of `PTFileStreamDataset` when enabled.

## Next Steps

After validating that training works with composable embeddings:

1. **Run full training** (all 378 batch files)
2. **Hyperparameter tuning** with Optuna (see spec: `/data/batch-jobs/specs/mlp-training-hyperparameter-tuning.md`)
3. **Compare performance** across embedding configurations
4. **Submit AWS Batch jobs** for distributed hyperparameter search

See the main specification for details on hyperparameter tuning and batch job setup.
