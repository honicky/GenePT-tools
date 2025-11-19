# MLP Evaluation Script Specification

## Overview

Create a general-purpose evaluation script (`scripts/evaluate_mlp.py`) that evaluates trained MLP models on any dataset with composable embeddings in parquet format.

## Motivation

We need a flexible evaluation tool to:
1. Evaluate trained models on test sets, validation sets, or custom datasets
2. Support any composable embedding configuration (GenePT, scGPT, metadata, etc.)
3. Compute comprehensive metrics including hierarchical Cell Ontology metrics
4. Provide a reusable evaluation pipeline independent of training code
5. Enable easy experimentation with different models and datasets

## Functional Requirements

### 1. Command-Line Interface

```bash
python scripts/evaluate_mlp.py \
  --checkpoint /path/to/checkpoint.pt \
  --config /path/to/config.json \
  --data-dir /path/to/data \
  --cell-counts /path/to/cell_counts.csv
```

**Required Arguments:**
- `--checkpoint`: Path to model checkpoint file (`.pt` format)
- `--config`: Path to config JSON file with model and data settings
- `--data-dir`: Base directory containing parquet files to evaluate on
- `--cell-counts`: Path to cell counts CSV file for cell type definitions

**Optional Arguments:**
- `--embedding-types`: Comma-separated list of embedding types (default: from config)
- `--genept-suffix`: Suffix for GenePT data directory (default: from config or `''`)
- `--tissue-suffix`: Suffix for tissue data directory (default: from config or `''`)
- `--metadata-suffix`: Suffix for metadata directory (default: from config or `''`)
- `--batch-size`: Inference batch size (default: 4096)
- `--device`: Device to run on (default: `cuda` if available, else `cpu`)
- `--output-dir`: Directory to save results (default: checkpoint directory)
- `--save-predictions`: Save prediction outputs (default: False)
- `--ontology-dir`: Cell Ontology cache directory (default: `data/ontology`)
- `--verbose`: Enable verbose output (default: False)

### 2. Core Functionality

#### 2.1 Configuration Loading
- Load config from JSON file (specified via `--config`)
- Extract model hyperparameters:
  - `n_hidden_layers`: Number of hidden layers
  - `dropout`: Dropout rate
  - `genept_dims`: GenePT embedding dimensions
  - `embedding_types`: List of embeddings to use
- Extract data config (with CLI overrides):
  - `cell_count_threshold`: Minimum samples per cell type
  - Embedding suffixes for data directories
- Support both standalone config files and checkpoint-associated configs

#### 2.2 Checkpoint Loading
- Load checkpoint from `.pt` file
- Extract `model_state_dict`
- Validate checkpoint format (warn if optimizer state missing)
- Compute input dimensions from config:
  - GenePT: `genept_dims` (e.g., 1536)
  - scGPT: 512 (fixed)
  - Metadata: Variable based on tissue embeddings
- Initialize `MLPClassifier` with config parameters
- Load weights from checkpoint

#### 2.3 Cell Type Loading
- Load cell types from CSV file (specified via `--cell-counts`)
- Expected CSV format: columns `cell_type`, `cell_count`
- Create sequential code mapping (0 to n-1)
- Apply filtering based on `cell_count_threshold`
- Create `code_remapping` dict: original_code → filtered_code or -100
- Build reverse mapping: cell_type → filtered_code

#### 2.4 Data Loading
- Use `ComposableTrainingDataset` in test mode
- Load parquet files from `--data-dir` with specified suffixes:
  - GenePT+scGPT embeddings: `{data_dir}/cellxgene_v2{genept_suffix}/`
  - Tissue embeddings: `{data_dir}/cellxgene_v2{tissue_suffix}/`
  - Metadata with labels: `{data_dir}/cellxgene_v2{metadata_suffix}/`
- Apply per-embedding scaling automatically (handled by ComposableDataset)
- Apply `code_remapping` to filter out low-count cell types
- Load all data into memory for evaluation
- Report dataset statistics (total samples, samples after filtering, class distribution)

#### 2.5 Model Inference
- Set model to eval mode (`model.eval()`)
- Move model to specified device
- Run batched inference with `torch.no_grad()`
- Use batch size from CLI arg (default: 4096)
- Collect predictions (logits and probabilities) and true labels

#### 2.6 Metrics Computation
- Compute all standard classification metrics:
  - `logloss`: Cross-entropy loss
  - `accuracy`: Overall accuracy
  - `macro_f1`, `macro_precision`, `macro_recall`: Macro-averaged metrics
  - `weighted_f1`, `weighted_precision`, `weighted_recall`: Weighted by class frequency
- Compute ranking metrics at k=2, 5, 10:
  - `recall_at_k`: Fraction where true class is in top-k predictions
  - `mrr_at_k`: Mean reciprocal rank (1/rank of true class, capped at k)
  - `dcg_at_k`: Discounted cumulative gain
- Compute hierarchical metrics using Cell Ontology:
  - `hierarchical_f1`, `hierarchical_precision`, `hierarchical_recall`
  - Load ontology from `--ontology-dir` (default: `data/ontology`)
  - Handle missing ontology gracefully (skip hierarchical metrics, warn user)

#### 2.7 Output
- Print comprehensive summary to console:
  ```
  Evaluation Results
  ==================
  Checkpoint: /path/to/checkpoint.pt
  Config:     /path/to/config.json
  Data:       /path/to/data

  Dataset Statistics:
  -------------------
  Total samples loaded:      150,000
  Samples after filtering:   125,000
  Number of classes:         234

  Model Configuration:
  --------------------
  Input dimensions:     2048 (GenePT: 1536, scGPT: 512)
  Hidden layers:        6
  Dropout:              0.120
  Output classes:       234

  Evaluation Metrics:
  -------------------
  logloss:              1.2345
  accuracy:             0.7234
  macro_f1:             0.8567
  macro_precision:      0.8612
  macro_recall:         0.8523
  weighted_f1:          0.8701

  Ranking Metrics:
  ----------------
  recall_at_2:          0.9234
  recall_at_5:          0.9678
  recall_at_10:         0.9812
  mrr_at_2:             0.8745
  mrr_at_5:             0.8801
  mrr_at_10:            0.8823
  dcg_at_2:             0.8899
  dcg_at_5:             0.9012
  dcg_at_10:            0.9089

  Hierarchical Metrics (Cell Ontology):
  --------------------------------------
  hierarchical_f1:          0.8901
  hierarchical_precision:   0.8945
  hierarchical_recall:      0.8858
  ```

- Save outputs to `--output-dir`:
  - `evaluation_results.json`: All metrics in structured JSON format
  - `predictions.npz`: Predictions (probs, logits) and labels (if `--save-predictions`)
  - `evaluation_summary.txt`: Console output captured to file
  - `class_distribution.csv`: Per-class sample counts and metrics

## Technical Design

### Code Reuse Strategy

The script should reuse existing infrastructure with minimal duplication:

**Import and reuse from `scripts/train_cellxgene_mlp.py`:**
- `load_cell_types_from_counts()` - Load cell types from counts file
- `create_code_remapping()` - Apply cell type filtering

**Import and reuse from `src/training/metrics.py`:**
- `evaluate_with_hierarchy()` - Full evaluation pipeline with all metrics
- `inference_all()` - Batched inference helper
- Individual metric functions as needed

**Import and reuse from `src/data_loading/composable_dataset.py`:**
- `ComposableTrainingDataset` - Load and combine embeddings from parquet

**Import and reuse from `src/models/mlp_classifier.py`:**
- `MLPClassifier` - Model architecture

**New code in evaluate_mlp.py:**
- Argument parsing and validation
- Config loading and merging with CLI args
- Main evaluation orchestration
- Output formatting and file saving

### Data Flow

```
1. Parse CLI arguments
   ↓
2. Load config JSON file
   ↓
3. Merge config with CLI overrides
   ↓
4. Load cell types from cell_counts CSV
   ↓
5. Create code_remapping (apply threshold filtering)
   ↓
6. Create ComposableTrainingDataset
   - Load parquet files from data_dir
   - Apply code_remapping for label filtering
   - Use per-embedding scaling
   ↓
7. Load all data into memory (X_eval, y_eval)
   ↓
8. Initialize MLPClassifier from config
   ↓
9. Load checkpoint weights
   ↓
10. Run batched inference (collect predictions)
   ↓
11. Compute all metrics (standard + hierarchical)
   ↓
12. Print results to console
   ↓
13. Save results to output_dir
```

### Error Handling

**File Validation:**
- Validate checkpoint file exists and is readable
- Validate config file exists and is valid JSON
- Validate cell_counts CSV file exists
- Validate data_dir exists and contains expected subdirectories
- Validate ontology_dir exists (or warn and skip hierarchical metrics)

**Data Validation:**
- Validate at least one sample remains after filtering
- Warn if many samples are filtered (> 50%)
- Validate label range matches model output dimensions
- Validate input feature dimensions match model expectations

**Model Validation:**
- Validate checkpoint contains model_state_dict
- Validate model architecture matches config
- Handle missing/extra keys in state_dict gracefully

**Graceful Degradation:**
- If ontology unavailable, skip hierarchical metrics and warn
- If GPU unavailable, fall back to CPU with warning
- If output_dir not writable, use current directory

## Implementation Notes

### Cell Type Filtering
- Filtering applies same logic as training via `code_remapping`
- Labels mapped to -100 are excluded from evaluation dataset
- Remaining labels are sequentially mapped (0 to n-1)
- Class count must match model output dimensions

### Data Scaling
- ComposableTrainingDataset handles per-embedding scaling automatically
- GenePT embeddings: divided by 0.021
- scGPT embeddings: divided by 0.044
- Metadata: divided by appropriate scale factor
- Scaling is transparent to evaluation script

### Data Format Requirements
- **Parquet files** in composable embedding format
- Directory structure:
  ```
  data_dir/
    cellxgene_v2{genept_suffix}/
      *.parquet  # GenePT + scGPT embeddings
    cellxgene_v2{tissue_suffix}/
      *.parquet  # Tissue embeddings
    cellxgene_v2{metadata_suffix}/
      *.parquet  # Metadata with cell_type labels
  ```
- ComposableDataset automatically discovers and loads all parquet files
- Suffixes allow using different datasets (train, test, validation)

### Config File Format
- JSON file with model and data configuration
- Can use checkpoint-associated config (e.g., `checkpoint_config.json`)
- Or create custom config file
- Required fields:
  ```json
  {
    "n_hidden_layers": 6,
    "dropout": 0.12,
    "genept_dims": 1536,
    "embedding_types": ["genept", "scgpt", "metadata"],
    "cell_count_threshold": 10000
  }
  ```
- Optional fields (can override via CLI):
  ```json
  {
    "genept_suffix": "_test_v1_scgpt",
    "tissue_suffix": "_test_v1_tissue",
    "metadata_suffix": "_test_v1"
  }
  ```

### Hierarchical Metrics
- Requires Cell Ontology graph files in ontology_dir
- If unavailable, gracefully skip hierarchical metrics
- Use same ontology version as training for valid comparisons

### Memory Considerations
- Evaluation dataset loaded fully into memory
- Typical test set size: ~120k samples, ~1-2 GB
- Inference runs in batches to avoid GPU OOM
- Predictions optionally saved to disk (can be large)

## Example Usage

### Example 1: Evaluate on Test Set
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750.pt \
  --config /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750_config.json \
  --data-dir /data/training_data \
  --cell-counts /data/batch-jobs/cell_counts.csv
```
This uses the suffixes from the config file (`_test_v1_scgpt`, `_test_v1_tissue`, `_test_v1`).

### Example 2: Evaluate with Custom Suffixes
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /path/to/model.pt \
  --config /path/to/config.json \
  --data-dir /data/validation_data \
  --cell-counts /data/cell_counts.csv \
  --genept-suffix "_val_v2" \
  --tissue-suffix "_val_v2" \
  --metadata-suffix "_val_v2"
```
This evaluates on a different dataset (e.g., validation set with different suffixes).

### Example 3: Evaluate with Custom Settings
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /path/to/model.pt \
  --config /path/to/config.json \
  --data-dir /path/to/data \
  --cell-counts /path/to/cell_counts.csv \
  --cell-count-threshold 5000 \
  --batch-size 2048 \
  --device cuda:1 \
  --output-dir /path/to/results \
  --save-predictions \
  --verbose
```
This overrides the cell count threshold, uses smaller batch size, specific GPU, saves predictions, and enables verbose logging.

### Example 4: Minimal Config File
If checkpoint doesn't have an associated config, create a minimal config JSON:
```json
{
  "n_hidden_layers": 6,
  "dropout": 0.12,
  "genept_dims": 1536,
  "embedding_types": ["genept", "scgpt", "metadata"],
  "cell_count_threshold": 10000,
  "genept_suffix": "_test_v1_scgpt",
  "tissue_suffix": "_test_v1_tissue",
  "metadata_suffix": "_test_v1"
}
```

Then run:
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /path/to/model.pt \
  --config /path/to/custom_config.json \
  --data-dir /data/training_data \
  --cell-counts /data/batch-jobs/cell_counts.csv
```

## Testing Strategy

### Unit Tests (`test/test_evaluate_mlp.py`)
- Test config loading and validation
- Test CLI argument parsing and merging with config
- Test cell type loading and remapping logic
- Test dimension calculation from config
- Test output file creation

### Integration Tests
- Test with small synthetic dataset (mock parquet files)
- Validate metrics computation produces expected ranges
- Validate output files contain correct structure
- Test error handling for missing files/directories

### Validation Test (Manual)
- Run on actual checkpoint: `checkpoint_epoch2_batch267_step1750.pt`
- Run on actual test data: `/data/training_data/cellxgene_v2_test_v1*`
- Manually compare output metrics with WandB step 1750 metrics
- Verify metrics are reasonable (within expected ranges)

## Dependencies

### Existing Dependencies
- `torch` - Model loading and inference
- `numpy`, `pandas` - Data manipulation
- `wandb` - Metrics fetching (optional)
- Existing project modules:
  - `src.models.mlp_classifier`
  - `src.training.metrics`
  - `src.data_loading.composable_dataset`

### No New Dependencies Required

## Future Enhancements

1. **Per-Class Metrics**: Export detailed per-cell-type performance (precision, recall, F1, sample count)
2. **Confusion Matrix**: Generate and save confusion matrices (full and top-N classes)
3. **Calibration Analysis**: Compute calibration metrics and reliability diagrams
4. **Multiple Checkpoints**: Evaluate multiple checkpoints in one run (compare across epochs)
5. **WandB Logging**: Add `--log-to-wandb` to create evaluation run in WandB
6. **Embedding-Only Evaluation**: Support evaluating on subsets of embeddings (e.g., GenePT only)
7. **CSV Export**: Export predictions in human-readable CSV format with cell type names

## Success Criteria

- ✅ Script is general-purpose and works with any checkpoint/config/data combination
- ✅ Reuses existing training infrastructure (no code duplication)
- ✅ Produces comprehensive metrics output (standard + hierarchical)
- ✅ Handles errors gracefully with informative messages
- ✅ Completes evaluation in < 5 minutes on GPU for ~120k test samples
- ✅ Output files are structured and readable (JSON, txt, npz)
- ✅ Can be used for test sets, validation sets, or custom datasets
