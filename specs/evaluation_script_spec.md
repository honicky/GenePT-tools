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
- `--per-file-metrics`: Compute and report per-file metrics (default: True)
- `--skip-per-file`: Skip per-file metrics computation for faster evaluation

**Constrained Output Arguments:**
- `--enable-constraints`: Enable tissue-based constrained output evaluation (default: False)
- `--constraint-mode`: Constraint mode(s) to evaluate: `allowlist`, `soft_prior`, or `both` (default: `both`)
- `--constraints-dir`: Directory containing constraint data files (default: `data/cellxgene_constraints`)
- `--allowlist-path`: Override path to `tissue_allowlists.json` (default: None, uses constraints-dir)
- `--counts-path`: Override path to `tissue_celltype_counts.json` for soft priors (default: None, uses constraints-dir)
- `--alpha`: Alpha value(s) for soft prior strength (default: 0.5, can specify multiple: `--alpha 0.3 0.5 0.7`)
- `--alpha-sweep`: Enable alpha sweep mode, evaluates [0.25, 0.5, 0.75, 1.0] (default: False)
- `--save-per-tissue-metrics`: Save detailed per-tissue metrics to CSV (default: False)

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
- **If `--enable-constraints` is set**: Extract `tissue_ontology_term_id` from metadata parquet files
  - Modify data loading to return tissue IDs alongside embeddings and labels
  - Track tissue information in `file_info` structure for per-file and per-tissue reporting
- Load all data into memory for evaluation
- Report dataset statistics (total samples, samples after filtering, class distribution, tissue distribution if constraints enabled)

#### 2.5 Constraint Loading (if enabled)
**Skip this section if `--enable-constraints` is False**

- Load constraint data files from `constraints_dir`:
  - **`tissue_allowlists.json`**: Maps tissue IDs to lists of allowed cell type class indices (in full 832-class vocabulary)
  - **`tissue_celltype_counts.json`**: Maps tissue IDs to cell type names and counts (used to compute soft priors dynamically)

- Build constraint infrastructure:
  ```python
  import json
  from src.inference.constraints import build_allowed_mask_from_set, counts_to_logp

  # Load allowlists
  with open(allowlist_path) as f:
      tissue_allowlists_full = json.load(f)  # Maps tissue_id -> [class indices in 832 vocab]

  # Load counts
  with open(counts_path) as f:
      counts_data = json.load(f)
      tissue_counts = counts_data['counts']  # Maps tissue_id -> {cell_type_name: count}

  # Filter allowlists to model's vocabulary using code_remapping
  tissue_allowlists_filtered = {}
  for tissue_id, full_indices in tissue_allowlists_full.items():
      # Filter to only include classes that map to valid model outputs (not -100)
      filtered_indices = [
          code_remapping[idx] for idx in full_indices
          if idx in code_remapping and code_remapping[idx] != -100
      ]
      tissue_allowlists_filtered[tissue_id] = filtered_indices

  # Compute soft priors from counts, filtered to model's vocabulary
  tissue_soft_priors = {}
  for tissue_id, cell_type_counts in tissue_counts.items():
      # Map cell type names to filtered class indices
      filtered_counts = {}
      for cell_type_name, count in cell_type_counts.items():
          if cell_type_name in cell_type_to_idx:
              filtered_idx = cell_type_to_idx[cell_type_name]
              filtered_counts[filtered_idx] = count

      # Convert counts to log probabilities
      if filtered_counts:
          logp = counts_to_logp(filtered_counts, num_classes)
          tissue_soft_priors[tissue_id] = logp.to(device)
  ```

- **Key principle**: Constraints are filtered dynamically based on the model's actual vocabulary (determined by `code_remapping` and `cell_type_to_idx`). This allows the same constraint files to work with any filtering threshold.

- Validate constraint coverage:
  - Check which evaluation tissues have constraint data available
  - Warn if some tissues lack constraints (will fall back to baseline for those tissues)
  - Report coverage statistics (e.g., "43/45 tissues covered")

#### 2.6 Model Inference
- Set model to eval mode (`model.eval()`)
- Move model to specified device
- Run batched inference with `torch.no_grad()`
- Use batch size from CLI arg (default: 4096)
- Collect predictions (logits and probabilities) and true labels
- **If constraints enabled**: Collect raw logits before softmax (needed for constraint application)

#### 2.7 Constrained Inference (if enabled)
**Skip this section if `--enable-constraints` is False**

For each constraint mode requested:

**Baseline Mode (always computed first):**
- Standard inference without constraints (as in section 2.6)
- Used as reference for delta metrics

**Allowlist Mode (if `constraint_mode` includes `allowlist` or `both`):**
- Group evaluation samples by tissue for homogeneous batch processing
  - Assumption: Each batch contains samples from a single tissue
  - Implementation: Either process files sequentially (files are typically tissue-homogeneous) or explicitly group by `tissue_ontology_term_id`
- For each tissue group:
  ```python
  # Get raw logits from model
  logits = model(X_tissue_batch)

  # Apply allowlist constraints (hard masking)
  constrained_logits = constraints.apply_allowlist(logits, tissue_id)
  # This sets logits to -1e9 for biologically impossible cell types

  # Get predictions from constrained logits
  preds = constrained_logits.argmax(dim=-1)
  ```
- Track constraint statistics:
  - Number of samples evaluated per tissue
  - Number of predictions changed vs baseline
  - List of cell types masked out per tissue

**Soft Prior Mode (if `constraint_mode` includes `soft_prior` or `both`):**
- For each alpha value specified (single value or sweep):
  - Group samples by tissue (same as allowlist mode)
  - For each tissue group:
    ```python
    # Get raw logits from model
    logits = model(X_tissue_batch)

    # Apply soft prior bias
    constrained_logits = constraints.apply_soft_prior(logits, tissue_id, alpha)
    # This adds: logits + alpha * log_prior_probs

    # Get predictions from biased logits
    preds = constrained_logits.argmax(dim=-1)
    ```
- If `--alpha-sweep` is enabled:
  - Evaluate with alpha values: [0.25, 0.5, 0.75, 1.0]
  - Store results for each alpha separately
  - Identify optimal alpha based on target metric (e.g., hierarchical F1)

**Tissue Grouping Strategy:**
- **Option A (Recommended)**: Process files one at a time
  - Each file typically represents one tissue/dataset (homogeneous)
  - Extract dominant tissue from file metadata
  - Apply constraints at file level
  - More efficient, leverages existing per-file infrastructure

- **Option B (Strict)**: Explicit tissue grouping
  - Group all samples by `tissue_ontology_term_id` after loading
  - Create tissue-specific batches for inference
  - Handles edge cases where files mix tissues
  - More memory overhead, slower

**Handling Missing Constraints:**
- If a tissue has no constraint data available:
  - Log warning: "Tissue {tissue_id} not found in constraints, using baseline predictions"
  - Fall back to unconstrained predictions for that tissue
  - Track coverage in constraint statistics

#### 2.8 Metrics Computation

**Overall Metrics (Baseline):**
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

**Per-File Metrics (if enabled):**
- Compute metrics separately for each parquet file in the test set
- Track file-level performance to identify:
  - Dataset quality issues (files performing significantly worse)
  - Distribution shift across data batches
  - Inconsistencies in test set composition
- Report per-file: sample count, logloss, accuracy, macro_f1, recall@2, recall@5, recall@10, hierarchical_f1 (if ontology available)
- **If constraints enabled**: Add per-file constraint metrics:
  - Tissue ID (extracted from file metadata)
  - Allowlist accuracy, Δaccuracy vs baseline
  - Soft prior accuracy (for each alpha), Δaccuracy vs baseline
  - Number of allowed cell types for this tissue
  - Percentage of predictions changed by constraints
- Compute summary statistics: mean, std, min, max across all files
- Can be disabled with `--skip-per-file` for faster evaluation on large datasets

**Constrained Metrics (if constraints enabled):**

For each constraint mode (allowlist, soft_prior):
- Compute all standard metrics on constrained predictions:
  - logloss, accuracy, macro_f1, hierarchical_f1, recall@k, etc.
- Compute **delta metrics** (improvement vs baseline):
  - `delta_accuracy = constrained_accuracy - baseline_accuracy`
  - `delta_macro_f1 = constrained_macro_f1 - baseline_macro_f1`
  - `delta_hierarchical_f1 = constrained_hierarchical_f1 - baseline_hierarchical_f1`
- Track **constraint impact statistics**:
  - Total samples evaluated with constraints
  - Samples per tissue (breakdown)
  - Predictions changed count and percentage
  - Tissues covered vs total tissues in dataset

**Per-Tissue Metrics (if constraints enabled and `--save-per-tissue-metrics`):**
- Aggregate results by tissue across all files
- For each tissue, report:
  ```python
  {
    'tissue_id': 'UBERON:0000178',           # UBERON ID
    'tissue_label': 'blood',                 # Human-readable name
    'samples': 15234,                        # Sample count
    'num_files': 3,                          # Files containing this tissue
    # Baseline metrics
    'baseline_accuracy': 0.721,
    'baseline_macro_f1': 0.856,
    'baseline_hierarchical_f1': 0.891,
    # Allowlist metrics
    'allowlist_accuracy': 0.734,
    'allowlist_delta_accuracy': 0.013,       # vs baseline
    'allowlist_macro_f1': 0.867,
    'allowlist_delta_macro_f1': 0.011,
    'num_allowed_classes': 89,               # Out of 302 total
    'allowlist_preds_changed_pct': 8.2,      # % predictions modified
    # Soft prior metrics (per alpha)
    'soft_prior_alpha_0.5_accuracy': 0.728,
    'soft_prior_alpha_0.5_delta_accuracy': 0.007,
    'soft_prior_alpha_0.5_macro_f1': 0.861,
    'soft_prior_preds_changed_pct': 12.5
  }
  ```

**Alpha Sweep Results (if `--alpha-sweep` enabled):**
- Tabular comparison across alpha values:
  ```python
  {
    'alpha_values': [0.25, 0.5, 0.75, 1.0],
    'accuracy': [0.723, 0.728, 0.731, 0.729],
    'macro_f1': [0.858, 0.861, 0.863, 0.860],
    'hierarchical_f1': [0.893, 0.896, 0.898, 0.895],
    'delta_accuracy': [0.002, 0.007, 0.010, 0.008],
    'optimal_alpha': 0.75  # Based on hierarchical_f1
  }
  ```

#### 2.9 Output
- Print comprehensive summary to console (example with constraints enabled):
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

  Per-File Metrics:
  -----------------
  File                                    Samples  Logloss  Accuracy  Macro F1  Hier. F1  Recall@2  Recall@5  Recall@10
  -----------------------------------------------------------------------------------------------------------------------
  batch_001.parquet                         8,432    1.201     0.735     0.862     0.891     0.925     0.968     0.982
  batch_002.parquet                         7,891    1.245     0.720     0.851     0.885     0.918     0.961     0.979
  batch_003.parquet                         8,127    1.189     0.741     0.869     0.897     0.931     0.972     0.985
  ...

  Summary Statistics Across Files:
  ---------------------------------
  Logloss:         mean=1.234 std=0.045 min=1.189 max=1.301
  Accuracy:        mean=0.728 std=0.018 min=0.695 max=0.752
  Macro F1:        mean=0.857 std=0.021 min=0.821 max=0.882
  Hierarchical F1: mean=0.890 std=0.008 min=0.878 max=0.904
  Recall@2:        mean=0.923 std=0.012 min=0.901 max=0.941
  Recall@10:       mean=0.981 std=0.008 min=0.968 max=0.991

  Constrained Evaluation Results:
  ================================

  Constraint Coverage:
  --------------------
  Tissues in dataset:       45
  Tissues with constraints: 43 (95.6%)
  Samples covered:          124,567 / 125,000 (99.7%)

  Allowlist Results:
  ------------------
  accuracy:                 0.734  (Δ +0.006)
  macro_f1:                 0.867  (Δ +0.010)
  hierarchical_f1:          0.898  (Δ +0.008)
  predictions_changed:      8,234  (6.6%)

  Soft Prior Results (alpha=0.5):
  --------------------------------
  accuracy:                 0.728  (Δ +0.000)
  macro_f1:                 0.861  (Δ +0.004)
  hierarchical_f1:          0.896  (Δ +0.006)
  predictions_changed:      15,789 (12.6%)

  Alpha Sweep (Soft Prior):
  --------------------------
  Alpha    Accuracy  Macro F1  Hier. F1  Δ Hier. F1
  -------  --------  --------  --------  ----------
  0.25     0.725     0.859     0.894     +0.004
  0.50     0.728     0.861     0.896     +0.006
  0.75     0.731     0.863     0.898     +0.008  ← Optimal
  1.00     0.729     0.860     0.895     +0.005

  Per-Tissue Metrics (top 5 by sample count):
  --------------------------------------------
  Tissue                    Samples  Baseline  Allowlist  Δ Acc  Soft(0.5)  Δ Acc  Classes
  ------------------------  -------  --------  ---------  -----  ---------  -----  -------
  blood                      21,450     0.721      0.734  +0.013     0.728  +0.007    89
  brain                      18,932     0.682      0.695  +0.013     0.688  +0.006   124
  lung                       12,845     0.745      0.758  +0.013     0.751  +0.006    76
  liver                       9,234     0.701      0.712  +0.011     0.706  +0.005    52
  heart                       8,123     0.734      0.746  +0.012     0.739  +0.005    68
  ```

- Save outputs to `--output-dir`:
  - `evaluation_results.json`: All metrics in structured JSON format, including:
    - Overall metrics (logloss, F1, recall@k, etc.)
    - `per_file_metrics`: List of per-file metric dicts (if per-file enabled)
    - `per_file_summary_stats`: Summary statistics across files (mean, std, min, max)
    - **If constraints enabled**:
      - `constrained_results`: Dict with keys `allowlist`, `soft_prior` containing constrained metrics
      - `constraint_statistics`: Coverage, predictions changed, etc.
      - `alpha_sweep_results`: Results for each alpha value (if sweep enabled)
  - `evaluation_summary.txt`: Complete console output including per-file and constrained metrics tables
  - `predictions.npz`: Predictions (probs, logits) and labels (if `--save-predictions`)
    - **If constraints enabled**: Also save `predictions_allowlist`, `predictions_soft_prior`
  - `class_distribution.csv`: Per-class sample counts
  - **If constraints enabled and `--save-per-tissue-metrics`**:
    - `per_tissue_metrics.csv`: Detailed tissue-level breakdown with baseline and constrained metrics
    - `alpha_sweep_results.csv`: Alpha sweep comparison table (if sweep enabled)

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

### Example 1: Basic Evaluation on Test Set
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750.pt \
  --config /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750_config.json \
  --data-dir /data/training_data \
  --cell-counts /data/batch-jobs/cell_counts.csv
```
This uses the suffixes from the config file (`_test_v1_scgpt`, `_test_v1_tissue`, `_test_v1`).

### Example 1b: Evaluate with Constrained Output
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750.pt \
  --config /data/GenePT-tools/checkpoints/vflydg4e_retrain/e66hluzl/checkpoint_epoch2_batch267_step1750_config.json \
  --data-dir /data/training_data \
  --cell-counts /data/batch-jobs/cell_counts.csv \
  --enable-constraints \
  --constraint-mode both \
  --alpha 0.5 \
  --save-per-tissue-metrics
```
This evaluates with both allowlist and soft prior constraints, using alpha=0.5 for the soft prior, and saves detailed per-tissue metrics to CSV.

### Example 1c: Evaluate with Alpha Sweep
```bash
python scripts/evaluate_mlp.py \
  --checkpoint /path/to/model.pt \
  --config /path/to/config.json \
  --data-dir /data/training_data \
  --cell-counts /data/batch-jobs/cell_counts.csv \
  --enable-constraints \
  --constraint-mode soft_prior \
  --alpha-sweep \
  --save-per-tissue-metrics
```
This evaluates soft prior constraints with alpha values [0.25, 0.5, 0.75, 1.0] to find the optimal setting.

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

## Use Cases for Per-File Metrics

Per-file metrics help identify several important issues:

1. **Dataset Quality Issues**: Files with significantly lower performance may indicate:
   - Data corruption or processing errors
   - Different preprocessing applied to some batches
   - Label noise concentrated in specific files

2. **Distribution Shift**: Performance variation across files may reveal:
   - Temporal drift (if files represent different collection times)
   - Batch effects (if files come from different experimental batches)
   - Dataset composition changes

3. **Test Set Validation**: Consistent per-file metrics indicate:
   - Proper shuffling and data splitting
   - Representative sampling across batches
   - No systematic biases in file creation

4. **Debugging**: When overall metrics are unexpected:
   - Identify which specific files are problematic
   - Focus investigation on outlier files
   - Validate file loading and preprocessing

## Constrained Output Implementation Notes

### Constraint Data Requirements

The constrained evaluation requires constraint data files:
- **Location**: `data/cellxgene_constraints/` (default, configurable via `--constraints-dir`)
- **Required files**:
  - **`tissue_allowlists.json`**: Maps UBERON tissue IDs to lists of allowed cell type class indices (in full CellXGene vocabulary, ~832 classes)
  - **`tissue_celltype_counts.json`**: Maps UBERON tissue IDs to cell type names and observed cell counts in CellXGene corpus
    - Used to dynamically compute soft priors (log probabilities) filtered to model's vocabulary
    - Structure: `{"tissue_id_to_label": {...}, "counts": {tissue_id: {cell_type_name: count}}}`
  - **`metadata.json`**: Human-readable tissue labels and metadata (optional, for display)

**Important**: These files contain data for the full CellXGene vocabulary (~832 cell types). The evaluation script **dynamically filters** the constraints to match the model's vocabulary (e.g., 302 classes after applying `cell_count_threshold`). This allows the same constraint files to work with models trained with different filtering thresholds.

### Class Vocabulary Alignment

**Challenge**: Constraint data was built with full CellXGene vocabulary (832 cell types), but models are trained with filtered vocabularies (e.g., 302 types after applying `cell_count_threshold=10000`).

**Solution**: Dynamic filtering in evaluation script using JSON constraint files:

1. **Allowlist filtering** (`tissue_allowlists.json` contains indices 0-831):
   ```python
   # Load full allowlists (indices in 832-class vocabulary)
   tissue_allowlists_full = load_json("tissue_allowlists.json")

   # Filter to model's vocabulary using code_remapping
   tissue_allowlists_filtered = {}
   for tissue_id, full_indices in tissue_allowlists_full.items():
       filtered_indices = [
           code_remapping[idx] for idx in full_indices
           if idx in code_remapping and code_remapping[idx] != -100
       ]
       tissue_allowlists_filtered[tissue_id] = filtered_indices
   ```

2. **Soft prior computation** (`tissue_celltype_counts.json` has cell type names as keys):
   ```python
   # Load counts (keyed by cell type NAME, not index)
   tissue_counts = load_json("tissue_celltype_counts.json")['counts']

   # Map cell type names to filtered model indices
   tissue_soft_priors = {}
   for tissue_id, name_to_count in tissue_counts.items():
       filtered_counts = {}
       for cell_type_name, count in name_to_count.items():
           if cell_type_name in cell_type_to_idx:  # Model's vocabulary
               filtered_idx = cell_type_to_idx[cell_type_name]
               filtered_counts[filtered_idx] = count

       # Convert counts to log probabilities for model's 302 classes
       logp = counts_to_logp(filtered_counts, num_classes=302)
       tissue_soft_priors[tissue_id] = logp
   ```

**Key advantages of JSON-based approach**:
- Same constraint files work with any model vocabulary (determined by `cell_count_threshold`)
- No need to rebuild constraint files when changing filtering
- Cell type names in counts file provide human-readable mapping
- Soft priors computed on-the-fly from counts, ensuring correct normalization for filtered vocabulary

### Tissue Extraction from Metadata

Per-file and per-tissue metrics require extracting tissue IDs from parquet files:

**Implementation**:
1. Modify `ComposableTrainingDataset._load_parquet_file()` to return tissue IDs
2. Load metadata parquet: `pd.read_parquet(metadata_file, columns=['tissue_ontology_term_id'])`
3. Store tissue IDs in `file_info` dict: `{'filename': ..., 'tissue_ids': np.array([...])}`
4. Determine dominant tissue per file (most common tissue_id) for file-level constraints
5. For strict per-sample constraints, track tissue for every sample

**Edge Cases**:
- Files with missing `tissue_ontology_term_id`: Skip constraints, warn user
- Files with multiple tissues: Either pick dominant or split file into tissue groups
- Tissues not in constraint data: Fall back to baseline, track coverage

## Future Enhancements

1. **Per-Class Metrics**: Export detailed per-cell-type performance (precision, recall, F1, sample count)
2. **Confusion Matrix**: Generate and save confusion matrices (full and top-N classes)
3. **Confusion Matrix with Constraints**: Compare confusion matrices baseline vs constrained
4. **Calibration Analysis**: Compute calibration metrics and reliability diagrams
5. **Multiple Checkpoints**: Evaluate multiple checkpoints in one run (compare across epochs)
6. **WandB Logging**: Add `--log-to-wandb` to create evaluation run in WandB with constrained metrics
7. **Embedding-Only Evaluation**: Support evaluating on subsets of embeddings (e.g., GenePT only)
8. **CSV Export**: Export predictions in human-readable CSV format with cell type names
9. **Statistical Significance**: Add paired t-tests for constraint improvements
10. **Constraint Visualization**: Generate plots showing per-tissue constraint impact

## Success Criteria

**Baseline Evaluation (Already Implemented):**
- ✅ Script is general-purpose and works with any checkpoint/config/data combination
- ✅ Reuses existing training infrastructure (no code duplication)
- ✅ Produces comprehensive metrics output (standard + hierarchical)
- ✅ Handles errors gracefully with informative messages
- ✅ Completes evaluation in < 5 minutes on GPU for ~120k test samples
- ✅ Output files are structured and readable (JSON, txt, npz)
- ✅ Can be used for test sets, validation sets, or custom datasets

**Constrained Evaluation (To Be Implemented):**
- ⬜ Constraints are opt-in via `--enable-constraints` flag (backward compatible)
- ⬜ Supports both allowlist (hard) and soft prior (probabilistic) constraint modes
- ⬜ Dynamically filters constraint vocabulary to match model's filtered classes
- ⬜ Extracts tissue IDs from metadata parquet files
- ⬜ Groups samples by tissue for homogeneous batch processing
- ⬜ Computes delta metrics (constrained vs baseline) for easy comparison
- ⬜ Reports per-tissue metrics to identify which tissues benefit most
- ⬜ Handles missing constraint data gracefully (falls back to baseline)
- ⬜ Alpha sweep mode identifies optimal soft prior strength
- ⬜ Adds < 30% overhead to evaluation time (efficient tissue grouping)
- ⬜ Saves constrained predictions alongside baseline predictions
- ⬜ Outputs structured JSON with constrained results embedded
