# CellXGene MLP Training Script Implementation Plan

## Overview
Step-by-step implementation checklist for converting the notebook to a production script using pre-shuffled S3 data.

## Phase 1: Core Data Infrastructure
### Week 1: Data Loading Foundation

- [ ] **1.1 Create basic project structure**
  - [ ] Create `src/data_loading/__init__.py`
  - [ ] Create `src/models/__init__.py`
  - [ ] Create `src/training/__init__.py`
  - [ ] Create `src/utils/__init__.py`
  - [ ] Create `scripts/` directory

- [ ] **1.2 Implement S3 data utilities** (`src/data_loading/utils.py`)
  - [ ] Function to list S3 files with boto3
  - [ ] Function to download single file from S3
  - [ ] Function to check if local file exists
  - [ ] Basic test: List files from `s3://pythiomicsdata/cellxgene_v2/training_v1_suffled/`

- [ ] **1.3 Implement basic dataset class** (`src/data_loading/s3_dataset.py`)
  - [ ] Create `S3ParquetStreamDataset` class skeleton
  - [ ] Implement `__init__` with configuration
  - [ ] Implement file listing logic
  - [ ] Implement local file checking
  - [ ] Test: Initialize dataset and list files

- [ ] **1.4 Add data loading logic**
  - [ ] Implement single file loading from S3 or local
  - [ ] Add parquet file reading
  - [ ] Extract embeddings and labels
  - [ ] Implement cell type encoding (matching notebook)
  - [ ] Test: Load and inspect one batch file

- [ ] **1.5 Implement iteration logic**
  - [ ] Add epoch-level file shuffling
  - [ ] Add intra-file sample shuffling
  - [ ] Create mini-batch generation
  - [ ] Implement `__iter__` method
  - [ ] Test: Iterate through subset of data (e.g., 3 files)

## Phase 2: Model and Training Components
### Week 2: Core Training Infrastructure

- [ ] **2.1 Port MLP architecture** (`src/models/mlp_classifier.py`)
  - [ ] Copy MLP creation logic from notebook
  - [ ] Implement linear interpolation for hidden dimensions
  - [ ] Add BatchNorm and Dropout layers
  - [ ] Test: Create model and verify architecture matches notebook

- [ ] **2.2 Port evaluation metrics** (`src/training/metrics.py`)
  - [ ] Copy metric functions from notebook (recall@k, mrr@k, dcg@k)
  - [ ] Add macro F1, precision, recall
  - [ ] Implement evaluation function
  - [ ] Test: Compute metrics on dummy data

- [ ] **2.3 Create configuration system** (`src/training/config.py`)
  - [ ] Create `TrainingConfig` dataclass
  - [ ] Add all hyperparameters with defaults from notebook
  - [ ] Add data paths configuration
  - [ ] Test: Create config and verify parameters

- [ ] **2.4 Implement checkpoint utilities** (`src/utils/checkpoint.py`)
  - [ ] Save checkpoint function (model, optimizer, epoch, batch)
  - [ ] Load checkpoint function
  - [ ] Best model tracking
  - [ ] Test: Save and load checkpoint

- [ ] **2.5 Create trainer class** (`src/training/trainer.py`)
  - [ ] Implement `MLPTrainer.__init__`
  - [ ] Add model creation method
  - [ ] Add optimizer creation (AdamW)
  - [ ] Add single batch training step
  - [ ] Test: Train on single batch

## Phase 3: Training Loop Implementation
### Week 3: Complete Training Pipeline

- [ ] **3.1 Implement full training epoch**
  - [ ] Add epoch training loop
  - [ ] Integrate with dataset iterator
  - [ ] Add loss tracking
  - [ ] Add progress bar with tqdm
  - [ ] Test: Train one epoch on subset

- [ ] **3.2 Add evaluation logic**
  - [ ] Load validation datasets (5k and 120k)
  - [ ] Implement periodic evaluation (every 10 batches for 5k)
  - [ ] Implement less frequent evaluation (every 250 batches for 120k)
  - [ ] Test: Verify metrics computation matches notebook

- [ ] **3.3 Integrate W&B logging**
  - [ ] Add W&B initialization
  - [ ] Log training loss
  - [ ] Log validation metrics
  - [ ] Log hyperparameters
  - [ ] Test: Verify W&B logging works

- [ ] **3.4 Add checkpointing to training**
  - [ ] Save checkpoint every 1000 batches
  - [ ] Save best model based on validation loss
  - [ ] Save final model at end of training
  - [ ] Implement resume from checkpoint
  - [ ] Test: Train, interrupt, resume

- [ ] **3.5 Complete trainer.run() method**
  - [ ] Full training loop for all epochs
  - [ ] Return final metrics
  - [ ] Handle keyboard interrupts gracefully
  - [ ] Test: Complete training run on subset

## Phase 4: CLI and Integration
### Week 4: Production Ready Script

- [ ] **4.1 Create CLI script** (`scripts/train_cellxgene_mlp.py`)
  - [ ] Add argparse for all parameters
  - [ ] Load configuration from CLI args
  - [ ] Initialize trainer
  - [ ] Run training
  - [ ] Test: Run with various CLI arguments

- [ ] **4.2 Add data validation**
  - [ ] Verify local data directory structure
  - [ ] Check S3 credentials if downloading
  - [ ] Validate cell type codes match
  - [ ] Test: Run with missing data, verify error handling

- [ ] **4.3 Hyperparameter optimization support**
  - [ ] Add optional Optuna trial parameter
  - [ ] Report metrics to trial
  - [ ] Create example optimization script
  - [ ] Test: Run simple hyperparameter search

- [ ] **4.4 Final validation**
  - [ ] Train with notebook's best hyperparameters
  - [ ] Verify metrics match notebook results
  - [ ] Ensure reproducibility with fixed seed
  - [ ] Document any differences from notebook

## Testing Checkpoints

### After Phase 1:
- Can load and iterate through pre-shuffled data
- Batches are created correctly
- File and sample shuffling work as expected

### After Phase 2:
- Model architecture matches notebook
- Metrics computation is correct
- Can save and load checkpoints

### After Phase 3:
- Full training loop works
- W&B logging functional
- Can resume from checkpoint
- Evaluation matches notebook

### After Phase 4:
- CLI interface works correctly
- **Final validation metrics match notebook:**
  - Recall@10: ~87%
  - MRR@10: ~56%
  - DCG@10: ~63.5%

## Test Commands

```bash
# Test data loading (Phase 1)
python -c "from src.data_loading.s3_dataset import S3ParquetStreamDataset; 
dataset = S3ParquetStreamDataset(local_data_dir='./test_cache', end_batch_file=3);
for i, (X, y) in enumerate(dataset): 
    print(f'Batch {i}: X={X.shape}, y={y.shape}'); 
    if i > 5: break"

# Test model creation (Phase 2)
python -c "from src.models.mlp_classifier import create_mlp;
model = create_mlp(500, 377, 3, 0.053);
print(model)"

# Test training on subset (Phase 3)
python scripts/train_cellxgene_mlp.py \
    --local-data-dir ./test_cache \
    --end-batch-file 10 \
    --epochs 1 \
    --checkpoint-dir ./test_checkpoints

# Full training run (Phase 4)
python scripts/train_cellxgene_mlp.py \
    --local-data-dir /data/cellxgene/training_shuffled \
    --test-data-dir /data/cellxgene/test \
    --n-dims 500 \
    --batch-size 1024 \
    --learning-rate 4.4e-5 \
    --dropout 0.053 \
    --n-hidden-layers 3 \
    --epochs 10 \
    --checkpoint-dir ./checkpoints \
    --wandb-project cellxgene-mlp-replication
```

## Success Metrics

1. **Data Loading**: Successfully iterate through all 376 batch files
2. **Model Match**: Architecture identical to notebook implementation
3. **Training Stability**: No loss spikes or distribution shifts
4. **Performance Match**: Achieve notebook's validation metrics
5. **Reproducibility**: Fixed seed produces consistent results

## Notes

- Start with minimal implementation, add optimizations later
- Test each component in isolation before integration
- Keep notebook open for reference during implementation
- Document any deviations from notebook approach
- Focus on correctness over performance initially