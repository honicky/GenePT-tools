# Model Retraining Status - Run nh5mzl7m

## Training Configuration

**Original Run:** https://wandb.ai/honicky/cellxgene-metadata-sweep-debug/runs/nh5mzl7m

**Retraining Config:** `/data/GenePT-tools/configs/retrain_nh5mzl7m.yaml`

**Key Changes from Original:**
- Epochs: 2 → 3
- Checkpointing: disabled → enabled every 250 batches
- Full evaluation: never → every 250 batches
- Checkpoint directory: `/data/GenePT-tools/checkpoints/nh5mzl7m_retrain/`

## Model Architecture

- Input dimension: 2048 (GenePT 1536 + scGPT 512)
- Hidden layers: 4 layers [1698, 1349, 1000, 651]
- Parameters: 7,979,104
- Dropout: 0.122
- Learning rate: 2.21e-05
- Weight decay: 3.37e-05
- Optimizer: Adam
- Batch size: 4096

## Data

- Training samples: 3,029,453 (80.4% of total)
- Cell types: 302 (filtered from 549 with threshold 10000)
- Validation samples: 108,855 (5k subset)
- Embeddings: GenePT (1536 dims) + scGPT (512 dims) + metadata (binary)

## Training Progress

**Status:** RUNNING

**Started:** 2025-11-11 04:36:22 UTC

**Log file:** `/tmp/retrain_nh5mzl7m.log`

**Wandb:** https://wandb.ai/honicky/cellxgene-metadata-retrain/runs/8jo1rck0

**Expected completion:** ~1.5-2.5 hours (3 epochs × 921 batches × 1-2 sec/batch)

**Current metrics:**
- Epoch 0, Batch 11/921
- Loss: 5.45 (decreasing from 5.78)

## Checkpointing

Checkpoints will be saved to:
```
/data/GenePT-tools/checkpoints/nh5mzl7m_retrain/
```

Every 250 batches:
- Batch 250 (Epoch 0)
- Batch 500 (Epoch 0)
- Batch 750 (Epoch 0)
- Batch 1000 (Epoch 1 + 79 batches)
- ... and so on

Best model tracked by validation logloss (minimize).

## Next Steps - Constrained Output Evaluation

Once training completes, we will evaluate the best checkpoint on brain tissue test data using three modes:

### Test Data
- File: `/mmc-scratch/scratch/cellxgene_v2_test_v1/dc30c3ec-46d6-4cd8-8ec1-b544a3d0f503.parquet`
- Tissue: Brain (UBERON:0000955)
- Cells: 6540
- Cell types: 11 (neuron, endothelial cell, astrocyte, etc.)

### Evaluation Modes

1. **Baseline (No Constraints)**
   - Standard softmax over all 302 cell types
   - No tissue-specific information used

2. **Allowlist (Hard Constraints)**
   - Only cell types allowed in brain tissue
   - Forbidden types set to -inf before softmax
   - Uses: `/data/GenePT-tools/data/cellxgene_constraints/tissue_allowlists.json`

3. **Soft Prior (Probabilistic Bias)**
   - Adds tissue-specific log prior to logits: `logits + alpha * log P(class | tissue)`
   - Alpha: 0.5 (recommended range: 0.25-1.0)
   - Uses: `/data/GenePT-tools/data/cellxgene_constraints/tissue_class_logprobs.pt`

### Evaluation Script

**Location:** `/data/GenePT-tools/scripts/eval_constrained_output.py`

**Command:**
```bash
cd /data/GenePT-tools && uv run python scripts/eval_constrained_output.py \
  --test-data /mmc-scratch/scratch/cellxgene_v2_test_v1/dc30c3ec-46d6-4cd8-8ec1-b544a3d0f503.parquet \
  --checkpoint /data/GenePT-tools/checkpoints/nh5mzl7m_retrain/best_model.pt \
  --class-labels /data/GenePT-tools/artifacts/cell_type_mapping_nh5mzl7m:v0/cell_type_mapping.csv \
  --constraints-dir /data/GenePT-tools/data/cellxgene_constraints \
  --alpha 0.5 \
  --device cuda \
  --batch-size 512
```

### Metrics

For each mode, we will compute:
- Accuracy
- Macro F1 (equal weight per class)
- Weighted F1 (weighted by support)
- Per-class precision, recall, F1
- Hierarchical metrics (if applicable)

### Expected Results

**Hypothesis:**
- Baseline: Good performance on common cell types
- Allowlist: May improve by preventing impossible predictions
- Soft prior: May balance between model confidence and tissue priors

We'll compare the three modes to see which provides the best performance for tissue-aware cell type classification.

## Monitoring Training

Check progress:
```bash
tail -f /tmp/retrain_nh5mzl7m.log
```

Check checkpoints:
```bash
ls -lh /data/GenePT-tools/checkpoints/nh5mzl7m_retrain/
```

View in Wandb:
```
https://wandb.ai/honicky/cellxgene-metadata-retrain/runs/8jo1rck0
```
