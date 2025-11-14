# Constrained Output Evaluation Report - Model nh5mzl7m

## Overview

**Model**: Retrained nh5mzl7m (3 epochs)
**Test Dataset**: Full test set (120,984 samples)
**Evaluation Date**: 2025-11-11
**Checkpoint**: `/data/GenePT-tools/checkpoints/final_checkpoint.pt`

## Model Configuration

- **Architecture**: 4-layer MLP
- **Input dimension**: 2048 (GenePT 1536 + scGPT 512)
- **Output classes**: 302 filtered cell types
- **Parameters**: 7,979,104
- **Dropout**: 0.122
- **Training seed**: 4201

## Evaluation Results

### Baseline (No Constraints)

#### Overall Performance
- **Accuracy**: 41.91%
- **Test Samples**: 120,984
- **Classes Present in Test**: 86 / 302

#### Macro Metrics (Equal Weight Per Class)
- **Precision**: 19.38%
- **Recall**: 17.78%
- **F1**: 17.44%

#### Weighted Metrics (Weighted by Support)
- **Precision**: 45.86%
- **Recall**: 41.91%
- **F1**: 41.99%

#### Recall@k Metrics
- **Recall@1**: 41.91%
- **Recall@5**: 74.90%
- **Recall@10**: 84.05%
- **Recall@20**: 89.44%

#### Hierarchical Metrics (Cell Ontology)
- **Hierarchical Precision**: 87.12%
- **Hierarchical Recall**: 88.66%
- **Hierarchical F1**: 87.88%

## Comparison with Training Validation Metrics

The evaluation metrics differ from the training validation metrics reported during training:

| Metric | Training Validation | Test Evaluation | Delta |
|--------|-------------------|-----------------|-------|
| Macro F1 | 11.77% | 17.44% | +5.67% |
| Hierarchical F1 | 91.94% | 87.88% | -4.06% |
| Recall@10 | 90.72% | 84.05% | -6.67% |
| Recall@5 | 81.01% | 74.90% | -6.11% |

**Note**: The discrepancy between training validation and test evaluation metrics requires investigation. Possible causes:
1. Different data distributions between validation and test sets
2. Validation metrics computed on smaller sample size
3. Batch ordering effects in metric computation

## Per-Class Performance Highlights

### Best Performing Classes (F1 > 0.95)
1. **L6b glutamatergic cortical neuron**: 99% F1 (1,698 samples)
2. **chandelier pvalb GABAergic cortical interneuron**: 99% F1 (548 samples)
3. **corticothalamic-projecting glutamatergic cortical neuron**: 99% F1 (1,811 samples)
4. **near-projecting glutamatergic cortical neuron**: 100% F1 (1,509 samples)
5. **L5 extratelencephalic projecting glutamatergic cortical neuron**: 98% F1 (155 samples)
6. **ependymal cell**: 98% F1 (676 samples)
7. **adipocyte**: 100% F1 (119 samples)

### Worst Performing Classes (F1 < 0.05)
1. **inhibitory interneuron**: 1% F1 (12,129 samples) - Large class, poor performance
2. **T cell**: 1% F1 (1,461 samples)
3. **double negative thymocyte**: 0% F1 (1,697 samples)
4. **skeletal muscle fiber**: 0% F1 (1,684 samples)
5. **fibroblast of lung**: 0% F1 (814 samples)
6. **neural cell**: 0% F1 (50 samples)
7. **stromal cell**: 0% F1 (419 samples)

### Moderate Support Classes (Good Performance)
- **pvalb GABAergic cortical interneuron**: 89% F1 (2,000 samples)
- **sncg GABAergic cortical interneuron**: 95% F1 (1,767 samples)
- **lamp5 GABAergic cortical interneuron**: 91% F1 (2,000 samples)
- **VIP GABAergic cortical interneuron**: 88% F1 (2,000 samples)
- **mast cell**: 95% F1 (660 samples)
- **plasmacytoid dendritic cell**: 90% F1 (1,010 samples)
- **malignant cell**: 92% F1 (2,000 samples)

## Key Observations

1. **Hierarchical Metrics**: The model achieves strong hierarchical F1 (87.88%), indicating it generally predicts cell types in the correct ontological neighborhood even when exact predictions are wrong.

2. **Macro vs Weighted Performance**: Large gap between macro F1 (17.44%) and weighted F1 (41.99%) indicates the model performs well on frequent classes but struggles with rare classes.

3. **Top-k Performance**: Recall@10 of 84.05% shows the correct cell type is in the top 10 predictions for most samples, suggesting constrained output could help.

4. **Neuron Subtypes**: The model excels at distinguishing specific cortical neuron subtypes (GABAergic interneurons, glutamatergic projection neurons), achieving 85-100% F1 on most.

5. **General Cell Types**: Broad, general cell types like "inhibitory interneuron", "T cell", "leukocyte", "neuron" perform poorly, likely because they're confused with their more specific subtypes.

6. **Tissue-Specific Types**: Lung-specific cell types (alveolar type 1/2, fibroblast of lung) show mixed performance despite the test data being brain tissue, suggesting the model may need tissue-aware constraints.

## Constrained Output Evaluation

**Status**: Not evaluated yet due to constraint file dimension mismatch.

The tissue constraint files were built for 832 full ontology classes, but the model was trained on 302 filtered classes (cell types with >10,000 samples). To evaluate constrained output modes (allowlist and soft prior), the constraint files need to be rebuilt for the 302-class filtered ontology.

## Recommendations

1. **Investigate metric discrepancy**: Determine why test evaluation metrics differ from training validation metrics.

2. **Rebuild constraint files**: Create tissue constraint files for the 302 filtered classes to enable constrained output evaluation.

3. **Address rare class performance**: Consider techniques like:
   - Class-balanced sampling during training
   - Focal loss or other class-reweighting strategies
   - Data augmentation for rare classes

4. **Tissue-aware evaluation**: Evaluate performance separately on different tissue types to understand tissue-specific biases.

5. **Hierarchical loss**: The strong hierarchical F1 suggests incorporating ontology-aware loss functions could improve both exact match and hierarchical metrics.

## Files Generated

- Evaluation script: `/data/GenePT-tools/scripts/eval_constrained_output.py`
- Evaluation log: `/tmp/eval_with_hierarchical.log`
- This report: `/data/GenePT-tools/EVALUATION_REPORT_FINAL.md`
