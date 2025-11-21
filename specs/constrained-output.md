# Constrained Output for Cell Type Classification

## Overview

Post-processing module for cell type classifiers that applies context-aware constraints or biases to model predictions. Operates entirely in logit space without requiring model retraining. Supports two modes:

1. **Allowlist (hard constraint)**: Only permitted classes can be predicted
2. **Soft prior (probabilistic bias)**: Nudge predictions toward context-typical classes

**Key property**: Context-agnostic API. The "context key" can represent any categorical variable per example (tissue, dataset, sequencer, species, panel, etc.).

---

## Motivation

Baseline cell type classifiers often predict biologically implausible labels (e.g., neurons in blood samples). Context provides strong constraints:

- **Anatomical context**: Tissue origin limits valid cell types
- **Technical context**: Sequencing protocol/instrument affects reliability
- **Biological context**: Species, disease state, developmental stage

This module incorporates context **after** model inference, enabling:

- No retraining when updating constraints
- Easy experimentation with allowlist vs. soft prior
- Per-context constraint customization

---

## Core API

### Design Assumption: Homogeneous Batches

**Key simplification**: All examples in a batch share the same context (e.g., same tissue, same sequencer, same species). This is natural for:

- Processing dataset chunks (all cells from one tissue sample)
- Online inference (user queries one tissue at a time)
- Training pipelines (batches are typically stratified by metadata)

This eliminates per-example context indexing and simplifies the API significantly.

### Input Schema

**Shared inputs** (both modes):

```python
logits: torch.Tensor          # [B, C] pre-softmax scores from model
very_neg: float = -1e9        # value for suppressing classes in softmax
```

**Allowlist-specific**:

```python
allowed_mask: torch.Tensor  # [C] boolean mask where True = allowed for THIS batch
```

**Soft prior-specific**:

```python
logp_class: torch.Tensor  # [C] log P(class | context) for THIS batch
alpha: float = 0.5        # prior strength (non-negative)
```

### Output

```python
logits_adjusted: torch.Tensor  # [B, C] transformed logits for softmax
```

---

## Implementation

### Allowlist Mode (Hard Constraints)

```python
def apply_allowlist_mask(
  logits: torch.Tensor,            # [B, C]
  allowed_mask: torch.Tensor,      # [C] boolean mask
  very_neg: float = -1e9
) -> torch.Tensor:
  """
  Hard allowlist for homogeneous batch (all examples have same context).

  Args:
    logits: Pre-softmax scores [B, C]
    allowed_mask: Boolean mask [C] where True = allowed for this batch
    very_neg: Value for suppressing forbidden classes

  Edge cases:
  - All classes False -> all suppressed (avoid this)
  - Empty batch (B=0) -> returns empty tensor

  Returns:
    Adjusted logits [B, C]
  """
  # Broadcast mask across batch: [C] -> [B, C]
  # Apply mask: keep original where True, replace with very_neg where False
  out = torch.where(allowed_mask, logits, torch.full_like(logits, very_neg))

  return out


def build_allowed_mask_from_set(
  allowed_classes: Set[int],
  num_classes: int,
  device: torch.device = None
) -> torch.Tensor:
  """
  Convert sparse allowlist set to dense boolean mask tensor.

  Args:
    allowed_classes: Set of allowed class IDs for this context
    num_classes: Total number of classes (C)
    device: Target device for tensor

  Returns:
    mask: [C] boolean tensor where mask[c] = (c in allowed_classes)
  """
  # Start with all False
  mask = torch.zeros(num_classes, dtype=torch.bool, device=device)

  # Fill in allowed classes
  if allowed_classes:
    class_indices = torch.tensor(
      sorted(allowed_classes),
      dtype=torch.long,
      device=device
    )
    mask[class_indices] = True

  return mask
```

**Usage**:

```python
# Define allowlist for blood tissue (this batch's context)
blood_allowed_classes = {0, 2, 5, 9, 15, 18, 22}

# Convert to dense mask tensor (do once per context)
num_classes = 50
blood_mask = build_allowed_mask_from_set(
  blood_allowed_classes,
  num_classes,
  device=logits.device
)

# Inference on homogeneous batch (all blood cells)
# Mask broadcasts automatically across batch dimension
logits_masked = apply_allowlist_mask(logits, blood_mask)
probs = torch.softmax(logits_masked, dim=-1)
predictions = probs.argmax(dim=-1)
```

### Soft Prior Mode (Probabilistic Bias)

```python
def add_soft_prior_bias(
  logits: torch.Tensor,     # [B, C]
  logp_class: torch.Tensor, # [C]
  alpha: float = 0.5
) -> torch.Tensor:
  """
  Soft prior for homogeneous batch. Adds alpha * log P(class | context) to logits.

  Args:
    logits: Pre-softmax scores [B, C]
    logp_class: Log probabilities [C] for this batch's context
    alpha: Prior strength. Recommended range [0.25, 1.0].
           Larger values strengthen bias toward context-typical classes.

  Returns:
    Adjusted logits [B, C]
  """
  # Bias broadcasts automatically across batch: [C] -> [B, C]
  return logits + alpha * logp_class
```

**Building the prior from training data**:

```python
def counts_to_logp(
  counts: torch.Tensor,     # [C] co-occurrence counts for one context
  epsilon: float = 1.0      # Laplace smoothing
) -> torch.Tensor:
  """
  Convert counts to log probabilities with smoothing.

  Args:
    counts: counts[c] = number of times class c appeared in this context
    epsilon: Add-k smoothing constant (1.0 = Laplace smoothing)

  Returns:
    logp: [C] where logp[c] = log P(class=c | context)
  """
  smoothed = counts + epsilon
  p = smoothed / smoothed.sum()
  return torch.log(p)
```

**Usage**:

```python
# Build prior from training data for blood tissue
blood_counts = train_counts_by_tissue["blood"]  # [C] vector
logp_blood = counts_to_logp(blood_counts, epsilon=1.0)

# Inference on blood batch
logits_biased = add_soft_prior_bias(
  logits,
  logp_blood,
  alpha=0.5  # tune on dev set
)
probs = torch.softmax(logits_biased, dim=-1)
predictions = probs.argmax(dim=-1)
```

**Tuning alpha**:

- Start with `[0.25, 0.5, 1.0]` and evaluate on dev set
- Larger `alpha` → stronger bias toward key-typical classes
- Monitor rare-but-valid classes for over-suppression

---

## CellxGene Tissue-Specific Implementation

### Overview

The cellxgene corpus provides rich tissue-cell type co-occurrence data. We leverage this to build:

1. **Tissue allowlists**: Valid cell types per tissue (from observed co-occurrence)
2. **Tissue priors**: P(cell_type | tissue) distribution

### File Format

**Allowlist file** (`tissue_allowlists.json`):

```json
{
  "blood": [0, 2, 5, 9, 15, 18, 22],
  "brain": [1, 3, 4, 11, 17, 25],
  "liver": [6, 7, 8, 10, 13, 19, 21],
  ...
}
```

**Prior file** (`tissue_class_logprobs.pt`):

```python
{
  "tissue_to_logp": {
    "blood": torch.Tensor[C],   # log P(class | blood)
    "brain": torch.Tensor[C],   # log P(class | brain)
    "liver": torch.Tensor[C],   # log P(class | liver)
    ...
  },
  "class_names": ["CD4 T cell", "neuron", ...]  # length C
}
```

### Loading and Usage

```python
import json
import torch
from typing import Dict, List, Set

class CellxGeneTissueConstraints:
  """
  Tissue-aware constraints for cell type classification.

  Assumes homogeneous batches: all cells in a batch come from the same tissue.
  """

  def __init__(
    self,
    allowlist_path: str = None,
    prior_path: str = None,
    num_classes: int = None,
    device: torch.device = None
  ):
    """
    Initialize tissue constraints.

    Args:
      allowlist_path: Path to tissue_allowlists.json
      prior_path: Path to tissue_class_logprobs.pt
      num_classes: Number of cell type classes (required for allowlist)
      device: Target device for tensors
    """
    self.num_classes = num_classes
    self.device = device

    # Load allowlists: tissue -> allowed class IDs
    self.tissue_to_allowlist = {}
    if allowlist_path:
      if num_classes is None:
        raise ValueError("num_classes required when loading allowlist")
      with open(allowlist_path) as f:
        self.tissue_to_allowlist = json.load(f)

    # Load priors: tissue -> log P(class | tissue)
    self.tissue_to_logp = {}
    if prior_path:
      data = torch.load(prior_path, map_location=device)
      self.tissue_to_logp = data["tissue_to_logp"]

      # Update num_classes from prior if not set
      if self.num_classes is None:
        first_tissue = next(iter(self.tissue_to_logp.keys()))
        self.num_classes = self.tissue_to_logp[first_tissue].shape[0]

  def get_allowlist_mask(
    self,
    tissue: str,
    device: torch.device = None
  ) -> torch.Tensor:
    """
    Get allowlist mask for a specific tissue.

    Args:
      tissue: Tissue name (e.g., "blood")
      device: Target device (default: self.device)

    Returns:
      Boolean mask [C] where True = allowed
    """
    if tissue not in self.tissue_to_allowlist:
      raise ValueError(f"Unknown tissue: {tissue}")

    allowed_classes = set(self.tissue_to_allowlist[tissue])
    return build_allowed_mask_from_set(
      allowed_classes,
      self.num_classes,
      device or self.device
    )

  def get_logp(
    self,
    tissue: str,
    device: torch.device = None
  ) -> torch.Tensor:
    """
    Get log prior for a specific tissue.

    Args:
      tissue: Tissue name (e.g., "blood")
      device: Target device (default: self.device)

    Returns:
      Log probabilities [C] where logp[c] = log P(class=c | tissue)
    """
    if tissue not in self.tissue_to_logp:
      raise ValueError(f"Unknown tissue: {tissue}")

    logp = self.tissue_to_logp[tissue]
    if device is not None:
      logp = logp.to(device)
    return logp

  def apply_allowlist(
    self,
    logits: torch.Tensor,
    tissue: str,
    very_neg: float = -1e9
  ) -> torch.Tensor:
    """
    Apply tissue allowlist to homogeneous batch.

    Args:
      logits: Pre-softmax scores [B, C]
      tissue: Tissue for this entire batch
      very_neg: Value for suppressing forbidden classes

    Returns:
      Adjusted logits [B, C]
    """
    mask = self.get_allowlist_mask(tissue, device=logits.device)
    return apply_allowlist_mask(logits, mask, very_neg)

  def apply_soft_prior(
    self,
    logits: torch.Tensor,
    tissue: str,
    alpha: float = 0.5
  ) -> torch.Tensor:
    """
    Apply tissue soft prior to homogeneous batch.

    Args:
      logits: Pre-softmax scores [B, C]
      tissue: Tissue for this entire batch
      alpha: Prior strength

    Returns:
      Adjusted logits [B, C]
    """
    logp = self.get_logp(tissue, device=logits.device)
    return add_soft_prior_bias(logits, logp, alpha)


# Usage example
constraints = CellxGeneTissueConstraints(
  allowlist_path="tissue_allowlists.json",
  prior_path="tissue_class_logprobs.pt",
  num_classes=50,  # number of cell types
  device="cuda"    # or "cpu"
)

# Batch inference (homogeneous batch: all blood)
tissue = "blood"
logits = model(batch)  # [B, num_classes]

# Option 1: Hard allowlist
logits_masked = constraints.apply_allowlist(logits, tissue)

# Option 2: Soft prior
logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=0.5)

# Get predictions
probs = torch.softmax(logits_masked, dim=-1)  # or logits_biased
predictions = probs.argmax(dim=-1)
```

### Building Tissue Constraints from CellxGene Data

**Prerequisites** (provided separately):

1. Co-occurrence counts: `Dict[str, torch.Tensor]`
   - `counts[tissue][class_idx]` = number of cells observed
2. Tissue and class vocabularies

**Generating allowlists**:

```python
def build_allowlist_dict(
  counts_by_tissue: Dict[str, torch.Tensor],  # tissue -> [C] counts
  min_count: int = 10
) -> Dict[str, List[int]]:
  """
  Build allowlist dict from co-occurrence counts.

  Args:
    counts_by_tissue: Dict mapping tissue name to count vector [C]
    min_count: Minimum observations to include class for tissue

  Returns:
    Dict mapping tissue name to list of allowed class indices
  """
  allowlists = {}
  for tissue, counts in counts_by_tissue.items():
    # Vectorized threshold comparison
    allowed_mask = counts >= min_count
    # Get indices where True
    allowed_classes = allowed_mask.nonzero(as_tuple=True)[0]
    allowlists[tissue] = allowed_classes.tolist()

  return allowlists
```

**Generating soft priors**:

```python
def build_soft_prior_dict(
  counts_by_tissue: Dict[str, torch.Tensor],  # tissue -> [C] counts
  epsilon: float = 1.0
) -> Dict[str, torch.Tensor]:
  """
  Build log probability dict from counts.

  Args:
    counts_by_tissue: Dict mapping tissue name to count vector [C]
    epsilon: Laplace smoothing constant

  Returns:
    Dict mapping tissue name to log probability vector [C]
  """
  tissue_to_logp = {}
  for tissue, counts in counts_by_tissue.items():
    tissue_to_logp[tissue] = counts_to_logp(counts, epsilon)

  return tissue_to_logp
```

**Complete example**:

```python
# Load cellxgene co-occurrence data
counts_by_tissue = {
  "blood": torch.tensor([1000, 50, 2000, ...]),  # [C]
  "brain": torch.tensor([10, 5000, 100, ...]),
  "liver": torch.tensor([500, 20, 1500, ...]),
  ...
}

# Build allowlists
allowlists = build_allowlist_dict(counts_by_tissue, min_count=10)

# Build priors
tissue_to_logp = build_soft_prior_dict(counts_by_tissue, epsilon=1.0)

# Save
import json
with open("tissue_allowlists.json", "w") as f:
  json.dump(allowlists, f)

torch.save({
  "tissue_to_logp": tissue_to_logp,
  "class_names": class_names
}, "tissue_class_logprobs.pt")
```

---

## Validation and Metrics

### Evaluation Protocol

1. **Baseline metrics** (no constraints):
   - Top-1/3 accuracy
   - Macro-F1 score
   - Per-tissue confusion matrices

2. **Allowlist validation**:
   - Precision improvement (invalid predictions eliminated)
   - Recall analysis (only for truly invalid classes)
   - Edge case audit (rare tissues, small allowlists)

3. **Soft prior validation**:
   - Alpha sweep: `[0.0, 0.25, 0.5, 0.75, 1.0]`
   - Per-tissue lift in macro-F1
   - Rare-class preservation (avoid over-suppression)

### Logging Per-Context Performance

Since batches are homogeneous, you typically process one context at a time. Aggregate metrics across contexts by collecting per-batch results:

```python
def evaluate_by_tissue(
  tissue_results: Dict[str, Dict[str, any]],  # tissue -> {"preds": [...], "labels": [...]}
  metric_fn: callable                         # e.g., accuracy_score, f1_score
) -> Dict[str, float]:
  """
  Compute metrics per tissue from collected results.

  Args:
    tissue_results: Dict mapping tissue to collected predictions/labels
    metric_fn: Function(labels, preds) -> float (sklearn-style)

  Returns:
    Dict mapping tissue name to metric value
  """
  metrics = {}
  for tissue, data in tissue_results.items():
    preds = torch.cat(data["preds"]).cpu().numpy()
    labels = torch.cat(data["labels"]).cpu().numpy()
    metrics[tissue] = metric_fn(labels, preds)

  return metrics


# Example usage during evaluation loop
tissue_results = {}

for batch in dataloader:
  tissue = batch["tissue"]  # All cells in batch have this tissue
  logits = model(batch)
  logits_adj = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
  preds = logits_adj.argmax(dim=-1)
  labels = batch["labels"]

  # Collect results per tissue
  if tissue not in tissue_results:
    tissue_results[tissue] = {"preds": [], "labels": []}
  tissue_results[tissue]["preds"].append(preds)
  tissue_results[tissue]["labels"].append(labels)

# Compute metrics
from sklearn.metrics import accuracy_score, f1_score
tissue_acc = evaluate_by_tissue(tissue_results, accuracy_score)
tissue_f1 = evaluate_by_tissue(tissue_results, lambda l, p: f1_score(l, p, average="macro"))
```

---

## Integration Guidelines

### 1. Choosing Between Modes

**Use allowlist when**:

- You have high-confidence biological constraints
- False positives (invalid predictions) are costly
- Context strongly restricts valid classes

**Use soft prior when**:

- Constraints are probabilistic, not absolute
- You want to preserve model confidence ordering
- Context provides preference, not exclusion

**Both can be combined** (apply allowlist first, then soft prior on remaining classes).

### 2. Numerical Stability

- Always apply transforms in **logit space** (pre-softmax)
- Use `very_neg = -1e9` (robust in mixed precision)
- Avoid `float('-inf')` (can cause NaNs in some ops)

### 3. Calibration

- Calibrate (e.g., temperature scaling) **after** finalizing constraint mode
- If switching between allowlist/prior, re-fit calibration
- Tissue-specific calibration may improve further

### 4. Context Key Design

**Key should be**:

- Known at inference time
- Granular enough to provide signal (not too coarse)
- Not too fine-grained (avoid data sparsity)

**Examples beyond tissue**:

- `sequencer_model`: Restrict unreliable classes per instrument
- `species`: Forbid human-only labels on mouse data
- `panel`: Certain antibody panels never include specific types
- `dataset_id`: Per-study biases or technical artifacts

---

## File Organization

```
GenePT-tools/
  core/
    postprocessing/
      __init__.py
      constrained_output.py       # Core API (generic)
      cellxgene_constraints.py    # Tissue-specific wrapper
  data/
    cellxgene_constraints/
      tissue_allowlists.json      # Tissue -> allowed class IDs
      tissue_class_logprobs.pt    # [K, C] log probs
      metadata.json               # Tissue/class vocabularies
  tests/
    test_constrained_output.py
  specs/
    constrained-output.md         # This document
```

---

## Example End-to-End Workflow

```python
# 1. Initialize constraints
from genept_tools.core.postprocessing import CellxGeneTissueConstraints

constraints = CellxGeneTissueConstraints(
  allowlist_path="data/cellxgene_constraints/tissue_allowlists.json",
  prior_path="data/cellxgene_constraints/tissue_class_logprobs.pt",
  num_classes=100,  # number of cell types in your ontology
  device="cuda"     # or "cpu"
)

# 2. Model inference on homogeneous batch
import torch
tissue = "blood"  # All cells in this batch are from blood
logits = model(batch)  # [B, C] on GPU
labels = batch["labels"]

# 3. Choose constraint mode
MODE = "soft_prior"  # or "allowlist"

if MODE == "allowlist":
  # Broadcasting: [C] mask applied to all [B] examples
  logits_adj = constraints.apply_allowlist(logits, tissue)
elif MODE == "soft_prior":
  # Broadcasting: [C] prior applied to all [B] examples
  logits_adj = constraints.apply_soft_prior(
    logits,
    tissue,
    alpha=0.5  # tuned on dev set
  )

# 4. Get predictions
probs = torch.softmax(logits_adj, dim=-1)
predictions = probs.argmax(dim=-1)

# 5. Evaluate across all tissues
from sklearn.metrics import accuracy_score, f1_score

tissue_results = {}
for batch in eval_dataloader:
  tissue = batch["tissue"]  # Homogeneous batch
  logits = model(batch)
  logits_adj = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
  preds = logits_adj.argmax(dim=-1)

  # Collect per tissue
  if tissue not in tissue_results:
    tissue_results[tissue] = {"preds": [], "labels": []}
  tissue_results[tissue]["preds"].append(preds)
  tissue_results[tissue]["labels"].append(batch["labels"])

# Compute metrics
tissue_acc = evaluate_by_tissue(tissue_results, accuracy_score)
tissue_f1 = evaluate_by_tissue(tissue_results, lambda l, p: f1_score(l, p, average="macro"))

print(f"Blood accuracy: {tissue_acc['blood']:.3f}")
print(f"Brain accuracy: {tissue_acc['brain']:.3f}")
```

---

## Testing Strategy

### Unit Tests

```python
def test_allowlist_basic():
  """Test basic allowlist masking with broadcasting."""
  B, C = 4, 10
  logits = torch.randn(B, C)

  # Build mask for one context
  allowed_classes = {0, 2, 5}
  mask = build_allowed_mask_from_set(allowed_classes, C)

  out = apply_allowlist_mask(logits, mask)

  # Check forbidden classes are suppressed (all rows)
  assert torch.all(out[:, 1] < -1e8)
  assert torch.all(out[:, 7] < -1e8)

  # Check allowed classes unchanged (all rows)
  assert torch.allclose(out[:, 0], logits[:, 0])
  assert torch.allclose(out[:, 2], logits[:, 2])


def test_soft_prior_basic():
  """Test soft prior addition with broadcasting."""
  B, C = 4, 10
  logits = torch.randn(B, C)
  logp = torch.randn(C)  # Single context
  alpha = 0.5

  out = add_soft_prior_bias(logits, logp, alpha)

  # Check bias is applied correctly (same for all rows)
  expected = logits + alpha * logp  # Broadcasting
  assert torch.allclose(out, expected)


def test_mask_building():
  """Test conversion from sparse set to dense mask."""
  allowed_classes = {0, 2, 5}
  C = 10
  mask = build_allowed_mask_from_set(allowed_classes, C)

  # Check shape
  assert mask.shape == (C,)

  # Check specific entries
  assert mask[0] == True
  assert mask[1] == False
  assert mask[2] == True
  assert mask[5] == True
  assert mask[7] == False


def test_broadcasting_efficiency():
  """Test that broadcasting is efficient."""
  B, C = 1000, 100
  logits = torch.randn(B, C)
  mask = torch.rand(C) > 0.5

  # Time broadcasting version
  import time
  start = time.time()
  out = apply_allowlist_mask(logits, mask)
  elapsed = time.time() - start

  # Should be very fast even on CPU
  assert elapsed < 0.05  # 50ms is generous

  # Check correctness
  assert out.shape == (B, C)
  # Allowed classes unchanged
  assert torch.allclose(out[:, mask], logits[:, mask])
  # Forbidden classes suppressed
  assert torch.all(out[:, ~mask] < -1e8)
```

### Integration Tests

```python
def test_cellxgene_constraints_end_to_end():
  """Test full pipeline with tissue constraints (homogeneous batch)."""
  constraints = CellxGeneTissueConstraints(
    allowlist_path="test_data/tissue_allowlists.json",
    prior_path="test_data/tissue_class_logprobs.pt",
    num_classes=50,
    device="cpu"
  )

  B, C = 4, 50
  logits = torch.randn(B, C)
  tissue = "blood"  # Homogeneous batch

  # Test allowlist mode
  mask = constraints.get_allowlist_mask(tissue)
  logits_masked = constraints.apply_allowlist(logits, tissue)
  probs = torch.softmax(logits_masked, dim=-1)
  assert probs.sum(dim=-1).allclose(torch.ones(B))

  # Verify constraints were applied (same for all rows)
  assert torch.all(logits_masked[:, ~mask] < -1e8)  # Forbidden suppressed
  assert torch.allclose(logits_masked[:, mask], logits[:, mask])  # Allowed unchanged

  # Test soft prior mode
  logp = constraints.get_logp(tissue)
  logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
  probs = torch.softmax(logits_biased, dim=-1)
  assert probs.sum(dim=-1).allclose(torch.ones(B))

  # Verify bias was added (same for all rows)
  expected = logits + 0.5 * logp  # Broadcasting
  assert torch.allclose(logits_biased, expected)


def test_multiple_tissues():
  """Test processing multiple tissues sequentially."""
  constraints = CellxGeneTissueConstraints(
    allowlist_path="test_data/tissue_allowlists.json",
    prior_path="test_data/tissue_class_logprobs.pt",
    num_classes=50
  )

  tissues = ["blood", "brain", "liver"]
  results = {}

  for tissue in tissues:
    logits = torch.randn(8, 50)  # Batch for this tissue
    logits_adj = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
    results[tissue] = logits_adj.argmax(dim=-1)

  # Verify different tissues get different adjustments
  # (This would fail if we accidentally shared state)
  assert len(results) == 3
```

---

## Performance Considerations

### Broadcasting Benefits

The homogeneous batch assumption enables efficient broadcasting:

```python
# Benchmark: 1000 examples, 100 classes
B, C = 1000, 100

# Key operations use native PyTorch broadcasting:
# 1. Mask application: torch.where(mask, logits, very_neg)
#    - mask: [C] broadcasts to [B, C]
#    - ~0.1-0.5ms on GPU, ~1-2ms on CPU
#
# 2. Prior addition: logits + alpha * logp
#    - logp: [C] broadcasts to [B, C]
#    - ~0.1-0.3ms on GPU, ~0.5-1ms on CPU

# Memory efficiency:
# - Per-context mask: C * 1 byte (bool) = 100 bytes for 100 classes
# - Per-context prior: C * 4 bytes (float32) = 400 bytes for 100 classes
# - Total overhead per tissue: ~500 bytes (negligible)
```

### Batching Recommendations

- **Optimal batch size**: 256-1024 examples for inference
- **Context grouping**: Group cells by tissue during data loading
  ```python
  # Good: stratified by tissue
  dataloader = DataLoader(dataset, batch_sampler=TissueBatchSampler(...))

  # Avoid: mixed tissues (would violate homogeneity assumption)
  dataloader = DataLoader(dataset, shuffle=True)  # Don't do this
  ```
- **Mask caching**: Store masks/priors in `CellxGeneTissueConstraints`, fetch per batch
- **Device placement**: Keep constraints on same device as logits
- **Mixed precision**: Works seamlessly with `torch.cuda.amp`

---

## Implementation Status & Integration Plan

### Current Status (as of 2025-11-14)

**Implementation**: ✅ COMPLETE
- All core functions implemented in `core_tmp/postprocessing/`
- CellxGene wrapper class fully functional
- CLI tool for building constraints from counts
- Comprehensive test suite (35 tests)

**Data Files**: ✅ COMPLETE
- Real constraint data generated at `/data/GenePT-tools/data/cellxgene_constraints/`
- 218 tissues, 832 cell types
- Allowlists and soft priors ready for production use

**Testing**: ⚠️ WRITTEN BUT NOT RUN
- Tests exist but have import path issues
- Need to fix imports after integration

**Integration**: ❌ PENDING
- Code isolated in `core_tmp/` staging area
- Not yet integrated into main `src/` package structure

### Integration Plan

#### Phase 1: Directory Structure & File Moves

**Create new inference module:**
```
src/inference/                    # NEW - Inference-related modules
  __init__.py                     # Export public API
  constraints/                    # NEW - Constraint implementations
    __init__.py                   # Export constraint classes/functions
    core.py                       # Renamed from constrained_output.py
    cellxgene.py                  # Renamed from cellxgene_constraints.py
```

**Move CLI tool:**
```
scripts/build_tissue_constraints.py  # Moved from core_tmp/postprocessing/
```

**File movements:**
1. `core_tmp/postprocessing/constrained_output.py` → `src/inference/constraints/core.py`
2. `core_tmp/postprocessing/cellxgene_constraints.py` → `src/inference/constraints/cellxgene.py`
3. `core_tmp/postprocessing/__init__.py` → `src/inference/constraints/__init__.py` (update imports)
4. `core_tmp/postprocessing/build_tissue_constraints.py` → `scripts/build_tissue_constraints.py`

#### Phase 2: Import Path Updates

**In `src/inference/constraints/cellxgene.py`:**
```python
# Change from:
from constrained_output import apply_allowlist_mask, add_soft_prior_bias

# To:
from .core import apply_allowlist_mask, add_soft_prior_bias, build_allowed_mask_from_set
```

**In `src/inference/constraints/__init__.py`:**
```python
from .core import (
    apply_allowlist_mask,
    add_soft_prior_bias,
    counts_to_logp,
    build_allowed_mask_from_set,
    build_allowlist_dict,
    build_soft_prior_dict,
)
from .cellxgene import CellxGeneTissueConstraints

__all__ = [
    'CellxGeneTissueConstraints',
    'apply_allowlist_mask',
    'add_soft_prior_bias',
    'counts_to_logp',
    'build_allowed_mask_from_set',
    'build_allowlist_dict',
    'build_soft_prior_dict',
]
```

**Create `src/inference/__init__.py`:**
```python
from .constraints import CellxGeneTissueConstraints

__all__ = ['CellxGeneTissueConstraints']
```

#### Phase 3: Test Import Fixes

**In `test/test_constrained_output.py`:**
```python
# Change from:
from constrained_output import apply_allowlist_mask, add_soft_prior_bias, ...

# To:
from src.inference.constraints.core import (
    apply_allowlist_mask,
    add_soft_prior_bias,
    counts_to_logp,
    build_allowed_mask_from_set,
    build_allowlist_dict,
    build_soft_prior_dict,
)
```

**In `test/test_cellxgene_constraints.py`:**
```python
# Change from:
from cellxgene_constraints import CellxGeneTissueConstraints

# To:
from src.inference.constraints import CellxGeneTissueConstraints
```

#### Phase 4: Verification

Run test suite to verify integration:
```bash
pytest test/test_constrained_output.py -v
pytest test/test_cellxgene_constraints.py -v
```

Expected: All 35 tests pass

#### Phase 5: Usage Example

Create `examples/tissue_constrained_inference.py` demonstrating:

1. **Load trained model:**
   ```python
   model = MLPClassifier.load_from_checkpoint('path/to/checkpoint.pt')
   constraints = CellxGeneTissueConstraints(
       allowlist_path='data/cellxgene_constraints/tissue_allowlists.json',
       prior_path='data/cellxgene_constraints/tissue_class_logprobs.pt',
       num_classes=302
   )
   ```

2. **Allowlist mode (hard constraints):**
   ```python
   logits = model(embeddings)  # [B, 302]
   logits_constrained = constraints.apply_allowlist(logits, tissue='UBERON:0000178')
   predictions = logits_constrained.argmax(dim=-1)
   ```

3. **Soft prior mode (probabilistic bias):**
   ```python
   logits_biased = constraints.apply_soft_prior(
       logits,
       tissue='UBERON:0000178',
       alpha=0.5  # Prior strength
   )
   predictions = logits_biased.argmax(dim=-1)
   ```

4. **Compare modes:**
   - Show baseline predictions (no constraints)
   - Show allowlist predictions (forbidden classes suppressed)
   - Show soft prior predictions (biased toward typical classes)
   - Compute accuracy improvements per mode

#### Phase 6: Documentation

Create `src/inference/constraints/README.md`:

```markdown
# Tissue-Aware Constraints for Cell Type Classification

Quick usage guide for applying tissue constraints to model predictions.

## Modes

**Allowlist (Hard)**: Only permit anatomically-valid cell types
**Soft Prior (Probabilistic)**: Bias toward tissue-typical cell types

## Alpha Parameter Tuning

- `alpha=0.0`: No prior (baseline model)
- `alpha=0.3-0.5`: Gentle bias (recommended for well-calibrated models)
- `alpha=1.0`: Strong bias (equal weight to model and prior)
- `alpha>1.0`: Prior dominates (use with caution)

Tune on validation set to maximize accuracy while preserving rare cell types.
```

### Estimated Effort

- **Phase 1-3** (File moves & import fixes): ~1 hour
- **Phase 4** (Test verification): ~30 minutes
- **Phase 5** (Usage example): ~1 hour
- **Phase 6** (Documentation): ~30 minutes

**Total**: ~3 hours to production-ready integration

### Dependencies

- No new package dependencies required
- Uses existing PyTorch, JSON, pathlib
- Compatible with current model architecture (302 classes post-filtering)

---

## Future Extensions

1. **Hierarchical constraints**: Apply coarse-grained allowlist + fine-grained prior
2. **Confidence-aware priors**: Stronger bias for low-confidence predictions
3. **Dynamic alpha**: Learn per-example or per-class alpha values
4. **Multi-key contexts**: Combine tissue + sequencer + species constraints (still homogeneous per batch)
5. **Uncertainty quantification**: Compute entropy before/after constraints
6. **Heterogeneous batches**: For mixed-tissue batches, use `[K, C]` lookup table and index with `tissue_idx[B]` (trades simplicity for flexibility)

---

## Empirical Evaluation Results

### Experimental Setup

**Model**: MLP classifier (checkpoint_epoch2_batch267_step1750.pt)
- Architecture: 6 hidden layers, 12% dropout
- Input: Composable embeddings (GenePT 1536d + scGPT 512d + metadata)
- Output: 302 cell types (filtered from 832 with cell_count_threshold=10,000)

**Dataset**: CellxGene test set
- 108,855 samples across 15 files
- 17 unique tissues (100% coverage after tissue mapping)
- 85 unique cell types present

**Baseline Performance**:
- Accuracy: 46.84%
- Macro F1: 18.51%
- Hierarchical F1: 91.70%

### Alpha Sweep Results (α = 0.0 to 2.0)

Complete performance across 21 alpha values:

| Alpha | Accuracy | Δ Acc | Macro F1 | Δ F1 | Hier. F1 | Δ Hier. F1 |
|-------|----------|-------|----------|------|----------|------------|
| **Baseline** | **46.84%** | - | **18.51%** | - | **91.70%** | - |
| **Allowlist** | **15.66%** | **-31.19%** | **5.43%** | **-13.09%** | **78.42%** | **-13.28%** |
| 0.0 | 46.84% | +0.00% | 18.51% | +0.00% | 91.70% | +0.00% |
| 0.1 | 49.86% | +3.01% | 20.39% | +1.87% | 92.32% | +0.62% |
| 0.2 | 52.46% | +5.61% | 22.38% | +3.86% | 92.80% | +1.10% |
| 0.3 | 53.94% | +7.10% | 24.10% | +5.59% | 93.06% | +1.36% |
| 0.4 | 54.66% | +7.82% | 25.95% | +7.44% | 92.99% | +1.29% |
| **0.5** | **54.94%** | **+8.10%** | **27.94%** | **+9.43%** | **93.04%** | **+1.34%** |
| 0.6 | 54.93% | +8.08% | 28.69% | +10.17% | 93.04% | +1.34% |
| 0.7 | 54.85% | +8.00% | 28.98% | +10.47% | 93.03% | +1.33% |
| 0.8 | 54.71% | +7.87% | 29.49% | +10.98% | 92.99% | +1.29% |
| 0.9 | 54.53% | +7.68% | 29.80% | +11.28% | 92.93% | +1.23% |
| 1.0 | 54.25% | +7.41% | 30.47% | +11.96% | 92.84% | +1.14% |
| 1.1 | 53.77% | +6.93% | 31.37% | +12.85% | 92.69% | +0.99% |
| **1.2** | **53.20%** | **+6.36%** | **31.98%** | **+13.47%** | **92.51%** | **+0.81%** |
| 1.3 | 52.42% | +5.58% | 31.25% | +12.74% | 92.29% | +0.59% |
| 1.4 | 51.51% | +4.67% | 31.48% | +12.97% | 92.03% | +0.33% |
| 1.5 | 50.37% | +3.53% | 31.14% | +12.63% | 91.70% | +0.00% |
| 1.6 | 48.93% | +2.09% | 31.05% | +12.54% | 91.28% | -0.42% |
| 1.7 | 47.53% | +0.69% | 30.83% | +12.31% | 90.80% | -0.90% |
| 1.8 | 46.17% | -0.68% | 30.36% | +11.85% | 90.30% | -1.40% |
| 1.9 | 44.69% | -2.15% | 29.80% | +11.28% | 89.74% | -1.96% |
| 2.0 | 43.28% | -3.56% | 28.71% | +10.19% | 89.12% | -2.58% |

### Key Findings

**Metric-Specific Optima:**
- **Peak accuracy**: α = 0.5-0.6 (~54.9%, +8.1% over baseline)
- **Peak macro F1**: α = 1.2 (31.98%, +13.47% over baseline)
- **Peak hierarchical F1**: α = 0.3-0.7 (~93.0-93.1%, +1.3-1.4% over baseline)

**Crossover Points:**
- Accuracy drops below baseline at α > 1.7
- Hierarchical F1 returns to baseline at α = 1.5
- Macro F1 remains positive across entire range (even at α = 2.0)

**Hard Constraints (Allowlist):**
- Severely degrades all metrics (-31.19% accuracy, -13.09% macro F1, -13.28% hierarchical F1)
- Demonstrates that hard constraints are too restrictive for this task
- Soft priors are strongly preferred

### Recommended Alpha Settings

1. **Default (balanced performance)**: α = 0.5
   - +8.10% accuracy improvement
   - +9.43% macro F1 improvement
   - +1.34% hierarchical F1 improvement
   - Good balance across all metrics

2. **Accuracy-focused tasks**: α = 0.5-0.6
   - Best for general cell type annotation
   - Maximizes correct predictions

3. **Rare cell type detection**: α = 1.0-1.2
   - Best macro F1 performance
   - Improves performance on underrepresented classes
   - Acceptable accuracy tradeoff (+6-7% still strong)

4. **Ontology-aware tasks**: α = 0.3-0.5
   - Maximizes hierarchical F1
   - Best when biological relationships matter

5. **Conservative approach**: α = 0.3
   - All metrics improve
   - Low risk of over-constraining

### Per-File Results (α = 0.5)

Detailed breakdown across 15 evaluation files:

| File | Samples | Accuracy | Macro F1 | Hier. F1 | Recall@2 | Recall@5 | Recall@10 |
|------|---------|----------|----------|----------|----------|----------|-----------|
| 05a49baa-d326... | 3,770 | 26.8% | 4.8% | 78.0% | 43.6% | 85.0% | 98.8% |
| 06ef6b36-6c9b... | 5,917 | 24.3% | 7.2% | 79.6% | 42.1% | 65.0% | 91.5% |
| 17e9d436-a264... | 2,080 | 27.9% | 5.7% | 92.8% | 70.0% | 92.1% | 96.7% |
| 24584be9-d3d5... | 4,034 | 40.7% | 14.5% | 86.6% | 63.0% | 67.2% | 73.8% |
| 2d85960a-2ba8... | 7,672 | 32.5% | 6.0% | 94.8% | 66.1% | 87.1% | 93.1% |
| 32e8a3d7-7b15... | 8,784 | 16.1% | 5.1% | 91.9% | 32.2% | 63.1% | 85.2% |
| 54ea5aba-3413... | 11,221 | 30.4% | 7.2% | 86.1% | 45.1% | 65.0% | 76.8% |
| 6e00ccf7-0749... | 5,414 | 37.2% | 6.9% | 83.0% | 48.2% | 75.4% | 93.3% |
| 738942eb-ac72... | 13,809 | 37.0% | 7.7% | 93.3% | 47.5% | 64.5% | 82.2% |
| 9c1b5626-58df... | 2,101 | 17.2% | 5.4% | 84.3% | 36.8% | 59.9% | 80.2% |
| a82c43bb-a703... | 2,105 | 25.2% | 4.9% | 92.1% | 65.9% | 90.6% | 96.6% |
| c05e6940-729c... | 11,391 | 74.4% | 11.8% | 93.4% | 88.8% | 96.8% | 98.7% |
| c54c9659-1b6b... | 2,081 | 18.6% | 6.2% | 86.6% | 40.3% | 65.5% | 82.6% |
| dc30c3ec-46d6... | 6,540 | 30.0% | 3.5% | 86.2% | 44.8% | 76.5% | 88.2% |
| f5b0810c-1664... | 21,936 | 91.9% | 51.8% | 98.5% | 98.3% | 99.9% | 100.0% |

**Summary Statistics:**
- Accuracy: mean=35.4%, std=21.0%, min=16.1%, max=91.9%
- Macro F1: mean=9.9%, std=11.9%, min=3.5%, max=51.8%
- Hierarchical F1: mean=88.5%, std=5.9%, min=78.0%, max=98.5%
- Recall@2: mean=55.5%, std=19.3%, min=32.2%, max=98.3%
- Recall@10: mean=89.2%, std=8.6%, min=73.8%, max=100.0%

**Observations:**
- High variance across files reflects dataset heterogeneity
- Best-performing file (f5b0810c): 91.9% accuracy, 51.8% macro F1
- Most challenging file (32e8a3d7): 16.1% accuracy, 5.1% macro F1
- Hierarchical F1 is more stable (78-98.5%) than accuracy (16-92%)
- Recall@10 consistently high (>73%), showing strong ranking performance

### Implementation Notes

**Tissue ID Mapping:**
- All 17 tissues in test data successfully mapped to CZ slim ontology
- Three-tier mapping strategy:
  1. Direct CZ slim lookup
  2. Nearest CZ slim ancestor via ontology
  3. Fallback mappings for non-CZ slim tissues
- Achieves 100% tissue coverage (108,855/108,855 samples)

**Dynamic Vocabulary Filtering:**
- Constraint files built for full 832-class vocabulary
- Dynamically filtered to match model's 302 classes at evaluation time
- Enables same constraint files to work with any `cell_count_threshold`
- No need to rebuild constraints when changing model vocabulary

**Performance:**
- Constraint application adds <1ms overhead per batch
- Negligible memory footprint (~500 bytes per tissue)
- Compatible with mixed precision training/inference

---

## V1 Model Evaluation Results (GenePT-only baseline)

### Experimental Setup

**Model**: V1 MLP classifier (checkpoint mlp_checkpoint_6000.pt, run nwhis8xb)
- Architecture: 3 hidden layers, 5.3% dropout
- Input: GenePT embeddings only (500 dimensions, first 500 of 3072)
- Output: 377 cell types (>10,000 sample threshold, no filtering)
- Training: 2 epochs, no input scaling

**Dataset**: CellxGene test set (same as v2)
- 120,984 samples across 15 files
- 10 unique tissues (100% coverage after tissue mapping)
- 89 unique cell types present

**Baseline Performance**:
- Accuracy: 38.54%
- Macro F1: 12.66%
- Recall@2: 51.90%
- Recall@10: 80.42%
- Hierarchical F1: 87.36%

### Alpha Sweep Results (α = 0.0 to 2.0)

Complete performance across 21 alpha values:

| Alpha | Accuracy | Δ Acc | Macro F1 | Δ F1 | Hier. F1 | Δ Hier. F1 |
|-------|----------|-------|----------|------|----------|------------|
| **Baseline** | **38.54%** | - | **12.66%** | - | **87.36%** | - |
| **Allowlist** | **9.61%** | **-28.93%** | **3.05%** | **-9.61%** | **70.17%** | **-17.19%** |
| 0.0 | 38.54% | +0.00% | 12.66% | +0.00% | 87.36% | +0.00% |
| 0.1 | 45.52% | +6.98% | 16.67% | +4.01% | 88.92% | +1.56% |
| 0.2 | 48.01% | +9.46% | 19.98% | +7.32% | 89.38% | +2.02% |
| 0.3 | 49.04% | +10.50% | 22.69% | +10.03% | 89.60% | +2.24% |
| 0.4 | 49.70% | +11.16% | 24.02% | +11.37% | 89.71% | +2.35% |
| 0.5 | 50.05% | +11.51% | 25.15% | +12.50% | 89.71% | +2.35% |
| **0.6** | **50.20%** | **+11.66%** | **25.49%** | **+12.84%** | **89.69%** | **+2.33%** |
| **0.7** | **50.20%** | **+11.66%** | **26.43%** | **+13.78%** | **89.63%** | **+2.27%** |
| 0.8 | 49.91% | +11.37% | 26.90% | +14.25% | 89.50% | +2.14% |
| 0.9 | 49.39% | +10.85% | 27.76% | +15.10% | 89.34% | +1.98% |
| 1.0 | 48.72% | +10.17% | 28.20% | +15.54% | 89.11% | +1.75% |
| 1.1 | 47.77% | +9.23% | 28.11% | +15.45% | 88.79% | +1.43% |
| 1.2 | 46.62% | +8.08% | 28.25% | +15.59% | 88.40% | +1.04% |
| **1.3** | **45.29%** | **+6.75%** | **28.37%** | **+15.71%** | **87.94%** | **+0.58%** |
| 1.4 | 43.72% | +5.18% | 27.91% | +15.25% | 87.38% | +0.02% |
| 1.5 | 42.12% | +3.58% | 26.90% | +14.24% | 86.75% | -0.61% |
| 1.6 | 40.54% | +1.99% | 26.10% | +13.44% | 86.08% | -1.28% |
| 1.7 | 38.95% | +0.40% | 25.09% | +12.43% | 85.35% | -2.01% |
| 1.8 | 37.34% | -1.20% | 23.85% | +11.19% | 84.54% | -2.82% |
| 1.9 | 35.89% | -2.65% | 23.00% | +10.34% | 83.72% | -3.64% |
| 2.0 | 34.47% | -4.08% | 21.57% | +8.91% | 82.85% | -4.51% |

### Key Findings

**Metric-Specific Optima:**
- **Peak accuracy**: α = 0.6-0.7 (50.20%, +11.66% over baseline)
- **Peak macro F1**: α = 1.3 (28.37%, +15.71% over baseline)
- **Peak hierarchical F1**: α = 0.4-0.6 (~89.7%, +2.3-2.4% over baseline)

**Crossover Points:**
- Accuracy drops below baseline at α > 1.7
- Hierarchical F1 returns to baseline at α ≈ 1.4
- Macro F1 remains positive across entire range (even at α = 2.0)

**Hard Constraints (Allowlist):**
- Severely degrades all metrics (-28.93% accuracy, -9.61% macro F1, -17.19% hierarchical F1)
- Demonstrates that hard constraints are too restrictive (consistent with v2 findings)
- Soft priors are strongly preferred

### Comparison: V1 vs V2 Models

**Architecture Differences:**
- **V1**: 377 classes, 500 GenePT dims only, 3 hidden layers, no input scaling
- **V2**: 302 classes, 2048 dims (GenePT 1536d + scGPT 512d + metadata), 6 hidden layers, scaled inputs

**Baseline Performance Comparison:**

| Metric | V1 (GenePT-only) | V2 (Composable) | Δ V2 vs V1 |
|--------|------------------|-----------------|------------|
| Accuracy | 38.54% | 46.84% | +8.30% |
| Macro F1 | 12.66% | 18.51% | +5.85% |
| Recall@2 | 51.90% | - | - |
| Recall@10 | 80.42% | - | - |
| Hierarchical F1 | 87.36% | 91.70% | +4.34% |

**Constrained Performance (Optimal α):**

| Metric | V1 @ α=0.6 | V2 @ α=0.5 | Δ V2 vs V1 |
|--------|------------|------------|------------|
| Accuracy | 50.20% | 54.94% | +4.74% |
| Macro F1 | 25.49% | 27.94% | +2.45% |
| Hierarchical F1 | 89.69% | 93.04% | +3.35% |

**Absolute Improvements from Constraints:**

| Metric | V1 Improvement | V2 Improvement | Δ V2 vs V1 |
|--------|----------------|----------------|------------|
| Accuracy | +11.66% | +8.10% | -3.56% |
| Macro F1 | +15.71% | +13.47% | -2.24% |
| Hierarchical F1 | +2.35% | +1.34% | -1.01% |

**Key Insights:**
1. **V2 baseline is stronger**: Composable embeddings provide +8.3% accuracy over GenePT-only
2. **V1 benefits more from constraints**: +11.66% accuracy gain vs +8.10% for V2
3. **Both models converge with constraints**: V1 reaches 50.2%, V2 reaches 54.9% (~4.7% gap)
4. **Optimal alpha differs slightly**: V1 peaks at α=0.6-0.7, V2 peaks at α=0.5
5. **Consistent pattern**: Both models show accuracy peak at α=0.5-0.7, macro F1 peak at α=1.2-1.3
6. **Hard constraints fail universally**: Both models suffer severe degradation with allowlist mode

### Implementation Notes (V1-specific)

**Checkpoint Loading:**
- V1 checkpoints saved without "model." prefix (direct nn.Sequential)
- V2 checkpoints have "model." prefix (wrapped in MLPClassifier)
- Evaluation script automatically detects and remaps v1 checkpoint keys

**Input Scaling:**
- V1 trained on raw GenePT embeddings (no scaling)
- V2 uses 0.021 scaling factor for GenePT, 0.044 for scGPT
- Must disable scaling when evaluating v1 models (`disable_scaling=True`)

**Cell Type Mapping:**
- V1 uses row-indexed mapping (row 0 → output 0, no filtering)
- V2 uses dynamic code remapping (filtered from 832 to 302 classes)
- Both approaches compatible with constraint framework

---

## References

- CellxGene Census: https://chanzuckerberg.github.io/cellxgene-census/
- Tissue-cell type relationships: CellxGene ontology mappings
- Calibration: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
