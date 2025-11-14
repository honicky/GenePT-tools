"""
Constrained output for context-aware classification.

Post-processing module that applies context-aware constraints or biases to model
predictions. Operates entirely in logit space without requiring model retraining.

Supports two modes:
1. Allowlist (hard constraint): Only permitted classes can be predicted
2. Soft prior (probabilistic bias): Nudge predictions toward context-typical classes

Key property: Homogeneous batch assumption - all examples in a batch share the
same context (e.g., same tissue, same sequencer, same species).
"""

import torch
from typing import Set, Dict


def apply_allowlist_mask(
    logits: torch.Tensor,
    allowed_mask: torch.Tensor,
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


def add_soft_prior_bias(
    logits: torch.Tensor,
    logp_class: torch.Tensor,
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


def counts_to_logp(
    counts: torch.Tensor,
    epsilon: float = 1.0
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


def build_allowlist_dict(
    counts_by_context: Dict[str, torch.Tensor],
    min_count: int = 10
) -> Dict[str, list]:
    """
    Build allowlist dict from co-occurrence counts.

    Args:
        counts_by_context: Dict mapping context name to count vector [C]
        min_count: Minimum observations to include class for context

    Returns:
        Dict mapping context name to list of allowed class indices
    """
    allowlists = {}
    for context, counts in counts_by_context.items():
        # Vectorized threshold comparison
        allowed_mask = counts >= min_count
        # Get indices where True
        allowed_classes = allowed_mask.nonzero(as_tuple=True)[0]
        allowlists[context] = allowed_classes.tolist()

    return allowlists


def build_soft_prior_dict(
    counts_by_context: Dict[str, torch.Tensor],
    epsilon: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Build log probability dict from counts.

    Args:
        counts_by_context: Dict mapping context name to count vector [C]
        epsilon: Laplace smoothing constant

    Returns:
        Dict mapping context name to log probability vector [C]
    """
    context_to_logp = {}
    for context, counts in counts_by_context.items():
        context_to_logp[context] = counts_to_logp(counts, epsilon)

    return context_to_logp
