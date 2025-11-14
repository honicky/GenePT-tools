"""
Post-processing module for context-aware classification constraints.
"""

from .core import (
    apply_allowlist_mask,
    build_allowed_mask_from_set,
    add_soft_prior_bias,
    counts_to_logp,
    build_allowlist_dict,
    build_soft_prior_dict,
)
from .cellxgene import CellxGeneTissueConstraints

__all__ = [
    # Core API
    "apply_allowlist_mask",
    "build_allowed_mask_from_set",
    "add_soft_prior_bias",
    "counts_to_logp",
    "build_allowlist_dict",
    "build_soft_prior_dict",
    # CellxGene-specific
    "CellxGeneTissueConstraints",
]
