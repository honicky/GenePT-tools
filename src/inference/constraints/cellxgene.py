"""
CellxGene tissue-specific constraints for cell type classification.

Provides tissue-aware allowlists and soft priors based on CellxGene corpus
co-occurrence data.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .core import (
    apply_allowlist_mask,
    add_soft_prior_bias,
    build_allowed_mask_from_set,
)


class CellxGeneTissueConstraints:
    """
    Tissue-aware constraints for cell type classification.

    Assumes homogeneous batches: all cells in a batch come from the same tissue.
    """

    def __init__(
        self,
        allowlist_path: Optional[str] = None,
        prior_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize tissue constraints.

        Args:
            allowlist_path: Path to tissue_allowlists.json
            prior_path: Path to tissue_class_logprobs.pt
            num_classes: Number of cell type classes (required for allowlist)
            device: Target device for tensors

        Raises:
            ValueError: If num_classes not provided when loading allowlist
        """
        self.num_classes = num_classes
        self.device = device

        # Load priors first (so we can infer num_classes if needed)
        self.tissue_to_logp: Dict[str, torch.Tensor] = {}
        self.tissue_vocab = []
        self.celltype_vocab = []
        self.tissue_id_to_label = {}

        if prior_path:
            data = torch.load(prior_path, map_location=device, weights_only=False)
            self.tissue_to_logp = data["tissue_to_logp"]
            self.tissue_vocab = data.get("tissue_vocab", [])
            self.celltype_vocab = data.get("celltype_vocab", [])
            self.tissue_id_to_label = data.get("tissue_id_to_label", {})

            # Update num_classes from prior if not set
            if self.num_classes is None and self.tissue_to_logp:
                first_tissue = next(iter(self.tissue_to_logp.keys()))
                self.num_classes = self.tissue_to_logp[first_tissue].shape[0]

        # Load allowlists: tissue -> allowed class IDs
        self.tissue_to_allowlist: Dict[str, list] = {}
        if allowlist_path:
            if self.num_classes is None:
                raise ValueError(
                    "num_classes required when loading allowlist "
                    "(must be provided explicitly or via prior_path)"
                )
            with open(allowlist_path) as f:
                self.tissue_to_allowlist = json.load(f)

    def get_allowlist_mask(
        self,
        tissue: str,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get allowlist mask for a specific tissue.

        Args:
            tissue: Tissue name/ID (e.g., "UBERON:0000178" for blood)
            device: Target device (default: self.device)

        Returns:
            Boolean mask [C] where True = allowed

        Raises:
            ValueError: If tissue not found in allowlist
        """
        if tissue not in self.tissue_to_allowlist:
            raise ValueError(
                f"Unknown tissue: {tissue}. "
                f"Available tissues: {sorted(self.tissue_to_allowlist.keys())[:10]}..."
            )

        allowed_classes = set(self.tissue_to_allowlist[tissue])
        return build_allowed_mask_from_set(
            allowed_classes,
            self.num_classes,
            device or self.device
        )

    def get_logp(
        self,
        tissue: str,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Get log prior for a specific tissue.

        Args:
            tissue: Tissue name/ID (e.g., "UBERON:0000178" for blood)
            device: Target device (default: self.device)

        Returns:
            Log probabilities [C] where logp[c] = log P(class=c | tissue)

        Raises:
            ValueError: If tissue not found in priors
        """
        if tissue not in self.tissue_to_logp:
            raise ValueError(
                f"Unknown tissue: {tissue}. "
                f"Available tissues: {sorted(self.tissue_to_logp.keys())[:10]}..."
            )

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
            alpha: Prior strength. Recommended range [0.25, 1.0].

        Returns:
            Adjusted logits [B, C]
        """
        logp = self.get_logp(tissue, device=logits.device)
        return add_soft_prior_bias(logits, logp, alpha)

    def get_tissue_label(self, tissue_id: str) -> str:
        """
        Get human-readable label for tissue ID.

        Args:
            tissue_id: Tissue ontology ID (e.g., "UBERON:0000178")

        Returns:
            Human-readable label (e.g., "blood")
        """
        return self.tissue_id_to_label.get(tissue_id, tissue_id)

    def available_tissues(self) -> list:
        """
        Get list of all available tissues.

        Returns:
            List of tissue IDs with constraints
        """
        # Union of tissues from allowlists and priors
        tissues = set(self.tissue_to_allowlist.keys()) | set(self.tissue_to_logp.keys())
        return sorted(tissues)

    def __repr__(self) -> str:
        """String representation."""
        n_tissues_allowlist = len(self.tissue_to_allowlist)
        n_tissues_prior = len(self.tissue_to_logp)
        return (
            f"CellxGeneTissueConstraints("
            f"num_classes={self.num_classes}, "
            f"allowlists={n_tissues_allowlist} tissues, "
            f"priors={n_tissues_prior} tissues)"
        )
