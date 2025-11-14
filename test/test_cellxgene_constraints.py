"""
Integration tests for CellxGeneTissueConstraints.

Tests the full pipeline with tissue constraints using real CellxGene data.
"""

import pytest
import torch
import json
from pathlib import Path

from src.inference.constraints import CellxGeneTissueConstraints


@pytest.fixture
def constraint_paths():
    """Paths to real CellxGene constraint files."""
    data_dir = Path("/data/GenePT-tools/data/cellxgene_constraints")
    return {
        "allowlist": data_dir / "tissue_allowlists.json",
        "prior": data_dir / "tissue_class_logprobs.pt",
    }


@pytest.fixture
def constraints(constraint_paths):
    """Initialize CellxGeneTissueConstraints with real data."""
    return CellxGeneTissueConstraints(
        allowlist_path=str(constraint_paths["allowlist"]),
        prior_path=str(constraint_paths["prior"]),
        device="cpu"
    )


class TestInitialization:
    """Test constraint initialization."""

    def test_load_allowlists_only(self, constraint_paths):
        """Test loading only allowlists."""
        # Need to specify num_classes when loading allowlists
        constraints = CellxGeneTissueConstraints(
            allowlist_path=str(constraint_paths["allowlist"]),
            num_classes=832,  # From CellxGene data
            device="cpu"
        )

        assert len(constraints.tissue_to_allowlist) > 0
        assert len(constraints.tissue_to_logp) == 0

    def test_load_priors_only(self, constraint_paths):
        """Test loading only priors."""
        constraints = CellxGeneTissueConstraints(
            prior_path=str(constraint_paths["prior"]),
            device="cpu"
        )

        assert len(constraints.tissue_to_logp) > 0
        assert len(constraints.tissue_to_allowlist) == 0
        # num_classes should be inferred from priors
        assert constraints.num_classes == 832

    def test_load_both(self, constraint_paths):
        """Test loading both allowlists and priors."""
        constraints = CellxGeneTissueConstraints(
            allowlist_path=str(constraint_paths["allowlist"]),
            prior_path=str(constraint_paths["prior"]),
            device="cpu"
        )

        assert len(constraints.tissue_to_allowlist) > 0
        assert len(constraints.tissue_to_logp) > 0
        assert constraints.num_classes == 832

    def test_missing_num_classes_error(self, constraint_paths):
        """Test that missing num_classes raises error for allowlist."""
        with pytest.raises(ValueError, match="num_classes required"):
            CellxGeneTissueConstraints(
                allowlist_path=str(constraint_paths["allowlist"]),
                device="cpu"
            )

    def test_repr(self, constraints):
        """Test string representation."""
        repr_str = repr(constraints)
        assert "CellxGeneTissueConstraints" in repr_str
        assert "num_classes" in repr_str


class TestAllowlistMode:
    """Test allowlist functionality with real data."""

    def test_get_allowlist_mask(self, constraints):
        """Test getting allowlist mask for a tissue."""
        # Get a tissue that exists
        tissues = constraints.available_tissues()
        assert len(tissues) > 0

        tissue = tissues[0]
        mask = constraints.get_allowlist_mask(tissue)

        # Check shape
        assert mask.shape == (constraints.num_classes,)
        assert mask.dtype == torch.bool

        # At least some classes should be allowed
        assert torch.any(mask)

    def test_get_allowlist_unknown_tissue(self, constraints):
        """Test error for unknown tissue."""
        with pytest.raises(ValueError, match="Unknown tissue"):
            constraints.get_allowlist_mask("INVALID_TISSUE_ID")

    def test_apply_allowlist(self, constraints):
        """Test applying allowlist to a batch."""
        tissue = constraints.available_tissues()[0]
        B, C = 8, constraints.num_classes
        logits = torch.randn(B, C)

        # Apply allowlist
        logits_masked = constraints.apply_allowlist(logits, tissue)

        # Check shape
        assert logits_masked.shape == (B, C)

        # Check that forbidden classes are suppressed
        mask = constraints.get_allowlist_mask(tissue)
        assert torch.all(logits_masked[:, ~mask] < -1e8)

        # Check that allowed classes are unchanged
        assert torch.allclose(logits_masked[:, mask], logits[:, mask])

    def test_allowlist_probabilities_normalized(self, constraints):
        """Test that probabilities are normalized after allowlist."""
        tissue = constraints.available_tissues()[0]
        B, C = 8, constraints.num_classes
        logits = torch.randn(B, C)

        logits_masked = constraints.apply_allowlist(logits, tissue)
        probs = torch.softmax(logits_masked, dim=-1)

        # Check normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))


class TestSoftPriorMode:
    """Test soft prior functionality with real data."""

    def test_get_logp(self, constraints):
        """Test getting log prior for a tissue."""
        tissues = constraints.available_tissues()
        tissue = tissues[0]

        logp = constraints.get_logp(tissue)

        # Check shape
        assert logp.shape == (constraints.num_classes,)

        # Check that it's a valid log probability distribution
        p = torch.exp(logp)
        assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)

        # All should be finite
        assert torch.isfinite(logp).all()

    def test_get_logp_unknown_tissue(self, constraints):
        """Test error for unknown tissue."""
        with pytest.raises(ValueError, match="Unknown tissue"):
            constraints.get_logp("INVALID_TISSUE_ID")

    def test_apply_soft_prior(self, constraints):
        """Test applying soft prior to a batch."""
        tissue = constraints.available_tissues()[0]
        B, C = 8, constraints.num_classes
        logits = torch.randn(B, C)
        alpha = 0.5

        # Apply soft prior
        logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=alpha)

        # Check shape
        assert logits_biased.shape == (B, C)

        # Check that bias was added
        logp = constraints.get_logp(tissue)
        expected = logits + alpha * logp
        assert torch.allclose(logits_biased, expected)

    def test_soft_prior_probabilities_normalized(self, constraints):
        """Test that probabilities are normalized after soft prior."""
        tissue = constraints.available_tissues()[0]
        B, C = 8, constraints.num_classes
        logits = torch.randn(B, C)

        logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
        probs = torch.softmax(logits_biased, dim=-1)

        # Check normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))

    def test_different_alpha_values(self, constraints):
        """Test soft prior with different alpha values."""
        tissue = constraints.available_tissues()[0]
        B, C = 8, constraints.num_classes
        logits = torch.randn(B, C)

        # Alpha = 0 (no bias)
        logits_0 = constraints.apply_soft_prior(logits, tissue, alpha=0.0)
        assert torch.allclose(logits_0, logits)

        # Alpha = 1.0 (full bias)
        logp = constraints.get_logp(tissue)
        logits_1 = constraints.apply_soft_prior(logits, tissue, alpha=1.0)
        assert torch.allclose(logits_1, logits + logp)


class TestMultipleTissues:
    """Test processing multiple tissues."""

    def test_multiple_tissues_sequentially(self, constraints):
        """Test processing multiple tissues in sequence."""
        tissues = constraints.available_tissues()[:3]  # First 3 tissues
        results = {}

        for tissue in tissues:
            B, C = 8, constraints.num_classes
            logits = torch.randn(B, C)
            logits_adj = constraints.apply_soft_prior(logits, tissue, alpha=0.5)
            results[tissue] = logits_adj.argmax(dim=-1)

        # Verify we got results for all tissues
        assert len(results) == len(tissues)

    def test_different_tissues_different_constraints(self, constraints):
        """Test that different tissues get different constraints."""
        tissues = constraints.available_tissues()
        if len(tissues) < 2:
            pytest.skip("Need at least 2 tissues")

        tissue1, tissue2 = tissues[0], tissues[1]

        # Get allowlists
        mask1 = constraints.get_allowlist_mask(tissue1)
        mask2 = constraints.get_allowlist_mask(tissue2)

        # They should be different (unless by coincidence)
        assert not torch.all(mask1 == mask2)

        # Get priors
        logp1 = constraints.get_logp(tissue1)
        logp2 = constraints.get_logp(tissue2)

        # They should be different
        assert not torch.allclose(logp1, logp2)


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_tissue_label(self, constraints):
        """Test getting tissue labels."""
        tissues = constraints.available_tissues()
        if not tissues:
            pytest.skip("No tissues available")

        tissue_id = tissues[0]
        label = constraints.get_tissue_label(tissue_id)

        # Should return a string
        assert isinstance(label, str)
        assert len(label) > 0

    def test_available_tissues(self, constraints):
        """Test getting available tissues."""
        tissues = constraints.available_tissues()

        # Should return a list
        assert isinstance(tissues, list)
        assert len(tissues) > 0

        # Should be sorted
        assert tissues == sorted(tissues)


class TestDevicePlacement:
    """Test device placement for tensors."""

    def test_mask_device(self, constraints):
        """Test that masks are placed on correct device."""
        tissue = constraints.available_tissues()[0]

        # CPU
        mask_cpu = constraints.get_allowlist_mask(tissue, device="cpu")
        assert mask_cpu.device.type == "cpu"

        # GPU (if available)
        if torch.cuda.is_available():
            mask_gpu = constraints.get_allowlist_mask(tissue, device="cuda")
            assert mask_gpu.device.type == "cuda"

    def test_logp_device(self, constraints):
        """Test that log priors are placed on correct device."""
        tissue = constraints.available_tissues()[0]

        # CPU
        logp_cpu = constraints.get_logp(tissue, device="cpu")
        assert logp_cpu.device.type == "cpu"

        # GPU (if available)
        if torch.cuda.is_available():
            logp_gpu = constraints.get_logp(tissue, device="cuda")
            assert logp_gpu.device.type == "cuda"


class TestRealWorldScenario:
    """Test realistic end-to-end scenarios."""

    def test_blood_tissue_constraints(self, constraints):
        """Test constraints for blood tissue specifically."""
        # Look for blood tissue (UBERON:0000178)
        blood_tissue = "UBERON:0000178"
        if blood_tissue not in constraints.available_tissues():
            pytest.skip("Blood tissue not in dataset")

        # Verify blood tissue has allowlist
        mask = constraints.get_allowlist_mask(blood_tissue)
        num_allowed = mask.sum().item()
        assert num_allowed > 0
        print(f"Blood tissue: {num_allowed} allowed cell types")

        # Verify blood tissue has prior
        logp = constraints.get_logp(blood_tissue)
        assert torch.isfinite(logp).all()

        # Apply both modes
        B, C = 16, constraints.num_classes
        logits = torch.randn(B, C)

        logits_allowlist = constraints.apply_allowlist(logits, blood_tissue)
        logits_prior = constraints.apply_soft_prior(logits, blood_tissue, alpha=0.5)

        # Both should maintain probability normalization
        probs_allowlist = torch.softmax(logits_allowlist, dim=-1)
        probs_prior = torch.softmax(logits_prior, dim=-1)

        assert torch.allclose(probs_allowlist.sum(dim=-1), torch.ones(B))
        assert torch.allclose(probs_prior.sum(dim=-1), torch.ones(B))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
