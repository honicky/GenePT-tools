"""
Unit tests for constrained output module.

Tests the core API functions for allowlist and soft prior modes.
"""

import pytest
import torch

from src.inference.constraints.core import (
    apply_allowlist_mask,
    build_allowed_mask_from_set,
    add_soft_prior_bias,
    counts_to_logp,
    build_allowlist_dict,
    build_soft_prior_dict,
)


class TestAllowlistMode:
    """Test allowlist (hard constraint) functionality."""

    def test_allowlist_basic(self):
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

    def test_mask_building(self):
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

    def test_empty_allowlist(self):
        """Test with empty allowlist (all forbidden)."""
        C = 10
        mask = build_allowed_mask_from_set(set(), C)

        assert mask.shape == (C,)
        assert torch.all(~mask)  # All False

    def test_full_allowlist(self):
        """Test with full allowlist (all allowed)."""
        C = 10
        all_classes = set(range(C))
        mask = build_allowed_mask_from_set(all_classes, C)

        assert mask.shape == (C,)
        assert torch.all(mask)  # All True

    def test_broadcasting_efficiency(self):
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
        assert elapsed < 0.1  # 100ms is generous

        # Check correctness
        assert out.shape == (B, C)
        # Allowed classes unchanged
        assert torch.allclose(out[:, mask], logits[:, mask])
        # Forbidden classes suppressed
        assert torch.all(out[:, ~mask] < -1e8)

    def test_device_placement(self):
        """Test mask building on different devices."""
        allowed_classes = {0, 2, 5}
        C = 10

        # CPU
        mask_cpu = build_allowed_mask_from_set(allowed_classes, C, device="cpu")
        assert mask_cpu.device.type == "cpu"

        # GPU (if available)
        if torch.cuda.is_available():
            mask_gpu = build_allowed_mask_from_set(allowed_classes, C, device="cuda")
            assert mask_gpu.device.type == "cuda"
            assert torch.all(mask_cpu == mask_gpu.cpu())


class TestSoftPriorMode:
    """Test soft prior (probabilistic bias) functionality."""

    def test_soft_prior_basic(self):
        """Test soft prior addition with broadcasting."""
        B, C = 4, 10
        logits = torch.randn(B, C)
        logp = torch.randn(C)  # Single context
        alpha = 0.5

        out = add_soft_prior_bias(logits, logp, alpha)

        # Check bias is applied correctly (same for all rows)
        expected = logits + alpha * logp  # Broadcasting
        assert torch.allclose(out, expected)

    def test_counts_to_logp(self):
        """Test conversion from counts to log probabilities."""
        counts = torch.tensor([100.0, 50.0, 25.0, 0.0])
        epsilon = 1.0

        logp = counts_to_logp(counts, epsilon)

        # Check shape
        assert logp.shape == counts.shape

        # Check normalization (exp and sum should be 1)
        p = torch.exp(logp)
        assert torch.allclose(p.sum(), torch.tensor(1.0))

        # Check smoothing (zero count should not be -inf)
        assert torch.isfinite(logp).all()

        # Check ordering preserved
        assert logp[0] > logp[1] > logp[2] > logp[3]

    def test_counts_to_logp_uniform(self):
        """Test with uniform counts."""
        counts = torch.ones(5) * 10.0
        epsilon = 1.0

        logp = counts_to_logp(counts, epsilon)

        # All should be equal (uniform distribution)
        assert torch.allclose(logp, logp[0].expand_as(logp))

        # Should equal log(1/5)
        expected = torch.log(torch.tensor(1.0 / 5.0))
        assert torch.allclose(logp[0], expected)

    def test_alpha_values(self):
        """Test different alpha values."""
        B, C = 4, 10
        logits = torch.randn(B, C)
        logp = torch.randn(C)

        # Alpha = 0 (no bias)
        out_0 = add_soft_prior_bias(logits, logp, alpha=0.0)
        assert torch.allclose(out_0, logits)

        # Alpha = 1.0 (full bias)
        out_1 = add_soft_prior_bias(logits, logp, alpha=1.0)
        assert torch.allclose(out_1, logits + logp)

        # Alpha = 0.5
        out_05 = add_soft_prior_bias(logits, logp, alpha=0.5)
        assert torch.allclose(out_05, logits + 0.5 * logp)


class TestBuilders:
    """Test builder functions for constraints."""

    def test_build_allowlist_dict(self):
        """Test building allowlists from counts."""
        counts_by_tissue = {
            "blood": torch.tensor([100, 5, 200, 50, 2, 0]),
            "brain": torch.tensor([10, 150, 5, 2, 100, 0]),
        }
        min_count = 10

        allowlists = build_allowlist_dict(counts_by_tissue, min_count)

        # Check blood allowlist
        assert set(allowlists["blood"]) == {0, 2, 3}  # counts >= 10

        # Check brain allowlist
        assert set(allowlists["brain"]) == {0, 1, 4}  # counts >= 10

    def test_build_soft_prior_dict(self):
        """Test building soft priors from counts."""
        counts_by_tissue = {
            "blood": torch.tensor([100.0, 50.0, 25.0]),
            "brain": torch.tensor([10.0, 150.0, 5.0]),
        }
        epsilon = 1.0

        tissue_to_logp = build_soft_prior_dict(counts_by_tissue, epsilon)

        # Check keys match
        assert set(tissue_to_logp.keys()) == {"blood", "brain"}

        # Check each is normalized
        for tissue, logp in tissue_to_logp.items():
            p = torch.exp(logp)
            assert torch.allclose(p.sum(), torch.tensor(1.0))

        # Check blood prior favors class 0
        assert tissue_to_logp["blood"][0] > tissue_to_logp["blood"][1]

        # Check brain prior favors class 1
        assert tissue_to_logp["brain"][1] > tissue_to_logp["brain"][0]


class TestEndToEnd:
    """End-to-end tests with realistic scenarios."""

    def test_allowlist_preserves_probabilities(self):
        """Test that allowlist doesn't break probability normalization."""
        B, C = 8, 20
        logits = torch.randn(B, C)
        allowed_classes = {0, 5, 10, 15}
        mask = build_allowed_mask_from_set(allowed_classes, C)

        logits_masked = apply_allowlist_mask(logits, mask)
        probs = torch.softmax(logits_masked, dim=-1)

        # Check normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))

        # Check forbidden classes have ~0 probability
        for c in range(C):
            if c not in allowed_classes:
                assert torch.all(probs[:, c] < 1e-8)

    def test_soft_prior_preserves_probabilities(self):
        """Test that soft prior doesn't break probability normalization."""
        B, C = 8, 20
        logits = torch.randn(B, C)
        counts = torch.tensor([float(i + 1) for i in range(C)])
        logp = counts_to_logp(counts, epsilon=1.0)

        logits_biased = add_soft_prior_bias(logits, logp, alpha=0.5)
        probs = torch.softmax(logits_biased, dim=-1)

        # Check normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))

    def test_combined_allowlist_and_prior(self):
        """Test applying both allowlist and soft prior."""
        B, C = 8, 20
        logits = torch.randn(B, C)

        # First apply allowlist
        allowed_classes = {0, 5, 10, 15}
        mask = build_allowed_mask_from_set(allowed_classes, C)
        logits_masked = apply_allowlist_mask(logits, mask)

        # Then apply soft prior (only on allowed classes)
        counts = torch.zeros(C)
        counts[list(allowed_classes)] = torch.tensor([100.0, 50.0, 200.0, 25.0])
        logp = counts_to_logp(counts, epsilon=1.0)
        logits_final = add_soft_prior_bias(logits_masked, logp, alpha=0.5)

        probs = torch.softmax(logits_final, dim=-1)

        # Check normalization
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B))

        # Check forbidden classes still have ~0 probability
        for c in range(C):
            if c not in allowed_classes:
                assert torch.all(probs[:, c] < 1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
