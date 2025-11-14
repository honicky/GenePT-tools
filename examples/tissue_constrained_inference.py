#!/usr/bin/env python3
"""
Tissue-Constrained Inference Example

Demonstrates how to apply tissue-aware constraints to cell type predictions.
Shows both allowlist (hard constraints) and soft prior (probabilistic bias) modes.
"""

import torch
import numpy as np
from pathlib import Path

from src.models.mlp_classifier import MLPClassifier
from src.inference.constraints import CellxGeneTissueConstraints


def load_model(checkpoint_path: str) -> MLPClassifier:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model config
    config = checkpoint.get('config', {})
    input_dim = checkpoint['model_state_dict']['model.0.weight'].shape[1]
    num_classes = checkpoint['model_state_dict'][list(checkpoint['model_state_dict'].keys())[-1]].shape[0]
    n_hidden_layers = config.get('n_hidden_layers', 4)
    dropout = config.get('dropout', 0.1)

    # Create model and load weights
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model: {input_dim}D -> {num_classes} classes")
    return model


def generate_sample_data(batch_size: int = 16, input_dim: int = 2048):
    """Generate sample embeddings for demonstration."""
    # In real usage, load actual cell embeddings
    embeddings = torch.randn(batch_size, input_dim)

    # Normalize to match training distribution
    embeddings = embeddings / embeddings.std(dim=0, keepdim=True).clamp(min=1e-6)

    return embeddings


def demo_baseline_inference(model, embeddings):
    """Baseline inference without constraints."""
    print("\n" + "="*80)
    print("BASELINE INFERENCE (No Constraints)")
    print("="*80)

    with torch.no_grad():
        logits = model(embeddings)
        probs = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    print(f"Logits shape: {logits.shape}")
    print(f"Top-5 predictions: {predictions[:5].tolist()}")
    print(f"Top prediction confidence: {probs[0, predictions[0]].item():.4f}")

    return logits, predictions


def demo_allowlist_mode(model, embeddings, tissue='UBERON:0000178'):
    """Demonstrate allowlist (hard constraint) mode."""
    print("\n" + "="*80)
    print(f"ALLOWLIST MODE (Hard Constraints) - Tissue: {tissue}")
    print("="*80)

    # Initialize constraints
    constraints = CellxGeneTissueConstraints(
        allowlist_path='data/cellxgene_constraints/tissue_allowlists.json',
        prior_path='data/cellxgene_constraints/tissue_class_logprobs.pt',
        num_classes=302  # Adjust to your model's num_classes
    )

    # Get tissue label
    tissue_label = constraints.get_tissue_label(tissue)
    print(f"Tissue: {tissue_label}")

    # Get baseline predictions
    with torch.no_grad():
        logits = model(embeddings)
        baseline_preds = logits.argmax(dim=-1)

    # Apply allowlist
    logits_constrained = constraints.apply_allowlist(logits, tissue)
    constrained_preds = logits_constrained.argmax(dim=-1)

    # Compare
    changed = (baseline_preds != constrained_preds).sum().item()
    print(f"\nPredictions changed: {changed}/{len(baseline_preds)}")
    print(f"Baseline predictions: {baseline_preds[:5].tolist()}")
    print(f"Allowlist predictions: {constrained_preds[:5].tolist()}")

    # Show probability mass before/after
    probs_baseline = torch.softmax(logits, dim=-1)
    probs_constrained = torch.softmax(logits_constrained, dim=-1)

    print(f"\nExample 0:")
    print(f"  Baseline top prediction: class {baseline_preds[0].item()} "
          f"(confidence: {probs_baseline[0, baseline_preds[0]].item():.4f})")
    print(f"  Allowlist top prediction: class {constrained_preds[0].item()} "
          f"(confidence: {probs_constrained[0, constrained_preds[0]].item():.4f})")

    return logits_constrained, constrained_preds


def demo_soft_prior_mode(model, embeddings, tissue='UBERON:0000178', alpha=0.5):
    """Demonstrate soft prior (probabilistic bias) mode."""
    print("\n" + "="*80)
    print(f"SOFT PRIOR MODE (Probabilistic Bias) - Tissue: {tissue}, alpha={alpha}")
    print("="*80)

    # Initialize constraints
    constraints = CellxGeneTissueConstraints(
        allowlist_path='data/cellxgene_constraints/tissue_allowlists.json',
        prior_path='data/cellxgene_constraints/tissue_class_logprobs.pt',
        num_classes=302
    )

    # Get baseline predictions
    with torch.no_grad():
        logits = model(embeddings)
        baseline_preds = logits.argmax(dim=-1)

    # Apply soft prior
    logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=alpha)
    biased_preds = logits_biased.argmax(dim=-1)

    # Compare
    changed = (baseline_preds != biased_preds).sum().item()
    print(f"\nPredictions changed: {changed}/{len(baseline_preds)}")
    print(f"Baseline predictions: {baseline_preds[:5].tolist()}")
    print(f"Soft prior predictions: {biased_preds[:5].tolist()}")

    # Show probability shifts
    probs_baseline = torch.softmax(logits, dim=-1)
    probs_biased = torch.softmax(logits_biased, dim=-1)

    print(f"\nExample 0 probability shift:")
    top5_baseline = torch.topk(probs_baseline[0], 5)
    top5_biased = torch.topk(probs_biased[0], 5)

    print("  Baseline top-5:")
    for i, (prob, idx) in enumerate(zip(top5_baseline.values, top5_baseline.indices)):
        print(f"    {i+1}. Class {idx.item()}: {prob.item():.4f}")

    print("  After soft prior top-5:")
    for i, (prob, idx) in enumerate(zip(top5_biased.values, top5_biased.indices)):
        print(f"    {i+1}. Class {idx.item()}: {prob.item():.4f}")

    return logits_biased, biased_preds


def demo_alpha_tuning(model, embeddings, tissue='UBERON:0000178'):
    """Demonstrate effect of different alpha values."""
    print("\n" + "="*80)
    print("ALPHA PARAMETER TUNING")
    print("="*80)

    constraints = CellxGeneTissueConstraints(
        allowlist_path='data/cellxgene_constraints/tissue_allowlists.json',
        prior_path='data/cellxgene_constraints/tissue_class_logprobs.pt',
        num_classes=302
    )

    with torch.no_grad():
        logits = model(embeddings)
        baseline_pred = logits[0].argmax().item()

    print(f"\nBaseline prediction for example 0: class {baseline_pred}")
    print("\nEffect of alpha on prediction:")

    for alpha in [0.0, 0.3, 0.5, 1.0, 2.0]:
        logits_biased = constraints.apply_soft_prior(logits, tissue, alpha=alpha)
        pred = logits_biased[0].argmax().item()
        prob = torch.softmax(logits_biased[0], dim=-1)[pred].item()

        changed = "←" if pred != baseline_pred else " "
        print(f"  alpha={alpha:.1f}: class {pred:3d} (conf: {prob:.4f}) {changed}")

    print("\nRecommendations:")
    print("  - alpha=0.0: No prior (baseline model)")
    print("  - alpha=0.3-0.5: Gentle bias (recommended)")
    print("  - alpha=1.0: Strong bias (prior weight = model weight)")
    print("  - alpha>1.0: Prior dominates (use with caution)")


def main():
    """Run all demonstrations."""
    print("="*80)
    print("Tissue-Constrained Inference Demonstration")
    print("="*80)

    # Configuration
    checkpoint_path = "checkpoints/best_model.pt"  # Update with actual path
    tissue = "UBERON:0000178"  # Blood

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\nWARNING: Checkpoint not found at {checkpoint_path}")
        print("Using random embeddings for demonstration only.")
        print("To use with a real model, provide a valid checkpoint path.\n")

        # For demo purposes, create a dummy model
        model = MLPClassifier(input_dim=2048, num_classes=302, n_hidden_layers=4, dropout=0.1)
        model.eval()
    else:
        model = load_model(checkpoint_path)

    # Generate sample data
    embeddings = generate_sample_data(batch_size=16, input_dim=2048)

    # Run demonstrations
    demo_baseline_inference(model, embeddings)
    demo_allowlist_mode(model, embeddings, tissue=tissue)
    demo_soft_prior_mode(model, embeddings, tissue=tissue, alpha=0.5)
    demo_alpha_tuning(model, embeddings, tissue=tissue)

    print("\n" + "="*80)
    print("Demonstration Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Load real cell embeddings and tissue labels")
    print("  2. Tune alpha on validation set")
    print("  3. Evaluate accuracy improvements per tissue")
    print("  4. Analyze rare cell type preservation")


if __name__ == "__main__":
    main()
