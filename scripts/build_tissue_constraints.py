#!/usr/bin/env python3
"""
Build tissue allowlists and soft priors from CellxGene co-occurrence counts.

Takes tissue_celltype_counts.pt and generates:
1. tissue_allowlists.json - Hard constraints (min_count threshold)
2. tissue_class_logprobs.pt - Soft priors (Laplace smoothing)
"""

import argparse
import json
from pathlib import Path

import torch

from constrained_output import build_allowlist_dict, build_soft_prior_dict


def main():
    parser = argparse.ArgumentParser(
        description="Build tissue constraints from CellxGene co-occurrence counts"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/GenePT-tools/data/cellxgene_constraints/tissue_celltype_counts.pt"),
        help="Path to tissue_celltype_counts.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/GenePT-tools/data/cellxgene_constraints"),
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum cell count to include cell type in allowlist"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Laplace smoothing constant for soft priors"
    )
    args = parser.parse_args()

    # Load co-occurrence counts
    print(f"Loading counts from {args.input}")
    data = torch.load(args.input, map_location="cpu")

    tissue_vocab = data["tissue_vocab"]
    tissue_id_to_label = data["tissue_id_to_label"]
    celltype_vocab = data["celltype_vocab"]
    tissue_to_counts = data["tissue_to_counts"]  # Dict[tissue_id, Tensor[C]]

    print(f"  Tissues: {len(tissue_vocab)}")
    print(f"  Cell types: {len(celltype_vocab)}")
    print(f"  Total cells: {sum(counts.sum().item() for counts in tissue_to_counts.values()):,}")

    # Build allowlists
    print(f"\nBuilding allowlists (min_count={args.min_count})")
    allowlists = build_allowlist_dict(tissue_to_counts, min_count=args.min_count)

    # Show statistics
    avg_allowed = sum(len(v) for v in allowlists.values()) / len(allowlists)
    print(f"  Average allowed cell types per tissue: {avg_allowed:.1f}")
    print(f"  Min allowed: {min(len(v) for v in allowlists.values())}")
    print(f"  Max allowed: {max(len(v) for v in allowlists.values())}")

    # Build soft priors
    print(f"\nBuilding soft priors (epsilon={args.epsilon})")
    tissue_to_logp = build_soft_prior_dict(tissue_to_counts, epsilon=args.epsilon)

    # Save allowlists as JSON
    allowlist_path = args.output_dir / "tissue_allowlists.json"
    print(f"\nSaving allowlists to {allowlist_path}")
    with open(allowlist_path, "w") as f:
        json.dump(allowlists, f, indent=2)

    # Save soft priors as PyTorch file
    prior_path = args.output_dir / "tissue_class_logprobs.pt"
    print(f"Saving soft priors to {prior_path}")
    torch.save({
        "tissue_to_logp": tissue_to_logp,
        "tissue_vocab": tissue_vocab,
        "tissue_id_to_label": tissue_id_to_label,
        "celltype_vocab": celltype_vocab,
        "metadata": {
            "min_count": args.min_count,
            "epsilon": args.epsilon,
            "num_tissues": len(tissue_vocab),
            "num_celltypes": len(celltype_vocab),
        }
    }, prior_path)

    # Save metadata JSON for human readability
    metadata_path = args.output_dir / "metadata.json"
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump({
            "num_tissues": len(tissue_vocab),
            "num_celltypes": len(celltype_vocab),
            "tissue_vocab": tissue_vocab,
            "tissue_id_to_label": tissue_id_to_label,
            "celltype_vocab": celltype_vocab[:100],  # First 100 for readability
            "constraint_params": {
                "min_count": args.min_count,
                "epsilon": args.epsilon,
            },
            "sample_allowlists": {
                tissue_id: allowlists[tissue_id][:10]  # First 10 allowed classes
                for tissue_id in list(tissue_vocab)[:5]  # First 5 tissues
            }
        }, f, indent=2)

    print("\nDone!")
    print(f"  {allowlist_path.name}")
    print(f"  {prior_path.name}")
    print(f"  {metadata_path.name}")


if __name__ == "__main__":
    main()
