#!/usr/bin/env python3
"""
Evaluate constrained output on brain tissue test data.

Compares three modes:
1. Baseline (no constraints)
2. Allowlist (hard constraints)
3. Soft prior (probabilistic bias)

Uses macro-F1 and hierarchical metrics for evaluation.
"""

import argparse
from pathlib import Path
import sys
import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.mlp_classifier import MLPClassifier
from core.postprocessing import CellxGeneTissueConstraints
from data_loading.composable_dataset import ComposableTrainingDataset
from training.metrics import evaluate_with_hierarchy, inference_all
import numpy as np


def load_test_data_composable(
    base_data_dir: str,
    embedding_types: list,
    genept_dims: int,
    code_remapping: dict,
    cell_type_codes: dict,
    test_genept_suffix: str = '_test_v1_scgpt',
    test_tissue_suffix: str = '_test_v1_tissue',
    test_metadata_suffix: str = '_test_v1',
    seed: int = 42
):
    """Load test data using ComposableTrainingDataset (same as training)."""
    print(f"\nLoading test data using ComposableTrainingDataset...")
    print(f"  Base dir: {base_data_dir}")
    print(f"  Embedding types: {embedding_types}")
    print(f"  GenePT dims: {genept_dims}")

    # Create test dataset
    test_dataset = ComposableTrainingDataset(
        base_dir=Path(base_data_dir),
        embedding_types=embedding_types,
        batch_size=1024,
        genept_dims=genept_dims,
        code_remapping=code_remapping,
        track_invalid_embeddings=False,
        seed=seed,
        # Test mode parameters
        is_test_mode=True,
        test_genept_suffix=test_genept_suffix,
        test_tissue_suffix=test_tissue_suffix,
        test_metadata_suffix=test_metadata_suffix,
        cell_type_codes=cell_type_codes,
        verbose=False
    )

    print(f"  Loading {len(test_dataset.file_list)} test files...")

    # Load all test data
    X_list = []
    y_list = []

    for X_batch, y_batch in test_dataset:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    print(f"  Loaded {len(X)} samples (before filtering)")

    # CRITICAL: Apply the same filtering as training (trainer.py:351)
    # Filter to valid cell types: y >= 0 AND y < num_classes
    num_classes = len(cell_type_codes)
    valid_mask = (y >= 0) & (y < num_classes)

    total_samples = len(y)
    valid_samples = valid_mask.sum()
    filtered_samples = total_samples - valid_samples

    if filtered_samples > 0:
        print(f"  Filtering out {filtered_samples:,} samples ({filtered_samples/total_samples*100:.1f}%):")
        negative_y = (y < 0).sum()
        out_of_range = (y >= num_classes).sum()
        if negative_y > 0:
            print(f"    - {negative_y:,} with y < 0 (not in vocabulary or below threshold)")
        if out_of_range > 0:
            print(f"    - {out_of_range:,} with y >= {num_classes} (out of range)")

    # Apply filter
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"  Valid samples: {valid_samples:,} ({valid_samples/total_samples*100:.1f}%)")
    print(f"  Input dim: {X.shape[1]}")
    print(f"  Unique labels: {len(np.unique(y))}")

    return torch.tensor(X), torch.tensor(y)


def load_test_data(parquet_path: str, scgpt_path: str = None, genept_dims: int = 1536):
    """Load test data from parquet file.

    Args:
        parquet_path: Path to main test data file (contains GenePT embeddings)
        scgpt_path: Optional path to scGPT embeddings file
        genept_dims: Number of GenePT dimensions to use

    Returns:
        X: Embeddings tensor [N, D]
        y_true: True cell type labels (strings)
        tissue_ids: Tissue ontology IDs (strings)
    """
    print(f"\nLoading test data from {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} cells")

    # Extract GenePT embeddings (columns named as integers)
    embedding_cols = [str(i) for i in range(genept_dims)]
    X_genept = torch.tensor(df[embedding_cols].values, dtype=torch.float32)
    print(f"  GenePT embedding dim: {X_genept.shape[1]}")

    # Load scGPT embeddings if path provided
    if scgpt_path:
        print(f"  Loading scGPT embeddings from {scgpt_path}")
        df_scgpt = pd.read_parquet(scgpt_path)
        scgpt_cols = [c for c in df_scgpt.columns if c.startswith('emb_')]
        scgpt_cols = sorted(scgpt_cols, key=lambda x: int(x.split('_')[1]))
        X_scgpt = torch.tensor(df_scgpt[scgpt_cols].values, dtype=torch.float32)
        print(f"  scGPT embedding dim: {X_scgpt.shape[1]}")

        # Concatenate embeddings
        X = torch.cat([X_genept, X_scgpt], dim=1)
        print(f"  Combined embedding dim: {X.shape[1]}")
    else:
        X = X_genept

    # Extract labels
    y_true = df['cell_type'].values

    # Extract tissue IDs
    tissue_ids = df['tissue_ontology_term_id'].values

    print(f"  Unique cell types: {len(set(y_true))}")
    print(f"  Unique tissues: {set(tissue_ids)}")

    return X, y_true, tissue_ids


def load_model(checkpoint_path: str, num_classes: int, input_dim: int, device: str = "cpu"):
    """Load trained MLP model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}")

    # Try to load config file to get exact architecture
    config_path = checkpoint_path.replace('.pt', '_config.json')

    if Path(config_path).exists():
        print(f"  Loading config from {config_path}")
        with open(config_path) as f:
            import json
            config = json.load(f)
            n_hidden_layers = config.get('n_hidden_layers', 3)
            dropout = config.get('dropout', 0.05)
            print(f"  Config: n_hidden_layers={n_hidden_layers}, dropout={dropout}")

    # Load checkpoint to check input_dim
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Infer input_dim from first layer
    first_layer_key = 'model.0.weight'
    if first_layer_key in state_dict:
        actual_input_dim = state_dict[first_layer_key].shape[1]
        print(f"  Detected input_dim from checkpoint: {actual_input_dim}")
        input_dim = actual_input_dim

    # Create model with correct architecture
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        n_hidden_layers=n_hidden_layers,
        dropout=dropout
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"  Model loaded successfully")
    print(f"  Parameters: {model.count_parameters():,}")

    return model


def get_tissue_for_batch(tissue_ids):
    """
    Get the tissue for a batch.

    Assumes homogeneous batch (all cells same tissue).
    If mixed, uses the most common tissue.
    """
    unique_tissues = set(tissue_ids)

    if len(unique_tissues) == 1:
        return list(unique_tissues)[0]

    # If mixed, use most common
    from collections import Counter
    tissue_counts = Counter(tissue_ids)
    most_common = tissue_counts.most_common(1)[0][0]
    print(f"  Warning: Mixed tissues in batch. Using most common: {most_common}")
    return most_common


def run_inference(
    model,
    X,
    constraints=None,
    tissue=None,
    mode="baseline",
    alpha=0.5,
    batch_size=512,
    device="cpu"
):
    """
    Run inference with optional constraints.

    Args:
        model: Trained MLP model
        X: Input embeddings [N, input_dim]
        constraints: CellxGeneTissueConstraints instance (or None for baseline)
        tissue: Tissue ID for constraints
        mode: "baseline", "allowlist", or "soft_prior"
        alpha: Prior strength for soft_prior mode
        batch_size: Batch size for inference
        device: torch device

    Returns:
        predictions: Predicted class indices [N]
        probs: Predicted probabilities [N, num_classes]
    """
    print(f"\nRunning inference ({mode} mode)")

    all_probs = []
    num_batches = (len(X) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))

            batch = X[start_idx:end_idx].to(device)

            # Get logits from model
            logits = model(batch)

            # Apply constraints based on mode
            if mode == "allowlist" and constraints is not None:
                logits = constraints.apply_allowlist(logits, tissue)
            elif mode == "soft_prior" and constraints is not None:
                logits = constraints.apply_soft_prior(logits, tissue, alpha=alpha)
            # else: baseline, no constraints

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

    # Concatenate all batches
    all_probs = torch.cat(all_probs, dim=0)
    predictions = all_probs.argmax(dim=-1).numpy()

    print(f"  Completed {num_batches} batches")

    return predictions, all_probs.numpy()


def evaluate_predictions(y_true, y_pred, y_probs, class_names, mode_name, ontology=None):
    """Evaluate predictions and print comprehensive metrics.

    Args:
        y_true: True labels (numpy array of indices)
        y_pred: Predicted labels (numpy array of indices)
        y_probs: Prediction probabilities (numpy array of shape [N, num_classes])
        class_names: List of class names
        mode_name: Name of evaluation mode
        ontology: Optional CellOntology for hierarchical metrics
    """
    print(f"\n{'='*60}")
    print(f"Results for {mode_name}")
    print(f"{'='*60}")

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Macro metrics (equal weight per class)
    # IMPORTANT: Must specify labels=np.arange(len(class_names)) to include ALL classes,
    # not just those present in y_true. This matches the training evaluation.
    from sklearn.metrics import precision_recall_fscore_support
    num_classes = len(class_names)
    all_labels = np.arange(num_classes)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=all_labels, zero_division=0
    )
    print(f"\nMacro Metrics (equal weight per class):")
    print(f"  Precision: {macro_prec:.4f}")
    print(f"  Recall:    {macro_rec:.4f}")
    print(f"  F1:        {macro_f1:.4f}")

    # Weighted metrics (weighted by support)
    # Also specify labels to be consistent, though for 'weighted' it shouldn't matter
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=all_labels, zero_division=0
    )
    print(f"\nWeighted Metrics (weighted by support):")
    print(f"  Precision: {weighted_prec:.4f}")
    print(f"  Recall:    {weighted_rec:.4f}")
    print(f"  F1:        {weighted_f1:.4f}")

    # Recall@k metrics
    print(f"\nRecall@k Metrics:")
    for k in [1, 5, 10, 20]:
        # Get top-k predictions
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        # Check if true label is in top-k
        recall_at_k = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        print(f"  Recall@{k:2d}: {recall_at_k:.4f}")

    # Hierarchical metrics (if ontology provided)
    hier_prec, hier_rec, hier_f1 = None, None, None
    if ontology is not None:
        try:
            from training.hierarchical_metrics import calculate_hierarchical_f_score
            # Convert indices to labels
            y_true_labels = [class_names[i] for i in y_true]
            y_pred_labels = [class_names[i] for i in y_pred]

            hier_metrics = calculate_hierarchical_f_score(
                y_true_labels, y_pred_labels, ontology.graph
            )
            hier_prec = hier_metrics['hierarchical_precision']
            hier_rec = hier_metrics['hierarchical_recall']
            hier_f1 = hier_metrics['hierarchical_f1']

            print(f"\nHierarchical Metrics:")
            print(f"  Hierarchical Precision: {hier_prec:.4f}")
            print(f"  Hierarchical Recall:    {hier_rec:.4f}")
            print(f"  Hierarchical F1:        {hier_f1:.4f}")
        except Exception as e:
            print(f"\nWarning: Could not compute hierarchical metrics: {e}")
            import traceback
            traceback.print_exc()

    # Per-class report (only for classes present in test data)
    print("\nPer-class metrics:")
    unique_labels = sorted(set(y_true))
    report = classification_report(
        y_true, y_pred, labels=unique_labels,
        target_names=[class_names[i] if i < len(class_names) else f"class_{i}" for i in unique_labels],
        zero_division=0
    )
    print(report)

    return {
        'accuracy': acc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate constrained output on test data")
    parser.add_argument(
        "--test-data",
        type=str,
        default="/mmc-scratch/scratch/cellxgene_v2_test_v1/dc30c3ec-46d6-4cd8-8ec1-b544a3d0f503.parquet",
        help="Path to test parquet file"
    )
    parser.add_argument(
        "--scgpt-data",
        type=str,
        default=None,
        help="Path to scGPT embeddings file (optional)"
    )
    parser.add_argument(
        "--genept-dims",
        type=int,
        default=1536,
        help="Number of GenePT dimensions to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        required=True,
        help="Path to class labels file (mapping indices to class names)"
    )
    parser.add_argument(
        "--constraints-dir",
        type=str,
        default="/data/GenePT-tools/data/cellxgene_constraints",
        help="Directory containing tissue constraints"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Prior strength for soft_prior mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference"
    )
    args = parser.parse_args()

    # Load class label mapping first (needed for data loading)
    print(f"\nLoading class labels from {args.class_labels}")

    # Check if it's CSV or JSON
    class_idx_mapping = None
    if args.class_labels.endswith('.csv'):
        # Read CSV file
        df_labels = pd.read_csv(args.class_labels)
        # Sort by filtered_code to ensure correct order
        df_labels = df_labels.sort_values('filtered_code')
        class_names = df_labels['cell_type'].tolist()
        idx_to_class = {str(i): name for i, name in enumerate(class_names)}

        # Get mapping from filtered indices to original ontology indices
        if 'original_code' in df_labels.columns:
            class_idx_mapping = df_labels['original_code'].tolist()
            print(f"  Loaded class index mapping (filtered -> original ontology)")
    else:
        # Read JSON file
        with open(args.class_labels) as f:
            import json
            class_labels = json.load(f)

        if isinstance(class_labels, dict):
            # If it's a dict, get the ordered list
            idx_to_class = class_labels
            class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
        else:
            # If it's a list
            class_names = class_labels
            idx_to_class = {str(i): name for i, name in enumerate(class_names)}

    num_classes = len(class_names)
    print(f"  Loaded {num_classes} class labels")

    # Create cell_type_codes and code_remapping from the mapping file
    cell_type_codes = {name: i for i, name in enumerate(class_names)}
    code_remapping = None  # Will be built by dataset if needed
    if class_idx_mapping:
        # code_remapping maps ALL original codes (0-548) to either:
        # - filtered code (0-301) for cell types >= threshold
        # - -100 for cell types < threshold
        # Build complete mapping from cell_counts.csv
        cell_counts_file = "/data/batch-jobs/cell_counts.csv"
        if os.path.exists(cell_counts_file):
            cell_counts_df = pd.read_csv(cell_counts_file)

            # Get the threshold from the mapping CSV if available
            if 'cell_count' in df_labels.columns:
                threshold = df_labels['cell_count'].min()
            else:
                threshold = 10000  # Default from config

            # Create mapping for ALL cell types
            code_remapping = {}
            for idx, row in cell_counts_df.iterrows():
                orig_code = idx  # Row index is the original code
                if row['cell_count'] >= threshold:
                    # Find this cell type in the filtered mapping
                    try:
                        filt_code = class_names.index(row['cell_type'])
                        code_remapping[orig_code] = filt_code
                    except ValueError:
                        # Cell type not in filtered list, map to -100
                        code_remapping[orig_code] = -100
                else:
                    # Below threshold, map to -100
                    code_remapping[orig_code] = -100

            print(f"  Created code_remapping for {len(code_remapping)} cell types")
            print(f"    - {sum(1 for v in code_remapping.values() if v >= 0)} mapped to filtered codes")
            print(f"    - {sum(1 for v in code_remapping.values() if v == -100)} mapped to -100 (below threshold)")
        else:
            # Fallback: use simple mapping (only for filtered types)
            code_remapping = {orig: filt for filt, orig in enumerate(class_idx_mapping)}

    # Load test data using the same method as training
    X, y_true = load_test_data_composable(
        base_data_dir="/localdata/training_data",
        embedding_types=["genept", "scgpt", "metadata"],
        genept_dims=args.genept_dims,
        code_remapping=code_remapping,
        cell_type_codes=cell_type_codes,
        test_genept_suffix="_test_v1_scgpt",
        test_tissue_suffix="_test_v1_tissue",
        test_metadata_suffix="_test_v1",
        seed=4201
    )

    y_true = y_true.numpy()  # Convert to numpy for metrics

    # Load model
    input_dim = X.shape[1]
    model = load_model(args.checkpoint, num_classes, input_dim, device=args.device)

    # Load Cell Ontology for hierarchical metrics
    ontology_graph = None
    try:
        from training.ontology import CellOntologyManager
        ontology_cache_dir = Path("/data/GenePT-tools/data/ontology")
        ontology_manager = CellOntologyManager(cache_dir=ontology_cache_dir)
        ontology_graph = ontology_manager.build_cell_type_graph()
        print(f"\nLoaded Cell Ontology with {len(ontology_graph.nodes())} cell types")
    except Exception as e:
        print(f"\nWarning: Could not load Cell Ontology: {e}")
        import traceback
        traceback.print_exc()

    # Skip constraints for now (would need tissue information)
    print("\nSkipping constrained modes (no tissue information in dataset)")
    constraints = None
    run_constrained = False

    # Run evaluations using the SAME function as training
    print("\n" + "="*60)
    print("Running Evaluation (Using Training's evaluate_with_hierarchy)")
    print("="*60)

    # Convert to numpy for evaluation
    X_np = X.numpy() if torch.is_tensor(X) else X
    y_np = y_true if isinstance(y_true, np.ndarray) else y_true.numpy()

    # Create cell type to index mapping
    cell_type_to_idx = {ct: i for i, ct in enumerate(class_names)}

    # Use training's batch size for consistency
    batch_size = 4096  # Same as training
    device_obj = torch.device(args.device)

    # Run evaluation with hierarchy (exact same as training)
    metrics, _, _, _ = evaluate_with_hierarchy(
        model=model,
        X=X_np,
        y=y_np,
        cell_types=class_names,
        cell_type_to_idx=cell_type_to_idx,
        ontology_graph=ontology_graph,
        batch_size=batch_size,
        device=device_obj
    )

    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Samples: {len(y_np)}")
    print(f"  Batch size: {batch_size}")
    print(f"\nMetrics:")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    results = {'baseline': metrics}

    if run_constrained:
        # 2. Allowlist (hard constraints)
        y_pred_allowlist, probs_allowlist = run_inference(
            model, X, constraints=constraints, tissue=tissue,
            mode="allowlist", batch_size=args.batch_size, device=args.device
        )
        results['allowlist'] = evaluate_predictions(
            y_true, y_pred_allowlist, probs_allowlist, class_names,
            "Allowlist (Hard Constraints)", ontology=ontology
        )

        # 3. Soft prior (probabilistic bias)
        y_pred_soft_prior, probs_soft_prior = run_inference(
            model, X, constraints=constraints, tissue=tissue,
            mode="soft_prior", alpha=args.alpha,
            batch_size=args.batch_size, device=args.device
        )
        results['soft_prior'] = evaluate_predictions(
            y_true, y_pred_soft_prior, probs_soft_prior, class_names,
            f"Soft Prior (alpha={args.alpha})", ontology=ontology
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
