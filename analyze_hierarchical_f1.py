#!/usr/bin/env python
"""
Analyze hierarchical F1 score for cell type predictions.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import networkx as nx
from typing import List

# Import functions from src/metrics.py
from src.metrics import (
    load_cell_ontology, 
    build_cell_type_graph,
    calculate_hierarchical_f_score
)

def main():
    # File paths
    h5ad_path = Path("/Users/rj/Downloads/32e8a3d7-7b15-4f80-a0ff-6d2fc531e972_subset_test_prediction.h5ad")
    data_dir = Path("/Users/rj/personal/GenePT-tools/data")
    
    print("=" * 60)
    print("Hierarchical F1 Score Analysis")
    print("=" * 60)
    
    # Step 1: Load the h5ad file
    print("\n1. Loading h5ad file...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"   Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Check available columns
    print("\n   Available metadata columns:")
    for col in adata.obs.columns:
        print(f"   - {col}")
    
    # Extract ground truth and predictions
    if 'cell_type' not in adata.obs.columns:
        raise ValueError("'cell_type' column not found in adata.obs")
    if 'cell_type_cdiam_miratyper_v1' not in adata.obs.columns:
        raise ValueError("'cell_type_cdiam_miratyper_v1' column not found in adata.obs")
    
    y_true_labels = adata.obs['cell_type'].tolist()
    y_pred_labels = adata.obs['cell_type_cdiam_miratyper_v1'].tolist()
    
    print(f"\n   Ground truth cell types: {len(set(y_true_labels))} unique types")
    print(f"   Predicted cell types: {len(set(y_pred_labels))} unique types")
    
    # Show sample of predictions
    print("\n   Sample predictions (first 5):")
    for i in range(min(5, len(y_true_labels))):
        print(f"   True: {y_true_labels[i][:40]:40s} | Pred: {y_pred_labels[i][:40]}")
    
    # Step 2: Load Cell Ontology
    print("\n2. Loading Cell Ontology...")
    ontology = load_cell_ontology(data_dir)
    
    # Step 3: Build cell type graph
    print("\n3. Building cell type hierarchy graph...")
    all_cell_types = list(set(y_true_labels + y_pred_labels))
    ontology_graph = build_cell_type_graph(ontology, all_cell_types)
    print(f"   Graph has {len(ontology_graph.nodes)} nodes and {len(ontology_graph.edges)} edges")
    
    # Step 4: Calculate hierarchical metrics
    print("\n4. Calculating hierarchical F1 score...")
    hierarchical_metrics = calculate_hierarchical_f_score(
        y_true_labels, 
        y_pred_labels, 
        ontology_graph
    )
    
    # Step 5: Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nHierarchical Precision: {hierarchical_metrics['hierarchical_precision']:.4f}")
    print(f"Hierarchical Recall:    {hierarchical_metrics['hierarchical_recall']:.4f}")
    print(f"Hierarchical F1 Score:  {hierarchical_metrics['hierarchical_f1']:.4f}")
    
    # Additional analysis: exact match accuracy
    exact_matches = sum(1 for true, pred in zip(y_true_labels, y_pred_labels) if true == pred)
    accuracy = exact_matches / len(y_true_labels)
    print(f"\nExact Match Accuracy:   {accuracy:.4f}")
    print(f"Total cells evaluated:  {len(y_true_labels)}")
    
    # Analysis of cell types not in ontology
    print("\n" + "-" * 60)
    print("Cell Type Coverage Analysis:")
    
    true_types_in_graph = sum(1 for label in set(y_true_labels) if label in ontology_graph)
    pred_types_in_graph = sum(1 for label in set(y_pred_labels) if label in ontology_graph)
    
    print(f"Ground truth types in ontology: {true_types_in_graph}/{len(set(y_true_labels))}")
    print(f"Predicted types in ontology:    {pred_types_in_graph}/{len(set(y_pred_labels))}")
    
    # Show types not in ontology (if any)
    true_not_in_ontology = [t for t in set(y_true_labels) if t not in ontology_graph]
    if true_not_in_ontology:
        print(f"\nGround truth types NOT in ontology ({len(true_not_in_ontology)}):")
        for t in true_not_in_ontology[:5]:
            print(f"  - {t}")
        if len(true_not_in_ontology) > 5:
            print(f"  ... and {len(true_not_in_ontology) - 5} more")
    
    pred_not_in_ontology = [t for t in set(y_pred_labels) if t not in ontology_graph]
    if pred_not_in_ontology:
        print(f"\nPredicted types NOT in ontology ({len(pred_not_in_ontology)}):")
        for t in pred_not_in_ontology[:5]:
            print(f"  - {t}")
        if len(pred_not_in_ontology) > 5:
            print(f"  ... and {len(pred_not_in_ontology) - 5} more")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    
    return hierarchical_metrics

if __name__ == "__main__":
    metrics = main()