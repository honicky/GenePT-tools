"""Hierarchical evaluation metrics using Cell Ontology.

This module provides pure functions for calculating hierarchical precision,
recall, and F1 scores based on ontological relationships between cell types.
"""

import networkx as nx
from typing import Set, List, Dict


def get_ancestors(node: str, graph: nx.DiGraph) -> Set[str]:
    """Get all ancestors of a node including the node itself.
    
    This is a pure function that returns the transitive closure of ancestors
    for a given node in a directed graph.
    
    Args:
        node: The node to find ancestors for
        graph: Directed graph representing the ontology
        
    Returns:
        Set of all ancestors including the node itself
    """
    ancestors = {node}
    if node in graph:
        ancestors.update(nx.ancestors(graph, node))
    return ancestors


def calculate_hierarchical_f_score(
    y_true_labels: List[str], 
    y_pred_labels: List[str], 
    ontology_graph: nx.DiGraph
) -> Dict[str, float]:
    """Calculate hierarchical precision, recall, and F-score.
    
    Based on Kiritchenko et al. (2005), this function computes hierarchical
    metrics by considering the overlap of ancestor sets in the ontology.
    
    This is a pure function with no side effects.
    
    Args:
        y_true_labels: List of true cell type labels
        y_pred_labels: List of predicted cell type labels
        ontology_graph: Directed graph representing the cell type ontology
        
    Returns:
        Dictionary with keys:
        - hierarchical_precision: Fraction of predicted ancestors that are correct
        - hierarchical_recall: Fraction of true ancestors that were predicted
        - hierarchical_f1: Harmonic mean of precision and recall
    """
    # Handle empty inputs
    if not y_true_labels or not y_pred_labels:
        return {
            "hierarchical_precision": 0.0,
            "hierarchical_recall": 0.0,
            "hierarchical_f1": 0.0
        }
    
    total_true_ancestors = 0
    total_pred_ancestors = 0
    total_intersection = 0
    
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        true_ancestors = get_ancestors(true_label, ontology_graph)
        pred_ancestors = get_ancestors(pred_label, ontology_graph)
        
        intersection = len(true_ancestors & pred_ancestors)
        
        total_true_ancestors += len(true_ancestors)
        total_pred_ancestors += len(pred_ancestors)
        total_intersection += intersection
    
    # Calculate hierarchical precision and recall
    h_precision = total_intersection / total_pred_ancestors if total_pred_ancestors > 0 else 0
    h_recall = total_intersection / total_true_ancestors if total_true_ancestors > 0 else 0
    
    # Calculate F-score
    if h_precision + h_recall == 0:
        h_f1 = 0
    else:
        h_f1 = 2 * (h_precision * h_recall) / (h_precision + h_recall)
    
    return {
        "hierarchical_precision": h_precision,
        "hierarchical_recall": h_recall,
        "hierarchical_f1": h_f1
    }