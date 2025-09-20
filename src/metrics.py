"""
Evaluation metrics for cell type classification.
Includes standard metrics and hierarchical metrics for cell type ontologies.
"""

import networkx as nx
import numpy as np
import obonet
import pandas as pd
import requests
from typing import Dict, List, Set, Tuple, Optional, Union
from sklearn.metrics import (
    log_loss, f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix
)
import warnings


# Suppress sklearn warnings for rare cell types
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')


def mrr_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """
    Calculate Mean Reciprocal Rank at k.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for each class
        k: Number of top predictions to consider
    
    Returns:
        Mean reciprocal rank score
    """
    topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
    rr = []
    for i in range(len(y_true)):
        if y_true[i] in topk_preds[i]:
            rank = np.where(topk_preds[i] == y_true[i])[0][0]
            rr.append(1.0 / (rank + 1))
        else:
            rr.append(0.0)
    return np.mean(rr)


def dcg_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for each class
        k: Number of top predictions to consider
    
    Returns:
        Mean DCG score
    """
    topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
    dcg = []
    for i in range(len(y_true)):
        if y_true[i] in topk_preds[i]:
            rank = np.where(topk_preds[i] == y_true[i])[0][0]
            dcg.append(1.0 / np.log2(rank + 2))
        else:
            dcg.append(0.0)
    return np.mean(dcg)


def recall_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """
    Calculate Recall at k (also known as top-k accuracy).
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities for each class
        k: Number of top predictions to consider
    
    Returns:
        Recall at k score
    """
    topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
    hits = [y_true[i] in topk_preds[i] for i in range(len(y_true))]
    return np.mean(hits)


def load_cell_ontology(data_dir):
    cell_ontology_path = data_dir / "cl.obo"
    if not cell_ontology_path.exists():
        print("Downloading Cell Ontology...")
        url = "http://purl.obolibrary.org/obo/cl.obo"
        response = requests.get(url)
        with open(cell_ontology_path, 'wb') as f:
            f.write(response.content)
        print("Downloaded!")

    # Load the ontology
    ontology = obonet.read_obo(cell_ontology_path)
    print(f"Loaded Cell Ontology with {len(ontology)} terms")
    return ontology


def build_cell_type_graph(ontology, cell_types):
    """
    Build a graph of cell type relationships from the Cell Ontology.
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Map cell type names to ontology IDs (this is simplified - real implementation would need proper mapping)
    # For demonstration, we'll create a simple hierarchy
    for node_id, node_data in ontology.nodes(data=True):
        if 'name' in node_data:
            G.add_node(node_data['name'])
    
    # Add edges based on is_a relationships
    for node_id, node_data in ontology.nodes(data=True):
        if 'name' in node_data and 'is_a' in node_data:
            child_name = node_data['name']
            for parent_id in node_data['is_a']:
                if parent_id in ontology:
                    parent_name = ontology.nodes[parent_id].get('name')
                    if parent_name:
                        G.add_edge(parent_name, child_name)
    
    return G

def get_ancestors(node: str, graph: nx.DiGraph) -> Set[str]:
    """Get all ancestors of a node including the node itself"""
    ancestors = {node}
    if node in graph:
        ancestors.update(nx.ancestors(graph, node))
    return ancestors

def calculate_hierarchical_f_score(y_true_labels: List[str], 
                                 y_pred_labels: List[str], 
                                 ontology_graph: nx.DiGraph) -> Dict[str, float]:
    """
    Calculate hierarchical precision, recall, and F-score.
    Based on Kiritchenko et al. (2005).
    """
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

def evaluate_stardard_metrics(model, X, y, batch_size=1024):
    """Standard evaluation from cellxgene_v2_mlp.ipynb"""
    y = np.asarray(y)
    model.eval()
    all_preds = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            preds = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    
    all_preds = np.concatenate(all_preds, axis=0)
    y_pred = all_preds.argmax(axis=1)
    
    num_classes = all_preds.shape[1]
    logloss = log_loss(y, all_preds, labels=np.arange(num_classes))
    macro_f1 = f1_score(y, y_pred, average='macro', labels=np.arange(num_classes))
    macro_precision = precision_score(y, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)
    macro_recall = recall_score(y, y_pred, average='macro', labels=np.arange(num_classes), zero_division=0)

    metrics = {
        "logloss": logloss,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }

    for k in [2, 5, 10]:
        metrics[f"recall_at_{k}"] = recall_at_k(y, all_preds, k)
        metrics[f"mrr_at_{k}"] = mrr_at_k(y, all_preds, k)
        metrics[f"dcg_at_{k}"] = dcg_at_k(y, all_preds, k)

    return metrics, y, all_preds, y_pred

def evaluate_with_hierarchy(model, X, y, cell_types, full_count_codes_pdf, ontology_graph, batch_size=1024):
    """
    Evaluate model with both standard and hierarchical metrics.
    """
    # Get standard metrics
    standard_metrics, y_true, all_preds, y_pred = evaluate_stardard_metrics(model, X, y, batch_size)
    
    # Convert indices to cell type labels
    # Note: y_true and y_pred are indices into full_count_codes_pdf, not 'code' values
    # We need to use .iloc to access by position
    try:
        y_true_labels = [full_count_codes_pdf.iloc[idx]['cell_type'] for idx in y_true]
        y_pred_labels = [full_count_codes_pdf.iloc[idx]['cell_type'] for idx in y_pred]
    except IndexError as e:
        print(f"IndexError occurred: {e}")
        print(f"full_count_codes_pdf shape: {full_count_codes_pdf.shape}")
        print(f"Max index in y_true: {max(y_true) if len(y_true) > 0 else 'N/A'}")
        print(f"Max index in y_pred: {max(y_pred) if len(y_pred) > 0 else 'N/A'}")
        print("First few values in y_true:", y_true[:10])
        print("First few values in y_pred:", y_pred[:10])
        raise
    
    # Calculate hierarchical metrics
    hierarchical_metrics = calculate_hierarchical_f_score(y_true_labels, y_pred_labels, ontology_graph)
    
    # Combine all metrics
    all_metrics = {**standard_metrics, **hierarchical_metrics}
    
    return all_metrics, y_true, all_preds, y_pred