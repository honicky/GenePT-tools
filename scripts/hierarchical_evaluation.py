#!/usr/bin/env python3
"""
Hierarchical evaluation of cell type predictions using OnClass and existing metrics.
Based on cellxgene_v2_mlp.ipynb evaluation functions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import OnClass - we'll install it if needed
try:
    import OnClass
except ImportError:
    print("OnClass not found. Please install with: pip install onclass")
    print("Or follow instructions at: https://github.com/wangshenguiuc/OnClass")


class CellTypeEvaluator:
    """Evaluator for cell type predictions with hierarchical metrics."""
    
    def __init__(self, cell_types: List[str], cell_ontology_file: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            cell_types: List of all possible cell types
            cell_ontology_file: Path to Cell Ontology OBO file (optional)
        """
        self.cell_types = cell_types
        self.num_classes = len(cell_types)
        self.cell_ontology_file = cell_ontology_file
        
    def mrr_at_k(self, y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
        """Mean Reciprocal Rank at k."""
        topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
        rr = []
        for i in range(len(y_true)):
            if y_true[i] in topk_preds[i]:
                rank = np.where(topk_preds[i] == y_true[i])[0][0]
                rr.append(1.0 / (rank + 1))
            else:
                rr.append(0.0)
        return np.mean(rr)
    
    def dcg_at_k(self, y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
        """Discounted Cumulative Gain at k."""
        topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
        dcg = []
        for i in range(len(y_true)):
            if y_true[i] in topk_preds[i]:
                rank = np.where(topk_preds[i] == y_true[i])[0][0]
                dcg.append(1.0 / np.log2(rank + 2))
            else:
                dcg.append(0.0)
        return np.mean(dcg)
    
    def recall_at_k(self, y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
        """Recall at k."""
        topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
        hits = [y_true[i] in topk_preds[i] for i in range(len(y_true))]
        return np.mean(hits)
    
    def evaluate_standard_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_probs: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate standard (non-hierarchical) metrics.
        
        Args:
            y_true: True labels (as indices)
            y_pred_probs: Predicted probabilities for each class
            y_pred: Predicted labels (optional, will be computed if not provided)
            
        Returns:
            Dictionary of metric names to values
        """
        if y_pred is None:
            y_pred = y_pred_probs.argmax(axis=1)
            
        # Ensure y_true is numpy array
        y_true = np.asarray(y_true)
        
        # Standard metrics
        logloss = log_loss(y_true, y_pred_probs, labels=np.arange(self.num_classes))
        macro_f1 = f1_score(y_true, y_pred, average='macro', labels=np.arange(self.num_classes))
        macro_precision = precision_score(
            y_true, y_pred, average='macro', 
            labels=np.arange(self.num_classes), zero_division=0
        )
        macro_recall = recall_score(
            y_true, y_pred, average='macro', 
            labels=np.arange(self.num_classes), zero_division=0
        )
        
        metrics = {
            "logloss": logloss,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall
        }
        
        # Ranking metrics
        for k in [2, 5, 10]:
            metrics[f"recall_at_{k}"] = self.recall_at_k(y_true, y_pred_probs, k)
            metrics[f"mrr_at_{k}"] = self.mrr_at_k(y_true, y_pred_probs, k)
            metrics[f"dcg_at_{k}"] = self.dcg_at_k(y_true, y_pred_probs, k)
        
        return metrics
    
    def evaluate_hierarchical_metrics(
        self,
        y_true_labels: List[str],
        y_pred_labels: List[str],
        y_pred_probs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate hierarchical metrics using OnClass.
        
        Args:
            y_true_labels: True cell type labels (as strings)
            y_pred_labels: Predicted cell type labels (as strings)
            y_pred_probs: Predicted probabilities (optional, for AUROC)
            
        Returns:
            Dictionary of hierarchical metric names to values
        """
        hierarchical_metrics = {}
        
        try:
            # Import OnClass evaluation functions
            from OnClass.OnClassUtils import evaluate_predictions
            
            # Basic hierarchical precision, recall, F1
            # Note: OnClass expects specific format, may need adjustment
            h_results = evaluate_predictions(
                y_true_labels,
                y_pred_labels,
                ontology_file=self.cell_ontology_file
            )
            
            hierarchical_metrics.update({
                "hierarchical_precision": h_results.get('precision', 0.0),
                "hierarchical_recall": h_results.get('recall', 0.0),
                "hierarchical_f1": h_results.get('f1', 0.0),
            })
            
        except Exception as e:
            print(f"OnClass evaluation failed: {e}")
            print("Falling back to basic hierarchical metrics")
            
            # Fallback: Simple exact match accuracy
            hierarchical_metrics["exact_match_accuracy"] = np.mean([
                yt == yp for yt, yp in zip(y_true_labels, y_pred_labels)
            ])
        
        return hierarchical_metrics
    
    def evaluate_with_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1024,
        device: Optional[torch.device] = None
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a PyTorch model.
        
        Args:
            model: PyTorch model
            X: Input features
            y: True labels (as indices)
            batch_size: Batch size for evaluation
            device: Device to run on
            
        Returns:
            Tuple of (metrics_dict, y_true, y_pred_probs, y_pred)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        y = np.asarray(y)
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
                logits = model(xb)
                preds = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                
        all_preds = np.concatenate(all_preds, axis=0)
        y_pred = all_preds.argmax(axis=1)
        
        # Get standard metrics
        metrics = self.evaluate_standard_metrics(y, all_preds, y_pred)
        
        # Get hierarchical metrics if we have cell type mappings
        if hasattr(self, 'index_to_celltype'):
            y_true_labels = [self.index_to_celltype[idx] for idx in y]
            y_pred_labels = [self.index_to_celltype[idx] for idx in y_pred]
            h_metrics = self.evaluate_hierarchical_metrics(
                y_true_labels, y_pred_labels, all_preds
            )
            metrics.update(h_metrics)
        
        return metrics, y, all_preds, y_pred


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    cell_types: List[str]
) -> pd.DataFrame:
    """
    Create normalized confusion matrix from predictions.
    
    Args:
        y_true: True labels (as indices)
        y_pred_probs: Predicted probabilities
        cell_types: List of cell type names
        
    Returns:
        Confusion matrix as DataFrame
    """
    n_classes = y_pred_probs.shape[1]
    densities = np.zeros([n_classes, n_classes], dtype=np.float32)
    
    # Accumulate predictions
    for i in range(y_pred_probs.shape[0]):
        densities[y_true[i]] += y_pred_probs[i]
    
    # Normalize by class counts
    for label, count in zip(*np.unique(y_true, return_counts=True)):
        if count > 0:
            densities[label] /= count
    
    # Create DataFrame with cell type labels
    confusion_df = pd.DataFrame(
        densities,
        index=cell_types[:n_classes],
        columns=cell_types[:n_classes]
    )
    
    return confusion_df


# Example usage
if __name__ == "__main__":
    print("Cell Type Hierarchical Evaluation Script")
    print("========================================")
    print()
    print("This script provides functions for evaluating cell type predictions")
    print("with both standard and hierarchical metrics.")
    print()
    print("Key functions:")
    print("- CellTypeEvaluator: Main evaluation class")
    print("- evaluate_standard_metrics: Log loss, F1, precision, recall, ranking metrics")
    print("- evaluate_hierarchical_metrics: OnClass-based hierarchical metrics")
    print("- create_confusion_matrix: Generate normalized confusion matrix")
    print()
    print("To use with your model:")
    print("1. Create evaluator: evaluator = CellTypeEvaluator(cell_types)")
    print("2. Evaluate: metrics, y, probs, preds = evaluator.evaluate_with_model(model, X, y)")
    print("3. Get confusion matrix: cm = create_confusion_matrix(y, probs, cell_types)")