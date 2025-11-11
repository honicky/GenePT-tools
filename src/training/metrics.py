"""Evaluation metrics for CellXGene MLP training."""

import numpy as np
import torch
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from typing import Dict, Tuple, Optional, List
import networkx as nx

from .hierarchical_metrics import calculate_hierarchical_f_score


def mrr_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
  """Calculate Mean Reciprocal Rank at k.
  
  Args:
    y_true: True labels (shape: [n_samples])
    y_pred_probs: Predicted probabilities (shape: [n_samples, n_classes])
    k: Number of top predictions to consider
    
  Returns:
    MRR@k score
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
  """Calculate Discounted Cumulative Gain at k.
  
  Args:
    y_true: True labels (shape: [n_samples])
    y_pred_probs: Predicted probabilities (shape: [n_samples, n_classes])
    k: Number of top predictions to consider
    
  Returns:
    DCG@k score
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
  """Calculate Recall at k.
  
  Args:
    y_true: True labels (shape: [n_samples])
    y_pred_probs: Predicted probabilities (shape: [n_samples, n_classes])
    k: Number of top predictions to consider
    
  Returns:
    Recall@k score
  """
  topk_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
  hits = [y_true[i] in topk_preds[i] for i in range(len(y_true))]
  return np.mean(hits)


def inference_batch(
    model: torch.nn.Module, 
    X: np.ndarray, 
    device: torch.device
) -> np.ndarray:
  """Run inference on a batch of data.
  
  Args:
    model: PyTorch model
    X: Input data (numpy array)
    device: Device to run inference on
    
  Returns:
    Predicted probabilities
  """
  xb = torch.tensor(X, dtype=torch.float32).to(device)
  with torch.no_grad():
    logits = model(xb)
    preds = torch.softmax(logits, dim=1).cpu().numpy()
  return preds


def inference_all(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
  """Run inference on all data in batches.
  
  Args:
    model: PyTorch model
    X: Input data (numpy array)
    batch_size: Batch size for inference
    device: Device to run inference on
    
  Returns:
    Tuple of (predicted labels, predicted probabilities)
  """
  if len(X) == 0:
    # Return empty arrays with correct shapes
    return np.array([], dtype=np.int64), np.array([]).reshape(0, -1)
  
  model.eval()
  all_preds = []
  
  with torch.no_grad():
    for i in range(0, len(X), batch_size):
      preds = inference_batch(model, X[i:i+batch_size], device)
      all_preds.append(preds)
  
  all_preds = np.concatenate(all_preds, axis=0)
  y_pred = all_preds.argmax(axis=1)
  return y_pred, all_preds


def evaluate_and_return_predictions(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    k_values: Tuple[int, ...] = (2, 5, 10)
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
  """Base evaluation function that runs inference once and calculates standard metrics.

  This is the base function that all other evaluation functions should call.
  It performs a single inference pass and calculates all standard metrics.

  Args:
    model: PyTorch model
    X: Input features
    y: True labels
    num_classes: Total number of classes
    batch_size: Batch size for inference
    device: Device to run inference on
    k_values: Values of k for recall@k, MRR@k, DCG@k metrics

  Returns:
    Tuple of (metrics dict, true labels, predicted probabilities, predicted labels)
  """
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Ensure y is numpy array
  y = np.asarray(y)

  # Handle empty dataset
  if len(X) == 0 or len(y) == 0:
    return {}, y, np.array([]), np.array([])

  # Get predictions (single inference pass)
  y_pred, all_preds = inference_all(model, X, batch_size, device)

  # Calculate standard metrics
  metrics = {}

  # Log loss
  metrics["logloss"] = log_loss(y, all_preds, labels=np.arange(num_classes))

  # Classification metrics (computed over classes present in y)
  metrics["macro_f1"] = f1_score(y, y_pred, average='macro', zero_division=0)
  metrics["macro_precision"] = precision_score(
    y, y_pred, average='macro', zero_division=0
  )
  metrics["macro_recall"] = recall_score(
    y, y_pred, average='macro', zero_division=0
  )

  # Ranking metrics at different k values
  for k in k_values:
    metrics[f"recall_at_{k}"] = recall_at_k(y, all_preds, k)
    metrics[f"mrr_at_{k}"] = mrr_at_k(y, all_preds, k)
    metrics[f"dcg_at_{k}"] = dcg_at_k(y, all_preds, k)

  return metrics, y, all_preds, y_pred


def evaluate_with_hierarchy(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cell_types: List[str],
    cell_type_to_idx: Dict[str, int],
    ontology_graph: nx.DiGraph,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    k_values: Tuple[int, ...] = (2, 5, 10)
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
  """Evaluate model with both standard and hierarchical metrics.

  Calls evaluate_and_return_predictions to get standard metrics, then adds
  hierarchical precision, recall, and F1 scores based on cell type ontology.

  Args:
    model: PyTorch model
    X: Input features
    y: True labels (indices)
    cell_types: List of cell type names
    cell_type_to_idx: Mapping from cell type names to indices
    ontology_graph: NetworkX directed graph of cell type relationships (REQUIRED)
    batch_size: Batch size for inference
    device: Device to run inference on
    k_values: Values of k for recall@k, MRR@k, DCG@k metrics

  Returns:
    Tuple of (metrics dict, true labels, predicted probabilities, predicted labels)
  """
  # Get standard metrics from base evaluation function
  num_classes = len(cell_types)
  metrics, y_true, all_preds, y_pred = evaluate_and_return_predictions(
    model=model,
    X=X,
    y=y,
    num_classes=num_classes,
    batch_size=batch_size,
    device=device,
    k_values=k_values
  )

  # Convert indices to cell type labels
  y_true_labels = [cell_types[idx] for idx in y_true]
  y_pred_labels = [cell_types[idx] for idx in y_pred]

  # Calculate hierarchical metrics
  hierarchical_metrics = calculate_hierarchical_f_score(
    y_true_labels, y_pred_labels, ontology_graph
  )

  # Add to metrics dict
  metrics.update(hierarchical_metrics)

  return metrics, y_true, all_preds, y_pred