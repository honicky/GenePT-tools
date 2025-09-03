"""Tests for integrated hierarchical evaluation functionality."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from unittest.mock import MagicMock, patch
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def test_evaluate_with_hierarchy():
    """Test evaluate_with_hierarchy function."""
    from src.training.metrics import evaluate_with_hierarchy
    
    # Create test data
    n_samples = 10
    n_features = 5
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create cell types list
    cell_types = ["T cell", "B cell", "macrophage"]
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Create a simple ontology graph
    ontology_graph = nx.DiGraph()
    ontology_graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
        ("immune cell", "macrophage"),
    ])
    
    # Create a simple model
    model = SimpleModel(n_features, n_classes)
    model.eval()
    
    # Test with hierarchy
    metrics = evaluate_with_hierarchy(
        model, X, y, cell_types, cell_type_to_idx, 
        ontology_graph, batch_size=5
    )
    
    # Check that all expected metrics are present
    assert "logloss" in metrics
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    
    # Check ranking metrics
    for k in [2, 5, 10]:
        assert f"recall_at_{k}" in metrics
        assert f"mrr_at_{k}" in metrics
        assert f"dcg_at_{k}" in metrics
    
    # Check hierarchical metrics
    assert "hierarchical_precision" in metrics
    assert "hierarchical_recall" in metrics
    assert "hierarchical_f1" in metrics
    
    # Check value ranges
    assert 0 <= metrics["hierarchical_precision"] <= 1
    assert 0 <= metrics["hierarchical_recall"] <= 1
    assert 0 <= metrics["hierarchical_f1"] <= 1


def test_evaluate_with_hierarchy_no_graph():
    """Test evaluate_with_hierarchy when no graph is provided."""
    from src.training.metrics import evaluate_with_hierarchy
    
    # Create test data
    n_samples = 10
    n_features = 5
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    cell_types = ["T cell", "B cell", "macrophage"]
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    model = SimpleModel(n_features, n_classes)
    model.eval()
    
    # Test without hierarchy (should work like normal evaluate)
    metrics = evaluate_with_hierarchy(
        model, X, y, cell_types, cell_type_to_idx,
        ontology_graph=None, batch_size=5
    )
    
    # Should have standard metrics but not hierarchical ones
    assert "logloss" in metrics
    assert "macro_f1" in metrics
    assert "hierarchical_precision" not in metrics
    assert "hierarchical_recall" not in metrics
    assert "hierarchical_f1" not in metrics


def test_evaluate_with_hierarchy_empty_data():
    """Test evaluate_with_hierarchy with empty data."""
    from src.training.metrics import evaluate_with_hierarchy
    
    X = np.array([]).reshape(0, 5)
    y = np.array([])
    
    cell_types = ["T cell", "B cell", "macrophage"]
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    model = SimpleModel(5, 3)
    model.eval()
    
    ontology_graph = nx.DiGraph()
    
    metrics = evaluate_with_hierarchy(
        model, X, y, cell_types, cell_type_to_idx,
        ontology_graph, batch_size=5
    )
    
    # Should return empty dict for empty data
    assert metrics == {}


def test_evaluate_with_hierarchy_perfect_predictions():
    """Test with perfect predictions to verify metrics."""
    from src.training.metrics import evaluate_with_hierarchy
    
    n_samples = 100
    n_features = 5
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    cell_types = ["T cell", "B cell", "macrophage"]
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Create ontology
    ontology_graph = nx.DiGraph()
    ontology_graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
        ("immune cell", "macrophage"),
    ])
    
    # Create a model that makes perfect predictions based on input pattern
    class PerfectModel(nn.Module):
        def __init__(self, X_full, y_true):
            super().__init__()
            self.X_full = torch.tensor(X_full, dtype=torch.float32)
            self.y_true = torch.tensor(y_true)
            
        def forward(self, x):
            # Find which samples these are by matching against full X
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, n_classes) - 10
            
            # Simple approach: use sum of features as a unique identifier
            x_sums = x.sum(dim=1)
            full_sums = self.X_full.sum(dim=1)
            
            for i in range(batch_size):
                # Find the matching sample in the full dataset
                diffs = torch.abs(full_sums - x_sums[i])
                idx = torch.argmin(diffs).item()
                true_class = self.y_true[idx].item()
                logits[i, true_class] = 10
            
            return logits
    
    model = PerfectModel(X, y)
    model.eval()
    
    metrics = evaluate_with_hierarchy(
        model, X, y, cell_types, cell_type_to_idx,
        ontology_graph, batch_size=10
    )
    
    # With perfect predictions
    assert metrics["recall_at_2"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["hierarchical_f1"] == 1.0


def test_evaluate_with_hierarchy_sibling_confusion():
    """Test hierarchical metrics when confusing sibling cell types."""
    from src.training.metrics import evaluate_with_hierarchy
    
    n_samples = 50
    n_features = 5
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    # All samples are T cells (index 0)
    y = np.zeros(n_samples, dtype=int)
    
    cell_types = ["T cell", "B cell", "macrophage"]
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # Create ontology where T cell and B cell are siblings
    ontology_graph = nx.DiGraph()
    ontology_graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
        ("immune cell", "macrophage"),
    ])
    
    # Create a model that confuses T cells with B cells
    class ConfusedModel(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, n_classes) - 10
            # Always predict B cell (index 1)
            logits[:, 1] = 10
            return logits
    
    model = ConfusedModel()
    model.eval()
    
    metrics = evaluate_with_hierarchy(
        model, X, y, cell_types, cell_type_to_idx,
        ontology_graph, batch_size=10
    )
    
    # Standard metrics should be poor (0% accuracy)
    assert metrics["recall_at_2"] < 1.0  # Should find T cell in top 2
    assert metrics["macro_f1"] < 0.5
    
    # But hierarchical metrics should be better (both are immune cells)
    assert metrics["hierarchical_f1"] > metrics["macro_f1"]
    assert metrics["hierarchical_f1"] > 0.5  # Should be around 0.67