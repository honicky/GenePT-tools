"""Tests for hierarchical metrics functionality."""

import numpy as np
import pytest
import networkx as nx
from typing import Set, List, Dict
from unittest.mock import MagicMock, patch


def test_get_ancestors_single_node():
    """Test getting ancestors for a single node with no parents."""
    from src.training.hierarchical_metrics import get_ancestors
    
    # Create a simple graph
    graph = nx.DiGraph()
    graph.add_node("cell")
    
    ancestors = get_ancestors("cell", graph)
    assert ancestors == {"cell"}


def test_get_ancestors_with_hierarchy():
    """Test getting ancestors in a hierarchical graph."""
    from src.training.hierarchical_metrics import get_ancestors
    
    # Create a hierarchical graph
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
        ("T cell", "CD4 T cell"),
        ("T cell", "CD8 T cell"),
    ])
    
    # Test leaf node
    ancestors = get_ancestors("CD4 T cell", graph)
    assert ancestors == {"CD4 T cell", "T cell", "immune cell", "cell"}
    
    # Test intermediate node
    ancestors = get_ancestors("T cell", graph)
    assert ancestors == {"T cell", "immune cell", "cell"}
    
    # Test root node
    ancestors = get_ancestors("cell", graph)
    assert ancestors == {"cell"}


def test_get_ancestors_node_not_in_graph():
    """Test handling of nodes not in the graph."""
    from src.training.hierarchical_metrics import get_ancestors
    
    graph = nx.DiGraph()
    graph.add_node("cell")
    
    ancestors = get_ancestors("unknown cell", graph)
    assert ancestors == {"unknown cell"}  # Should return just the node itself


def test_calculate_hierarchical_f_score_perfect_match():
    """Test hierarchical F-score with perfect predictions."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    # Create a simple hierarchy
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "T cell"),
        ("cell", "B cell"),
    ])
    
    y_true = ["T cell", "B cell", "T cell"]
    y_pred = ["T cell", "B cell", "T cell"]
    
    metrics = calculate_hierarchical_f_score(y_true, y_pred, graph)
    
    assert metrics["hierarchical_precision"] == 1.0
    assert metrics["hierarchical_recall"] == 1.0
    assert metrics["hierarchical_f1"] == 1.0


def test_calculate_hierarchical_f_score_parent_child_confusion():
    """Test hierarchical F-score when confusing parent and child."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    # Create a hierarchy
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
    ])
    
    # Predict parent when truth is child (should have some credit)
    y_true = ["T cell"]
    y_pred = ["immune cell"]
    
    metrics = calculate_hierarchical_f_score(y_true, y_pred, graph)
    
    # T cell ancestors: {T cell, immune cell, cell}
    # immune cell ancestors: {immune cell, cell}
    # Intersection: {immune cell, cell} = 2 elements
    # Precision: 2/2 = 1.0 (all predicted ancestors are correct)
    # Recall: 2/3 = 0.667 (got 2 out of 3 true ancestors)
    assert metrics["hierarchical_precision"] == 1.0
    assert abs(metrics["hierarchical_recall"] - 0.667) < 0.01
    assert abs(metrics["hierarchical_f1"] - 0.8) < 0.01


def test_calculate_hierarchical_f_score_sibling_confusion():
    """Test hierarchical F-score when confusing siblings."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    # Create a hierarchy
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
    ])
    
    # Predict sibling (T cell instead of B cell)
    y_true = ["B cell"]
    y_pred = ["T cell"]
    
    metrics = calculate_hierarchical_f_score(y_true, y_pred, graph)
    
    # B cell ancestors: {B cell, immune cell, cell}
    # T cell ancestors: {T cell, immune cell, cell}
    # Intersection: {immune cell, cell} = 2 elements
    # Precision: 2/3 = 0.667
    # Recall: 2/3 = 0.667
    assert abs(metrics["hierarchical_precision"] - 0.667) < 0.01
    assert abs(metrics["hierarchical_recall"] - 0.667) < 0.01
    assert abs(metrics["hierarchical_f1"] - 0.667) < 0.01


def test_calculate_hierarchical_f_score_completely_wrong():
    """Test hierarchical F-score with completely unrelated predictions."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    # Create two separate hierarchies
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("cell", "epithelial cell"),
        ("epithelial cell", "squamous cell"),
    ])
    
    # Predict from different branch
    y_true = ["T cell"]
    y_pred = ["squamous cell"]
    
    metrics = calculate_hierarchical_f_score(y_true, y_pred, graph)
    
    # T cell ancestors: {T cell, immune cell, cell}
    # squamous cell ancestors: {squamous cell, epithelial cell, cell}
    # Intersection: {cell} = 1 element
    # Precision: 1/3 = 0.333
    # Recall: 1/3 = 0.333
    assert abs(metrics["hierarchical_precision"] - 0.333) < 0.01
    assert abs(metrics["hierarchical_recall"] - 0.333) < 0.01
    assert abs(metrics["hierarchical_f1"] - 0.333) < 0.01


def test_calculate_hierarchical_f_score_empty_inputs():
    """Test hierarchical F-score with empty inputs."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    graph = nx.DiGraph()
    
    metrics = calculate_hierarchical_f_score([], [], graph)
    
    assert metrics["hierarchical_precision"] == 0
    assert metrics["hierarchical_recall"] == 0
    assert metrics["hierarchical_f1"] == 0


def test_calculate_hierarchical_f_score_multiple_samples():
    """Test hierarchical F-score with multiple samples."""
    from src.training.hierarchical_metrics import calculate_hierarchical_f_score
    
    # Create a hierarchy
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("cell", "immune cell"),
        ("immune cell", "T cell"),
        ("immune cell", "B cell"),
        ("T cell", "CD4 T cell"),
        ("T cell", "CD8 T cell"),
    ])
    
    y_true = ["CD4 T cell", "CD8 T cell", "B cell"]
    y_pred = ["CD4 T cell", "T cell", "T cell"]  # One perfect, two partial
    
    metrics = calculate_hierarchical_f_score(y_true, y_pred, graph)
    
    # Verify the metrics are between 0 and 1
    assert 0.5 < metrics["hierarchical_precision"] <= 1.0
    assert 0.5 < metrics["hierarchical_recall"] <= 1.0
    assert 0.5 < metrics["hierarchical_f1"] <= 1.0