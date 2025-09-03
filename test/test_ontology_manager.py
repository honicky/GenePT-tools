"""Tests for Cell Ontology Manager."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import networkx as nx
import pickle


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_init_creates_cache_dir(temp_cache_dir):
    """Test that initialization creates cache directory if it doesn't exist."""
    from src.training.ontology import CellOntologyManager
    
    cache_dir = temp_cache_dir / "ontology_cache"
    assert not cache_dir.exists()
    
    manager = CellOntologyManager(cache_dir)
    assert cache_dir.exists()


def test_download_ontology_when_not_cached(temp_cache_dir):
    """Test downloading ontology when not in cache."""
    from src.training.ontology import CellOntologyManager
    
    manager = CellOntologyManager(temp_cache_dir)
    
    mock_response = MagicMock()
    mock_response.content = b"fake obo content"
    mock_response.raise_for_status = MagicMock()
    
    with patch('requests.get', return_value=mock_response):
        obo_path = manager.download_ontology()
        
        assert obo_path.exists()
        assert obo_path.name == "cl.obo"
        assert obo_path.read_bytes() == b"fake obo content"


def test_download_ontology_when_cached(temp_cache_dir):
    """Test that cached ontology is not re-downloaded."""
    from src.training.ontology import CellOntologyManager
    
    # Create a cached file
    cached_file = temp_cache_dir / "cl.obo"
    cached_file.write_text("cached content")
    
    manager = CellOntologyManager(temp_cache_dir)
    
    with patch('requests.get') as mock_get:
        obo_path = manager.download_ontology()
        
        # Should not make a request
        mock_get.assert_not_called()
        
        assert obo_path == cached_file
        assert obo_path.read_text() == "cached content"


def test_build_cell_type_graph_from_obo(temp_cache_dir):
    """Test building graph from OBO file."""
    from src.training.ontology import CellOntologyManager
    
    # Create a minimal OBO file content
    obo_content = """
[Term]
id: CL:0000000
name: cell

[Term]
id: CL:0000001
name: immune cell
is_a: CL:0000000

[Term]
id: CL:0000002
name: T cell
is_a: CL:0000001

[Term]
id: CL:0000003
name: B cell
is_a: CL:0000001
"""
    
    obo_file = temp_cache_dir / "cl.obo"
    obo_file.write_text(obo_content)
    
    manager = CellOntologyManager(temp_cache_dir)
    
    with patch.object(manager, 'download_ontology', return_value=obo_file):
        graph = manager.build_cell_type_graph()
    
    # Check graph structure
    assert "cell" in graph
    assert "immune cell" in graph
    assert "T cell" in graph
    assert "B cell" in graph
    
    # Check relationships (edges go from parent to child)
    assert graph.has_edge("cell", "immune cell")
    assert graph.has_edge("immune cell", "T cell")
    assert graph.has_edge("immune cell", "B cell")
    assert not graph.has_edge("T cell", "B cell")


def test_build_cell_type_graph_with_cache(temp_cache_dir):
    """Test that graph is loaded from cache when available."""
    from src.training.ontology import CellOntologyManager
    
    # Create a cached graph
    cached_graph = nx.DiGraph()
    cached_graph.add_edge("cached_cell", "cached_immune_cell")
    
    cache_file = temp_cache_dir / "cell_ontology_graph.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_graph, f)
    
    manager = CellOntologyManager(temp_cache_dir)
    
    with patch.object(manager, 'download_ontology') as mock_download:
        graph = manager.build_cell_type_graph()
        
        # Should not download OBO file
        mock_download.assert_not_called()
        
        # Should return cached graph
        assert "cached_cell" in graph
        assert graph.has_edge("cached_cell", "cached_immune_cell")


def test_map_cell_types_exact_match():
    """Test mapping cell types with exact matches."""
    from src.training.ontology import CellOntologyManager
    
    # Create a graph with some cell types
    graph = nx.DiGraph()
    graph.add_node("T cell")
    graph.add_node("B cell")
    graph.add_node("macrophage")
    
    manager = CellOntologyManager(Path("/tmp"))
    manager._graph = graph  # Inject the graph
    
    cell_types = ["T cell", "B cell", "unknown cell"]
    mapping = manager.map_cell_types_to_ontology(cell_types)
    
    assert mapping["T cell"] == "T cell"
    assert mapping["B cell"] == "B cell"
    assert mapping["unknown cell"] == "unknown cell"  # Not in ontology, maps to itself


def test_map_cell_types_fuzzy_match():
    """Test mapping cell types with fuzzy matching."""
    from src.training.ontology import CellOntologyManager
    
    # Create a graph with some cell types
    graph = nx.DiGraph()
    graph.add_node("T cell")
    graph.add_node("B cell")
    graph.add_node("CD4-positive, alpha-beta T cell")
    
    manager = CellOntologyManager(Path("/tmp"))
    manager._graph = graph
    
    cell_types = ["T-cell", "CD4+ T cell", "B-cell"]
    mapping = manager.map_cell_types_to_ontology(cell_types, fuzzy_match=True)
    
    # Print for debugging
    print(f"Mapping results: {mapping}")
    
    # Should match despite different formatting
    assert mapping["T-cell"] == "T cell"
    assert mapping["B-cell"] == "B cell"
    # CD4+ T cell should match to CD4-positive, alpha-beta T cell
    assert mapping["CD4+ T cell"] == "CD4-positive, alpha-beta T cell"


def test_get_graph_property():
    """Test getting the graph property."""
    from src.training.ontology import CellOntologyManager
    
    manager = CellOntologyManager(Path("/tmp"))
    
    # Initially None
    assert manager.graph is None
    
    # After setting, should return the graph
    test_graph = nx.DiGraph()
    test_graph.add_node("test")
    manager._graph = test_graph
    
    assert manager.graph is test_graph