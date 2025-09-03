"""Cell Ontology management for hierarchical evaluation."""

import pickle
import requests
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CellOntologyManager:
    """Manages Cell Ontology loading and graph construction.
    
    This class handles downloading, caching, and processing of the Cell Ontology
    for use in hierarchical evaluation metrics.
    """
    
    ONTOLOGY_URL = "http://purl.obolibrary.org/obo/cl.obo"
    OBO_FILENAME = "cl.obo"
    GRAPH_CACHE_FILENAME = "cell_ontology_graph.pkl"
    
    def __init__(self, cache_dir: Path):
        """Initialize with cache directory for ontology files.
        
        Args:
            cache_dir: Directory to cache ontology files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._graph = None
    
    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Get the ontology graph."""
        return self._graph
    
    def download_ontology(self) -> Path:
        """Download Cell Ontology OBO file if not cached.
        
        Returns:
            Path to the OBO file
        """
        obo_path = self.cache_dir / self.OBO_FILENAME
        
        if obo_path.exists():
            logger.info(f"Using cached ontology at {obo_path}")
            return obo_path
        
        logger.info(f"Downloading Cell Ontology from {self.ONTOLOGY_URL}")
        response = requests.get(self.ONTOLOGY_URL)
        response.raise_for_status()
        
        obo_path.write_bytes(response.content)
        logger.info(f"Saved ontology to {obo_path}")
        
        return obo_path
    
    def build_cell_type_graph(self) -> nx.DiGraph:
        """Build directed graph from ontology with caching.
        
        Returns:
            NetworkX directed graph of cell type relationships
        """
        # Check for cached graph
        graph_cache_path = self.cache_dir / self.GRAPH_CACHE_FILENAME
        
        if graph_cache_path.exists():
            logger.info(f"Loading cached graph from {graph_cache_path}")
            with open(graph_cache_path, 'rb') as f:
                self._graph = pickle.load(f)
            return self._graph
        
        # Download and parse OBO file
        obo_path = self.download_ontology()
        
        # Import obonet only when needed
        try:
            import obonet
        except ImportError:
            raise ImportError(
                "obonet is required for parsing OBO files. "
                "Install it with: pip install obonet"
            )
        
        logger.info(f"Parsing ontology from {obo_path}")
        ontology = obonet.read_obo(str(obo_path))
        
        # Build graph
        graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in ontology.nodes(data=True):
            if 'name' in node_data:
                graph.add_node(node_data['name'])
        
        # Add edges (parent -> child relationships)
        for node_id, node_data in ontology.nodes(data=True):
            if 'name' in node_data and 'is_a' in node_data:
                child_name = node_data['name']
                for parent_id in node_data['is_a']:
                    if parent_id in ontology:
                        parent_data = ontology.nodes.get(parent_id, {})
                        parent_name = parent_data.get('name')
                        if parent_name:
                            graph.add_edge(parent_name, child_name)
        
        logger.info(f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Cache the graph
        with open(graph_cache_path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"Cached graph to {graph_cache_path}")
        
        self._graph = graph
        return graph
    
    def map_cell_types_to_ontology(
        self, 
        cell_types: List[str],
        fuzzy_match: bool = False
    ) -> Dict[str, str]:
        """Map dataset cell types to ontology term names.
        
        Args:
            cell_types: List of cell type names from the dataset
            fuzzy_match: Whether to use fuzzy matching for cell type names
            
        Returns:
            Dictionary mapping dataset cell types to ontology terms
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_cell_type_graph() first.")
        
        mapping = {}
        ontology_names = set(self._graph.nodes())
        
        for cell_type in cell_types:
            # Try exact match first
            if cell_type in ontology_names:
                mapping[cell_type] = cell_type
            elif fuzzy_match:
                # Simple fuzzy matching - normalize and compare
                normalized = cell_type.lower().replace("-", " ").replace("_", " ")
                normalized = normalized.replace("+", " positive")
                
                best_match = None
                best_score = float('inf')  # Lower is better for this scoring
                
                for onto_name in ontology_names:
                    onto_normalized = onto_name.lower().replace("-", " ").replace("_", " ")
                    onto_normalized = onto_normalized.replace(",", "")
                    
                    # Calculate a simple similarity score
                    if normalized == onto_normalized:
                        # Exact match after normalization
                        best_match = onto_name
                        break
                    
                    # For CD markers, prioritize matches with CD markers
                    if "cd4" in normalized.lower():
                        if "cd4" in onto_normalized.lower():
                            # Both have CD4, this is a good match
                            score = abs(len(onto_normalized) - len(normalized))
                            if score < best_score:
                                best_match = onto_name
                                best_score = score
                    else:
                        # Check if one is substring of the other
                        if normalized in onto_normalized:
                            # Prefer shorter matches (more general)
                            score = len(onto_name) - len(normalized)
                            if score < best_score:
                                best_match = onto_name
                                best_score = score
                        elif onto_normalized in normalized:
                            score = len(normalized) - len(onto_name)
                            if score < best_score:
                                best_match = onto_name
                                best_score = score
                
                mapping[cell_type] = best_match if best_match else cell_type
            else:
                # No match found, map to itself
                mapping[cell_type] = cell_type
        
        return mapping