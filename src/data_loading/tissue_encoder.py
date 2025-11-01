"""
Hierarchical tissue encoder using UBERON ontology with global CellxGene vocabulary.

This module provides efficient encoding of tissue terms to one-hot vectors that include
tissue, organ(s), and system(s) information based on the UBERON ontology hierarchy.

Uses CellxGene's curated ontology lists to ensure consistent 126-dimensional encoding
across all datasets (train/validation/test).
"""

from typing import Dict, List
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import torch


class TissueEncoder:
  """
  Hierarchical tissue encoder using UBERON ontology with global CellxGene vocabulary.

  Converts tissue ontology terms to one-hot encoded vectors that include
  tissue, organ(s), and system(s) information. Multiple organs/systems
  for a single tissue result in multi-hot encoding.

  Uses CellxGene's curated ontology lists to ensure consistent encoding
  dimensions across all datasets (train/validation/test).

  Attributes:
      tissue_to_idx: Mapping from tissue term to index (81 tissues)
      organ_to_idx: Mapping from organ term to index (28 organs)
      system_to_idx: Mapping from system term to index (17 systems)
      tissue_dim: Dimension of tissue one-hot encoding (81)
      organ_dim: Dimension of organ one-hot encoding (28)
      system_dim: Dimension of system one-hot encoding (17)
      total_dim: Total dimension (126, constant)
  """

  def __init__(self):
    """
    Initialize encoder with global CellxGene vocabulary.

    No arguments needed - uses CellxGene's curated lists directly.
    """
    from cellxgene_ontology_guide.curated_ontology_term_lists import (
        CuratedOntologyTermList, get_curated_ontology_term_list)
    from cellxgene_ontology_guide.ontology_parser import OntologyParser

    self.ontology_parser = OntologyParser()

    # Load global vocabularies from CellxGene
    self._load_global_vocabularies()

    # Build vocabulary mappings
    self._build_vocabulary()

    # Pre-compute tissue mappings for efficiency
    self._build_mapping_cache()

    # Pre-build encoding lookup DataFrame for fast join-based encoding
    self._encoding_df = self._build_encoding_dataframe()

  def _load_global_vocabularies(self) -> None:
    """
    Load global vocabularies from CellxGene curated lists.
    """
    from cellxgene_ontology_guide.curated_ontology_term_lists import (
        CuratedOntologyTermList, get_curated_ontology_term_list)

    self.curated_tissues = get_curated_ontology_term_list(
        CuratedOntologyTermList.TISSUE_GENERAL)
    self.curated_organs = get_curated_ontology_term_list(
        CuratedOntologyTermList.ORGAN)
    self.curated_systems = get_curated_ontology_term_list(
        CuratedOntologyTermList.SYSTEM)

    # Convert to sets for fast lookup
    self.curated_organs_set = set(self.curated_organs)
    self.curated_systems_set = set(self.curated_systems)

  def _build_vocabulary(self) -> None:
    """
    Build vocabulary mappings for tissues, organs, and systems.

    Uses global CellxGene vocabularies (sorted for reproducibility).
    """
    # Create index mappings (sorted for reproducibility)
    self.tissue_to_idx = {
        t: i for i, t in enumerate(sorted(self.curated_tissues))
    }
    self.organ_to_idx = {
        o: i for i, o in enumerate(sorted(self.curated_organs))
    }
    self.system_to_idx = {
        s: i for i, s in enumerate(sorted(self.curated_systems))
    }

    # Store dimensions (constant)
    self.tissue_dim = len(self.tissue_to_idx)  # 81
    self.organ_dim = len(self.organ_to_idx)  # 28
    self.system_dim = len(self.system_to_idx)  # 17
    self.total_dim = self.tissue_dim + self.organ_dim + self.system_dim  # 126

  def _build_mapping_cache(self) -> None:
    """
    Pre-compute mappings from tissue to organ/system indices.

    Creates a cache for fast lookup during encoding.
    Uses UBERON ontology to find ancestors of each tissue.
    """
    self.tissue_to_organ_indices: Dict[str, List[int]] = {}
    self.tissue_to_system_indices: Dict[str, List[int]] = {}

    for tissue in self.tissue_to_idx.keys():
      # Get all ancestors of this tissue
      ancestors = self.ontology_parser.get_term_ancestors(
          tissue, include_self=True)

      # Find which curated organs/systems are ancestors
      matching_organs = [
          org for org in ancestors if org in self.curated_organs_set
      ]
      matching_systems = [
          sys for sys in ancestors if sys in self.curated_systems_set
      ]

      # Map to indices
      organ_indices = [
          self.organ_to_idx[o] for o in matching_organs
          if o in self.organ_to_idx
      ]
      system_indices = [
          self.system_to_idx[s] for s in matching_systems
          if s in self.system_to_idx
      ]

      self.tissue_to_organ_indices[tissue] = organ_indices
      self.tissue_to_system_indices[tissue] = system_indices

  def _build_encoding_dataframe(self) -> pd.DataFrame:
    """
    Pre-compute encoding vectors for all tissues as a DataFrame.

    Returns:
        DataFrame with tissue_id as index and encoding columns
        Shape: (81 tissues, 126 encoding dimensions)
    """
    rows = []

    for tissue_id in sorted(self.tissue_to_idx.keys()):
      # Create encoding vector for this tissue
      encoding = np.zeros(self.total_dim, dtype=np.float32)

      # Tissue one-hot
      tissue_idx = self.tissue_to_idx[tissue_id]
      encoding[tissue_idx] = 1.0

      # Organ multi-hot
      organ_indices = self.tissue_to_organ_indices.get(tissue_id, [])
      for organ_idx in organ_indices:
        encoding[self.tissue_dim + organ_idx] = 1.0

      # System multi-hot
      system_indices = self.tissue_to_system_indices.get(tissue_id, [])
      for system_idx in system_indices:
        encoding[self.tissue_dim + self.organ_dim + system_idx] = 1.0

      rows.append({'tissue_id': tissue_id, 'encoding': encoding})

    return pd.DataFrame(rows).set_index('tissue_id')

  def encode(self, tissues: pd.Series) -> torch.Tensor:
    """
    Encode a series of tissue terms to hierarchical one-hot vectors.

    Uses efficient join-based approach:
    1. Create mapping DataFrame (done once during init)
    2. Join tissues with pre-computed encodings
    3. Convert to tensor

    Args:
        tissues: Pandas Series of tissue ontology term IDs (can be categorical)

    Returns:
        Tensor of shape (n_cells, total_dim) containing concatenated
        one-hot encodings for tissue, organ(s), and system(s)

    Example:
        >>> tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])
        >>> encoded = encoder.encode(tissues)
        >>> encoded.shape
        torch.Size([2, 126])
    """
    # Build encoding lookup if not cached
    if not hasattr(self, '_encoding_df'):
      self._encoding_df = self._build_encoding_dataframe()

    # Handle empty input
    if len(tissues) == 0:
      return torch.zeros(0, self.total_dim, dtype=torch.float32)

    # Convert categorical to string if needed
    if pd.api.types.is_categorical_dtype(tissues):
      tissue_values = tissues.astype(str)
    else:
      tissue_values = tissues

    # Create DataFrame for joining
    tissue_df = pd.DataFrame({'tissue_id': tissue_values}).reset_index(
        drop=True)

    # Join with pre-computed encodings
    joined = tissue_df.join(self._encoding_df, on='tissue_id')

    # Handle unknown tissues (fill with zeros)
    encodings = joined['encoding'].values

    # Convert to list of arrays, handling NaN/None for unknown tissues
    encoding_list = []
    for enc in encodings:
      if enc is not None and not (isinstance(enc, float) and np.isnan(enc)):
        encoding_list.append(enc)
      else:
        # Unknown tissue - create zero vector
        encoding_list.append(np.zeros(self.total_dim, dtype=np.float32))

    # Stack into array
    encoding_matrix = np.stack(encoding_list)

    # Convert to tensor
    return torch.from_numpy(encoding_matrix)

  def encode_batch(self, tissues_list: List[pd.Series]) -> torch.Tensor:
    """
    Encode multiple batches of tissues efficiently.

    Args:
        tissues_list: List of pandas Series, each containing tissue terms

    Returns:
        Tensor of shape (total_cells, total_dim)
    """
    encoded_batches = [self.encode(tissues) for tissues in tissues_list]
    return torch.cat(encoded_batches, dim=0)

  def get_tissue_indices(self, tissues: pd.Series) -> torch.LongTensor:
    """
    Get integer indices for tissues (useful for embedding layers).

    Args:
        tissues: Pandas Series of tissue ontology term IDs

    Returns:
        LongTensor of shape (n_cells,) with tissue indices
    """
    if pd.api.types.is_categorical_dtype(tissues):
      tissue_values = tissues.astype(str)
    else:
      tissue_values = tissues

    indices = torch.zeros(len(tissues), dtype=torch.long)
    for i, tissue in enumerate(tissue_values):
      if tissue in self.tissue_to_idx:
        indices[i] = self.tissue_to_idx[tissue]

    return indices

  def save(self, path: Path) -> None:
    """
    Save encoder state to disk.

    Args:
        path: Path to save pickle file
    """
    state = {
        'tissue_to_idx': self.tissue_to_idx,
        'organ_to_idx': self.organ_to_idx,
        'system_to_idx': self.system_to_idx,
        'tissue_to_organ_indices': self.tissue_to_organ_indices,
        'tissue_to_system_indices': self.tissue_to_system_indices,
        'encoding_df': self._encoding_df,  # Save pre-computed encodings
        'dimensions': {
            'tissue_dim': self.tissue_dim,
            'organ_dim': self.organ_dim,
            'system_dim': self.system_dim,
            'total_dim': self.total_dim,
        }
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
      pickle.dump(state, f)

  @classmethod
  def load(cls, path: Path) -> 'TissueEncoder':
    """
    Load encoder state from disk.

    Args:
        path: Path to pickle file

    Returns:
        Loaded TissueEncoder instance

    Note: Since vocabulary is global, we could just recreate the encoder,
    but loading from cache is faster and ensures consistency.
    """
    with open(path, 'rb') as f:
      state = pickle.load(f)

    # Verify dimensions match expected global vocabulary
    if state['dimensions']['tissue_dim'] != 81:
      raise ValueError(
          f"Expected 81 tissues, got {state['dimensions']['tissue_dim']}")
    if state['dimensions']['organ_dim'] != 28:
      raise ValueError(
          f"Expected 28 organs, got {state['dimensions']['organ_dim']}")
    if state['dimensions']['system_dim'] != 17:
      raise ValueError(
          f"Expected 17 systems, got {state['dimensions']['system_dim']}")

    # Create instance
    encoder = cls.__new__(cls)

    # Initialize ontology parser
    from cellxgene_ontology_guide.ontology_parser import OntologyParser
    encoder.ontology_parser = OntologyParser()

    # Load vocabularies
    encoder._load_global_vocabularies()

    # Restore state
    encoder.tissue_to_idx = state['tissue_to_idx']
    encoder.organ_to_idx = state['organ_to_idx']
    encoder.system_to_idx = state['system_to_idx']
    encoder.tissue_to_organ_indices = state['tissue_to_organ_indices']
    encoder.tissue_to_system_indices = state['tissue_to_system_indices']
    encoder._encoding_df = state['encoding_df']  # Restore pre-computed encodings
    encoder.tissue_dim = state['dimensions']['tissue_dim']
    encoder.organ_dim = state['dimensions']['organ_dim']
    encoder.system_dim = state['dimensions']['system_dim']
    encoder.total_dim = state['dimensions']['total_dim']

    return encoder

  def get_tissue_info(self, tissue: str) -> Dict[str, any]:
    """
    Get human-readable information about a tissue's encoding.

    Args:
        tissue: Tissue ontology term ID

    Returns:
        Dictionary with tissue, organs, and systems information
    """
    if tissue not in self.tissue_to_idx:
      # Check if tissue is valid but not in curated list
      if self.ontology_parser.is_valid_term_id(tissue):
        return {
            'warning':
                f'Valid tissue but not in CellxGene curated list: {tissue}',
            'tissue':
                tissue,
            'tissue_label':
                self.ontology_parser.get_term_label(tissue),
            'in_vocabulary':
                False
        }
      return {'error': f'Unknown tissue: {tissue}'}

    # Get organ and system IDs from cache
    organ_indices = self.tissue_to_organ_indices.get(tissue, [])
    system_indices = self.tissue_to_system_indices.get(tissue, [])

    # Reverse lookup to get IDs
    idx_to_organ = {v: k for k, v in self.organ_to_idx.items()}
    idx_to_system = {v: k for k, v in self.system_to_idx.items()}

    organ_ids = [idx_to_organ[i] for i in organ_indices]
    system_ids = [idx_to_system[i] for i in system_indices]

    return {
        'tissue':
            tissue,
        'tissue_label':
            self.ontology_parser.get_term_label(tissue),
        'tissue_idx':
            self.tissue_to_idx[tissue],
        'organs': [(oid, self.ontology_parser.get_term_label(oid))
                   for oid in organ_ids],
        'organ_indices':
            organ_indices,
        'systems': [(sid, self.ontology_parser.get_term_label(sid))
                    for sid in system_ids],
        'system_indices':
            system_indices,
        'encoding_dim':
            self.total_dim,
    }


def create_tissue_encoder() -> TissueEncoder:
  """
  Create encoder with global CellxGene vocabulary.

  Returns:
      TissueEncoder with fixed vocabulary (81 tissues, 28 organs, 17 systems)

  Note: No need to pass parquet files - vocabulary is global and fixed.
  """
  return TissueEncoder()


def load_tissues_from_parquet(parquet_path: Path) -> pd.Series:
  """
  Load tissue column from parquet file.

  Args:
      parquet_path: Path to parquet file

  Returns:
      Pandas Series with tissue ontology term IDs (categorical)
  """
  df = pd.read_parquet(parquet_path, columns=['tissue_ontology_term_id'])
  return df['tissue_ontology_term_id']
