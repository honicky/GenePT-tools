"""
Tissue ontology utilities using CellXGene's official ontology guide.

Provides functions to query tissue-organ-system hierarchies and augment
dataframes with ontology information.

Example:
    >>> from src.data_loading.tissue_ontology import TissueOntology
    >>>
    >>> # Initialize once
    >>> tissue_ont = TissueOntology()
    >>>
    >>> # Query a single tissue
    >>> info = tissue_ont.get_tissue_info('UBERON:0000178')
    >>> print(info['tissue_label'])  # 'blood'
    >>>
    >>> # Augment a dataframe
    >>> df = tissue_ont.add_hierarchy_columns(
    ...     df,
    ...     tissue_id_col='tissue_ontology_term_id'
    ... )
"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from functools import lru_cache

from cellxgene_ontology_guide.ontology_parser import OntologyParser
from cellxgene_ontology_guide.curated_ontology_term_lists import (
  CuratedOntologyTermList,
  get_curated_ontology_term_list
)


class TissueOntology:
  """
  Interface to CellXGene tissue ontology for querying tissue-organ-system hierarchies.

  This class provides methods to:
  - Get organ and system classifications for tissues
  - Augment dataframes with ontology information
  - Query tissue relationships and hierarchies

  Attributes:
      ontology: OntologyParser instance for UBERON queries
      tissue_set: Set of curated tissue IDs from CellXGene
      organ_set: Set of curated organ IDs from CellXGene
      system_set: Set of curated system IDs from CellXGene
  """

  def __init__(self):
    """Initialize the tissue ontology with CellXGene curated lists."""
    self.ontology = OntologyParser()

    # Load curated term lists
    self.tissue_list = get_curated_ontology_term_list(CuratedOntologyTermList.TISSUE_GENERAL)
    self.organ_list = get_curated_ontology_term_list(CuratedOntologyTermList.ORGAN)
    self.system_list = get_curated_ontology_term_list(CuratedOntologyTermList.SYSTEM)

    # Convert to sets for fast lookup
    self.tissue_set = set(self.tissue_list)
    self.organ_set = set(self.organ_list)
    self.system_set = set(self.system_list)

  @lru_cache(maxsize=1024)
  def get_label(self, term_id: str) -> str:
    """
    Get human-readable label for an ontology term.

    Args:
        term_id: UBERON ID (e.g., 'UBERON:0000178')

    Returns:
        Human-readable label (e.g., 'blood')
    """
    return self.ontology.get_term_label(term_id)

  @lru_cache(maxsize=1024)
  def get_tissue_info(self, tissue_id: str) -> Dict:
    """
    Get complete hierarchy information for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Dictionary with keys:
            - tissue_id: Input tissue ID
            - tissue_label: Human-readable tissue name
            - organs: List of (organ_id, organ_label) tuples
            - systems: List of (system_id, system_label) tuples
            - is_curated_tissue: Whether this is in CellXGene's curated tissue list
            - is_curated_organ: Whether this is in CellXGene's curated organ list
            - is_curated_system: Whether this is in CellXGene's curated system list
    """
    # Get all ancestors (including the tissue itself)
    ancestors = self.ontology.get_term_ancestors(tissue_id, include_self=True)

    # Find which curated organs and systems are ancestors
    organs = [
      (oid, self.get_label(oid))
      for oid in ancestors
      if oid in self.organ_set
    ]
    systems = [
      (sid, self.get_label(sid))
      for sid in ancestors
      if sid in self.system_set
    ]

    return {
      'tissue_id': tissue_id,
      'tissue_label': self.get_label(tissue_id),
      'organs': organs,
      'systems': systems,
      'is_curated_tissue': tissue_id in self.tissue_set,
      'is_curated_organ': tissue_id in self.organ_set,
      'is_curated_system': tissue_id in self.system_set,
    }

  def get_primary_organ(self, tissue_id: str) -> Optional[Tuple[str, str]]:
    """
    Get the primary (first) organ for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Tuple of (organ_id, organ_label) or None if no organ
    """
    info = self.get_tissue_info(tissue_id)
    return info['organs'][0] if info['organs'] else None

  def get_primary_system(self, tissue_id: str) -> Optional[Tuple[str, str]]:
    """
    Get the primary (first) system for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Tuple of (system_id, system_label) or None if no system
    """
    info = self.get_tissue_info(tissue_id)
    return info['systems'][0] if info['systems'] else None

  def add_hierarchy_columns(
    self,
    df: pd.DataFrame,
    tissue_id_col: str = 'tissue_ontology_term_id',
    add_tissue_label: bool = True,
    add_organ: bool = True,
    add_system: bool = True,
    use_primary_only: bool = True,
    prefix: str = ''
  ) -> pd.DataFrame:
    """
    Add tissue hierarchy columns to a dataframe.

    Args:
        df: DataFrame with tissue ontology IDs
        tissue_id_col: Name of column containing tissue UBERON IDs
        add_tissue_label: Whether to add tissue_label column
        add_organ: Whether to add organ columns
        add_system: Whether to add system columns
        use_primary_only: If True, uses only the first (primary) organ/system.
                         If False, may create multiple rows per input row
                         if tissue belongs to multiple organs/systems.
        prefix: Prefix to add to new column names (e.g., 'cxg_')

    Returns:
        DataFrame with additional columns:
            - {prefix}tissue_label (if add_tissue_label=True)
            - {prefix}organ_label, {prefix}organ_uberon_id (if add_organ=True)
            - {prefix}system_label, {prefix}system_uberon_id (if add_system=True)
    """
    if tissue_id_col not in df.columns:
      raise ValueError(f"Column '{tissue_id_col}' not found in dataframe")

    if use_primary_only:
      # Add columns directly without expanding rows
      def get_hierarchy_info(tissue_id):
        if pd.isna(tissue_id):
          return pd.Series({
            f'{prefix}tissue_label': None,
            f'{prefix}organ_label': None,
            f'{prefix}organ_uberon_id': None,
            f'{prefix}system_label': None,
            f'{prefix}system_uberon_id': None,
          })

        info = self.get_tissue_info(tissue_id)
        organ = info['organs'][0] if info['organs'] else (None, None)
        system = info['systems'][0] if info['systems'] else (None, None)

        return pd.Series({
          f'{prefix}tissue_label': info['tissue_label'] if add_tissue_label else None,
          f'{prefix}organ_label': organ[1] if add_organ else None,
          f'{prefix}organ_uberon_id': organ[0] if add_organ else None,
          f'{prefix}system_label': system[1] if add_system else None,
          f'{prefix}system_uberon_id': system[0] if add_system else None,
        })

      hierarchy_df = df[tissue_id_col].apply(get_hierarchy_info)

      # Drop None columns
      hierarchy_df = hierarchy_df.loc[:, hierarchy_df.notna().any()]

      return pd.concat([df, hierarchy_df], axis=1)

    else:
      # Create multiple rows for tissues with multiple organs/systems
      rows = []

      for idx, row in df.iterrows():
        tissue_id = row[tissue_id_col]

        if pd.isna(tissue_id):
          # Keep row as-is with None values
          new_row = row.to_dict()
          if add_tissue_label:
            new_row[f'{prefix}tissue_label'] = None
          if add_organ:
            new_row[f'{prefix}organ_label'] = None
            new_row[f'{prefix}organ_uberon_id'] = None
          if add_system:
            new_row[f'{prefix}system_label'] = None
            new_row[f'{prefix}system_uberon_id'] = None
          rows.append(new_row)
          continue

        info = self.get_tissue_info(tissue_id)
        organs = info['organs'] if add_organ and info['organs'] else [(None, None)]
        systems = info['systems'] if add_system and info['systems'] else [(None, None)]

        # Create one row per organ-system combination
        for organ_id, organ_label in organs:
          for system_id, system_label in systems:
            new_row = row.to_dict()
            if add_tissue_label:
              new_row[f'{prefix}tissue_label'] = info['tissue_label']
            if add_organ:
              new_row[f'{prefix}organ_label'] = organ_label
              new_row[f'{prefix}organ_uberon_id'] = organ_id
            if add_system:
              new_row[f'{prefix}system_label'] = system_label
              new_row[f'{prefix}system_uberon_id'] = system_id
            rows.append(new_row)

      return pd.DataFrame(rows)

  def get_ancestors(self, tissue_id: str, include_self: bool = True) -> Set[str]:
    """
    Get all ancestor terms for a tissue.

    Args:
        tissue_id: UBERON tissue ID
        include_self: Whether to include the tissue itself in results

    Returns:
        Set of ancestor UBERON IDs
    """
    return self.ontology.get_term_ancestors(tissue_id, include_self=include_self)

  def get_descendants(self, tissue_id: str) -> Set[str]:
    """
    Get all descendant terms for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Set of descendant UBERON IDs
    """
    return self.ontology.get_term_descendants(tissue_id)

  def get_children(self, tissue_id: str) -> Set[str]:
    """
    Get direct children (one level down) for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Set of child UBERON IDs
    """
    return self.ontology.get_term_children(tissue_id)

  def get_parents(self, tissue_id: str) -> Set[str]:
    """
    Get direct parents (one level up) for a tissue.

    Args:
        tissue_id: UBERON tissue ID

    Returns:
        Set of parent UBERON IDs
    """
    return self.ontology.get_term_parents(tissue_id)

  def get_distance(self, term1: str, term2: str) -> Optional[int]:
    """
    Get the shortest path distance between two terms.

    Args:
        term1: First UBERON ID
        term2: Second UBERON ID

    Returns:
        Distance (number of edges) or None if no path exists
    """
    return self.ontology.get_distance_between_terms(term1, term2)

  def is_descendant_of(self, child_id: str, ancestor_id: str) -> bool:
    """
    Check if one tissue is a descendant of another.

    Args:
        child_id: UBERON ID of potential child
        ancestor_id: UBERON ID of potential ancestor

    Returns:
        True if child_id is a descendant of ancestor_id
    """
    ancestors = self.get_ancestors(child_id, include_self=False)
    return ancestor_id in ancestors

  def create_tissue_mapping_table(self) -> pd.DataFrame:
    """
    Create a complete mapping table for all CellXGene curated tissues.

    Returns:
        DataFrame with columns:
            - tissue_label
            - tissue_uberon_id
            - organ_label
            - organ_uberon_id
            - system_label
            - system_uberon_id

        Note: Tissues with multiple organs/systems will have multiple rows.
    """
    rows = []

    for tissue_id in self.tissue_list:
      info = self.get_tissue_info(tissue_id)

      # Handle cases with no organs or systems
      organs = info['organs'] if info['organs'] else [(None, None)]
      systems = info['systems'] if info['systems'] else [(None, None)]

      # Create one row per organ-system combination
      for organ_id, organ_label in organs:
        for system_id, system_label in systems:
          rows.append({
            'tissue_label': info['tissue_label'],
            'tissue_uberon_id': tissue_id,
            'organ_label': organ_label,
            'organ_uberon_id': organ_id,
            'system_label': system_label,
            'system_uberon_id': system_id,
          })

    df = pd.DataFrame(rows)
    return df.sort_values(
      ['system_label', 'organ_label', 'tissue_label'],
      na_position='last'
    )

  def get_curated_lists(self) -> Dict[str, List[str]]:
    """
    Get all CellXGene curated term lists.

    Returns:
        Dictionary with keys 'tissues', 'organs', 'systems' mapping to lists of IDs
    """
    return {
      'tissues': self.tissue_list.copy(),
      'organs': self.organ_list.copy(),
      'systems': self.system_list.copy(),
    }
