"""
Tests for tissue ontology API.
"""

import pytest
import pandas as pd
from src.data_loading.tissue_ontology import TissueOntology


@pytest.fixture(scope='module')
def tissue_ont():
  """Fixture to initialize TissueOntology once for all tests."""
  return TissueOntology()


def test_initialization(tissue_ont):
  """Test that TissueOntology initializes correctly."""
  assert len(tissue_ont.tissue_list) > 0
  assert len(tissue_ont.organ_list) > 0
  assert len(tissue_ont.system_list) > 0
  assert isinstance(tissue_ont.tissue_set, set)
  assert isinstance(tissue_ont.organ_set, set)
  assert isinstance(tissue_ont.system_set, set)


def test_get_label(tissue_ont):
  """Test label retrieval."""
  # Blood
  label = tissue_ont.get_label('UBERON:0000178')
  assert label == 'blood'

  # Lung
  label = tissue_ont.get_label('UBERON:0002048')
  assert label == 'lung'


def test_get_tissue_info_blood(tissue_ont):
  """Test tissue info for blood."""
  info = tissue_ont.get_tissue_info('UBERON:0000178')

  assert info['tissue_id'] == 'UBERON:0000178'
  assert info['tissue_label'] == 'blood'
  assert len(info['organs']) > 0
  assert len(info['systems']) > 0
  assert info['is_curated_tissue'] is True

  # Blood should have itself as an organ
  organ_ids = [oid for oid, _ in info['organs']]
  assert 'UBERON:0000178' in organ_ids

  # Blood should be under hematopoietic system
  system_ids = [sid for sid, _ in info['systems']]
  assert 'UBERON:0002390' in system_ids  # hematopoietic system


def test_get_tissue_info_lung(tissue_ont):
  """Test tissue info for lung."""
  info = tissue_ont.get_tissue_info('UBERON:0002048')

  assert info['tissue_label'] == 'lung'
  assert len(info['systems']) > 0

  # Lung should be under respiratory system
  system_ids = [sid for sid, _ in info['systems']]
  assert 'UBERON:0001004' in system_ids  # respiratory system


def test_get_primary_organ(tissue_ont):
  """Test getting primary organ."""
  # Blood has itself as organ
  organ = tissue_ont.get_primary_organ('UBERON:0000178')
  assert organ is not None
  organ_id, organ_label = organ
  assert organ_id == 'UBERON:0000178'
  assert organ_label == 'blood'


def test_get_primary_system(tissue_ont):
  """Test getting primary system."""
  # Lung has respiratory system
  system = tissue_ont.get_primary_system('UBERON:0002048')
  assert system is not None
  system_id, system_label = system
  assert system_label == 'respiratory system'


def test_add_hierarchy_columns_primary(tissue_ont):
  """Test adding hierarchy columns with primary_only=True."""
  # Create test dataframe
  df = pd.DataFrame({
    'cell_id': ['cell1', 'cell2', 'cell3'],
    'tissue_ontology_term_id': [
      'UBERON:0000178',  # blood
      'UBERON:0002048',  # lung
      'UBERON:0002113',  # kidney
    ]
  })

  # Add hierarchy columns
  result = tissue_ont.add_hierarchy_columns(df)

  # Check new columns exist
  assert 'tissue_label' in result.columns
  assert 'organ_label' in result.columns
  assert 'organ_uberon_id' in result.columns
  assert 'system_label' in result.columns
  assert 'system_uberon_id' in result.columns

  # Check original rows preserved
  assert len(result) == len(df)

  # Check values
  assert result.loc[0, 'tissue_label'] == 'blood'
  assert result.loc[1, 'tissue_label'] == 'lung'
  assert result.loc[2, 'tissue_label'] == 'kidney'


def test_add_hierarchy_columns_with_prefix(tissue_ont):
  """Test adding hierarchy columns with prefix."""
  df = pd.DataFrame({
    'tissue_ontology_term_id': ['UBERON:0000178']
  })

  result = tissue_ont.add_hierarchy_columns(df, prefix='cxg_')

  assert 'cxg_tissue_label' in result.columns
  assert 'cxg_organ_label' in result.columns
  assert 'cxg_system_label' in result.columns


def test_add_hierarchy_columns_selective(tissue_ont):
  """Test adding only selected columns."""
  df = pd.DataFrame({
    'tissue_ontology_term_id': ['UBERON:0000178']
  })

  # Only add system
  result = tissue_ont.add_hierarchy_columns(
    df,
    add_tissue_label=False,
    add_organ=False,
    add_system=True
  )

  assert 'tissue_label' not in result.columns
  assert 'organ_label' not in result.columns
  assert 'system_label' in result.columns


def test_add_hierarchy_columns_expand_rows(tissue_ont):
  """Test adding hierarchy columns with use_primary_only=False."""
  df = pd.DataFrame({
    'cell_id': ['cell1'],
    'tissue_ontology_term_id': ['UBERON:0002107']  # liver (multiple systems)
  })

  result = tissue_ont.add_hierarchy_columns(df, use_primary_only=False)

  # Liver belongs to multiple systems, so should have multiple rows
  assert len(result) >= len(df)


def test_get_ancestors(tissue_ont):
  """Test ancestor retrieval."""
  ancestors = tissue_ont.get_ancestors('UBERON:0000178')  # blood

  assert isinstance(ancestors, set)
  assert len(ancestors) > 0
  # Should include itself
  assert 'UBERON:0000178' in ancestors
  # Should include hematopoietic system
  assert 'UBERON:0002390' in ancestors


def test_get_descendants(tissue_ont):
  """Test descendant retrieval."""
  descendants = tissue_ont.get_descendants('UBERON:0000178')  # blood

  assert isinstance(descendants, set)
  # Blood should have venous blood as descendant
  assert 'UBERON:0013756' in descendants  # venous blood


def test_is_descendant_of(tissue_ont):
  """Test descendant relationship checking."""
  # Venous blood is descendant of blood
  assert tissue_ont.is_descendant_of('UBERON:0013756', 'UBERON:0000178') is True

  # Blood is not descendant of lung
  assert tissue_ont.is_descendant_of('UBERON:0000178', 'UBERON:0002048') is False


def test_create_tissue_mapping_table(tissue_ont):
  """Test creating complete mapping table."""
  df = tissue_ont.create_tissue_mapping_table()

  assert isinstance(df, pd.DataFrame)
  assert len(df) > 0
  assert 'tissue_label' in df.columns
  assert 'tissue_uberon_id' in df.columns
  assert 'organ_label' in df.columns
  assert 'system_label' in df.columns

  # Should have all curated tissues
  unique_tissues = df['tissue_uberon_id'].nunique()
  assert unique_tissues == len(tissue_ont.tissue_list)


def test_get_curated_lists(tissue_ont):
  """Test getting curated lists."""
  lists = tissue_ont.get_curated_lists()

  assert 'tissues' in lists
  assert 'organs' in lists
  assert 'systems' in lists

  assert len(lists['tissues']) > 0
  assert len(lists['organs']) > 0
  assert len(lists['systems']) > 0


def test_missing_column_error(tissue_ont):
  """Test that missing column raises appropriate error."""
  df = pd.DataFrame({'wrong_column': ['UBERON:0000178']})

  with pytest.raises(ValueError, match="Column .* not found"):
    tissue_ont.add_hierarchy_columns(df, tissue_id_col='tissue_ontology_term_id')


def test_null_tissue_ids(tissue_ont):
  """Test handling of null tissue IDs."""
  df = pd.DataFrame({
    'tissue_ontology_term_id': ['UBERON:0000178', None, 'UBERON:0002048']
  })

  result = tissue_ont.add_hierarchy_columns(df)

  # Should handle null without error
  assert len(result) == 3
  assert pd.isna(result.loc[1, 'tissue_label'])
