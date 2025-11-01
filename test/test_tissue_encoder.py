"""
Unit tests for TissueEncoder.

Tests the core functionality of hierarchical tissue encoding using
CellxGene's global vocabulary.
"""

import pytest
import pandas as pd
import torch
import numpy as np
from pathlib import Path

from src.data_loading.tissue_encoder import TissueEncoder, create_tissue_encoder


@pytest.fixture
def sample_encoder():
  """Create encoder with global CellxGene vocabulary."""
  return TissueEncoder()


def test_encoder_initialization(sample_encoder):
  """Test encoder initializes with correct global dimensions."""
  assert sample_encoder.tissue_dim == 81  # CellxGene curated tissues
  assert sample_encoder.organ_dim == 28  # CellxGene curated organs
  assert sample_encoder.system_dim == 17  # CellxGene curated systems
  assert sample_encoder.total_dim == 126  # Fixed total dimension


def test_single_tissue_encoding(sample_encoder):
  """Test encoding a single tissue."""
  tissues = pd.Series(['UBERON:0002107'])  # liver
  encoded = sample_encoder.encode(tissues)

  assert encoded.shape == (1, 126)  # Fixed global dimension
  assert encoded.dtype == torch.float32

  # Check one-hot encoding has at least tissue + organ + system
  # liver: 1 tissue + 1 organ (liver) + 1 system (digestive)
  assert encoded.sum().item() >= 3.0


def test_multi_system_encoding(sample_encoder):
  """Test tissue with multiple systems gets multi-hot encoded."""
  tissues = pd.Series(['UBERON:0002106'])  # spleen
  encoded = sample_encoder.encode(tissues)

  # spleen: 1 tissue + 1 organ (spleen) + 2 systems (hematopoietic, immune)
  assert encoded.sum().item() >= 4.0


def test_batch_encoding(sample_encoder):
  """Test encoding multiple tissues in a batch."""
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048', 'UBERON:0000178'])
  encoded = sample_encoder.encode(tissues)

  assert encoded.shape == (3, 126)  # Fixed global dimension

  # Each row should have at least 2 ones (tissue + at least one system)
  row_sums = encoded.sum(dim=1)
  assert all(row_sums >= 2.0)


def test_categorical_input(sample_encoder):
  """Test encoding works with categorical dtype."""
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'], dtype='category')
  encoded = sample_encoder.encode(tissues)

  assert encoded.shape == (2, 126)


def test_unknown_tissue_handling(sample_encoder):
  """Test encoder handles unknown tissues gracefully (zero vector)."""
  tissues = pd.Series(
      ['UBERON:0002107', 'UBERON:9999999'])  # One valid, one unknown
  encoded = sample_encoder.encode(tissues)

  # First row should have values, second row should be all zeros
  assert encoded[0].sum() > 0
  assert encoded[1].sum() == 0


def test_get_tissue_indices(sample_encoder):
  """Test converting tissues to integer indices."""
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])
  indices = sample_encoder.get_tissue_indices(tissues)

  assert indices.shape == (2, )
  assert indices.dtype == torch.long
  assert all(indices >= 0)
  assert all(indices < sample_encoder.tissue_dim)


def test_save_and_load(sample_encoder, tmp_path):
  """Test saving and loading encoder state."""
  save_path = tmp_path / "encoder.pkl"
  sample_encoder.save(save_path)

  loaded_encoder = TissueEncoder.load(save_path)

  # Verify global dimensions are preserved
  assert loaded_encoder.tissue_dim == 81
  assert loaded_encoder.organ_dim == 28
  assert loaded_encoder.system_dim == 17
  assert loaded_encoder.total_dim == 126
  assert loaded_encoder.tissue_to_idx == sample_encoder.tissue_to_idx


def test_get_tissue_info(sample_encoder):
  """Test retrieving human-readable tissue information."""
  info = sample_encoder.get_tissue_info('UBERON:0002107')

  assert info['tissue'] == 'UBERON:0002107'
  assert 'organs' in info
  assert 'systems' in info
  assert info['encoding_dim'] == sample_encoder.total_dim


def test_encoding_determinism(sample_encoder):
  """Test encoding is deterministic."""
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])

  encoded1 = sample_encoder.encode(tissues)
  encoded2 = sample_encoder.encode(tissues)

  assert torch.allclose(encoded1, encoded2)


def test_encoding_sparsity(sample_encoder):
  """Test encodings are sparse (mostly zeros)."""
  tissues = pd.Series(['UBERON:0002107'] * 100)
  encoded = sample_encoder.encode(tissues)

  # Calculate sparsity (proportion of zeros)
  sparsity = (encoded == 0).float().mean()

  # Should be mostly zeros (>80% zeros for hierarchical encoding)
  assert sparsity > 0.8


def test_create_tissue_encoder_helper():
  """Test the helper function creates encoder correctly."""
  encoder = create_tissue_encoder()

  assert isinstance(encoder, TissueEncoder)
  assert encoder.total_dim == 126


def test_encoding_values_are_binary(sample_encoder):
  """Test that all encoding values are 0 or 1."""
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048', 'UBERON:0000178'])
  encoded = sample_encoder.encode(tissues)

  # All values should be 0.0 or 1.0
  unique_values = torch.unique(encoded)
  assert len(unique_values) <= 2
  assert all(v in [0.0, 1.0] for v in unique_values.tolist())


def test_encoding_reproducibility_across_instances():
  """Test that different encoder instances produce identical encodings."""
  encoder1 = TissueEncoder()
  encoder2 = TissueEncoder()

  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])

  encoded1 = encoder1.encode(tissues)
  encoded2 = encoder2.encode(tissues)

  assert torch.allclose(encoded1, encoded2)


def test_tissue_dimension_structure(sample_encoder):
  """Test that encoding has correct structure (tissue + organ + system)."""
  tissues = pd.Series(['UBERON:0002107'])  # liver
  encoded = sample_encoder.encode(tissues)

  # Split encoding into components
  tissue_part = encoded[:, :sample_encoder.tissue_dim]
  organ_part = encoded[:, sample_encoder.tissue_dim:sample_encoder.tissue_dim +
                        sample_encoder.organ_dim]
  system_part = encoded[:, sample_encoder.tissue_dim +
                         sample_encoder.organ_dim:]

  # Tissue part should have exactly 1 hot value
  assert tissue_part.sum() == 1.0

  # Organ and system parts should have at least 1 hot value each
  assert organ_part.sum() >= 1.0
  assert system_part.sum() >= 1.0


def test_encode_batch(sample_encoder):
  """Test encoding multiple series in a batch."""
  tissues_list = [
      pd.Series(['UBERON:0002107']),
      pd.Series(['UBERON:0002048', 'UBERON:0000178']),
      pd.Series(['UBERON:0002113']),
  ]

  encoded = sample_encoder.encode_batch(tissues_list)

  # Should have 4 total cells (1 + 2 + 1)
  assert encoded.shape == (4, 126)


def test_encoding_matrix_type(sample_encoder):
  """Test that encoding returns proper PyTorch tensor."""
  tissues = pd.Series(['UBERON:0002107'])
  encoded = sample_encoder.encode(tissues)

  assert isinstance(encoded, torch.Tensor)
  assert encoded.dtype == torch.float32
  assert not encoded.requires_grad  # Should not have gradients by default


def test_empty_input_handling(sample_encoder):
  """Test encoder handles empty input gracefully."""
  tissues = pd.Series([], dtype=str)
  encoded = sample_encoder.encode(tissues)

  assert encoded.shape == (0, 126)


def test_large_batch_encoding(sample_encoder):
  """Test encoding works with large batches."""
  # Create a large batch (10,000 cells)
  tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'] * 5000)
  encoded = sample_encoder.encode(tissues)

  assert encoded.shape == (10000, 126)
  assert not torch.isnan(encoded).any()
  assert not torch.isinf(encoded).any()
