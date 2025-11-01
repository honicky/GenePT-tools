"""
Integration tests for TissueEncoder.

Tests the encoder with real CellxGene data from parquet files.
"""

import pytest
import pandas as pd
import torch
import time
from pathlib import Path

from src.data_loading.tissue_encoder import (TissueEncoder,
                                              create_tissue_encoder,
                                              load_tissues_from_parquet)


@pytest.fixture
def real_parquet_files():
  """Get real parquet files from test data."""
  data_dir = Path('/mnt/scratch/cellxgene_v2_test_v1')
  files = sorted(data_dir.glob('*.parquet'))[:5]  # Use first 5 files

  if not files:
    pytest.skip("Test data not available")

  return files


def test_load_tissues_from_real_parquet(real_parquet_files):
  """Test loading tissues from real parquet files."""
  tissues = load_tissues_from_parquet(real_parquet_files[0])

  assert isinstance(tissues, pd.Series)
  assert len(tissues) > 0
  assert all(tissues.str.startswith('UBERON:'))


def test_create_encoder_global_vocab():
  """Test creating encoder with global vocabulary."""
  encoder = create_tissue_encoder()

  # Verify fixed global dimensions
  assert encoder.tissue_dim == 81
  assert encoder.organ_dim == 28
  assert encoder.system_dim == 17
  assert encoder.total_dim == 126


def test_encode_real_data(real_parquet_files):
  """Test encoding tissues from real data."""
  encoder = create_tissue_encoder()

  tissues = load_tissues_from_parquet(real_parquet_files[0])
  encoded = encoder.encode(tissues)

  assert encoded.shape[0] == len(tissues)
  assert encoded.shape[1] == 126  # Fixed global dimension
  assert not torch.isnan(encoded).any()
  assert not torch.isinf(encoded).any()


def test_encode_multiple_files(real_parquet_files):
  """Test encoding tissues from multiple files with same encoder."""
  encoder = create_tissue_encoder()

  all_encoded = []
  for file in real_parquet_files:
    tissues = load_tissues_from_parquet(file)
    encoded = encoder.encode(tissues)
    all_encoded.append(encoded)

  # All should have same dimension
  combined = torch.cat(all_encoded, dim=0)
  assert combined.shape[1] == 126


def test_tissue_statistics(real_parquet_files):
  """Test statistics of real tissue encodings."""
  encoder = create_tissue_encoder()

  tissues = load_tissues_from_parquet(real_parquet_files[0])
  encoded = encoder.encode(tissues)

  # Log statistics
  print(f"\nTissue encoding statistics:")
  print(f"  Total dimension: {encoder.total_dim} (fixed)")
  print(f"  Tissue dim: {encoder.tissue_dim} (81 curated tissues)")
  print(f"  Organ dim: {encoder.organ_dim} (28 curated organs)")
  print(f"  System dim: {encoder.system_dim} (17 curated systems)")
  print(
      f"  Average ones per encoding: {encoded.sum(dim=1).mean().item():.2f}")
  print(f"  Sparsity: {(encoded == 0).float().mean().item():.2%}")


def test_encoder_persistence(real_parquet_files, tmp_path):
  """Test saving and loading encoder with real data."""
  encoder = create_tissue_encoder()

  save_path = tmp_path / "real_encoder.pkl"
  encoder.save(save_path)

  loaded_encoder = TissueEncoder.load(save_path)

  # Verify dimensions
  assert loaded_encoder.total_dim == 126

  # Test that loaded encoder produces same results
  tissues = load_tissues_from_parquet(real_parquet_files[0])
  encoded1 = encoder.encode(tissues)
  encoded2 = loaded_encoder.encode(tissues)

  assert torch.allclose(encoded1, encoded2)


def test_consistency_across_datasets(real_parquet_files):
  """Test that encoder produces consistent dimensions across different datasets."""
  # Create encoder once
  encoder = create_tissue_encoder()

  # Encode tissues from different files
  for file in real_parquet_files:
    tissues = load_tissues_from_parquet(file)
    encoded = encoder.encode(tissues)

    # All should have same dimension regardless of which tissues are present
    assert encoded.shape[1] == 126


def test_unknown_tissue_handling(real_parquet_files):
  """Test encoder handles tissues not in curated list."""
  encoder = create_tissue_encoder()

  # Mix known and potentially unknown tissues
  tissues = load_tissues_from_parquet(real_parquet_files[0])
  encoded = encoder.encode(tissues)

  # Should still produce valid output
  assert encoded.shape[1] == 126
  # Unknown tissues should have zeros in tissue encoding but may have organ/system info
  assert not torch.isnan(encoded).any()


def test_encoding_performance(real_parquet_files):
  """Test encoding performance on large dataset."""
  encoder = create_tissue_encoder()

  # Load large file
  tissues = load_tissues_from_parquet(real_parquet_files[0])

  # Time the encoding (should be very fast due to join)
  start = time.time()
  encoded = encoder.encode(tissues)
  elapsed = time.time() - start

  cells_per_second = len(tissues) / elapsed

  print(f"\nEncoding performance:")
  print(f"  Cells: {len(tissues)}")
  print(f"  Time: {elapsed:.3f}s")
  print(f"  Throughput: {cells_per_second:.0f} cells/second")

  # Should encode at >100k cells/second (conservative target)
  assert cells_per_second > 100_000, f"Too slow: {cells_per_second:.0f} cells/s"


def test_real_tissue_info(real_parquet_files):
  """Test getting tissue info for real tissues."""
  encoder = create_tissue_encoder()

  tissues = load_tissues_from_parquet(real_parquet_files[0])
  unique_tissues = tissues.unique()[:5]  # Test first 5 unique tissues

  for tissue in unique_tissues:
    info = encoder.get_tissue_info(tissue)

    # Should have valid info
    assert 'tissue' in info or 'error' in info or 'warning' in info

    if 'tissue' in info:
      print(f"\n{info['tissue_label']} ({info['tissue']}):")
      print(f"  Organs: {[label for _, label in info['organs']]}")
      print(f"  Systems: {[label for _, label in info['systems']]}")


def test_categorical_tissue_from_parquet(real_parquet_files):
  """Test handling categorical tissue columns from parquet."""
  encoder = create_tissue_encoder()

  # Load tissue column (may be categorical)
  df = pd.read_parquet(real_parquet_files[0],
                       columns=['tissue_ontology_term_id'])
  tissues = df['tissue_ontology_term_id']

  # Should handle categorical or string dtype
  encoded = encoder.encode(tissues)

  assert encoded.shape[0] == len(tissues)
  assert encoded.shape[1] == 126


def test_all_files_same_encoder(real_parquet_files):
  """Test processing all files with single encoder instance."""
  encoder = create_tissue_encoder()

  total_cells = 0
  for file in real_parquet_files:
    tissues = load_tissues_from_parquet(file)
    encoded = encoder.encode(tissues)

    # Track total cells processed
    total_cells += len(tissues)

    # Verify consistent output
    assert encoded.shape[1] == 126
    assert not torch.isnan(encoded).any()

  print(f"\nProcessed {total_cells:,} cells across {len(real_parquet_files)} files")


def test_encoding_unique_tissues(real_parquet_files):
  """Test that each unique tissue gets a unique encoding."""
  encoder = create_tissue_encoder()

  # Load tissues from first file
  tissues = load_tissues_from_parquet(real_parquet_files[0])
  unique_tissues = tissues.unique()

  # Encode each unique tissue
  encodings = {}
  for tissue in unique_tissues:
    tissue_series = pd.Series([tissue])
    encoded = encoder.encode(tissue_series)
    encodings[tissue] = encoded[0]

  # Verify all unique tissues have different encodings
  encoding_list = list(encodings.values())
  for i in range(len(encoding_list)):
    for j in range(i + 1, len(encoding_list)):
      # At least the tissue component should differ
      assert not torch.allclose(encoding_list[i], encoding_list[j])


def test_encoding_preserves_order(real_parquet_files):
  """Test that encoding preserves the order of input tissues."""
  encoder = create_tissue_encoder()

  tissues = load_tissues_from_parquet(real_parquet_files[0])[:100]

  # Encode in original order
  encoded1 = encoder.encode(tissues)

  # Encode one by one and stack
  encoded2 = torch.stack([encoder.encode(pd.Series([t]))[0] for t in tissues])

  # Should be identical
  assert torch.allclose(encoded1, encoded2)


def test_memory_efficiency(real_parquet_files):
  """Test that encoder doesn't consume excessive memory."""
  import sys

  encoder = create_tissue_encoder()

  # Get size of encoder
  encoder_size = sys.getsizeof(encoder.__dict__)

  # Should be relatively small (<1MB for lookup tables)
  print(f"\nEncoder memory footprint: {encoder_size / 1024:.1f} KB")
  assert encoder_size < 1_000_000  # Less than 1MB


def test_batch_vs_individual_encoding(real_parquet_files):
  """Test that batch encoding matches individual encoding."""
  encoder = create_tissue_encoder()

  tissues = load_tissues_from_parquet(real_parquet_files[0])[:50]

  # Encode all at once
  batch_encoded = encoder.encode(tissues)

  # Encode individually and stack
  individual_encoded = torch.stack(
      [encoder.encode(pd.Series([t]))[0] for t in tissues])

  # Should match
  assert torch.allclose(batch_encoded, individual_encoded)
