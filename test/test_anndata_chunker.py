import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import anndata as ad
from scipy import sparse
import torch

from src.utils import (
    AnnDataChunker,
    _load_csr_matrix_components,
    _load_var_metadata,
    _load_obs_metadata
)

@pytest.fixture
def temp_h5ad_file(tmp_path):
    """Create a temporary h5ad file for testing."""
    # Create a simple anndata object
    n_obs, n_vars = 100, 50
    X = sparse.random(n_obs, n_vars, density=0.1, format='csr')
    
    obs = pd.DataFrame({
        'cell_type': pd.Categorical(['type_' + str(i % 3) for i in range(n_obs)]),
        'condition': ['cond_' + str(i % 2) for i in range(n_obs)],
        'counts': np.random.randint(100, 1000, n_obs),
        'alt_index': [ str(i) for i in range(n_obs)],
    })
    
    var = pd.DataFrame({
        'gene_name': [f'gene_{i}' for i in range(n_vars)],
        'feature_type': pd.Categorical(['protein_coding' if i % 2 else 'lncRNA' 
                                      for i in range(n_vars)])
    })
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    file_path = tmp_path / "test.h5ad"
    adata.write_h5ad(file_path)
    return file_path

def test_anndata_chunker_init(temp_h5ad_file):
    """Test AnnDataChunker initialization."""
    # Test with valid inputs
    chunker = AnnDataChunker(temp_h5ad_file, ['cell_type', 'condition'])
    assert chunker.file_path_or_obj == temp_h5ad_file
    assert chunker.obs_columns == ['cell_type', 'condition']
    
    # Test with None obs_columns
    chunker = AnnDataChunker(temp_h5ad_file, None)
    assert chunker.obs_columns == []
    
    # Test with invalid inputs
    with pytest.raises(TypeError):
        AnnDataChunker(123, ['cell_type'])  # Invalid file_path type
    with pytest.raises(TypeError):
        AnnDataChunker(temp_h5ad_file, 'cell_type')  # Invalid obs_columns type

def test_anndata_chunker_context_manager(temp_h5ad_file):
    """Test context manager functionality."""
    with AnnDataChunker(temp_h5ad_file, None) as chunker:
        assert chunker.is_open
        assert chunker._file is not None
        assert isinstance(chunker._obs_df, pd.DataFrame)
        assert isinstance(chunker._var_df, pd.DataFrame)
    
    assert not chunker.is_open
    assert chunker._file is None
    assert chunker._obs_df is None
    assert chunker._var_df is None

def test_load_subset(temp_h5ad_file):
    """Test loading a subset of data."""
    with AnnDataChunker(temp_h5ad_file, ['cell_type', 'condition']) as chunker:
        # Test normal subset
        subset = chunker.load_subset(start_row=0, n_rows=10)
        assert subset.n_obs == 10
        assert subset.n_vars == 50
        assert 'cell_type' in subset.obs.columns
        assert 'condition' in subset.obs.columns
        
        # Test with valid_indices
        valid_indices = np.array([0, 1, 2])
        subset = chunker.load_subset(start_row=0, n_rows=10, valid_indices=valid_indices)
        assert subset.n_obs == 10
        assert subset.n_vars == len(valid_indices)
        
        # Test error cases
        with pytest.raises(ValueError):
            chunker.load_subset(start_row=-1, n_rows=10)  # Invalid start_row
        with pytest.raises(ValueError):
            chunker.load_subset(start_row=0, n_rows=0)    # Invalid n_rows
        with pytest.raises(ValueError):
            chunker.load_subset(start_row=1000, n_rows=10)  # start_row too large

def test_load_torch_csr_matrix(temp_h5ad_file):
    """Test loading data as torch CSR matrix."""
    with AnnDataChunker(temp_h5ad_file, None) as chunker:
        # Test normal loading
        matrix = chunker.load_torch_csr_matrix(start_row=0, n_rows=10)
        assert matrix.shape == (10, 50)
        assert matrix.layout == torch.sparse_csr
        assert matrix.dtype == torch.float32

        # Test with valid_indices
        valid_indices = np.array([0, 1, 2])
        full_matrix = chunker.load_torch_csr_matrix(start_row=0, n_rows=10)
        subset_matrix = chunker.load_torch_csr_matrix(start_row=0, n_rows=10, valid_indices=valid_indices)
        
        assert subset_matrix.shape == (10, len(valid_indices))  # Shape should match valid_indices length
        assert subset_matrix.layout == torch.sparse_csr
        assert subset_matrix.dtype == torch.float32
        
        # Convert to dense to easily compare values
        full_dense = full_matrix.to_dense()
        subset_dense = subset_matrix.to_dense()
        for i, idx in enumerate(valid_indices):
            assert torch.allclose(subset_dense[:, i], full_dense[:, idx])

def test_helper_functions(temp_h5ad_file):
    """Test the helper functions."""
    with h5py.File(temp_h5ad_file, 'r') as f:
        # Test _load_var_metadata
        var_df = _load_var_metadata(f)
        assert isinstance(var_df, pd.DataFrame)
        assert 'gene_name' in var_df.columns
        assert 'feature_type' in var_df.columns
        
        # Test _load_obs_metadata
        obs_df = _load_obs_metadata(f, start_row=0, n_rows=10, 
                                  obs_columns=['cell_type', 'condition'])
        assert isinstance(obs_df, pd.DataFrame)
        assert len(obs_df) == 10
        assert 'cell_type' in obs_df.columns
        assert 'condition' in obs_df.columns
        
        # Test _load_csr_matrix_components
        data, indices, indptr = _load_csr_matrix_components(f, start_row=0, n_rows=10)
        assert isinstance(data, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(indptr, np.ndarray)
        assert len(indptr) == 11  # n_rows + 1

def test_property_access(temp_h5ad_file):
    """Test property access for obs and var."""
    chunker = AnnDataChunker(temp_h5ad_file, None)
    
    # Test access when file is not open
    with pytest.raises(RuntimeError):
        _ = chunker.obs
    with pytest.raises(RuntimeError):
        _ = chunker.var
    
    # Test access when file is open
    with chunker:
        assert isinstance(chunker.obs, pd.DataFrame)
        assert isinstance(chunker.var, pd.DataFrame)

def test_iter_chunks(temp_h5ad_file):
    """Test the iter_chunks method of AnnDataChunker."""
    with AnnDataChunker(temp_h5ad_file, ['cell_type', 'condition']) as chunker:
        chunk_size = 30
        
        # Get original data for comparison
        full_data = chunker.load_subset(0, len(chunker.obs))
        total_rows = len(chunker.obs)
        
        # Collect all chunks
        chunks = list(chunker.iter_chunks(chunk_size))
        
        # Test number of chunks
        expected_num_chunks = (total_rows + chunk_size - 1) // chunk_size
        assert len(chunks) == expected_num_chunks
        
        # Test chunk sizes and total rows
        total_rows_in_chunks = 0
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                # All chunks except the last should be full size
                assert chunk.n_obs == chunk_size
            else:
                # Last chunk might be smaller
                expected_last_chunk_size = total_rows - (i * chunk_size)
                assert chunk.n_obs == expected_last_chunk_size
            total_rows_in_chunks += chunk.n_obs
        
        # Verify total number of rows
        assert total_rows_in_chunks == total_rows
        
        # Test concatenated data matches original
        concatenated = ad.concat(chunks, join='outer')
        assert concatenated.n_obs == full_data.n_obs
        assert concatenated.n_vars == full_data.n_vars
        
        # Check that obs data matches
        for col in ['cell_type', 'condition']:
            pd.testing.assert_series_equal(
                concatenated.obs[col],
                full_data.obs[col],
                check_names=False
            )
        
        # Test that sparse matrix data matches
        assert np.allclose(
            concatenated.X.toarray(),
            full_data.X.toarray()
        )

def test_iter_chunks_edge_cases(temp_h5ad_file):
    """Test iter_chunks with edge cases."""
    with AnnDataChunker(temp_h5ad_file, ['cell_type']) as chunker:
        total_rows = len(chunker.obs)
        
        # Test chunk_size = 1
        chunks = list(chunker.iter_chunks(1))
        assert len(chunks) == total_rows
        assert all(chunk.n_obs == 1 for chunk in chunks)
        
        # Test chunk_size = total_rows
        chunks = list(chunker.iter_chunks(total_rows))
        assert len(chunks) == 1
        assert chunks[0].n_obs == total_rows
        
        # Test invalid chunk sizes
        with pytest.raises(ValueError):
            list(chunker.iter_chunks(0))
        with pytest.raises(ValueError):
            list(chunker.iter_chunks(-1))
        
        # Test with closed file
        chunker.close()
        with pytest.raises(RuntimeError):
            list(chunker.iter_chunks(10))

def test_anndata_chunker_with_file_object(temp_h5ad_file):
    """Test AnnDataChunker with a file-like object."""
    # Test with an opened file object
    with open(temp_h5ad_file, 'rb') as file_obj:
        chunker = AnnDataChunker(file_obj, ['cell_type', 'condition'])
        assert not chunker._owns_file  # Should not own the file
        assert chunker.file_path_or_obj == file_obj
        
        # Test that we can use the chunker with the file object
        with chunker:
            assert chunker.is_open
            subset = chunker.load_subset(start_row=0, n_rows=10)
            assert subset.n_obs == 10
            assert 'cell_type' in subset.obs.columns
        
        # File should still be open since we didn't own it
        assert not file_obj.closed

def test_anndata_chunker_context_nesting(temp_h5ad_file):
    """Test nested context managers with file-like object."""
    with open(temp_h5ad_file, 'rb') as file_obj:
        with AnnDataChunker(file_obj, ['cell_type']) as chunker:
            assert chunker.is_open
            # Test basic functionality
            subset = chunker.load_subset(start_row=0, n_rows=5)
            assert subset.n_obs == 5
            
        # Chunker should be closed but not the file
        assert not chunker.is_open
        assert not file_obj.closed
    # Now the file should be closed
    assert file_obj.closed

def test_anndata_chunker_file_ownership(temp_h5ad_file):
    """Test file ownership behavior with different initialization methods."""
    # Test with path (should own file)
    chunker_path = AnnDataChunker(temp_h5ad_file, None)
    assert chunker_path._owns_file
    
    # Test with file object (should not own file)
    with open(temp_h5ad_file, 'rb') as file_obj:
        chunker_file = AnnDataChunker(file_obj, None)
        assert not chunker_file._owns_file
        
        # Test operations with non-owned file
        chunker_file.open()
        assert chunker_file.is_open
        chunker_file.close()
        assert not chunker_file.is_open
        assert not file_obj.closed  # Should not close the file we don't own

def test_anndata_chunker_iter_chunks_with_file_object(temp_h5ad_file):
    """Test iter_chunks method with a file-like object."""
    with open(temp_h5ad_file, 'rb') as file_obj:
        with AnnDataChunker(file_obj, ['cell_type']) as chunker:
            chunk_size = 30
            chunks = list(chunker.iter_chunks(chunk_size))
            
            # Basic checks
            assert len(chunks) > 0
            assert all(isinstance(chunk, ad.AnnData) for chunk in chunks)
            assert all('cell_type' in chunk.obs.columns for chunk in chunks)
            
            # Check total rows
            total_rows = sum(chunk.n_obs for chunk in chunks)
            assert total_rows == len(chunker.obs)

def test_load_index_column(temp_h5ad_file):
    """Test loading of index column from h5ad file."""
    chunker = AnnDataChunker(temp_h5ad_file, ['cell_type'])
    with chunker:
        # Test default behavior
        assert chunker._index_column == "_index"  # Default value
        assert "_index" in chunker.obs_columns
        assert "cell_type" in chunker.obs_columns

def test_load_index_column_with_no_obs_columns(temp_h5ad_file):
    """Test loading of index column when no obs columns are specified."""
    chunker = AnnDataChunker(temp_h5ad_file, None)
    with chunker:
        assert chunker._index_column == "_index"  # Default value
        assert "_index" in chunker.obs_columns

def test_load_index_column_fallback(temp_h5ad_file):
    """Test index column loading fallback behavior."""
    with h5py.File(temp_h5ad_file, 'r+') as f:
        # Test fallback when neither _index nor index exists in attrs
        if "_index" in f["obs"].attrs:
            del f["obs"].attrs["_index"]
        if "index" in f["obs"].attrs:
            del f["obs"].attrs["index"]
            
        chunker = AnnDataChunker(temp_h5ad_file, ['cell_type'])
        with chunker:
            assert chunker._index_column == "_index"  # Should default to "_index"
            assert "_index" in chunker.obs_columns

def test_load_index_column_alternative(temp_h5ad_file):
    """Test loading alternative index column name."""
    with h5py.File(temp_h5ad_file, 'r+') as f:
        # Test when index (not _index) is in attrs
        # Test fallback when neither _index nor index exists in attrs
        if "_index" in f["obs"].attrs:
            del f["obs"].attrs["_index"]
        f["obs"].attrs["index"] = "alt_index"
        
        chunker = AnnDataChunker(temp_h5ad_file, ['cell_type'])
        with chunker:
            assert chunker._index_column == "alt_index"
            assert "alt_index" in chunker.obs_columns 