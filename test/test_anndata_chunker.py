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
        'counts': np.random.randint(100, 1000, n_obs)
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
    assert chunker.file_path == temp_h5ad_file
    assert chunker.obs_columns == ['cell_type', 'condition']
    
    # Test with None obs_columns
    chunker = AnnDataChunker(temp_h5ad_file, None)
    assert chunker.obs_columns is None
    
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