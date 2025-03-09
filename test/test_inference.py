import pytest
import numpy as np
import torch
import pandas as pd
from scipy import sparse
from src.inference import (
    create_embedding_matrix,
    create_embedding_matrix_torch,
    create_cell_embeddings,
    create_cell_embeddings_torch,
)

@pytest.fixture
def sample_data():
    # Create sample embeddings DataFrame
    embeddings_data = {
        'ensembl_id': ['ENSG1', 'ENSG2', 'ENSG3', 'ENSG4'],
        'dim1': [0.1, 0.2, 0.3, 0.4],
        'dim2': [0.5, 0.6, 0.7, 0.8]
    }
    merged_embeddings = pd.DataFrame(embeddings_data)
    
    # Create sample gene IDs (note: ENSG3 is intentionally missing to test filtering)
    major_ensembl_ids = pd.Series(['ENSG1', 'ENSG2', 'ENSG4'])
    
    # Create sample expression matrix (3 cells x 3 genes)
    expression_matrix = sparse.csr_matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    return {
        'merged_embeddings': merged_embeddings,
        'major_ensembl_ids': major_ensembl_ids,
        'expression_matrix': expression_matrix
    }

def test_create_embedding_matrix(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix(
        sample_data['merged_embeddings'],
        sample_data['major_ensembl_ids']
    )
    
    # Check shapes
    assert embedding_matrix.shape == (2, 3)  # 2 dimensions, 3 valid genes
    assert len(valid_indices) == 3
    
    # Check values
    expected_matrix = np.array([
        [0.1, 0.2, 0.4],  # First embedding dimension
        [0.5, 0.6, 0.8]   # Second embedding dimension
    ])
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)
    
    # Check indices
    assert valid_indices == [0, 1, 2]  # Should match position in major_ensembl_ids

def test_create_embedding_matrix_torch(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix_torch(
        sample_data['merged_embeddings'],
        sample_data['major_ensembl_ids']
    )
    
    # Check type and shape
    assert isinstance(embedding_matrix, torch.Tensor)
    assert embedding_matrix.shape == (2, 3)
    assert len(valid_indices) == 3
    
    # Check values
    expected_matrix = torch.tensor([
        [0.1, 0.2, 0.4],
        [0.5, 0.6, 0.8]
    ], dtype=torch.float32)
    assert torch.allclose(embedding_matrix, expected_matrix)

def test_create_cell_embeddings(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix(
        sample_data['merged_embeddings'],
        sample_data['major_ensembl_ids']
    )
    
    cell_embeddings = create_cell_embeddings(
        sample_data['expression_matrix'],
        embedding_matrix,
        valid_indices
    )
    
    # Check shape
    assert cell_embeddings.shape == (3, 2)  # 3 cells, 2 dimensions
    
    # Check normalization
    norms = np.linalg.norm(cell_embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(3))

def test_create_cell_embeddings_torch(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix_torch(
        sample_data['merged_embeddings'],
        sample_data['major_ensembl_ids']
    )
    
    # Pre-filter the expression matrix
    filtered_expression = sample_data['expression_matrix'][:, valid_indices]
    
    # Convert scipy sparse matrix to torch sparse CSR tensor
    expression_tensor = torch.sparse_csr_tensor(
        torch.LongTensor(filtered_expression.indptr),
        torch.LongTensor(filtered_expression.indices),
        torch.FloatTensor(filtered_expression.data),
        size=filtered_expression.shape
    )
    
    cell_embeddings = create_cell_embeddings_torch(
        expression_tensor,
        embedding_matrix,
    )
    
    # Check type and shape
    assert isinstance(cell_embeddings, torch.Tensor)
    assert cell_embeddings.shape == (3, 2)
    
    # Check normalization
    norms = torch.norm(cell_embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))
    
    # Check actual values against numpy implementation
    numpy_embeddings = create_cell_embeddings(
        filtered_expression,  # use pre-filtered expression matrix
        embedding_matrix.cpu().numpy(),
        list(range(len(valid_indices)))  # since expression is already filtered
    )
    assert torch.allclose(cell_embeddings, torch.tensor(numpy_embeddings, dtype=torch.float32))

def test_device_handling():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Create small test data
        embeddings_data = {
            'ensembl_id': ['ENSG1'],
            'dim1': [0.1],
            'dim2': [0.5]
        }
        merged_embeddings = pd.DataFrame(embeddings_data)
        major_ensembl_ids = pd.Series(['ENSG1'])
        
        # Test device placement
        embedding_matrix, _ = create_embedding_matrix_torch(
            merged_embeddings,
            major_ensembl_ids,
            device=device
        )
        assert embedding_matrix.device.type == 'cuda' 