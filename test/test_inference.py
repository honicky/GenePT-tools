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
    # Update sample embeddings DataFrame to use numeric column names
    embeddings_data = {
        "ensembl_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        "0": [0.1, 0.2, 0.3, 0.4],
        "1": [0.5, 0.6, 0.7, 0.8],
        "metadata": ["a", "b", "c", "d"],
    }
    index = ["GENE1", "GENE2", "GENE3", "GENE4"]
    merged_embeddings = pd.DataFrame(embeddings_data, index=index)
    major_ensembl_ids = pd.Series(["ENSG1", "ENSG2", "ENSG4"])
    major_gene_ids = pd.Series(["GENE1", "GENE2", "GENE4"])

    expression_matrix = sparse.csr_matrix(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    return {
        "merged_embeddings": merged_embeddings,
        "major_ensembl_ids": major_ensembl_ids,
        "major_gene_ids": major_gene_ids,
        "expression_matrix": expression_matrix,
    }


def test_create_embedding_matrix(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix(
        sample_data["merged_embeddings"],
        sample_data["major_ensembl_ids"],
        id_column="ensembl_id",
    )

    # Check shapes
    assert embedding_matrix.shape == (2, 3)  # 2 dimensions, 3 valid genes
    assert len(valid_indices) == 3

    # Check values
    expected_matrix = np.array(
        [
            [0.1, 0.2, 0.4],  # First embedding dimension
            [0.5, 0.6, 0.8],  # Second embedding dimension
        ]
    )
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)

    # Check indices
    assert valid_indices == [0, 1, 2]  # Should match position in major_ensembl_ids


def test_create_embedding_matrix_torch(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix_torch(
        sample_data["merged_embeddings"],
        sample_data["major_gene_ids"],
    )

    # Check type and shape
    assert isinstance(embedding_matrix, torch.Tensor)
    assert embedding_matrix.shape == (2, 3)
    assert len(valid_indices) == 3

    # Check values
    expected_matrix = torch.tensor(
        [[0.1, 0.2, 0.4], [0.5, 0.6, 0.8]], dtype=torch.float32
    )
    assert torch.allclose(embedding_matrix, expected_matrix)


def test_create_cell_embeddings(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix(
        sample_data["merged_embeddings"], sample_data["major_gene_ids"]
    )

    cell_embeddings = create_cell_embeddings(
        sample_data["expression_matrix"], embedding_matrix, valid_indices
    )

    # Check shape
    assert cell_embeddings.shape == (3, 2)  # 3 cells, 2 dimensions

    # Check normalization
    norms = np.linalg.norm(cell_embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(3))


def test_create_cell_embeddings_torch(sample_data):
    embedding_matrix, valid_indices = create_embedding_matrix_torch(
        sample_data["merged_embeddings"],
        sample_data["major_gene_ids"],
    )

    # Pre-filter the expression matrix
    filtered_expression = sample_data["expression_matrix"][:, valid_indices]

    # Convert scipy sparse matrix to torch sparse CSR tensor
    expression_tensor = torch.sparse_csr_tensor(
        torch.LongTensor(filtered_expression.indptr),
        torch.LongTensor(filtered_expression.indices),
        torch.FloatTensor(filtered_expression.data),
        size=filtered_expression.shape,
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
        list(range(len(valid_indices))),  # since expression is already filtered
    )
    assert torch.allclose(
        cell_embeddings, torch.tensor(numpy_embeddings, dtype=torch.float32)
    )


def test_device_handling():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Create small test data
        embeddings_data = {"ensembl_id": ["ENSG1"], "dim1": [0.1], "dim2": [0.5]}
        merged_embeddings = pd.DataFrame(embeddings_data)
        major_ensembl_ids = pd.Series(["ENSG1"])

        # Test device placement
        embedding_matrix, _ = create_embedding_matrix_torch(
            merged_embeddings, major_ensembl_ids, device=device
        )
        assert embedding_matrix.device.type == "cuda"


def test_create_embedding_matrix_with_index():
    # Create sample embeddings DataFrame with index as gene IDs
    embeddings_data = {
        "0": [0.1, 0.2, 0.3, 0.4],
        "1": [0.5, 0.6, 0.7, 0.8],
        "metadata": ["a", "b", "c", "d"],  # Should be ignored
    }
    merged_embeddings = pd.DataFrame(
        embeddings_data, index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    )
    major_ensembl_ids = pd.Series(["ENSG1", "ENSG2", "ENSG4"])

    embedding_matrix, valid_indices = create_embedding_matrix(
        merged_embeddings, major_ensembl_ids
    )

    # Check shapes
    assert embedding_matrix.shape == (2, 3)  # 2 dimensions, 3 valid genes
    assert len(valid_indices) == 3

    # Check values
    expected_matrix = np.array(
        [
            [0.1, 0.2, 0.4],  # First embedding dimension
            [0.5, 0.6, 0.8],  # Second embedding dimension
        ]
    )
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)

    # Check indices
    assert valid_indices == [0, 1, 2]  # Should match position in major_ensembl_ids


def test_create_embedding_matrix_with_column():
    # Test with explicitly specified column
    embeddings_data = {
        "gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],  # Different column name
        "0": [0.1, 0.2, 0.3, 0.4],
        "1": [0.5, 0.6, 0.7, 0.8],
        "metadata": ["a", "b", "c", "d"],
    }
    merged_embeddings = pd.DataFrame(embeddings_data)
    major_ensembl_ids = pd.Series(["ENSG1", "ENSG2", "ENSG4"])

    embedding_matrix, valid_indices = create_embedding_matrix(
        merged_embeddings, major_ensembl_ids, id_column="gene_id"
    )

    # Check shapes and values
    assert embedding_matrix.shape == (2, 3)
    expected_matrix = np.array([[0.1, 0.2, 0.4], [0.5, 0.6, 0.8]])
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)


def test_embedding_column_filtering():
    # Test that only numeric columns are used
    embeddings_data = {
        "ensembl_id": ["ENSG1", "ENSG2"],
        "0": [0.1, 0.2],
        "1": [0.3, 0.4],
        "dim2": [0.5, 0.6],  # Should be ignored
        "-1": [0.7, 0.8],  # Should be ignored (negative)
        "metadata": ["a", "b"],
    }
    merged_embeddings = pd.DataFrame(embeddings_data)
    major_ensembl_ids = pd.Series(["ENSG1", "ENSG2"])

    embedding_matrix, valid_indices = create_embedding_matrix(
        merged_embeddings, major_ensembl_ids, id_column="ensembl_id"
    )

    # Check that only '0' and '1' columns were used
    assert embedding_matrix.shape == (2, 2)
    expected_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)


def test_create_embedding_matrix_with_gene_id_index():
    # Create sample embeddings DataFrame with gene IDs as index
    embeddings_data = {
        "0": [0.1, 0.2, 0.3, 0.4],
        "1": [0.5, 0.6, 0.7, 0.8],
        "2": [0.9, 1.0, 1.1, 1.2],
        "metadata": ["a", "b", "c", "d"],  # Should be ignored
    }
    index = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    merged_embeddings = pd.DataFrame(embeddings_data, index=index)

    # Create major_ensembl_ids with some genes missing and in different order
    major_ensembl_ids = pd.Series(["ENSG2", "ENSG4", "ENSG1"])

    # Test numpy version
    embedding_matrix, valid_indices = create_embedding_matrix(
        merged_embeddings, major_ensembl_ids
    )

    # Check shapes
    assert embedding_matrix.shape == (3, 3)  # 3 dimensions, 3 valid genes

    # Check values - should be ordered according to major_ensembl_ids
    expected_matrix = np.array(
        [
            [0.2, 0.4, 0.1],  # Values for ENSG2, ENSG4, ENSG1 from dimension 0
            [0.6, 0.8, 0.5],  # Values for ENSG2, ENSG4, ENSG1 from dimension 1
            [1.0, 1.2, 0.9],  # Values for ENSG2, ENSG4, ENSG1 from dimension 2
        ]
    )
    np.testing.assert_array_almost_equal(embedding_matrix, expected_matrix)

    # Check indices
    assert valid_indices == [0, 1, 2]  # Should match position in major_ensembl_ids

    # Test PyTorch version
    embedding_matrix_torch, valid_indices_torch = create_embedding_matrix_torch(
        merged_embeddings, major_ensembl_ids
    )

    # Check torch tensor
    assert isinstance(embedding_matrix_torch, torch.Tensor)
    assert embedding_matrix_torch.shape == (3, 3)
    assert torch.allclose(
        embedding_matrix_torch, torch.tensor(expected_matrix, dtype=torch.float32)
    )
    assert valid_indices_torch == valid_indices
