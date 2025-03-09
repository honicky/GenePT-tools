import numpy as np
import torch

def _get_embedding_indices(merged_embeddings, major_ensembl_ids):
    """
    Helper function to get embedding indices and valid indices for both numpy and torch versions.

    Args:
        merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings with 'ensembl_id' column
        major_ensembl_ids (pd.Series): Series of Ensembl IDs in the order they appear in expression matrix

    Returns:
        tuple: (embedding_cols, valid_indices, embedding_indices)
    """
    # Get the embedding values without the metadata columns
    embedding_cols = [
        col for col in merged_embeddings.columns if col not in ["ensembl_id"]
    ]

    # Create a mapping from major_ensembl_ids to column indices in cell_gene_matrix
    gene_idx_map = {gene_id: idx for idx, gene_id in enumerate(major_ensembl_ids)}

    # Find which embeddings correspond to genes in our expression matrix
    # and get their indices in the correct order
    valid_indices = []
    embedding_indices = []
    for i, ensembl_id in enumerate(merged_embeddings.ensembl_id):
        if ensembl_id in gene_idx_map:
            valid_indices.append(gene_idx_map[ensembl_id])
            embedding_indices.append(i)

    return embedding_cols, valid_indices, embedding_indices

def create_embedding_matrix(merged_embeddings, major_ensembl_ids):
    """
    Create a reordered embedding matrix that aligns gene embeddings with expression matrix columns.

    Args:
        merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings with 'ensembl_id' column
        major_ensembl_ids (pd.Series): Series of Ensembl IDs in the order they appear in expression matrix

    Returns:
        tuple: (embedding_matrix, valid_indices)
            - embedding_matrix: numpy array of shape (n_embedding_dims, n_valid_genes)
            - valid_indices: list of indices mapping to original expression matrix columns
    """
    embedding_cols, valid_indices, embedding_indices = _get_embedding_indices(
        merged_embeddings, major_ensembl_ids
    )
    
    # Create the reordered embedding matrix
    embedding_matrix = (
        merged_embeddings[embedding_cols].iloc[embedding_indices].values.T
    )

    return embedding_matrix, valid_indices

def create_embedding_matrix_torch(merged_embeddings, major_ensembl_ids, device='cpu'):
    """
    Create a reordered embedding matrix that aligns gene embeddings with expression matrix columns.
    PyTorch version that returns a torch.Tensor.

    Args:
        merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings with 'ensembl_id' column
        major_ensembl_ids (pd.Series): Series of Ensembl IDs in the order they appear in expression matrix
        device (str or torch.device): Device to place the tensor on ('cpu' or 'cuda')

    Returns:
        tuple: (embedding_matrix, valid_indices)
            - embedding_matrix: torch.Tensor of shape (n_embedding_dims, n_valid_genes)
            - valid_indices: list of indices mapping to original expression matrix columns
    """
    embedding_cols, valid_indices, embedding_indices = _get_embedding_indices(
        merged_embeddings, major_ensembl_ids
    )

    # Create the reordered embedding matrix as a PyTorch tensor on specified device
    embedding_matrix = torch.tensor(
        merged_embeddings[embedding_cols].iloc[embedding_indices].values.T,
        dtype=torch.float32,
        device=device
    )

    return embedding_matrix, valid_indices

def create_cell_embeddings(expression_matrix, embedding_matrix, valid_indices):
    """
    Create normalized cell embeddings from gene expression data and gene embeddings.

    Args:
        expression_matrix: scipy.sparse.csr_matrix or numpy array of shape (n_cells, n_genes)
        embedding_matrix: numpy array of shape (n_embedding_dims, n_valid_genes)
        valid_indices: list of indices to select genes that have embeddings

    Returns:
        numpy array of shape (n_cells, n_embedding_dims) containing normalized cell embeddings
    """
    # Select only the columns from expression matrix that have corresponding embeddings
    filtered_expression = expression_matrix[:, valid_indices]

    # Perform the matrix multiplication (n_cells x n_embedding_dimensions)
    cell_embeddings = filtered_expression @ embedding_matrix.T

    # Normalize the cell embeddings
    norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings = cell_embeddings / norms

    return cell_embeddings

def create_cell_embeddings_torch(expression_matrix, embedding_matrix, device='cpu'):
    """
    Create normalized cell embeddings using PyTorch operations.

    Args:
        expression_matrix: torch.sparse.FloatTensor in CSR format of shape (n_cells, n_genes)
        embedding_matrix: torch.Tensor of shape (n_embedding_dims, n_valid_genes)
        device (str or torch.device): Device to place the tensors on ('cpu' or 'cuda')

    Returns:
        torch.Tensor of shape (n_cells, n_embedding_dims) containing normalized cell embeddings
    """
    # Only move tensors if they're not already on the target device
    if expression_matrix.device != device:
        expression_matrix = expression_matrix.to(device)
    if embedding_matrix.device != device:
        embedding_matrix = embedding_matrix.to(device)
    
    # Perform sparse matrix multiplication
    cell_embeddings = torch.sparse.mm(expression_matrix,  embedding_matrix.T)
    
    # Normalize the cell embeddings
    norms = torch.norm(cell_embeddings, dim=1, keepdim=True)
    cell_embeddings = cell_embeddings / norms

    return cell_embeddings