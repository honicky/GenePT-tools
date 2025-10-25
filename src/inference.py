import numpy as np

try:
  import torch

  _torch_available = True
except ImportError:
  _torch_available = False
  pass


def _get_embedding_indices(merged_embeddings, selected_gene_ids, id_column=None):
  """
    Helper function to get embedding indices and valid indices for both numpy and torch versions.

    Args:
        merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings
        selected_gene_ids (pd.Series): Series of Ensembl IDs in the order they appear in expression matrix
        id_column (str, optional): Column name containing gene IDs. If None, uses DataFrame index.

    Returns:
        tuple: (embedding_cols, valid_indices, embedding_indices)
    """
  # Get the embedding values columns (must be non-negative integers)
  embedding_cols = [
    col for col in merged_embeddings.columns if str(col).isdigit() and int(col) >= 0
  ]

  # Get the gene IDs either from the specified column or index
  gene_ids = (
    merged_embeddings[id_column] if id_column is not None else merged_embeddings.index)

  # Create a mapping from gene IDs to their position in merged_embeddings
  gene_pos_map = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

  # Find which genes in selected_gene_ids exist in our embeddings
  valid_indices = []
  embedding_indices = []

  for idx, gene_id in enumerate(selected_gene_ids):
    if gene_id in gene_pos_map:
      valid_indices.append(idx)
      embedding_indices.append(gene_pos_map[gene_id])

  return embedding_cols, valid_indices, embedding_indices


def create_embedding_matrix(merged_embeddings, selected_gene_ids, id_column=None):
  """
    Create a reordered embedding matrix that aligns gene embeddings with expression matrix columns.

    Args:
        merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings
        selected_gene_ids (pd.Series): Series of Ensembl IDs in the order they appear in expression matrix
        id_column (str, optional): Column name containing gene IDs. If None, uses DataFrame index.

    Returns:
        tuple: (embedding_matrix, valid_indices)
            - embedding_matrix: numpy array of shape (n_embedding_dims, n_valid_genes)
            - valid_indices: list of indices mapping to original expression matrix columns
    """
  print(f"Creating embedding matrix for {len(selected_gene_ids)} genes")
  embedding_cols, valid_indices, embedding_indices = _get_embedding_indices(
    merged_embeddings, selected_gene_ids, id_column)

  # Create the reordered embedding matrix
  print(f"Selecting embedding inidices")
  embedding_matrix = (
    merged_embeddings[embedding_cols].iloc[embedding_indices].values.T)

  print(f"Embedding matrix shape: {embedding_matrix.shape}")
  return embedding_matrix, valid_indices


# Only define torch-related functions if torch was successfully imported
if _torch_available:

  def create_embedding_matrix_torch(merged_embeddings,
                                    selected_gene_ids,
                                    device="cpu",
                                    id_column=None):
    """
        Create a reordered embedding matrix that aligns gene embeddings with expression matrix columns.
        PyTorch version that returns a torch.Tensor.

        Args:
            merged_embeddings (pd.DataFrame): DataFrame containing gene embeddings
            selected_gene_ids (pd.Series): Series of Ensembl/Gene IDs in the order they appear in expression matrix
            device (str or torch.device): Device to place the tensor on ('cpu' or 'cuda')
            id_column (str, optional): Column name containing gene IDs. If None, uses DataFrame index.

        Returns:
            tuple: (embedding_matrix, valid_indices)
                - embedding_matrix: torch.Tensor of shape (n_embedding_dims, n_valid_genes)
                - valid_indices: list of indices mapping to original expression matrix columns
        """
    embedding_cols, valid_indices, embedding_indices = _get_embedding_indices(
      merged_embeddings, selected_gene_ids, id_column)

    print(f"Creating reordered embedding matrix")
    # Create the reordered embedding matrix as a PyTorch tensor on specified device
    embedding_matrix = torch.tensor(
      merged_embeddings[embedding_cols].iloc[embedding_indices].values.T,
      dtype=torch.float32,
      device=device,
    )

    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix, valid_indices

  def create_cell_embeddings_torch(expression_matrix, embedding_matrix, device="cpu"):
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
    cell_embeddings = torch.sparse.mm(expression_matrix, embedding_matrix.T)

    # Normalize the cell embeddings, avoiding division by zero
    norms = torch.norm(cell_embeddings, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    cell_embeddings = cell_embeddings / norms

    return cell_embeddings


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

  # Normalize the cell embeddings, avoiding division by zero
  norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
  norms[norms == 0] = 1
  cell_embeddings = cell_embeddings / norms

  return cell_embeddings
