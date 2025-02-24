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

    # Create the reordered embedding matrix
    embedding_matrix = (
        merged_embeddings[embedding_cols].iloc[embedding_indices].values.T
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