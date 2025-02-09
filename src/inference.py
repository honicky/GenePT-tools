import numpy as np
import pandas as pd


def create_gene_embedding_matrix(
    gene_names: list[str], gene_embedding: pd.DataFrame
) -> np.ndarray:
    """
    Create a lookup matrix for gene embeddings with vectorized operations.

    Args:
        gene_names (list): List of gene names to look up
        gene_embedding (pd.DataFrame): DataFrame containing gene embeddings with genes as index

    Returns:
        np.ndarray: Matrix of gene embeddings with shape (len(gene_names), embedding_dim)
    """
    EMBED_DIM = gene_embedding.shape[1]

    # Initialize zero matrix directly with numpy
    lookup_embed = np.zeros(shape=(len(gene_names), EMBED_DIM))

    # Fill in matching genes one by one to match notebook behavior
    count_missing = 0
    for i, gene in enumerate(gene_names):
        if gene in gene_embedding.index:
            lookup_embed[i, :] = gene_embedding.loc[gene]
        else:
            count_missing += 1

    print(
        f"Unable to match {count_missing} out of {len(gene_names)} genes in the GenePT-w embedding"
    )

    return lookup_embed


def gene_pt_w_embedding(
    gene_expression_counts: pd.DataFrame,
    experiment_ids: list[str],
    gene_embedding: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate the normalized GenePT-w embedding.

    Args:
        gene_expression_counts (pd.DataFrame): DataFrame containing gene expression counts with gene names as index
        experiment_ids (list[str]): List of experiment IDs to include in the embedding
        gene_embedding (pd.DataFrame): DataFrame containing gene embeddings with gene names as index

    Returns:
        np.ndarray: Normalized GenePT-w embedding
    """
    lookup_embed = create_gene_embedding_matrix(
        gene_expression_counts.gene_name, gene_embedding
    )
    # print(f"gene_expression_counts[experiment_ids].T.shape: {gene_expression_counts[experiment_ids].T.shape}")
    # print(f"lookup_embed.shape: {lookup_embed.shape}")
    unnormalized_gene_pt_w_embedding = np.dot(
        gene_expression_counts[experiment_ids].T, lookup_embed
    )
    # print(f"unnormalized_gene_pt_w_embedding.shape: {unnormalized_gene_pt_w_embedding.shape}")
    normalized_gene_pt_w_embedding = unnormalized_gene_pt_w_embedding / np.linalg.norm(
        unnormalized_gene_pt_w_embedding, axis=1, keepdims=True
    )
    return pd.DataFrame(normalized_gene_pt_w_embedding, index=experiment_ids)


def gene_pt_w_embedding_normalized(
    gene_expression_counts: pd.DataFrame,
    experiment_ids: list[str],
    gene_embedding: pd.DataFrame,
) -> np.ndarray:
    return gene_pt_w_embedding(
        gene_expression_counts, experiment_ids, gene_embedding
    ) / np.linalg.norm(
        gene_pt_w_embedding(gene_expression_counts, experiment_ids, gene_embedding),
        axis=1,
        keepdims=True,
    )
