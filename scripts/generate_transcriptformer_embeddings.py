#!/usr/bin/env python
"""
Generate cell embeddings using Transcriptformer foundation model.

Transcriptformer is a foundation model for single-cell RNA-seq data from CZI AI.
It generates embeddings that capture cell state and biological context.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: str, tissue: str) -> anndata.AnnData:
    """Load Tabula Sapiens dataset for a specific tissue.

    Args:
        data_dir: Base directory containing tissue datasets
        tissue: Tissue name (e.g., 'Blood', 'Bone_Marrow')

    Returns:
        AnnData object with the tissue data
    """
    # Look for files matching the tissue name pattern
    data_path = Path(data_dir)
    pattern = f"*{tissue}*v2_curated.h5ad"
    matching_files = list(data_path.glob(pattern))

    if not matching_files:
        # Try alternative pattern
        pattern = f"TabulaSapiens_{tissue}.h5ad"
        matching_files = list(data_path.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"Dataset not found for tissue: {tissue} in {data_dir}")

    file_path = matching_files[0]

    logger.info(f"Loading dataset from {file_path}")
    adata = sc.read_h5ad(file_path)

    # Log dataset info
    n_cells, n_genes = adata.shape
    logger.info(f"Loaded {n_cells:,} cells with {n_genes:,} genes")

    # Check if ensembl_id is available
    if 'ensembl_id' not in adata.var.columns:
        logger.warning("ensembl_id not found in var columns. Available columns:")
        logger.warning(f"{list(adata.var.columns)}")
        # Try to use gene_ids if available
        if 'gene_ids' in adata.var.columns:
            logger.info("Using gene_ids as ensembl_id")
            adata.var['ensembl_id'] = adata.var['gene_ids']

    # Log cell type information
    if 'cell_type' in adata.obs.columns:
        n_types = adata.obs['cell_type'].nunique()
        logger.info(f"Found {n_types} unique cell types")

    return adata


def run_transcriptformer_inference(
    adata: anndata.AnnData,
    checkpoint_path: str,
    output_dir: str,
    tissue: str,
    model_variant: str = "tf_metazoa",
    batch_size: int = 8
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Run Transcriptformer inference using the CLI.

    Args:
        adata: AnnData object with expression data
        checkpoint_path: Path to Transcriptformer checkpoint
        output_dir: Directory for output files
        tissue: Tissue name for file naming
        model_variant: Model variant to use
        batch_size: Batch size for inference

    Returns:
        Tuple of (embeddings array, metadata DataFrame)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save temporary h5ad file for Transcriptformer
    temp_file = output_path / f"temp_{tissue}.h5ad"
    logger.info(f"Saving temporary file for inference: {temp_file}")
    adata.write_h5ad(temp_file)

    # Output file name
    output_h5ad = output_path / f"transcriptformer_{model_variant}_{tissue}_embeddings.h5ad"

    # Build inference command
    cmd = [
        "transcriptformer", "inference",
        "--checkpoint-path", checkpoint_path,
        "--data-file", str(temp_file),
        "--output-path", str(output_path),
        "--output-filename", output_h5ad.name,
        "--batch-size", str(batch_size),
        "--emb-type", "cell",  # Get cell-level embeddings
        "--precision", "16-mixed",  # Use mixed precision for efficiency
        "--filter-to-vocabs",  # Filter to model vocabulary
    ]

    # Check for gene column
    if 'ensembl_id' in adata.var.columns:
        cmd.extend(["--gene-col-name", "ensembl_id"])
    else:
        # Use default or gene_ids if available
        logger.warning("Using default gene column name")

    logger.info(f"Running Transcriptformer inference...")
    logger.info(f"Command: {' '.join(cmd)}")

    # Run inference using subprocess with real-time output
    import subprocess
    start_time = time.time()

    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output line by line
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to console in real-time
        output_lines.append(line)

    # Wait for process to complete
    process.wait()

    if process.returncode != 0:
        error_output = ''.join(output_lines)
        logger.error(f"Inference failed with return code {process.returncode}")
        raise RuntimeError(f"Transcriptformer inference failed: {error_output}")

    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed:.1f} seconds")

    # Load the output embeddings
    if not output_h5ad.exists():
        raise FileNotFoundError(f"Output file not found: {output_h5ad}")

    logger.info(f"Loading embeddings from {output_h5ad}")
    adata_emb = sc.read_h5ad(output_h5ad)

    # Extract embeddings
    if 'X_transcriptformer' in adata_emb.obsm:
        embeddings = adata_emb.obsm['X_transcriptformer']
    else:
        # Check other possible keys
        logger.warning("X_transcriptformer not found in obsm. Available keys:")
        logger.warning(f"{list(adata_emb.obsm.keys())}")
        # Try to find embedding key
        emb_keys = [k for k in adata_emb.obsm.keys() if 'emb' in k.lower() or 'transcript' in k.lower()]
        if emb_keys:
            embeddings = adata_emb.obsm[emb_keys[0]]
            logger.info(f"Using embeddings from {emb_keys[0]}")
        else:
            raise KeyError("Could not find embeddings in output file")

    # Get metadata
    metadata = adata_emb.obs.copy()

    # Clean up temporary file
    temp_file.unlink()

    return embeddings, metadata


def save_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str,
    tissue: str,
    model_variant: str
):
    """Save embeddings to Parquet file for consistency with scGPT outputs.

    Args:
        embeddings: Cell embedding matrix
        metadata: Cell metadata DataFrame
        output_dir: Output directory
        tissue: Tissue name
        model_variant: Model variant name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with embeddings
    n_cells, embedding_dim = embeddings.shape
    logger.info(f"Saving {n_cells:,} cell embeddings of dimension {embedding_dim}")

    # Create column names for embeddings
    embedding_cols = [f"emb_{i}" for i in range(embedding_dim)]

    # Create DataFrame
    df = pd.DataFrame(embeddings, columns=embedding_cols, index=metadata.index)

    # Add metadata columns
    for col in ['cell_type', 'cell_type_code']:
        if col in metadata.columns:
            df[col] = metadata[col]

    # Save to Parquet
    output_file = output_path / f"transcriptformer_{model_variant}_{tissue}_embeddings.parquet"
    df.to_parquet(output_file)
    logger.info(f"Saved embeddings to {output_file}")

    # Log file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {size_mb:.1f} MB")


def main():
    """Main entry point for Transcriptformer embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate cell embeddings using Transcriptformer"
    )
    parser.add_argument(
        "--tissue",
        type=str,
        required=True,
        help="Tissue name (e.g., Blood, Bone_Marrow)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing Tabula Sapiens datasets"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/transcriptformer",
        help="Directory containing Transcriptformer models"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="tf_metazoa",
        choices=["tf_sapiens", "tf_exemplar", "tf_metazoa"],
        help="Transcriptformer model variant to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cz_benchmark/embeddings/transcriptformer",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 70)
    logger.info("TRANSCRIPTFORMER EMBEDDING GENERATION")
    logger.info("=" * 70)
    logger.info(f"Tissue: {args.tissue}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Model variant: {args.model_variant}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 70)

    # Check model path
    checkpoint_path = Path(args.model_dir) / args.model_variant
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logger.info(f"Using checkpoint: {checkpoint_path}")

    try:
        # Load dataset
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Loading dataset")
        logger.info("="*50)
        adata = load_dataset(args.data_dir, args.tissue)

        # Run inference
        logger.info("\n" + "="*50)
        logger.info("STEP 2: Running Transcriptformer inference")
        logger.info("="*50)
        embeddings, metadata = run_transcriptformer_inference(
            adata=adata,
            checkpoint_path=str(checkpoint_path),
            output_dir=args.output_dir,
            tissue=args.tissue,
            model_variant=args.model_variant,
            batch_size=args.batch_size
        )

        # Save embeddings
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Saving embeddings")
        logger.info("="*50)
        save_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            output_dir=args.output_dir,
            tissue=args.tissue,
            model_variant=args.model_variant
        )

        logger.info("\n" + "="*70)
        logger.info(f"✓ Successfully generated embeddings for {args.tissue}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())