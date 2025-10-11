#!/usr/bin/env python
"""
Generate scGPT embeddings for Tabula Sapiens v2 tissues using the official scGPT API.

This script uses scgpt.tasks.embed_data for zero-shot embedding generation.
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import scanpy as sc
import scgpt as scg
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_tissue_name(filename: Path) -> str:
    """Extract tissue name from Tabula Sapiens v2 filename.
    
    Args:
        filename: Path to h5ad file
        
    Returns:
        Tissue name (e.g., 'Blood', 'Bone_Marrow')
    """
    # Format: homo_sapiens_{uuid}_{tissue}_v2_curated.h5ad
    name = filename.stem  # Remove .h5ad
    
    # Split on '_v2_curated' first to get the base name
    if '_v2_curated' in name:
        base = name.split('_v2_curated')[0]
        # Find tissue name after UUID
        parts = base.split('_')
        # Skip homo_sapiens and UUID, get the rest as tissue name
        if len(parts) >= 3 and parts[0] == 'homo' and parts[1] == 'sapiens':
            # UUID is parts[2], tissue starts at parts[3]
            # Skip the UUID part (format: 10df7690-6d10-4029-a47e-0f071bb2df83)
            for i, part in enumerate(parts):
                if '-' in part and len(part) == 36:  # UUID format
                    # Tissue name is everything after UUID
                    return '_'.join(parts[i+1:])
    
    # Fallback for older format: TabulaSapiens_v2_{tissue}.h5ad
    if '_v2_' in name:
        parts = name.split('_v2_')
        if len(parts) == 2:
            return parts[1]
    
    return name


def process_tissue(
    tissue_name: str,
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: Optional[str] = None
) -> bool:
    """Process a single tissue using scGPT embed_data API.
    
    Args:
        tissue_name: Name of tissue to process
        data_dir: Directory containing h5ad files
        model_dir: Directory containing scGPT model
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        device: Device for computation (optional)
        
    Returns:
        True if successful
    """
    # Find tissue file - try multiple naming patterns
    possible_patterns = [
        f"*_{tissue_name}_v2_curated.h5ad",  # New format
        f"TabulaSapiens_v2_{tissue_name}.h5ad",  # Old format
        f"*{tissue_name}*.h5ad"  # Fallback
    ]
    
    h5ad_file = None
    for pattern in possible_patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            h5ad_file = matches[0]
            break
    
    if h5ad_file is None or not h5ad_file.exists():
        logger.error(f"Tissue file not found for: {tissue_name}")
        logger.error(f"Searched in: {data_dir}")
        return False
    
    logger.info(f"Processing tissue: {tissue_name} from {h5ad_file.name}")
    
    try:
        # Load data
        adata = sc.read_h5ad(h5ad_file)
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Clean gene names in feature_name column
        if 'feature_name' in adata.var.columns:
            original_names = adata.var['feature_name'].tolist()
            clean_names = []
            
            for gene in original_names:
                if '_ENSG' in gene:
                    # Extract symbol before _ENSG (e.g., "MATR3_ENSG00000015479" -> "MATR3")
                    clean_name = gene.split('_ENSG')[0]
                    clean_names.append(clean_name)
                elif gene.startswith('ENSG'):
                    # Keep ENSG IDs as is for now (scGPT might handle them)
                    clean_names.append(gene)
                else:
                    # Already clean symbol
                    clean_names.append(gene)
            
            # Count how many were cleaned
            n_cleaned = sum(1 for i, j in zip(original_names, clean_names) if i != j)
            if n_cleaned > 0:
                logger.info(f"Cleaned {n_cleaned} gene names (removed _ENSG suffixes)")
            
            # Update the feature_name column
            adata.var['feature_name'] = clean_names
            gene_col = 'feature_name'
            logger.info(f"Using gene column: {gene_col}")
        else:
            # Fallback to index if feature_name not available
            gene_col = 'index'
            logger.info(f"Using gene index as gene column")
        
        # Determine cell type column for metadata
        obs_to_save = []
        if 'cell_type' in adata.obs.columns:
            obs_to_save.append('cell_type')
        if 'donor_id' in adata.obs.columns:
            obs_to_save.append('donor_id')
        if 'tissue' in adata.obs.columns:
            obs_to_save.append('tissue')
        
        logger.info(f"Will save metadata columns: {obs_to_save}")
        
        # Generate embeddings using scGPT API
        logger.info("Generating embeddings with scGPT...")
        
        # Set device if specified
        if device:
            scg.utils.set_seed(42)  # Set seed for reproducibility
            # Note: device setting might be handled internally by embed_data
        
        # Use scGPT's embed_data function
        embed_adata = scg.tasks.embed_data(
            adata,
            model_dir,
            gene_col=gene_col,
            obs_to_save=obs_to_save if obs_to_save else None,
            batch_size=batch_size,
            return_new_adata=True,
        )
        
        logger.info(f"Generated embeddings with shape: {embed_adata.X.shape}")
        
        # Extract embeddings and metadata
        import numpy as np
        
        # Get embeddings (should be in X of returned adata)
        if hasattr(embed_adata.X, 'toarray'):
            embeddings = embed_adata.X.toarray()
        else:
            embeddings = np.array(embed_adata.X)
        
        # Create DataFrame with embeddings and metadata
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
        )
        
        # Add cell IDs
        embedding_df['cell_id'] = embed_adata.obs.index.tolist()
        
        # Add saved metadata
        for col in obs_to_save:
            if col in embed_adata.obs.columns:
                embedding_df[col] = embed_adata.obs[col].tolist()
        
        # Reorder columns to put metadata first
        metadata_cols = ['cell_id'] + [col for col in obs_to_save if col in embedding_df.columns]
        embedding_cols = [col for col in embedding_df.columns if col.startswith('embedding_')]
        embedding_df = embedding_df[metadata_cols + embedding_cols]
        
        # Save to parquet
        output_path = output_dir / f"scgpt_{tissue_name}_embeddings.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_df.to_parquet(output_path, index=False, compression='snappy')
        
        # Log file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(embedding_df)} embeddings to {output_path} ({file_size_mb:.1f} MB)")
        
        # Clean up memory
        del adata, embed_adata, embeddings, embedding_df
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {tissue_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate scGPT embeddings using official API")
    parser.add_argument(
        '--tissue',
        type=str,
        default='Bone_Marrow',
        help='Tissue name or "all" for all target tissues'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark'),
        help='Directory containing h5ad files'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('models/scgpt'),
        help='Directory containing scGPT model'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/cz_benchmark/embeddings/scgpt'),
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps' if torch.backends.mps.is_available() else 'cpu',
        help='Device for computation'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing embeddings'
    )
    
    args = parser.parse_args()
    
    # Target tissues
    target_tissues = ["Blood", "Bone_Marrow", "Lung", "Mammary", "Thymus"]
    
    # Determine which tissues to process
    if args.tissue == 'all':
        tissues_to_process = target_tissues
    else:
        if args.tissue not in target_tissues:
            logger.warning(f"{args.tissue} not in target list, processing anyway")
        tissues_to_process = [args.tissue]
    
    # Check if model directory exists
    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        logger.info("Please ensure the scGPT model is in the correct directory")
        sys.exit(1)
    
    # Check for model files
    expected_files = ['vocab.json', 'best_model.pt']
    missing_files = [f for f in expected_files if not (args.model_dir / f).exists()]
    if missing_files:
        logger.warning(f"Some expected files are missing: {missing_files}")
        logger.info("The embed_data function will try to load the model anyway")
    
    # Process tissues
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for tissue in tissues_to_process:
        # Check if already processed (for resume)
        output_path = args.output_dir / f"scgpt_{tissue}_embeddings.parquet"
        if args.resume and output_path.exists():
            logger.info(f"Skipping {tissue} - already processed")
            success_count += 1
            continue
        
        if process_tissue(
            tissue, 
            args.data_dir, 
            args.model_dir, 
            args.output_dir, 
            args.batch_size,
            args.device
        ):
            success_count += 1
            logger.info(f"Successfully processed {tissue}")
        else:
            logger.error(f"Failed to process {tissue}")
    
    # Summary
    logger.info(f"Processed {success_count}/{len(tissues_to_process)} tissues successfully")
    
    if success_count == len(tissues_to_process):
        logger.info("All tissues processed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tissues failed processing")
        sys.exit(1)


if __name__ == "__main__":
    main()