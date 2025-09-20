#!/usr/bin/env python3
"""
Generate scGPT embeddings for CellXGene datasets.
Can process local H5AD files or download from S3.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_scgpt_model(model_path: Path, device: str = 'cuda'):
    """
    Load pre-trained scGPT model.
    
    Args:
        model_path: Path to model checkpoint directory
        device: Device for computation
        
    Returns:
        Tuple of (model, vocab)
    """
    try:
        import scgpt as scg
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
    except ImportError:
        logger.error("scGPT not installed. Please install with: pip install scgpt")
        sys.exit(1)
    
    logger.info(f"Loading scGPT model from {model_path}")
    
    # Load vocabulary
    vocab_file = model_path / 'vocab.json'
    if not vocab_file.exists():
        # Try alternative paths
        vocab_file = model_path / 'gene_vocab.json'
    
    if vocab_file.exists():
        vocab = GeneVocab.from_file(str(vocab_file))
        logger.info(f"Loaded vocabulary with {len(vocab)} genes")
    else:
        logger.error(f"Vocabulary file not found at {vocab_file}")
        return None, None
    
    # Load model checkpoint
    checkpoint_file = model_path / 'best_model.pt'
    if not checkpoint_file.exists():
        checkpoint_file = model_path / 'model.pt'
    
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # Get model config from checkpoint or use defaults
        model_config = checkpoint.get('model_config', {})
        
        # Initialize model with config
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_config.get('d_model', 512),
            nhead=model_config.get('nhead', 8),
            d_hid=model_config.get('d_hid', 2048),
            nlayers=model_config.get('nlayers', 12),
            vocab=vocab,
            dropout=model_config.get('dropout', 0.1),
            pad_token="<pad>",
            pad_value=0,
            do_mvc=model_config.get('do_mvc', False),
            do_dab=model_config.get('do_dab', False),
            use_batch_labels=model_config.get('use_batch_labels', False),
            explicit_zero_prob=model_config.get('explicit_zero_prob', False),
        )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        
        return model, vocab
    else:
        logger.error(f"Model checkpoint not found at {checkpoint_file}")
        return None, None


def preprocess_adata(adata, gene_vocab, max_genes: int = 3000):
    """
    Preprocess AnnData object for scGPT input.
    
    Args:
        adata: AnnData object
        gene_vocab: Gene vocabulary from model
        max_genes: Maximum number of genes to use
        
    Returns:
        Preprocessed expression matrix
    """
    import scanpy as sc
    
    # Basic preprocessing if not already done
    if 'normalized' not in adata.layers:
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Store normalized data
        adata.layers['normalized'] = adata.X.copy()
    
    # Select highly variable genes if needed
    if adata.n_vars > max_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes)
        adata = adata[:, adata.var.highly_variable]
    
    # Map genes to vocabulary
    gene_names = adata.var_names.str.upper().tolist()
    gene_ids = []
    
    for gene in gene_names:
        if gene in gene_vocab:
            gene_ids.append(gene_vocab[gene])
        else:
            # Try alternative names or use unknown token
            gene_ids.append(gene_vocab.get('<unk>', 0))
    
    return adata, gene_ids


def generate_embeddings(
    adata,
    model,
    gene_vocab,
    batch_size: int = 256,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate scGPT embeddings for cells in AnnData.
    
    Args:
        adata: Preprocessed AnnData object
        model: scGPT model
        gene_vocab: Gene vocabulary
        batch_size: Batch size for inference
        device: Device for computation
        
    Returns:
        Embedding matrix (n_cells, embedding_dim)
    """
    n_cells = adata.n_obs
    embeddings = []
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_cells, batch_size), desc="Generating embeddings"):
            batch_end = min(i + batch_size, n_cells)
            
            # Get batch data
            if hasattr(adata.X, 'toarray'):
                batch_expr = adata.X[i:batch_end].toarray()
            else:
                batch_expr = adata.X[i:batch_end]
            
            # Convert to torch tensor
            batch_tensor = torch.FloatTensor(batch_expr).to(device)
            
            # Generate embeddings
            # Note: Actual implementation depends on scGPT model interface
            batch_embeddings = model.encode(batch_tensor)
            
            # Move to CPU and store
            embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    
    return embeddings


def process_h5ad_file(
    h5ad_path: Path,
    model,
    gene_vocab,
    output_dir: Path,
    batch_size: int = 256,
    device: str = 'cuda'
) -> Dict:
    """
    Process a single H5AD file and generate embeddings.
    
    Args:
        h5ad_path: Path to H5AD file
        model: scGPT model
        gene_vocab: Gene vocabulary
        output_dir: Output directory
        batch_size: Batch size for inference
        device: Device for computation
        
    Returns:
        Processing statistics
    """
    import scanpy as sc
    
    logger.info(f"Processing {h5ad_path.name}")
    
    # Load H5AD
    adata = sc.read_h5ad(h5ad_path)
    initial_cells = adata.n_obs
    initial_genes = adata.n_vars
    
    logger.info(f"Loaded {initial_cells:,} cells, {initial_genes:,} genes")
    
    # Preprocess
    adata, gene_ids = preprocess_adata(adata, gene_vocab)
    
    # Generate embeddings
    embeddings = generate_embeddings(
        adata, model, gene_vocab, batch_size, device
    )
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'cell_id': adata.obs_names.values,
        'scgpt_embedding': list(embeddings)
    })
    
    # Add metadata if available
    for col in ['cell_type', 'tissue', 'donor_id']:
        if col in adata.obs.columns:
            output_df[col] = adata.obs[col].values
    
    # Save embeddings
    output_file = output_dir / f"{h5ad_path.stem}_scgpt_embeddings.parquet"
    output_df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved embeddings to {output_file}")
    
    # Save as PT format for faster loading
    pt_output = output_dir / f"{h5ad_path.stem}_scgpt_embeddings.pt"
    torch_dict = {
        'embeddings': torch.FloatTensor(embeddings),
        'cell_ids': output_df['cell_id'].tolist(),
        'metadata': {
            'n_cells': len(output_df),
            'embedding_dim': embeddings.shape[1],
            'source_file': h5ad_path.name,
            'created_at': datetime.now().isoformat()
        }
    }
    torch.save(torch_dict, pt_output, pickle_protocol=4)
    
    return {
        'file': h5ad_path.name,
        'cells_processed': len(output_df),
        'embedding_dim': embeddings.shape[1],
        'output_files': [output_file.name, pt_output.name]
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings')
    parser.add_argument('--input-dir', type=Path, help='Directory with H5AD files')
    parser.add_argument('--input-files', nargs='+', help='List of H5AD files')
    parser.add_argument('--model-path', type=Path, required=True, help='Path to scGPT model')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, gene_vocab = load_scgpt_model(args.model_path, args.device)
    if model is None:
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Get list of files to process
    if args.input_files:
        h5ad_files = [Path(f) for f in args.input_files]
    elif args.input_dir:
        h5ad_files = list(args.input_dir.glob('*.h5ad'))
    else:
        logger.error("Must specify either --input-dir or --input-files")
        sys.exit(1)
    
    # Limit files if requested
    if args.max_files:
        h5ad_files = h5ad_files[:args.max_files]
    
    logger.info(f"Found {len(h5ad_files)} H5AD files to process")
    
    # Process each file
    results = []
    for h5ad_path in h5ad_files:
        try:
            stats = process_h5ad_file(
                h5ad_path,
                model,
                gene_vocab,
                args.output_dir,
                args.batch_size,
                args.device
            )
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {h5ad_path}: {e}")
            continue
    
    # Save processing summary
    summary_file = args.output_dir / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Processed {len(results)} files")
    total_cells = sum(r['cells_processed'] for r in results)
    logger.info(f"Total cells: {total_cells:,}")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()