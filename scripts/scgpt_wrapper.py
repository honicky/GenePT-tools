#!/usr/bin/env python3
"""
AWS Batch wrapper for scGPT embedding generation.
Handles S3 downloads, processes H5AD files, and uploads results.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
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


class ScGPTEmbeddingGenerator:
    """Generate scGPT embeddings for H5AD files from S3."""
    
    def __init__(
        self,
        model_path: str,
        output_bucket: str,
        output_prefix: str,
        batch_size: int = 256,
        device: str = 'cuda',
        aws_profile: Optional[str] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_path: S3 or local path to scGPT model checkpoint
            output_bucket: S3 bucket for output
            output_prefix: S3 prefix for output files
            batch_size: Batch size for inference
            device: Device for computation (cuda/cpu)
            aws_profile: AWS profile to use (optional)
        """
        self.model_path = model_path
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix
        self.batch_size = batch_size
        self.device = device
        
        # Initialize S3 client
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            self.s3_client = session.client('s3')
        else:
            self.s3_client = boto3.client('s3')
        
        # Load model
        self.model = None
        self.gene_vocab = None
        
    def download_from_s3(self, s3_path: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
            self.s3_client.download_file(bucket, key, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def upload_to_s3(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3."""
        try:
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
            self.s3_client.upload_file(local_path, bucket, key)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def load_model(self):
        """Load scGPT model and vocabulary."""
        import scgpt as scg
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Download model if on S3
        model_dir = Path('/tmp/scgpt_model')
        model_dir.mkdir(exist_ok=True)
        
        if self.model_path.startswith('s3://'):
            # Download model files
            # Assuming model checkpoint structure
            files_to_download = [
                'best_model.pt',
                'vocab.json',
                'config.json'
            ]
            
            for file_name in files_to_download:
                s3_file = f"{self.model_path.rstrip('/')}/{file_name}"
                local_file = model_dir / file_name
                if not self.download_from_s3(s3_file, str(local_file)):
                    logger.warning(f"Could not download {file_name}")
            
            model_path = model_dir
        else:
            model_path = Path(self.model_path)
        
        # Load vocabulary
        vocab_file = model_path / 'vocab.json'
        if vocab_file.exists():
            self.gene_vocab = GeneVocab.from_file(str(vocab_file))
            logger.info(f"Loaded vocabulary with {len(self.gene_vocab)} genes")
        
        # Load model
        checkpoint_file = model_path / 'best_model.pt'
        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            # Initialize model (adjust parameters based on checkpoint)
            self.model = TransformerModel(
                ntoken=len(self.gene_vocab),
                d_model=512,
                nhead=8,
                d_hid=2048,
                nlayers=12,
                vocab=self.gene_vocab,
                dropout=0.1,
                pad_token="<pad>",
                pad_value=0,
                do_mvc=False,
                do_dab=False,
                use_batch_labels=False,
                explicit_zero_prob=False,
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_file}")
    
    def process_h5ad(self, h5ad_path: str) -> pd.DataFrame:
        """
        Process a single H5AD file and generate embeddings.
        
        Args:
            h5ad_path: Path to H5AD file
            
        Returns:
            DataFrame with cell IDs and embeddings
        """
        import scanpy as sc
        import anndata
        import scgpt as scg
        
        logger.info(f"Processing {h5ad_path}")
        
        # Load H5AD file
        adata = sc.read_h5ad(h5ad_path)
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Generate embeddings using scGPT
        # This is the simplified version - actual implementation would need
        # proper preprocessing and gene mapping
        with torch.no_grad():
            # Use scGPT's embed_data function
            embed_adata = scg.tasks.embed_data(
                adata,
                model_dir=Path(self.model_path),
                gene_col='feature_name',  # Adjust based on actual data
                batch_size=self.batch_size,
                device=self.device
            )
        
        # Extract embeddings
        embeddings = embed_adata.obsm['X_scGPT']
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'cell_id': adata.obs.index.values,
            'scgpt_embedding': list(embeddings)
        })
        
        # Add metadata if available
        if 'cell_type' in adata.obs.columns:
            output_df['cell_type'] = adata.obs['cell_type'].values
        
        logger.info(f"Generated embeddings for {len(output_df)} cells")
        return output_df
    
    def process_file_list(self, file_list: List[str]) -> Dict[str, int]:
        """
        Process a list of H5AD files from S3.
        
        Args:
            file_list: List of S3 paths to H5AD files
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_files': len(file_list),
            'processed_files': 0,
            'failed_files': 0,
            'total_cells': 0
        }
        
        # Load model once
        if self.model is None:
            self.load_model()
        
        # Process each file
        for s3_path in tqdm(file_list, desc="Processing files"):
            try:
                # Download H5AD file
                local_h5ad = tempfile.mktemp(suffix='.h5ad')
                if not self.download_from_s3(s3_path, local_h5ad):
                    stats['failed_files'] += 1
                    continue
                
                # Process file
                embeddings_df = self.process_h5ad(local_h5ad)
                stats['total_cells'] += len(embeddings_df)
                
                # Save embeddings
                file_name = Path(s3_path).stem
                output_file = tempfile.mktemp(suffix='.parquet')
                embeddings_df.to_parquet(output_file, index=False)
                
                # Upload to S3
                s3_output = f"s3://{self.output_bucket}/{self.output_prefix}/{file_name}_scgpt_embeddings.parquet"
                if self.upload_to_s3(output_file, s3_output):
                    stats['processed_files'] += 1
                else:
                    stats['failed_files'] += 1
                
                # Cleanup
                os.remove(local_h5ad)
                os.remove(output_file)
                
            except Exception as e:
                logger.error(f"Failed to process {s3_path}: {e}")
                stats['failed_files'] += 1
        
        return stats


def main():
    """Main entry point for AWS Batch."""
    parser = argparse.ArgumentParser(description='Generate scGPT embeddings for H5AD files')
    parser.add_argument('--input-list', required=True, help='S3 path to JSON file with H5AD file list')
    parser.add_argument('--model-path', required=True, help='S3 or local path to scGPT model')
    parser.add_argument('--output-bucket', required=True, help='S3 bucket for output')
    parser.add_argument('--output-prefix', required=True, help='S3 prefix for output files')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    parser.add_argument('--device', default='cuda', help='Device for computation')
    
    args = parser.parse_args()
    
    # Download input list
    if args.input_list.startswith('s3://'):
        local_list = '/tmp/input_list.json'
        s3 = boto3.client('s3')
        parsed = urlparse(args.input_list)
        s3.download_file(parsed.netloc, parsed.path.lstrip('/'), local_list)
    else:
        local_list = args.input_list
    
    # Load file list
    with open(local_list, 'r') as f:
        file_list = json.load(f)
    
    logger.info(f"Processing {len(file_list)} H5AD files")
    
    # Initialize generator
    generator = ScGPTEmbeddingGenerator(
        model_path=args.model_path,
        output_bucket=args.output_bucket,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Process files
    stats = generator.process_file_list(file_list)
    
    # Log results
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Processed: {stats['processed_files']}")
    logger.info(f"Failed: {stats['failed_files']}")
    logger.info(f"Total cells: {stats['total_cells']:,}")
    logger.info("=" * 60)
    
    # Exit with error code if any files failed
    if stats['failed_files'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()