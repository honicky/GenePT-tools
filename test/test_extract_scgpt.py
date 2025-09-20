#!/usr/bin/env python3
"""
Test script for scGPT embedding extraction.
Tests with a small subset of data first.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_scgpt_training_embeddings import ScGPTEmbeddingExtractor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_test_data(test_dir: Path):
    """Create small test datasets for validation."""
    
    # Create test training batch
    training_dir = test_dir / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    n_cells = 100
    cell_ids = [f"cell_{i:04d}" for i in range(n_cells)]
    origin_files = ['file1.h5ad'] * 50 + ['file2.h5ad'] * 50
    
    training_df = pd.DataFrame({
        'cell_id': cell_ids,
        'origin_file': origin_files,
        'cell_type': ['T cell'] * 30 + ['B cell'] * 20 + ['Monocyte'] * 30 + ['NK cell'] * 20,
        'cell_type_code': [0] * 30 + [1] * 20 + [2] * 30 + [3] * 20,
        'embedding': [np.random.randn(3072).tolist() for _ in range(n_cells)]
    })
    
    batch_path = training_dir / 'batch_0000.parquet'
    training_df.to_parquet(batch_path, index=False)
    logger.info(f"Created test training batch: {batch_path}")
    
    # Create test scGPT embeddings
    scgpt_dir = test_dir / 'scgpt'
    scgpt_dir.mkdir(parents=True, exist_ok=True)
    
    for origin_file in ['file1', 'file2']:
        origin_cells = training_df[training_df['origin_file'] == f'{origin_file}.h5ad']['cell_id'].tolist()
        scgpt_df = pd.DataFrame({
            'cell_id': origin_cells,
            'scgpt_embedding': [np.random.randn(512).tolist() for _ in range(len(origin_cells))]
        })
        
        scgpt_path = scgpt_dir / f'{origin_file}_scgpt.parquet'
        scgpt_df.to_parquet(scgpt_path, index=False)
        logger.info(f"Created test scGPT embeddings: {scgpt_path}")
    
    return training_dir, scgpt_dir


def validate_output(output_dir: Path, format: str = 'both'):
    """Validate the output files."""
    
    logger.info("Validating output...")
    
    # Check parquet output
    if format in ['parquet', 'both']:
        parquet_dir = output_dir / 'parquet'
        if not parquet_dir.exists():
            logger.error(f"Parquet directory not found: {parquet_dir}")
            return False
        
        parquet_files = list(parquet_dir.glob('*.parquet'))
        if not parquet_files:
            logger.error("No parquet files found in output")
            return False
        
        for pfile in parquet_files:
            df = pd.read_parquet(pfile)
            logger.info(f"Parquet file {pfile.name}: {len(df)} rows")
            
            # Check columns
            required_cols = ['cell_id', 'origin_file', 'scgpt_embedding', 'cell_type', 'cell_type_code']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column {col} in {pfile.name}")
                    return False
            
            # Check embedding dimensions
            first_embedding = df['scgpt_embedding'].iloc[0]
            if len(first_embedding) != 512:
                logger.error(f"Wrong embedding dimension: {len(first_embedding)} (expected 512)")
                return False
    
    # Check PT output
    if format in ['pt', 'both']:
        pt_dir = output_dir / 'pt'
        if not pt_dir.exists():
            logger.error(f"PT directory not found: {pt_dir}")
            return False
        
        pt_files = list(pt_dir.glob('*.pt'))
        if not pt_files:
            logger.error("No PT files found in output")
            return False
        
        for pt_file in pt_files:
            data = torch.load(pt_file, map_location='cpu')
            logger.info(f"PT file {pt_file.name}:")
            logger.info(f"  - Embeddings shape: {data['embeddings'].shape}")
            logger.info(f"  - Cell type codes shape: {data['cell_type_codes'].shape}")
            logger.info(f"  - Metadata keys: {list(data['metadata'].keys())}")
            
            # Check dimensions
            if data['embeddings'].shape[1] != 512:
                logger.error(f"Wrong embedding dimension in PT: {data['embeddings'].shape[1]}")
                return False
    
    logger.info("✓ Validation passed!")
    return True


def test_incremental_processing(test_dir: Path):
    """Test incremental processing feature."""
    
    logger.info("Testing incremental processing...")
    
    # Create initial data
    training_dir, scgpt_dir = create_test_data(test_dir)
    output_dir = test_dir / 'output_incremental'
    
    # First run
    extractor = ScGPTEmbeddingExtractor(
        training_dir=str(training_dir),
        scgpt_dir=str(scgpt_dir),
        output_dir=str(output_dir),
        local_output_dir=str(output_dir),
        max_workers=1,
        output_format='both',
        incremental=True,
        manifest_file='test_manifest.json'
    )
    extractor.run()
    
    # Check manifest was created
    manifest_path = output_dir / 'test_manifest.json'
    if not manifest_path.exists():
        logger.error("Manifest file not created")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest1 = json.load(f)
    
    # Second run (should skip)
    extractor2 = ScGPTEmbeddingExtractor(
        training_dir=str(training_dir),
        scgpt_dir=str(scgpt_dir),
        output_dir=str(output_dir),
        local_output_dir=str(output_dir),
        max_workers=1,
        output_format='both',
        incremental=True,
        manifest_file='test_manifest.json'
    )
    extractor2.run()
    
    # Add new batch
    new_df = pd.DataFrame({
        'cell_id': [f"cell_{i:04d}" for i in range(100, 150)],
        'origin_file': ['file1.h5ad'] * 50,
        'cell_type': ['T cell'] * 50,
        'cell_type_code': [0] * 50,
        'embedding': [np.random.randn(3072).tolist() for _ in range(50)]
    })
    new_df.to_parquet(training_dir / 'batch_0001.parquet', index=False)
    
    # Third run (should process new batch only)
    extractor3 = ScGPTEmbeddingExtractor(
        training_dir=str(training_dir),
        scgpt_dir=str(scgpt_dir),
        output_dir=str(output_dir),
        local_output_dir=str(output_dir),
        max_workers=1,
        output_format='both',
        incremental=True,
        manifest_file='test_manifest.json'
    )
    extractor3.run()
    
    # Check manifest was updated
    with open(manifest_path, 'r') as f:
        manifest2 = json.load(f)
    
    if len(manifest2) != 2:
        logger.error(f"Manifest should have 2 entries, has {len(manifest2)}")
        return False
    
    logger.info("✓ Incremental processing test passed!")
    return True


def main():
    """Run tests."""
    parser = argparse.ArgumentParser(description="Test scGPT extraction")
    parser.add_argument('--test-dir', help='Directory for test data')
    parser.add_argument('--test-incremental', action='store_true', help='Test incremental processing')
    parser.add_argument('--test-real-data', action='store_true', help='Test with real data (first batch only)')
    
    args = parser.parse_args()
    
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        test_dir = Path(tempfile.mkdtemp(prefix='scgpt_test_'))
        logger.info(f"Using temp directory: {test_dir}")
    
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.test_real_data:
            # Test with real data (limited)
            logger.info("Testing with real data (first batch only)...")
            
            extractor = ScGPTEmbeddingExtractor(
                training_dir='/data/GenePT-tools/data/cellxgene_embeddings/training_v1_shuffled',
                scgpt_dir='s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1',
                output_dir=str(test_dir / 'real_output'),
                local_output_dir=str(test_dir / 'real_output'),
                aws_profile='xcellerate',
                max_workers=1,
                output_format='both'
            )
            
            # Process only first batch for testing
            origin_to_cells, batches = extractor.inventory_training_data()
            if batches:
                first_batch = batches[0]
                logger.info(f"Processing first batch: {first_batch}")
                # Load minimal embeddings
                test_origins = list(origin_to_cells.keys())[:2]
                origin_embeddings = {}
                for origin in test_origins:
                    emb = extractor.load_scgpt_embeddings(origin)
                    if emb:
                        origin_embeddings[origin] = emb
                
                result = extractor.process_batch(first_batch, origin_embeddings)
                if result:
                    logger.info(f"✓ Real data test passed! Output: {result}")
                else:
                    logger.error("Real data test failed")
        
        elif args.test_incremental:
            # Test incremental processing
            success = test_incremental_processing(test_dir)
            if not success:
                sys.exit(1)
        
        else:
            # Basic test with synthetic data
            logger.info("Running basic test with synthetic data...")
            
            training_dir, scgpt_dir = create_test_data(test_dir)
            output_dir = test_dir / 'output'
            
            # Run extraction
            extractor = ScGPTEmbeddingExtractor(
                training_dir=str(training_dir),
                scgpt_dir=str(scgpt_dir),
                output_dir=str(output_dir),
                local_output_dir=str(output_dir),
                max_workers=1,
                output_format='both'
            )
            
            extractor.run()
            
            # Validate output
            if validate_output(output_dir):
                logger.info("✓ All tests passed!")
            else:
                logger.error("✗ Tests failed")
                sys.exit(1)
    
    finally:
        if 'temp' in str(test_dir):
            logger.info(f"Cleaning up temp directory: {test_dir}")
            import shutil
            shutil.rmtree(test_dir)


if __name__ == '__main__':
    main()