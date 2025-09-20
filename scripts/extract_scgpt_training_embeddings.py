#!/usr/bin/env python3
"""
Extract scGPT embeddings for cells in the shuffled training set.
Maintains the exact order and structure of the training batches.
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import warnings

import boto3
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

# Suppress pandas future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_scgpt_embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScGPTEmbeddingExtractor:
    """Extract scGPT embeddings for training cells."""
    
    def __init__(
        self,
        training_dir: str,
        scgpt_dir: str,
        output_dir: str,
        local_output_dir: Optional[str] = None,
        aws_profile: Optional[str] = None,
        max_workers: int = 4,
        cache_gb: int = 16,
        output_format: str = 'both',
        incremental: bool = False,
        manifest_file: str = 'processed_batches.json'
    ):
        """
        Initialize the extractor.
        
        Args:
            training_dir: Path to training data (local or S3)
            scgpt_dir: Path to scGPT embeddings (local or S3)
            output_dir: Output directory for results (local or S3)
            local_output_dir: Local directory for output
            aws_profile: AWS profile for S3 access
            max_workers: Number of parallel workers
            cache_gb: GB of memory for caching embeddings
            output_format: 'parquet', 'pt', or 'both'
            incremental: Whether to do incremental processing
            manifest_file: File to track processed batches
        """
        self.training_dir = training_dir
        self.scgpt_dir = scgpt_dir
        self.output_dir = output_dir
        self.local_output_dir = local_output_dir or output_dir
        self.aws_profile = aws_profile
        self.max_workers = max_workers
        self.cache_size_bytes = cache_gb * 1024**3
        self.output_format = output_format
        self.incremental = incremental
        self.manifest_file = manifest_file
        
        # Setup S3 client if needed
        self.s3_client = None
        if any(path.startswith('s3://') for path in [training_dir, scgpt_dir, output_dir]):
            self.s3_client = self._setup_s3_client()
        
        # Cache for scGPT embeddings
        self.embedding_cache = {}
        self.cache_size_current = 0
        
        # Tracking
        self.processed_batches = self._load_manifest() if incremental else {}
        self.stats = defaultdict(int)
    
    def _setup_s3_client(self):
        """Setup S3 client with profile if specified."""
        if self.aws_profile:
            session = boto3.Session(profile_name=self.aws_profile)
            return session.client('s3')
        return boto3.client('s3')
    
    def _load_manifest(self) -> Dict:
        """Load manifest of processed batches."""
        manifest_path = Path(self.local_output_dir) / self.manifest_file
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        """Save manifest of processed batches."""
        manifest_path = Path(self.local_output_dir) / self.manifest_file
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(self.processed_batches, f, indent=2)
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        if file_path.startswith('s3://'):
            # For S3, use ETag
            bucket, key = self._parse_s3_path(file_path)
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response['ETag'].strip('"')
        else:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
    
    def _parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and key."""
        path = s3_path.replace('s3://', '')
        parts = path.split('/', 1)
        return parts[0], parts[1] if len(parts) > 1 else ''
    
    def _list_files(self, directory: str, pattern: str = '*.parquet') -> List[str]:
        """List files in directory (local or S3)."""
        if directory.startswith('s3://'):
            bucket, prefix = self._parse_s3_path(directory)
            response = self.s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix
            )
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        files.append(f"s3://{bucket}/{obj['Key']}")
            return sorted(files)
        else:
            path = Path(directory)
            files = list(path.glob(pattern))
            return sorted([str(f) for f in files])
    
    def _read_parquet(self, file_path: str) -> pd.DataFrame:
        """Read parquet file from local or S3."""
        if file_path.startswith('s3://'):
            # Download to temp file
            import tempfile
            bucket, key = self._parse_s3_path(file_path)
            with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
                self.s3_client.download_file(bucket, key, tmp.name)
                return pd.read_parquet(tmp.name)
        else:
            return pd.read_parquet(file_path)
    
    def _write_parquet(self, df: pd.DataFrame, file_path: str):
        """Write parquet file to local or S3."""
        # Always write locally first
        local_path = file_path
        if file_path.startswith('s3://'):
            import tempfile
            local_path = tempfile.mktemp(suffix='.parquet')
        
        df.to_parquet(local_path, index=False)
        
        # Upload to S3 if needed
        if file_path.startswith('s3://'):
            bucket, key = self._parse_s3_path(file_path)
            self.s3_client.upload_file(local_path, bucket, key)
            os.remove(local_path)
    
    def inventory_training_data(self, use_duckdb=True) -> Dict[str, Set[str]]:
        """
        Inventory all training batches and extract cell-origin mappings.
        
        Args:
            use_duckdb: Use DuckDB for fast parquet querying (default: True)
        
        Returns:
            Dictionary mapping origin_file -> Set[cell_id]
        """
        logger.info("Starting training data inventory...")
        
        # Use local training data if available
        local_training_dir = Path('/data/GenePT-tools/data/cellxgene_embeddings/training_v1_shuffled')
        if local_training_dir.exists():
            training_files = sorted(local_training_dir.glob('batch_*.parquet'))
            logger.info(f"Found {len(training_files)} local training batch files")
        else:
            training_files = self._list_files(self.training_dir, 'batch_*.parquet')
            logger.info(f"Found {len(training_files)} training batch files in {self.training_dir}")
        
        origin_to_cells = defaultdict(set)
        all_batches_info = []
        
        # Check for incremental processing
        batches_to_process = []
        for file_path in training_files:
            file_str = str(file_path)
            batch_name = Path(file_str).name
            
            if self.incremental:
                checksum = self._compute_checksum(file_str)
                if batch_name in self.processed_batches:
                    if self.processed_batches[batch_name]['checksum'] == checksum:
                        logger.info(f"Skipping already processed batch: {batch_name}")
                        continue
            
            batches_to_process.append(file_str)
        
        logger.info(f"Processing {len(batches_to_process)} batches...")
        
        if use_duckdb and batches_to_process:
            # Use DuckDB for fast inventory
            origin_to_cells, all_batches_info = self._inventory_with_duckdb(batches_to_process)
        else:
            # Original pandas-based inventory
            for batch_file in tqdm(batches_to_process, desc="Inventorying batches"):
                df = self._read_parquet(batch_file)
                
                # Track batch info
                batch_info = {
                    'file': batch_file,
                    'n_cells': len(df),
                    'cell_types': df['cell_type'].nunique() if 'cell_type' in df.columns else 0
                }
                all_batches_info.append(batch_info)
                
                # Detect column names and build origin->cells mapping
                if 'origin_file' in df.columns and 'cell_id' in df.columns:
                    # Test data format
                    for origin, cells in df.groupby('origin_file')['cell_id']:
                        origin_to_cells[origin].update(cells.tolist())
                elif 'source_file' in df.columns and 'observation_joinid' in df.columns:
                    # Real data format
                    for origin, cells in df.groupby('source_file')['observation_joinid']:
                        origin_to_cells[origin].update(cells.tolist())
                else:
                    logger.warning(f"Batch {batch_file} missing expected column pairs (origin_file/cell_id or source_file/observation_joinid)")
        
        # Log statistics
        total_cells = sum(len(cells) for cells in origin_to_cells.values())
        logger.info(f"Inventory complete:")
        logger.info(f"  - Total batches: {len(batches_to_process)}")
        logger.info(f"  - Unique origin files: {len(origin_to_cells)}")
        logger.info(f"  - Total cells: {total_cells:,}")
        
        self.stats['total_batches'] = len(batches_to_process)
        self.stats['unique_origins'] = len(origin_to_cells)
        self.stats['total_cells'] = total_cells
        
        return origin_to_cells, batches_to_process
    
    def _inventory_with_pandas(self, batch_files: List[str]) -> Tuple[Dict[str, Set[str]], List[Dict]]:
        """Fallback pandas-based inventory when DuckDB fails."""
        origin_to_cells = defaultdict(set)
        all_batches_info = []
        
        for batch_file in batch_files:
            df = self._read_parquet(batch_file)
            batch_info = {
                'file': batch_file,
                'n_cells': len(df),
                'cell_types': df['cell_type'].nunique() if 'cell_type' in df.columns else 0
            }
            all_batches_info.append(batch_info)
            
            # Detect column names
            if 'origin_file' in df.columns and 'cell_id' in df.columns:
                for origin, cells in df.groupby('origin_file')['cell_id']:
                    origin_to_cells[origin].update(cells.tolist())
            elif 'source_file' in df.columns and 'observation_joinid' in df.columns:
                for origin, cells in df.groupby('source_file')['observation_joinid']:
                    origin_to_cells[origin].update(cells.tolist())
        
        return dict(origin_to_cells), all_batches_info
    
    def _inventory_with_duckdb(self, batch_files: List[str]) -> Tuple[Dict[str, Set[str]], List[Dict]]:
        """
        Use DuckDB to quickly inventory parquet files.
        
        Args:
            batch_files: List of parquet file paths
            
        Returns:
            Tuple of (origin_to_cells mapping, batch info list)
        """
        logger.info("Using DuckDB for fast inventory...")
        origin_to_cells = defaultdict(set)
        all_batches_info = []
        
        # Create DuckDB connection
        conn = duckdb.connect(':memory:')
        
        # First, detect column names from first file
        first_file = batch_files[0]
        detect_query = f"SELECT * FROM read_parquet({first_file!r}) LIMIT 1"
        sample_df = conn.execute(detect_query).fetchdf()
        
        # Determine column names based on what exists
        if 'origin_file' in sample_df.columns:
            origin_col = 'origin_file'
            cell_col = 'cell_id'
        elif 'source_file' in sample_df.columns:
            origin_col = 'source_file'
            cell_col = 'observation_joinid'
        else:
            # Fall back to pandas if we can't determine columns
            logger.warning("Could not determine column names, falling back to pandas")
            conn.close()
            return self._inventory_with_pandas(batch_files)
        
        logger.info(f"Detected columns: origin={origin_col}, cell={cell_col}")
        
        try:
            # Process files in chunks for memory efficiency
            chunk_size = 50
            for i in range(0, len(batch_files), chunk_size):
                chunk_files = batch_files[i:i+chunk_size]
                
                # Create glob pattern or file list for DuckDB
                if len(chunk_files) == 1:
                    file_pattern = chunk_files[0]
                else:
                    # DuckDB can read multiple files with glob or list
                    file_pattern = chunk_files
                
                # Query to get unique origin-cell pairs
                query = f"""
                    SELECT 
                        {origin_col} as origin_file,
                        {cell_col} as cell_id,
                        COUNT(*) as count
                    FROM read_parquet({file_pattern!r})
                    GROUP BY {origin_col}, {cell_col}
                """
                
                try:
                    result_df = conn.execute(query).fetchdf()
                    
                    # Build origin->cells mapping
                    for origin_file, group in result_df.groupby('origin_file'):
                        origin_to_cells[origin_file].update(group['cell_id'].tolist())
                    
                    # Get batch statistics
                    for batch_file in chunk_files:
                        stats_query = f"""
                            SELECT 
                                COUNT(*) as n_cells,
                                COUNT(DISTINCT cell_type) as n_cell_types
                            FROM read_parquet({batch_file!r})
                        """
                        stats = conn.execute(stats_query).fetchone()
                        batch_info = {
                            'file': batch_file,
                            'n_cells': stats[0],
                            'cell_types': stats[1]
                        }
                        all_batches_info.append(batch_info)
                        
                except Exception as e:
                    logger.warning(f"DuckDB query failed for chunk, falling back to pandas: {e}")
                    # Fall back to pandas for this chunk
                    for batch_file in chunk_files:
                        df = self._read_parquet(batch_file)
                        batch_info = {
                            'file': batch_file,
                            'n_cells': len(df),
                            'cell_types': df['cell_type'].nunique() if 'cell_type' in df.columns else 0
                        }
                        all_batches_info.append(batch_info)
                        
                        # Detect column names
                        if 'origin_file' in df.columns and 'cell_id' in df.columns:
                            origin_col_pd, cell_col_pd = 'origin_file', 'cell_id'
                        elif 'source_file' in df.columns and 'observation_joinid' in df.columns:
                            origin_col_pd, cell_col_pd = 'source_file', 'observation_joinid'
                        else:
                            origin_col_pd, cell_col_pd = None, None
                            
                        if origin_col_pd and cell_col_pd:
                            for origin, cells in df.groupby(origin_col_pd)[cell_col_pd]:
                                origin_to_cells[origin].update(cells.tolist())
                
                # Log progress
                processed = min(i + chunk_size, len(batch_files))
                logger.info(f"Inventoried {processed}/{len(batch_files)} files...")
        
        finally:
            conn.close()
        
        return dict(origin_to_cells), all_batches_info
    
    def load_scgpt_embeddings(self, origin_file: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load scGPT embeddings for a specific origin file.
        
        Args:
            origin_file: Name of the origin h5ad file
            
        Returns:
            Dictionary mapping cell_id -> embedding vector
        """
        # Extract stem name (remove .h5ad extension)
        origin_stem = Path(origin_file).stem
        
        # Try different naming patterns
        possible_names = [
            f"{origin_stem}_scgpt.parquet",
            f"{origin_stem}.parquet",
            origin_stem
        ]
        
        for name in possible_names:
            scgpt_path = None
            
            # Check local first
            local_scgpt_dir = Path('/data/GenePT-tools/data/cellxgene_embeddings/scgpt_embeddings_v1')
            if local_scgpt_dir.exists():
                local_file = local_scgpt_dir / name
                if local_file.exists():
                    scgpt_path = str(local_file)
            
            # Check S3 if not found locally
            if not scgpt_path:
                if self.scgpt_dir.startswith('s3://'):
                    s3_path = f"{self.scgpt_dir.rstrip('/')}/{name}"
                    try:
                        # Check if exists
                        bucket, key = self._parse_s3_path(s3_path)
                        self.s3_client.head_object(Bucket=bucket, Key=key)
                        scgpt_path = s3_path
                    except:
                        continue
                else:
                    file_path = Path(self.scgpt_dir) / name
                    if file_path.exists():
                        scgpt_path = str(file_path)
            
            if scgpt_path:
                try:
                    logger.debug(f"Loading scGPT embeddings from {scgpt_path}")
                    df = self._read_parquet(scgpt_path)
                    
                    # Create cell_id -> embedding mapping
                    embeddings = {}
                    for _, row in df.iterrows():
                        cell_id = row['cell_id']
                        if 'scgpt_embedding' in row:
                            embedding = np.array(row['scgpt_embedding'])
                        elif 'embedding' in row:
                            embedding = np.array(row['embedding'])
                        else:
                            logger.warning(f"No embedding column found in {scgpt_path}")
                            return None
                        
                        embeddings[cell_id] = embedding
                    
                    logger.info(f"Loaded {len(embeddings)} scGPT embeddings for {origin_file}")
                    return embeddings
                    
                except Exception as e:
                    logger.error(f"Error loading scGPT embeddings from {scgpt_path}: {e}")
                    continue
        
        logger.warning(f"Could not find scGPT embeddings for {origin_file}")
        return None
    
    def create_pt_format(self, df: pd.DataFrame, output_path: str):
        """Convert DataFrame to PyTorch tensor format."""
        # Stack embeddings into tensor
        embeddings = np.vstack(df['scgpt_embedding'].values)
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        # Create tensor dict
        tensor_dict = {
            'embeddings': embeddings_tensor,
            'cell_type_codes': torch.tensor(df['cell_type_code'].values, dtype=torch.long),
            'metadata': {
                'cell_ids': df['cell_id'].tolist(),
                'origin_files': df['origin_file'].tolist(),
                'cell_types': df['cell_type'].tolist(),
                'n_cells': len(df),
                'embedding_dim': embeddings.shape[1],
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Save tensor file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_dict, output_path, pickle_protocol=4)
        logger.debug(f"Saved PT format to {output_path}")
    
    def process_batch(
        self,
        batch_file: str,
        origin_embeddings: Dict[str, Dict[str, np.ndarray]]
    ) -> Optional[str]:
        """
        Process a single training batch.
        
        Args:
            batch_file: Path to training batch file
            origin_embeddings: Cached scGPT embeddings by origin file
            
        Returns:
            Output file path if successful, None otherwise
        """
        batch_name = Path(batch_file).name
        logger.info(f"Processing batch: {batch_name}")
        
        try:
            # Read training batch
            df = self._read_parquet(batch_file)
            n_cells = len(df)
            
            # Detect column names
            if 'origin_file' in df.columns and 'cell_id' in df.columns:
                origin_col = 'origin_file'
                cell_col = 'cell_id'
            elif 'source_file' in df.columns and 'observation_joinid' in df.columns:
                origin_col = 'source_file'
                cell_col = 'observation_joinid'
            else:
                logger.error(f"Batch {batch_name}: Cannot find expected columns")
                return None
            
            # Extract scGPT embeddings for each cell
            scgpt_embeddings = []
            missing_cells = []
            
            for _, row in df.iterrows():
                origin_file = row[origin_col]
                cell_id = row[cell_col]
                
                if origin_file not in origin_embeddings:
                    missing_cells.append((origin_file, cell_id))
                    scgpt_embeddings.append(None)
                    continue
                
                if cell_id not in origin_embeddings[origin_file]:
                    missing_cells.append((origin_file, cell_id))
                    scgpt_embeddings.append(None)
                    continue
                
                scgpt_embeddings.append(origin_embeddings[origin_file][cell_id])
            
            # Check for missing cells
            if missing_cells:
                logger.warning(f"Batch {batch_name}: {len(missing_cells)} cells missing scGPT embeddings")
                for origin, cell in missing_cells[:5]:  # Show first 5
                    logger.debug(f"  Missing: {origin} / {cell}")
            
            # Filter out cells without embeddings
            valid_indices = [i for i, emb in enumerate(scgpt_embeddings) if emb is not None]
            if not valid_indices:
                logger.error(f"Batch {batch_name}: No valid scGPT embeddings found!")
                return None
            
            # Create output DataFrame with standardized column names
            output_df = pd.DataFrame({
                'cell_id': df.iloc[valid_indices][cell_col].values,
                'origin_file': df.iloc[valid_indices][origin_col].values,
                'scgpt_embedding': [scgpt_embeddings[i] for i in valid_indices],
                'cell_type': df.iloc[valid_indices]['cell_type'].values if 'cell_type' in df.columns else ['unknown'] * len(valid_indices),
                'cell_type_code': df.iloc[valid_indices]['cell_type_code'].values if 'cell_type_code' in df.columns else [0] * len(valid_indices)
            })
            
            # Create output directory structure
            output_base = Path(self.local_output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
            
            # Save in requested formats
            if self.output_format in ['parquet', 'both']:
                parquet_dir = output_base / 'parquet'
                parquet_dir.mkdir(exist_ok=True)
                parquet_path = parquet_dir / batch_name
                output_df.to_parquet(parquet_path, index=False)
                logger.debug(f"Saved parquet to {parquet_path}")
            
            if self.output_format in ['pt', 'both']:
                pt_dir = output_base / 'pt'
                pt_dir.mkdir(exist_ok=True)
                pt_name = batch_name.replace('.parquet', '.pt')
                pt_path = pt_dir / pt_name
                self.create_pt_format(output_df, pt_path)
            
            # Update manifest if incremental
            if self.incremental:
                checksum = self._compute_checksum(batch_file)
                self.processed_batches[batch_name] = {
                    'checksum': checksum,
                    'processed_at': datetime.now().isoformat(),
                    'n_cells': len(output_df),
                    'n_missing': len(missing_cells)
                }
                self._save_manifest()
            
            # Update stats
            self.stats['processed_cells'] += len(output_df)
            self.stats['missing_cells'] += len(missing_cells)
            self.stats['processed_batches'] += 1
            
            logger.info(f"Completed batch {batch_name}: {len(output_df)}/{n_cells} cells")
            return str(output_base / batch_name)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self):
        """Run the full extraction pipeline."""
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("Starting scGPT embedding extraction")
        logger.info(f"Training dir: {self.training_dir}")
        logger.info(f"scGPT dir: {self.scgpt_dir}")
        logger.info(f"Output dir: {self.local_output_dir}")
        logger.info(f"Output format: {self.output_format}")
        logger.info(f"Incremental: {self.incremental}")
        logger.info("="*60)
        
        # Step 1: Inventory training data
        origin_to_cells, batches_to_process = self.inventory_training_data()
        
        if not batches_to_process:
            logger.info("No batches to process (all up to date)")
            return
        
        # Step 2: Load all required scGPT embeddings
        logger.info("Loading scGPT embeddings...")
        origin_embeddings = {}
        
        for origin_file in tqdm(origin_to_cells.keys(), desc="Loading scGPT embeddings"):
            embeddings = self.load_scgpt_embeddings(origin_file)
            if embeddings:
                origin_embeddings[origin_file] = embeddings
            else:
                logger.warning(f"Failed to load embeddings for {origin_file}")
        
        logger.info(f"Loaded embeddings for {len(origin_embeddings)}/{len(origin_to_cells)} origin files")
        
        # Step 3: Process batches
        logger.info(f"Processing {len(batches_to_process)} batches...")
        
        if self.max_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for batch_file in batches_to_process:
                    future = executor.submit(self.process_batch, batch_file, origin_embeddings)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                    result = future.result()
                    if result:
                        logger.debug(f"Completed: {result}")
        else:
            # Sequential processing
            for batch_file in tqdm(batches_to_process, desc="Processing batches"):
                self.process_batch(batch_file, origin_embeddings)
        
        # Step 4: Upload to S3 if needed
        if self.output_dir.startswith('s3://') and self.output_dir != self.local_output_dir:
            logger.info(f"Uploading results to {self.output_dir}...")
            self._upload_to_s3()
        
        # Final statistics
        elapsed = datetime.now() - start_time
        logger.info("="*60)
        logger.info("Extraction complete!")
        logger.info(f"Time elapsed: {elapsed}")
        logger.info(f"Batches processed: {self.stats['processed_batches']}")
        logger.info(f"Cells processed: {self.stats['processed_cells']:,}")
        logger.info(f"Missing cells: {self.stats['missing_cells']:,}")
        total_cells = self.stats['processed_cells'] + self.stats['missing_cells']
        success_rate = self.stats['processed_cells'] / total_cells * 100 if total_cells > 0 else 100.0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60)
    
    def _upload_to_s3(self):
        """Upload local results to S3."""
        local_path = Path(self.local_output_dir)
        
        for local_file in local_path.rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path)
                s3_key = f"{self.output_dir.rstrip('/')}/{relative_path}"
                
                if self.output_dir.startswith('s3://'):
                    bucket, key_prefix = self._parse_s3_path(self.output_dir)
                    s3_key = f"{key_prefix}/{relative_path}"
                    
                    logger.debug(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
                    self.s3_client.upload_file(str(local_file), bucket, s3_key)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract scGPT embeddings for training cells"
    )
    
    parser.add_argument(
        '--training-dir',
        default='/data/GenePT-tools/data/cellxgene_embeddings/training_v1_shuffled',
        help='Directory containing training batch files'
    )
    parser.add_argument(
        '--scgpt-dir',
        default='s3://pythiomicsdata/cellxgene_v2/scgpt_embeddings_v1',
        help='Directory containing scGPT embeddings'
    )
    parser.add_argument(
        '--output-dir',
        default='s3://pythiomicsdata/cellxgene_v2/training_v1_scgpt_shuffled',
        help='S3 output directory'
    )
    parser.add_argument(
        '--local-output-dir',
        default='/data/GenePT-tools/data/cellxgene_embeddings/training_v1_scgpt_shuffled',
        help='Local output directory'
    )
    parser.add_argument(
        '--aws-profile',
        default='xcellerate',
        help='AWS profile for S3 access'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--cache-gb',
        type=int,
        default=16,
        help='GB of memory for caching embeddings'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'pt', 'both'],
        default='both',
        help='Output format'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only process new/changed batches'
    )
    parser.add_argument(
        '--manifest',
        default='processed_batches.json',
        help='Manifest file for incremental processing'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after processing'
    )
    parser.add_argument(
        '--local-mode',
        action='store_true',
        help='Use local files only (no S3)'
    )
    
    args = parser.parse_args()
    
    # Adjust paths for local mode
    if args.local_mode:
        args.scgpt_dir = '/data/GenePT-tools/data/cellxgene_embeddings/scgpt_embeddings_v1'
        args.output_dir = args.local_output_dir
    
    # Create extractor
    extractor = ScGPTEmbeddingExtractor(
        training_dir=args.training_dir,
        scgpt_dir=args.scgpt_dir,
        output_dir=args.output_dir,
        local_output_dir=args.local_output_dir,
        aws_profile=args.aws_profile if not args.local_mode else None,
        max_workers=args.max_workers,
        cache_gb=args.cache_gb,
        output_format=args.output_format,
        incremental=args.incremental,
        manifest_file=args.manifest
    )
    
    # Run extraction
    extractor.run()
    
    # Validation if requested
    if args.validate:
        logger.info("Running validation...")
        # TODO: Implement validation
        pass


if __name__ == '__main__':
    main()