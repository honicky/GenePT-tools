"""S3-based streaming dataset for CellXGene MLP training."""

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .utils import list_s3_files, get_or_download_file


class S3ParquetStreamDataset(IterableDataset):
  """Streaming dataset for pre-shuffled S3 parquet files.
  
  - Each file contains 10,000 pre-shuffled samples
  - Sequential reading with optional epoch-level file shuffling
  - Uses local files if available, downloads from S3 if not
  """
  
  def __init__(
      self,
      cell_types: List[str],
      cell_type_codes: pd.Series,
      s3_bucket: str = "pythiomicsdata",
      s3_prefix: str = "cellxgene_v2/training_v1_suffled",
      local_data_dir: Optional[Path] = None,
      n_dims: int = 500,
      batch_size: int = 1024,
      download_if_missing: bool = True,
      shuffle_files_per_epoch: bool = True,
      shuffle_within_files: bool = True,
      aws_profile: str = "xcellerate",
      start_batch_file: int = 0,
      end_batch_file: Optional[int] = None,
      seed: int = 42,
      verbose: bool = True
  ):
    """Initialize the dataset.
    
    Args:
      cell_types: List of all cell types
      cell_type_codes: Series mapping cell types to codes
      s3_bucket: S3 bucket name
      s3_prefix: S3 prefix for batch files
      local_data_dir: Local directory to check for files first
      n_dims: Number of embedding dimensions to use
      batch_size: Mini-batch size
      download_if_missing: Whether to download from S3 if not found locally
      shuffle_files_per_epoch: Whether to shuffle file order each epoch
      shuffle_within_files: Whether to shuffle samples within each file
      aws_profile: AWS profile to use for S3 access
      start_batch_file: Starting batch file index (for resuming/debugging)
      end_batch_file: Ending batch file index (for debugging with subset)
      seed: Random seed for reproducibility
      verbose: Whether to print progress messages
    """
    self.cell_types = cell_types
    self.cell_type_codes = cell_type_codes
    self.s3_bucket = s3_bucket
    self.s3_prefix = s3_prefix
    self.local_data_dir = Path(local_data_dir) if local_data_dir else Path.cwd() / "data_cache"
    self.n_dims = n_dims
    self.batch_size = batch_size
    self.download_if_missing = download_if_missing
    self.shuffle_files_per_epoch = shuffle_files_per_epoch
    self.shuffle_within_files = shuffle_within_files
    self.aws_profile = aws_profile
    self.start_batch_file = start_batch_file
    self.end_batch_file = end_batch_file
    self.seed = seed
    self.verbose = verbose
    
    # Set up random state
    self.rng = random.Random(seed)
    self.np_rng = np.random.RandomState(seed)
    
    # Get list of files
    self.s3_files = self._list_s3_files()
    
    # Create local data directory if needed
    self.local_data_dir.mkdir(parents=True, exist_ok=True)
    
    if self.verbose:
      print(f"Dataset initialized with {len(self.s3_files)} files")
      print(f"Local data directory: {self.local_data_dir}")
      print(f"Batch size: {batch_size}, n_dims: {n_dims}")
  
  def _list_s3_files(self) -> List[str]:
    """List and filter S3 files based on start/end indices."""
    all_files = list_s3_files(self.s3_bucket, self.s3_prefix, self.aws_profile)
    
    # Filter to just the batch files
    batch_files = [f for f in all_files if 'batch_' in f and f.endswith('.parquet')]
    batch_files.sort()  # Ensure consistent ordering
    
    # Apply start/end limits
    if self.end_batch_file is not None:
      batch_files = batch_files[self.start_batch_file:self.end_batch_file]
    else:
      batch_files = batch_files[self.start_batch_file:]
    
    return batch_files
  
  def _get_batch_file(self, s3_key: str) -> Optional[Path]:
    """Get a batch file, using local copy if available, downloading if not."""
    filename = Path(s3_key).name
    return get_or_download_file(
      filename=filename,
      local_dir=self.local_data_dir,
      bucket=self.s3_bucket,
      s3_prefix=self.s3_prefix,
      download_if_missing=self.download_if_missing,
      profile_name=self.aws_profile
    )
  
  def _encode_labels(self, cell_type_series: pd.Series) -> np.ndarray:
    """Encode cell type strings to integer codes.
    
    Args:
      cell_type_series: Series of cell type strings
      
    Returns:
      Array of integer codes
    """
    # Get the subset of cell types we're training on
    training_cell_types = [self.cell_types[i] for i in self.cell_type_codes.values]
    
    # Convert to categorical using only the training cell types
    categorical = cell_type_series.astype(
      pd.CategoricalDtype(categories=training_cell_types)
    )
    
    # Get codes (-1 for unknown cell types)
    codes = categorical.cat.codes
    
    # Filter out unknown cell types
    valid_mask = codes >= 0
    if not valid_mask.all():
      if self.verbose:
        print(f"Warning: Filtering out {(~valid_mask).sum()} samples with unknown cell types")
      codes = codes[valid_mask]
    
    return codes.values
  
  def _process_batch_file(self, file_path: Path) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Process a single batch file and yield mini-batches.
    
    Args:
      file_path: Path to the parquet file
      
    Yields:
      Tuples of (X, y) tensors for each mini-batch
    """
    # Read parquet file
    df = pd.read_parquet(file_path)
    
    # Get the subset of cell types we're training on
    training_cell_types = [self.cell_types[i] for i in self.cell_type_codes.values]
    
    # Filter to valid cell types
    valid_mask = df['cell_type'].isin(training_cell_types)
    df = df[valid_mask]
    
    if len(df) == 0:
      if self.verbose:
        print(f"Warning: No valid samples in {file_path}")
      return
    
    # Extract embedding columns
    embedding_cols = [str(i) for i in range(self.n_dims)]
    X = df[embedding_cols].values.astype(np.float32)
    
    # Encode labels
    y = self._encode_labels(df['cell_type'])
    
    # Shuffle within file if requested
    if self.shuffle_within_files:
      indices = self.np_rng.permutation(len(X))
      X = X[indices]
      y = y[indices]
    
    # Create mini-batches
    for i in range(0, len(X), self.batch_size):
      batch_X = torch.from_numpy(X[i:i + self.batch_size])
      batch_y = torch.from_numpy(y[i:i + self.batch_size]).long()
      yield batch_X, batch_y
  
  def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Iterate through all files and yield batches.
    
    Yields:
      Tuples of (X, y) tensors for each mini-batch
    """
    # Get file list
    files = self.s3_files.copy()
    
    # Shuffle file order if requested (but not on first epoch to match notebook)
    if self.shuffle_files_per_epoch:
      self.rng.shuffle(files)
    
    # Process each file
    for file_idx, s3_key in enumerate(files):
      if self.verbose and file_idx % 10 == 0:
        print(f"Processing file {file_idx + 1}/{len(files)}: {Path(s3_key).name}")
      
      # Get the file (download if needed)
      local_path = self._get_batch_file(s3_key)
      if local_path is None:
        print(f"Warning: Could not get file {s3_key}, skipping")
        continue
      
      # Process file and yield batches
      try:
        for batch in self._process_batch_file(local_path):
          yield batch
      except Exception as e:
        print(f"Error processing {local_path}: {e}")
        raise
  
  def __len__(self):
    """Estimate total number of batches (approximate)."""
    # Each file has ~10,000 samples
    samples_per_file = 10000
    total_samples = len(self.s3_files) * samples_per_file
    return total_samples // self.batch_size