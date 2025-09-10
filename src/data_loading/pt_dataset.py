"""Fast PyTorch tensor-based dataset for pre-processed data."""

import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import torch
from torch.utils.data import IterableDataset


class PTFileStreamDataset(IterableDataset):
  """Streaming dataset for pre-processed .pt files.
  
  Much faster than Parquet because:
  - No parsing/filtering needed
  - Direct tensor loading
  - Pre-computed labels
  """
  
  def __init__(
      self,
      data_dir: Path,
      batch_size: int = 1024,
      n_dims: Optional[int] = None,  # None means use all dimensions
      shuffle_files_per_epoch: bool = True,
      shuffle_within_files: bool = True,
      seed: int = 42,
      verbose: bool = False
  ):
    """Initialize the dataset.
    
    Args:
      data_dir: Directory containing .pt files
      batch_size: Mini-batch size
      n_dims: Number of dimensions to use (None for all)
      shuffle_files_per_epoch: Whether to shuffle file order each epoch
      shuffle_within_files: Whether to shuffle samples within each file
      seed: Random seed
      verbose: Whether to print progress
    """
    self.data_dir = Path(data_dir)
    self.batch_size = batch_size
    self.n_dims = n_dims
    self.shuffle_files_per_epoch = shuffle_files_per_epoch
    self.shuffle_within_files = shuffle_within_files
    self.seed = seed
    self.verbose = verbose
    
    # Set up random state
    self.rng = random.Random(seed)
    
    # Get list of .pt files (excluding metadata)
    self.pt_files = sorted([
      f for f in self.data_dir.glob("*.pt") 
      if f.name != "metadata.pt"
    ])
    
    if len(self.pt_files) == 0:
      raise ValueError(f"No .pt files found in {data_dir}")
    
    # Load metadata
    metadata_path = self.data_dir / "metadata.pt"
    if metadata_path.exists():
      self.metadata = torch.load(metadata_path, weights_only=True)
      self.n_classes = len(self.metadata['cell_types'])
    else:
      # Infer from first file
      sample_data = torch.load(self.pt_files[0], weights_only=True)
      self.n_classes = sample_data['y'].max().item() + 1
    
    # Get total samples from metadata if available
    if metadata_path.exists() and 'total_samples' in self.metadata:
      self.total_samples = self.metadata['total_samples']
    else:
      # Fallback: calculate from files (for older datasets)
      self.total_samples = 0
      if self.verbose:
        print("Calculating total samples (metadata doesn't have total_samples)...")
      for pt_file in self.pt_files:
        data = torch.load(pt_file, weights_only=True, map_location='cpu')
        self.total_samples += data.get('n_samples', len(data['X']))
    
    if self.verbose:
      print(f"Dataset initialized with {len(self.pt_files)} files")
      print(f"Total samples: {self.total_samples:,}")
      print(f"Number of classes: {self.n_classes}")
      if self.n_dims:
        print(f"Using first {self.n_dims} dimensions")
  
  def _load_and_process_file(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and process a single .pt file.
    
    Args:
      file_path: Path to the .pt file
      
    Returns:
      Tuple of (X, y) tensors for the file
    """
    # Load pre-processed tensors
    data = torch.load(file_path, map_location='cpu', weights_only=True)
    X = data['X']
    y = data['y']
    
    # Slice to desired dimensions if specified
    if self.n_dims is not None and X.shape[1] > self.n_dims:
      X = X[:, :self.n_dims]
    
    # Scale embeddings to have unit variance (important for neural network training)
    # OpenAI embeddings have very small std (~0.026) which causes training issues
    X = X / 0.026  # Approximate std of OpenAI embeddings
    
    # Shuffle within file if requested
    if self.shuffle_within_files:
      indices = torch.randperm(len(X))
      X = X[indices]
      y = y[indices]
    
    return X, y
  
  def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Iterate through all files and yield batches.
    
    Accumulates samples across file boundaries to ensure consistent batch sizes.
    
    Yields:
      Tuples of (X, y) tensors for each mini-batch
    """
    # Handle multi-worker data loading
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      # Split files among workers
      worker_id = worker_info.id
      num_workers = worker_info.num_workers
      files = [f for i, f in enumerate(self.pt_files) if i % num_workers == worker_id]
      
      if self.verbose and worker_id == 0:
        print(f"Worker {worker_id} processing {len(files)} files")
    else:
      files = self.pt_files.copy()
    
    # Shuffle file order if requested
    if self.shuffle_files_per_epoch:
      self.rng.shuffle(files)
    
    # Buffer for accumulating samples across files
    X_buffer = []
    y_buffer = []
    
    # Process each file
    for file_idx, pt_file in enumerate(files):
      if self.verbose and file_idx % 50 == 0:
        print(f"Processing file {file_idx + 1}/{len(files)}: {pt_file.name}")
      
      try:
        # Load and process the file
        X, y = self._load_and_process_file(pt_file)
        
        # Add to buffer
        X_buffer.append(X)
        y_buffer.append(y)
        
        # Concatenate buffer
        X_concat = torch.cat(X_buffer, dim=0)
        y_concat = torch.cat(y_buffer, dim=0)
        
        # Yield complete batches from buffer
        while len(X_concat) >= self.batch_size:
          batch_X = X_concat[:self.batch_size]
          batch_y = y_concat[:self.batch_size]
          yield batch_X, batch_y
          
          # Keep remainder in buffer
          X_concat = X_concat[self.batch_size:]
          y_concat = y_concat[self.batch_size:]
        
        # Update buffer with remainder
        if len(X_concat) > 0:
          X_buffer = [X_concat]
          y_buffer = [y_concat]
        else:
          X_buffer = []
          y_buffer = []
          
      except Exception as e:
        print(f"Error processing {pt_file}: {e}")
        continue
    
    # Yield final partial batch if any samples remain
    if len(X_buffer) > 0:
      X_final = torch.cat(X_buffer, dim=0) if len(X_buffer) > 1 else X_buffer[0]
      y_final = torch.cat(y_buffer, dim=0) if len(y_buffer) > 1 else y_buffer[0]
      if len(X_final) > 0:
        yield X_final, y_final
  
  def __len__(self):
    """Return exact number of batches.
    
    With cross-file batching, we get exactly total_samples // batch_size batches
    (plus 1 if there's a remainder for the final partial batch).
    """
    # Now that we accumulate across files, we get the expected number of batches
    return (self.total_samples + self.batch_size - 1) // self.batch_size  # Ceiling division