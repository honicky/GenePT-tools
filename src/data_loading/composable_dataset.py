"""Composable embedding dataset for loading pre-shuffled embeddings.

This module provides a dataset class for loading and combining multiple
embedding types (GenePT, scGPT, tissue) from pre-shuffled batch files.
"""

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import torch
from torch.utils.data import IterableDataset
import numpy as np


class ComposableTrainingDataset(IterableDataset):
    """
    Dataset for loading composable pre-shuffled embeddings.

    Loads and combines multiple embedding types (genept, scgpt, tissue)
    from pre-shuffled batch files in JuiceFS or local storage.

    Features:
    - Combines multiple embedding types
    - Automatic alignment by row_hash
    - Dimension slicing for GenePT (e.g., use only first 1536 dims)
    - File shuffling per epoch
    - Sample shuffling within files
    - Streaming for memory efficiency
    """

    def __init__(
        self,
        base_dir: Path,
        embedding_types: List[str],
        batch_size: int = 1024,
        start_batch_file: int = 0,
        end_batch_file: Optional[int] = None,
        genept_dims: Optional[int] = 1536,
        code_remapping: Optional[Dict[int, int]] = None,
        track_invalid_embeddings: bool = True,
        shuffle_files_per_epoch: bool = True,
        shuffle_within_files: bool = True,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Initialize dataset.

        Args:
            base_dir: Base directory containing embedding type subdirectories
                     (e.g., /mmc-scratch/scratch/)
            embedding_types: List of embedding types to load
                           (e.g., ['genept', 'tissue'])
            batch_size: Mini-batch size for training
            start_batch_file: Index of first batch file to load (for debugging)
            end_batch_file: Index of last batch file to load (None = all files)
            genept_dims: Number of GenePT dimensions to use (default: 1536)
                        Set to None to use all 3072 dimensions
            code_remapping: Optional dict mapping original codes to filtered codes (or -100)
            track_invalid_embeddings: Whether to filter out invalid embedding rows
            shuffle_files_per_epoch: Whether to shuffle file order each epoch
            shuffle_within_files: Whether to shuffle samples within each file
            seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.base_dir = Path(base_dir)
        self.embedding_types = embedding_types
        self.batch_size = batch_size
        self.start_batch_file = start_batch_file
        self.end_batch_file = end_batch_file
        self.genept_dims = genept_dims
        self.code_remapping = code_remapping
        self.track_invalid_embeddings = track_invalid_embeddings
        self.shuffle_files_per_epoch = shuffle_files_per_epoch
        self.shuffle_within_files = shuffle_within_files
        self.seed = seed
        self.verbose = verbose

        # Set up random state
        self.rng = random.Random(seed)

        # Build embedding directory paths
        self.embedding_dirs = {
            'genept': self.base_dir / 'cellxgene_v2_training_v1_shuffled_genept',
            'scgpt': self.base_dir / 'cellxgene_v2_training_v1_shuffled_scgpt',
            'tissue': self.base_dir / 'cellxgene_v2_training_v1_shuffled_tissue'
        }

        # Verify requested types exist
        for emb_type in embedding_types:
            if emb_type not in self.embedding_dirs:
                raise ValueError(f"Unknown embedding type: {emb_type}")
            if not self.embedding_dirs[emb_type].exists():
                raise FileNotFoundError(
                    f"Embedding directory not found: {self.embedding_dirs[emb_type]}"
                )

        # Load metadata to get dimensions and number of batches
        self.metadatas = {}
        for emb_type in embedding_types:
            metadata_path = self.embedding_dirs[emb_type] / 'metadata.pt'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            self.metadatas[emb_type] = torch.load(metadata_path, weights_only=True)

        # Determine batch file indices
        n_batches = min(m['n_batches'] for m in self.metadatas.values())
        if end_batch_file is None:
            end_batch_file = n_batches

        self.batch_indices = list(range(start_batch_file, min(end_batch_file, n_batches)))

        if len(self.batch_indices) == 0:
            raise ValueError(f"No batch files to load (start={start_batch_file}, end={end_batch_file}, (available={n_batches})")

        import json
        # Get number of classes from metadata
        # print(json.dumps(self.metadatas[self.embedding_types[0]], indent=2))
        self.n_classes = len(self.metadatas[self.embedding_types[0]]['cell_types'])

        # Estimate total samples (will be refined as we load)
        # Note: Actual count may be lower due to alignment filtering
        self.total_samples = self.metadatas[self.embedding_types[0]]['total_samples']
        if end_batch_file is not None:
            # Estimate samples for subset
            self.total_samples = int(self.total_samples * len(self.batch_indices) / n_batches)

        if self.verbose:
            print(f"ComposableTrainingDataset initialized:")
            print(f"  Embedding types: {embedding_types}")
            print(f"  Batch files: {start_batch_file} to {end_batch_file - 1} ({len(self.batch_indices)} files)")
            print(f"  Total dimensions: {self.get_total_dims()}")
            print(f"  Estimated samples: {self.total_samples:,}")
            print(f"  Number of classes: {self.n_classes}")
            if self.genept_dims:
                print(f"  GenePT dims: {self.genept_dims} (of {self.metadatas.get('genept', {}).get('n_dims', 'N/A')})")

    def get_total_dims(self) -> int:
        """Get total embedding dimensions across all types."""
        total_dims = 0
        for emb_type in self.embedding_types:
            if emb_type == 'genept' and self.genept_dims is not None:
                total_dims += self.genept_dims
            else:
                total_dims += self.metadatas[emb_type]['n_dims']
        return total_dims

    def _load_batch_tensors(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and align embeddings from all types, tracking invalid rows.

        Args:
            batch_idx: Index of batch file to load

        Returns:
            X: Combined embeddings [N_samples, total_dims]
            y: Cell type codes [N_samples] (original, before remapping)
            invalid_mask: Boolean mask [N_samples] indicating invalid rows
        """
        # Load embeddings from each type
        embeddings = {}
        hashes = {}
        labels = None
        invalid_mask = None

        for emb_type in self.embedding_types:
            batch_file = self.embedding_dirs[emb_type] / f"batch_{batch_idx:04d}.pt"
            if not batch_file.exists():
                raise FileNotFoundError(f"Batch file not found: {batch_file}")

            batch_data = torch.load(batch_file, map_location='cpu', weights_only=True)

            # Slice GenePT embeddings if specified
            X = batch_data['X']
            if emb_type == 'genept' and self.genept_dims is not None:
                X = X[:, :self.genept_dims]

            embeddings[emb_type] = X
            hashes[emb_type] = batch_data['row_hash']

            # Detect invalid embeddings (all-zero or NaN)
            if self.track_invalid_embeddings:
                emb_invalid = (X.sum(dim=1) == 0) | torch.isnan(X).any(dim=1)

                if invalid_mask is None:
                    invalid_mask = emb_invalid
                else:
                    # Invalid if ANY embedding type is invalid
                    invalid_mask = invalid_mask | emb_invalid

            # Get labels from first embedding type
            if emb_type == self.embedding_types[0] and 'y' in batch_data:
                labels = batch_data['y']

        # Align embeddings by row_hash
        # All embedding types MUST have identical row_hash order
        reference_type = self.embedding_types[0]
        reference_hashes = hashes[reference_type]

        # Verify all types have matching hashes (fail fast if not)
        for emb_type in self.embedding_types[1:]:
            if not torch.equal(reference_hashes, hashes[emb_type]):
                raise ValueError(
                    f"Row hash mismatch in batch {batch_idx}: "
                    f"{reference_type} vs {emb_type}. "
                    f"All embedding types must have identical row_hash order. "
                    f"This indicates a data integrity problem."
                )

        # All hashes match - concatenate embeddings
        X_list = [embeddings[t] for t in self.embedding_types]
        X_concatenated = torch.cat(X_list, dim=1)

        return X_concatenated, labels, invalid_mask

    def _load_and_align_batch(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load batch and apply combined filtering.

        Filtering logic:
        1. Remap cell type codes (included → 0-N, excluded → -100)
        2. Create boolean mask for valid samples (cell type included + embeddings valid)
        3. Filter out invalid samples

        Args:
            batch_idx: Index of batch file to load

        Returns:
            Tuple of (X, y) where X is filtered embeddings, y is filtered labels
        """
        # Load batch with invalid tracking
        X, y, invalid_mask = self._load_batch_tensors(batch_idx)

        # Step 1: Remap cell type codes
        if self.code_remapping is not None:
            max_code = max(self.code_remapping.keys())
            remap_tensor = torch.full((max_code + 1,), -100, dtype=torch.long)
            for old_code, new_code in self.code_remapping.items():
                remap_tensor[old_code] = new_code

            y = remap_tensor[y]

        # Step 2: Create combined valid mask
        valid_mask = (y != -100)  # Cell type is included

        if invalid_mask is not None and self.track_invalid_embeddings:
            valid_mask = valid_mask & (~invalid_mask)  # Also exclude invalid embeddings

        # Step 3: Filter
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate through all batch files and yield mini-batches.

        Yields:
            Tuples of (X, y) tensors for each mini-batch
        """
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split batch files among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = [idx for i, idx in enumerate(self.batch_indices) if i % num_workers == worker_id]

            if self.verbose and worker_id == 0:
                print(f"Worker {worker_id} processing {len(batch_indices)} batch files")
        else:
            batch_indices = self.batch_indices.copy()

        # Shuffle batch file order if requested
        if self.shuffle_files_per_epoch:
            self.rng.shuffle(batch_indices)

        # Buffer for accumulating samples across files
        X_buffer = []
        y_buffer = []

        # Process each batch file
        for file_idx, batch_idx in enumerate(batch_indices):
            if self.verbose and file_idx % 50 == 0:
                print(f"Processing batch file {file_idx + 1}/{len(batch_indices)}: batch_{batch_idx:04d}")

            try:
                # Load and align embeddings
                X, y = self._load_and_align_batch(batch_idx)

                # Shuffle within file if requested
                if self.shuffle_within_files:
                    indices = torch.randperm(len(X))
                    X = X[indices]
                    y = y[indices]

                # Add to buffer
                X_buffer.append(X)
                y_buffer.append(y)

                # Concatenate buffer
                X_concat = torch.cat(X_buffer, dim=0)
                y_concat = torch.cat(y_buffer, dim=0)

                # Yield complete mini-batches from buffer
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
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Yield final partial batch if any samples remain
        if len(X_buffer) > 0:
            X_final = torch.cat(X_buffer, dim=0) if len(X_buffer) > 1 else X_buffer[0]
            y_final = torch.cat(y_buffer, dim=0) if len(y_buffer) > 1 else y_buffer[0]
            if len(X_final) > 0:
                yield X_final, y_final

    def __len__(self):
        """Return estimated number of mini-batches."""
        return (self.total_samples + self.batch_size - 1) // self.batch_size
