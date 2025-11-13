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
import pandas as pd

# Add numpy to safe globals for PyTorch 2.6+ weights_only=True
# We need to allow numpy types since metadata batch files contain numpy arrays
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.core.multiarray.scalar,
])


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
        verbose: bool = False,
        # Test data parameters
        is_test_mode: bool = False,
        test_genept_suffix: str = "_test_v1_scgpt",
        test_tissue_suffix: str = "_test_v1_tissue",
        test_metadata_suffix: str = "_test_v1",
        cell_type_codes: Optional[pd.Series] = None
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
            is_test_mode: If True, load from parquet files instead of pt files (test mode)
            test_genept_suffix: Directory suffix for test GenePT embeddings
            test_tissue_suffix: Directory suffix for test tissue embeddings
            test_metadata_suffix: Directory suffix for test metadata
            cell_type_codes: Series mapping cell type strings to codes (required for test mode)
        """
        self.base_dir = Path(base_dir)
        self.embedding_types = embedding_types
        self.batch_size = batch_size
        self.start_batch_file = start_batch_file
        self.end_batch_file = end_batch_file
        self.genept_dims = genept_dims
        self.code_remapping = code_remapping
        self.track_invalid_embeddings = track_invalid_embeddings
        self.seed = seed
        self.verbose = verbose

        # Per-embedding-type scaling factors (measured from raw embedding distributions)
        # Each embedding type is scaled to have std ~1.0 independently
        self.embedding_scales = {
            'genept': 0.021,  # Typical std of raw GenePT embeddings
            'scgpt': 0.044,   # Typical std of raw scGPT embeddings
        }

        # Test mode parameters
        self.is_test_mode = is_test_mode
        self.test_genept_suffix = test_genept_suffix
        self.test_tissue_suffix = test_tissue_suffix
        self.test_metadata_suffix = test_metadata_suffix
        self.cell_type_codes = cell_type_codes

        # Disable shuffling in test mode
        if is_test_mode:
            self.shuffle_files_per_epoch = False
            self.shuffle_within_files = False
        else:
            self.shuffle_files_per_epoch = shuffle_files_per_epoch
            self.shuffle_within_files = shuffle_within_files

        # Set up random state
        self.rng = random.Random(seed)

        # Build embedding directory paths based on mode
        if is_test_mode:
            # Test mode: use parquet files with test suffixes
            self.embedding_dirs = {
                'genept': self.base_dir / f'cellxgene_v2{test_genept_suffix}',
                'scgpt': self.base_dir / f'cellxgene_v2{test_genept_suffix}',
                'tissue': self.base_dir / f'cellxgene_v2{test_tissue_suffix}',
                'metadata': self.base_dir / f'cellxgene_v2{test_metadata_suffix}'
            }
        else:
            # Training mode: use pt files with shuffled suffixes
            self.embedding_dirs = {
                'genept': self.base_dir / 'cellxgene_v2_training_v1_shuffled_genept',
                'scgpt': self.base_dir / 'cellxgene_v2_training_v1_shuffled_scgpt',
                'tissue': self.base_dir / 'cellxgene_v2_training_v1_shuffled_tissue',
                'metadata': self.base_dir / 'cellxgene_v2_training_v1_shuffled_metadata'
            }

        # Verify requested types exist
        for emb_type in embedding_types:
            if emb_type not in self.embedding_dirs:
                raise ValueError(f"Unknown embedding type: {emb_type}")
            if not self.embedding_dirs[emb_type].exists():
                raise FileNotFoundError(
                    f"Embedding directory not found: {self.embedding_dirs[emb_type]}"
                )

        # Load metadata and determine files to process
        if is_test_mode:
            # Test mode: list parquet files by UUID
            # Get list of parquet files from first embedding directory
            first_emb_dir = self.embedding_dirs[embedding_types[0]]
            parquet_files = sorted(first_emb_dir.glob('*.parquet'))
            self.file_list = [f.stem for f in parquet_files]  # Store UUIDs (filenames without extension)

            # Apply start/end batch file limits if specified
            if end_batch_file is not None:
                self.file_list = self.file_list[start_batch_file:end_batch_file]
            elif start_batch_file > 0:
                self.file_list = self.file_list[start_batch_file:]

            if len(self.file_list) == 0:
                raise ValueError(f"No parquet files found in {first_emb_dir}")

            # For test mode, we don't have metadata.pt files
            self.metadatas = {}
            self.total_samples = None  # Will be determined during iteration

        else:
            # Training mode: load metadata from pt files
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
            self.file_list = self.batch_indices  # For consistency with test mode

            if len(self.batch_indices) == 0:
                raise ValueError(f"No batch files to load (start={start_batch_file}, end={end_batch_file}, (available={n_batches})")

        import json
        # Get number of classes from metadata
        if is_test_mode:
            # Test mode: infer from data or use provided cell_type_codes
            if cell_type_codes is not None:
                self.n_classes = len(cell_type_codes)
            else:
                self.n_classes = None  # Will be inferred from data
        else:
            # Training mode: get from metadata
            if 'metadata' in self.embedding_types and 'cell_types' in self.metadatas['metadata']:
                self.n_classes = len(self.metadatas['metadata']['cell_types'])
            else:
                # Fallback to first embedding type (for backward compatibility)
                first_type = self.embedding_types[0]
                if 'cell_types' in self.metadatas[first_type]:
                    self.n_classes = len(self.metadatas[first_type]['cell_types'])
                else:
                    # If no cell_types found, we'll need to infer from data
                    print("Warning: No cell_types found in metadata, will infer from data")
                    self.n_classes = None

        # Estimate total samples
        if is_test_mode:
            self.total_samples = None  # Will be determined during iteration
        else:
            # Training mode: estimate from metadata
            first_metadata = self.metadatas[self.embedding_types[0]]
            self.total_samples = first_metadata.get('total_samples') or first_metadata.get('n_total_samples', 0)
            if end_batch_file is not None:
                # Estimate samples for subset
                n_batches = min(m['n_batches'] for m in self.metadatas.values())
                self.total_samples = int(self.total_samples * len(self.batch_indices) / n_batches)

        if self.verbose:
            mode_str = "Test" if is_test_mode else "Training"
            print(f"Composable{mode_str}Dataset initialized:")
            print(f"  Embedding types: {embedding_types}")
            if is_test_mode:
                print(f"  Parquet files: {len(self.file_list)} files")
            else:
                print(f"  Batch files: {start_batch_file} to {end_batch_file - 1} ({len(self.batch_indices)} files)")
            print(f"  Total dimensions: {self.get_total_dims()}")
            if self.total_samples:
                print(f"  Estimated samples: {self.total_samples:,}")
            print(f"  Number of classes: {self.n_classes}")
            if self.genept_dims and not is_test_mode:
                print(f"  GenePT dims: {self.genept_dims} (of {self.metadatas.get('genept', {}).get('n_dims', 'N/A')})")

    def get_total_dims(self) -> int:
        """Get total embedding dimensions across all types (excluding metadata)."""
        total_dims = 0

        # Hard-coded dimension mapping for test mode
        dimension_map = {
            'genept': 3072,  # Will be sliced by genept_dims
            'scgpt': 512,
            'tissue': 126,
        }

        for emb_type in self.embedding_types:
            # Skip metadata - it's not part of X
            if emb_type == 'metadata':
                continue

            if emb_type == 'genept' and self.genept_dims is not None:
                total_dims += self.genept_dims
            elif self.is_test_mode:
                # Use hard-coded dimensions for test mode
                if emb_type in dimension_map:
                    total_dims += dimension_map[emb_type]
                else:
                    raise ValueError(f"Unknown embedding type for test mode: {emb_type}")
            else:
                # Training mode: get from metadata
                total_dims += self.metadatas[emb_type]['n_dims']
        return total_dims

    def _load_parquet_file(self, file_uuid: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and align embeddings from parquet files (test mode).

        Args:
            file_uuid: UUID of the parquet file to load (without extension)

        Returns:
            X: Combined embeddings [N_samples, total_dims]
            y: Cell type codes [N_samples]
            invalid_mask: Boolean mask [N_samples] indicating invalid rows
        """
        embeddings = {}
        observation_joinids = {}
        labels = None
        invalid_mask = None

        for emb_type in self.embedding_types:
            parquet_file = self.embedding_dirs[emb_type] / f"{file_uuid}.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

            # Load parquet file
            df = pd.read_parquet(parquet_file)

            # Extract observation_joinid for alignment
            if 'observation_joinid' in df.columns:
                observation_joinids[emb_type] = df['observation_joinid'].values
            else:
                raise ValueError(f"Missing observation_joinid column in {parquet_file}")

            # Handle metadata separately
            if emb_type == 'metadata':
                # Extract cell type labels
                if 'cell_type' in df.columns:
                    # Map cell type strings to codes
                    if self.cell_type_codes is None:
                        raise ValueError("cell_type_codes must be provided for test mode with metadata")
                    cell_types = df['cell_type'].values
                    labels = torch.tensor([self.cell_type_codes.get(ct, -1) for ct in cell_types], dtype=torch.long)
                elif 'cell_type_ontology_term_id' in df.columns:
                    # Use ontology term IDs directly (need mapping)
                    raise NotImplementedError("cell_type_ontology_term_id mapping not yet implemented")

                # Extract GenePT embeddings from metadata file (columns '0' to '3071')
                # Only select the columns we need to reduce I/O
                if self.genept_dims is not None:
                    genept_cols = [str(i) for i in range(self.genept_dims)]  # Only load needed dimensions
                else:
                    genept_cols = [str(i) for i in range(3072)]  # Load all dimensions

                if all(col in df.columns for col in genept_cols[:10]):  # Check if GenePT columns exist
                    X = df[genept_cols].values.astype(np.float32)  # Load only selected columns
                    embeddings['genept'] = torch.from_numpy(X)
                continue

            # Regular embedding type (scgpt, tissue)
            if emb_type == 'scgpt' or emb_type == 'genept':
                # scGPT embeddings: columns emb_0 to emb_511
                emb_cols = [f'emb_{i}' for i in range(512)]
                if all(col in df.columns for col in emb_cols):
                    X = df[emb_cols].values.astype(np.float32)
                    embeddings[emb_type] = torch.from_numpy(X)
                else:
                    raise ValueError(f"Missing embedding columns in {parquet_file}")

            elif emb_type == 'tissue':
                # Tissue embeddings: columns tissue_0 to tissue_125
                tissue_cols = [f'tissue_{i}' for i in range(126)]
                if all(col in df.columns for col in tissue_cols):
                    X = df[tissue_cols].values.astype(np.float32)
                    embeddings[emb_type] = torch.from_numpy(X)
                else:
                    raise ValueError(f"Missing tissue columns in {parquet_file}")

            # Detect invalid embeddings (all-zero or NaN)
            if self.track_invalid_embeddings and emb_type in embeddings:
                X_tensor = embeddings[emb_type]
                emb_invalid = (X_tensor.sum(dim=1) == 0) | torch.isnan(X_tensor).any(dim=1)

                if invalid_mask is None:
                    invalid_mask = emb_invalid
                else:
                    invalid_mask = invalid_mask | emb_invalid

        # Verify all types have matching observation_joinids
        reference_type = self.embedding_types[0]
        reference_joinids = observation_joinids[reference_type]

        for emb_type in self.embedding_types[1:]:
            if not np.array_equal(reference_joinids, observation_joinids[emb_type]):
                raise ValueError(
                    f"observation_joinid mismatch in {file_uuid}: "
                    f"{reference_type} vs {emb_type}. "
                    f"All embedding types must have identical observation_joinid order."
                )

        # Apply per-embedding-type scaling (same as training data)
        for emb_type in embeddings:
            if emb_type in self.embedding_scales:
                embeddings[emb_type] = embeddings[emb_type] / self.embedding_scales[emb_type]

        # Concatenate embeddings (exclude metadata)
        embedding_types_only = [t for t in self.embedding_types if t != 'metadata']
        X_list = [embeddings[t] for t in embedding_types_only if t in embeddings]
        X_concatenated = torch.cat(X_list, dim=1)

        return X_concatenated, labels, invalid_mask

    def _load_batch_tensors(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and align embeddings from all types, tracking invalid rows.

        Args:
            batch_idx: Index of batch file to load (or UUID string for test mode)

        Returns:
            X: Combined embeddings [N_samples, total_dims]
            y: Cell type codes [N_samples] (original, before remapping)
            invalid_mask: Boolean mask [N_samples] indicating invalid rows
        """
        # Dispatch to appropriate loader
        if self.is_test_mode:
            return self._load_parquet_file(batch_idx)  # batch_idx is actually a UUID string in test mode
        # Load embeddings from each type (excluding metadata)
        embeddings = {}
        hashes = {}
        labels = None
        invalid_mask = None

        for emb_type in self.embedding_types:
            batch_file = self.embedding_dirs[emb_type] / f"batch_{batch_idx:04d}.pt"
            if not batch_file.exists():
                raise FileNotFoundError(f"Batch file not found: {batch_file}")

            # Load batch data
            # Use weights_only=False for metadata (contains NumPy object arrays with strings)
            # Use weights_only=True for embeddings (only numeric tensors)
            weights_only = (emb_type != 'metadata')
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=weights_only)

            # Handle metadata separately - extract labels only, don't add to embeddings
            if emb_type == 'metadata':
                # Metadata has 'y' with cell type codes
                if 'y' in batch_data:
                    labels = batch_data['y']
                # Store hash for alignment verification
                hashes[emb_type] = batch_data['row_hash']
                continue  # Don't add to embeddings dict

            # Regular embedding type - add to embeddings
            X = batch_data['X']

            # Slice GenePT embeddings if specified
            # NOTE: For .pt files, we must load the full tensor from disk first,
            # then slice in memory. This wastes I/O bandwidth. For better performance,
            # pre-process .pt files to only contain needed dimensions, or use parquet files
            # which support column-level access (see parquet loading above).
            if emb_type == 'genept' and self.genept_dims is not None:
                X = X[:, :self.genept_dims]

            # Scale embedding to have std ~1.0 (normalize each type independently)
            if emb_type in self.embedding_scales:
                X = X / self.embedding_scales[emb_type]

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

        # If no labels from metadata, try to get from first embedding type
        if labels is None:
            first_type = [t for t in self.embedding_types if t != 'metadata'][0]
            batch_file = self.embedding_dirs[first_type] / f"batch_{batch_idx:04d}.pt"
            batch_data = torch.load(batch_file, map_location='cpu', weights_only=True)
            if 'y' in batch_data:
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

        # All hashes match - concatenate embeddings (exclude metadata)
        embedding_types_only = [t for t in self.embedding_types if t != 'metadata']
        X_list = [embeddings[t] for t in embedding_types_only]
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
        if self.code_remapping is not None and len(self.code_remapping) > 0:
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
            file_list = [f for i, f in enumerate(self.file_list) if i % num_workers == worker_id]

            if self.verbose and worker_id == 0:
                print(f"Worker {worker_id} processing {len(file_list)} files")
        else:
            file_list = self.file_list.copy()

        # Shuffle file order if requested (disabled in test mode)
        if self.shuffle_files_per_epoch:
            self.rng.shuffle(file_list)

        # Buffer for accumulating samples across files
        X_buffer = []
        y_buffer = []

        # Process each file (batch file or parquet file)
        for file_idx, file_id in enumerate(file_list):
            if self.verbose and file_idx % 50 == 0:
                if self.is_test_mode:
                    print(f"Processing file {file_idx + 1}/{len(file_list)}: {file_id}.parquet")
                else:
                    print(f"Processing batch file {file_idx + 1}/{len(file_list)}: batch_{file_id:04d}")

            try:
                # Load and align embeddings
                X, y = self._load_and_align_batch(file_id)

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
                if self.is_test_mode:
                    print(f"Error processing file {file_id}.parquet: {e}")
                else:
                    print(f"Error processing batch {file_id}: {e}")
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
