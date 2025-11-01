# Tissue Encoding with Hierarchical Ontology Specification

## Status: 📝 SPECIFICATION

## Overview
Implement hierarchical tissue encoding that captures tissue, organ, and system information using the UBERON ontology. This encoding will be used as additional features for cell type classification models, providing anatomical context beyond the cell's molecular profile.

**Key Design Decision**: Uses CellxGene's **global curated vocabulary** (81 tissues, 28 organs, 17 systems) to ensure **fixed 126-dimensional encoding** across all datasets. This guarantees:
- Consistent model architecture across train/validation/test splits
- No vocabulary mismatch between datasets
- Reproducible results across different data subsets
- Ability to train models on one dataset and apply to another

## Background
Current cell type classification models use only gene expression embeddings. However, cells from different tissues can have distinct characteristics even within the same cell type (e.g., macrophages in lung vs. liver). By encoding tissue information hierarchically, we can:

1. Provide anatomical context to improve classification accuracy
2. Handle tissue-specific variations in gene expression
3. Enable the model to learn tissue-specific cell type patterns
4. Support multi-tissue training with explicit tissue conditioning

## Objectives
1. Create efficient one-hot encoding of tissue, organ(s), and system(s)
2. Use **global vocabulary** from CellxGene's curated ontology (81 tissues, 28 organs, 17 systems)
3. Handle multi-valued mappings (one tissue → multiple organs/systems)
4. Integrate with existing PyTorch-based data loading pipeline
5. Maintain memory efficiency for large-scale training
6. Support categorical tissue inputs from parquet files
7. Provide comprehensive test coverage using real CellxGene data
8. Ensure consistent encoding across all datasets (train, validation, test)

## Data Source and Format

### Input Data
- **Source**: Parquet files in `/mnt/scratch/cellxgene_v2_test_v1/*.parquet`
- **Tissue Column**: `tissue_ontology_term_id` (pandas categorical)
- **Format**: UBERON ontology term IDs (e.g., "UBERON:0002107", "UBERON:0000178")
- **Characteristics**:
  - Each file may contain 1-5 different tissues
  - Each cell has exactly one tissue ID
  - Tissue IDs are standardized UBERON terms

### CellxGene Global Vocabulary

The encoder uses CellxGene's curated ontology lists to ensure consistent encoding across all datasets:

```python
from cellxgene_ontology_guide.curated_ontology_term_lists import (
    CuratedOntologyTermList,
    get_curated_ontology_term_list
)

# Global vocabularies (fixed for all datasets)
CURATED_TISSUES = get_curated_ontology_term_list(CuratedOntologyTermList.TISSUE_GENERAL)  # 81 tissues
CURATED_ORGANS = get_curated_ontology_term_list(CuratedOntologyTermList.ORGAN)  # 28 organs
CURATED_SYSTEMS = get_curated_ontology_term_list(CuratedOntologyTermList.SYSTEM)  # 17 systems
```

**Encoding Dimensions (Fixed)**:
- Tissue dimension: 81 (all CellxGene curated tissues)
- Organ dimension: 28 (all CellxGene curated organs)
- System dimension: 17 (all CellxGene curated systems)
- **Total dimension: 126** (constant across all datasets)

### Example Data
```python
# Sample tissue IDs from test data
tissues = [
    'UBERON:0000178',  # blood → organs: [blood], systems: [hematopoietic system]
    'UBERON:0002048',  # lung → organs: [lung], systems: [respiratory system]
    'UBERON:0002107',  # liver → organs: [liver], systems: [digestive system]
    'UBERON:0002185',  # bronchus → organs: [], systems: [respiratory system]
    'UBERON:0002082',  # cardiac ventricle → organs: [heart], systems: [cardiovascular system, circulatory system]
]
```

## Implementation Architecture

### 1. Core Encoder Class

Create `src/data_loading/tissue_encoder.py`:

```python
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle


class TissueEncoder:
    """
    Hierarchical tissue encoder using UBERON ontology with global CellxGene vocabulary.

    Converts tissue ontology terms to one-hot encoded vectors that include
    tissue, organ(s), and system(s) information. Multiple organs/systems
    for a single tissue result in multi-hot encoding.

    Uses CellxGene's curated ontology lists to ensure consistent encoding
    dimensions across all datasets (train/validation/test).

    Attributes:
        tissue_to_idx: Mapping from tissue term to index (81 tissues)
        organ_to_idx: Mapping from organ term to index (28 organs)
        system_to_idx: Mapping from system term to index (17 systems)
        tissue_dim: Dimension of tissue one-hot encoding (81)
        organ_dim: Dimension of organ one-hot encoding (28)
        system_dim: Dimension of system one-hot encoding (17)
        total_dim: Total dimension (126, constant)
    """

    def __init__(self):
        """
        Initialize encoder with global CellxGene vocabulary.

        No arguments needed - uses CellxGene's curated lists directly.
        """
        from cellxgene_ontology_guide.curated_ontology_term_lists import (
            CuratedOntologyTermList,
            get_curated_ontology_term_list
        )
        from cellxgene_ontology_guide.ontology_parser import OntologyParser

        self.ontology_parser = OntologyParser()

        # Load global vocabularies from CellxGene
        self._load_global_vocabularies()

        # Build vocabulary mappings
        self._build_vocabulary()

        # Pre-compute tissue mappings for efficiency
        self._build_mapping_cache()

        # Pre-build encoding lookup DataFrame for fast join-based encoding
        self._encoding_df = self._build_encoding_dataframe()

    def _load_global_vocabularies(self) -> None:
        """
        Load global vocabularies from CellxGene curated lists.
        """
        from cellxgene_ontology_guide.curated_ontology_term_lists import (
            CuratedOntologyTermList,
            get_curated_ontology_term_list
        )

        self.curated_tissues = get_curated_ontology_term_list(
            CuratedOntologyTermList.TISSUE_GENERAL
        )
        self.curated_organs = get_curated_ontology_term_list(
            CuratedOntologyTermList.ORGAN
        )
        self.curated_systems = get_curated_ontology_term_list(
            CuratedOntologyTermList.SYSTEM
        )

        # Convert to sets for fast lookup
        self.curated_organs_set = set(self.curated_organs)
        self.curated_systems_set = set(self.curated_systems)

    def _build_vocabulary(self) -> None:
        """
        Build vocabulary mappings for tissues, organs, and systems.

        Uses global CellxGene vocabularies (sorted for reproducibility).
        """
        # Create index mappings (sorted for reproducibility)
        self.tissue_to_idx = {t: i for i, t in enumerate(sorted(self.curated_tissues))}
        self.organ_to_idx = {o: i for i, o in enumerate(sorted(self.curated_organs))}
        self.system_to_idx = {s: i for i, s in enumerate(sorted(self.curated_systems))}

        # Store dimensions (constant)
        self.tissue_dim = len(self.tissue_to_idx)  # 81
        self.organ_dim = len(self.organ_to_idx)    # 28
        self.system_dim = len(self.system_to_idx)  # 17
        self.total_dim = self.tissue_dim + self.organ_dim + self.system_dim  # 126

    def _build_mapping_cache(self) -> None:
        """
        Pre-compute mappings from tissue to organ/system indices.

        Creates a cache for fast lookup during encoding.
        Uses UBERON ontology to find ancestors of each tissue.
        """
        self.tissue_to_organ_indices: Dict[str, List[int]] = {}
        self.tissue_to_system_indices: Dict[str, List[int]] = {}

        for tissue in self.tissue_to_idx.keys():
            # Get all ancestors of this tissue
            ancestors = self.ontology_parser.get_term_ancestors(
                tissue,
                include_self=True
            )

            # Find which curated organs/systems are ancestors
            matching_organs = [
                org for org in ancestors
                if org in self.curated_organs_set
            ]
            matching_systems = [
                sys for sys in ancestors
                if sys in self.curated_systems_set
            ]

            # Map to indices
            organ_indices = [
                self.organ_to_idx[o]
                for o in matching_organs
                if o in self.organ_to_idx
            ]
            system_indices = [
                self.system_to_idx[s]
                for s in matching_systems
                if s in self.system_to_idx
            ]

            self.tissue_to_organ_indices[tissue] = organ_indices
            self.tissue_to_system_indices[tissue] = system_indices

    def _build_encoding_dataframe(self) -> pd.DataFrame:
        """
        Pre-compute encoding vectors for all tissues as a DataFrame.

        Returns:
            DataFrame with tissue_id as index and encoding columns
            Shape: (81 tissues, 126 encoding dimensions)
        """
        rows = []

        for tissue_id in sorted(self.tissue_to_idx.keys()):
            # Create encoding vector for this tissue
            encoding = np.zeros(self.total_dim, dtype=np.float32)

            # Tissue one-hot
            tissue_idx = self.tissue_to_idx[tissue_id]
            encoding[tissue_idx] = 1.0

            # Organ multi-hot
            organ_indices = self.tissue_to_organ_indices.get(tissue_id, [])
            for organ_idx in organ_indices:
                encoding[self.tissue_dim + organ_idx] = 1.0

            # System multi-hot
            system_indices = self.tissue_to_system_indices.get(tissue_id, [])
            for system_idx in system_indices:
                encoding[self.tissue_dim + self.organ_dim + system_idx] = 1.0

            rows.append({
                'tissue_id': tissue_id,
                'encoding': encoding
            })

        return pd.DataFrame(rows).set_index('tissue_id')

    def encode(self, tissues: pd.Series) -> torch.Tensor:
        """
        Encode a series of tissue terms to hierarchical one-hot vectors.

        Uses efficient join-based approach:
        1. Create mapping DataFrame (done once during init)
        2. Join tissues with pre-computed encodings
        3. Convert to tensor

        Args:
            tissues: Pandas Series of tissue ontology term IDs (can be categorical)

        Returns:
            Tensor of shape (n_cells, total_dim) containing concatenated
            one-hot encodings for tissue, organ(s), and system(s)

        Example:
            >>> tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])
            >>> encoded = encoder.encode(tissues)
            >>> encoded.shape
            torch.Size([2, total_dim])
        """
        # Build encoding lookup if not cached
        if not hasattr(self, '_encoding_df'):
            self._encoding_df = self._build_encoding_dataframe()

        # Convert categorical to string if needed
        if pd.api.types.is_categorical_dtype(tissues):
            tissue_values = tissues.astype(str)
        else:
            tissue_values = tissues

        # Create DataFrame for joining
        tissue_df = pd.DataFrame({'tissue_id': tissue_values}).reset_index(drop=True)

        # Join with pre-computed encodings
        joined = tissue_df.join(self._encoding_df, on='tissue_id')

        # Handle unknown tissues (fill with zeros)
        encodings = joined['encoding'].values

        # Stack into array (handle None for unknown tissues)
        encoding_matrix = np.stack([
            enc if enc is not None else np.zeros(self.total_dim, dtype=np.float32)
            for enc in encodings
        ])

        # Convert to tensor
        return torch.from_numpy(encoding_matrix)

    def encode_batch(self, tissues_list: List[pd.Series]) -> torch.Tensor:
        """
        Encode multiple batches of tissues efficiently.

        Args:
            tissues_list: List of pandas Series, each containing tissue terms

        Returns:
            Tensor of shape (total_cells, total_dim)
        """
        encoded_batches = [self.encode(tissues) for tissues in tissues_list]
        return torch.cat(encoded_batches, dim=0)

    def get_tissue_indices(self, tissues: pd.Series) -> torch.LongTensor:
        """
        Get integer indices for tissues (useful for embedding layers).

        Args:
            tissues: Pandas Series of tissue ontology term IDs

        Returns:
            LongTensor of shape (n_cells,) with tissue indices
        """
        if pd.api.types.is_categorical_dtype(tissues):
            tissue_values = tissues.astype(str)
        else:
            tissue_values = tissues

        indices = torch.zeros(len(tissues), dtype=torch.long)
        for i, tissue in enumerate(tissue_values):
            if tissue in self.tissue_to_idx:
                indices[i] = self.tissue_to_idx[tissue]

        return indices

    def save(self, path: Path) -> None:
        """
        Save encoder state to disk.

        Args:
            path: Path to save pickle file
        """
        state = {
            'tissue_to_idx': self.tissue_to_idx,
            'organ_to_idx': self.organ_to_idx,
            'system_to_idx': self.system_to_idx,
            'tissue_to_organ_indices': self.tissue_to_organ_indices,
            'tissue_to_system_indices': self.tissue_to_system_indices,
            'encoding_df': self._encoding_df,  # Save pre-computed encodings
            'dimensions': {
                'tissue_dim': self.tissue_dim,
                'organ_dim': self.organ_dim,
                'system_dim': self.system_dim,
                'total_dim': self.total_dim,
            }
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> 'TissueEncoder':
        """
        Load encoder state from disk.

        Args:
            path: Path to pickle file

        Returns:
            Loaded TissueEncoder instance

        Note: Since vocabulary is global, we could just recreate the encoder,
        but loading from cache is faster and ensures consistency.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Verify dimensions match expected global vocabulary
        if state['dimensions']['tissue_dim'] != 81:
            raise ValueError(f"Expected 81 tissues, got {state['dimensions']['tissue_dim']}")
        if state['dimensions']['organ_dim'] != 28:
            raise ValueError(f"Expected 28 organs, got {state['dimensions']['organ_dim']}")
        if state['dimensions']['system_dim'] != 17:
            raise ValueError(f"Expected 17 systems, got {state['dimensions']['system_dim']}")

        # Create instance
        encoder = cls.__new__(cls)

        # Initialize ontology parser
        from cellxgene_ontology_guide.ontology_parser import OntologyParser
        encoder.ontology_parser = OntologyParser()

        # Load vocabularies
        encoder._load_global_vocabularies()

        # Restore state
        encoder.tissue_to_idx = state['tissue_to_idx']
        encoder.organ_to_idx = state['organ_to_idx']
        encoder.system_to_idx = state['system_to_idx']
        encoder.tissue_to_organ_indices = state['tissue_to_organ_indices']
        encoder.tissue_to_system_indices = state['tissue_to_system_indices']
        encoder._encoding_df = state['encoding_df']  # Restore pre-computed encodings
        encoder.tissue_dim = state['dimensions']['tissue_dim']
        encoder.organ_dim = state['dimensions']['organ_dim']
        encoder.system_dim = state['dimensions']['system_dim']
        encoder.total_dim = state['dimensions']['total_dim']

        return encoder

    def get_tissue_info(self, tissue: str) -> Dict[str, any]:
        """
        Get human-readable information about a tissue's encoding.

        Args:
            tissue: Tissue ontology term ID

        Returns:
            Dictionary with tissue, organs, and systems information
        """
        if tissue not in self.tissue_to_idx:
            # Check if tissue is valid but not in curated list
            if self.ontology_parser.is_valid_term_id(tissue):
                return {
                    'warning': f'Valid tissue but not in CellxGene curated list: {tissue}',
                    'tissue': tissue,
                    'tissue_label': self.ontology_parser.get_term_label(tissue),
                    'in_vocabulary': False
                }
            return {'error': f'Unknown tissue: {tissue}'}

        # Get organ and system IDs from cache
        organ_indices = self.tissue_to_organ_indices.get(tissue, [])
        system_indices = self.tissue_to_system_indices.get(tissue, [])

        # Reverse lookup to get IDs
        idx_to_organ = {v: k for k, v in self.organ_to_idx.items()}
        idx_to_system = {v: k for k, v in self.system_to_idx.items()}

        organ_ids = [idx_to_organ[i] for i in organ_indices]
        system_ids = [idx_to_system[i] for i in system_indices]

        return {
            'tissue': tissue,
            'tissue_label': self.ontology_parser.get_term_label(tissue),
            'tissue_idx': self.tissue_to_idx[tissue],
            'organs': [(oid, self.ontology_parser.get_term_label(oid)) for oid in organ_ids],
            'organ_indices': organ_indices,
            'systems': [(sid, self.ontology_parser.get_term_label(sid)) for sid in system_ids],
            'system_indices': system_indices,
            'encoding_dim': self.total_dim,
        }
```

### 2. No Additional Dependencies Required

The encoder uses `cellxgene-ontology-guide` directly (already in project dependencies):

```python
from cellxgene_ontology_guide.ontology_parser import OntologyParser
from cellxgene_ontology_guide.curated_ontology_term_lists import (
    CuratedOntologyTermList,
    get_curated_ontology_term_list
)

# Example: Get tissue hierarchy
parser = OntologyParser()

def get_tissue_hierarchy(tissue_id: str) -> dict:
    """Get organs and systems for a tissue."""
    # Get all ancestors
    ancestors = parser.get_term_ancestors(tissue_id, include_self=True)

    # Load curated lists
    curated_organs = set(get_curated_ontology_term_list(CuratedOntologyTermList.ORGAN))
    curated_systems = set(get_curated_ontology_term_list(CuratedOntologyTermList.SYSTEM))

    # Find matching organs and systems
    organs = [a for a in ancestors if a in curated_organs]
    systems = [a for a in ancestors if a in curated_systems]

    return {'organs': organs, 'systems': systems}
```

**Note**: The existing `src/data_loading/tissue_ontology.py` is NOT required. The encoder uses CellxGene's official ontology API directly.

### 3. Memory-Efficient Batch Construction

For integration with data loading:

```python
def load_tissues_from_parquet(parquet_path: Path) -> pd.Series:
    """
    Load tissue column from parquet file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        Pandas Series with tissue ontology term IDs (categorical)
    """
    df = pd.read_parquet(parquet_path, columns=['tissue_ontology_term_id'])
    return df['tissue_ontology_term_id']


def create_tissue_encoder() -> TissueEncoder:
    """
    Create encoder with global CellxGene vocabulary.

    Returns:
        TissueEncoder with fixed vocabulary (81 tissues, 28 organs, 17 systems)

    Note: No need to pass parquet files - vocabulary is global and fixed.
    """
    return TissueEncoder()
```

## Testing Strategy

### Unit Tests

Create `test/test_tissue_encoder.py`:

```python
import pytest
import pandas as pd
import torch
from pathlib import Path
from src.data_loading.tissue_encoder import TissueEncoder
from src.data_loading.tissue_ontology import TissueOntologyLookup


@pytest.fixture
def sample_encoder():
    """Create encoder with global CellxGene vocabulary."""
    return TissueEncoder()


def test_encoder_initialization(sample_encoder):
    """Test encoder initializes with correct global dimensions."""
    assert sample_encoder.tissue_dim == 81  # CellxGene curated tissues
    assert sample_encoder.organ_dim == 28   # CellxGene curated organs
    assert sample_encoder.system_dim == 17  # CellxGene curated systems
    assert sample_encoder.total_dim == 126  # Fixed total dimension


def test_single_tissue_encoding(sample_encoder):
    """Test encoding a single tissue."""
    tissues = pd.Series(['UBERON:0002107'])  # liver
    encoded = sample_encoder.encode(tissues)

    assert encoded.shape == (1, 126)  # Fixed global dimension
    assert encoded.dtype == torch.float32

    # Check one-hot encoding has at least tissue + organ + system
    # liver: 1 tissue + 1 organ (liver) + 1 system (digestive)
    assert encoded.sum().item() >= 3.0


def test_multi_system_encoding(sample_encoder):
    """Test tissue with multiple systems gets multi-hot encoded."""
    tissues = pd.Series(['UBERON:0002082'])  # cardiac ventricle
    encoded = sample_encoder.encode(tissues)

    # cardiac ventricle: 1 tissue + 1 organ (heart) + 2 systems (cardiovascular, circulatory)
    assert encoded.sum().item() >= 4.0


def test_batch_encoding(sample_encoder):
    """Test encoding multiple tissues in a batch."""
    tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048', 'UBERON:0000178'])
    encoded = sample_encoder.encode(tissues)

    assert encoded.shape == (3, 126)  # Fixed global dimension

    # Each row should have at least 2 ones (tissue + at least one system)
    row_sums = encoded.sum(dim=1)
    assert all(row_sums >= 2.0)


def test_categorical_input(sample_encoder):
    """Test encoding works with categorical dtype."""
    tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'], dtype='category')
    encoded = sample_encoder.encode(tissues)

    assert encoded.shape == (2, sample_encoder.total_dim)


def test_unknown_tissue_handling(sample_encoder):
    """Test encoder handles unknown tissues gracefully (zero vector)."""
    tissues = pd.Series(['UBERON:0002107', 'UBERON:9999999'])  # One valid, one unknown
    encoded = sample_encoder.encode(tissues)

    # First row should have values, second row should be all zeros
    assert encoded[0].sum() > 0
    assert encoded[1].sum() == 0


def test_get_tissue_indices(sample_encoder):
    """Test converting tissues to integer indices."""
    tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])
    indices = sample_encoder.get_tissue_indices(tissues)

    assert indices.shape == (2,)
    assert indices.dtype == torch.long
    assert all(indices >= 0)
    assert all(indices < sample_encoder.tissue_dim)


def test_save_and_load(sample_encoder, tmp_path):
    """Test saving and loading encoder state."""
    save_path = tmp_path / "encoder.pkl"
    sample_encoder.save(save_path)

    loaded_encoder = TissueEncoder.load(save_path)

    # Verify global dimensions are preserved
    assert loaded_encoder.tissue_dim == 81
    assert loaded_encoder.organ_dim == 28
    assert loaded_encoder.system_dim == 17
    assert loaded_encoder.total_dim == 126
    assert loaded_encoder.tissue_to_idx == sample_encoder.tissue_to_idx


def test_get_tissue_info(sample_encoder):
    """Test retrieving human-readable tissue information."""
    info = sample_encoder.get_tissue_info('UBERON:0002107')

    assert info['tissue'] == 'UBERON:0002107'
    assert 'organs' in info
    assert 'systems' in info
    assert info['encoding_dim'] == sample_encoder.total_dim


def test_encoding_determinism(sample_encoder):
    """Test encoding is deterministic."""
    tissues = pd.Series(['UBERON:0002107', 'UBERON:0002048'])

    encoded1 = sample_encoder.encode(tissues)
    encoded2 = sample_encoder.encode(tissues)

    assert torch.allclose(encoded1, encoded2)


def test_encoding_sparsity(sample_encoder):
    """Test encodings are sparse (mostly zeros)."""
    tissues = pd.Series(['UBERON:0002107'] * 100)
    encoded = sample_encoder.encode(tissues)

    # Calculate sparsity (proportion of zeros)
    sparsity = (encoded == 0).float().mean()

    # Should be mostly zeros (>80% zeros for hierarchical encoding)
    assert sparsity > 0.8
```

### Integration Tests

Create `test/test_tissue_encoder_integration.py`:

```python
import pytest
import pandas as pd
from pathlib import Path
from src.data_loading.tissue_encoder import (
    TissueEncoder,
    create_tissue_encoder_from_parquet_files,
    load_tissues_from_parquet
)
from src.data_loading.tissue_ontology import TissueOntologyLookup


@pytest.fixture
def real_parquet_files():
    """Get real parquet files from test data."""
    data_dir = Path('/mnt/scratch/cellxgene_v2_test_v1')
    files = sorted(data_dir.glob('*.parquet'))[:5]  # Use first 5 files

    if not files:
        pytest.skip("Test data not available")

    return files


def test_load_tissues_from_real_parquet(real_parquet_files):
    """Test loading tissues from real parquet files."""
    tissues = load_tissues_from_parquet(real_parquet_files[0])

    assert isinstance(tissues, pd.Series)
    assert len(tissues) > 0
    assert all(tissues.str.startswith('UBERON:'))


def test_create_encoder_global_vocab():
    """Test creating encoder with global vocabulary."""
    encoder = create_tissue_encoder()

    # Verify fixed global dimensions
    assert encoder.tissue_dim == 81
    assert encoder.organ_dim == 28
    assert encoder.system_dim == 17
    assert encoder.total_dim == 126


def test_encode_real_data(real_parquet_files):
    """Test encoding tissues from real data."""
    encoder = create_tissue_encoder()

    tissues = load_tissues_from_parquet(real_parquet_files[0])
    encoded = encoder.encode(tissues)

    assert encoded.shape[0] == len(tissues)
    assert encoded.shape[1] == 126  # Fixed global dimension
    assert not torch.isnan(encoded).any()
    assert not torch.isinf(encoded).any()


def test_encode_multiple_files(real_parquet_files):
    """Test encoding tissues from multiple files with same encoder."""
    encoder = create_tissue_encoder()

    all_encoded = []
    for file in real_parquet_files:
        tissues = load_tissues_from_parquet(file)
        encoded = encoder.encode(tissues)
        all_encoded.append(encoded)

    # All should have same dimension
    combined = torch.cat(all_encoded, dim=0)
    assert combined.shape[1] == 126


def test_tissue_statistics(real_parquet_files):
    """Test statistics of real tissue encodings."""
    encoder = create_tissue_encoder()

    tissues = load_tissues_from_parquet(real_parquet_files[0])
    encoded = encoder.encode(tissues)

    # Log statistics
    print(f"\nTissue encoding statistics:")
    print(f"  Total dimension: {encoder.total_dim} (fixed)")
    print(f"  Tissue dim: {encoder.tissue_dim} (81 curated tissues)")
    print(f"  Organ dim: {encoder.organ_dim} (28 curated organs)")
    print(f"  System dim: {encoder.system_dim} (17 curated systems)")
    print(f"  Average ones per encoding: {encoded.sum(dim=1).mean().item():.2f}")
    print(f"  Sparsity: {(encoded == 0).float().mean().item():.2%}")


def test_encoder_persistence(real_parquet_files, tmp_path):
    """Test saving and loading encoder with real data."""
    encoder = create_tissue_encoder()

    save_path = tmp_path / "real_encoder.pkl"
    encoder.save(save_path)

    loaded_encoder = TissueEncoder.load(save_path)

    # Verify dimensions
    assert loaded_encoder.total_dim == 126

    # Test that loaded encoder produces same results
    tissues = load_tissues_from_parquet(real_parquet_files[0])
    encoded1 = encoder.encode(tissues)
    encoded2 = loaded_encoder.encode(tissues)

    assert torch.allclose(encoded1, encoded2)


def test_consistency_across_datasets(real_parquet_files):
    """Test that encoder produces consistent dimensions across different datasets."""
    # Create encoder once
    encoder = create_tissue_encoder()

    # Encode tissues from different files
    for file in real_parquet_files:
        tissues = load_tissues_from_parquet(file)
        encoded = encoder.encode(tissues)

        # All should have same dimension regardless of which tissues are present
        assert encoded.shape[1] == 126


def test_unknown_tissue_handling(real_parquet_files):
    """Test encoder handles tissues not in curated list."""
    encoder = create_tissue_encoder()

    # Mix known and potentially unknown tissues
    tissues = load_tissues_from_parquet(real_parquet_files[0])
    encoded = encoder.encode(tissues)

    # Should still produce valid output
    assert encoded.shape[1] == 126
    # Unknown tissues should have zeros in tissue encoding but may have organ/system info
    assert not torch.isnan(encoded).any()


def test_encoding_performance(real_parquet_files):
    """Test encoding performance on large dataset."""
    import time

    encoder = create_tissue_encoder()

    # Load large file
    tissues = load_tissues_from_parquet(real_parquet_files[0])

    # Time the encoding (should be very fast due to join)
    start = time.time()
    encoded = encoder.encode(tissues)
    elapsed = time.time() - start

    cells_per_second = len(tissues) / elapsed

    print(f"\nEncoding performance:")
    print(f"  Cells: {len(tissues)}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {cells_per_second:.0f} cells/second")

    # Should encode at >100k cells/second (conservative target)
    assert cells_per_second > 100_000, f"Too slow: {cells_per_second:.0f} cells/s"
```

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from src.data_loading.tissue_encoder import TissueEncoder, create_tissue_encoder

# Create encoder with global CellxGene vocabulary
# No need to pass training files - vocabulary is fixed
encoder = create_tissue_encoder()

print(f"Fixed encoding dimensions:")
print(f"  Tissues: {encoder.tissue_dim}")
print(f"  Organs: {encoder.organ_dim}")
print(f"  Systems: {encoder.system_dim}")
print(f"  Total: {encoder.total_dim}")  # Always 126

# Save encoder for later use
encoder.save(Path('data/tissue_encoder.pkl'))

# Encode tissues from any file - dimension is always 126
train_file = Path('/mnt/scratch/cellxgene_v2_train/file1.parquet')
tissues = pd.read_parquet(train_file)['tissue_ontology_term_id']
tissue_features = encoder.encode(tissues)

print(f"Encoded shape: {tissue_features.shape}")  # (n_cells, 126)
print(f"Sparsity: {(tissue_features == 0).float().mean():.2%}")

# Same encoder works for validation and test data
val_file = Path('/mnt/scratch/cellxgene_v2_test_v1/file2.parquet')
val_tissues = pd.read_parquet(val_file)['tissue_ontology_term_id']
val_features = encoder.encode(val_tissues)
print(f"Val encoded shape: {val_features.shape}")  # (n_cells, 126) - same dimension!
```

### Integration with Training Pipeline

```python
class CellDatasetWithTissue(torch.utils.data.Dataset):
    """Dataset that includes tissue encodings."""

    def __init__(self, parquet_file: Path, tissue_encoder: TissueEncoder):
        self.data = pd.read_parquet(parquet_file)
        self.tissue_encoder = tissue_encoder

        # Pre-encode all tissues once
        self.tissue_features = tissue_encoder.encode(
            self.data['tissue_ontology_term_id']
        )

        # Extract other features
        self.embeddings = torch.tensor(
            self.data.iloc[:, 3:3075].values,  # Embedding columns
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.data['labels'].values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'tissue': self.tissue_features[idx],
            'label': self.labels[idx]
        }


# Usage
encoder = TissueEncoder.load(Path('data/tissue_encoder.pkl'))
dataset = CellDatasetWithTissue(Path('train.parquet'), encoder)

# Model that uses tissue features
class CellClassifierWithTissue(nn.Module):
    def __init__(self, embedding_dim, tissue_dim, num_classes):
        super().__init__()
        self.embedding_proj = nn.Linear(embedding_dim, 512)
        self.tissue_proj = nn.Linear(tissue_dim, 128)
        self.classifier = nn.Linear(512 + 128, num_classes)

    def forward(self, embeddings, tissue):
        emb_features = self.embedding_proj(embeddings)
        tissue_features = self.tissue_proj(tissue)
        combined = torch.cat([emb_features, tissue_features], dim=1)
        return self.classifier(combined)
```

## Performance Considerations

### Efficient Join-Based Approach

The encoder uses a highly efficient join-based approach instead of per-cell loops:

**Previous approach (naive)**:
```python
# BAD: Loop over every cell
for i, tissue in enumerate(tissues):
    encoding[i] = lookup_encoding(tissue)  # Millions of lookups!
```

**Current approach (efficient)**:
```python
# GOOD: Pre-compute all encodings, then join
# 1. Build lookup once: 81 tissues → 81 encoding vectors (done during init)
# 2. Join tissue column with lookup table (vectorized pandas operation)
# 3. Convert to tensor
```

### Performance Characteristics

**Initialization (one-time cost)**:
- Load CellxGene ontology: ~500ms
- Build encoding lookup table (81 tissues): ~10ms
- **Total: ~500ms** (done once, cached for reuse)

**Encoding (per-file)**:
- Join operation: O(n) where n = number of cells
- Numpy stack operation: O(n)
- Tensor conversion: O(n)
- **Expected throughput: ~1M cells/second on CPU**
- **100x faster than loop-based approach**

### Memory Efficiency
- Encoding lookup table: 81 tissues × 126 dims × 4 bytes = **40KB** (negligible)
- One-hot encoding is sparse (~97% zeros for typical tissue distributions)
- Temporary DataFrame during join: minimal overhead due to pandas optimization
- Consider sparse tensor representation for storage:
  ```python
  def encode_sparse(self, tissues: pd.Series) -> torch.sparse.FloatTensor:
      """Return sparse tensor representation."""
      dense = self.encode(tissues)
      return dense.to_sparse()
  ```

### Batch Processing Recommendations
- **Encode once per file**, store results alongside embeddings
- Don't re-encode tissues in training loop
- Pre-computed encodings can be saved as part of preprocessed data:
  ```python
  # Preprocess once
  encoder = TissueEncoder()
  tissues = pd.read_parquet(file)['tissue_ontology_term_id']
  tissue_features = encoder.encode(tissues)  # Fast join

  # Save with embeddings
  torch.save({
      'embeddings': embeddings,
      'tissue_features': tissue_features,  # Pre-computed
      'labels': labels
  }, 'preprocessed.pt')
  ```

## Validation and Quality Checks

### Encoding Validation
1. **Completeness**: All tissues have at least one organ and one system
2. **Consistency**: Same tissue always produces same encoding
3. **Sparsity**: Encodings should be >80% zeros
4. **Non-empty**: No all-zero encodings for known tissues

### Ontology Coverage
```python
def validate_ontology_coverage(encoder: TissueEncoder, tissues: List[str]) -> dict:
    """
    Validate that all tissues have complete ontology information.
    """
    stats = {
        'total_tissues': len(tissues),
        'tissues_with_organs': 0,
        'tissues_with_systems': 0,
        'tissues_with_multiple_organs': 0,
        'tissues_with_multiple_systems': 0,
        'missing_info': []
    }

    for tissue in tissues:
        info = encoder.get_tissue_info(tissue)

        if info.get('organs'):
            stats['tissues_with_organs'] += 1
            if len(info['organs']) > 1:
                stats['tissues_with_multiple_organs'] += 1

        if info.get('systems'):
            stats['tissues_with_systems'] += 1
            if len(info['systems']) > 1:
                stats['tissues_with_multiple_systems'] += 1

        if not info.get('organs') or not info.get('systems'):
            stats['missing_info'].append(tissue)

    return stats
```

## Dependencies

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
]
```

## Success Criteria

1. ✅ Encoder initializes from real CellxGene parquet files
2. ✅ Handles multi-valued organ/system mappings correctly (multi-hot encoding)
3. ✅ Encodes 10,000+ cells per second
4. ✅ Memory efficient (sparse representation)
5. ✅ All unit tests pass with >95% coverage
6. ✅ Integration tests pass with real data from `/mnt/scratch/cellxgene_v2_test_v1/`
7. ✅ Saves and loads without data loss
8. ✅ Produces deterministic encodings

## Future Extensions

1. **Learned Embeddings**: Replace one-hot with learned tissue embeddings
2. **Hierarchical Attention**: Attention over tissue→organ→system hierarchy
3. **Cross-Tissue Transfer**: Use tissue encodings for domain adaptation
4. **Tissue-Specific Batch Normalization**: Condition normalization on tissue
5. **Sparse Tensor Support**: Native sparse tensor operations for efficiency

## References

- UBERON Ontology: http://uberon.github.io/
- CellxGene Schema: https://github.com/chanzuckerberg/cellxgene-census
- Tissue Ontology Guide: https://github.com/obophenotype/uberon
