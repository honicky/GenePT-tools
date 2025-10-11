# Foundation Model Embedding Generation Specification

## Overview
Generate cell embeddings using scGPT and Transcriptformer foundation models for 5 key Tabula Sapiens v2 Curated Benchmark tissues. These embeddings will be used for downstream cell type classification and benchmarking.

## Objectives
1. Process 5 selected tissue datasets from Tabula Sapiens v2
2. Generate embeddings using scGPT (zero-shot mode)
3. Generate embeddings using Transcriptformer
4. Save embeddings in efficient format for downstream analysis
5. Ensure reproducibility and consistent preprocessing

## Dataset Information

### Input Datasets
- **Location**: `/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark/`
- **Format**: H5AD files with standardized structure
- **Count**: 5 tissue types
- **Key Fields**:
  - `X_original`: Raw count matrix
  - `obs['cell_type']`: Cell type annotations
  - `obs['donor_id']`: Donor identifiers
  - `var['feature_name']`: Gene symbols in mixed format (SYMBOL_ENSGID)
  
### Tissues to Process
```python
tissues_to_evaluate = ["Blood", "Bone_Marrow", "Lung", "Mammary", "Thymus"]

def extract_tissue_name(filepath):
    """Extract tissue name from Tabula Sapiens v2 filename."""
    stem = filepath.stem
    # Remove the known prefix
    tissue_part = stem.replace('homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_', '')
    # Remove the known suffix
    tissue_name = tissue_part.replace('_v2_curated', '')
    return tissue_name
```

### Gene Name Processing
- **Input Format**: Mixed format in `var['feature_name']`: "SYMBOL_ENSGID" (e.g., "CD3E_ENSG00000198851")
- **Required Cleaning**: Extract symbol portion before "_ENSG"
```python
def get_clean_gene_symbols(adata):
    """Extract clean gene symbols from feature_name column."""
    feature_names = adata.var['feature_name'].tolist()
    clean_genes = []
    for gene in feature_names:
        if '_ENSG' in gene:
            symbol = gene.split('_ENSG')[0]
            clean_genes.append(symbol.upper())
        elif not gene.startswith('ENSG'):
            clean_genes.append(gene.upper())
    return clean_genes
```

## Model Specifications

### scGPT
- **Model**: Zero-shot scGPT (whole human model)
- **Embedding Dimension**: 512
- **Input Requirements**:
  - Raw counts (from X_original layer)
  - Gene symbols matching model vocabulary
  - Batch size: 32 (adjust based on GPU memory)
- **Processing Mode**: Zero-shot (no fine-tuning)
- **Normalization**: Model-specific (log1p, scaling to 10,000)

### Transcriptformer
- **Model**: Latest checkpoint from authors
- **Embedding Dimension**: 128 or 256 (check model config)
- **Input Requirements**:
  - Preprocessed expression values
  - Gene matching to model vocabulary
  - Tokenization per model specifications
- **Processing Mode**: Zero-shot inference
- **Normalization**: Follow model documentation

## Implementation Architecture

### 1. Data Pipeline

```python
class FoundationModelEmbeddingGenerator:
    def __init__(self, model_type: str, device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.gene_vocab = None
        
    def load_model(self):
        """Load pre-trained foundation model."""
        if self.model_type == "scgpt":
            self.load_scgpt()
        elif self.model_type == "transcriptformer":
            self.load_transcriptformer()
    
    def preprocess_adata(self, adata: AnnData) -> torch.Tensor:
        """
        Preprocess AnnData for model input.
        - Match genes to model vocabulary
        - Apply model-specific normalization
        - Convert to appropriate tensor format
        """
        pass
    
    def generate_embeddings(self, adata: AnnData) -> np.ndarray:
        """
        Generate embeddings for all cells in batches.
        Returns: [n_cells, embedding_dim] array
        """
        pass
```

### 2. Batch Processing Strategy

```python
class BatchProcessor:
    def __init__(self, batch_size: int = 32, max_cells: int = 50000):
        self.batch_size = batch_size
        self.max_cells = max_cells  # Subsample if needed
    
    def process_tissue(self, tissue_path: Path, model: FoundationModelEmbeddingGenerator):
        """
        Process single tissue file:
        1. Load AnnData
        2. Subsample if > max_cells (stratified by cell type)
        3. Generate embeddings in batches
        4. Save results
        """
        pass
```

### 3. Gene Matching and Vocabulary Analysis

#### Pre-Processing Vocabulary Analysis

Before implementing the full pipeline, conduct a vocabulary mismatch analysis to verify gene mapping quality:

1. **Load Model Vocabulary**: Extract the gene vocabulary from the pre-trained model (e.g., `vocab.json` for scGPT). This typically contains gene symbols as keys mapped to token indices.

2. **Analyze Dataset Gene Formats**: For each Tabula Sapiens v2 dataset:
   - Identify available gene identifier columns (`feature_name`, `gene_name`, `ensembl_id`)
   - Detect mixed format genes (e.g., "GENE_ENSG00000123456" format where gene symbol and Ensembl ID are concatenated)
   - Calculate the percentage of genes in this mixed format

3. **Test Mapping Strategy**:
   - Extract gene symbols from mixed format by splitting on "_ENSG" 
   - Convert all gene names to uppercase (scGPT uses uppercase HGNC symbols)
   - Calculate coverage: what percentage of dataset genes exist in the model vocabulary
   - Identify categories of missing genes (mitochondrial MT-*, ribosomal RPL*/RPS*, non-coding RNAs)

4. **Acceptance Criteria**:
   - Mean coverage across all tissues should be >85%
   - Minimum coverage for any single tissue should be >75%
   - Most missing genes should be non-critical (MT genes, pseudogenes, lncRNAs)
   - Core marker genes (GAPDH, ACTB, CD3E, CD4, CD8A, etc.) should be retained

5. **Generate Mismatch Report**:
   - Coverage statistics per tissue
   - Most frequently missing genes across all tissues
   - Categories of missing genes
   - Recommendation on whether to proceed with existing mapping

```python
class GeneMapper:
    def __init__(self, model_dir: Path, model_type: str):
        self.model_type = model_type
        self.vocab = self.load_vocabulary(model_dir)
        self.vocab_upper = set(g.upper() for g in self.vocab.keys())
    
    def analyze_vocabulary_coverage(self, adata) -> dict:
        """
        Analyze how well dataset genes match model vocabulary.
        Returns statistics for decision-making.
        """
        # Extract and clean gene names
        gene_names = self.extract_gene_names(adata)
        gene_symbols = self.clean_mixed_format(gene_names)
        gene_symbols_upper = gene_symbols.str.upper()
        
        # Calculate coverage
        in_vocab = gene_symbols_upper.isin(self.vocab_upper)
        coverage = in_vocab.mean() * 100
        
        # Categorize missing genes
        missing = gene_symbols[~in_vocab]
        
        return {
            'coverage_pct': coverage,
            'n_matched': in_vocab.sum(),
            'n_total': len(gene_symbols),
            'missing_genes': missing.head(20).tolist()
        }
```

#### Implementation Notes for Gene Mapping

After vocabulary analysis confirms acceptable coverage:

1. **Gene Filtering Strategy**:
   - Retain only genes present in model vocabulary for embedding generation
   - Document the number and percentage of genes filtered per tissue
   - Store filtered gene lists for reproducibility

2. **Quality Assurance**:
   
   **Key Marker Gene Verification**:
   - Define tissue-specific marker gene sets based on known biology:
     - Blood/Immune: CD3E, CD4, CD8A, CD14, CD19, MS4A1, NCAM1, FCGR3A
     - Epithelial: EPCAM, KRT8, KRT18, KRT19, CDH1
     - Endothelial: PECAM1, CDH5, VWF, FLT1
     - Stromal: COL1A1, COL3A1, VIM, ACTA2
     - Neural: RBFOX3, MAP2, GFAP, OLIG2
   - After filtering, verify >90% of tissue-relevant markers are retained
   - Generate warning if critical lineage markers are missing
   
   **Cell Type Coverage Analysis**:
   - For each cell type in the dataset, calculate the percentage of its highly variable genes that remain after filtering
   - Flag any cell type where <70% of its top 100 differentially expressed genes are retained
   - Ensure rare cell types (< 1% of cells) maintain adequate gene representation for distinguishability
   - Compare cell type separation (via UMAP or clustering) before and after gene filtering to ensure biological structure is preserved
   
   **Minimum Gene Thresholds**:
   - Absolute minimum: 1000 genes (below this, embeddings lose biological meaning)
   - Recommended minimum: 2000 genes for robust cell type identification
   - Optimal range: 3000-5000 genes balancing coverage with computational efficiency
   - If a tissue falls below threshold after filtering, consider:
     - Using a different foundation model with broader vocabulary
     - Implementing gene synonym mapping to increase matches
     - Documenting the limitation and expected impact on downstream analysis

## Processing Workflow

### Step 1: Environment Setup

Due to conflicting Python version requirements (scGPT uses Python 3.10 with torch 2.1.2, Transcriptformer requires Python ≥3.11 with torch 2.5.1), we use separate virtual environments:

**Note**: The dependency groups approach was replaced with separate requirements files due to conflicts.

#### Configure pyproject.toml

```toml
[project]
name = "genept-tools"
requires-python = ">=3.10"
dependencies = []  # Keep base deps minimal

[dependency-groups]
# scGPT stack (Python 3.10)
scgpt = [
  "torch==2.1.2",
  "torchvision==0.16.2",
  "scgpt @ git+https://github.com/bowang-lab/scGPT.git",
  "flash-attn>=2.0.0",
  "scanpy>=1.9.0",
  "anndata>=0.9.0",
  "pandas>=2.0.0",
  "numpy<2.0.0",  # scGPT compatibility
  "h5py>=3.0.0",
  "pyarrow>=14.0.0",
  "tqdm>=4.0.0",
]

# Transcriptformer stack (Python 3.11+)
transcriptformer = [
  "torch==2.5.1", 
  "torchvision==0.20.1",
  "transcriptformer @ git+https://github.com/czi-ai/transcriptformer.git",
  "scanpy>=1.9.0",
  "anndata>=0.9.0",
  "pandas>=2.0.0",
  "numpy>=1.24.0",
  "h5py>=3.0.0",
  "pyarrow>=14.0.0",
  "tqdm>=4.0.0",
]

# Platform-specific PyTorch indexes
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu121", marker = "sys_platform != 'darwin'" }
]
torchvision = [
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
  { index = "pytorch-cu121", marker = "sys_platform != 'darwin'" }
]
```

#### Create Separate Environments

**Actual Implementation Approach:**

```bash
# Create Python 3.10 environment for scGPT
uv venv -p 3.10 .venv-scgpt

# Create Python 3.11 environment for Transcriptformer  
uv venv -p 3.11 .venv-transcriptformer

# Install dependencies using requirements files (preferred over dependency groups)
source .venv-scgpt/bin/activate && uv pip install -r requirements-scgpt.txt
source .venv-transcriptformer/bin/activate && uv pip install -r requirements-transcriptformer.txt

# Install scGPT package directly
source .venv-scgpt/bin/activate && uv pip install scgpt
```

**Helper Scripts Created:**
- `scripts/run_scgpt.sh` - Activates scGPT environment and runs commands
- `scripts/run_transcriptformer.sh` - Activates Transcriptformer environment and runs commands

### Step 2: Model Download
```python
# Download pre-trained models
def download_models():
    # scGPT
    scgpt_checkpoint = "models/pretrained/scgpt_human_zero_shot_checkpoint_name.pt"
    
    # Transcriptformer
    tf_checkpoint = "models/pretrained/transcriptformer_pretrained_checkpoint_name.pt"
```

### Step 3: Pre-Analysis (Notebook)

Manually run vocabulary analysis in a Jupyter notebook before generating embeddings. Open `notebooks/scgpt_transcriptformer_vocabulary_analysis.ipynb` in your preferred environment.

The notebook should:
1. Load both model vocabularies (scGPT and Transcriptformer)
2. Analyze gene coverage for all 26 tissues
3. Generate coverage report with statistics
4. Identify problematic tissues or cell types
5. Save analysis results to `data/vocabulary_analysis_results.csv`

Only proceed to Step 4 if coverage meets acceptance criteria (>85% mean, >75% minimum).

### Step 4: Embedding Generation (Scripts)

Run each model separately to avoid memory issues and maintain clarity:

#### Generate scGPT Embeddings

**Actual Implementation Notes:**

1. **Use Official scGPT API**: The `scg.tasks.embed_data()` function handles most complexity
2. **Required Model Files**:
   - `best_model.pt` (checkpoint, ~205MB)
   - `args.json` (model configuration)
   - `vocab.json` (gene vocabulary, 60,697 genes)

3. **macOS Compatibility Fixes Required**:
   - Replace `os.sched_getaffinity()` with `os.cpu_count()`
   - Set `num_workers=0` in DataLoader to avoid multiprocessing issues

4. **Gene Name Cleaning**:
   - 252 genes in SYMBOL_ENSG format need splitting on "_ENSG"
   - 5,840 ENSG-only IDs are kept as-is
   - 74% gene coverage achieved (19,359/26,167 genes)

Create `scripts/generate_scgpt_embeddings_v2.py` (preferred implementation):
```python
#!/usr/bin/env python
"""
Generate scGPT embeddings using official API.
Run with: ./scripts/run_scgpt.sh python scripts/generate_scgpt_embeddings_v2.py
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

def main(args):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/scgpt_embedding_{args.tissue}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize scGPT model once
    model = load_scgpt_model(args.checkpoint_path)
    
    if args.tissue == "all":
        # Process all tissues sequentially
        tissue_files = sorted(Path(args.data_dir).glob("*.h5ad"))
        for tissue_file in tqdm(tissue_files, desc="Processing tissues"):
            process_single_tissue(tissue_file, model, args.output_dir)
    else:
        # Process single tissue
        tissue_file = Path(args.data_dir) / f"{args.tissue}.h5ad"
        process_single_tissue(tissue_file, model, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", default="all", help="Tissue name or 'all'")
    parser.add_argument("--data-dir", default="/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark")
    parser.add_argument("--output-dir", default="data/cz_benchmark/embeddings/scgpt")
    parser.add_argument("--checkpoint-path", required=True, help="Path to scGPT checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    main(args)
```

Run with:
```bash
# Process single tissue
./scripts/run_scgpt.sh python scripts/generate_scgpt_embeddings.py \
    --tissue Blood \
    --checkpoint-path models/scgpt_human.pt

# Process all tissues
./scripts/run_scgpt.sh python scripts/generate_scgpt_embeddings.py \
    --tissue all \
    --checkpoint-path models/scgpt_human.pt \
    --resume
```

#### Generate Transcriptformer Embeddings

Create `scripts/generate_transcriptformer_embeddings.py` with similar structure:
```bash
# Process all tissues with Transcriptformer
./scripts/run_transcriptformer.sh python scripts/generate_transcriptformer_embeddings.py \
    --tissue all \
    --checkpoint-path models/transcriptformer.pt \
    --resume
```

### Step 5: Basic Validation

Include basic validation checks in the embedding generation scripts:

```python
def validate_embeddings(embeddings: np.ndarray, expected_dim: int, tissue_name: str):
    """
    Basic sanity checks for generated embeddings.
    """
    # Check for invalid values
    assert not np.isnan(embeddings).any(), f"{tissue_name}: Contains NaN values"
    assert not np.isinf(embeddings).any(), f"{tissue_name}: Contains Inf values"
    
    # Check dimensions
    assert embeddings.ndim == 2, f"{tissue_name}: Wrong number of dimensions"
    assert embeddings.shape[1] == expected_dim, f"{tissue_name}: Wrong embedding dimension"
    
    # Check for non-zero variance (not all same value)
    assert embeddings.std() > 0, f"{tissue_name}: Zero variance in embeddings"
    
    # Check value range is reasonable (not exploded gradients)
    assert embeddings.min() > -100, f"{tissue_name}: Contains extremely negative values"
    assert embeddings.max() < 100, f"{tissue_name}: Contains extremely positive values"
    
    logging.info(f"{tissue_name} validation passed: shape={embeddings.shape}, "
                f"range=[{embeddings.min():.2f}, {embeddings.max():.2f}]")
```

These checks should be run immediately after generating embeddings for each tissue, before saving to disk. Any validation failures should halt processing and be logged for investigation.

## Output Specifications

### File Structure
```
data/cz_benchmark/embeddings/
├── scgpt/
│   ├── scgpt_Blood_embeddings.parquet
│   ├── scgpt_Bone_Marrow_embeddings.parquet
│   └── ...
├── transcriptformer/
│   ├── transcriptformer_Blood_embeddings.parquet
│   ├── transcriptformer_Bone_Marrow_embeddings.parquet
│   └── ...
└── processing_logs/
    ├── scgpt_processing.log
    └── transcriptformer_processing.log
```

### Embedding Format
```python
# Parquet format with embeddings and metadata
import pyarrow as pa
import pyarrow.parquet as pq

# Method 1: Direct DataFrame construction without copy
# Create metadata DataFrame first
metadata_df = pd.DataFrame({
    "cell_id": cell_ids,
    "cell_type": cell_types,
    "donor_id": donor_ids,
    "tissue": tissue_name,
    "n_genes_detected": n_genes,
    "total_counts": total_counts,
    "n_counts_original": n_counts_original  # Raw counts before normalization
})

# Create embedding DataFrame directly from array (avoids per-column copy)
embedding_df = pd.DataFrame(
    embeddings, 
    columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
)

# Concatenate along columns (more memory efficient than dict construction)
full_df = pd.concat([metadata_df, embedding_df], axis=1)

# Alternative Method 2: Using PyArrow RecordBatch (most efficient)
# This approach minimizes copies by working with contiguous memory blocks
# import pyarrow as pa
# import pyarrow.compute as pc
#
# # Convert metadata to PyArrow arrays (potentially zero-copy for some types)
# metadata_table = pa.Table.from_pandas(metadata_df)
# 
# # For embeddings, create a single tensor/fixed-size-list column (most efficient)
# # Option A: Store as tensor (requires all cells have same embedding dim)
# embeddings_tensor = pa.Tensor.from_numpy(embeddings)
# 
# # Option B: Store as FixedSizeListArray (more flexible, still efficient)
# flat_array = pa.array(embeddings.ravel())  # Flatten to 1D
# embeddings_array = pa.FixedSizeListArray.from_arrays(flat_array, embeddings.shape[1])
#
# # Create table with embedding as single column (avoids many columns)
# table = metadata_table.append_column("embeddings", embeddings_array)

# Create Parquet metadata for scalar attributes
metadata = {
    "model": model_name,
    "model_checkpoint": checkpoint_path,
    "embedding_dim": str(embedding_dim),
    "n_cells": str(n_cells),
    "n_genes_in_model": str(n_genes_in_model),
    "gene_coverage": str(gene_coverage),  # Percentage of genes mapped
    "processing_date": datetime.now().isoformat(),
    "preprocessing_method": preprocessing_method,
    "model_version": model_version
}

# Write Parquet with metadata
table = pa.Table.from_pandas(full_df)
# Add metadata to schema
existing_metadata = table.schema.metadata or {}
combined_metadata = {**existing_metadata, **{k.encode(): v.encode() for k, v in metadata.items()}}
table = table.replace_schema_metadata(combined_metadata)

# Save with compression
pq.write_table(
    table, 
    f"{model_name}_{tissue_name}_embeddings.parquet",
    compression='snappy'  # Fast compression for large embedding matrices
)
```

### Reading Embeddings
```python
# Read Parquet file with embeddings and metadata
def read_embeddings(filepath: Path) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Read embeddings from Parquet file.
    Returns: (embeddings, metadata, attributes)
    """
    # Read table
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    # Extract embeddings
    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
    embeddings = df[embedding_cols].values
    
    # Extract metadata
    metadata_cols = [col for col in df.columns if not col.startswith("embedding_")]
    metadata = df[metadata_cols]
    
    # Extract scalar attributes from Parquet metadata
    attributes = {}
    if table.schema.metadata:
        attributes = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    
    return embeddings, metadata, attributes
```

## Memory and Performance Optimization

### GPU Memory Management
```python
def process_with_memory_management(adata, model, batch_size=32):
    """
    Process in batches with automatic batch size adjustment.
    """
    embeddings = []
    
    try:
        # Try default batch size
        embeddings = process_batches(adata, model, batch_size)
    except torch.cuda.OutOfMemoryError:
        # Reduce batch size and retry
        torch.cuda.empty_cache()
        batch_size = batch_size // 2
        embeddings = process_batches(adata, model, batch_size)
    
    return embeddings
```

### Checkpointing
```python
def process_with_checkpointing(tissue_files, model):
    """
    Save progress after each tissue.
    """
    checkpoint_file = "processing_checkpoint.json"
    
    # Load checkpoint if exists
    processed = load_checkpoint(checkpoint_file)
    
    for tissue_file in tissue_files:
        if tissue_file.name in processed:
            continue
            
        # Process tissue
        embeddings = generate_embeddings(tissue_file, model)
        
        # Save and update checkpoint
        save_embeddings(embeddings)
        update_checkpoint(checkpoint_file, tissue_file.name)
```

## Validation and Testing

### Unit Tests
1. Test gene mapping between datasets and models
2. Test preprocessing pipelines for each model
3. Test batch processing with various sizes
4. Test embedding dimension consistency

### Integration Tests
1. Process smallest real dataset end-to-end 
2. Verify output file formats
3. Test checkpoint/resume functionality
4. Process largest real dataset end-to-end

## Error Handling

### Common Issues and Solutions

1. **Gene Vocabulary Mismatch**
   - Log missing genes
   - Use zero-padding for missing genes
   - Report coverage statistics

2. **Memory Overflow**
   - Automatic batch size reduction
   - Cell subsampling for large datasets
   - CPU offloading if needed

3. **Model Loading Failures**
   - Verify checkpoint integrity
   - Check CUDA/PyTorch compatibility
   - Fallback to CPU if GPU unavailable

## Dependencies and Requirements

### Environment Management Strategy

Due to conflicting dependencies between foundation models:
- **scGPT**: Requires Python 3.10, PyTorch 2.1.2
- **Transcriptformer**: Requires Python ≥3.11, PyTorch 2.5.1

We use `uv` dependency groups to maintain isolated, reproducible environments for each model. This approach ensures:
1. No dependency conflicts between models
2. Reproducible builds across different machines
3. Platform-specific PyTorch builds (CPU for macOS, CUDA for Linux)

### Reproducible Dependency Locks

Generate pinned requirements for exact reproducibility:
```bash
# Generate locked requirements for each environment
UV_PROJECT_ENVIRONMENT=.venv-scgpt uv pip compile --group scgpt -o requirements.scgpt.lock
UV_PROJECT_ENVIRONMENT=.venv-transcriptformer uv pip compile --group transcriptformer -o requirements.transcriptformer.lock

# Recreate exact environment from locks
UV_PROJECT_ENVIRONMENT=.venv-scgpt uv pip sync requirements.scgpt.lock
UV_PROJECT_ENVIRONMENT=.venv-transcriptformer uv pip sync requirements.transcriptformer.lock
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥16GB VRAM (A100 preferred for batch processing)
  - scGPT: Requires CUDA 12.1 compatible GPU
  - Transcriptformer: Can use CUDA 12.1 or newer
- **RAM**: 64GB minimum (128GB recommended for processing multiple tissues in parallel)
- **Storage**: 
  - Models: ~10GB for checkpoints
  - Input data: ~50GB for all Tabula Sapiens v2 tissues
  - Output embeddings: ~200GB (Parquet compressed)
  - Working space: ~100GB for intermediate files
  - Total: 500GB recommended

### Software Requirements
- **Python**: 3.10 (for scGPT) and 3.11+ (for Transcriptformer)
- **uv**: Latest version for dependency management
- **CUDA**: 12.1 or compatible version (Linux systems)
- **Git**: For cloning model repositories

## Execution Plan

### Phase 1: Setup and Pre-Analysis
1. Configure dual virtual environments using dependency groups
2. Download model checkpoints (scGPT and Transcriptformer)
3. Run vocabulary analysis notebook to assess gene coverage
4. Review coverage statistics and decide whether to proceed
5. Test embedding generation on smallest tissue file

### Phase 2: scGPT Embedding Generation
1. Activate scGPT environment (.venv-scgpt)
2. Run embedding generation script for all 26 tissues
3. Monitor logs for any failures or warnings
4. Basic validation checks (NaN, dimensions, value ranges)
5. Save embeddings as Parquet files with metadata

### Phase 3: Transcriptformer Embedding Generation
1. Activate Transcriptformer environment (.venv-transcriptformer)
2. Run embedding generation script for all 26 tissues
3. Monitor logs for any failures or warnings
4. Basic validation checks (NaN, dimensions, value ranges)
5. Save embeddings as Parquet files with metadata

### Phase 4: Completion
1. Verify all tissues processed successfully
2. Check total file sizes and integrity
3. Generate summary report of what was produced
4. Document any tissues that failed or had issues
5. Package embeddings for downstream analysis

## Success Criteria

1. ✅ All 26 tissues processed successfully
2. ✅ Embeddings generated for >95% of cells
3. ✅ No NaN/Inf values in embeddings
4. ✅ Reproducible results (same input → same output)
5. ✅ Efficient storage format (<100GB total)
6. ✅ Processing completed within 1 week
7. ✅ Clear documentation of parameters used

## Future Extensions

1. Add more foundation models (Geneformer, CellPLM)
2. Implement fine-tuning capabilities
3. Add streaming processing for very large datasets
4. Create unified embedding format for model comparison
5. Implement distributed processing across multiple GPUs