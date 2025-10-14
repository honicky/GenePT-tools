# CZ Benchmarks Evaluation Specification

## Overview

This specification outlines the implementation of a comprehensive evaluation notebook using CZ Benchmarks framework to evaluate GenePT embeddings alone and in combination with foundation model embeddings (scGPT and Transcriptformer) for cell type classification tasks.

## Objectives

1. **Benchmark GenePT at 1024 dimensions** using CZ's label prediction task
2. **Evaluate embedding combinations**:
   - GenePT (1024d) alone 
   - GenePT (1024d) + scGPT (512d) = 1536d total
   - GenePT (1024d) + Transcriptformer (2048d) = 3072d total
3. **Use precomputed embeddings** for 5 tissue types: Blood, Bone_Marrow, Lung, Mammary, Thymus
4. **Leverage CZ's standardized evaluation** for reproducible benchmarking

## Data Architecture

### Input Data Sources
- **Tabula Sapiens v2 Curated Datasets**: Local h5ad files in `/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark/`
  - Loaded via custom `load_tabula_sapiens_dataset()` function
  - Pattern: `*_{tissue_name}_v2_curated.h5ad`
- **GenePT Gene Embeddings**: `data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet`
  - 33,703 genes with 3072d embeddings
  - Gene names in index (not column)
- **Precomputed Cell Embeddings**:
  - scGPT: `data/cz_benchmark/embeddings/scgpt/{tissue}_embeddings.parquet` (512d)
  - Transcriptformer: `data/cz_benchmark/embeddings/transcriptformer/{tissue}_embeddings.parquet` (2048d)
  - GenePT 1024d: Generated in Phase 1, saved to `data/cz_benchmark/embeddings/genept_1024d/{tissue}_embeddings.parquet`

### GenePT 1024d Generation Strategy
Since OpenAI's text-embedding-3-large uses Matryoshka embeddings, we can efficiently create 1024d GenePT embeddings by:
1. **Use existing 3072d GenePT embeddings** (already computed and stored)
2. **Truncate to first 1024 dimensions** leveraging Matryoshka embedding properties
3. **Apply same workflow** as `cz_benchmark_evaluation.ipynb` for consistency
4. **Create cell embeddings** using expression-weighted averaging via `create_cell_embeddings()`

## Implementation Design

### Core Classes and Functions

```python
@dataclass
class CZBenchmarkConfig:
    """Configuration for CZ benchmarks evaluation"""
    tissue_types: List[str] = field(default_factory=lambda: [
        'Blood', 'Bone_Marrow', 'Lung', 'Mammary', 'Thymus'
    ])
    embedding_combinations: List[str] = field(default_factory=lambda: [
        'genept_1024d', 
        'genept_1024d_scgpt_512d', 
        'genept_1024d_transcriptformer_2048d'
    ])
    cv_folds: int = 5
    min_class_size: int = 10
    random_seed: int = 42

class CZBenchmarkEvaluator:
    """Handles CZ benchmarks evaluation workflow"""

    def load_tabula_sapiens_dataset(self, tissue_type: str) -> AnnData:
        """Load Tabula Sapiens v2 curated dataset from local h5ad file"""

    def load_precomputed_embeddings(self, tissue_type: str, model_type: str) -> pd.DataFrame:
        """Load precomputed cell embeddings from parquet files"""

    def generate_genept_1024d_embeddings(self, tissue_type: str) -> np.ndarray:
        """Generate GenePT cell embeddings at 1024 dimensions by truncating existing 3072d gene embeddings"""

    def combine_embeddings(self, genept_emb: np.ndarray, foundation_emb: np.ndarray) -> np.ndarray:
        """Concatenate GenePT with foundation model embeddings"""

    def run_label_prediction_task(self, embeddings: np.ndarray, adata: AnnData,
                                config: CZBenchmarkConfig) -> Dict:
        """Execute CZ label prediction task"""

    def evaluate_all_combinations(self) -> pd.DataFrame:
        """Run evaluation across all tissue types and embedding combinations"""
```

### Evaluation Workflow

1. **Data Loading Phase**
   - Load Tabula Sapiens v2 curated datasets from local h5ad files
   - Load precomputed scGPT and Transcriptformer cell embeddings
   - Generate GenePT cell embeddings at 1024 dimensions (Phase 1 complete)

2. **Embedding Preparation Phase**
   - Align cell barcodes across all embedding types
   - Handle missing cells/genes gracefully
   - Create embedding combinations via concatenation

3. **CZ Task Execution Phase**
   - Initialize `MetadataLabelPredictionTask` with standardized parameters
   - Run cross-validation for each embedding combination
   - Collect metrics: accuracy, F1, precision, recall, AUROC

4. **Results Analysis Phase**
   - Compare performance across embedding combinations
   - Analyze tissue-specific performance patterns
   - Generate comparative visualizations

### Key Technical Considerations

#### GenePT Embedding Generation
```python
# Generate 1024d GenePT embeddings by truncating existing 3072d embeddings
def generate_genept_1024d_for_tissue(tissue_type: str):
    # Load existing 3072d GenePT gene embeddings from huggingface_model directory
    gene_embeddings_df = pd.read_parquet('data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet')
    # NOTE: Gene names are in the index, not a column (index.name = 'gene_name')

    # Get embedding columns (0-3071) and truncate to first 1024 dimensions
    embedding_cols = [col for col in gene_embeddings_df.columns if str(col).isdigit() and int(col) >= 0]
    embedding_cols_1024 = [col for col in embedding_cols if int(col) < 1024]  # First 1024 dimensions

    # Create truncated embedding dataframe (gene names remain in index)
    gene_embeddings_1024d = gene_embeddings_df[embedding_cols_1024].copy()

    # Load tissue dataset from local h5ad file
    adata = load_tabula_sapiens_dataset(tissue_type)

    # CRITICAL: Process gene names to handle mixed formats in feature_name column
    gene_list = []
    for gene in adata.var['feature_name']:
        # Handle different formats:
        # 1. "GENE_ENSG00000123456" -> extract "GENE"
        # 2. "ENSG00000123456.15" -> use as-is (will likely not match)
        # 3. "GENE" -> use as-is
        if '_ENSG' in gene:
            # Extract gene symbol before _ENSG
            gene_symbol = gene.split('_ENSG')[0]
            gene_list.append(gene_symbol)
        else:
            # Use as-is for normal gene symbols or ENSG IDs
            gene_list.append(gene)

    print(f"Processed feature_name column for gene symbols")
    print(f"Example genes after processing: {gene_list[:5]}")

    # Reset index to make gene_name a column for create_embedding_matrix
    gene_embeddings_with_names = gene_embeddings_1024d.reset_index()

    embedding_matrix, valid_indices = create_embedding_matrix(
        gene_embeddings_with_names, gene_list, id_column='gene_name'
    )

    cell_embeddings = create_cell_embeddings(
        adata.X, embedding_matrix, valid_indices
    )

    return cell_embeddings
```

#### Embedding Combination Strategy
```python
def create_combined_embeddings(genept_1024d, foundation_model_emb):
    """Concatenate embeddings along feature dimension"""
    # Ensure cell alignment
    aligned_cells = align_cell_barcodes(genept_1024d, foundation_model_emb)
    
    # Concatenate along feature axis
    combined_emb = np.concatenate([
        genept_1024d[aligned_cells], 
        foundation_model_emb[aligned_cells]
    ], axis=1)
    
    return combined_emb, aligned_cells
```

#### CZ Task Integration
```python
def run_cz_label_prediction(embeddings, adata, tissue_type):
    """Run CZ label prediction task with standardized parameters"""
    from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
    
    # Initialize task
    task = MetadataLabelPredictionTask()
    
    # Prepare task input
    task_input = MetadataLabelPredictionTaskInput(
        adata=adata,
        cell_representations=embeddings,
        labels_key="cell_type",  # Target cell type labels
        n_folds=5,
        min_class_size=10,
        random_seed=42
    )
    
    # Execute task
    results = task.run(task_input)
    
    return results
```

## Expected Results Structure

### Metrics Collection
- **Per-tissue performance** for each embedding combination
- **Aggregated statistics** across all tissues
- **Comparative analysis** showing improvement from combination strategies

### Performance Matrix
```
Tissue         | GenePT_1024d | GenePT+scGPT | GenePT+TF | Best_Combination
---------------|--------------|--------------|-----------|----------------
Blood          | 0.85         | 0.89         | 0.91      | GenePT+TF
Bone_Marrow    | 0.82         | 0.86         | 0.88      | GenePT+TF  
Lung           | 0.78         | 0.83         | 0.85      | GenePT+TF
Mammary        | 0.80         | 0.82         | 0.84      | GenePT+TF
Thymus         | 0.83         | 0.87         | 0.89      | GenePT+TF
```

## Implementation Priority

1. **Phase 1**: Generate GenePT 1024d embeddings for all 5 tissues ✅
   - **Status**: Implemented in `notebooks/cz_benchmarks_genept_1024d.ipynb`
   - **Key Functions**:
     - `load_tabula_sapiens_dataset()` - Load h5ad files from local directory
     - `load_genept_3072d_embeddings()` - Load gene embeddings from huggingface_model
     - `truncate_to_1024d()` - Truncate embeddings using Matryoshka property
     - `process_gene_names()` - Handle GENE_ENSG format in feature_name column
     - `generate_genept_1024d_for_tissue()` - Generate cell embeddings for each tissue
   - **Output**: Parquet files saved to `data/cz_benchmark/embeddings/genept_1024d/`

2. **Phase 2**: Implement embedding combination and alignment logic ✅
   - **Status**: Implemented in `notebooks/cz_benchmarks_combine_embeddings.ipynb`
   - **Key Functions**:
     - `load_embeddings_for_tissue()` - Load and align all embedding types
     - `create_combined_embeddings()` - Create GenePT alone, GenePT+scGPT, GenePT+Transcriptformer
     - `save_combined_embeddings()` - Save with metadata to parquet
   - **Output**: Combined embeddings saved to `data/cz_benchmark/embeddings/combined/`

3. **Phase 3-5**: CZ Benchmarks Integration and Evaluation ✅
   - **Status**: Implemented in `notebooks/cz_benchmarks_label_prediction.ipynb`
   - **Key Components**:
     - `run_cz_label_prediction()` - Wrapper for CZ's MetadataLabelPredictionTask
     - `evaluate_tissue_embedding_combination()` - Complete evaluation pipeline
     - Results collection with accuracy, F1, precision, recall metrics
     - Comparative visualizations and performance analysis
   - **Output**: Results and figures saved to `data/cz_benchmark/results/`

## Dependencies

- `scanpy` and `anndata` for loading h5ad files
- `pandas` and `numpy` for data manipulation
- `src.inference` module (`create_embedding_matrix`, `create_cell_embeddings`)
- Existing precomputed scGPT and Transcriptformer cell embeddings
- GenePT gene embeddings from `data/huggingface_model/`
- `czbenchmarks` package (for Phase 3 - label prediction task integration)

## Success Criteria

- **Reproducible benchmarking** using CZ's standardized framework
- **Comprehensive comparison** across embedding strategies
- **Actionable insights** on optimal embedding combination approaches
- **Performance quantification** with statistical significance testing

This evaluation will provide definitive evidence of GenePT's performance alone and its synergistic benefits when combined with foundation model embeddings.