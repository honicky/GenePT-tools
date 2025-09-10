# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenePT-tools is a Python bioinformatics toolkit for generating and analyzing gene embeddings, particularly building upon the GenePT paper. The project focuses on creating composable embeddings that capture various biological aspects of genes and enables cell type classification using single-cell RNA sequencing data.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync
uv pip install -e '.[dev]'
```

### Code Formatting
```bash
# Format imports
uv run isort --gitignore .

# Format code with yapf (88 column limit, 2-space indentation)
uv run yapf --in-place --recursive src/ test/

# Format notebooks
uv run black --target-version py310 notebooks/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/test_inference.py

# Run with verbose output
uv run pytest -v
```

## Core Architecture

### Key Components

1. **src/embeddings.py** - Batch processing engine for OpenAI API
   - `BatchInfo` class manages batch processing metadata
   - `get_gene_embedding_batch_requests()` creates embedding requests
   - `create_gene_descriptions_dataframe()` processes GPT responses

2. **src/inference.py** - Embedding matrix operations
   - `create_embedding_matrix()` aligns gene embeddings with expression data
   - `create_cell_embeddings()` generates cell-level embeddings via weighted averaging
   - Optional PyTorch support for GPU acceleration

3. **src/utils.py** - Data infrastructure
   - `AnnDataChunker` for efficient processing of large single-cell datasets
   - File download/extraction utilities
   - HuggingFace integration for model/dataset handling

4. **src/prompt_templates.py** - Prompt engineering templates
   - `NCBI_UNIPROT_ASSOCIATED_CELL_TYPE_TISSUE_DRUG_PATHWAY_PROMPT_V1` main prompt
   - Specialized variants for aging, cell types, drugs, pathways

5. **src/training/** - ML training infrastructure
   - `trainer.py` - Main training loop with WandB integration, validation, checkpointing
   - `optuna_manager.py` - Hyperparameter tuning with Optuna
   - `metrics.py` - Evaluation metrics including hierarchical F1 with Cell Ontology
   - `config.py` - Training configuration dataclass

6. **src/data_loading/** - Efficient data loading
   - `pt_dataset.py` - Fast PyTorch tensor dataset with cross-file batching
   - `s3_dataset.py` - S3/Parquet streaming dataset  
   - Input scaling (÷0.026) for OpenAI embeddings
   - Proper label encoding (sequential 0-n codes)

### Workflow Pattern

1. **Data Setup**: Download embeddings from Zenodo, initialize data directories
2. **Description Enhancement**: Use GPT-4 to enhance NCBI/UniProt gene summaries
3. **Embedding Generation**: Convert descriptions to 3072-dim embeddings via text-embedding-3-large
4. **Cell Embedding Creation**: Generate cell embeddings through expression-weighted averaging

### Data Organization

- `data/` - Contains CellXGene datasets, Tabula Sapiens embeddings, generated embeddings
- `save/` - Model saves and processed outputs
- `notebooks/` - Analysis and experimentation notebooks
- `img/` - Visualization outputs
- `specs/` - High level and detailed impementation instructions for features

## Configuration

- Uses `.env` files for API keys (OpenAI, HuggingFace)
- Python 3.10 required for scGPT compatibility
- Dependencies managed through `pyproject.toml`
- No explicit config files - settings embedded in code

## Testing Patterns

- Tests located in `test/` directory following pytest conventions
- Uses fixtures for test data setup
- Includes conditional imports for optional dependencies (torch)
- Tests both unit functionality and integration scenarios
  - unit tests should run quickly, and should separated into different suites than integrations tests
  - core logic functions should be pure functions (no side effects) as long as that doesn't introduce undue complexity.  This will make unit testing much cleaner
  - integration tests should not leave any state in production systems
  

## Important Notes

- The project uses `uv` for modern Python package management
- Optional PyTorch support - check imports before using torch functionality
- Large files handled through chunked processing to manage memory
- API rate limiting built into batch processing
- Sparse matrix operations used for memory efficiency with single-cell data
- Always use torch.load with weights_only=True unless there is a good reason not to
- PT (PyTorch tensor) format provides ~10x faster data loading than Parquet
- Cross-file batching ensures consistent batch sizes and optimal GPU utilization
- OpenAI embeddings require scaling (÷0.026) for proper neural network training
- Cell type codes must be sequential (0 to n-1) for proper model output dimensions
- Hyperparameter tuning automatically retries failed trials to reach target count
- WandB artifacts are used for checkpoint storage during hyperparameter optimization