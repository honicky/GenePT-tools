# GenePT-tools

Tools to use and expand the capabilities of the original GenePT. This repository contains utilities and notebooks for working with gene embeddings and single-cell RNA sequencing data.

## Overview

This project builds upon the GenePT paper and provides tools to:
- Compare different embedding approaches (GenePT vs scGPT)
- Work with large single-cell datasets like Tabula Sapiens
- Generate composable embeddings across different dimensions
- Perform cell type classification using embeddings

## Setup

### Requirements
- Python 3.10 (required for scGPT compatibility)
- Standard scientific Python packages (pandas, numpy, scikit-learn)
- Special dependencies:
  - scGPT
  - AnnData
  - Hugging Face datasets/models

### Installation
```bash
# Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode with all tools
pip install -e ".[dev]"
```

### Running tools
```bash
# Format code
black .
isort --gitignore .

# Run tests
pytest
```

## Features
```
GenePT-tools/
├── src/        # utility functions
└── notebooks/  # analysis notebooks
```
### Notebooks

| Notebook                                        | Description                                                                       |
|-------------------------------------------------|-----------------------------------------------------------------------------------|
| [`tabula_sapiens_embed_genept.ipynb`](notebooks/tabula_sapiens_embed_genept.ipynb) | Evaluates GenePT embeddings' cell classification performance on Tabula Sapiens    |
| [`port.ipynb`](notebooks/port.ipynb)             | Handles data preparation and uploading of GenePT embeddings to HuggingFace Hub    |
| [`lupus_data_analysis.ipynb`](notebooks/lupus_data_analysis.ipynb) | Analyzes gene expression patterns in lupus patients using GenePT embeddings       |
| [`brain_age_data_analysis.ipynb`](notebooks/brain_age_data_analysis.ipynb) | Predicts brain age from gene expression data using balanced bootstrap sampling    |
| [`blood_age_data_analysis.ipynb`](notebooks/blood_age_data_analysis.ipynb) | Analyzes blood-based gene expression patterns for age prediction                  |
| [`brain_age_data_analysis_full_embeddings.ipynb`](notebooks/brain_age_data_analysis_full_embeddings.ipynb) | Extended brain age prediction using complete gene embedding features              |
| [`tabula_sapiens_eda.ipynb`](notebooks/tabula_sapiens_eda.ipynb) | Exploratory analysis of the Tabula Sapiens single-cell dataset                    |
| [`aging_and_related_gene_query.ipynb`](notebooks/aging_and_related_gene_query.ipynb) | Queries and analyzes aging-related genes using NCBI gene summaries                |
| [`tabula_sapiens_embed_scgpt.ipynb`](notebooks/tabula_sapiens_embed_scgpt.ipynb) | Implements scGPT embeddings for Tabula Sapiens cell classification                |

### Data Processing
- Support for loading and processing large sparse AnnData files
- Integration with Hugging Face datasets

### Embedding Generation
- GenePT original embeddings
- scGPT embeddings
- Composable embeddings across different dimensions:
  - Associated genes
  - Aging related information
  - Drug interactions
  - Pathways and biological processes

### Analysis Tools
- Cell type classification
- Embedding comparison utilities
- Visualization tools for high-dimensional data

## Project Status

- [x] Exact comparison between scGPT and GenePT embeddings
- [x] Minimum cell count filtering per cell type
- [x] AnnData integration
- [x] Original GenePT embeddings support
- [ ] scGPT with batch tokens
- [ ] scGPT with modality tokens
- [ ] scGPT with combined batch/modality tokens
- [ ] Complete Tabula Sapiens cell embedding
- [ ] Cell-document bidirectional lookups
- [ ] Cell separation analysis

## Contributing

This is a preliminary repository with work in progress. Code is mostly untested but being actively developed. Contributions and collaborations are welcome.

## License

This project is licensed under the MIT License. The original GenePT weights are governed by the license of the original GenePT repository.
