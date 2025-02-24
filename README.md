# GenePT-tools

Tools to use and expand the capabilities of the original GenePT. This repository contains utilities and notebooks for working with gene embeddings and single-cell RNA sequencing data.

## Overview

This project builds upon the GenePT paper and provides tools to:
- Compare different embedding approaches (GenePT vs scGPT)
- Work with large single-cell datasets like Tabula Sapiens
- Generate composable embeddings across different dimensions
- Perform cell type classification using embeddings

### Results

The following image shows a detailed summary of the results of the comparison between GenePT and scGPT zero-shot classification so far:
![Comparison of embedding methods](./img/comparison_with_small.png)
We used a [Google Sheet](https://docs.google.com/spreadsheets/d/1Epjhj0ZBFEdY5iIONLi9-I8geACofHlaFGvKu561I5U/edit?usp=sharing) to format the output


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

## Important files
```
GenePT-tools/
├── src/        # utility functions
└── notebooks/  # analysis notebooks
```
### Notebooks

Take a look at `generate_genept_embeddings.ipynb` to see how to generate a GenePT embeddings and dataset and upload them to HuggingFace Hub.  `create_hf_repos.ipynb` will create a new repository for the embeddings and dataset.

Take a look at `tabula_sapiens_*.ipynb` for a comparison of cell type classification using GenePT and scGPT embeddings.

| Notebook                                                                                                   | Description                                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`generate_genept_embeddings.ipynb`](notebooks/generate_genept_embeddings.ipynb)                           | Generates the GenePT embeddings and dataset for upload to HuggingFace Hub         |
| [`tabula_sapiens_embed_genept.ipynb`](notebooks/tabula_sapiens_embed_genept.ipynb)                         | Evaluates GenePT embeddings' cell classification performance on Tabula Sapiens    |
| [`create_hf_repos.ipynb`](notebooks/create_hf_repos.ipynb)                                                 | Creates the initial HuggingFace repositories for the GenePT embeddings and dataset|
| [`tabula_sapiens_eda.ipynb`](notebooks/tabula_sapiens_eda.ipynb)                                           | Exploratory analysis of the Tabula Sapiens single-cell dataset                    |
| [`tabula_sapiens_embed_genept.ipynb`](notebooks/tabula_sapiens_embed_genene.ipynb)                         | Embed a subset of the Tabula Sapiens dataset using GenePT embeddings              |
| [`tabula_sapiens_embed_scgpt.ipynb`](notebooks/tabula_sapiens_embed_scgpt.ipynb)                           | Embed a subset of the Tabula Sapiens dataset using scGPT embeddings               |
| [`tabula_sapiens_analysis_all.ipynb`](notebooks/tabula_sapiens_analysis_all.ipynb)                         | A comparison of GenePT and scGPT embeddings for cell type classification on TS    |

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
- [ ] prompt improvements
    [x] remove aging
    [x] add cell type
    [ ] add tissue type
    [ ] add disfuctional cell type
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
