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

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more detail on getting up and running.

### Requirements
- Python 3.10 (required for scGPT compatibility)
- Standard scientific Python packages (pandas, numpy, scikit-learn)
- Special dependencies:
  - scGPT
  - AnnData
  - Hugging Face datasets/models

### Installation
We use `uv` for package management.  Install using the instructions at https://docs.astral.sh/uv/getting-started/installation/ 

Then:

```bash
uv sync
uv pip install -e '.[dev]' # to use testing tools etc
```

### Running tools
```bash
# Format code
scripts/format.sh
uv run isort --gitignore .

# run tests
uv run pytest
```

### Training the CellXGene MLP Model

The `scripts/train_cellxgene_mlp.py` script trains an MLP classifier on CellXGene embeddings for cell type classification. It supports both standard training and hyperparameter optimization using Optuna.

#### Basic Training

```bash
# Train with local data
python scripts/train_cellxgene_mlp.py \
  --local-data-dir data/cellxgene_embeddings/training_v1_shuffled \
  --test-data-dir data/cellxgene_embeddings/test_v1 \
  --cell-types-file cell_types_filtered.csv \
  --epochs 10 \
  --batch-size 1024 \
  --learning-rate 4.366e-5 \
  --device cuda \
  --wandb-project cellxgene-mlp

# Stream directly from S3 (requires AWS credentials)
python scripts/train_cellxgene_mlp.py \
  --s3-bucket pythiomicsdata \
  --s3-prefix cellxgene_v2/training_v1_shuffled \
  --test-data-dir data/cellxgene_embeddings/test_v1 \
  --download-if-missing \
  --epochs 10 \
  --device cuda

# Resume from checkpoint
python scripts/train_cellxgene_mlp.py \
  --resume-from checkpoints/checkpoint_epoch5_batch1000.pt \
  --local-data-dir data/cellxgene_embeddings/training_v1_shuffled \
  --test-data-dir data/cellxgene_embeddings/test_v1
```

#### Hyperparameter Tuning with Optuna

The training script supports automatic hyperparameter optimization using Optuna. Simply provide a YAML configuration file with `--tuning-config`:

```bash
# Quick tuning with example config (10 trials)
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_quick.yaml \
  --local-data-dir data/cellxgene_embeddings/training_v1_shuffled \
  --test-data-dir data/cellxgene_embeddings/test_v1 \
  --wandb-project cellxgene-tuning

# Full hyperparameter search (100 trials)
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_full.yaml \
  --tuning-n-trials 100 \
  --local-data-dir data/cellxgene_embeddings/training_v1_shuffled \
  --test-data-dir data/cellxgene_embeddings/test_v1

# Resume previous tuning study
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_full.yaml \
  --tuning-storage sqlite:///optuna_study.db \
  --tuning-n-trials 50  # Run 50 more trials
```

##### Example Tuning Configurations

**Quick Search** (`specs/examples/tuning_quick.yaml`):
- 10-20 trials for rapid exploration
- Tests key hyperparameters: learning rate, dropout, hidden layers
- 2 epochs per trial for quick evaluation
- Good for initial exploration

**Full Search** (`specs/examples/tuning_full.yaml`):
- Comprehensive hyperparameter space
- Includes optimizer choice, learning rate scheduling
- Warm-starts from best known configurations
- Suitable for production model optimization

**Custom Tuning**:
Create your own YAML config following this structure:
```yaml
optuna:
  study_name: "my_study"
  direction: "minimize"
  metric_to_optimize: "val_loss"
  n_trials: 30
  n_epochs_per_trial: 2

hyperparameters:
  # Parameters to optimize
  dropout:
    type: "float"
    low: 0.0
    high: 0.5
  
  batch_size:
    type: "categorical"
    choices: [512, 1024, 2048]

fixed_params:
  # Parameters with fixed values (not optimized)
  learning_rate: 1e-4  # Fixed learning rate
  n_hidden_layers: 3    # Fixed architecture
  epochs: 10            # For final training
  device: "cuda"
```

**Important Notes**: 
- Any parameter you want to keep fixed should go in `fixed_params`, not `hyperparameters`
- Parameters in `hyperparameters` will be optimized, while those in `fixed_params` remain constant across all trials
- **Command-line arguments override config values**: If you specify a parameter both in the config file and on the command line, the command-line value takes precedence for certain parameters (data paths, checkpoints, wandb settings)

#### Key Parameters

**Training Parameters:**
- `--local-data-dir`: Directory with pre-shuffled training parquet files
- `--test-data-dir`: Directory with validation data
- `--cell-types-file`: CSV file mapping cell type names to codes
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 1024)
- `--learning-rate`: Learning rate (default: 4.366e-05)
- `--n-hidden-layers`: Number of hidden layers (default: 3)
- `--dropout`: Dropout rate (default: 0.053)
- `--checkpoint-dir`: Where to save model checkpoints

**Hyperparameter Tuning:**
- `--tuning-config`: Path to YAML configuration file (enables tuning mode)
- `--tuning-n-trials`: Override number of trials from config
- `--tuning-timeout`: Maximum time in seconds for tuning
- `--tuning-storage`: Database URL for study persistence (e.g., `sqlite:///study.db`)

**Monitoring:**
- `--wandb-project`: Weights & Biases project for experiment tracking
- `--wandb-entity`: W&B team/organization
- `--enable-hierarchical-metrics`: Use Cell Ontology for hierarchical evaluation

For full parameter documentation:
```bash
python scripts/train_cellxgene_mlp.py --help
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
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
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
- [ ] Prompt improvements
  - [x] Remove aging
  - [x] Add cell type
  - [ ] Add tissue type
  - [ ] Add dysfunctional cell type
- [ ] scGPT with batch tokens
- [ ] scGPT with modality tokens
- [ ] scGPT with combined batch/modality tokens
- [ ] Complete Tabula Sapiens cell embedding
- [ ] Cell-document bidirectional lookups
- [ ] Cell separation analysis
  
## Contributing

This is a preliminary repository with work in progress. Code is mostly untested but being actively developed. Contributions and collaborations are welcome.

## License

This project is subject to the x   described in the LICENSE.md file.
