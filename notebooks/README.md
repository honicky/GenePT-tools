# GenePT-tools Notebooks

This directory contains Jupyter notebooks for analyzing gene embeddings and single-cell RNA sequencing data using the GenePT methodology. Below is a comprehensive summary of each notebook and its purpose.

## Setup and Core Infrastructure

### `notebook_setup.ipynb`
**Purpose**: Common setup notebook sourced by other notebooks  
**Key Functions**:
- Enables autoreload for development
- Sets up `repo_dir` and `data_dir` variables
- Configures pandas display options
- Downloads and extracts required data files

## Gene Embedding Generation

### `generate_genept_embeddings.ipynb`
**Purpose**: Generate custom GenePT embeddings using OpenAI's text-embedding-3-large model  
**Workflow**:
1. Load gene descriptions from NCBI and UniProt
2. Generate enhanced descriptions using GPT-4-mini with specialized prompts
3. Create 3072-dimensional embeddings capturing gene associations, cell types, drug interactions, and pathways
4. Handle duplicate genes by averaging embeddings
5. Upload results to HuggingFace datasets

**Key Output**: `embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet`

## Exploratory Data Analysis (EDA)

### `genept_data_scratch.ipynb`
**Purpose**: Basic exploration of GenePT gene descriptions and embeddings  
**Analysis**: 
- Examines generated gene descriptions from GPT models
- Loads NCBI/UniProt gene summaries
- Tests basic embedding functionality with blood age analysis dataset

### `human_pancreas_eda.ipynb`
**Purpose**: Exploratory analysis of human pancreas single-cell data  
**Dataset**: Human pancreas scRNA-seq (16,382 cells × 19,093 genes)  
**Analysis**: Cell type distribution across pancreatic cell types (alpha, beta, ductal, acinar, delta, gamma, etc.)

### `pbmc3k_eda.ipynb`
**Purpose**: Basic exploration of PBMC3K dataset  
**Dataset**: 2,700 peripheral blood mononuclear cells × 32,738 genes  
**Analysis**: Standard single-cell data exploration

### `tabula_sapiens_eda.ipynb`
**Purpose**: Comprehensive analysis of Tabula Sapiens dataset  
**Dataset**: 1+ million cells from CellXGene  
**Analysis**:
- Cell type and tissue distribution analysis  
- Quality control metrics
- Cross-tabulation of donors vs cell types
- Memory-efficient data loading strategies

## Age-Related Analysis

### `blood_age_data_analysis.ipynb`
**Purpose**: Predict age from blood gene expression using GenePT embeddings  
**Dataset**: Blood samples from diabetes study (493 samples)  
**Methods**:
- Creates GenePT-w embeddings via weighted averaging
- Trains LightGBM regression models for age prediction
- Group cross-validation by subject
- UMAP visualization of embedding space
- Feature importance analysis

**Key Results**: Mean R² of ~0.05 for age prediction

### `brain_age_data_analysis.ipynb`
**Purpose**: Age prediction from brain tissue gene expression  
**Dataset**: GTEx brain samples (2,931 samples across multiple brain regions)  
**Methods**:
- Similar to blood analysis but with brain-specific data
- Cross-validation with brain region and sex as features
- Hierarchical age group classification
- Confusion matrix analysis for age categories

**Key Results**: Similar performance to blood analysis

### `brain_age_data_analysis_full_embeddings.ipynb`
**Purpose**: Enhanced brain age analysis with full embedding dimensions
**Methods**: Uses complete embedding space rather than reduced dimensions

## Disease and Drug Analysis

### `lupus_data_analysis.ipynb`
**Purpose**: Analyze lupus patient drug response using GenePT embeddings  
**Dataset**: Lupus clinical trial data (468 samples)  
**Analysis**:
- Interferon status prediction from gene expression
- Drug dose effect analysis (Placebo, 10mg, 50mg, 200mg)
- Longitudinal analysis (baseline vs 24-week timepoints)
- Patient response classification

**Key Results**: 91-93% accuracy for interferon status classification

### `myeloid_logistic_regression.ipynb`
**Purpose**: Cell type classification for myeloid cells  
**Dataset**: Multiple sclerosis brain tissue data  
**Methods**:
- Binary classification between cell types
- Logistic regression with GenePT embeddings
- Feature importance analysis for cell type markers

## scGPT Integration

### `tabula_sapiens_embed_scgpt.ipynb`
**Purpose**: Generate scGPT embeddings for Tabula Sapiens data  
**Methods**:
- Uses pre-trained scGPT model for cell embeddings
- Processes 100K cells with 512-dimensional embeddings  
- Cross-validation with donor holdout
- Comparison of KNN, Random Forest, and LightGBM classifiers

**Key Results**: Demonstrates scGPT embedding generation and evaluation

### `tabula_sapiens_embed_genept.ipynb` / `tabula_sapiens_embed_genept_all.ipynb`
**Purpose**: Apply GenePT embeddings to Tabula Sapiens for cell type prediction  
**Methods**: Similar to scGPT analysis but using GenePT methodology

### `tabula_sapiens_analysis_all.ipynb`
**Purpose**: Comprehensive analysis combining multiple embedding approaches

## CellXGene v2 Analysis

### `cellxgene_v2_training_data_eda.ipynb`
**Purpose**: Exploratory analysis of CellXGene v2 datasets for training set creation  
**Analysis**:
- Metadata analysis of 961 datasets
- Cell type frequency analysis across datasets
- Quality control and filtering criteria
- Training/test split strategy based on publication dates

### `cellxgene_v2_training_data.ipynb`
**Purpose**: Extract and prepare training data from CellXGene v2  
**Methods**:
- Optimized data extraction using Integer Linear Programming
- Efficient S3 data access with row group optimization
- Cell type balancing (10K cells per type where possible)
- Parquet file generation for training

### `cellxgene_v2_test_data.ipynb`
**Purpose**: Prepare test set from post-2023 CellXGene data
**Methods**: Similar to training data preparation but for held-out test set

### `cellxgene_v2_mlp.ipynb`
**Purpose**: Train deep learning models on CellXGene v2 data  
**Architecture**: Multi-layer perceptron with batch normalization and dropout  
**Methods**:
- Optuna hyperparameter optimization
- Out-of-memory training with double buffering
- Weights & Biases experiment tracking
- Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Ranking metrics (Recall@k, MRR@k, DCG@k)

**Key Results**: 
- 87% Recall@10 for cell type prediction
- Hierarchical clustering analysis of confusion patterns
- Interactive confusion heatmap visualization

## Advanced Evaluation

### `hierarchical_evaluation_demo.ipynb`
**Purpose**: Demonstrate hierarchical evaluation using Cell Ontology  
**Methods**:
- Integrates OnClass methodology for hierarchical metrics
- Cell Ontology-based evaluation
- Hierarchical precision, recall, and F1 scores
- Comparison of standard vs hierarchical metrics

**Key Results**: 
- Standard F1: ~0.09
- Hierarchical F1: ~0.91 (gives partial credit for biologically related predictions)

## Repository Creation

### `create_hf_repos.ipynb`
**Purpose**: Create and manage HuggingFace repositories for model and data sharing  
**Functions**: Repository setup and dataset uploading workflows

## Data Sources and Key Metrics

- **Primary Datasets**: Tabula Sapiens, CellXGene v2, GTEx, PBMC3K, Human Pancreas
- **Cell Types**: 700+ unique cell types across all analyses  
- **Embedding Dimensions**: 3072 (GenePT), 512 (scGPT)
- **Model Performance**: Cell type classification accuracy ranges from 60-90% depending on dataset complexity
- **Key Innovation**: GenePT-w (weighted) embeddings that incorporate gene expression levels with semantic embeddings

## Usage Patterns

1. **Start with**: `notebook_setup.ipynb` (always run first)
2. **For new embeddings**: `generate_genept_embeddings.ipynb`  
3. **For EDA**: Dataset-specific `*_eda.ipynb` notebooks
4. **For modeling**: `cellxgene_v2_mlp.ipynb` or disease-specific analysis notebooks
5. **For evaluation**: `hierarchical_evaluation_demo.ipynb`

## Key Dependencies

- OpenAI API (for embedding generation)
- scGPT (for comparative embeddings)  
- PyTorch (for deep learning models)
- scanpy/anndata (for single-cell analysis)
- HuggingFace datasets (for data sharing)
- Weights & Biases (for experiment tracking)
- OnClass (for hierarchical evaluation)