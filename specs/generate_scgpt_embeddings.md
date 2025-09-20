# scGPT Embedding Generation Specification

## Overview
Create a script to generate scGPT embeddings for cells in the CellXGene dataset using the pre-trained scGPT model. This will process raw gene expression data and generate 512-dimensional embeddings that capture cell type and biological state information.

## Background
- scGPT is a foundation model for single-cell biology that generates embeddings from gene expression data
- The model takes raw gene expression profiles and outputs 512-dimensional embeddings
- We need to process the CellXGene v2 dataset to generate embeddings for all cells
- These embeddings will be used alongside GenePT embeddings for cell type classification

## Requirements

### Input Data
1. **Raw expression data**: Located in CellXGene v2 dataset files
   - Format: H5AD files with expression matrices
   - Contains: Gene expression counts/normalized values
   - Cell metadata including cell types

2. **Pre-trained scGPT model**
   - Model checkpoint from HuggingFace or local storage
   - Model configuration and vocabulary files
   - Gene token mappings

### Output
1. **scGPT embeddings**: 512-dimensional vectors per cell
   - Format: Parquet files with columns:
     - `cell_id`: Unique cell identifier
     - `scgpt_embedding`: 512-dim numpy array
   - Organized by source file for alignment with training data

2. **PyTorch tensor format** (optional for faster loading)
   - `.pt` files with batched tensors
   - Metadata including cell IDs and source files

## Technical Architecture

### Components

1. **Data Loader**
   - Load H5AD files efficiently
   - Extract expression matrices (X)
   - Handle sparse matrices
   - Batch cells for GPU processing

2. **Gene Vocabulary Alignment**
   - Map dataset genes to scGPT vocabulary
   - Handle missing genes with padding/defaults
   - Normalize gene names (uppercase, symbol mapping)

3. **Expression Preprocessing**
   - Library size normalization
   - Log transformation if needed
   - Scaling to model's expected input range
   - Binning for discrete tokens (if required by model)

4. **Model Inference**
   - Load pre-trained scGPT checkpoint
   - Configure for inference mode (no gradients)
   - Process batches through encoder
   - Extract embeddings from specified layer

5. **Batch Processing**
   - Process cells in batches (e.g., 64-512 cells)
   - GPU memory management
   - Progress tracking
   - Checkpoint saving for recovery

6. **Output Management**
   - Save embeddings in parquet format
   - Optional PT tensor format
   - Maintain cell-file mappings
   - Compression for storage efficiency

## Implementation Plan

### Phase 1: Setup and Model Loading
1. Install scGPT package and dependencies
2. Download pre-trained model checkpoint
3. Test model loading and basic inference
4. Verify embedding dimensions and format

### Phase 2: Data Processing Pipeline
1. Create H5AD file reader with batch support
2. Implement gene vocabulary mapping
3. Add expression preprocessing steps
4. Test on small subset of data

### Phase 3: Batch Inference System
1. Implement batched model inference
2. Add GPU memory management
3. Create checkpoint/recovery system
4. Add progress tracking with tqdm

### Phase 4: Output Generation
1. Implement parquet writer for embeddings
2. Add PT tensor format option
3. Create file organization structure
4. Add validation checks

### Phase 5: Scaling and Optimization
1. Optimize batch sizes for GPU
2. Add multi-GPU support if needed
3. Implement incremental processing
4. Add S3 upload for results

## Configuration Options

```yaml
model:
  checkpoint_path: "path/to/scgpt_checkpoint"
  model_type: "scGPT"
  embedding_layer: -1  # Last layer
  device: "cuda"
  
data:
  input_dir: "data/cellxgene_v2/h5ad_files"
  gene_vocab_file: "scgpt_gene_vocab.json"
  batch_size: 256
  max_genes: 3000  # Top variable genes
  
preprocessing:
  normalize: true
  log_transform: true
  scaling: "standard"
  binning: true
  n_bins: 51
  
output:
  output_dir: "data/scgpt_embeddings"
  format: ["parquet", "pt"]
  compression: "snappy"
  checkpoint_every: 1000
```

## Error Handling

1. **Missing genes**: Use zero padding or model's default token
2. **Memory errors**: Reduce batch size automatically
3. **Corrupted files**: Skip and log, continue processing
4. **Model errors**: Save checkpoint and retry with smaller batch

## Performance Targets

- Process 1000 cells/second on single GPU
- Generate embeddings for 1M cells in ~20 minutes
- Memory usage < 16GB GPU RAM
- Output file size ~2GB per 100k cells

## Validation

1. Check embedding dimensions (512)
2. Verify embedding magnitudes are reasonable
3. Test clustering of known cell types
4. Compare with existing scGPT embeddings (if available)

## Dependencies

```python
# Core dependencies
scgpt  # or geneformer, depending on model choice
scanpy  # For H5AD file handling
anndata  # For single-cell data structures
torch  # For model inference
pandas  # For data manipulation
numpy  # For array operations
tqdm  # For progress bars
pyarrow  # For parquet files
```

## Notes

- The pre-trained model may be from scGPT, Geneformer, or similar foundation models
- Consider using half-precision (fp16) for faster inference if model supports it
- May need to process in chunks due to memory constraints
- Coordinate with existing data organization (source files, cell IDs)