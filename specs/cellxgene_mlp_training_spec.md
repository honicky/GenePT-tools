# CellXGene MLP Training Script Specification

## Overview
Convert the notebook `cellxgene_v2_mlp.ipynb` into a production-ready training script that efficiently handles out-of-core datasets using pre-shuffled data from S3. The script will stream data sequentially from pre-shuffled batch files, eliminating the need for complex in-memory shuffling.

## Key Requirements

### 1. Data Management
- **Pre-shuffled data**: Use 376 pre-shuffled batch files (~238MB each) from S3
- **Streaming from S3**: Direct streaming or local caching of batch files
- **Sequential processing**: Read files in order, relying on pre-shuffling
- **Out-of-core processing**: Handle 87GB dataset without loading into memory

### 2. Architecture Components

#### 2.1 Data Pipeline
```
S3ParquetStreamDataset (Simplified Implementation)
├── S3 Integration
│   ├── List batch files from S3 bucket
│   ├── Stream or cache files locally
│   └── Handle network failures gracefully
├── Sequential Processing
│   ├── Read batch files in order (batch_0000.parquet to batch_0375.parquet)
│   ├── Apply cell type filtering if needed
│   └── Convert to tensor format
├── Batch Generation
│   ├── Create mini-batches from each file
│   ├── Efficient tensor conversion
│   └── GPU memory management
└── Epoch Management
    ├── File-order shuffling and intra-file shuffling per epoch
    ├── Progress tracking
    └── Resume from specific batch
```

#### 2.2 Training Loop
```
MLPTrainer
├── Model Architecture
│   ├── Configurable MLP with n_hidden_layers
│   ├── BatchNorm and Dropout layers
│   └── Linear interpolation for hidden dimensions
├── Optimization
│   ├── Simple AdamW optimizer (advanced optimizers as future work)
│   └── Hyperparameter-friendly design for integration with Optuna/Ray Tune
├── Checkpointing
│   ├── Periodic model saves
│   ├── Resume from checkpoint
│   ├── Make sure to checkpoint/save the final model as well
│   └── Best model tracking
└── Monitoring
    ├── Weights & Biases integration
    ├── Multi-level evaluation (5k, 120k samples)
    └── Comprehensive metrics tracking
```

## Implementation Plan

### Phase 1: Data Loading Infrastructure

#### 1.1 S3ParquetStreamDataset Class
```python
class S3ParquetStreamDataset(IterableDataset):
    """
    Streaming dataset for pre-shuffled S3 parquet files
    - Each file contains 10,000 pre-shuffled samples
    - Sequential reading with optional epoch-level file shuffling
    - Uses local files if available, downloads from S3 if not
    """
    def __init__(
        self,
        s3_bucket: str = "pythiomicsdata",
        s3_prefix: str = "cellxgene_v2/training_v1_suffled",
        local_data_dir: Optional[Path] = None,  # Check here first for existing files
        cell_types: List[str],
        cell_type_codes: np.ndarray,
        n_dims: int = 500,  # Reduced embedding dimension
        batch_size: int = 1024,
        download_if_missing: bool = True,  # Download from S3 if not found locally
        shuffle_files_per_epoch: bool = True,
        aws_profile: str = "xcellerate",
        start_batch_file: int = 0,  # For resuming
        end_batch_file: Optional[int] = None,  # For debugging with subset
        seed: int = 42
    )
    
    def _list_s3_files(self) -> List[str]
    def _download_or_stream_file(self, s3_key: str) -> pd.DataFrame
    def _process_batch_file(self, df: pd.DataFrame) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]
```

#### 1.2 Data Location Strategy
```python
# The dataset will intelligently handle data location:
# 1. Check if file exists locally first
# 2. Download from S3 only if needed
# 3. Optionally keep downloaded files for future use

def get_batch_file(filename: str, local_dir: Path, s3_bucket: str, s3_key: str) -> Path:
    """Get a batch file, using local copy if available, downloading if not"""
    local_path = local_dir / filename
    
    if local_path.exists():
        print(f"Using local file: {local_path}")
        return local_path
    
    if download_if_missing:
        print(f"Downloading from S3: {s3_key}")
        # Download from S3 to local_dir
        s3_client.download_file(s3_bucket, s3_key, local_path)
        return local_path
    else:
        raise FileNotFoundError(f"File not found locally and download disabled: {filename}")

# Example usage patterns:

# 1. Use pre-downloaded data (no S3 access needed)
dataset = S3ParquetStreamDataset(
    local_data_dir=Path("/data/cellxgene/training_shuffled"),
    download_if_missing=False,  # Only use local files
    ...
)

# 2. Check local first, download if missing
dataset = S3ParquetStreamDataset(
    local_data_dir=Path("./cache/training"),
    download_if_missing=True,  # Download missing files
    ...
)

# 3. Always download fresh (no local storage)
dataset = S3ParquetStreamDataset(
    local_data_dir=Path("/tmp/training"),  # Temporary location
    download_if_missing=True,
    ...
)
```

### Phase 2: Model Training Components

#### 2.1 Configuration Management
```python
@dataclass
class TrainingConfig:
    # Data parameters
    data_dir: Path
    n_dims: int = 500
    batch_size: int = 1024
    buffer_size: int = 100_000
    chunk_size: int = 50_000
    
    # Model parameters
    n_hidden_layers: int = 3
    dropout: float = 0.05
    
    # Training parameters
    learning_rate: float = 4.4e-5
    weight_decay: float = 1e-5
    epochs: int = 10
    
    # Evaluation parameters
    eval_every_n_batches: int = 10
    checkpoint_every_n_batches: int = 1000
    
    # System parameters
    device: str = "auto"
    num_workers: int = 4
    mixed_precision: bool = True
```

#### 2.2 MLPTrainer Class
```python
class MLPTrainer:
    """
    Main training orchestrator
    Designed to be usable both standalone and within hyperparameter optimization
    """
    def __init__(self, config: TrainingConfig, trial=None)  # Optional Optuna trial
    def create_model(self) -> nn.Module
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer
    def train_epoch(self, dataloader: DataLoader, epoch: int)
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]
    def save_checkpoint(self, batch_idx: int)
    def load_checkpoint(self, checkpoint_path: Path)
    def run(self) -> Dict[str, float]  # Returns final metrics for hyperparam optimization
    
# Example usage in hyperparameter optimization
def objective(trial):
    config = TrainingConfig(
        learning_rate=trial.suggest_loguniform('lr', 1e-5, 1e-2),
        dropout=trial.suggest_uniform('dropout', 0.0, 0.5),
        n_hidden_layers=trial.suggest_int('n_hidden', 1, 4),
        batch_size=trial.suggest_categorical('batch_size', [512, 1024, 2048])
    )
    trainer = MLPTrainer(config, trial=trial)
    metrics = trainer.run()
    return metrics['val_loss']  # Or any metric to optimize
```

### Phase 3: Simple Sequential Implementation

#### 3.1 Core Design Principles
- **Correctness over performance**: Focus on accurate training dynamics
- **Sequential processing**: One file at a time, no parallelism
- **Simple batching**: Straightforward mini-batch creation from each file
- **Minimal complexity**: No prefetching, caching, or optimization initially

#### 3.2 Data Processing Flow
1. **List files**: Get ordered list of batch files from S3
2. **Shuffle file order**: Randomize file processing order each epoch
3. **Sequential download**: Download one file at a time as needed
4. **Process file**: Load parquet, filter cell types if needed
5. **Shuffle within file**: Randomize sample order within each file
6. **Create mini-batches**: Generate batches from shuffled samples
7. **Train on batches**: Forward pass, backward pass, optimizer step
8. **Move to next file**: Repeat until all files processed
9. **Next epoch**: Repeat with different file order and intra-file shuffling

#### 3.3 Key Simplifications
- **No parallel downloading**: Process files sequentially
- **No local cache**: Re-download files each epoch (can add caching later)
- **No buffer management**: Each file processed independently
- **No advanced optimizations**: Focus on correct gradient computation

### Phase 4: Monitoring & Evaluation

#### 4.1 Metrics
- **Training**: Loss, learning rate, gradient norms
- **Validation (5k)**: Fast feedback every 10 batches
  - Log loss, macro F1, precision, recall
  - Recall@k, MRR@k, DCG@k (k=2,5,10)
- **Validation (120k)**: Full evaluation every 250 batches

#### 4.2 Logging
- **Weights & Biases**: Full experiment tracking
- **Console**: Progress bars with tqdm
- **Checkpoints**: Automatic best model selection

### Phase 5: CLI Interface

```bash
# Use local data if available
python scripts/train_cellxgene_mlp.py \
    --local-data-dir /data/cellxgene/training_shuffled \
    --test-data-dir /data/cellxgene/test \
    --download-if-missing \
    --n-dims 500 \
    --batch-size 1024 \
    --epochs 10 \
    --checkpoint-dir checkpoints/ \
    --wandb-project cellxgene-mlp-v2

# Use only local data (no S3 access)
python scripts/train_cellxgene_mlp.py \
    --local-data-dir /data/cellxgene/training_shuffled \
    --test-data-dir /data/cellxgene/test \
    --no-download \
    --n-dims 500 \
    --batch-size 1024 \
    --epochs 10

# Download to cache directory if needed
python scripts/train_cellxgene_mlp.py \
    --local-data-dir ./cache/training \
    --test-data-dir ./cache/test \
    --download-if-missing \
    --s3-bucket pythiomicsdata \
    --s3-prefix cellxgene_v2/training_v1_suffled \
    --aws-profile xcellerate \
    --n-dims 500 \
    --batch-size 1024 \
    --epochs 10
```

## File Structure

```
src/
├── data_loading/
│   ├── __init__.py
│   ├── s3_dataset.py               # S3ParquetStreamDataset
│   └── utils.py                    # Helper functions for data processing
├── models/
│   ├── __init__.py
│   └── mlp_classifier.py           # MLP architecture
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # MLPTrainer class
│   ├── metrics.py                  # Evaluation metrics
│   └── config.py                   # Configuration classes
└── utils/
    ├── __init__.py
    ├── checkpoint.py                # Checkpoint utilities
    └── logging.py                   # Logging setup

scripts/
└── train_cellxgene_mlp.py          # Main training script entrypoint
```

## Testing Strategy

### Unit Tests (pytest-based)

All tests use pytest and are located in the `test/` directory. Tests are designed to avoid side effects and minimize mocking.

#### Model Tests (`test/test_mlp_classifier.py`)
- MLP architecture creation with correct dimensions
- Hidden layer dimension interpolation
- Layer ordering (Linear, BatchNorm, ReLU, Dropout)
- Forward pass shape validation
- Training/eval mode behavior
- Parameter counting
- Device placement

#### Metrics Tests (`test/test_metrics.py`)
- Recall@k, MRR@k, DCG@k calculations
- Perfect/partial/zero prediction cases
- Batch inference functionality
- Full evaluation pipeline
- Multi-device support

#### Configuration Tests (`test/test_config.py`)
- Default value initialization
- Path object conversion
- Device auto-detection
- Dictionary serialization for logging
- Optuna integration

#### Checkpoint Tests (`test/test_checkpoint.py`)
- Save/load checkpoint functionality
- Best model tracking
- Checkpoint manager with automatic cleanup
- Final model saving
- Config persistence

#### Data Loading Tests (`test/test_s3_dataset.py`)
- S3 file listing and downloading
- Local cache usage
- Parquet file processing
- Label encoding
- Batch generation
- Shuffling behavior

#### Trainer Tests (`test/test_trainer.py`)
- Model and optimizer creation
- Single batch training
- Epoch training loop
- Validation evaluation
- W&B integration (mocked)
- Resume from checkpoint

### Integration Tests
- End-to-end training on 3-file subset
- Checkpoint save and resume
- Metric computation verification
- Memory usage monitoring

### Performance Tests
- Throughput benchmarking
- Memory scaling analysis
- GPU utilization measurement


## Success Criteria

1. **Primary Goal: Match Notebook Performance**
   - Achieve same validation metrics as notebook implementation
   - Recall@10: ~87%, MRR@10: ~56%, DCG@10: ~63.5%
   - Use same hyperparameters that worked in notebook (learning_rate: 4.4e-5, dropout: 0.053, n_hidden_layers: 3)

2. **Correctness**
   - No distribution shift between batches (solved by pre-shuffled data)
   - Reproducible training with fixed seed
   - Consistent evaluation metrics

3. **Simplicity**
   - Clean, readable implementation
   - Focus on correctness over optimization
   - Easy to debug and modify

## Notes on Current Implementation Issues

The existing `ParquetBatchDataset` in the notebook has these problems:
1. **Distribution shift**: Files are processed sequentially, causing loss spikes
2. **No cross-file shuffling**: Each file becomes a separate distribution
3. **Memory inefficiency**: Loads entire chunks before processing

The new implementation addresses these via:
- **Pre-shuffled data**: Data has been shuffled across all samples before creating batch files
- **Sequential simplicity**: Each file contains a representative sample of the full distribution
- **Minimal memory footprint**: Process one 238MB file at a time
- **Focus on correctness**: Simple, debuggable implementation without premature optimization

## Data Format

The pre-shuffled data on S3 has the following characteristics:
- **Location**: `s3://pythiomicsdata/cellxgene_v2/training_v1_suffled/`
- **Files**: 376 batch files (batch_0000.parquet to batch_0375.parquet)
- **File size**: ~238MB each
- **Samples per file**: 10,000 pre-shuffled samples
- **Total size**: 87.36GB
- **Columns**: 3094 total (embeddings 0-3071 + metadata including cell_type)
- **AWS Profile**: `xcellerate`