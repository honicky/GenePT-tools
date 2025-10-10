# CZ Benchmark Adaptation Specification for GenePT Cell Type Classification

## Executive Summary

This specification describes the implementation strategy for adapting the CZ benchmarks cell-type classification task to evaluate GenePT model performance. Instead of using embeddings with multiple downstream classifiers, we'll leverage our pre-trained MLP classifier by freezing its weights and fine-tuning only the output layer for each dataset's specific cell types.

## Architecture Overview

### Base Model Structure
- **Pre-trained Model**: MLPClassifier from best WandB run "nwhis8xb"
  - Input: 500 dimensional GenePT embeddings
  - Architecture: 3 hidden layers with batch normalization and dropout=0.053
  - Learning rate: 0.00132 (1.32e-3)
  - Batch size: 1024
  - Pre-trained on CellXGene dataset with 377 cell types
  - Model checkpoint: Available via WandB as model files (not artifacts)

### Adaptation Strategy
1. **Freeze Pre-trained Layers**: Lock all weights except the output layer
2. **Replace Output Layer**: New linear layer mapping to dataset-specific cell types
3. **Optional Adapter Layer**: Consider adding intermediate layer if direct output replacement underperforms

## Implementation Components

### 1. Data Preprocessing Module
```python
class CZBenchmarkDataProcessor:
    """Process benchmark datasets for GenePT evaluation."""
    
    def __init__(self, config: AdaptationConfig, embedding_dim: int = 500):
        self.config = config
        self.embedding_dim = embedding_dim
        self.gene_embeddings = None
        self.label_encoder = LabelEncoder()
    
    def load_gene_embeddings(self, embedding_path=None):
        """Load pre-computed GenePT embeddings from parquet file."""
        if embedding_path is None:
            embedding_path = "data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet"
        embeddings_df = pd.read_parquet(embedding_path)
        # Gene names are in the index, select first embedding_dim columns
        embedding_cols = [str(i) for i in range(min(self.embedding_dim, len(embeddings_df.columns)))]
        self.gene_embeddings = embeddings_df[embedding_cols].reset_index()
        return self.gene_embeddings
    
    def prepare_dataset(self, adata, cell_type_col='cell_type'):
        """
        Process AnnData object for benchmark evaluation.
        NOTE: Datasets are pre-curated, no filtering needed.
        
        Steps:
        1. Handle layers (e.g., X_original in Tabula Sapiens v2) 
        2. Align gene embeddings with expression data
        3. Create cell embeddings via weighted averaging
        4. Map cell type labels to sequential codes
        5. Scale embeddings by 0.026 for OpenAI embeddings
        """
```

### 2. Cross-Validation Framework
```python
class CrossValidationEvaluator:
    """5-fold cross-validation following CZ benchmark protocol."""
    
    def __init__(self, wandb_run_id, config):
        self.wandb_run_id = wandb_run_id  # "nwhis8xb" for best model
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.load_base_model()
    
    def load_base_model(self):
        """Load best model from WandB."""
        import wandb
        api = wandb.Api()
        run = api.run(f"honicky/cellxgene-mlp-v1/{self.wandb_run_id}")
        
        # Download checkpoint files (not artifacts)
        for file in run.files():
            if file.name.endswith('.pt'):
                file.download(root="./wandb_downloads", replace=True)
                checkpoint_path = f"./wandb_downloads/{file.name}"
                break
        
        # Load model configuration from run
        self.base_config = run.config
        return checkpoint_path
    
    def run_cross_validation(self, X, y, cell_types):
        """
        Execute 5-fold stratified cross-validation with WandB tracking.
        
        Returns:
            - Per-fold metrics
            - Aggregated mean/std metrics
            - Best performing fold model
        """
```

### 3. Model Adaptation Module
```python
class ModelAdapter:
    """Adapt pre-trained model for new cell type sets."""
    
    def create_adapted_model(self, base_model, num_target_classes, 
                            freeze_base=True, add_adapter=False):
        """
        Create task-specific model from pre-trained base.
        
        Args:
            base_model: Pre-trained MLPClassifier
            num_target_classes: Number of cell types in target dataset
            freeze_base: Whether to freeze pre-trained layers
            add_adapter: Add intermediate adaptation layer
        
        Returns:
            Adapted model ready for fine-tuning
        """
        model = copy.deepcopy(base_model)
        
        # Freeze all but output layer
        if freeze_base:
            for param in model.model[:-1].parameters():
                param.requires_grad = False
        
        # Replace output layer
        in_features = model.model[-1].in_features
        
        if add_adapter:
            # Optional: Add adapter layer
            adapter_dim = (in_features + num_target_classes) // 2
            new_output = nn.Sequential(
                nn.Linear(in_features, adapter_dim),
                nn.BatchNorm1d(adapter_dim),
                nn.ReLU(),
                nn.Dropout(model.dropout),
                nn.Linear(adapter_dim, num_target_classes)
            )
        else:
            new_output = nn.Linear(in_features, num_target_classes)
        
        model.model[-1] = new_output
        return model
```

### 4. Training Strategy
```python
class AdaptiveTrainer:
    """Fine-tuning trainer with early stopping and WandB tracking."""
    
    def __init__(self, config, wandb_project="cz-benchmark-genept"):
        self.config = config
        self.device = torch.device(config.device)
        self.wandb_project = wandb_project
    
    def train_fold(self, model, train_loader, val_loader, fold_idx, dataset_name):
        """
        Train single fold with validation-based early stopping.
        
        Hyperparameters (based on nwhis8xb):
        - Learning rate: 5e-4 to 1e-3 (lower than original 1.32e-3)
        - Epochs: 5-20 with early stopping
        - Batch size: 512-1024
        - Optimizer: Adam with weight decay
        """
        import wandb
        
        # Initialize WandB run for this fold
        run = wandb.init(
            project=self.wandb_project,
            name=f"{dataset_name}_fold_{fold_idx}",
            config={
                "base_model": "nwhis8xb",
                "dataset": dataset_name,
                "fold": fold_idx,
                "adaptation_lr": self.config.adaptation_lr,
                "batch_size": self.config.batch_size,
                "freeze_base": self.config.freeze_base_model,
                "use_adapter": self.config.use_adapter_layer
            },
            reinit=True
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.adaptation_lr,
            weight_decay=self.config.weight_decay
        )
        
        best_val_score = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_adaptation_epochs):
            # Training loop
            train_loss = self.train_epoch(model, train_loader, optimizer)
            
            # Validation
            val_metrics = self.evaluate(model, val_loader)
            
            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics['loss'],
                "val_accuracy": val_metrics['accuracy'],
                "val_f1_macro": val_metrics['f1_macro']
            })
            
            # Early stopping
            if val_metrics['loss'] < best_val_score:
                best_val_score = val_metrics['loss']
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                # Save best model as WandB artifact
                artifact = wandb.Artifact(
                    f"model_{dataset_name}_fold_{fold_idx}",
                    type="model",
                    metadata={"val_loss": best_val_score}
                )
                torch.save(best_model_state, "best_model.pt")
                artifact.add_file("best_model.pt")
                run.log_artifact(artifact)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break
        
        model.load_state_dict(best_model_state)
        run.finish()
        return model, best_val_score
```

### 5. Metrics Implementation
```python
class MetricsCalculator:
    """Calculate benchmark metrics matching CZ implementation."""
    
    def compute_metrics(self, y_true, y_pred, y_prob):
        """
        Compute all required metrics:
        - Accuracy
        - F1 Score (macro)
        - Precision (macro)
        - Recall (macro)
        - AUROC (if applicable)
        - Hierarchical F1 (using Cell Ontology)
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
        }
        
        # Add hierarchical metrics if ontology available
        if self.ontology_manager:
            metrics['hierarchical_f1'] = self.compute_hierarchical_f1(
                y_true, y_pred, cell_type_mapping
            )
        
        return metrics
```

## Hyperparameter Tuning Strategy

### Two-Stage Optimization

#### Stage 1: Quick Grid Search (Per Dataset)
Based on best model (nwhis8xb) hyperparameters:
- **Learning Rate**: [2.5e-4, 5e-4, 7.5e-4] (lower than base 1.32e-3)
- **Adapter Layer**: [True, False]
- **Freeze Strategy**: [all_but_output, last_two_layers]
- **Validation**: 20% holdout from training fold
- **Epochs**: 5 with early stopping
- **Batch Size**: 1024 (same as base model)

#### Stage 2: Fine-Tuning (Best Configuration)
- Use best configuration from Stage 1
- Train for more epochs (up to 20)
- Learning rate scheduling (cosine or step decay)
- More aggressive early stopping

### Hyperparameter Configuration
```python
@dataclass
class AdaptationConfig:
    # Base model reference
    base_wandb_run: str = "nwhis8xb"
    base_project: str = "honicky/cellxgene-mlp-v1"
    
    # Model adaptation
    freeze_base_model: bool = True
    use_adapter_layer: bool = False
    adapter_dropout: float = 0.053  # Match base model dropout
    
    # Training parameters (tuned from base)
    adaptation_lr: float = 5e-4  # ~40% of base LR
    weight_decay: float = 1e-5
    batch_size: int = 1024  # Match base model
    max_adaptation_epochs: int = 20
    patience: int = 3
    
    # Validation strategy
    validation_split: float = 0.2  # From training fold
    min_delta: float = 0.001  # Minimum improvement
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine"  # or "step"
    scheduler_params: dict = field(default_factory=lambda: {
        "T_max": 10,  # for cosine
        "eta_min": 1e-5
    })
    
    # WandB tracking
    wandb_project: str = "cz-benchmark-genept"
    wandb_entity: Optional[str] = None
```

## Training and Measurement Process

### 1. Dataset Preparation
```python
def prepare_benchmark_dataset(tissue_name):
    """Load and prepare Tabula Sapiens v2 benchmark dataset."""
    from pathlib import Path
    
    # Load pre-curated dataset
    benchmark_dir = Path("/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark")
    file_path = benchmark_dir / f"homo_sapiens_*_{tissue_name}_v2_curated.h5ad"
    adata = sc.read_h5ad(file_path)
    
    # Handle layers - use X_original if X is empty
    if adata.X is None or (hasattr(adata.X, 'nnz') and adata.X.nnz == 0):
        adata.X = adata.layers['X_original'].copy()
    
    # Load gene embeddings
    embeddings_df = pd.read_parquet(
        "data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet"
    )
    gene_embeddings = embeddings_df[['0', '1', ..., '499']].reset_index()  # First 500 dims
    
    # Align embeddings and create cell embeddings
    embedding_matrix, valid_indices = create_embedding_matrix(
        gene_embeddings, adata.var_names.tolist(), id_column='gene_name'
    )
    X = create_cell_embeddings(adata.X, embedding_matrix, valid_indices)
    X = X / 0.026  # Scale for OpenAI embeddings
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(adata.obs['cell_type'])
    
    return X, y, label_encoder.classes_.tolist()
```

### 2. Cross-Validation Execution
```python
def run_benchmark_evaluation(dataset_name, wandb_run_id="nwhis8xb", config=None):
    """Complete benchmark evaluation pipeline with WandB tracking."""
    import wandb
    
    # Initialize WandB sweep for full benchmark
    wandb.init(
        project="cz-benchmark-genept",
        name=f"benchmark_{dataset_name}",
        config={
            "base_model_run": wandb_run_id,
            "dataset": dataset_name,
            "n_folds": 5,
            "config": config.to_dict() if config else None
        }
    )
    
    # Load data
    X, y, cell_types = prepare_benchmark_dataset(dataset_name)
    
    # Load base model from WandB
    api = wandb.Api()
    base_run = api.run(f"honicky/cellxgene-mlp-v1/{wandb_run_id}")
    base_config = base_run.config
    
    # Download model checkpoint
    artifacts = base_run.logged_artifacts()
    checkpoint_artifact = [a for a in artifacts if 'checkpoint' in a.name][0]
    checkpoint_path = checkpoint_artifact.download()
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        
        # Create validation split from training data
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_full)),
            test_size=config.validation_split,
            stratify=y_train_full
        )
        
        X_train = X_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_train = y_train_full[train_idx]
        y_val = y_train_full[val_idx]
        
        # Load base model and adapt
        base_model = load_pretrained_model(checkpoint_path, base_config)
        adapted_model = adapt_model_for_dataset(
            base_model, 
            num_classes=len(cell_types),
            config=config
        )
        
        # Train on fold
        trained_model = train_adapted_model(
            adapted_model,
            X_train, y_train,
            X_val, y_val,
            config
        )
        
        # Evaluate on test fold
        test_metrics = evaluate_model(trained_model, X_test, y_test)
        fold_results.append(test_metrics)
    
    # Aggregate results
    final_metrics = aggregate_metrics(fold_results)
    
    # Log summary to WandB
    wandb.log({
        "final_accuracy_mean": final_metrics['accuracy']['mean'],
        "final_accuracy_std": final_metrics['accuracy']['std'],
        "final_f1_macro_mean": final_metrics['f1_macro']['mean'],
        "final_f1_macro_std": final_metrics['f1_macro']['std']
    })
    
    # Create summary table
    summary_table = wandb.Table(
        columns=["Fold", "Accuracy", "F1_Macro", "Precision", "Recall"],
        data=[[i, r['accuracy'], r['f1_macro'], r['precision'], r['recall']] 
              for i, r in enumerate(fold_results)]
    )
    wandb.log({"fold_results": summary_table})
    
    wandb.finish()
    
    return final_metrics
```

### 3. Results Reporting
```python
def generate_benchmark_report(results, dataset_name):
    """Generate comprehensive benchmark report."""
    
    report = {
        'dataset': dataset_name,
        'method': 'GenePT-MLP-Adapted',
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'accuracy': {
                'mean': np.mean([r['accuracy'] for r in results]),
                'std': np.std([r['accuracy'] for r in results]),
                'per_fold': [r['accuracy'] for r in results]
            },
            'f1_macro': {
                'mean': np.mean([r['f1_macro'] for r in results]),
                'std': np.std([r['f1_macro'] for r in results]),
                'per_fold': [r['f1_macro'] for r in results]
            },
            # ... other metrics
        },
        'comparison': {
            'vs_logistic_regression': None,  # To be filled
            'vs_knn': None,  # To be filled
            'vs_random_forest': None  # To be filled
        }
    }
    
    return report
```

## Integration with Existing Code

### Reusing CZ Benchmarks Components
- **Metrics Calculation**: Adapt `_compute_metrics()` from label_prediction.py
- **Cross-Validation**: Follow same StratifiedKFold setup
- **Data Filtering**: Use same minimum cell type thresholds

### Reusing GenePT Components
- **Model Architecture**: Use existing MLPClassifier
- **Training Infrastructure**: Leverage existing Trainer class
- **Metrics**: Use existing hierarchical F1 implementation
- **Data Loading**: Adapt existing PTFileStreamDataset for benchmark data

## Validation Strategy

### 1. Sanity Checks
- Verify model adaptation preserves base performance on original classes
- Ensure frozen layers remain unchanged during training
- Check gradient flow only through unfrozen layers

### 2. Baseline Comparison
- Run traditional classifiers (LR, KNN, RF) on same embeddings
- Compare with published CZ benchmark results
- Measure improvement from pre-training vs random initialization

### 3. Ablation Studies
- Effect of adapter layer vs direct output replacement
- Impact of different freezing strategies
- Benefit of pre-training vs training from scratch

## Expected Outcomes

### Performance Targets
- **Training Time**: <5 minutes per fold on GPU

### Comparison Goals
- Outperform standard classifiers on GenePT embeddings
- Competitive with or better than CZ benchmark baselines
- Demonstrate transfer learning benefits

## Implementation Example

### Loading and Using the Best Model
```python
import wandb
import torch
from src.models.mlp_classifier import MLPClassifier

def load_best_genept_model():
    """Load the best pre-trained GenePT model from WandB."""
    
    # Initialize WandB API
    api = wandb.Api()
    run = api.run("honicky/cellxgene-mlp-v1/nwhis8xb")
    
    # Get configuration
    config = run.config
    
    # Download checkpoint - find the best checkpoint artifact
    artifacts = run.logged_artifacts()
    # Filter for checkpoint artifacts (e.g., mlp_checkpoint_6000.pt)
    checkpoint_artifacts = [a for a in artifacts if 'checkpoint' in a.name]
    
    # Download the artifact
    artifact = checkpoint_artifacts[-1]  # Get latest/best
    artifact_dir = artifact.download()
    
    # Find the .pt file in the downloaded directory
    import os
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    
    # Initialize model with original configuration
    model = MLPClassifier(
        input_dim=config.get('input_dim', 500),  # Handle flat config
        num_classes=config.get('num_classes', 377),
        n_hidden_layers=config.get('n_hidden_layers', 3),
        dropout=config.get('dropout', 0.053)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # Handle state dict prefix mismatch
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Add "model." prefix if needed
    if not list(state_dict.keys())[0].startswith("model."):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    return model, config

# Usage in benchmark
def run_cz_benchmark():
    # Load base model
    base_model, base_config = load_best_genept_model()
    
    # Initialize WandB for benchmark tracking
    wandb.init(
        project="cz-benchmark-genept",
        config={
            "base_model": "nwhis8xb",
            "base_config": base_config
        }
    )
    
    # Rest of benchmark implementation...
```

## Dependencies and Requirements

### Python Packages
- `torch >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `scanpy >= 1.9.0`
- `anndata >= 0.10.0`
- `optuna >= 3.0.0` (for hyperparameter tuning)
- `wandb >= 0.15.0` (for experiment tracking)
- Existing GenePT dependencies

### Data Requirements
- **Benchmark Datasets**: Tabula Sapiens v2 Curated Benchmark
  - Location: `/Users/rj/personal/Tabula_Sapiens_v2_Curated_Benchmark/`
  - Tissues evaluated: Blood, Bone_Marrow, Lung, Mammary, Thymus
  - Format: H5AD files with layers (X_original contains raw counts)
  - Pre-curated: No additional filtering needed
- **Gene Embeddings**: `data/huggingface_model/embedding_associations_cell_type_tissue_drug_pathway_openai_large.parquet`
  - 33,703 genes × 3,072 dimensions (use first 500 for model)
  - Gene names in index
- **Pre-trained Model**: WandB run nwhis8xb
  - Downloaded as checkpoint files (not artifacts)
  - State dict keys may need "model." prefix
- Cell Ontology for hierarchical metrics (optional)

### Compute Resources
- GPU recommended for faster training
- ~16GB RAM for loading embeddings
- ~10GB disk space for checkpoints
- Internet connection for WandB tracking and model downloads
