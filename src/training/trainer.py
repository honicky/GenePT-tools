"""Trainer class for CellXGene MLP model."""

import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.mlp_classifier import MLPClassifier
from ..data_loading.s3_dataset import S3ParquetStreamDataset
from ..data_loading.pt_dataset import PTFileStreamDataset
from ..data_loading.composable_dataset import ComposableTrainingDataset
from ..utils.checkpoint import CheckpointManager, load_checkpoint, save_best_model, save_checkpoint
from .metrics import evaluate, evaluate_with_hierarchy
from .config import TrainingConfig
from .ontology import CellOntologyManager

# Optional wandb import
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  wandb = None
  WANDB_AVAILABLE = False


class MLPTrainer:
  """Main training orchestrator for CellXGene MLP model.
  
  Designed to be usable both standalone and within hyperparameter optimization.
  """
  
  def __init__(
      self,
      config: TrainingConfig,
      cell_types: list,
      cell_type_codes: pd.Series,
      code_remapping: dict = None,
      mapping_df: pd.DataFrame = None,
      trial=None  # Optional Optuna trial
  ):
    """Initialize the trainer.

    Args:
      config: Training configuration
      cell_types: List of all cell types
      cell_type_codes: Series mapping cell types to codes
      code_remapping: Optional dict mapping original codes to filtered codes
      mapping_df: Optional DataFrame with cell type mapping for WandB
      trial: Optional Optuna trial for hyperparameter optimization
    """
    self.config = config
    self.cell_types = cell_types
    self.cell_type_codes = cell_type_codes
    self.code_remapping = code_remapping
    self.mapping_df = mapping_df
    self.trial = trial
    
    # Set random seeds for reproducibility
    if config.seed is not None:
      random.seed(config.seed)
      np.random.seed(config.seed)
      torch.manual_seed(config.seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
      # For additional reproducibility (may impact performance)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    
    # Set device
    self.device = torch.device(config.device)
    print(f"Using device: {self.device}")
    
    # Initialize model
    self.num_classes = len(cell_type_codes)
    self.model = self.create_model()
    
    # Initialize optimizer
    self.optimizer = self.create_optimizer(self.model)
    
    # Initialize loss function
    self.criterion = nn.CrossEntropyLoss()
    
    # Initialize checkpoint manager
    self.checkpoint_manager = CheckpointManager(
      checkpoint_dir=config.checkpoint_dir,
      save_every_n_batches=config.checkpoint_every_n_batches,
      keep_last_n=5,
      track_metric='val_logloss',
      metric_mode='min'
    )
    
    # Training state
    self.start_epoch = 0
    self.start_batch = 0
    self.global_step = 0
    
    # Track validation thresholds - initialize to the first interval
    # (not 0, which would trigger immediately on step 1)
    self.next_eval_5k_step = config.eval_every_n_batches
    self.next_eval_full_step = config.eval_full_every_n_batches
    
    # Load checkpoint if resuming
    if config.resume_from is not None:
      self.load_checkpoint(config.resume_from)
    
    # Load validation data
    self.X_val_5k = None
    self.y_val_5k = None
    self.X_val_120k = None
    self.y_val_120k = None
    if self.config.load_validation_data:
      self.load_validation_data()
    
    # Initialize W&B if configured
    self.wandb_run = None
    if config.wandb_project is not None:
      self.init_wandb()
    
    # Initialize ontology for hierarchical metrics
    self.ontology_graph = None
    if config.enable_hierarchical_metrics:
      try:
        print("Loading Cell Ontology for hierarchical metrics...")
        ontology_manager = CellOntologyManager(config.ontology_cache_dir)
        self.ontology_graph = ontology_manager.build_cell_type_graph()
        print(f"Loaded ontology with {len(self.ontology_graph.nodes)} cell types")
      except Exception as e:
        print(f"Warning: Could not load Cell Ontology: {e}")
        print("Continuing without hierarchical metrics")
  
  def create_model(self) -> nn.Module:
    """Create the MLP model."""
    model = MLPClassifier(
      input_dim=self.config.n_dims,
      num_classes=self.num_classes,
      n_hidden_layers=self.config.n_hidden_layers,
      dropout=self.config.dropout
    )
    model = model.to(self.device)
    
    print(f"Created model with {model.count_parameters():,} parameters")
    print(f"Hidden dimensions: {model.get_hidden_dims()}")
    
    return model
  
  def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
    """Create the optimizer with differential weight decay."""
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue
      
      # Don't apply weight decay to biases and normalization parameters
      if name.endswith("bias") or "norm" in name.lower() or "bn" in name.lower():
        no_decay_params.append(param)
      else:
        decay_params.append(param)
    
    param_groups = [
      {"params": decay_params, "weight_decay": self.config.weight_decay},
      {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Create optimizer based on config
    if self.config.optimizer_type == "adam":
      optimizer = optim.Adam(param_groups, lr=self.config.learning_rate)
    elif self.config.optimizer_type == "adamw":
      optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)
    elif self.config.optimizer_type == "sgd":
      optimizer = optim.SGD(param_groups, lr=self.config.learning_rate, momentum=0.9)
    else:
      raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    return optimizer
  
  def load_validation_data(self):
    """Load validation datasets from directory, supporting both .parquet and .pt formats."""
    print("Loading validation data...")
    
    if self.config.test_data_dir is None:
      print("No test_data_dir specified, skipping validation data loading")
      return
    
    test_dir = Path(self.config.test_data_dir)
    if not test_dir.exists():
      print(f"Test directory {test_dir} does not exist")
      return
    
    # Check for .pt files first (faster format)
    pt_files = sorted(test_dir.glob("*.pt"))
    # Filter out metadata file
    pt_files = [f for f in pt_files if f.name != "metadata.pt"]
    
    if pt_files:
      print(f"Loading validation data from {len(pt_files)} .pt files")
      X_list = []
      y_list = []
      
      for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=True)
        # Slice to configured dimensions if needed
        X = data['X']
        if self.config.n_dims is not None and X.shape[1] > self.config.n_dims:
          X = X[:, :self.config.n_dims]
        # Scale embeddings to match training data scaling
        X = X / 0.026  # Same scaling as training data
        X_list.append(X.numpy())
        y_list.append(data['y'].numpy())
      
      # Combine all files
      X_combined = np.concatenate(X_list, axis=0).astype(np.float32)
      y_combined = np.concatenate(y_list, axis=0).astype(np.int64)
      
      # Filter to valid cell types (y values should already be in range 0 to num_classes-1)
      valid_mask = (y_combined >= 0) & (y_combined < self.num_classes)
      self.X_val_120k = X_combined[valid_mask]
      self.y_val_120k = y_combined[valid_mask]
      
    else:
      # Fall back to loading parquet files
      parquet_files = sorted(test_dir.glob("*.parquet"))
      
      if not parquet_files:
        print(f"No .pt or .parquet files found in {test_dir}")
        return
      
      print(f"Loading validation data from {len(parquet_files)} .parquet files")
      dfs = []
      for parquet_file in parquet_files:
        dfs.append(pd.read_parquet(parquet_file))
      
      if dfs:
        df = pd.concat(dfs, ignore_index=True)
        self.X_val_120k, self.y_val_120k = self._process_validation_df(df)
    
    if self.X_val_120k is not None:
      print(f"Loaded validation set: {self.X_val_120k.shape}")
      
      # Create 5k subset
      if len(self.X_val_120k) > 5000:
        indices = np.random.choice(len(self.X_val_120k), 5000, replace=False)
        self.X_val_5k = self.X_val_120k[indices]
        self.y_val_5k = self.y_val_120k[indices]
        print(f"Created 5k validation subset")
      else:
        self.X_val_5k = self.X_val_120k
        self.y_val_5k = self.y_val_120k
  
  def _process_validation_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Process validation dataframe to extract features and labels."""
    # Extract embedding columns
    embedding_cols = [str(i) for i in range(self.config.n_dims)]
    
    # Filter to columns that exist
    available_cols = [col for col in embedding_cols if col in df.columns]
    if len(available_cols) < self.config.n_dims:
      print(f"Warning: Only {len(available_cols)} embedding dimensions available")
    
    X = df[available_cols].values.astype(np.float32)
    
    # Scale embeddings to match training data scaling
    X = X / 0.026  # Same scaling as training data
    
    # Get the subset of cell types we're training on
    # cell_types is the list of cell type names we're training on
    # cell_type_codes is a Series with cell type names as index and their codes as values
    training_cell_types = self.cell_types
    
    # Encode labels using only the training cell types
    y = df["cell_type"].astype(
      pd.CategoricalDtype(categories=training_cell_types)
    ).cat.codes
    
    # Filter out samples with unknown cell types (-1)
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask].values.astype(np.int64)
    
    # The encoded values are already correct (0 to num_classes-1)
    # since we encoded using only the training cell types
    
    return X, y
  
  def train_batch(self, X: torch.Tensor, y: torch.Tensor) -> float:
    """Train on a single batch.
    
    Args:
      X: Input features
      y: Target labels
      
    Returns:
      Loss value
    """
    self.model.train()
    
    # Move to device
    X = X.to(self.device)
    y = y.to(self.device)
    
    # Forward pass
    self.optimizer.zero_grad()
    outputs = self.model(X)
    loss = self.criterion(outputs, y)
    
    # Backward pass
    loss.backward()
    self.optimizer.step()
    
    return loss.item()
  
  def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
      dataloader: Training data loader
      epoch: Current epoch number
      
    Returns:
      Dictionary of training metrics
    """
    self.model.train()
    
    epoch_losses = []
    batch_times = []
    
    # Progress bar with validation metrics tracking
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    val_postfix = {}  # Store latest validation metrics for display
    
    # Get expected number of batches (this is an estimate)
    expected_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
    
    for batch_idx, (X, y) in enumerate(pbar):
      # Check if we're near the end of the epoch (last 10% of batches)
      # and suppress the IterableDataset length warning for those
      if expected_batches and batch_idx > expected_batches * 0.9:
        warnings.filterwarnings('ignore', message='.*Length of IterableDataset.*was reported to be.*')
      else:
        # Re-enable the warning for normal batches
        warnings.filterwarnings('default', message='.*Length of IterableDataset.*was reported to be.*')
      batch_start = time.time()
      
      # Train on batch
      loss = self.train_batch(X, y)
      epoch_losses.append(loss)
      
      batch_time = time.time() - batch_start
      batch_times.append(batch_time)
      
      # Update progress bar with loss and any validation metrics
      postfix_dict = {'loss': f'{loss:.4f}'}
      postfix_dict.update(val_postfix)
      pbar.set_postfix(postfix_dict)
      
      # Report to Optuna trial if present (before incrementing step)
      if self.trial is not None:
        self.trial.report(loss, self.global_step)
      
      # Increment step counter before any logging (to ensure correct WandB step ordering)
      self.global_step += 1
      
      # Log to W&B with the incremented step (don't commit yet)
      if self.wandb_run is not None:
        self.wandb_run.log({
          'train/loss': loss,
          'train/batch_time': batch_time,
          'epoch': epoch,
          'global_step': self.global_step
        }, step=self.global_step, commit=False)
      
      # Periodic evaluation on 5k validation set (threshold-based)
      if self.global_step >= self.next_eval_5k_step and self.X_val_5k is not None:
        val_metrics = self.evaluate_validation(use_5k=True)
        if val_metrics and 'logloss' in val_metrics:
          # Update progress bar postfix with validation metrics
          val_postfix['val_loss'] = f"{val_metrics['logloss']:.3f}"
          val_postfix['val_r@10'] = f"{val_metrics['recall_at_10']:.3f}"
          if 'hierarchical_f1' in val_metrics:
            val_postfix['val_hF1'] = f"{val_metrics['hierarchical_f1']:.3f}"
        # Set next threshold
        self.next_eval_5k_step = self.global_step + self.config.eval_every_n_batches
      
      # Less frequent evaluation on full validation set (threshold-based)
      if self.global_step >= self.next_eval_full_step and self.X_val_120k is not None:
        # Debug: Check if we're at the exact threshold
        if self.global_step == self.next_eval_full_step and self.config.verbose:
          print(f"[DEBUG] Running full validation at step {self.global_step} (threshold was {self.next_eval_full_step})")
        val_metrics = self.evaluate_validation(use_5k=False)
        if val_metrics and 'logloss' in val_metrics:
          # Update with full validation metrics (overwrite the 5k metrics)
          val_postfix['val_loss'] = f"{val_metrics['logloss']:.3f}"
          val_postfix['val_r@10'] = f"{val_metrics['recall_at_10']:.3f}"
          val_postfix['val_mrr'] = f"{val_metrics['mrr_at_10']:.3f}"
          if 'hierarchical_f1' in val_metrics:
            val_postfix['val_hF1'] = f"{val_metrics['hierarchical_f1']:.3f}"
          # Add a marker to show this is full validation
          val_postfix['full'] = "✓"
        # Set next threshold
        self.next_eval_full_step = self.global_step + self.config.eval_full_every_n_batches
      
      # Always commit at the end of each step (after training and any validations)
      if self.wandb_run is not None:
        # Commit with an empty log to finalize this step
        self.wandb_run.log({}, step=self.global_step, commit=True)
      
      # Check if we've reached max steps per epoch (for quick testing)
      if (self.config.max_steps_per_epoch is not None and 
          batch_idx + 1 >= self.config.max_steps_per_epoch):
        # Early stop message will be visible in the progress bar description
        pbar.set_description(f"Epoch {epoch} (stopped at {self.config.max_steps_per_epoch} steps)")
        break
      
      # Checkpoint after incrementing (so we don't checkpoint at step 0)
      if self.checkpoint_manager.should_save(self.global_step):
        checkpoint_path = None
        
        # Save local checkpoint (only if enabled)
        if self.config.local_checkpoints:
          checkpoint_path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
            config=self.config
          )
        
        # Save checkpoint as WandB artifact (regardless of local checkpoint setting)
        if self.config.wandb_save_artifacts and self.wandb_run:
          # If no local checkpoint was saved, create a temporary one for WandB
          if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_dir / f"temp_checkpoint_step_{self.global_step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
              checkpoint_path=checkpoint_path,
              model=self.model,
              optimizer=self.optimizer,
              epoch=epoch,
              batch_idx=batch_idx,
              global_step=self.global_step,
              config=self.config
            )
          
          self.save_artifact(
            file_path=checkpoint_path,
            artifact_name=f"checkpoint_step_{self.global_step}",
            artifact_type="checkpoint",
            description=f"Training checkpoint at step {self.global_step} (epoch {epoch}, batch {batch_idx})"
          )
          
          # Clean up temporary checkpoint if it was created just for WandB
          if not self.config.local_checkpoints and checkpoint_path.name.startswith("temp_"):
            checkpoint_path.unlink(missing_ok=True)
    
    # Reset warning filter to default for next epoch
    warnings.filterwarnings('default', message='.*Length of IterableDataset.*was reported to be.*')
    
    # Return epoch metrics
    return {
      'epoch': epoch,
      'avg_loss': np.mean(epoch_losses),
      'avg_batch_time': np.mean(batch_times)
    }
  
  def evaluate_validation(self, use_5k: bool = True) -> Dict[str, float]:
    """Evaluate on validation set.
    
    Args:
      use_5k: Whether to use 5k subset or full validation set
      
    Returns:
      Dictionary of validation metrics
    """
    if use_5k and self.X_val_5k is not None:
      X, y = self.X_val_5k, self.y_val_5k
      prefix = "val5k"
    elif self.X_val_120k is not None:
      X, y = self.X_val_120k, self.y_val_120k
      prefix = "val120k"
    else:
      return {}
    
    # Evaluate with hierarchical metrics if available
    if self.ontology_graph is not None:
      # Create cell type to index mapping
      cell_type_to_idx = {ct: i for i, ct in enumerate(self.cell_types[:self.num_classes])}
      
      metrics = evaluate_with_hierarchy(
        model=self.model,
        X=X,
        y=y,
        cell_types=self.cell_types[:self.num_classes],
        cell_type_to_idx=cell_type_to_idx,
        ontology_graph=self.ontology_graph,
        batch_size=self.config.batch_size,
        device=self.device
      )
    else:
      # Standard evaluation
      metrics = evaluate(
        model=self.model,
        X=X,
        y=y,
        num_classes=self.num_classes,
        batch_size=self.config.batch_size,
        device=self.device
      )
    
    # Add prefix to metrics
    prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    # Log to W&B
    if self.wandb_run is not None:
      # Store the step we're logging to (in case global_step changes)
      log_step = self.global_step
      # Don't commit here - let the training loop handle commits
      self.wandb_run.log(prefixed_metrics, step=log_step, commit=False)
    
    # Update best metric for checkpointing (unless explicitly disabled with "none")
    if (prefix == "val120k" and 
        self.config.best_model_metric != "none" and 
        self.config.best_model_metric in metrics):
      metric_key = f"val_{self.config.best_model_metric}"
      
      # Save locally only if local checkpoints are enabled
      new_best_value = None
      if self.config.local_checkpoints:
        new_best_value = save_best_model(
          model=self.model,
          metrics={metric_key: metrics[self.config.best_model_metric]},
          metric_name=metric_key,
          checkpoint_dir=self.config.checkpoint_dir,
          mode=self.config.best_model_mode,
          current_best=self.checkpoint_manager.best_metric
        )
      else:
        # Just check if this would be a new best without saving locally
        current_value = metrics[self.config.best_model_metric]
        current_best = self.checkpoint_manager.best_metric
        
        is_best = False
        if current_best is None:
          is_best = True
        elif self.config.best_model_mode == 'min' and current_value < current_best:
          is_best = True
        elif self.config.best_model_mode == 'max' and current_value > current_best:
          is_best = True
          
        new_best_value = current_value if is_best else current_best
      
      # Save as WandB artifact if a new best model was identified (regardless of local saving)
      if (new_best_value is not None and new_best_value != self.checkpoint_manager.best_metric and
          self.config.wandb_save_artifacts and self.wandb_run):
        current_metric_value = metrics[self.config.best_model_metric]
        
        # If local checkpoints are disabled, we need to create temporary files for WandB
        if not self.config.local_checkpoints:
          best_model_path = self.config.checkpoint_dir / 'temp_best_model.pt'
          best_model_path.parent.mkdir(parents=True, exist_ok=True)
          torch.save(self.model.state_dict(), best_model_path)
          
          metrics_path = self.config.checkpoint_dir / 'temp_best_model_metrics.json'
          with open(metrics_path, 'w') as f:
            import json
            json.dump({metric_key: current_metric_value}, f, indent=2)
        else:
          # Use the locally saved files
          best_model_path = self.config.checkpoint_dir / 'best_model.pt'
          metrics_path = self.config.checkpoint_dir / 'best_model_metrics.json'
        
        # Save model artifact
        self.save_artifact(
          file_path=best_model_path,
          artifact_name="best_model",
          artifact_type="model",
          description=f"Best model with {metric_key}={current_metric_value:.4f}"
        )
        
        # Save metrics artifact
        if metrics_path.exists():
          self.save_artifact(
            file_path=metrics_path,
            artifact_name="best_model_metrics",
            artifact_type="metrics",
            description=f"Metrics for best model ({metric_key}={current_metric_value:.4f})"
          )
        
        # Clean up temporary files if they were created just for WandB
        if not self.config.local_checkpoints:
          best_model_path.unlink(missing_ok=True)
          metrics_path.unlink(missing_ok=True)
      
      self.checkpoint_manager.best_metric = new_best_value
    
    return metrics
  
  def load_checkpoint(self, checkpoint_path: Path):
    """Load from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = load_checkpoint(
      checkpoint_path=checkpoint_path,
      model=self.model,
      optimizer=self.optimizer,
      device=self.device
    )
    
    self.start_epoch = checkpoint.get('epoch', 0)
    self.start_batch = checkpoint.get('batch_idx', 0)
    self.global_step = checkpoint.get('global_step', 0)
    
    print(f"Resumed from epoch {self.start_epoch}, batch {self.start_batch}")
  
  def init_wandb(self):
    """Initialize Weights & Biases logging."""
    if not WANDB_AVAILABLE:
      print("Warning: wandb not installed, skipping W&B logging")
      self.wandb_run = None
      return
    
    self.wandb_run = wandb.init(
      project=self.config.wandb_project,
      entity=self.config.wandb_entity,
      name=self.config.wandb_run_name,
      tags=self.config.wandb_tags,
      config=self.config.to_dict(),
      reinit=True
    )

    # Log cell type mapping if filtering is enabled
    if self.mapping_df is not None:
      self._log_cell_type_mapping()

    # Watch model (reduced frequency to avoid overwhelming wandb)
    wandb.watch(self.model, log='gradients', log_freq=200)

    print(f"Initialized W&B run: {self.wandb_run.name}")

  def _log_cell_type_mapping(self):
    """Log cell type mapping to WandB as config metadata."""
    if self.wandb_run is None or self.mapping_df is None:
      return

    # 1. Log as WandB Table (viewable in UI)
    mapping_table = wandb.Table(dataframe=self.mapping_df)
    wandb.log({"cell_type_mapping": mapping_table})

    # 2. Save as CSV artifact for reproducibility
    artifact = wandb.Artifact(
      name=f"cell_type_mapping_{wandb.run.id}",
      type="dataset_metadata",
      description=f"Filtered cell type mapping (threshold={self.config.cell_count_threshold})"
    )

    mapping_file = self.config.checkpoint_dir / "cell_type_mapping.csv"
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    self.mapping_df.to_csv(mapping_file, index=False)
    artifact.add_file(str(mapping_file))
    wandb.log_artifact(artifact)

    # 3. Log summary statistics to wandb.config
    wandb.config.update({
      "cell_count_threshold": self.config.cell_count_threshold,
      "n_cell_types_filtered": len(self.mapping_df),
      "sample_coverage_pct": 100 * self.mapping_df['cell_count'].sum() / self.mapping_df['cell_count'].sum()  # Will be calculated properly in actual use
    }, allow_val_change=True)

    # 4. Log top 10 cell types as summary
    top_10 = self.mapping_df.head(10)[['filtered_code', 'cell_type', 'cell_count']].to_dict('records')
    wandb.summary['top_10_cell_types'] = top_10

    if self.config.verbose:
      print(f"Logged cell type mapping to WandB:")
      print(f"  - Table: cell_type_mapping ({len(self.mapping_df)} rows)")
      print(f"  - Artifact: cell_type_mapping_{wandb.run.id}")
      print(f"  - Config: threshold={self.config.cell_count_threshold}, filtered={len(self.mapping_df)}")

  def save_artifact(self, file_path: Path, artifact_name: str, artifact_type: str, description: str = ""):
    """Save a file as WandB artifact.
    
    Args:
      file_path: Path to file to save as artifact
      artifact_name: Name for the artifact  
      artifact_type: Type of artifact (e.g. 'model', 'checkpoint')
      description: Optional description
    """
    if self.wandb_run is None or not self.config.wandb_save_artifacts:
      return
    
    if not file_path.exists():
      print(f"Warning: File {file_path} does not exist, skipping artifact save")
      return
    
    try:
      artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description
      )
      artifact.add_file(str(file_path))
      self.wandb_run.log_artifact(artifact)
      print(f"Saved {artifact_type} artifact: {artifact_name}")
    except Exception as e:
      print(f"Warning: Failed to save artifact {artifact_name}: {e}")
  
  def run(self) -> Dict[str, float]:
    """Run the full training loop.
    
    Returns:
      Final validation metrics for hyperparameter optimization
    """
    print(f"Starting training for {self.config.epochs} epochs")
    print(f"[DEBUG] Config batch_size: {self.config.batch_size}")
    print(f"[DEBUG] Config learning_rate: {self.config.learning_rate}")
    
    # Create dataset - auto-detect format based on directory contents
    local_data_path = Path(self.config.local_data_dir) if self.config.local_data_dir else None
    
    # Check if directory contains .pt files (fast format)
    if local_data_path and local_data_path.exists():
      print(f"[DEBUG] Local data path exists: {local_data_path}")
      pt_files = list(local_data_path.glob("*.pt"))
      print(f"[DEBUG] Found {len(pt_files)} .pt files")
      if pt_files:
        print(f"[DEBUG] First few .pt files: {[f.name for f in pt_files[:5]]}")
    else:
      if local_data_path:
        print(f"[DEBUG] Local data path does not exist: {local_data_path}")
        # In AWS Batch, if the path doesn't exist, there might be a mount issue
        # Let's check what directories do exist
        parent = local_data_path.parent
        if parent.exists():
          print(f"[DEBUG] Parent directory exists: {parent}")
          print(f"[DEBUG] Contents: {list(parent.iterdir())[:10]}")
      else:
        print(f"[DEBUG] Local data path is None")
      pt_files = []
    
    # Check if using composable dataset (new system)
    if self.config.use_composable_dataset:
      # Use composable embedding dataset
      if self.config.verbose:
        print(f"Using composable embedding dataset")
        print(f"  Base dir: {self.config.base_data_dir}")
        print(f"  Embedding types: {self.config.embedding_types}")
        print(f"  GenePT dims: {self.config.genept_dims}")

      dataset = ComposableTrainingDataset(
        base_dir=self.config.base_data_dir,
        embedding_types=self.config.embedding_types,
        batch_size=self.config.batch_size,
        start_batch_file=self.config.start_batch_file,
        end_batch_file=self.config.end_batch_file,
        genept_dims=self.config.genept_dims,
        code_remapping=self.code_remapping,
        track_invalid_embeddings=self.config.track_invalid_embeddings,
        shuffle_files_per_epoch=self.config.shuffle_files_per_epoch,
        shuffle_within_files=self.config.shuffle_within_files,
        seed=self.config.seed,
        verbose=self.config.verbose
      )
    elif pt_files and any(f.name.startswith("batch_") for f in pt_files):
      # Use fast PT dataset
      if self.config.verbose:
        print(f"Using fast PT dataset from {local_data_path}")
      dataset = PTFileStreamDataset(
        data_dir=local_data_path,
        batch_size=self.config.batch_size,
        n_dims=self.config.n_dims,
        shuffle_files_per_epoch=self.config.shuffle_files_per_epoch,
        shuffle_within_files=self.config.shuffle_within_files,
        seed=self.config.seed,
        verbose=self.config.verbose
      )
    else:
      # Use original S3/Parquet dataset
      if self.config.verbose:
        print(f"Using S3/Parquet dataset")
      dataset = S3ParquetStreamDataset(
        cell_types=self.cell_types,
        cell_type_codes=self.cell_type_codes,
        s3_bucket=self.config.s3_bucket,
        s3_prefix=self.config.s3_prefix,
        local_data_dir=self.config.local_data_dir,
        n_dims=self.config.n_dims,
        batch_size=self.config.batch_size,
        download_if_missing=self.config.download_if_missing,
        shuffle_files_per_epoch=self.config.shuffle_files_per_epoch,
        shuffle_within_files=self.config.shuffle_within_files,
        aws_profile=self.config.aws_profile,
        start_batch_file=self.config.start_batch_file,
        end_batch_file=self.config.end_batch_file,
        seed=self.config.seed,
        verbose=self.config.verbose
      )
    
    # Create dataloader
    dataloader = DataLoader(
      dataset,
      batch_size=None,  # Dataset returns batches
      num_workers=self.config.num_workers,
      pin_memory=True if self.device.type == 'cuda' else False,
      prefetch_factor=2 if self.config.num_workers > 0 else None,
      persistent_workers=True if self.config.num_workers > 0 else False
    )
    
    # Training loop
    for epoch in range(self.start_epoch, self.config.epochs):
      self.train_epoch(dataloader, epoch)
      
      # Epoch completion is already shown in the progress bar
      # Validation already happens during the epoch via threshold triggers
      # No need for additional end-of-epoch validation which causes step ordering issues
    
    # Final validation at end of all training (since we may not end on a validation boundary)
    final_metrics = {}
    if self.X_val_120k is not None:
      # Increment step for final validation to avoid conflicts
      self.global_step += 1
      final_metrics = self.evaluate_validation(use_5k=False)
      # Commit the final validation metrics since we're done training
      if self.wandb_run is not None:
        self.wandb_run.log({}, step=self.global_step, commit=True)
    
    final_checkpoint_path = None
    
    # Save final checkpoint locally (only if enabled)
    if self.config.local_checkpoints:
      final_checkpoint_path = self.checkpoint_manager.save_final(
        model=self.model,
        optimizer=self.optimizer,
        epoch=self.config.epochs,
        batch_idx=0,
        global_step=self.global_step,
        metrics=final_metrics,
        config=self.config
      )
    
    # Save final checkpoint as WandB artifact (regardless of local setting)
    if self.config.wandb_save_artifacts and self.wandb_run:
      # If no local checkpoint was saved, create a temporary one for WandB
      if final_checkpoint_path is None:
        final_checkpoint_path = self.config.checkpoint_dir / "temp_final_checkpoint.pt"
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
          checkpoint_path=final_checkpoint_path,
          model=self.model,
          optimizer=self.optimizer,
          epoch=self.config.epochs,
          batch_idx=0,
          global_step=self.global_step,
          best_metrics={'best_' + self.checkpoint_manager.track_metric: self.checkpoint_manager.best_metric} if self.checkpoint_manager.best_metric else None,
          final_metrics=final_metrics,
          config=self.config
        )
      
      self.save_artifact(
        file_path=final_checkpoint_path,
        artifact_name="final_checkpoint",
        artifact_type="checkpoint",
        description=f"Final training checkpoint after {self.config.epochs} epochs ({self.global_step} steps)"
      )
      
      # Clean up temporary checkpoint if it was created just for WandB
      if not self.config.local_checkpoints and final_checkpoint_path.name.startswith("temp_"):
        final_checkpoint_path.unlink(missing_ok=True)
    
    # Close W&B run
    if self.wandb_run is not None:
      self.wandb_run.finish()
    
    return final_metrics