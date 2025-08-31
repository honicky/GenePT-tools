"""Trainer class for CellXGene MLP model."""

import time
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
from ..utils.checkpoint import CheckpointManager, load_checkpoint
from .metrics import evaluate
from .config import TrainingConfig


class MLPTrainer:
  """Main training orchestrator for CellXGene MLP model.
  
  Designed to be usable both standalone and within hyperparameter optimization.
  """
  
  def __init__(
      self,
      config: TrainingConfig,
      cell_types: list,
      cell_type_codes: pd.Series,
      trial=None  # Optional Optuna trial
  ):
    """Initialize the trainer.
    
    Args:
      config: Training configuration
      cell_types: List of all cell types
      cell_type_codes: Series mapping cell types to codes
      trial: Optional Optuna trial for hyperparameter optimization
    """
    self.config = config
    self.cell_types = cell_types
    self.cell_type_codes = cell_type_codes
    self.trial = trial
    
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
    
    # Load checkpoint if resuming
    if config.resume_from is not None:
      self.load_checkpoint(config.resume_from)
    
    # Load validation data
    self.X_val_5k = None
    self.y_val_5k = None
    self.X_val_120k = None
    self.y_val_120k = None
    self.load_validation_data()
    
    # Initialize W&B if configured
    self.wandb_run = None
    if config.wandb_project is not None:
      self.init_wandb()
  
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
    
    optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)
    return optimizer
  
  def load_validation_data(self):
    """Load validation datasets."""
    print("Loading validation data...")
    
    # Try to load from test data directory if specified
    if self.config.test_data_dir is not None:
      test_dir = Path(self.config.test_data_dir)
      
      # Load 5k validation set
      val_5k_path = test_dir / "val_5k.parquet"
      if val_5k_path.exists():
        df = pd.read_parquet(val_5k_path)
        self.X_val_5k, self.y_val_5k = self._process_validation_df(df)
        print(f"Loaded 5k validation set: {self.X_val_5k.shape}")
      
      # Load 120k validation set
      val_120k_path = test_dir / "val_120k.parquet"
      if val_120k_path.exists():
        df = pd.read_parquet(val_120k_path)
        self.X_val_120k, self.y_val_120k = self._process_validation_df(df)
        print(f"Loaded 120k validation set: {self.X_val_120k.shape}")
    
    # Fall back to loading from data directory
    if self.X_val_120k is None:
      test_path = self.config.data_dir / "cellxgene_embeddings" / "test_v1"
      if test_path.exists():
        # Load all parquet files in test directory
        dfs = []
        for file in test_path.glob("*.parquet"):
          dfs.append(pd.read_parquet(file))
        
        if dfs:
          df = pd.concat(dfs, ignore_index=True)
          self.X_val_120k, self.y_val_120k = self._process_validation_df(df)
          print(f"Loaded validation set from {test_path}: {self.X_val_120k.shape}")
          
          # Create 5k subset
          if len(self.X_val_120k) > 5000:
            indices = np.random.choice(len(self.X_val_120k), 5000, replace=False)
            self.X_val_5k = self.X_val_120k[indices]
            self.y_val_5k = self.y_val_120k[indices]
            print(f"Created 5k validation subset")
  
  def _process_validation_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Process validation dataframe to extract features and labels."""
    # Extract embedding columns
    embedding_cols = [str(i) for i in range(self.config.n_dims)]
    
    # Filter to columns that exist
    available_cols = [col for col in embedding_cols if col in df.columns]
    if len(available_cols) < self.config.n_dims:
      print(f"Warning: Only {len(available_cols)} embedding dimensions available")
    
    X = df[available_cols].values.astype(np.float32)
    
    # Encode labels
    y = df["cell_type"].astype(
      pd.CategoricalDtype(categories=self.cell_types)
    ).cat.codes
    
    # Filter to valid codes
    valid_mask = y.isin(self.cell_type_codes.values)
    X = X[valid_mask]
    y = y[valid_mask].values
    
    # Map to our code system (matching notebook's y_to_code)
    y_series = pd.Series(y, name='cell_type_code')
    merged = pd.merge(
      y_series,
      self.cell_type_codes.reset_index().rename(columns={'index': 'cell_type', 0: 'code'}),
      left_on='cell_type_code',
      right_on='code',
      how='left'
    )
    y = merged.index.values
    
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
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (X, y) in enumerate(pbar):
      batch_start = time.time()
      
      # Train on batch
      loss = self.train_batch(X, y)
      epoch_losses.append(loss)
      
      batch_time = time.time() - batch_start
      batch_times.append(batch_time)
      
      # Update progress bar
      pbar.set_postfix({'loss': f'{loss:.4f}'})
      
      # Log to W&B
      if self.wandb_run is not None:
        self.wandb_run.log({
          'train/loss': loss,
          'train/batch_time': batch_time,
          'epoch': epoch,
          'global_step': self.global_step
        })
      
      # Report to Optuna trial if present
      if self.trial is not None:
        self.trial.report(loss, self.global_step)
      
      # Periodic evaluation on 5k validation set
      if self.global_step % self.config.eval_every_n_batches == 0 and self.X_val_5k is not None:
        val_metrics = self.evaluate_validation(use_5k=True)
        if self.config.verbose:
          print(f"\n[Step {self.global_step}] Val-5k metrics: "
                f"loss={val_metrics['logloss']:.4f}, "
                f"recall@10={val_metrics['recall_at_10']:.4f}")
      
      # Less frequent evaluation on full validation set
      if self.global_step % self.config.eval_full_every_n_batches == 0 and self.X_val_120k is not None:
        val_metrics = self.evaluate_validation(use_5k=False)
        if self.config.verbose:
          print(f"\n[Step {self.global_step}] Val-120k metrics: "
                f"loss={val_metrics['logloss']:.4f}, "
                f"recall@10={val_metrics['recall_at_10']:.4f}, "
                f"MRR@10={val_metrics['mrr_at_10']:.4f}")
      
      # Checkpoint
      if self.checkpoint_manager.should_save(self.global_step):
        self.checkpoint_manager.save(
          model=self.model,
          optimizer=self.optimizer,
          epoch=epoch,
          batch_idx=batch_idx,
          global_step=self.global_step,
          config=self.config
        )
      
      self.global_step += 1
    
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
    
    # Evaluate
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
      self.wandb_run.log(prefixed_metrics)
    
    # Update best metric for checkpointing
    if prefix == "val120k":
      self.checkpoint_manager.best_metric = self.checkpoint_manager.save_best_model(
        model=self.model,
        metrics={'val_logloss': metrics['logloss']},
        metric_name='val_logloss',
        checkpoint_dir=self.config.checkpoint_dir,
        mode='min',
        current_best=self.checkpoint_manager.best_metric
      )
    
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
    try:
      import wandb
      
      self.wandb_run = wandb.init(
        project=self.config.wandb_project,
        entity=self.config.wandb_entity,
        name=self.config.wandb_run_name,
        tags=self.config.wandb_tags,
        config=self.config.to_dict(),
        reinit=True
      )
      
      # Watch model
      wandb.watch(self.model, log='all', log_freq=100)
      
      print(f"Initialized W&B run: {self.wandb_run.name}")
    except ImportError:
      print("Warning: wandb not installed, skipping W&B logging")
      self.wandb_run = None
  
  def run(self) -> Dict[str, float]:
    """Run the full training loop.
    
    Returns:
      Final validation metrics for hyperparameter optimization
    """
    print(f"Starting training for {self.config.epochs} epochs")
    
    # Create dataset
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
      num_workers=self.config.num_workers
    )
    
    # Training loop
    for epoch in range(self.start_epoch, self.config.epochs):
      epoch_metrics = self.train_epoch(dataloader, epoch)
      
      print(f"Epoch {epoch} completed: avg_loss={epoch_metrics['avg_loss']:.4f}")
      
      # Full validation at end of epoch
      if self.X_val_120k is not None:
        val_metrics = self.evaluate_validation(use_5k=False)
        print(f"Validation metrics: {val_metrics}")
    
    # Save final checkpoint
    final_metrics = {}
    if self.X_val_120k is not None:
      final_metrics = self.evaluate_validation(use_5k=False)
    
    self.checkpoint_manager.save_final(
      model=self.model,
      optimizer=self.optimizer,
      epoch=self.config.epochs,
      batch_idx=0,
      global_step=self.global_step,
      metrics=final_metrics,
      config=self.config
    )
    
    # Close W&B run
    if self.wandb_run is not None:
      self.wandb_run.finish()
    
    return final_metrics