"""Checkpoint utilities for saving and loading model state."""

import torch
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    global_step: int,
    best_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
    **kwargs
) -> Path:
  """Save a training checkpoint.
  
  Args:
    checkpoint_path: Path to save checkpoint
    model: PyTorch model
    optimizer: Optimizer
    epoch: Current epoch
    batch_idx: Current batch index
    global_step: Total number of training steps
    best_metrics: Best validation metrics so far
    config: Training configuration (will be converted to dict)
    **kwargs: Additional items to save in checkpoint
    
  Returns:
    Path to saved checkpoint
  """
  # Create checkpoint directory if needed
  checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
  
  # Build checkpoint dict
  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'batch_idx': batch_idx,
    'global_step': global_step,
    'timestamp': datetime.now().isoformat(),
  }
  
  # Add optional items
  if best_metrics is not None:
    checkpoint['best_metrics'] = best_metrics
  
  if config is not None:
    # Convert config to dict if it has to_dict method
    if hasattr(config, 'to_dict'):
      checkpoint['config'] = config.to_dict()
    else:
      checkpoint['config'] = str(config)
  
  # Add any additional kwargs
  checkpoint.update(kwargs)
  
  # Save checkpoint
  torch.save(checkpoint, checkpoint_path)
  
  # Also save config as JSON for easy inspection
  if config is not None and hasattr(config, 'to_dict'):
    config_path = checkpoint_path.parent / f"{checkpoint_path.stem}_config.json"
    with open(config_path, 'w') as f:
      json.dump(config.to_dict(), f, indent=2, default=str)
  
  return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
  """Load a training checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint file
    model: PyTorch model to load state into (optional)
    optimizer: Optimizer to load state into (optional)
    device: Device to load tensors to
    
  Returns:
    Dictionary containing all checkpoint data
  """
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
  
  # Load checkpoint with weights_only=True for security
  # This is safe since we only save standard PyTorch objects and basic Python types
  if device is not None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
  else:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
  
  # Load model state if model provided
  if model is not None and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
  
  # Load optimizer state if optimizer provided
  if optimizer is not None and 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  
  return checkpoint


def save_best_model(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    metric_name: str,
    checkpoint_dir: Path,
    mode: str = 'min',
    current_best: Optional[float] = None
) -> Optional[float]:
  """Save model if it's the best so far based on a metric.
  
  Args:
    model: PyTorch model
    metrics: Current metrics dictionary
    metric_name: Name of metric to track
    checkpoint_dir: Directory to save best model
    mode: 'min' if lower is better, 'max' if higher is better
    current_best: Current best metric value
    
  Returns:
    New best metric value if model was saved, otherwise current_best
  """
  if metric_name not in metrics:
    print(f"Warning: Metric '{metric_name}' not found in metrics")
    return current_best
  
  current_value = metrics[metric_name]
  
  # Check if this is the best model
  is_best = False
  if current_best is None:
    is_best = True
  elif mode == 'min' and current_value < current_best:
    is_best = True
  elif mode == 'max' and current_value > current_best:
    is_best = True
  
  if is_best:
    # Save the model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / 'best_model.pt'
    torch.save(model.state_dict(), best_model_path)
    
    # Save metrics
    metrics_path = checkpoint_dir / 'best_model_metrics.json'
    with open(metrics_path, 'w') as f:
      json.dump(metrics, f, indent=2)
    
    print(f"Saved new best model with {metric_name}={current_value:.4f}")
    return current_value
  
  return current_best


class CheckpointManager:
  """Manages checkpointing during training."""
  
  def __init__(
      self,
      checkpoint_dir: Path,
      save_every_n_batches: int = 1000,
      keep_last_n: int = 5,
      track_metric: str = 'val_loss',
      metric_mode: str = 'min'
  ):
    """Initialize checkpoint manager.
    
    Args:
      checkpoint_dir: Directory to save checkpoints
      save_every_n_batches: Save checkpoint every N batches
      keep_last_n: Keep only the last N checkpoints
      track_metric: Metric to track for best model
      metric_mode: 'min' or 'max' for metric tracking
    """
    self.checkpoint_dir = Path(checkpoint_dir)
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    self.save_every_n_batches = save_every_n_batches
    self.keep_last_n = keep_last_n
    self.track_metric = track_metric
    self.metric_mode = metric_mode
    self.best_metric = None
    self.checkpoints = []
  
  def should_save(self, global_step: int) -> bool:
    """Check if we should save a checkpoint at this step."""
    return global_step % self.save_every_n_batches == 0
  
  def save(
      self,
      model: torch.nn.Module,
      optimizer: torch.optim.Optimizer,
      epoch: int,
      batch_idx: int,
      global_step: int,
      metrics: Optional[Dict[str, float]] = None,
      config: Optional[Any] = None
  ) -> Optional[Path]:
    """Save a checkpoint and manage old checkpoints."""
    # Create checkpoint filename
    checkpoint_name = f"checkpoint_epoch{epoch}_batch{batch_idx}_step{global_step}.pt"
    checkpoint_path = self.checkpoint_dir / checkpoint_name
    
    # Save checkpoint
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=epoch,
      batch_idx=batch_idx,
      global_step=global_step,
      best_metrics={'best_' + self.track_metric: self.best_metric} if self.best_metric else None,
      config=config
    )
    
    # Track checkpoint
    self.checkpoints.append(checkpoint_path)
    
    # Remove old checkpoints if needed
    if len(self.checkpoints) > self.keep_last_n:
      old_checkpoint = self.checkpoints.pop(0)
      if old_checkpoint.exists():
        old_checkpoint.unlink()
        # Also remove config file if exists
        config_path = old_checkpoint.parent / f"{old_checkpoint.stem}_config.json"
        if config_path.exists():
          config_path.unlink()
    
    # Check if this is the best model
    if metrics is not None and self.track_metric in metrics:
      self.best_metric = save_best_model(
        model=model,
        metrics=metrics,
        metric_name=self.track_metric,
        checkpoint_dir=self.checkpoint_dir,
        mode=self.metric_mode,
        current_best=self.best_metric
      )
    
    return checkpoint_path
  
  def save_final(
      self,
      model: torch.nn.Module,
      optimizer: torch.optim.Optimizer,
      epoch: int,
      batch_idx: int,
      global_step: int,
      metrics: Optional[Dict[str, float]] = None,
      config: Optional[Any] = None
  ) -> Path:
    """Save final checkpoint at end of training."""
    checkpoint_path = self.checkpoint_dir / "final_checkpoint.pt"
    
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=epoch,
      batch_idx=batch_idx,
      global_step=global_step,
      best_metrics={'best_' + self.track_metric: self.best_metric} if self.best_metric else None,
      final_metrics=metrics,
      config=config
    )
    
    print(f"Saved final checkpoint to {checkpoint_path}")
    return checkpoint_path