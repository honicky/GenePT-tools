"""Configuration classes for CellXGene MLP training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TrainingConfig:
  """Configuration for training the CellXGene MLP model.
  
  Default values are from the best performing run in the notebook.
  """
  
  # Data parameters
  data_dir: Path = Path("data")
  local_data_dir: Optional[Path] = None
  test_data_dir: Optional[Path] = None
  s3_bucket: str = "pythiomicsdata"
  s3_prefix: str = "cellxgene_v2/training_v1_suffled"
  aws_profile: str = None
  download_if_missing: bool = True
  n_dims: int = 500  # Embedding dimension used in best notebook run
  
  # Model parameters (from notebook's best hyperparameters)
  n_hidden_layers: int = 3
  dropout: float = 0.053  # 0.05310516410376493 rounded
  
  # Training parameters (from notebook's best run)
  learning_rate: float = 4.366e-5  # 4.366013566028419e-05
  weight_decay: float = 1e-5
  batch_size: int = 1024
  epochs: int = 10
  
  # Advanced optimization parameters (for hyperparameter tuning)
  optimizer_type: str = "adam"  # Options: adam, adamw, sgd
  lr_scheduler: str = "none"  # Options: none, cosine, step, exponential
  lr_scheduler_params: dict = field(default_factory=dict)  # Scheduler-specific params
  label_smoothing: float = 0.0  # Label smoothing factor (0.0 = no smoothing)
  gradient_clip_val: Optional[float] = None  # Max gradient norm for clipping
  
  # Evaluation parameters
  eval_every_n_batches: int = 10  # Evaluate on 5k validation set
  eval_full_every_n_batches: int = 250  # Evaluate on 120k validation set
  checkpoint_every_n_batches: int = 1000
  
  # System parameters
  device: str = "auto"  # auto, cuda, mps, or cpu
  num_workers: int = 0  # DataLoader workers (0 for no multiprocessing)
  mixed_precision: bool = False  # Use mixed precision training
  seed: int = 42
  load_validation_data: bool = True  # Set to False for unit tests to speed them up
  
  # Best model tracking (automatically derived from Optuna config when using hyperparameter tuning)
  best_model_metric: str = "logloss"  # Default metric for best model tracking (matches common Optuna usage)
  best_model_mode: str = "min"  # "min" for lower=better, "max" for higher=better
  
  # File shuffling
  shuffle_files_per_epoch: bool = True
  shuffle_within_files: bool = True
  
  # Subset for debugging
  start_batch_file: int = 0
  end_batch_file: Optional[int] = None
  max_steps_per_epoch: Optional[int] = None  # Limit training steps per epoch for quick testing  # None means use all files
  
  # Paths
  checkpoint_dir: Path = Path("checkpoints")
  resume_from: Optional[Path] = None
  
  # Logging
  wandb_project: Optional[str] = None
  wandb_entity: Optional[str] = None
  wandb_run_name: Optional[str] = None
  wandb_tags: List[str] = field(default_factory=list)
  wandb_save_artifacts: bool = True  # Save checkpoints as WandB artifacts
  local_checkpoints: bool = True  # Save local filesystem checkpoints (disable when using WandB artifacts)
  verbose: bool = True
  
  # Hierarchical metrics
  enable_hierarchical_metrics: bool = True
  ontology_cache_dir: Path = Path("data/ontology")
  
  def __post_init__(self):
    """Convert string paths to Path objects and validate config."""
    # Convert paths
    self.data_dir = Path(self.data_dir)
    self.checkpoint_dir = Path(self.checkpoint_dir)
    
    if self.local_data_dir is not None:
      self.local_data_dir = Path(self.local_data_dir)
    
    if self.test_data_dir is not None:
      self.test_data_dir = Path(self.test_data_dir)
    
    if self.resume_from is not None:
      self.resume_from = Path(self.resume_from)
    
    # Auto-detect device
    if self.device == "auto":
      import torch
      if torch.cuda.is_available():
        self.device = "cuda"
      elif torch.backends.mps.is_available():
        self.device = "mps"
      else:
        self.device = "cpu"
  
  def to_dict(self) -> dict:
    """Convert config to dictionary for logging."""
    config_dict = {}
    for key, value in self.__dict__.items():
      if isinstance(value, Path):
        config_dict[key] = str(value)
      else:
        config_dict[key] = value
    return config_dict


@dataclass
class OptunaConfig:
  """Configuration for hyperparameter optimization with Optuna."""
  
  # Search space
  learning_rate_min: float = 1e-5
  learning_rate_max: float = 1e-2
  dropout_min: float = 0.0
  dropout_max: float = 0.5
  n_hidden_layers_min: int = 1
  n_hidden_layers_max: int = 4
  batch_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
  
  # Optimization settings
  n_trials: int = 100
  n_epochs_per_trial: int = 2  # Quick evaluation
  metric_to_optimize: str = "val_loss"  # or "macro_f1"
  direction: str = "minimize"  # or "maximize"
  
  # Optuna specific
  study_name: str = "cellxgene_mlp_optimization"
  storage: Optional[str] = None  # e.g., "sqlite:///optuna.db"
  load_if_exists: bool = True
  
  def suggest_config(self, trial, base_config: TrainingConfig) -> TrainingConfig:
    """Suggest hyperparameters for a trial.
    
    Args:
      trial: Optuna trial object
      base_config: Base configuration to modify
      
    Returns:
      Modified TrainingConfig with suggested hyperparameters
    """
    import copy
    config = copy.deepcopy(base_config)
    
    # Suggest hyperparameters
    config.learning_rate = trial.suggest_loguniform('learning_rate', self.learning_rate_min, self.learning_rate_max)
    config.dropout = trial.suggest_uniform('dropout', self.dropout_min, self.dropout_max)
    config.n_hidden_layers = trial.suggest_int('n_hidden_layers', self.n_hidden_layers_min, self.n_hidden_layers_max)
    config.batch_size = trial.suggest_categorical('batch_size', self.batch_sizes)
    
    # Use fewer epochs for optimization
    config.epochs = self.n_epochs_per_trial
    
    return config