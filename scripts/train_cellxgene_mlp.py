#!/usr/bin/env python
"""Training script for CellXGene MLP model.

This script trains an MLP model on pre-shuffled CellXGene data from S3.
It supports both local data caching and direct S3 streaming.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import MLPTrainer
from src.training.config import TrainingConfig

# Optional import for hyperparameter tuning
try:
  from src.training.optuna_manager import OptunaManager
  import optuna
  OPTUNA_AVAILABLE = True
except ImportError:
  OPTUNA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Train CellXGene MLP model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  
  # Data parameters
  parser.add_argument(
    "--local-data-dir",
    type=Path,
    default=None,
    help="Local directory containing training data (or for caching S3 data)"
  )
  parser.add_argument(
    "--test-data-dir",
    type=Path,
    default=None,
    help="Directory containing validation data"
  )
  parser.add_argument(
    "--s3-bucket",
    type=str,
    default="pythiomicsdata",
    help="S3 bucket containing training data"
  )
  parser.add_argument(
    "--s3-prefix",
    type=str,
    default="cellxgene_v2/training_v1_suffled",
    help="S3 prefix for training data"
  )
  parser.add_argument(
    "--aws-profile",
    type=str,
    default="xcellerate",
    help="AWS profile to use for S3 access"
  )
  parser.add_argument(
    "--download-if-missing",
    action="store_true",
    help="Download files from S3 if not found locally"
  )
  parser.add_argument(
    "--no-download",
    action="store_true",
    help="Only use local files, don't download from S3"
  )
  
  # Model parameters
  parser.add_argument(
    "--n-dims",
    type=int,
    default=500,
    help="Number of embedding dimensions"
  )
  parser.add_argument(
    "--n-hidden-layers",
    type=int,
    default=3,
    help="Number of hidden layers in MLP"
  )
  parser.add_argument(
    "--dropout",
    type=float,
    default=0.053,
    help="Dropout rate"
  )
  
  # Training parameters
  parser.add_argument(
    "--learning-rate",
    type=float,
    default=4.366e-5,
    help="Learning rate"
  )
  parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-5,
    help="Weight decay for AdamW"
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=1024,
    help="Batch size"
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train"
  )
  
  # Evaluation parameters
  parser.add_argument(
    "--eval-every-n-batches",
    type=int,
    default=10,
    help="Evaluate on 5k validation set every N batches"
  )
  parser.add_argument(
    "--eval-full-every-n-batches",
    type=int,
    default=250,
    help="Evaluate on full validation set every N batches"
  )
  parser.add_argument(
    "--checkpoint-every-n-batches",
    type=int,
    default=1000,
    help="Save checkpoint every N batches"
  )
  
  # System parameters
  parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cpu", "cuda", "mps"],
    help="Device to use for training"
  )
  parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of data loading workers"
  )
  parser.add_argument(
    "--mixed-precision",
    action="store_true",
    help="Use mixed precision training"
  )
  
  # Output and logging
  parser.add_argument(
    "--checkpoint-dir",
    type=Path,
    default=Path("checkpoints"),
    help="Directory to save checkpoints"
  )
  parser.add_argument(
    "--wandb-project",
    type=str,
    default=None,
    help="Weights & Biases project name"
  )
  parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="Weights & Biases entity/team name"
  )
  parser.add_argument(
    "--wandb-run-name",
    type=str,
    default=None,
    help="Weights & Biases run name"
  )
  parser.add_argument(
    "--wandb-tags",
    nargs="+",
    default=None,
    help="Weights & Biases tags"
  )
  
  # Resume and subset
  parser.add_argument(
    "--resume-from",
    type=Path,
    default=None,
    help="Path to checkpoint to resume from"
  )
  parser.add_argument(
    "--start-batch-file",
    type=int,
    default=0,
    help="Start from this batch file (for debugging)"
  )
  parser.add_argument(
    "--end-batch-file",
    type=int,
    default=None,
    help="End at this batch file (for debugging)"
  )
  
  # Shuffling
  parser.add_argument(
    "--no-shuffle-files",
    action="store_true",
    help="Don't shuffle file order per epoch"
  )
  parser.add_argument(
    "--no-shuffle-within-files",
    action="store_true",
    help="Don't shuffle samples within files"
  )
  
  # Other
  parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose output"
  )
  parser.add_argument(
    "--cell-types-file",
    type=Path,
    default=None,
    help="Path to file containing cell types and codes (optional)"
  )
  
  # Hierarchical metrics parameters
  parser.add_argument(
    "--enable-hierarchical-metrics",
    action="store_true",
    default=True,
    help="Enable hierarchical evaluation using Cell Ontology (default: enabled)"
  )
  parser.add_argument(
    "--disable-hierarchical-metrics",
    action="store_true",
    help="Disable hierarchical evaluation"
  )
  parser.add_argument(
    "--ontology-cache-dir",
    type=Path,
    default=Path("data/ontology"),
    help="Directory to cache Cell Ontology files"
  )
  
  # Hyperparameter tuning parameters
  parser.add_argument(
    "--tuning-config",
    type=Path,
    default=None,
    help="Path to hyperparameter tuning configuration file (enables tuning mode when provided)"
  )
  parser.add_argument(
    "--tuning-n-trials",
    type=int,
    default=None,
    help="Override number of trials from config"
  )
  parser.add_argument(
    "--tuning-timeout",
    type=int,
    default=None,
    help="Maximum time in seconds for tuning"
  )
  parser.add_argument(
    "--tuning-storage",
    type=str,
    default=None,
    help="Optuna study database URL (e.g., sqlite:///optuna.db)"
  )
  
  return parser.parse_args()


def load_cell_types(cell_types_file: Path = None):
  """Load cell types and codes.
  
  Args:
    cell_types_file: Optional path to file with cell types and codes
    
  Returns:
    Tuple of (cell_types list, cell_type_codes Series)
  """
  if cell_types_file and cell_types_file.exists():
    # Load from file
    df = pd.read_csv(cell_types_file)
    cell_types = df['cell_type'].tolist()
    # Create series with cell type names as index and codes as values
    cell_type_codes = pd.Series(df['code'].values, index=df['cell_type'].values)
  else:
    # Use default from notebook (simplified for demo)
    # In production, load from a reference file
    print("Warning: Using simplified cell type list. Provide --cell-types-file for full list.")
    cell_types = [f"type_{i}" for i in range(377)]  # Placeholder
    cell_type_codes = pd.Series(range(377))
  
  return cell_types, cell_type_codes


def run_training_with_config(config: TrainingConfig, cell_types: list, cell_type_codes: pd.Series, trial=None):
  """Run training with a given configuration.
  
  Args:
    config: Training configuration
    cell_types: List of cell type names
    cell_type_codes: Series mapping cell types to codes
    trial: Optional Optuna trial for hyperparameter tuning
    
  Returns:
    Final metrics dictionary
  """
  # Create trainer
  trainer = MLPTrainer(
    config=config,
    cell_types=cell_types,
    cell_type_codes=cell_type_codes
  )
  
  # Add Optuna trial if provided
  if trial is not None:
    trainer.optuna_trial = trial
  
  # Run training
  print("Starting training...")
  final_metrics = trainer.run()
  
  return final_metrics


def main():
  """Main training function."""
  args = parse_args()
  
  # Load cell types
  cell_types, cell_type_codes = load_cell_types(args.cell_types_file)
  print(f"Loaded {len(cell_types)} cell types, training on {len(cell_type_codes)} codes")
  
  # Check if we're in tuning mode
  if args.tuning_config:
    if not OPTUNA_AVAILABLE:
      print("Error: Optuna is required for hyperparameter tuning. Install with: pip install optuna")
      sys.exit(1)
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning Mode")
    print("="*60)
    
    # Create OptunaManager
    manager = OptunaManager(
      config_path=args.tuning_config,
      storage=args.tuning_storage
    )
    
    # Create trainer factory
    def create_and_run_trainer(trial: optuna.Trial):
      # Get suggested config
      config = manager.suggest_hyperparameters(trial)
      
      # Override with command-line arguments if provided
      if args.local_data_dir:
        config.local_data_dir = args.local_data_dir
      if args.test_data_dir:
        config.test_data_dir = args.test_data_dir
      if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
      if args.wandb_project:
        config.wandb_project = args.wandb_project
      if args.wandb_entity:
        config.wandb_entity = args.wandb_entity
      
      # Add trial number to wandb run name
      if config.wandb_project:
        config.wandb_run_name = f"trial_{trial.number}"
        config.wandb_tags = ["optuna", f"trial_{trial.number}"]
      
      # Run training
      final_metrics = run_training_with_config(config, cell_types, cell_type_codes, trial)
      
      # Report intermediate values for pruning
      for step in range(0, len(final_metrics.get('val_loss_history', [])), 10):
        if 'val_loss_history' in final_metrics and step < len(final_metrics['val_loss_history']):
          trial.report(final_metrics['val_loss_history'][step], step)
          
          # Check if trial should be pruned
          if trial.should_prune():
            raise optuna.TrialPruned()
      
      # Return optimization metric
      return final_metrics
    
    # Run optimization
    manager.run_optimization(
      trainer_factory=create_and_run_trainer,
      n_trials=args.tuning_n_trials,
      timeout=args.tuning_timeout
    )
    
    # Get best configuration
    print("\n" + "="*60)
    print("Optimization Complete - Training Final Model")
    print("="*60)
    
    best_config = manager.get_best_config()
    
    # Override with command-line arguments for final training
    if args.local_data_dir:
      best_config.local_data_dir = args.local_data_dir
    if args.test_data_dir:
      best_config.test_data_dir = args.test_data_dir
    if args.checkpoint_dir:
      best_config.checkpoint_dir = args.checkpoint_dir
    
    # Run final training with best config
    final_metrics = run_training_with_config(best_config, cell_types, cell_type_codes)
    
    # Save optimization results
    results_file = best_config.checkpoint_dir / "optuna_results.json"
    manager.save_results(results_file)
    
  else:
    # Normal training mode (non-tuning)
    # Determine if hierarchical metrics should be enabled
    enable_hierarchical = args.enable_hierarchical_metrics and not args.disable_hierarchical_metrics
    
    # Create configuration
    config = TrainingConfig(
      # Data
      local_data_dir=args.local_data_dir,
      test_data_dir=args.test_data_dir,
      s3_bucket=args.s3_bucket,
      s3_prefix=args.s3_prefix,
      aws_profile=args.aws_profile,
      download_if_missing=args.download_if_missing and not args.no_download,
      # Model
      n_dims=args.n_dims,
      n_hidden_layers=args.n_hidden_layers,
      dropout=args.dropout,
      # Training
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      batch_size=args.batch_size,
      epochs=args.epochs,
      # Evaluation
      eval_every_n_batches=args.eval_every_n_batches,
      eval_full_every_n_batches=args.eval_full_every_n_batches,
      checkpoint_every_n_batches=args.checkpoint_every_n_batches,
      # System
      device=args.device,
      num_workers=args.num_workers,
      mixed_precision=args.mixed_precision,
      # Output
      checkpoint_dir=args.checkpoint_dir,
      wandb_project=args.wandb_project,
      wandb_entity=args.wandb_entity,
      wandb_run_name=args.wandb_run_name,
      wandb_tags=args.wandb_tags,
      # Resume and subset
      resume_from=args.resume_from,
      start_batch_file=args.start_batch_file,
      end_batch_file=args.end_batch_file,
      # Shuffling
      shuffle_files_per_epoch=not args.no_shuffle_files,
      shuffle_within_files=not args.no_shuffle_within_files,
      # Other
      seed=args.seed,
      verbose=args.verbose,
      # Hierarchical metrics
      enable_hierarchical_metrics=enable_hierarchical,
      ontology_cache_dir=args.ontology_cache_dir
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    config_dict = config.to_dict()
    for key, value in config_dict.items():
      print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Run training
    final_metrics = run_training_with_config(config, cell_types, cell_type_codes)
  
  # Print final metrics
  print("\n" + "="*60)
  print("Final Metrics:")
  print("="*60)
  for key, value in final_metrics.items():
    if isinstance(value, float):
      print(f"  {key}: {value:.4f}")
    else:
      print(f"  {key}: {value}")
  print("="*60)
  
  # Save metrics to file
  metrics_file = config.checkpoint_dir / "final_metrics.json"
  with open(metrics_file, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    json_metrics = {}
    for k, v in final_metrics.items():
      if hasattr(v, 'item'):
        json_metrics[k] = v.item()
      elif isinstance(v, np.ndarray):
        json_metrics[k] = v.tolist()
      else:
        json_metrics[k] = v
    json.dump(json_metrics, f, indent=2)
  print(f"\nSaved final metrics to {metrics_file}")
  
  print("\nTraining complete!")


if __name__ == "__main__":
  main()