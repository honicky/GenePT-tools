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
    default=None,
    help="AWS profile to use for S3 access (None for default credentials)"
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
  parser.add_argument(
    "--use-composable-dataset",
    action="store_true",
    help="Use composable embedding dataset (new system)"
  )
  parser.add_argument(
    "--base-data-dir",
    type=Path,
    default=None,
    help="Base directory for composable embeddings (e.g., /mmc-scratch/scratch/)"
  )
  parser.add_argument(
    "--embedding-types",
    nargs="+",
    default=["genept"],
    help="Embedding types to use (e.g., genept tissue scgpt)"
  )
  parser.add_argument(
    "--genept-dims",
    type=int,
    default=1536,
    help="Number of GenePT dimensions to use (default: 1536, use 0 for all 3072)"
  )
  # cell-count-threshold removed - must be specified in tuning config fixed_params
  parser.add_argument(
    "--cell-counts-file",
    type=Path,
    default=None,
    help="CSV file with cell type counts (columns: cell_type, cell_count)"
  )

  # Model parameters
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
  parser.add_argument(
    "--wandb-save-artifacts",
    action="store_true",
    default=True,
    help="Save checkpoints as WandB artifacts (default: True)"
  )
  parser.add_argument(
    "--no-wandb-artifacts",
    action="store_true",
    help="Disable saving checkpoints as WandB artifacts"
  )
  parser.add_argument(
    "--local-checkpoints",
    action="store_true",
    default=True,
    help="Save checkpoints to local filesystem (default: True, automatically disabled during hyperparameter optimization with WandB)"
  )
  parser.add_argument(
    "--no-local-checkpoints",
    action="store_true",
    help="Disable local filesystem checkpoints (useful when using WandB artifacts)"
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
  parser.add_argument(
    "--max-steps-per-epoch",
    type=int,
    default=None,
    help="Maximum training steps per epoch (for quick testing)"
  )
  
  # Best model tracking
  parser.add_argument(
    "--best-model-metric",
    type=str,
    default="logloss",
    help="Metric to use for saving best model (default: logloss). Use 'none' to disable best model tracking. Automatically overridden to match Optuna optimization metric when using hyperparameter tuning."
  )
  parser.add_argument(
    "--best-model-mode",
    type=str,
    default="min",
    choices=["min", "max"],
    help="Whether best metric should be minimized or maximized (default: min for logloss). Automatically overridden to match Optuna direction when using hyperparameter tuning."
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
    # Create sequential codes from 0 to n-1 (ignore the 'code' column which has non-sequential values)
    # The order in the CSV file defines the mapping
    cell_type_codes = pd.Series(range(len(cell_types)), index=cell_types)
  else:
    # Use default from notebook (simplified for demo)
    # In production, load from a reference file
    print("Warning: Using simplified cell type list. Provide --cell-types-file for full list.")
    cell_types = [f"type_{i}" for i in range(377)]  # Placeholder
    cell_type_codes = pd.Series(range(377), index=cell_types)

  return cell_types, cell_type_codes


def load_cell_types_from_counts(cell_counts_file):
  """Load cell types directly from cell counts file.

  The cell_counts.csv file contains all cell types with their counts.
  This serves as the canonical source for cell type ordering.

  Args:
    cell_counts_file: Path to CSV file with columns (cell_type, cell_count)
                     Can be string or Path object

  Returns:
    Tuple of (cell_types list, cell_type_codes Series)
  """
  # Convert to Path if string
  cell_counts_file = Path(cell_counts_file)

  if not cell_counts_file.exists():
    raise FileNotFoundError(f"Cell counts file not found: {cell_counts_file}")

  # Load cell counts file
  counts_df = pd.read_csv(cell_counts_file)

  # Verify required columns
  if 'cell_type' not in counts_df.columns:
    raise ValueError(f"cell_counts_file must have 'cell_type' column")
  if 'cell_count' not in counts_df.columns:
    raise ValueError(f"cell_counts_file must have 'cell_count' column")

  # Extract cell types (order in file defines the canonical ordering)
  cell_types = counts_df['cell_type'].tolist()

  # Create sequential codes from 0 to n-1
  cell_type_codes = pd.Series(range(len(cell_types)), index=cell_types)

  return cell_types, cell_type_codes


def create_code_remapping(
    cell_types: list,
    cell_type_codes: pd.Series,
    cell_counts_file: Path,
    min_count: int
):
  """Create code remapping that maps excluded types to -100 (for filtering).

  Args:
    cell_types: List of all cell types
    cell_type_codes: Series mapping cell types to original codes
    cell_counts_file: Path to CSV with columns (cell_type, cell_count)
    min_count: Minimum number of samples required

  Returns:
    filtered_cell_types: List of included cell types
    filtered_codes: Sequential codes for included types (0 to N-1)
    code_remapping: Dict mapping ALL original codes to filtered codes OR -100
    mapping_df: DataFrame with mapping info for WandB logging
  """
  # Load cell counts
  counts_df = pd.read_csv(cell_counts_file)

  # Separate included vs excluded types
  included_df = counts_df[counts_df['cell_count'] >= min_count].copy()
  excluded_df = counts_df[counts_df['cell_count'] < min_count].copy()

  # Create sequential codes for included types
  filtered_cell_types = included_df['cell_type'].tolist()
  filtered_codes = pd.Series(range(len(filtered_cell_types)), index=filtered_cell_types)

  # Create remapping for ALL types (included + excluded)
  code_remapping = {}

  # Map included types to new sequential codes (0, 1, 2, ..., N-1)
  for cell_type in filtered_cell_types:
    if cell_type in cell_type_codes.index:
      original_code = cell_type_codes[cell_type]
      new_code = filtered_codes[cell_type]
      code_remapping[original_code] = new_code

  # Map excluded types to -100 (marker for filtering)
  for cell_type in excluded_df['cell_type']:
    if cell_type in cell_type_codes.index:
      original_code = cell_type_codes[cell_type]
      code_remapping[original_code] = -100

  # Create mapping DataFrame (only for included types that exist in cell_type_codes)
  mapping_rows = []
  for i, ct in enumerate(filtered_cell_types):
    if ct in cell_type_codes.index:
      mapping_rows.append({
        'cell_type': ct,
        'filtered_code': i,
        'original_code': cell_type_codes[ct],
        'cell_count': counts_df[counts_df['cell_type'] == ct]['cell_count'].iloc[0]
      })
  mapping_df = pd.DataFrame(mapping_rows).sort_values('cell_count', ascending=False)

  # Print filtering summary
  total_types = len(cell_types)
  total_samples = counts_df['cell_count'].sum()
  filtered_samples = included_df['cell_count'].sum()

  print(f"\nCell Type Filtering (threshold={min_count}):")
  print(f"  Original cell types: {total_types}")
  print(f"  Filtered cell types: {len(filtered_cell_types)} ({100*len(filtered_cell_types)/total_types:.1f}%)")
  print(f"  Cell types excluded: {total_types - len(filtered_cell_types)}")
  print(f"  Sample coverage: {filtered_samples:,} / {total_samples:,} ({100*filtered_samples/total_samples:.1f}%)")
  print(f"  Top 10 included cell types by count:")
  for i, row in mapping_df.head(10).iterrows():
    print(f"    {row['filtered_code']:3d}: {row['cell_type'][:60]:60s} ({row['cell_count']:6,d} samples)")

  return filtered_cell_types, filtered_codes, code_remapping, mapping_df


def run_training_with_config(
    config: TrainingConfig,
    cell_types: list,
    cell_type_codes: pd.Series,
    code_remapping: dict = None,
    mapping_df: pd.DataFrame = None,
    trial=None
):
  """Run training with a given configuration.

  Args:
    config: Training configuration
    cell_types: List of cell type names
    cell_type_codes: Series mapping cell types to codes
    code_remapping: Optional dict mapping original codes to filtered codes
    mapping_df: Optional DataFrame with cell type mapping for WandB
    trial: Optional Optuna trial for hyperparameter tuning

  Returns:
    Final metrics dictionary
  """
  # Create trainer
  trainer = MLPTrainer(
    config=config,
    cell_types=cell_types,
    cell_type_codes=cell_type_codes,
    code_remapping=code_remapping,
    mapping_df=mapping_df
  )

  # Add Optuna trial if provided
  if trial is not None:
    trainer.optuna_trial = trial

  # Run training
  print("Starting training...")
  final_metrics = trainer.run()

  return final_metrics


def validate_required_parameters(args, config=None):
  """Validate that all required parameters are provided.
  
  Args:
    args: Parsed command line arguments
    config: Optional tuning config dict (for tuning mode)
    
  Returns:
    List of error messages (empty if all required params present)
  """
  errors = []
  
  # Get effective values (command line overrides config)
  def get_effective_value(param_name):
    # Command line value takes precedence (but False from store_true doesn't count as "specified")
    cli_value = getattr(args, param_name, None)

    # For boolean flags (store_true), only use CLI value if it's True (explicitly set)
    # Otherwise check config. This allows config to set True when CLI didn't specify the flag.
    if param_name == 'use_composable_dataset':
      if cli_value is True:  # Only override if flag was explicitly provided
        return cli_value
    elif cli_value is not None:
      return cli_value

    # Check config fixed_params if in tuning mode
    if config and 'fixed_params' in config:
      return config['fixed_params'].get(param_name)

    return None
  
  # Check if using composable dataset
  use_composable = get_effective_value('use_composable_dataset')

  # DEBUG: Print what we detected
  if config:
    print(f"DEBUG: use_composable_dataset = {use_composable}")
    print(f"DEBUG: fixed_params keys = {list(config.get('fixed_params', {}).keys())}")

  if use_composable:
    # For composable dataset, require base_data_dir
    required_params = [
      ('base_data_dir', 'base directory for composable embeddings'),
    ]
  else:
    # For S3/parquet dataset, require local_data_dir and test_data_dir
    required_params = [
      ('local_data_dir', 'training data directory'),
      ('test_data_dir', 'validation/test data directory'),
    ]

  # Check each required parameter
  for param_name, description in required_params:
    effective_value = get_effective_value(param_name)
    if not effective_value:
      errors.append(f"Missing required parameter: {param_name} ({description})")

  return errors


def main():
  """Main training function."""
  args = parse_args()

  # If using tuning mode, load config early to get filtering parameters
  tuning_config_dict = None
  if args.tuning_config:
    import yaml
    with open(args.tuning_config, 'r') as f:
      tuning_config_dict = yaml.safe_load(f)

    # Override args with fixed_params from tuning config if not specified on command line
    fixed_params = tuning_config_dict.get('fixed_params', {})
    if args.cell_counts_file is None and 'cell_counts_file' in fixed_params:
      args.cell_counts_file = fixed_params['cell_counts_file']

    # cell_count_threshold must be specified in config (no CLI argument, no default)
    if 'cell_count_threshold' not in fixed_params:
      raise ValueError(
        "cell_count_threshold must be specified in tuning config fixed_params. "
        "Use 0 for no filtering."
      )
    cell_count_threshold = fixed_params['cell_count_threshold']
  else:
    # Non-tuning mode: cell_count_threshold must be specified
    raise ValueError(
      "cell_count_threshold is required. Use --tuning-config with fixed_params.cell_count_threshold specified."
    )

  # Load cell types from cell_counts_file (required)
  if args.cell_counts_file is None:
    raise ValueError(
      "Cell counts file is required. Specify via --cell-counts-file or in tuning config fixed_params."
    )

  cell_types, cell_type_codes = load_cell_types_from_counts(args.cell_counts_file)
  print(f"Loaded {len(cell_types)} cell types from {args.cell_counts_file}")

  # Apply cell type filtering if threshold specified
  code_remapping = None
  mapping_df = None
  if cell_count_threshold > 0:
    filtered_cell_types, filtered_codes, code_remapping, mapping_df = create_code_remapping(
      cell_types=cell_types,
      cell_type_codes=cell_type_codes,
      cell_counts_file=args.cell_counts_file,
      min_count=cell_count_threshold
    )
    # Use filtered cell types for training
    cell_types = filtered_cell_types
    cell_type_codes = filtered_codes

  # Check if we're in tuning mode
  if tuning_config_dict:
    if not OPTUNA_AVAILABLE:
      print("Error: Optuna is required for hyperparameter tuning. Install with: pip install optuna")
      sys.exit(1)

    # Validate required parameters
    validation_errors = validate_required_parameters(args, tuning_config_dict)
    if validation_errors:
      print("Error: Missing required parameters:")
      for error in validation_errors:
        print(f"  - {error}")
      print(f"\nEither specify these parameters on the command line or add them to the")
      print(f"'fixed_params' section of your config file: {args.tuning_config}")
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
    def create_trainer(trial: optuna.Trial):
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
        # Format key parameters for run name
        lr_str = f"{config.learning_rate:.1e}"  # Scientific notation
        wd_str = f"{config.weight_decay:.1e}" if config.weight_decay > 0 else "0"
        optimizer = getattr(config, 'optimizer_type', 'adam')
        scheduler = getattr(config, 'lr_scheduler', 'none')
        grad_clip = getattr(config, 'gradient_clip_val', 'none')
        
        # Create descriptive run name with trial number for sorting
        # For composable datasets, use genept_dims or "composable" as placeholder
        dims_str = f"{config.genept_dims}" if hasattr(config, 'genept_dims') and config.genept_dims else "500"
        config.wandb_run_name = (
          f"{trial.number:03d}_dims{dims_str}_layers{config.n_hidden_layers}_"
          f"drop{config.dropout:.3f}_bs{config.batch_size}_lr{lr_str}_wd{wd_str}_"
          f"{optimizer}_{scheduler}"
        )
        
        # Add gradient clipping to name if used
        if grad_clip and grad_clip != 'none' and grad_clip is not None:
          config.wandb_run_name += f"_clip{grad_clip}"
        
        config.wandb_tags = ["optuna", f"trial_{trial.number}"]
      
      # Create trainer (don't run it yet - OptunaManager will call .run())
      trainer = MLPTrainer(
        config=config,
        cell_types=cell_types,
        cell_type_codes=cell_type_codes,
        code_remapping=code_remapping,
        mapping_df=mapping_df
      )
      
      # Add Optuna trial for pruning support
      trainer.optuna_trial = trial
      
      return trainer
    
    # Run optimization
    print(f"Starting optimization with n_trials={args.tuning_n_trials}, timeout={args.tuning_timeout}")
    manager.run_optimization(
      trainer_factory=create_trainer,
      n_trials=args.tuning_n_trials,
      timeout=args.tuning_timeout
    )
    
    # Optimization complete
    print("\n" + "="*60)
    print("Hyperparameter Optimization Complete")
    print("="*60)
    
    # Save optimization results
    results_file = args.checkpoint_dir / "optuna_results.json"
    manager.save_results(results_file)
    
    # Print best results
    best_trial = manager.study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in best_trial.params.items():
      print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {results_file}")
    print("\nTo train with the best parameters, use the saved configuration.")
    
    # Exit after optimization (no final training)
    return
    
  else:
    # Normal training mode (non-tuning)
    # Validate required parameters
    validation_errors = validate_required_parameters(args)
    if validation_errors:
      print("Error: Missing required parameters:")
      for error in validation_errors:
        print(f"  - {error}")
      print(f"\nPlease specify these parameters on the command line.")
      sys.exit(1)
    
    # Determine if hierarchical metrics should be enabled
    enable_hierarchical = args.enable_hierarchical_metrics and not args.disable_hierarchical_metrics
    
    # Determine if WandB artifacts should be enabled
    enable_wandb_artifacts = args.wandb_save_artifacts and not args.no_wandb_artifacts
    
    # Determine if local checkpoints should be enabled
    enable_local_checkpoints = args.local_checkpoints and not args.no_local_checkpoints
    
    # Handle genept_dims (0 means use all dimensions)
    genept_dims = args.genept_dims if args.genept_dims > 0 else None

    # Create configuration
    config = TrainingConfig(
      # Data
      local_data_dir=args.local_data_dir,
      test_data_dir=args.test_data_dir,
      s3_bucket=args.s3_bucket,
      s3_prefix=args.s3_prefix,
      aws_profile=args.aws_profile,
      download_if_missing=args.download_if_missing and not args.no_download,
      # Composable embeddings
      use_composable_dataset=args.use_composable_dataset,
      base_data_dir=args.base_data_dir,
      embedding_types=args.embedding_types,
      genept_dims=genept_dims,
      # Cell type filtering
      cell_count_threshold=cell_count_threshold,
      cell_counts_file=args.cell_counts_file,
      # Model
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
      wandb_save_artifacts=enable_wandb_artifacts,
      local_checkpoints=enable_local_checkpoints,
      # Resume and subset
      resume_from=args.resume_from,
      start_batch_file=args.start_batch_file,
      end_batch_file=args.end_batch_file,
      max_steps_per_epoch=args.max_steps_per_epoch,
      # Best model tracking
      best_model_metric=args.best_model_metric,
      best_model_mode=args.best_model_mode,
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
    final_metrics = run_training_with_config(
      config, cell_types, cell_type_codes,
      code_remapping=code_remapping,
      mapping_df=mapping_df
    )
  
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