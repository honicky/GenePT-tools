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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import MLPTrainer
from src.training.config import TrainingConfig


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


def main():
  """Main training function."""
  args = parse_args()
  
  # Load cell types
  cell_types, cell_type_codes = load_cell_types(args.cell_types_file)
  print(f"Loaded {len(cell_types)} cell types, training on {len(cell_type_codes)} codes")
  
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
  
  # Create trainer
  trainer = MLPTrainer(
    config=config,
    cell_types=cell_types,
    cell_type_codes=cell_type_codes
  )
  
  # Run training
  print("Starting training...")
  final_metrics = trainer.run()
  
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