#!/usr/bin/env python3
"""
General-purpose evaluation script for MLP models trained on composable embeddings.

This script evaluates trained MLP classifiers on any dataset with composable embeddings
in parquet format, computing comprehensive metrics including hierarchical Cell Ontology metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mlp_classifier import MLPClassifier
from src.data_loading.composable_dataset import ComposableTrainingDataset
from src.training.metrics import evaluate_with_hierarchy, evaluate_and_return_predictions


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


def map_tissue_to_cz_slim(tissue_id: str) -> Optional[str]:
  """Map arbitrary tissue ID to CZ slim tissue using three-tier strategy.

  Uses the same logic as TissueEncoder:
  Tier 1: Return as-is if already in CZ slim
  Tier 2: Find nearest CZ slim ancestor via ontology
  Tier 3: Use manual fallback mapping

  Args:
    tissue_id: Tissue ontology term ID (UBERON or CL)

  Returns:
    CZ slim tissue ID, or None if no mapping found
  """
  from cellxgene_ontology_guide.curated_ontology_term_lists import (
    CuratedOntologyTermList, get_curated_ontology_term_list
  )
  from cellxgene_ontology_guide.ontology_parser import OntologyParser
  from src.data_loading.tissue_encoder import FALLBACK_TISSUE_MAPPINGS

  # Get curated tissue list (CZ slim)
  curated_tissues = get_curated_ontology_term_list(CuratedOntologyTermList.TISSUE_GENERAL)
  curated_tissues_set = set(curated_tissues)

  # Tier 1: Already in CZ slim
  if tissue_id in curated_tissues_set:
    return tissue_id

  # Tier 2: Find nearest CZ slim ancestor
  try:
    ontology_parser = OntologyParser()
    ancestors = ontology_parser.get_term_ancestors(tissue_id, include_self=False)
    for ancestor in ancestors:
      if ancestor in curated_tissues_set:
        return ancestor
  except:
    pass

  # Tier 3: Use manual fallback mapping
  # Accept any tissue in FALLBACK_TISSUE_MAPPINGS, even if not in CZ slim
  # (these are tissues we have constraints for)
  if tissue_id in FALLBACK_TISSUE_MAPPINGS:
    mapped_tissue = FALLBACK_TISSUE_MAPPINGS[tissue_id]['tissue']
    # Return the mapped tissue (could be itself or different)
    return mapped_tissue

  # No mapping found
  return None


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

  return filtered_cell_types, filtered_codes, code_remapping, mapping_df


def parse_args():
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
    description='Evaluate trained MLP models on composable embedding datasets',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Evaluate on test set
  python scripts/evaluate_mlp.py \\
    --checkpoint /path/to/checkpoint.pt \\
    --config /path/to/config.json \\
    --data-dir /data/training_data \\
    --cell-counts /data/batch-jobs/cell_counts.csv

  # Evaluate with custom suffixes
  python scripts/evaluate_mlp.py \\
    --checkpoint /path/to/checkpoint.pt \\
    --config /path/to/config.json \\
    --data-dir /data/validation_data \\
    --cell-counts /data/cell_counts.csv \\
    --genept-suffix "_val_v2" \\
    --tissue-suffix "_val_v2" \\
    --metadata-suffix "_val_v2"
    """
  )

  # Required arguments
  parser.add_argument('--checkpoint', type=str, required=True,
                     help='Path to model checkpoint file (.pt format)')
  parser.add_argument('--config', type=str, required=True,
                     help='Path to config JSON file with model and data settings')
  parser.add_argument('--data-dir', type=str, required=True,
                     help='Base directory containing parquet files to evaluate on')
  parser.add_argument('--cell-counts', type=str, required=True,
                     help='Path to cell counts CSV file for cell type definitions')

  # Optional arguments
  parser.add_argument('--embedding-types', type=str, default=None,
                     help='Comma-separated list of embedding types (default: from config)')
  parser.add_argument('--genept-suffix', type=str, default=None,
                     help='Suffix for GenePT data directory (default: from config or "")')
  parser.add_argument('--tissue-suffix', type=str, default=None,
                     help='Suffix for tissue data directory (default: from config or "")')
  parser.add_argument('--metadata-suffix', type=str, default=None,
                     help='Suffix for metadata directory (default: from config or "")')
  parser.add_argument('--batch-size', type=int, default=4096,
                     help='Inference batch size (default: 4096)')
  parser.add_argument('--device', type=str, default=None,
                     help='Device to run on (default: cuda if available, else cpu)')
  parser.add_argument('--output-dir', type=str, default=None,
                     help='Directory to save results (default: checkpoint directory)')
  parser.add_argument('--save-predictions', action='store_true',
                     help='Save prediction outputs (default: False)')
  parser.add_argument('--cell-count-threshold', type=int, default=None,
                     help='Override minimum cell count threshold (default: from config)')
  parser.add_argument('--ontology-dir', type=str, default='data/ontology',
                     help='Cell Ontology cache directory (default: data/ontology)')
  parser.add_argument('--verbose', action='store_true',
                     help='Enable verbose output (default: False)')
  parser.add_argument('--skip-per-file', action='store_true',
                     help='Skip per-file metrics computation for faster evaluation')

  # Constrained output arguments
  parser.add_argument('--enable-constraints', action='store_true',
                     help='Enable tissue-based constrained output evaluation (default: False)')
  parser.add_argument('--constraint-mode', type=str, default='both',
                     choices=['allowlist', 'soft_prior', 'both'],
                     help='Constraint mode(s) to evaluate (default: both)')
  parser.add_argument('--constraints-dir', type=str, default='data/cellxgene_constraints',
                     help='Directory containing constraint data files (default: data/cellxgene_constraints)')
  parser.add_argument('--allowlist-path', type=str, default=None,
                     help='Override path to tissue_allowlists.json (default: None, uses constraints-dir)')
  parser.add_argument('--counts-path', type=str, default=None,
                     help='Override path to tissue_celltype_counts.json (default: None, uses constraints-dir)')
  parser.add_argument('--alpha', type=float, nargs='+', default=[0.5],
                     help='Alpha value(s) for soft prior strength (default: 0.5)')
  parser.add_argument('--alpha-sweep', action='store_true',
                     help='Enable alpha sweep mode, evaluates [0.0, 0.1, 0.2, ..., 1.0] (default: False)')
  parser.add_argument('--save-per-tissue-metrics', action='store_true',
                     help='Save detailed per-tissue metrics to CSV (default: False)')

  return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
  """Load configuration from JSON file."""
  config_path = Path(config_path)

  if not config_path.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")

  try:
    with open(config_path, 'r') as f:
      config = json.load(f)
  except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in config file: {e}")

  # Validate required fields
  required_fields = ['n_hidden_layers', 'dropout', 'genept_dims', 'embedding_types']
  missing = [f for f in required_fields if f not in config]
  if missing:
    raise ValueError(f"Config missing required fields: {missing}")

  return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
  """Merge config with CLI argument overrides."""
  merged = config.copy()

  # Override with CLI arguments if provided
  if args.embedding_types is not None:
    merged['embedding_types'] = [t.strip() for t in args.embedding_types.split(',')]

  if args.cell_count_threshold is not None:
    merged['cell_count_threshold'] = args.cell_count_threshold
  elif 'cell_count_threshold' not in merged:
    # Default if not in config or CLI
    merged['cell_count_threshold'] = 10000

  # Handle suffixes with priority: CLI > config > default empty string
  if args.genept_suffix is not None:
    merged['genept_suffix'] = args.genept_suffix
  elif 'genept_suffix' not in merged and 'test_genept_suffix' not in merged:
    merged['genept_suffix'] = ''
  elif 'test_genept_suffix' in merged and 'genept_suffix' not in merged:
    merged['genept_suffix'] = merged['test_genept_suffix']

  if args.tissue_suffix is not None:
    merged['tissue_suffix'] = args.tissue_suffix
  elif 'tissue_suffix' not in merged and 'test_tissue_suffix' not in merged:
    merged['tissue_suffix'] = ''
  elif 'test_tissue_suffix' in merged and 'tissue_suffix' not in merged:
    merged['tissue_suffix'] = merged['test_tissue_suffix']

  if args.metadata_suffix is not None:
    merged['metadata_suffix'] = args.metadata_suffix
  elif 'metadata_suffix' not in merged and 'test_metadata_suffix' not in merged:
    merged['metadata_suffix'] = ''
  elif 'test_metadata_suffix' in merged and 'metadata_suffix' not in merged:
    merged['metadata_suffix'] = merged['test_metadata_suffix']

  return merged


def compute_input_dim(config: Dict[str, Any]) -> int:
  """Compute input dimensions from config."""
  input_dim = 0
  embedding_types = config['embedding_types']

  if 'genept' in embedding_types:
    input_dim += config['genept_dims']

  if 'scgpt' in embedding_types:
    input_dim += 512  # scGPT fixed dimension

  # Metadata embeddings handled by ComposableDataset
  # Just count the embedding types that contribute to input

  return input_dim


def load_evaluation_data(
  data_dir: str,
  config: Dict[str, Any],
  code_remapping: Dict[int, int],
  cell_type_codes: pd.Series,
  verbose: bool = False,
  track_files: bool = True,
  extract_tissue_ids: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[list]]:
  """Load evaluation data using ComposableTrainingDataset.

  Returns:
    X_eval: Feature matrix
    y_eval: Labels
    tissue_ids: Tissue ontology IDs for each sample (if extract_tissue_ids=True), else None
    file_info: List of dicts with file-level info (if track_files=True), else None
               Each dict contains: {'filename', 'start_idx', 'end_idx', 'X', 'y', 'tissue_ids'}
  """

  if verbose:
    print(f"\nLoading evaluation data from {data_dir}...")
    print(f"  Embedding types: {config['embedding_types']}")
    print(f"  GenePT suffix: '{config.get('genept_suffix', '')}'")
    print(f"  Tissue suffix: '{config.get('tissue_suffix', '')}'")
    print(f"  Metadata suffix: '{config.get('metadata_suffix', '')}'")

  # Create dataset in test mode
  dataset = ComposableTrainingDataset(
    base_dir=Path(data_dir),
    embedding_types=config['embedding_types'],
    batch_size=1024,  # Fixed batch size for loading
    genept_dims=config['genept_dims'],
    code_remapping=code_remapping,
    track_invalid_embeddings=config.get('track_invalid_embeddings', True),
    seed=config.get('seed', 42),
    # Test mode parameters
    is_test_mode=True,
    test_genept_suffix=config.get('genept_suffix', ''),
    test_tissue_suffix=config.get('tissue_suffix', ''),
    test_metadata_suffix=config.get('metadata_suffix', ''),
    cell_type_codes=cell_type_codes,
    verbose=verbose
  )

  if verbose:
    print(f"  Found {len(dataset.file_list)} files to load")

  # Load all data into memory, tracking file boundaries if requested
  X_list = []
  y_list = []
  tissue_ids_list = [] if extract_tissue_ids else None
  file_info = [] if track_files else None
  current_idx = 0

  # Metadata directory for tissue ID extraction
  metadata_dir = None
  if extract_tissue_ids:
    metadata_suffix = config.get('metadata_suffix', '')
    metadata_dir = Path(data_dir) / f"cellxgene_v2{metadata_suffix}"
    if not metadata_dir.exists():
      print(f"Warning: Metadata directory not found: {metadata_dir}")
      print("  Tissue IDs will not be extracted")
      extract_tissue_ids = False
      tissue_ids_list = None

  # Load files one by one to track boundaries
  for file_path in dataset.file_list:
    # Create a dataset for just this file
    file_dataset = ComposableTrainingDataset(
      base_dir=Path(data_dir),
      embedding_types=config['embedding_types'],
      batch_size=100000,  # Large batch to get all file data at once
      genept_dims=config['genept_dims'],
      code_remapping=code_remapping,
      track_invalid_embeddings=config.get('track_invalid_embeddings', True) if not track_files else False,
      seed=config.get('seed', 42),
      is_test_mode=True,
      test_genept_suffix=config.get('genept_suffix', ''),
      test_tissue_suffix=config.get('tissue_suffix', ''),
      test_metadata_suffix=config.get('metadata_suffix', ''),
      cell_type_codes=cell_type_codes,
      verbose=False
    )
    file_dataset.file_list = [file_path]

    file_X_list = []
    file_y_list = []
    for X_batch, y_batch in file_dataset:
      file_X_list.append(X_batch.numpy())
      file_y_list.append(y_batch.numpy())

    if file_X_list:
      file_X = np.concatenate(file_X_list, axis=0).astype(np.float32)
      file_y = np.concatenate(file_y_list, axis=0).astype(np.int64)

      # Extract tissue IDs from metadata if requested
      file_tissue_ids = None
      if extract_tissue_ids and metadata_dir:
        filename = file_path.name if hasattr(file_path, 'name') else Path(file_path).name
        # Look for corresponding metadata file (add .parquet if not present)
        if not filename.endswith('.parquet'):
          filename = filename + '.parquet'
        metadata_file = metadata_dir / filename
        if metadata_file.exists():
          try:
            metadata_df = pd.read_parquet(metadata_file, columns=['tissue_ontology_term_id'])
            file_tissue_ids = metadata_df['tissue_ontology_term_id'].values

            # Map tissue IDs to CZ slim tissues for constraint lookup
            mapped_tissue_ids = []
            for tid in file_tissue_ids:
              if tid == 'unknown':
                mapped_tissue_ids.append('unknown')
              else:
                mapped_tid = map_tissue_to_cz_slim(tid)
                # If no mapping found, keep as 'unknown' for constraint purposes
                mapped_tissue_ids.append(mapped_tid if mapped_tid is not None else 'unknown')
            file_tissue_ids = np.array(mapped_tissue_ids)

            # Apply code_remapping filtering to align with labels
            # Tissue IDs should match the samples after filtering
            if len(file_tissue_ids) != len(file_y):
              # Tissue IDs need to be filtered to match processed labels
              # For now, slice to match (assumes same filtering applied)
              if len(file_tissue_ids) >= len(file_y):
                file_tissue_ids = file_tissue_ids[:len(file_y)]
              else:
                print(f"Warning: Tissue ID count mismatch for {filename}: {len(file_tissue_ids)} vs {len(file_y)}")
                file_tissue_ids = None
          except Exception as e:
            if verbose:
              print(f"Warning: Could not load tissue IDs from {metadata_file}: {e}")
            file_tissue_ids = None
        else:
          if verbose:
            print(f"Warning: Metadata file not found: {metadata_file}")

      # Add to overall lists
      X_list.append(file_X)
      y_list.append(file_y)
      if extract_tissue_ids:
        if file_tissue_ids is not None:
          tissue_ids_list.append(file_tissue_ids)
        else:
          # Fill with None or placeholder if tissue IDs not available for this file
          tissue_ids_list.append(np.array(['unknown'] * len(file_y)))

      # Track file info if requested
      if track_files:
        end_idx = current_idx + len(file_X)
        filename = file_path.name if hasattr(file_path, 'name') else Path(file_path).name
        file_info.append({
          'filename': filename,
          'start_idx': current_idx,
          'end_idx': end_idx,
          'X': file_X,
          'y': file_y,
          'tissue_ids': file_tissue_ids
        })
        current_idx = end_idx

  if not X_list:
    raise ValueError("No evaluation data loaded. Check data directory and suffixes.")

  # Combine all batches
  X_eval = np.concatenate(X_list, axis=0).astype(np.float32)
  y_eval = np.concatenate(y_list, axis=0).astype(np.int64)
  tissue_ids_eval = None
  if extract_tissue_ids and tissue_ids_list:
    tissue_ids_eval = np.concatenate(tissue_ids_list, axis=0)

  if verbose:
    print(f"  Loaded {len(X_eval):,} samples")
    print(f"  Input dimensions: {X_eval.shape[1]}")
    print(f"  Unique classes: {len(np.unique(y_eval))}")
    if extract_tissue_ids and tissue_ids_eval is not None:
      unique_tissues = np.unique(tissue_ids_eval)
      print(f"  Unique tissues: {len(unique_tissues)}")
    if track_files and file_info:
      print(f"  Tracked {len(file_info)} files for per-file metrics")

  return X_eval, y_eval, tissue_ids_eval, file_info


def format_metrics_output(
  metrics: Dict[str, float],
  X_eval: np.ndarray,
  y_eval: np.ndarray,
  config: Dict[str, Any],
  num_classes: int,
  checkpoint_path: str,
  config_path: str,
  data_dir: str
) -> str:
  """Format metrics for console output."""

  output = []
  output.append("\n" + "="*80)
  output.append("Evaluation Results")
  output.append("="*80)
  output.append(f"Checkpoint: {checkpoint_path}")
  output.append(f"Config:     {config_path}")
  output.append(f"Data:       {data_dir}")
  output.append(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

  output.append("\nDataset Statistics:")
  output.append("-" * 80)
  output.append(f"Total samples:        {len(X_eval):,}")
  output.append(f"Number of classes:    {num_classes}")
  output.append(f"Input dimensions:     {X_eval.shape[1]}")

  output.append("\nModel Configuration:")
  output.append("-" * 80)
  output.append(f"Hidden layers:        {config['n_hidden_layers']}")
  output.append(f"Dropout:              {config['dropout']:.3f}")
  output.append(f"GenePT dims:          {config['genept_dims']}")
  output.append(f"Embedding types:      {', '.join(config['embedding_types'])}")
  output.append(f"Output classes:       {num_classes}")

  output.append("\nEvaluation Metrics:")
  output.append("-" * 80)

  # Standard metrics
  if 'logloss' in metrics:
    output.append(f"logloss:              {metrics['logloss']:.4f}")
  if 'accuracy' in metrics:
    output.append(f"accuracy:             {metrics['accuracy']:.4f}")
  if 'macro_f1' in metrics:
    output.append(f"macro_f1:             {metrics['macro_f1']:.4f}")
  if 'macro_precision' in metrics:
    output.append(f"macro_precision:      {metrics['macro_precision']:.4f}")
  if 'macro_recall' in metrics:
    output.append(f"macro_recall:         {metrics['macro_recall']:.4f}")
  if 'weighted_f1' in metrics:
    output.append(f"weighted_f1:          {metrics['weighted_f1']:.4f}")
  if 'weighted_precision' in metrics:
    output.append(f"weighted_precision:   {metrics['weighted_precision']:.4f}")
  if 'weighted_recall' in metrics:
    output.append(f"weighted_recall:      {metrics['weighted_recall']:.4f}")

  # Ranking metrics
  ranking_keys = [k for k in metrics.keys() if 'recall_at' in k or 'mrr_at' in k or 'dcg_at' in k]
  if ranking_keys:
    output.append("\nRanking Metrics:")
    output.append("-" * 80)
    for k in sorted(ranking_keys):
      output.append(f"{k:20s}  {metrics[k]:.4f}")

  # Hierarchical metrics
  hierarchical_keys = [k for k in metrics.keys() if 'hierarchical' in k]
  if hierarchical_keys:
    output.append("\nHierarchical Metrics (Cell Ontology):")
    output.append("-" * 80)
    for k in sorted(hierarchical_keys):
      output.append(f"{k:25s} {metrics[k]:.4f}")

  output.append("="*80 + "\n")

  return "\n".join(output)


def compute_per_file_metrics(
  model: torch.nn.Module,
  file_info: list,
  cell_types: list,
  cell_type_to_idx: Dict[str, int],
  ontology_graph: Optional[Any],  # nx.DiGraph when available
  batch_size: int,
  device: torch.device
) -> pd.DataFrame:
  """Compute metrics for each file separately.

  Args:
    model: Trained model
    file_info: List of dicts with file data
    cell_types: List of cell type names
    cell_type_to_idx: Mapping from cell types to indices
    ontology_graph: Cell ontology graph (optional)
    batch_size: Batch size for inference
    device: Device for inference

  Returns:
    DataFrame with per-file metrics
  """
  from src.training.metrics import evaluate_and_return_predictions, evaluate_with_hierarchy

  per_file_results = []

  for file_data in file_info:
    filename = file_data['filename']
    X_file = file_data['X']
    y_file = file_data['y']

    # Skip empty files
    if len(X_file) == 0:
      continue

    # Compute metrics for this file
    if ontology_graph is not None:
      file_metrics, y_true_file, _, y_pred_file = evaluate_with_hierarchy(
        model=model,
        X=X_file,
        y=y_file,
        cell_types=cell_types,
        cell_type_to_idx=cell_type_to_idx,
        ontology_graph=ontology_graph,
        batch_size=batch_size,
        device=device
      )
    else:
      file_metrics, y_true_file, _, y_pred_file = evaluate_and_return_predictions(
        model=model,
        X=X_file,
        y=y_file,
        num_classes=len(cell_types),
        batch_size=batch_size,
        device=device
      )

    # Compute accuracy manually since it's not in the metrics
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true_file, y_pred_file)

    # Extract key metrics
    result = {
      'filename': filename,
      'samples': len(X_file),
      'logloss': file_metrics.get('logloss', np.nan),
      'accuracy': accuracy,
      'macro_f1': file_metrics.get('macro_f1', np.nan),
      'recall_at_2': file_metrics.get('recall_at_2', np.nan),
      'recall_at_5': file_metrics.get('recall_at_5', np.nan),
      'recall_at_10': file_metrics.get('recall_at_10', np.nan),
    }

    # Add hierarchical metrics if available
    if 'hierarchical_f1' in file_metrics:
      result['hierarchical_f1'] = file_metrics['hierarchical_f1']

    per_file_results.append(result)

  return pd.DataFrame(per_file_results)


def format_per_file_metrics(per_file_df: pd.DataFrame) -> str:
  """Format per-file metrics for console output.

  Args:
    per_file_df: DataFrame with per-file metrics

  Returns:
    Formatted string for console output
  """
  if per_file_df.empty:
    return ""

  output = []
  output.append("\nPer-File Metrics:")
  output.append("-" * 120)

  # Determine columns to display
  has_hierarchical = 'hierarchical_f1' in per_file_df.columns

  # Create header
  if has_hierarchical:
    header = f"{'File':<45} {'Samples':>8}  {'Logloss':>8}  {'Accuracy':>8}  {'Macro F1':>8}  {'Hier. F1':>8}  {'Recall@2':>8}  {'Recall@5':>8}  {'Recall@10':>9}"
  else:
    header = f"{'File':<45} {'Samples':>8}  {'Logloss':>8}  {'Accuracy':>8}  {'Macro F1':>8}  {'Recall@2':>8}  {'Recall@5':>8}  {'Recall@10':>9}"

  output.append(header)
  output.append("-" * 120)

  # Add rows
  for _, row in per_file_df.iterrows():
    filename = row['filename'][:44]  # Truncate long names
    if has_hierarchical:
      line = f"{filename:<45} {row['samples']:>8,}  {row['logloss']:>8.3f}  {row['accuracy']:>8.3f}  {row['macro_f1']:>8.3f}  {row['hierarchical_f1']:>8.3f}  {row['recall_at_2']:>8.3f}  {row['recall_at_5']:>8.3f}  {row['recall_at_10']:>9.3f}"
    else:
      line = f"{filename:<45} {row['samples']:>8,}  {row['logloss']:>8.3f}  {row['accuracy']:>8.3f}  {row['macro_f1']:>8.3f}  {row['recall_at_2']:>8.3f}  {row['recall_at_5']:>8.3f}  {row['recall_at_10']:>9.3f}"
    output.append(line)

  # Add summary statistics
  output.append("")
  output.append("Summary Statistics Across Files:")
  output.append("-" * 120)

  metrics_to_summarize = ['logloss', 'accuracy', 'macro_f1', 'recall_at_2', 'recall_at_10']
  if has_hierarchical:
    metrics_to_summarize.insert(3, 'hierarchical_f1')

  for metric in metrics_to_summarize:
    if metric in per_file_df.columns:
      mean_val = per_file_df[metric].mean()
      std_val = per_file_df[metric].std()
      min_val = per_file_df[metric].min()
      max_val = per_file_df[metric].max()

      metric_label = metric.replace('_', ' ').title().replace('F1', 'F1').replace('At', '@')
      output.append(f"{metric_label:18s} mean={mean_val:.3f}  std={std_val:.3f}  min={min_val:.3f}  max={max_val:.3f}")

  return "\n".join(output)


class FilteredTissueConstraints:
  """Tissue-based constraints with dynamic vocabulary filtering.

  Loads constraint data from JSON files and filters to match the model's
  vocabulary, enabling the same constraint files to work with any cell_count_threshold.
  """

  def __init__(
    self,
    allowlist_path: str,
    counts_path: str,
    code_remapping: dict,
    cell_type_to_idx: dict,
    num_classes: int,
    device: torch.device
  ):
    """Initialize constraints with dynamic filtering.

    Args:
      allowlist_path: Path to tissue_allowlists.json
      counts_path: Path to tissue_celltype_counts.json
      code_remapping: Maps full vocab indices (832) to filtered indices or -100
      cell_type_to_idx: Maps cell type names to filtered indices
      num_classes: Number of classes in filtered vocabulary
      device: Torch device
    """
    import json

    self.num_classes = num_classes
    self.device = device
    self.tissue_to_allowlist = {}
    self.tissue_to_logp = {}

    # Load and filter allowlists
    if allowlist_path and Path(allowlist_path).exists():
      with open(allowlist_path) as f:
        tissue_allowlists_full = json.load(f)

      for tissue_id, full_indices in tissue_allowlists_full.items():
        # Filter to model's vocabulary
        filtered_indices = [
          code_remapping[idx] for idx in full_indices
          if idx in code_remapping and code_remapping[idx] != -100
        ]
        if filtered_indices:
          self.tissue_to_allowlist[tissue_id] = filtered_indices

    # Load counts and compute soft priors
    if counts_path and Path(counts_path).exists():
      with open(counts_path) as f:
        counts_data = json.load(f)
        tissue_counts = counts_data.get('counts', {})

      for tissue_id, cell_type_counts in tissue_counts.items():
        # Map cell type names to filtered indices
        filtered_counts = {}
        for cell_type_name, count in cell_type_counts.items():
          if cell_type_name in cell_type_to_idx:
            filtered_idx = cell_type_to_idx[cell_type_name]
            filtered_counts[filtered_idx] = count

        if filtered_counts:
          # Compute log probabilities
          logp = self._counts_to_logp(filtered_counts)
          self.tissue_to_logp[tissue_id] = logp.to(device)

  def _counts_to_logp(self, counts_dict: dict) -> torch.Tensor:
    """Convert counts dict to log probabilities tensor.

    Args:
      counts_dict: {class_idx: count}

    Returns:
      logp: [num_classes] log probability tensor
    """
    counts = torch.zeros(self.num_classes, dtype=torch.float32)
    for idx, count in counts_dict.items():
      counts[idx] = count

    # Normalize to probabilities and take log
    probs = counts / counts.sum()
    logp = torch.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
    return logp

  def apply_allowlist(self, logits: torch.Tensor, tissue_id: str) -> torch.Tensor:
    """Apply allowlist constraints to logits.

    Args:
      logits: [batch_size, num_classes] logits
      tissue_id: Tissue ontology ID

    Returns:
      constrained_logits: [batch_size, num_classes] with disallowed classes set to -1e9
    """
    if tissue_id not in self.tissue_to_allowlist:
      return logits

    constrained_logits = logits.clone()
    allowlist = self.tissue_to_allowlist[tissue_id]

    # Create mask: all False, then set allowed indices to True
    mask = torch.zeros(self.num_classes, dtype=torch.bool, device=logits.device)
    mask[allowlist] = True

    # Set disallowed logits to -1e9
    constrained_logits[:, ~mask] = -1e9

    return constrained_logits

  def apply_soft_prior(
    self,
    logits: torch.Tensor,
    tissue_id: str,
    alpha: float
  ) -> torch.Tensor:
    """Apply soft prior constraints to logits.

    Args:
      logits: [batch_size, num_classes] logits
      tissue_id: Tissue ontology ID
      alpha: Prior strength (0 = no prior, 1 = equal weight to logits)

    Returns:
      biased_logits: [batch_size, num_classes] with soft prior added
    """
    if tissue_id not in self.tissue_to_logp:
      return logits

    logp = self.tissue_to_logp[tissue_id]
    # Broadcast and add: [batch_size, num_classes] + [num_classes]
    biased_logits = logits + alpha * logp

    return biased_logits


def evaluate_with_constraints(
  model: torch.nn.Module,
  X: np.ndarray,
  y: np.ndarray,
  tissue_ids: np.ndarray,
  file_info: list,
  constraints: Any,
  constraint_mode: str,
  alpha_values: list,
  cell_types: list,
  cell_type_to_idx: dict,
  ontology_graph: Any,
  batch_size: int,
  device: torch.device,
  verbose: bool = False
) -> dict:
  """Evaluate model with tissue-based constraints.

  Args:
    model: Trained model
    X: Feature matrix [N, D]
    y: True labels [N]
    tissue_ids: Tissue ontology IDs [N]
    file_info: List of file metadata dicts with tissue_ids
    constraints: CellxGeneTissueConstraints object
    constraint_mode: 'allowlist', 'soft_prior', or 'both'
    alpha_values: List of alpha values for soft prior
    cell_types: List of cell type names
    cell_type_to_idx: Dict mapping cell type to index
    ontology_graph: Cell Ontology graph (optional)
    batch_size: Inference batch size
    device: Torch device
    verbose: Enable verbose output

  Returns:
    results: Dict with keys 'allowlist', 'soft_prior', 'alpha_sweep', 'statistics'
  """
  from sklearn.metrics import accuracy_score
  from src.training.metrics import evaluate_with_hierarchy, evaluate_and_return_predictions

  results = {}
  model.eval()

  # Process file by file (assuming files are tissue-homogeneous)
  file_results_by_mode = {
    'allowlist': [],
    'soft_prior': {alpha: [] for alpha in alpha_values}
  }

  for file_dict in file_info:
    file_X = file_dict['X']
    file_y = file_dict['y']
    file_tissue_ids = file_dict.get('tissue_ids')

    if file_tissue_ids is None or len(file_tissue_ids) == 0:
      if verbose:
        print(f"Warning: No tissue IDs for file {file_dict['filename']}, skipping constraints")
      continue

    # Determine dominant tissue for this file
    unique_tissues, counts = np.unique(file_tissue_ids, return_counts=True)
    dominant_tissue = unique_tissues[np.argmax(counts)]

    if dominant_tissue == 'unknown':
      continue

    # Check if tissue has constraints
    has_allowlist = dominant_tissue in constraints.tissue_to_allowlist
    has_prior = dominant_tissue in constraints.tissue_to_logp

    # Process file in batches
    file_X_tensor = torch.tensor(file_X, dtype=torch.float32)
    num_samples = len(file_X)

    # Allowlist mode
    if constraint_mode in ['allowlist', 'both'] and has_allowlist:
      file_preds_allowlist = []
      for i in range(0, num_samples, batch_size):
        batch_X = file_X_tensor[i:i+batch_size].to(device)
        with torch.no_grad():
          logits = model(batch_X)
          # Apply allowlist
          constrained_logits = constraints.apply_allowlist(logits, dominant_tissue)
          preds = constrained_logits.argmax(dim=-1).cpu().numpy()
          file_preds_allowlist.append(preds)

      file_preds_allowlist = np.concatenate(file_preds_allowlist)
      file_results_by_mode['allowlist'].append({
        'filename': file_dict['filename'],
        'tissue': dominant_tissue,
        'y_true': file_y,
        'y_pred': file_preds_allowlist
      })

    # Soft prior mode
    if constraint_mode in ['soft_prior', 'both'] and has_prior:
      for alpha in alpha_values:
        file_preds_soft = []
        for i in range(0, num_samples, batch_size):
          batch_X = file_X_tensor[i:i+batch_size].to(device)
          with torch.no_grad():
            logits = model(batch_X)
            # Apply soft prior
            constrained_logits = constraints.apply_soft_prior(logits, dominant_tissue, alpha)
            preds = constrained_logits.argmax(dim=-1).cpu().numpy()
            file_preds_soft.append(preds)

        file_preds_soft = np.concatenate(file_preds_soft)
        file_results_by_mode['soft_prior'][alpha].append({
          'filename': file_dict['filename'],
          'tissue': dominant_tissue,
          'y_true': file_y,
          'y_pred': file_preds_soft
        })

  # Aggregate results across files
  from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
  from scipy.special import softmax

  # Helper function to compute metrics from predictions
  def compute_metrics_from_predictions(y_true, y_pred, num_classes, cell_types_list=None, ontology=None):
    """Compute metrics from predictions.

    Args:
      y_true: True labels
      y_pred: Predicted labels
      num_classes: Number of classes
      cell_types_list: List of cell type names (optional, for hierarchical metrics)
      ontology: Cell ontology graph (optional, for hierarchical metrics)

    Returns:
      Dictionary of metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Compute hierarchical metrics if ontology available
    if ontology is not None and cell_types_list is not None:
      from src.training.hierarchical_metrics import calculate_hierarchical_f_score
      # Convert indices to labels
      y_true_labels = [cell_types_list[idx] for idx in y_true]
      y_pred_labels = [cell_types_list[idx] for idx in y_pred]
      # Calculate hierarchical metrics
      hier_metrics = calculate_hierarchical_f_score(y_true_labels, y_pred_labels, ontology)
      metrics.update(hier_metrics)

    return metrics

  # Allowlist results
  if file_results_by_mode['allowlist']:
    y_true_all = np.concatenate([r['y_true'] for r in file_results_by_mode['allowlist']])
    y_pred_all = np.concatenate([r['y_pred'] for r in file_results_by_mode['allowlist']])

    metrics_allowlist = compute_metrics_from_predictions(
      y_true_all, y_pred_all, len(cell_types),
      cell_types_list=cell_types,
      ontology=ontology_graph
    )

    results['allowlist'] = {
      'metrics': metrics_allowlist,
      'y_true': y_true_all,
      'y_pred': y_pred_all,
      'num_samples': len(y_true_all)
    }

  # Soft prior results
  soft_prior_results = {}
  for alpha in alpha_values:
    if file_results_by_mode['soft_prior'][alpha]:
      y_true_all = np.concatenate([r['y_true'] for r in file_results_by_mode['soft_prior'][alpha]])
      y_pred_all = np.concatenate([r['y_pred'] for r in file_results_by_mode['soft_prior'][alpha]])

      metrics_soft = compute_metrics_from_predictions(
        y_true_all, y_pred_all, len(cell_types),
        cell_types_list=cell_types,
        ontology=ontology_graph
      )

      soft_prior_results[alpha] = {
        'metrics': metrics_soft,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'num_samples': len(y_true_all)
      }

  if soft_prior_results:
    results['soft_prior'] = soft_prior_results

  return results


def save_results(
  output_dir: Path,
  metrics: Dict[str, float],
  summary_text: str,
  y_true: np.ndarray,
  y_pred: np.ndarray,
  all_preds: np.ndarray,
  cell_types: list,
  save_predictions: bool,
  per_file_df: Optional[pd.DataFrame] = None
):
  """Save evaluation results to files."""

  output_dir.mkdir(parents=True, exist_ok=True)

  # Prepare metrics JSON with per-file metrics if available
  metrics_output = metrics.copy()
  if per_file_df is not None and not per_file_df.empty:
    # Convert per-file dataframe to list of dicts for JSON
    per_file_metrics = per_file_df.to_dict('records')
    metrics_output['per_file_metrics'] = per_file_metrics

    # Add summary statistics
    summary_stats = {}
    for col in ['logloss', 'accuracy', 'macro_f1', 'hierarchical_f1', 'recall_at_2', 'recall_at_5', 'recall_at_10']:
      if col in per_file_df.columns:
        summary_stats[col] = {
          'mean': float(per_file_df[col].mean()),
          'std': float(per_file_df[col].std()),
          'min': float(per_file_df[col].min()),
          'max': float(per_file_df[col].max())
        }
    metrics_output['per_file_summary_stats'] = summary_stats

  # Save metrics JSON
  metrics_file = output_dir / 'evaluation_results.json'
  with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)
  print(f"Saved metrics to: {metrics_file}")

  # Save summary text
  summary_file = output_dir / 'evaluation_summary.txt'
  with open(summary_file, 'w') as f:
    f.write(summary_text)
  print(f"Saved summary to: {summary_file}")

  # Save class distribution
  unique, counts = np.unique(y_true, return_counts=True)
  dist_df = pd.DataFrame({
    'class_id': unique,
    'cell_type': [cell_types[i] if i < len(cell_types) else f"class_{i}" for i in unique],
    'sample_count': counts
  })
  dist_file = output_dir / 'class_distribution.csv'
  dist_df.to_csv(dist_file, index=False)
  print(f"Saved class distribution to: {dist_file}")

  # Save predictions if requested
  if save_predictions:
    pred_file = output_dir / 'predictions.npz'
    np.savez_compressed(
      pred_file,
      y_true=y_true,
      y_pred=y_pred,
      all_preds=all_preds
    )
    print(f"Saved predictions to: {pred_file}")


def main():
  """Main evaluation function."""
  args = parse_args()

  # Validate files exist
  checkpoint_path = Path(args.checkpoint)
  if not checkpoint_path.exists():
    print(f"Error: Checkpoint file not found: {checkpoint_path}")
    sys.exit(1)

  config_path = Path(args.config)
  if not config_path.exists():
    print(f"Error: Config file not found: {config_path}")
    sys.exit(1)

  cell_counts_path = Path(args.cell_counts)
  if not cell_counts_path.exists():
    print(f"Error: Cell counts file not found: {cell_counts_path}")
    sys.exit(1)

  data_dir = Path(args.data_dir)
  if not data_dir.exists():
    print(f"Error: Data directory not found: {data_dir}")
    sys.exit(1)

  # Set device
  if args.device:
    device = torch.device(args.device)
  else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if args.verbose:
    print(f"Using device: {device}")

  # Load and merge config
  if args.verbose:
    print(f"\nLoading config from {config_path}...")

  config = load_config(args.config)
  config = merge_config_with_args(config, args)

  if args.verbose:
    print(f"  Cell count threshold: {config['cell_count_threshold']}")
    print(f"  Embedding types: {config['embedding_types']}")

  # Load cell types and create code remapping
  if args.verbose:
    print(f"\nLoading cell types from {cell_counts_path}...")

  cell_types, cell_type_codes = load_cell_types_from_counts(cell_counts_path)

  filtered_cell_types, filtered_codes, code_remapping, mapping_df = create_code_remapping(
    cell_types,
    cell_type_codes,
    cell_counts_path,
    config['cell_count_threshold']
  )

  num_classes = len(filtered_cell_types)

  if args.verbose:
    print(f"  Total cell types: {len(cell_types)}")
    print(f"  Filtered cell types (>= {config['cell_count_threshold']} samples): {num_classes}")

  # Compute input dimensions
  input_dim = compute_input_dim(config)

  if args.verbose:
    print(f"\nModel architecture:")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Hidden layers: {config['n_hidden_layers']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Output classes: {num_classes}")

  # Load evaluation data
  compute_per_file = not args.skip_per_file
  X_eval, y_eval, tissue_ids_eval, file_info = load_evaluation_data(
    str(data_dir),
    config,
    code_remapping,
    filtered_codes,
    verbose=args.verbose,
    track_files=compute_per_file,
    extract_tissue_ids=args.enable_constraints
  )

  # Validate dimensions
  if X_eval.shape[1] != input_dim:
    print(f"Warning: Data dimensions ({X_eval.shape[1]}) don't match expected input ({input_dim})")
    print(f"Using data dimensions: {X_eval.shape[1]}")
    input_dim = X_eval.shape[1]

  # Initialize model
  if args.verbose:
    print(f"\nInitializing model...")

  model = MLPClassifier(
    input_dim=input_dim,
    num_classes=num_classes,
    n_hidden_layers=config['n_hidden_layers'],
    dropout=config['dropout']
  )

  # Load checkpoint
  if args.verbose:
    print(f"Loading checkpoint from {checkpoint_path}...")

  checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

  if 'model_state_dict' not in checkpoint:
    print("Error: Checkpoint does not contain 'model_state_dict'")
    sys.exit(1)

  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(device)
  model.eval()

  if args.verbose:
    print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

  # Create cell type to index mapping
  cell_type_to_idx = {ct: i for i, ct in enumerate(filtered_cell_types)}

  # Load ontology for hierarchical metrics
  ontology_dir = Path(args.ontology_dir)
  ontology_graph = None
  enable_hierarchical = False

  if ontology_dir.exists():
    try:
      # Import here to avoid errors if ontology not available
      from src.training.ontology import CellOntologyManager

      if args.verbose:
        print(f"\nLoading Cell Ontology from {ontology_dir}...")

      ontology_manager = CellOntologyManager(ontology_dir)
      ontology_graph = ontology_manager.build_cell_type_graph()

      if ontology_graph is not None and len(ontology_graph.nodes) > 0:
        enable_hierarchical = True
        if args.verbose:
          print(f"  Loaded ontology with {len(ontology_graph.nodes)} cell types")
      else:
        print(f"\nWarning: Ontology graph is empty")
        print("Skipping hierarchical metrics...")
    except Exception as e:
      print(f"\nWarning: Could not load Cell Ontology: {e}")
      print("Skipping hierarchical metrics...")
  else:
    if args.verbose:
      print(f"\nWarning: Ontology directory not found: {ontology_dir}")
      print("Skipping hierarchical metrics...")

  # Load constraints if enabled
  constraints = None
  alpha_values = args.alpha
  if args.enable_constraints:
    if args.verbose:
      print(f"\nLoading tissue-based constraints...")

    # Handle alpha sweep
    if args.alpha_sweep:
      alpha_values = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
      if args.verbose:
        print(f"  Alpha sweep mode enabled: {alpha_values}")
    elif args.verbose:
      print(f"  Alpha value(s): {alpha_values}")

    try:
      # Build constraint paths
      constraints_dir = Path(args.constraints_dir)
      allowlist_path = args.allowlist_path or (constraints_dir / "tissue_allowlists.json")
      counts_path = args.counts_path or (constraints_dir / "tissue_celltype_counts.json")

      # Check if files exist
      allowlist_exists = allowlist_path.exists() if allowlist_path else False
      counts_exists = counts_path.exists() if counts_path else False

      if not allowlist_exists and not counts_exists:
        print(f"Warning: No constraint files found in {constraints_dir}")
        print("  Expected: tissue_allowlists.json and/or tissue_celltype_counts.json")
        constraints = None
      else:
        if not allowlist_exists:
          print(f"Warning: Allowlist file not found: {allowlist_path}")
          allowlist_path = None
        if not counts_exists:
          print(f"Warning: Counts file not found: {counts_path}")
          counts_path = None

        # Create FilteredTissueConstraints with dynamic vocabulary filtering
        constraints = FilteredTissueConstraints(
          allowlist_path=str(allowlist_path) if allowlist_path else None,
          counts_path=str(counts_path) if counts_path else None,
          code_remapping=code_remapping,
          cell_type_to_idx=cell_type_to_idx,
          num_classes=num_classes,
          device=device
        )

        if args.verbose:
          if allowlist_path:
            print(f"  Loaded allowlists: {len(constraints.tissue_to_allowlist)} tissues")
          if counts_path:
            print(f"  Loaded soft priors: {len(constraints.tissue_to_logp)} tissues")
          if tissue_ids_eval is not None:
            unique_tissues = np.unique(tissue_ids_eval)
            unique_tissues = [t for t in unique_tissues if t != 'unknown']
            print(f"  Tissues in evaluation data: {len(unique_tissues)}")

    except Exception as e:
      print(f"Warning: Could not load constraints: {e}")
      print("Continuing with baseline evaluation only...")
      import traceback
      if args.verbose:
        traceback.print_exc()
      constraints = None

  # Run evaluation
  if args.verbose:
    print(f"\nRunning evaluation on {len(X_eval):,} samples...")

  # Use hierarchical evaluation if ontology available, otherwise use standard
  if enable_hierarchical:
    metrics, y_true, all_preds, y_pred = evaluate_with_hierarchy(
      model=model,
      X=X_eval,
      y=y_eval,
      cell_types=filtered_cell_types,
      cell_type_to_idx=cell_type_to_idx,
      ontology_graph=ontology_graph,
      batch_size=args.batch_size,
      device=device
    )
  else:
    metrics, y_true, all_preds, y_pred = evaluate_and_return_predictions(
      model=model,
      X=X_eval,
      y=y_eval,
      num_classes=num_classes,
      batch_size=args.batch_size,
      device=device
    )

  # Format and print results
  summary_text = format_metrics_output(
    metrics=metrics,
    X_eval=X_eval,
    y_eval=y_eval,
    config=config,
    num_classes=num_classes,
    checkpoint_path=str(checkpoint_path),
    config_path=str(config_path),
    data_dir=str(data_dir)
  )

  print(summary_text)

  # Run constrained evaluation if enabled
  constrained_results = None
  if constraints and file_info:
    if args.verbose:
      print(f"\nRunning constrained evaluation...")

    try:
      constrained_results = evaluate_with_constraints(
        model=model,
        X=X_eval,
        y=y_eval,
        tissue_ids=tissue_ids_eval,
        file_info=file_info,
        constraints=constraints,
        constraint_mode=args.constraint_mode,
        alpha_values=alpha_values,
        cell_types=filtered_cell_types,
        cell_type_to_idx=cell_type_to_idx,
        ontology_graph=ontology_graph if enable_hierarchical else None,
        batch_size=args.batch_size,
        device=device,
        verbose=args.verbose
      )

      # Print constrained results summary with baseline comparison
      if constrained_results:
        print("\n" + "="*80)
        print("Constrained Evaluation Results (Baseline vs Constrained)")
        print("="*80)

        # Get baseline metrics for comparison
        # Note: baseline may not have 'accuracy' from evaluate_with_hierarchy, compute it
        from sklearn.metrics import accuracy_score
        baseline_accuracy = accuracy_score(y_true, y_pred)
        baseline_macro_f1 = metrics.get('macro_f1', 0.0)

        # Create comparison table
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(f"{'Mode':<25} {'Samples':>12} {'Accuracy':>12} {'Macro F1':>12} {'Δ Acc':>10} {'Δ F1':>10}")
        print("-" * 80)

        # Baseline row
        print(f"{'Baseline (unconstrained)':<25} {len(y_eval):>12,} {baseline_accuracy:>12.4f} {baseline_macro_f1:>12.4f} {'-':>10} {'-':>10}")

        # Allowlist results
        if 'allowlist' in constrained_results:
          allowlist_metrics = constrained_results['allowlist']['metrics']
          allowlist_acc = allowlist_metrics.get('accuracy', 0.0)
          allowlist_f1 = allowlist_metrics.get('macro_f1', 0.0)
          delta_acc = allowlist_acc - baseline_accuracy
          delta_f1 = allowlist_f1 - baseline_macro_f1
          num_samples = constrained_results['allowlist']['num_samples']
          print(f"{'Allowlist (hard)':<25} {num_samples:>12,} {allowlist_acc:>12.4f} {allowlist_f1:>12.4f} {delta_acc:>+10.4f} {delta_f1:>+10.4f}")

        # Soft prior results
        if 'soft_prior' in constrained_results:
          for alpha, result in sorted(constrained_results['soft_prior'].items()):
            soft_metrics = result['metrics']
            soft_acc = soft_metrics.get('accuracy', 0.0)
            soft_f1 = soft_metrics.get('macro_f1', 0.0)
            delta_acc = soft_acc - baseline_accuracy
            delta_f1 = soft_f1 - baseline_macro_f1
            num_samples = result['num_samples']
            print(f"{'Soft prior (α=' + str(alpha) + ')':<25} {num_samples:>12,} {soft_acc:>12.4f} {soft_f1:>12.4f} {delta_acc:>+10.4f} {delta_f1:>+10.4f}")

        print("-" * 80)

        # Additional metrics if hierarchical
        if enable_hierarchical and 'hierarchical_f1' in metrics:
          print("\nHierarchical Metrics:")
          print("-" * 60)
          print(f"{'Mode':<25} {'Hier. F1':>12} {'Hier. Prec':>12} {'Hier. Rec':>12}")
          print("-" * 60)

          baseline_hier_f1 = metrics.get('hierarchical_f1', 0.0)
          baseline_hier_prec = metrics.get('hierarchical_precision', 0.0)
          baseline_hier_rec = metrics.get('hierarchical_recall', 0.0)
          print(f"{'Baseline':<25} {baseline_hier_f1:>12.4f} {baseline_hier_prec:>12.4f} {baseline_hier_rec:>12.4f}")

          if 'allowlist' in constrained_results:
            allowlist_metrics = constrained_results['allowlist']['metrics']
            print(f"{'Allowlist':<25} {allowlist_metrics.get('hierarchical_f1', 0.0):>12.4f} {allowlist_metrics.get('hierarchical_precision', 0.0):>12.4f} {allowlist_metrics.get('hierarchical_recall', 0.0):>12.4f}")

          if 'soft_prior' in constrained_results:
            for alpha, result in sorted(constrained_results['soft_prior'].items()):
              soft_metrics = result['metrics']
              print(f"{'Soft prior (α=' + str(alpha) + ')':<25} {soft_metrics.get('hierarchical_f1', 0.0):>12.4f} {soft_metrics.get('hierarchical_precision', 0.0):>12.4f} {soft_metrics.get('hierarchical_recall', 0.0):>12.4f}")

          print("-" * 60)

    except Exception as e:
      print(f"Warning: Constrained evaluation failed: {e}")
      import traceback
      traceback.print_exc()
      constrained_results = None

  # Compute and display per-file metrics if requested
  per_file_df = None
  per_file_output = ""
  if compute_per_file and file_info:
    if args.verbose:
      print(f"\nComputing per-file metrics for {len(file_info)} files...")

    per_file_df = compute_per_file_metrics(
      model=model,
      file_info=file_info,
      cell_types=filtered_cell_types,
      cell_type_to_idx=cell_type_to_idx,
      ontology_graph=ontology_graph if enable_hierarchical else None,
      batch_size=args.batch_size,
      device=device
    )

    # Print per-file metrics
    per_file_output = format_per_file_metrics(per_file_df)
    if per_file_output:
      print(per_file_output)

  # Combine summary with per-file output
  full_summary = summary_text
  if per_file_output:
    full_summary = summary_text + per_file_output

  # Save results
  if args.output_dir:
    output_dir = Path(args.output_dir)
  else:
    output_dir = checkpoint_path.parent

  save_results(
    output_dir=output_dir,
    metrics=metrics,
    summary_text=full_summary,
    y_true=y_true,
    y_pred=y_pred,
    all_preds=all_preds,
    cell_types=filtered_cell_types,
    save_predictions=args.save_predictions,
    per_file_df=per_file_df
  )

  print(f"\nEvaluation complete!")


if __name__ == '__main__':
  main()
