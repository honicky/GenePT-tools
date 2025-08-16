#!/usr/bin/env python3
"""
Extract subsets of AnnData files based on observation_joinid values from parquet files.

This script takes .h5ad files from data/cellxgene/test_v1 and corresponding 
.parquet files from data/cellxgene_embeddings/test_v1, and creates new AnnData 
files containing only the rows that have embeddings in the parquet files.
"""

import os
import pandas as pd
import anndata as ad
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_anndata_subset(h5ad_path, parquet_path, output_path):
  """
  Extract subset of AnnData file based on observation_joinid values from parquet file.
  
  Args:
    h5ad_path: Path to the input .h5ad file
    parquet_path: Path to the corresponding .parquet file with embeddings
    output_path: Path where to save the extracted subset
  """
  logger.info(f"Processing {h5ad_path.name}")
  
  # Load parquet file to get observation_joinid values
  logger.info("Loading parquet file to get observation_joinid values...")
  df_parquet = pd.read_parquet(parquet_path)
  target_joinids = set(df_parquet['observation_joinid'])
  logger.info(f"Found {len(target_joinids)} observation_joinid values in parquet file")
  
  # Load AnnData in backed mode to avoid loading entire file into memory
  logger.info("Loading AnnData file in backed mode...")
  adata = ad.read_h5ad(h5ad_path, backed='r')
  logger.info(f"AnnData shape: {adata.shape}")
  
  # Find indices of matching observation_joinid values
  logger.info("Finding matching observation_joinid values...")
  obs_joinids = adata.obs['observation_joinid']
  mask = obs_joinids.isin(target_joinids)
  matching_indices = mask[mask].index
  logger.info(f"Found {len(matching_indices)} matching rows out of {adata.shape[0]} total rows")
  
  # Extract the subset - only X, var, and obs
  logger.info("Extracting subset...")
  # Get the subset of observations
  subset_obs = adata.obs.loc[matching_indices].copy()
  subset_var = adata.var.copy()  # Keep all variables
  
  # Extract X data for the subset of observations
  # This is memory-efficient as we only load the needed rows
  subset_X = adata[matching_indices, :].X
  
  # Create new AnnData object with only the essential components
  logger.info("Creating new AnnData object...")
  adata_subset = ad.AnnData(
    X=subset_X,
    obs=subset_obs,
    var=subset_var
  )
  
  logger.info(f"Subset AnnData shape: {adata_subset.shape}")
  
  # Save the subset
  logger.info(f"Saving to {output_path}")
  adata_subset.write_h5ad(output_path)
  logger.info("Done!")

def main():
  """Main function to process all file pairs."""
  # Set up paths
  h5ad_dir = Path("data/cellxgene/test_v1")
  parquet_dir = Path("data/cellxgene_embeddings/test_v1")
  output_dir = Path("data/cellxgene_subsets")
  
  # Create output directory
  output_dir.mkdir(exist_ok=True)
  
  # Get list of .h5ad files (no extension)
  h5ad_files = [f for f in h5ad_dir.iterdir() if f.is_file()]
  
  logger.info(f"Found {len(h5ad_files)} .h5ad files to process")
  
  processed = 0
  errors = 0
  
  for h5ad_file in h5ad_files:
    # Construct corresponding parquet file path
    parquet_file = parquet_dir / f"{h5ad_file.name}.parquet"
    
    if not parquet_file.exists():
      logger.warning(f"No corresponding parquet file found for {h5ad_file.name}")
      continue
    
    # Construct output file path
    output_file = output_dir / f"{h5ad_file.name}_subset.h5ad"
    
    # Skip if output already exists
    if output_file.exists():
      logger.info(f"Output file {output_file.name} already exists, skipping...")
      continue
    
    try:
      extract_anndata_subset(h5ad_file, parquet_file, output_file)
      processed += 1
    except Exception as e:
      logger.error(f"Error processing {h5ad_file.name}: {str(e)}")
      errors += 1
      continue
  
  logger.info(f"Processing complete! Successfully processed {processed} files, {errors} errors")

if __name__ == "__main__":
  main() 