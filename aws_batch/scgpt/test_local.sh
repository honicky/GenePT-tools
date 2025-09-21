#!/bin/bash
# Quick test script for scGPT on EC2 with local files

set -e

echo "=== Testing scGPT Locally on EC2 ==="
echo

# Check for GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo

# Create test directory
TEST_DIR="/tmp/scgpt_test"
mkdir -p $TEST_DIR
cd $TEST_DIR

echo "1. Downloading a sample H5AD file..."
# Download a small test file (you can replace with your own)
if [ ! -f "test.h5ad" ]; then
    # Using a public dataset for testing
    wget -O test.h5ad "https://datasets.cellxgene.cziscience.com/default/b4645fc5-f793-4207-9ad2-93b6c5d2c07b.h5ad" \
        || echo "Could not download test file, please provide your own H5AD file"
fi

echo "2. Creating Python test script..."
cat > test_scgpt.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Minimal test of scGPT embedding generation
"""
import sys
import numpy as np
import pandas as pd
import scanpy as sc

print("Testing scGPT embedding generation...")

# Try to import scGPT
try:
    import scgpt as scg
    print("✓ scGPT imported successfully")
except ImportError:
    print("✗ scGPT not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scgpt", "flash-attn<1.0.5"])
    import scgpt as scg
    print("✓ scGPT installed and imported")

# Load test data
print("\nLoading test H5AD file...")
adata = sc.read_h5ad("test.h5ad")
print(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")

# Subsample for quick test
if adata.n_obs > 1000:
    print(f"  Subsampling to 1000 cells for quick test...")
    sc.pp.subsample(adata, n_obs=1000)

# Basic preprocessing
print("\nPreprocessing...")
if 'counts' not in adata.layers:
    adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("  ✓ Normalized and log-transformed")

# Select highly variable genes
if adata.n_vars > 3000:
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable]
    print(f"  ✓ Selected {adata.n_vars} highly variable genes")

print("\nGenerating mock embeddings (real model not loaded)...")
# Since we don't have the model, generate random embeddings for testing
mock_embeddings = np.random.randn(adata.n_obs, 512).astype(np.float32)

# Create output DataFrame
output_df = pd.DataFrame({
    'cell_id': adata.obs_names.values,
    'scgpt_embedding': list(mock_embeddings)
})

# Add metadata if available
if 'cell_type' in adata.obs.columns:
    output_df['cell_type'] = adata.obs['cell_type'].values

# Save output
output_file = 'test_embeddings.parquet'
output_df.to_parquet(output_file, index=False)
print(f"\n✓ Saved embeddings to {output_file}")
print(f"  Shape: {output_df.shape}")
print(f"  Embedding dimension: {mock_embeddings.shape[1]}")

# Verify output
df_check = pd.read_parquet(output_file)
print(f"\n✓ Verified output file")
print(f"  Cells: {len(df_check)}")
print(f"  Columns: {df_check.columns.tolist()}")
print(f"  First embedding shape: {len(df_check.iloc[0]['scgpt_embedding'])}")

print("\n=== Test completed successfully ===")
PYTHON_SCRIPT

echo "3. Running Python test..."
python3 test_scgpt.py

echo
echo "4. Checking output..."
ls -la test_embeddings.parquet

echo
echo "=== Test complete ==="
echo
echo "Next steps:"
echo "1. Build the full Docker image with: docker build -t scgpt-embeddings docker/scgpt/"
echo "2. Run on real data with: ./aws_batch/scgpt/run_on_ec2.sh"
echo "3. Or use the manual Docker commands in EC2_QUICKSTART.md"