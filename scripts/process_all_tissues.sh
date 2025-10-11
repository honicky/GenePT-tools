#!/bin/bash
# Process all 5 target tissues for scGPT embedding generation

set -e  # Exit on error

# Target tissues from the spec
TISSUES=("Blood" "Bone_Marrow" "Lung" "Mammary" "Thymus")

# Configuration
DATA_DIR="/data/Tabula_Sapiens_v2_Curated_Benchmark"
MODEL_DIR="models/scgpt"
OUTPUT_DIR="data/cz_benchmark/embeddings/scgpt"
BATCH_SIZE=32
SCRIPT="scripts/generate_scgpt_embeddings_v2.py"

echo "======================================================================"
echo "Starting scGPT embedding generation for 5 target tissues"
echo "======================================================================"
echo "Tissues to process: ${TISSUES[@]}"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "======================================================================"

# Process each tissue
for tissue in "${TISSUES[@]}"; do
    echo ""
    echo "Processing $tissue..."
    echo "----------------------------------------------------------------------"

    # Check if already processed
    output_file="$OUTPUT_DIR/scgpt_${tissue}_embeddings.parquet"
    if [ -f "$output_file" ]; then
        echo "✓ $tissue already processed, skipping..."
        continue
    fi

    # Run embedding generation
    ./scripts/run_scgpt.sh python $SCRIPT \
        --tissue "$tissue" \
        --data-dir "$DATA_DIR" \
        --model-dir "$MODEL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE

    # Check if successful
    if [ -f "$output_file" ]; then
        size=$(du -h "$output_file" | cut -f1)
        echo "✓ Successfully generated embeddings for $tissue ($size)"
    else
        echo "✗ Failed to generate embeddings for $tissue"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "All tissues processed successfully!"
echo "======================================================================"

# Summary
echo ""
echo "Generated files:"
ls -lah "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "No files found"

echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR" 2>/dev/null || echo "Directory not found"