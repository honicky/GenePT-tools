#!/bin/bash
# Process remaining 3 tissues for Transcriptformer embedding generation

set -e  # Exit on error

# Remaining tissues (Blood is running/done, Bone_Marrow is done)
TISSUES=("Lung" "Mammary" "Thymus")

# Configuration
DATA_DIR="/data/Tabula_Sapiens_v2_Curated_Benchmark"
MODEL_DIR="models/transcriptformer"
MODEL_VARIANT="tf_metazoa"
OUTPUT_DIR="data/cz_benchmark/embeddings/transcriptformer"
BATCH_SIZE=6
SCRIPT="scripts/generate_transcriptformer_embeddings.py"

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "======================================================================"
echo "Starting Transcriptformer embedding generation (Batch Size 6)"
echo "======================================================================"
echo "Tissues to process: ${TISSUES[@]}"
echo "Model variant: $MODEL_VARIANT"
echo "Batch size: $BATCH_SIZE"
echo "Memory optimization: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "======================================================================"

# Process each tissue
for tissue in "${TISSUES[@]}"; do
    echo ""
    echo "Processing $tissue..."
    echo "----------------------------------------------------------------------"

    # Check if already processed
    output_file="$OUTPUT_DIR/transcriptformer_${MODEL_VARIANT}_${tissue}_embeddings.parquet"
    if [ -f "$output_file" ]; then
        echo "✓ $tissue already processed, skipping..."
        continue
    fi

    # Run embedding generation
    start_time=$(date +%s)
    echo "Starting at $(date)"

    ./scripts/run_transcriptformer.sh python $SCRIPT \
        --tissue "$tissue" \
        --data-dir "$DATA_DIR" \
        --model-dir "$MODEL_DIR" \
        --model-variant "$MODEL_VARIANT" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    # Check if successful
    if [ -f "$output_file" ]; then
        size=$(du -h "$output_file" | cut -f1)
        echo "✓ Successfully generated embeddings for $tissue ($size) in ${elapsed}s"
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
echo "Generated Transcriptformer files:"
ls -lah "$OUTPUT_DIR"/transcriptformer_*.parquet 2>/dev/null || echo "No files found"

echo ""
echo "Comparison with scGPT:"
echo "----------------------------------------------------------------------"
echo "scGPT embeddings:"
ls -lah "data/cz_benchmark/embeddings/scgpt"/*.parquet 2>/dev/null | awk '{print $NF": "$5}'
echo ""
echo "Transcriptformer embeddings:"
ls -lah "$OUTPUT_DIR"/*.parquet 2>/dev/null | awk '{print $NF": "$5}'