#!/bin/bash
# Wrapper script for training that handles S3 downloads
set -e

echo "Starting training wrapper at $(date)"

# AWS credentials should be available through ECS task role
# The container gets temporary credentials via the task role
echo "Checking AWS credentials..."
if [ -n "$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" ]; then
    echo "Running in ECS with task role credentials"
elif [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Using AWS credentials from environment variables"
else
    echo "Warning: No AWS credentials detected. S3 downloads may fail."
fi

# Function to download from S3 if path starts with s3://
download_if_s3() {
    local input_path=$1
    local param_name=$2
    
    if [[ $input_path == s3://* ]]; then
        echo "Detected S3 path for $param_name: $input_path"
        
        # Extract filename from S3 path
        local filename=$(basename $input_path)
        local local_path="/tmp/$filename"
        
        echo "Downloading $input_path to $local_path"
        aws s3 cp $input_path $local_path
        
        if [ $? -eq 0 ]; then
            echo "Successfully downloaded $param_name to $local_path"
            echo $local_path
        else
            echo "Failed to download $param_name from S3"
            exit 1
        fi
    else
        echo $input_path
    fi
}

# Parse arguments and download S3 files
ARGS=()
i=0
while [ $i -lt $# ]; do
    arg=${!i}
    next_i=$((i+1))
    next_arg=${!next_i}
    
    case $arg in
        --tuning-config)
            if [ $next_i -lt $# ]; then
                local_path=$(download_if_s3 "$next_arg" "tuning-config")
                ARGS+=("$arg" "$local_path")
                i=$((i+2))
            else
                ARGS+=("$arg")
                i=$((i+1))
            fi
            ;;
        --cell-types-file)
            if [ $next_i -lt $# ]; then
                local_path=$(download_if_s3 "$next_arg" "cell-types-file")
                ARGS+=("$arg" "$local_path")
                i=$((i+2))
            else
                ARGS+=("$arg")
                i=$((i+1))
            fi
            ;;
        --resume-from)
            if [ $next_i -lt $# ]; then
                local_path=$(download_if_s3 "$next_arg" "resume-from")
                ARGS+=("$arg" "$local_path")
                i=$((i+2))
            else
                ARGS+=("$arg")
                i=$((i+1))
            fi
            ;;
        *)
            ARGS+=("$arg")
            i=$((i+1))
            ;;
    esac
done

# Special handling for standard S3 config/cell types if not provided
if [[ ! " ${ARGS[@]} " =~ " --cell-types-file " ]]; then
    # Check if standard cell types file exists in S3
    echo "No --cell-types-file provided, checking for default in S3..."
    aws s3 ls s3://miratyper-training-configs/cell_types_filtered.csv > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Found default cell_types_filtered.csv in S3, downloading..."
        aws s3 cp s3://miratyper-training-configs/cell_types_filtered.csv /tmp/cell_types_filtered.csv
        ARGS+=("--cell-types-file" "/tmp/cell_types_filtered.csv")
    fi
fi

echo "Final arguments: ${ARGS[@]}"
echo "Starting training script..."

# Run the actual training script with processed arguments
exec python /app/scripts/train_cellxgene_mlp.py "${ARGS[@]}"