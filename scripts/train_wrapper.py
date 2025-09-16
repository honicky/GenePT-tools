#!/usr/bin/env python3
"""
Wrapper script for training that handles S3 downloads.
Uses boto3 which automatically picks up ECS task role credentials.
"""

import os
import sys
import boto3
import subprocess
from pathlib import Path
from urllib.parse import urlparse

def download_from_s3(s3_path, local_path):
    """Download a file from S3 to local path."""
    print(f"Downloading {s3_path} to {local_path}")
    
    # Parse S3 URL
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    # Download using boto3 (will use ECS task role credentials automatically)
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, local_path)
        print(f"Successfully downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"Failed to download from S3: {e}")
        return False

def process_args(args):
    """Process command line arguments and download S3 files."""
    processed_args = []
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        # Check if this is an argument that might have an S3 path
        if arg in ['--tuning-config', '--cell-types-file', '--resume-from']:
            processed_args.append(arg)
            if i + 1 < len(args):
                next_arg = args[i + 1]
                if next_arg.startswith('s3://'):
                    # Download S3 file to /tmp
                    filename = os.path.basename(next_arg)
                    local_path = f'/tmp/{filename}'
                    
                    if download_from_s3(next_arg, local_path):
                        processed_args.append(local_path)
                    else:
                        print(f"Error: Failed to download {next_arg}")
                        sys.exit(1)
                else:
                    processed_args.append(next_arg)
                i += 2
            else:
                i += 1
        else:
            processed_args.append(arg)
            i += 1
    
    # Add default cell types file if not provided
    if '--cell-types-file' not in processed_args:
        print("No --cell-types-file provided, checking for default in S3...")
        default_s3_path = 's3://miratyper-training-configs/cell_types_filtered.csv'
        local_path = '/tmp/cell_types_filtered.csv'
        
        if download_from_s3(default_s3_path, local_path):
            processed_args.extend(['--cell-types-file', local_path])
    
    return processed_args

def main():
    # Get original arguments (skip script name)
    original_args = sys.argv[1:]
    
    print(f"Original arguments: {original_args}")
    
    # Process arguments and download S3 files
    processed_args = process_args(original_args)
    
    print(f"Processed arguments: {processed_args}")
    
    # Run the actual training script
    cmd = ['python', '/app/scripts/train_cellxgene_mlp.py'] + processed_args
    print(f"Running: {' '.join(cmd)}")
    
    # Execute and pass through the exit code
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()