#!/usr/bin/env python3
"""
Submit scGPT embedding generation jobs to AWS Batch.
Splits large file lists into smaller batches for parallel processing.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import boto3


def split_file_list(file_list: List[str], chunk_size: int) -> List[List[str]]:
    """Split file list into chunks."""
    chunks = []
    for i in range(0, len(file_list), chunk_size):
        chunks.append(file_list[i:i + chunk_size])
    return chunks


def upload_file_list(s3_client, file_list: List[str], bucket: str, key: str):
    """Upload file list to S3."""
    json_data = json.dumps(file_list, indent=2)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json_data.encode('utf-8'),
        ContentType='application/json'
    )
    return f"s3://{bucket}/{key}"


def submit_batch_job(
    batch_client,
    job_name: str,
    job_queue: str,
    job_definition: str,
    input_list_s3: str,
    model_path: str,
    output_bucket: str,
    output_prefix: str
):
    """Submit a job to AWS Batch."""
    response = batch_client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        parameters={
            'input_list': input_list_s3,
            'model_path': model_path,
            'output_bucket': output_bucket,
            'output_prefix': output_prefix
        },
        containerOverrides={
            'command': [
                'python', '/app/scgpt_wrapper.py',
                '--input-list', 'Ref::input_list',
                '--model-path', 'Ref::model_path',
                '--output-bucket', 'Ref::output_bucket',
                '--output-prefix', 'Ref::output_prefix',
                '--batch-size', '256',
                '--device', 'cuda'
            ]
        }
    )
    return response['jobId']


def main():
    parser = argparse.ArgumentParser(description='Submit scGPT jobs to AWS Batch')
    parser.add_argument('--file-list', required=True, help='Path to JSON file with H5AD S3 paths')
    parser.add_argument('--job-queue', default='gpu-queue', help='AWS Batch job queue')
    parser.add_argument('--job-definition', default='scgpt-embedding-generation', help='Job definition')
    parser.add_argument('--model-path', default='s3://pythiomicsdata/models/scgpt/whole_human', help='Model S3 path')
    parser.add_argument('--output-bucket', default='pythiomicsdata', help='Output S3 bucket')
    parser.add_argument('--output-prefix', default='cellxgene_v2/scgpt_embeddings_v2', help='Output prefix')
    parser.add_argument('--files-per-job', type=int, default=10, help='Files per job')
    parser.add_argument('--job-prefix', default='scgpt-embed', help='Job name prefix')
    parser.add_argument('--aws-profile', help='AWS profile to use')
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    
    args = parser.parse_args()
    
    # Load file list
    with open(args.file_list, 'r') as f:
        all_files = json.load(f)
    
    print(f"Total files to process: {len(all_files)}")
    
    # Split into chunks
    chunks = split_file_list(all_files, args.files_per_job)
    print(f"Split into {len(chunks)} jobs ({args.files_per_job} files each)")
    
    if args.dry_run:
        print("\nDry run - would submit these jobs:")
        for i, chunk in enumerate(chunks):
            print(f"  Job {i+1}: {len(chunk)} files")
            print(f"    First file: {chunk[0]}")
            print(f"    Last file: {chunk[-1]}")
        return
    
    # Initialize AWS clients
    session = boto3.Session(profile_name=args.aws_profile) if args.aws_profile else boto3.Session()
    batch_client = session.client('batch', region_name='us-west-2')
    s3_client = session.client('s3')
    
    # Submit jobs
    job_ids = []
    for i, chunk in enumerate(chunks):
        # Upload chunk file list to S3
        chunk_key = f"{args.output_prefix}/job_lists/chunk_{i:04d}.json"
        chunk_s3 = upload_file_list(
            s3_client,
            chunk,
            args.output_bucket,
            chunk_key
        )
        print(f"Uploaded chunk {i+1} to {chunk_s3}")
        
        # Submit job
        job_name = f"{args.job_prefix}-{i:04d}"
        job_id = submit_batch_job(
            batch_client,
            job_name,
            args.job_queue,
            args.job_definition,
            chunk_s3,
            args.model_path,
            args.output_bucket,
            f"{args.output_prefix}/chunk_{i:04d}"
        )
        job_ids.append(job_id)
        print(f"Submitted job {job_name}: {job_id}")
    
    # Save job IDs
    job_info = {
        'total_files': len(all_files),
        'files_per_job': args.files_per_job,
        'total_jobs': len(job_ids),
        'job_ids': job_ids,
        'job_queue': args.job_queue,
        'job_definition': args.job_definition,
        'model_path': args.model_path,
        'output_location': f"s3://{args.output_bucket}/{args.output_prefix}"
    }
    
    job_info_file = f"{args.job_prefix}_job_info.json"
    with open(job_info_file, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    print(f"\nSubmitted {len(job_ids)} jobs")
    print(f"Job information saved to {job_info_file}")


if __name__ == '__main__':
    main()