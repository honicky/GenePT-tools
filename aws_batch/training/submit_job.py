#!/usr/bin/env python3
"""Submit and monitor AWS Batch training jobs using AWS CLI."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

def submit_training_job(
    job_name: str,
    job_queue: str = "miratyper-memverge-queue",
    job_definition: str = "genept-training-job",
    parameters: Optional[Dict[str, str]] = None,
    config_file: Optional[Path] = None,
    monitor: bool = True,
    region: str = "us-west-2",
    profile: str = "memverge"
) -> str:
  """Submit a training job to AWS Batch using AWS CLI.
  
  Args:
    job_name: Name for this job
    job_queue: AWS Batch job queue name
    job_definition: AWS Batch job definition name
    parameters: Dictionary of parameters to override
    config_file: Path to config file to upload to S3
    monitor: Whether to monitor job progress
    region: AWS region
    profile: AWS profile to use
    
  Returns:
    Job ID
  """
  
  # Upload config file if provided
  config_s3_path = None
  if config_file and config_file.exists():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_key = f"configs/{job_name}_{timestamp}.yaml"
    config_s3_path = f"s3://miratyper-training-configs/{config_key}"
    
    print(f"Uploading config to {config_s3_path}")
    cmd = [
      "aws", "s3", "cp",
      str(config_file),
      config_s3_path,
      "--profile", profile
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
      print(f"Error uploading config: {result.stderr}", file=sys.stderr)
      sys.exit(1)
  
  # Build parameters
  job_params = {}
  if config_s3_path:
    job_params['config_file'] = config_s3_path
  if parameters:
    job_params.update(parameters)
  
  # Submit job
  print(f"Submitting job: {job_name}")
  print(f"Parameters: {json.dumps(job_params, indent=2)}")
  
  cmd = [
    "aws", "batch", "submit-job",
    "--job-name", job_name,
    "--job-queue", job_queue,
    "--job-definition", job_definition,
    "--region", region,
    "--profile", profile
  ]
  
  if job_params:
    param_list = [f"{k}={v}" for k, v in job_params.items()]
    cmd.extend(["--parameters", ",".join(param_list)])
  
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"Error submitting job: {result.stderr}", file=sys.stderr)
    sys.exit(1)
  
  response = json.loads(result.stdout)
  job_id = response['jobId']
  print(f"Job submitted successfully!")
  print(f"Job ID: {job_id}")
  print(f"Job ARN: {response['jobArn']}")
  
  if monitor:
    monitor_job(job_id, region, profile)
  
  return job_id


def monitor_job(job_id: str, region: str = "us-west-2", profile: str = "memverge"):
  """Monitor a running job until completion.
  
  Args:
    job_id: AWS Batch job ID
    region: AWS region
    profile: AWS profile to use
  """
  
  print(f"\nMonitoring job {job_id}...")
  print("Press Ctrl+C to stop monitoring (job will continue running)")
  
  last_status = None
  log_stream_name = None
  next_token = None
  
  try:
    while True:
      # Get job status
      cmd = [
        "aws", "batch", "describe-jobs",
        "--jobs", job_id,
        "--region", region,
        "--profile", profile
      ]
      result = subprocess.run(cmd, capture_output=True, text=True)
      if result.returncode != 0:
        print(f"Error getting job status: {result.stderr}", file=sys.stderr)
        break
      
      response = json.loads(result.stdout)
      if not response['jobs']:
        print(f"Job {job_id} not found")
        break
      
      job = response['jobs'][0]
      status = job['status']
      
      # Print status change
      if status != last_status:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
        last_status = status
        
        # Get log stream name when job starts running
        if status == 'RUNNING' and 'logStreamName' in job['container']:
          log_stream_name = job['container']['logStreamName']
          print(f"Log stream: {log_stream_name}")
      
      # Stream logs if available
      if log_stream_name and status in ['RUNNING', 'SUCCEEDED', 'FAILED']:
        try:
          cmd = [
            "aws", "logs", "get-log-events",
            "--log-group-name", "/aws/batch/miratyper-training",
            "--log-stream-name", log_stream_name,
            "--start-from-head",
            "--region", region,
            "--profile", profile
          ]
          if next_token:
            cmd.extend(["--next-token", next_token])
          
          result = subprocess.run(cmd, capture_output=True, text=True)
          if result.returncode == 0:
            log_response = json.loads(result.stdout)
            
            for event in log_response.get('events', []):
              timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
              print(f"[{timestamp.strftime('%H:%M:%S')}] {event['message']}", end='')
            
            next_token = log_response.get('nextForwardToken')
        except Exception as e:
          # Log stream might not be ready yet
          pass
      
      # Check if job is complete
      if status in ['SUCCEEDED', 'FAILED']:
        print(f"\nJob completed with status: {status}")
        
        if status == 'FAILED' and 'statusReason' in job:
          print(f"Failure reason: {job['statusReason']}")
        
        # Print final metrics if available
        if status == 'SUCCEEDED':
          print("\n✅ Training completed successfully!")
          print("Check WandB for detailed metrics and model artifacts")
          print(f"Outputs available in s3://miratyper-training-outputs/{job_name}/")
        
        break
      
      # Wait before next check
      time.sleep(5)
      
  except KeyboardInterrupt:
    print("\n\nStopped monitoring (job continues running in background)")
    print(f"To check status: aws batch describe-jobs --jobs {job_id} --profile memverge")
    print(f"To view logs: aws logs tail /aws/batch/miratyper-training --follow --profile memverge")


def main():
  parser = argparse.ArgumentParser(description='Submit GenePT training jobs to AWS Batch')
  
  parser.add_argument(
    '--job-name',
    required=True,
    help='Name for this training job'
  )
  
  parser.add_argument(
    '--job-queue',
    default='miratyper-memverge-queue',
    help='AWS Batch job queue (default: miratyper-memverge-queue)'
  )
  
  parser.add_argument(
    '--job-definition',
    default='genept-training-job',
    help='AWS Batch job definition (default: genept-training-job)'
  )
  
  parser.add_argument(
    '--config',
    type=Path,
    help='Path to training config YAML file'
  )
  
  parser.add_argument(
    '--epochs',
    type=str,
    help='Number of training epochs'
  )
  
  parser.add_argument(
    '--batch-size',
    type=str,
    help='Training batch size'
  )
  
  parser.add_argument(
    '--learning-rate',
    type=str,
    help='Learning rate'
  )
  
  parser.add_argument(
    '--wandb-project',
    type=str,
    help='WandB project name'
  )
  
  parser.add_argument(
    '--no-monitor',
    action='store_true',
    help='Submit job without monitoring'
  )
  
  parser.add_argument(
    '--region',
    default='us-west-2',
    help='AWS region (default: us-west-2)'
  )
  
  parser.add_argument(
    '--profile',
    default='memverge',
    help='AWS profile to use (default: memverge)'
  )
  
  args = parser.parse_args()
  
  # Build parameters from command line
  parameters = {}
  if args.epochs:
    parameters['epochs'] = args.epochs
  if args.batch_size:
    parameters['batch_size'] = args.batch_size
  if args.learning_rate:
    parameters['learning_rate'] = args.learning_rate
  if args.wandb_project:
    parameters['wandb_project'] = args.wandb_project
  
  # Submit job
  job_id = submit_training_job(
    job_name=args.job_name,
    job_queue=args.job_queue,
    job_definition=args.job_definition,
    parameters=parameters,
    config_file=args.config,
    monitor=not args.no_monitor,
    region=args.region,
    profile=args.profile
  )
  
  if args.no_monitor:
    print(f"\nJob submitted: {job_id}")
    print(f"Check status: aws batch describe-jobs --jobs {job_id} --profile {args.profile}")


if __name__ == '__main__':
  main()