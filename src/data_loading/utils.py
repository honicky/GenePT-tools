"""Utilities for S3 data access and file management."""

from pathlib import Path
from typing import List, Optional
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def get_s3_client(profile_name: str = None):
  """Create an S3 client with the specified AWS profile.
  
  Args:
    profile_name: AWS profile to use for credentials. If None or empty string,
                  uses default credentials (IAM role, environment vars, etc.)
    
  Returns:
    boto3 S3 client
  """
  # If profile_name is None, empty string, or "none", use default credentials (e.g., ECS task role)
  if not profile_name or profile_name.lower() == "none":
    return boto3.client('s3')
  
  # Otherwise use the specified profile
  session = boto3.Session(profile_name=profile_name)
  return session.client('s3')


def list_s3_files(
    bucket: str,
    prefix: str,
    profile_name: str = None
) -> List[str]:
  """List all files in an S3 bucket with given prefix.
  
  Args:
    bucket: S3 bucket name
    prefix: S3 key prefix
    profile_name: AWS profile to use
    
  Returns:
    List of S3 keys
  """
  s3_client = get_s3_client(profile_name)
  
  files = []
  paginator = s3_client.get_paginator('list_objects_v2')
  
  try:
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
      if 'Contents' in page:
        for obj in page['Contents']:
          # Only include .parquet files
          if obj['Key'].endswith('.parquet'):
            files.append(obj['Key'])
  except (NoCredentialsError, ClientError) as e:
    raise RuntimeError(f"Failed to list S3 files: {e}")
  
  # Sort files to ensure consistent ordering
  files.sort()
  return files


def download_s3_file(
    bucket: str,
    s3_key: str,
    local_path: Path,
    profile_name: str = None
) -> Path:
  """Download a single file from S3.
  
  Args:
    bucket: S3 bucket name
    s3_key: S3 object key
    local_path: Local path to save file
    profile_name: AWS profile to use
    
  Returns:
    Path to downloaded file
  """
  s3_client = get_s3_client(profile_name)
  
  # Create parent directory if needed
  local_path.parent.mkdir(parents=True, exist_ok=True)
  
  try:
    s3_client.download_file(bucket, s3_key, str(local_path))
    return local_path
  except (NoCredentialsError, ClientError) as e:
    raise RuntimeError(f"Failed to download {s3_key}: {e}")


def check_local_file(
    local_path: Path,
    expected_size: Optional[int] = None
) -> bool:
  """Check if a local file exists and optionally verify its size.
  
  Args:
    local_path: Path to local file
    expected_size: Expected file size in bytes (optional)
    
  Returns:
    True if file exists and matches expected size (if provided)
  """
  if not local_path.exists():
    return False
  
  if expected_size is not None:
    actual_size = local_path.stat().st_size
    return actual_size == expected_size
  
  return True


def get_or_download_file(
    filename: str,
    local_dir: Path,
    bucket: str,
    s3_prefix: str,
    download_if_missing: bool = True,
    profile_name: str = None
) -> Optional[Path]:
  """Get a file from local directory or download from S3 if needed.
  
  Args:
    filename: Name of the file (e.g., 'batch_0000.parquet')
    local_dir: Local directory to check/save file
    bucket: S3 bucket name
    s3_prefix: S3 prefix for the file
    download_if_missing: Whether to download if not found locally
    profile_name: AWS profile to use
    
  Returns:
    Path to the file, or None if not found and download disabled
  """
  local_path = local_dir / filename
  
  # Check if file exists locally
  if check_local_file(local_path):
    # print(f"Using local file: {local_path}")
    return local_path
  
  # Download if enabled
  if download_if_missing:
    s3_key = f"{s3_prefix}/{filename}" if s3_prefix else filename
    print(f"Downloading from S3: s3://{bucket}/{s3_key}")
    return download_s3_file(bucket, s3_key, local_path, profile_name)
  
  print(f"File not found locally and download disabled: {filename}")
  return None