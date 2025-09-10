#!/usr/bin/env python3
"""
Setup PyTorch with appropriate CUDA support based on available hardware.
This script detects GPU availability and installs the correct PyTorch version.
"""

import subprocess
import sys
import os


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_cuda_version():
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            # Parse CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    # Extract version like "12.1" from "CUDA Version: 12.1"
                    version = line.split('CUDA Version:')[1].split()[0]
                    return version
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_pytorch_index_url(cuda_version):
    """Get PyTorch index URL based on CUDA version."""
    if not cuda_version:
        print("No CUDA detected, using CPU-only PyTorch")
        return "https://download.pytorch.org/whl/cpu"
    
    # Parse major.minor version
    major, minor = cuda_version.split('.')[:2]
    cuda_short = f"{major}{minor}"
    
    # Map to PyTorch CUDA versions (as of 2024)
    cuda_map = {
        "121": "cu121",  # CUDA 12.1
        "118": "cu118",  # CUDA 11.8
        "117": "cu117",  # CUDA 11.7
    }
    
    if cuda_short in cuda_map:
        cuda_tag = cuda_map[cuda_short]
        print(f"CUDA {cuda_version} detected, using PyTorch with {cuda_tag}")
        return f"https://download.pytorch.org/whl/{cuda_tag}"
    else:
        print(f"CUDA {cuda_version} detected but no exact match, using latest CUDA build")
        return "https://download.pytorch.org/whl/cu121"  # Use latest as fallback


def install_pytorch(index_url):
    """Install PyTorch with the appropriate index URL."""
    cmd = [
        "uv", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", index_url,
        "--force-reinstall"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def verify_installation():
    """Verify PyTorch installation and CUDA availability."""
    try:
        import torch
        print("\n" + "="*60)
        print("PyTorch Installation Verification")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("="*60 + "\n")
        return True
    except ImportError as e:
        print(f"Error: Failed to import torch: {e}")
        return False


def main():
    print("🔧 Setting up PyTorch with appropriate hardware support...\n")
    
    # Check for GPU
    has_gpu = check_nvidia_gpu()
    
    if has_gpu:
        cuda_version = get_cuda_version()
        index_url = get_pytorch_index_url(cuda_version)
    else:
        print("No NVIDIA GPU detected, installing CPU-only PyTorch")
        index_url = "https://download.pytorch.org/whl/cpu"
    
    # Install PyTorch
    print(f"\nInstalling PyTorch from: {index_url}")
    if not install_pytorch(index_url):
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Verify installation
    if verify_installation():
        print("✅ PyTorch successfully installed and configured!")
    else:
        print("❌ PyTorch installation verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()