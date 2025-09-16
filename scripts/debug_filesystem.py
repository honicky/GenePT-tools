#!/usr/bin/env python3
"""Debug script to explore filesystem structure in AWS Batch container."""

import os
import sys
from pathlib import Path

def explore_filesystem():
    """Explore and print filesystem structure."""
    print("=" * 80)
    print("FILESYSTEM EXPLORATION")
    print("=" * 80)
    
    # Check root directories
    print("\n### Root directory contents:")
    for item in sorted(os.listdir('/')):
        path = Path('/') / item
        if path.is_dir():
            print(f"  DIR:  {item}/")
        else:
            print(f"  FILE: {item}")
    
    # Check /data if it exists
    if Path('/data').exists():
        print("\n### /data directory exists!")
        print("Contents of /data:")
        try:
            for item in sorted(os.listdir('/data')):
                path = Path('/data') / item
                if path.is_dir():
                    print(f"  DIR:  {item}/")
                    # Check subdirectories
                    try:
                        subitems = list(path.iterdir())[:5]
                        for subitem in subitems:
                            print(f"    - {subitem.name}")
                        if len(list(path.iterdir())) > 5:
                            print(f"    ... and {len(list(path.iterdir())) - 5} more items")
                    except:
                        print(f"    (cannot read)")
                else:
                    print(f"  FILE: {item}")
        except Exception as e:
            print(f"  Error reading /data: {e}")
        
        # Check specific paths
        test_paths = [
            '/data/GenePT-Tools',
            '/data/GenePT-Tools/data',
            '/data/GenePT-Tools/data/cellxgene_embeddings',
            '/data/GenePT-Tools/data/cellxgene_embeddings/training_v1_shuffled_pt',
            '/data/cellxgene_embeddings',
            '/data/training_v1_shuffled_pt',
        ]
        
        print("\n### Checking specific paths:")
        for test_path in test_paths:
            p = Path(test_path)
            if p.exists():
                print(f"  ✓ {test_path}")
                if p.is_dir():
                    try:
                        files = list(p.glob('*'))[:3]
                        print(f"    First few items: {[f.name for f in files]}")
                    except:
                        print(f"    (cannot read contents)")
            else:
                print(f"  ✗ {test_path} (does not exist)")
    else:
        print("\n### /data directory does NOT exist")
    
    # Check /scratch if it exists
    if Path('/scratch').exists():
        print("\n### /scratch directory exists!")
        print("Contents of /scratch:")
        try:
            for item in sorted(os.listdir('/scratch')):
                print(f"  - {item}")
        except Exception as e:
            print(f"  Error reading /scratch: {e}")
    
    # Check mounted volumes
    print("\n### Mount points (from /proc/mounts):")
    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    device, mount_point = parts[0], parts[1]
                    if mount_point.startswith('/data') or mount_point.startswith('/scratch'):
                        print(f"  {device} -> {mount_point}")
    except:
        print("  Could not read /proc/mounts")
    
    # Check environment variables
    print("\n### Relevant environment variables:")
    for key in sorted(os.environ.keys()):
        if 'DATA' in key or 'PATH' in key.upper() or 'DIR' in key:
            print(f"  {key}={os.environ[key]}")
    
    # Check disk usage
    print("\n### Disk usage:")
    os.system("df -h | grep -E '(^Filesystem|/data|/scratch|/$)'")
    
    # Look for any .pt or .parquet files
    print("\n### Searching for data files...")
    
    # Search in common locations
    search_paths = ['/data', '/scratch', '/tmp', '/app']
    for search_path in search_paths:
        if Path(search_path).exists():
            print(f"\nSearching {search_path} for .pt and .parquet files...")
            os.system(f"find {search_path} -type f \\( -name '*.pt' -o -name '*.parquet' \\) 2>/dev/null | head -20")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    explore_filesystem()
    # Exit with success so job doesn't fail
    sys.exit(0)