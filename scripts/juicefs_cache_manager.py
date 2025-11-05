"""JuiceFS cache management and warmup for training data."""

import subprocess
import time
from pathlib import Path
import json
import sys


def run_command(cmd, timeout=30):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_juicefs_mount():
    """Check if JuiceFS is mounted and get mount info."""
    print("="*60)
    print("JuiceFS Mount Information")
    print("="*60)

    # Check if /mmc-scratch is mounted
    returncode, stdout, stderr = run_command("df -h | grep juicefs")

    if returncode != 0:
        print("JuiceFS not found in mounted filesystems")
        return None

    print("Mounted JuiceFS filesystems:")
    print(stdout)

    # Parse mount info
    lines = stdout.strip().split('\n')
    mounts = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            mounts.append({
                'filesystem': parts[0],
                'size': parts[1],
                'used': parts[2],
                'available': parts[3],
                'use_pct': parts[4],
                'mountpoint': parts[5]
            })

    return mounts


def check_cache_config():
    """Check JuiceFS cache configuration."""
    print("\n" + "="*60)
    print("JuiceFS Cache Configuration")
    print("="*60)

    # Try to find cache directory
    cache_locations = [
        "/var/jfsCache",
        "/tmp/jfsCache",
        "~/.juicefs/cache",
        "/root/.juicefs/cache"
    ]

    cache_dir = None
    for loc in cache_locations:
        expanded = Path(loc).expanduser()
        if expanded.exists():
            cache_dir = expanded
            print(f"Cache directory found: {cache_dir}")
            break

    if cache_dir is None:
        print("Cache directory not found in standard locations")
        print("Trying to find via mount options...")

        returncode, stdout, stderr = run_command("mount | grep juicefs")
        if returncode == 0:
            print("Mount options:")
            print(stdout)
    else:
        # Get cache size
        returncode, stdout, stderr = run_command(f"du -sh {cache_dir}")
        if returncode == 0:
            print(f"Cache size: {stdout.strip()}")

        # Get cache file count
        returncode, stdout, stderr = run_command(f"find {cache_dir} -type f | wc -l")
        if returncode == 0:
            print(f"Cached files: {stdout.strip()}")

        # Get cache disk usage
        returncode, stdout, stderr = run_command(f"df -h {cache_dir}")
        if returncode == 0:
            print("\nCache filesystem:")
            print(stdout)

    return cache_dir


def get_cache_stats():
    """Get JuiceFS cache statistics."""
    print("\n" + "="*60)
    print("JuiceFS Cache Statistics")
    print("="*60)

    # Try juicefs stats command
    returncode, stdout, stderr = run_command("juicefs stats /mmc-scratch/scratch --raw", timeout=5)

    if returncode == 0 and stdout:
        print("Raw cache stats:")
        print(stdout)
        return stdout
    else:
        print("Unable to get cache stats (juicefs stats not available or timed out)")
        print(f"Error: {stderr}")
        return None


def warmup_training_data(base_dir="/mmc-scratch/scratch/", batch_start=0, batch_end=100):
    """Warmup JuiceFS cache with training data."""
    print("\n" + "="*60)
    print(f"Warming up JuiceFS Cache (batches {batch_start}-{batch_end})")
    print("="*60)

    embedding_dirs = [
        'cellxgene_v2_training_v1_shuffled_genept',
        'cellxgene_v2_training_v1_shuffled_tissue',
        'cellxgene_v2_training_v1_shuffled_metadata'
    ]

    base_path = Path(base_dir)

    # Count total files to warmup
    total_files = 0
    for emb_dir in embedding_dirs:
        dir_path = base_path / emb_dir
        if dir_path.exists():
            batch_files = [dir_path / f"batch_{i:04d}.pt" for i in range(batch_start, batch_end)]
            total_files += sum(1 for f in batch_files if f.exists())

    print(f"Found {total_files} batch files to warmup")

    if total_files == 0:
        print("No files found to warmup")
        return

    # Warmup by reading files
    print("Starting warmup (reading files to trigger cache)...")
    start_time = time.time()

    files_read = 0
    for emb_dir in embedding_dirs:
        dir_path = base_path / emb_dir
        if not dir_path.exists():
            print(f"Skipping {emb_dir} (not found)")
            continue

        print(f"\nWarming up {emb_dir}...")
        for i in range(batch_start, batch_end):
            batch_file = dir_path / f"batch_{i:04d}.pt"
            if batch_file.exists():
                try:
                    # Just read the file to trigger caching
                    with open(batch_file, 'rb') as f:
                        # Read in chunks to avoid loading entire file into memory
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                    files_read += 1

                    if files_read % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = files_read / elapsed
                        eta = (total_files - files_read) / rate if rate > 0 else 0
                        print(f"  Progress: {files_read}/{total_files} files "
                              f"({files_read/total_files*100:.1f}%) - "
                              f"{rate:.1f} files/sec - "
                              f"ETA: {eta/60:.1f} min")
                except Exception as e:
                    print(f"  Error reading {batch_file}: {e}")

    elapsed = time.time() - start_time
    print(f"\nWarmup complete!")
    print(f"  Files read: {files_read}")
    print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Rate: {files_read/elapsed:.1f} files/sec")


def estimate_dataset_size(base_dir="/mmc-scratch/scratch/", batch_start=0, batch_end=100):
    """Estimate total size of dataset to warmup."""
    print("\n" + "="*60)
    print(f"Estimating Dataset Size (batches {batch_start}-{batch_end})")
    print("="*60)

    embedding_dirs = [
        'cellxgene_v2_training_v1_shuffled_genept',
        'cellxgene_v2_training_v1_shuffled_tissue',
        'cellxgene_v2_training_v1_shuffled_metadata'
    ]

    base_path = Path(base_dir)

    total_size = 0
    total_files = 0

    for emb_dir in embedding_dirs:
        dir_path = base_path / emb_dir
        if not dir_path.exists():
            continue

        dir_size = 0
        dir_files = 0

        for i in range(batch_start, batch_end):
            batch_file = dir_path / f"batch_{i:04d}.pt"
            if batch_file.exists():
                size = batch_file.stat().st_size
                dir_size += size
                dir_files += 1

        total_size += dir_size
        total_files += dir_files

        print(f"{emb_dir}:")
        print(f"  Files: {dir_files}")
        print(f"  Size: {dir_size / 1024**3:.2f} GB")

    print(f"\nTotal:")
    print(f"  Files: {total_files}")
    print(f"  Size: {total_size / 1024**3:.2f} GB")

    return total_size, total_files


def monitor_cache_during_training(duration_sec=60):
    """Monitor cache hit rate during training."""
    print("\n" + "="*60)
    print(f"Monitoring Cache Performance ({duration_sec}s)")
    print("="*60)

    # This would need juicefs stats to work
    print("Note: Real-time cache monitoring requires 'juicefs stats' command")
    print("Run: juicefs stats /mmc-scratch/scratch")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JuiceFS cache management for training")
    parser.add_argument("command", choices=["check", "warmup", "estimate", "monitor"],
                       help="Command to run")
    parser.add_argument("--batch-start", type=int, default=0,
                       help="First batch index to warmup")
    parser.add_argument("--batch-end", type=int, default=None,
                       help="Last batch index to warmup (default: all)")
    parser.add_argument("--base-dir", type=str, default="/mmc-scratch/scratch/",
                       help="Base directory for embeddings")

    args = parser.parse_args()

    if args.command == "check":
        mounts = check_juicefs_mount()
        cache_dir = check_cache_config()
        stats = get_cache_stats()

    elif args.command == "estimate":
        # Estimate for all batches if not specified
        if args.batch_end is None:
            # Count total batches
            genept_dir = Path(args.base_dir) / "cellxgene_v2_training_v1_shuffled_genept"
            if genept_dir.exists():
                batch_files = list(genept_dir.glob("batch_*.pt"))
                args.batch_end = len(batch_files)
                print(f"Auto-detected {args.batch_end} total batches")

        estimate_dataset_size(args.base_dir, args.batch_start, args.batch_end)

    elif args.command == "warmup":
        # Default to first 100 batches if not specified
        if args.batch_end is None:
            args.batch_end = 100

        # First estimate size
        total_size, total_files = estimate_dataset_size(args.base_dir, args.batch_start, args.batch_end)

        print("\nProceed with warmup? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            warmup_training_data(args.base_dir, args.batch_start, args.batch_end)
        else:
            print("Warmup cancelled")

    elif args.command == "monitor":
        monitor_cache_during_training()
