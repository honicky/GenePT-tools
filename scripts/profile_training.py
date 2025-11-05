"""Profile training performance to identify bottlenecks."""

import torch
import time
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import TrainingConfig
from src.data_loading.composable_dataset import ComposableTrainingDataset


def profile_data_loading():
    """Profile data loading from composable dataset."""
    print("\n" + "="*60)
    print("Profiling Data Loading")
    print("="*60)

    config = TrainingConfig(
        use_composable_dataset=True,
        base_data_dir=Path("/mmc-scratch/scratch/"),
        embedding_types=["genept", "tissue", "metadata"],
        genept_dims=1536,
        cell_count_threshold=5000,
        cell_counts_file=Path("/data/batch-jobs/cell_counts.csv"),
        batch_size=1024,
        start_batch_file=0,
        end_batch_file=10,  # Only test with 10 files
        verbose=False
    )

    # Load cell type filtering
    print("Loading cell type filtering...")
    counts_df = pd.read_csv(config.cell_counts_file)
    included_df = counts_df[counts_df['cell_count'] >= config.cell_count_threshold]

    # Load cell types
    cell_types_path = config.cell_counts_file.parent / "cell_types.csv"
    cell_types_df = pd.read_csv(cell_types_path)
    cell_type_codes = pd.Series(range(len(cell_types_df)), index=cell_types_df['cell_type'])

    # Create code remapping
    filtered_cell_types = included_df['cell_type'].tolist()
    filtered_codes = pd.Series(range(len(filtered_cell_types)), index=filtered_cell_types)

    code_remapping = {}
    for cell_type in filtered_cell_types:
        if cell_type in cell_type_codes.index:
            original_code = cell_type_codes[cell_type]
            new_code = filtered_codes[cell_type]
            code_remapping[original_code] = new_code

    for cell_type in counts_df[counts_df['cell_count'] < config.cell_count_threshold]['cell_type']:
        if cell_type in cell_type_codes.index:
            original_code = cell_type_codes[cell_type]
            code_remapping[original_code] = -100

    print(f"Filtered to {len(filtered_cell_types)} cell types")

    # Create dataset
    print("\nCreating dataset...")
    dataset = ComposableTrainingDataset(
        base_dir=config.base_data_dir,
        embedding_types=config.embedding_types,
        batch_size=config.batch_size,
        start_batch_file=config.start_batch_file,
        end_batch_file=config.end_batch_file,
        genept_dims=config.genept_dims,
        code_remapping=code_remapping,
        track_invalid_embeddings=True,
        shuffle_files_per_epoch=False,  # No shuffle for profiling
        shuffle_within_files=False,
        seed=42,
        verbose=False
    )

    print(f"Dataset ready: {len(dataset)} estimated batches")

    # Profile data loading
    print("\nProfiling batches...")
    times = {
        'iteration': [],
        'data_loading': [],
    }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False
    )

    iterator = iter(dataloader)
    i = 0
    while i < 50:
        try:
            t0 = time.time()
            X, y = next(iterator)
            t_load = time.time() - t0

            times['iteration'].append(i)
            times['data_loading'].append(t_load * 1000)  # ms

            if i % 10 == 0:
                print(f"  Batch {i:2d}: load={t_load*1000:6.1f}ms  shape={X.shape}")
            i += 1
        except StopIteration:
            print(f"  Dataset exhausted after {i} batches")
            break

    df = pd.DataFrame(times)
    print(f"\nData Loading Stats (ms):")
    print(f"  Mean: {df['data_loading'].mean():.1f}")
    print(f"  Median: {df['data_loading'].median():.1f}")
    print(f"  Std: {df['data_loading'].std():.1f}")
    print(f"  Min: {df['data_loading'].min():.1f}")
    print(f"  Max: {df['data_loading'].max():.1f}")

    return df


def profile_training_loop():
    """Profile full training loop with GPU."""
    print("\n" + "="*60)
    print("Profiling Training Loop (GPU)")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling")
        return None

    device = torch.device("cuda")

    # Create simple model
    from src.models.mlp import MLPClassifier

    input_dim = 1662  # genept 1536 + tissue 126
    n_classes = 360  # After filtering
    model = MLPClassifier(
        n_dims=input_dim,
        n_classes=n_classes,
        n_hidden_layers=4,
        dropout=0.1
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load dataset
    config = TrainingConfig(
        use_composable_dataset=True,
        base_data_dir=Path("/mmc-scratch/scratch/"),
        embedding_types=["genept", "tissue", "metadata"],
        genept_dims=1536,
        cell_count_threshold=5000,
        cell_counts_file=Path("/data/batch-jobs/cell_counts.csv"),
        batch_size=1024,
        start_batch_file=0,
        end_batch_file=10,
        verbose=False
    )

    # Load filtering (simplified)
    counts_df = pd.read_csv(config.cell_counts_file)
    included_df = counts_df[counts_df['cell_count'] >= config.cell_count_threshold]
    cell_types_path = config.cell_counts_file.parent / "cell_types.csv"
    cell_types_df = pd.read_csv(cell_types_path)
    cell_type_codes = pd.Series(range(len(cell_types_df)), index=cell_types_df['cell_type'])

    filtered_cell_types = included_df['cell_type'].tolist()
    filtered_codes = pd.Series(range(len(filtered_cell_types)), index=filtered_cell_types)

    code_remapping = {}
    for cell_type in filtered_cell_types:
        if cell_type in cell_type_codes.index:
            code_remapping[cell_type_codes[cell_type]] = filtered_codes[cell_type]
    for cell_type in counts_df[counts_df['cell_count'] < config.cell_count_threshold]['cell_type']:
        if cell_type in cell_type_codes.index:
            code_remapping[cell_type_codes[cell_type]] = -100

    dataset = ComposableTrainingDataset(
        base_dir=config.base_data_dir,
        embedding_types=config.embedding_types,
        batch_size=config.batch_size,
        start_batch_file=config.start_batch_file,
        end_batch_file=config.end_batch_file,
        genept_dims=config.genept_dims,
        code_remapping=code_remapping,
        track_invalid_embeddings=True,
        shuffle_files_per_epoch=False,
        shuffle_within_files=False,
        seed=42,
        verbose=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=False
    )

    # Profile training loop
    print("\nProfiling training iterations...")
    times = {
        'iteration': [],
        'data_loading': [],
        'h2d_transfer': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': [],
    }

    model.train()
    iterator = iter(dataloader)

    i = 0
    while i < 50:
        try:
            t_iter_start = time.time()

            # Data loading
            t0 = time.time()
            X, y = next(iterator)
            t_load = time.time() - t0

            # Transfer to GPU
            t0 = time.time()
            X = X.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            torch.cuda.synchronize()
            t_h2d = time.time() - t0

            # Forward pass
            t0 = time.time()
            logits = model(X)
            loss = criterion(logits, y)
            torch.cuda.synchronize()
            t_forward = time.time() - t0

            # Backward pass
            t0 = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            t_backward = time.time() - t0

            # Optimizer step
            t0 = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            t_optim = time.time() - t0

            t_iter_total = time.time() - t_iter_start

            times['iteration'].append(i)
            times['data_loading'].append(t_load * 1000)
            times['h2d_transfer'].append(t_h2d * 1000)
            times['forward'].append(t_forward * 1000)
            times['backward'].append(t_backward * 1000)
            times['optimizer'].append(t_optim * 1000)
            times['total'].append(t_iter_total * 1000)

            if i % 10 == 0:
                print(f"  Batch {i:2d}: load={t_load*1000:5.1f}ms  h2d={t_h2d*1000:5.1f}ms  "
                      f"fwd={t_forward*1000:5.1f}ms  bwd={t_backward*1000:5.1f}ms  "
                      f"opt={t_optim*1000:5.1f}ms  total={t_iter_total*1000:6.1f}ms")
            i += 1
        except StopIteration:
            print(f"  Dataset exhausted after {i} batches")
            break

    df = pd.DataFrame(times)

    print(f"\nTraining Loop Breakdown (ms, excluding first 5 warmup batches):")
    df_warm = df[df['iteration'] >= 5]

    for col in ['data_loading', 'h2d_transfer', 'forward', 'backward', 'optimizer', 'total']:
        mean_time = df_warm[col].mean()
        pct = (mean_time / df_warm['total'].mean() * 100) if col != 'total' else 100.0
        print(f"  {col:15s}: {mean_time:6.1f}ms  ({pct:4.1f}%)")

    print(f"\nThroughput:")
    samples_per_batch = 1024
    total_time_sec = df_warm['total'].mean() / 1000
    samples_per_sec = samples_per_batch / total_time_sec
    print(f"  {samples_per_sec:.0f} samples/sec")
    print(f"  {3682 * total_time_sec / 60:.1f} minutes for full epoch (3682 batches)")

    return df


def check_gpu_utilization():
    """Check current GPU utilization."""
    print("\n" + "="*60)
    print("GPU Utilization Check")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpu_util, mem_util, mem_used, mem_total = result.stdout.strip().split(',')
            print(f"GPU Utilization: {gpu_util.strip()}%")
            print(f"Memory Utilization: {mem_util.strip()}%")
            print(f"Memory Used: {mem_used.strip()} MB / {mem_total.strip()} MB")
        else:
            print("Failed to query nvidia-smi")
    except Exception as e:
        print(f"Error checking GPU: {e}")


if __name__ == "__main__":
    print("Training Performance Profiler")
    print("="*60)

    # Check GPU first
    check_gpu_utilization()

    # Profile data loading
    df_load = profile_data_loading()

    # Profile training loop
    df_train = profile_training_loop()

    # Final summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if df_load is not None:
        print(f"Data loading: {df_load['data_loading'].mean():.1f}ms per batch (avg)")

    if df_train is not None:
        df_warm = df_train[df_train['iteration'] >= 5]
        total = df_warm['total'].mean()
        print(f"Total per iteration: {total:.1f}ms")
        print(f"  Data loading: {df_warm['data_loading'].mean()/total*100:.1f}%")
        print(f"  GPU transfer: {df_warm['h2d_transfer'].mean()/total*100:.1f}%")
        print(f"  Computation: {(df_warm['forward'].mean() + df_warm['backward'].mean() + df_warm['optimizer'].mean())/total*100:.1f}%")

        epoch_time_min = 3682 * total / 1000 / 60
        print(f"\nEstimated time per epoch: {epoch_time_min:.1f} minutes ({epoch_time_min/60:.1f} hours)")
