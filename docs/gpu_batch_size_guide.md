# GPU Memory and Batch Size Performance Guide

## Overview
GPU memory is often the primary constraint when training neural networks. This guide explains how GPU memory impacts batch size selection and performance for the CellXGene MLP training pipeline.

## Current Model Memory Requirements

### Model Architecture
- **Input dimension**: 500-3078 (configurable via `n_dims`)
- **Hidden layers**: 1-5 layers (typically 3)
- **Output classes**: ~150 cell types
- **Dropout**: 0.053
- **Mixed precision**: FP16 available (halves memory usage)

### Memory Formula (Approximate)
```
Memory (GB) ≈ (Model Parameters × 4 bytes + Batch Size × Feature Dim × 4 bytes) / 1e9
```

With mixed precision (FP16):
```
Memory (GB) ≈ (Model Parameters × 2 bytes + Batch Size × Feature Dim × 2 bytes) / 1e9
```

## GPU Memory Tiers and Recommended Batch Sizes

### Consumer GPUs
| GPU | Memory | FP32 Batch Size | FP16 Batch Size | Notes |
|-----|--------|-----------------|-----------------|-------|
| RTX 3060 | 12 GB | 1024-2048 | 4096-8192 | Good entry point |
| RTX 3070 | 8 GB | 512-1024 | 2048-4096 | Memory limited |
| RTX 3080 | 10-12 GB | 1024-2048 | 4096-8192 | Good price/performance |
| RTX 3090 | 24 GB | 4096-8192 | 16384-32768 | Excellent for development |
| RTX 4090 | 24 GB | 4096-8192 | 16384-32768 | Fastest, best for production |

### Data Center GPUs (AWS/Cloud)
| GPU | Memory | FP32 Batch Size | FP16 Batch Size | AWS Instance | $/hour |
|-----|--------|-----------------|-----------------|--------------|--------|
| T4 | 16 GB | 2048-4096 | 8192-16384 | g4dn.xlarge | ~$0.53 |
| V100 | 16-32 GB | 2048-8192 | 8192-32768 | p3.2xlarge | ~$3.06 |
| A10G | 24 GB | 4096-8192 | 16384-32768 | g5.xlarge | ~$1.01 |
| A100 | 40-80 GB | 8192-32768 | 32768-131072 | p4d.24xlarge | ~$32.77 |

## Performance Impact of Batch Size

### 1. **Training Speed**
```python
# Larger batches = better GPU utilization
Batch Size 256:  ~30% GPU utilization (memory bound)
Batch Size 1024: ~60% GPU utilization (balanced)
Batch Size 4096: ~85% GPU utilization (compute bound)
Batch Size 16384: ~95% GPU utilization (optimal for throughput)
```

### 2. **Convergence Characteristics**
- **Small batches (256-512)**: 
  - ✅ Better generalization
  - ✅ Faster convergence (in epochs)
  - ❌ Slower training (in wall time)
  - ❌ Noisy gradients

- **Medium batches (1024-4096)**:
  - ✅ Good balance of speed and accuracy
  - ✅ Stable training
  - ✅ Efficient GPU usage
  - 🎯 **Recommended default**

- **Large batches (8192+)**:
  - ✅ Maximum throughput
  - ✅ Smooth gradients
  - ❌ May need learning rate scaling
  - ❌ Risk of worse generalization

### 3. **Learning Rate Scaling**
When increasing batch size, scale learning rate:
```python
# Linear scaling rule (works well up to batch size ~8192)
new_lr = base_lr * (new_batch_size / base_batch_size)

# Square root scaling (for very large batches)
new_lr = base_lr * sqrt(new_batch_size / base_batch_size)
```

## Optimization Strategies

### 1. **Gradient Accumulation** (Simulate larger batches)
```python
# Effective batch size = batch_size * accumulation_steps
--batch-size 1024 --gradient-accumulation-steps 4  # Effective: 4096
```

### 2. **Mixed Precision Training** (Double batch size)
```python
# Enable in training script
--mixed-precision  # Uses FP16, ~2x memory savings
```

### 3. **Dynamic Batch Sizing**
```python
# Start with large batch, reduce if OOM
def find_optimal_batch_size():
    batch_size = 8192
    while batch_size >= 256:
        try:
            train_with_batch_size(batch_size)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            torch.cuda.empty_cache()
```

## Practical Recommendations

### For Your CellXGene MLP Model

1. **Development/Testing** (T4 16GB or RTX 3060):
   ```bash
   --batch-size 2048 --mixed-precision
   ```

2. **Standard Training** (A10G 24GB or RTX 3090):
   ```bash
   --batch-size 4096 --mixed-precision
   ```

3. **Fast Training** (A100 40GB+):
   ```bash
   --batch-size 16384 --mixed-precision --learning-rate 0.0002
   ```

### Memory Monitoring
```python
# Add to training loop
if step % 100 == 0:
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### Batch Size Search Space for Optuna
```yaml
# In tuning config
batch_size:
  type: "categorical"
  choices: [512, 1024, 2048, 4096, 8192]  # Adjust based on GPU
```

## Key Takeaways

1. **GPU memory directly limits maximum batch size**
2. **Larger batches → faster training but may need LR tuning**
3. **Mixed precision effectively doubles your batch size capacity**
4. **Batch size 1024-4096 is usually optimal for this model**
5. **Use gradient accumulation if memory-limited**
6. **Monitor GPU utilization to ensure efficient training**

## Quick Reference Commands

```bash
# Check GPU memory
nvidia-smi

# Monitor during training
watch -n 1 nvidia-smi

# PyTorch memory summary
python -c "import torch; print(torch.cuda.memory_summary())"

# Find optimal batch size
python scripts/train_cellxgene_mlp.py \
  --batch-size 8192 \
  --mixed-precision \
  --max-steps-per-epoch 10  # Quick test
```

## Cost-Performance Analysis

| Setup | Batch Size | Training Time | Cost/Epoch | Quality |
|-------|------------|---------------|------------|---------|
| CPU (current) | 1024 | ~2 hours | $0.20 | Baseline |
| T4 GPU | 4096 | ~20 min | $0.18 | Same |
| A10G GPU | 8192 | ~12 min | $0.20 | Same |
| A100 GPU | 32768 | ~5 min | $2.73 | Same |

**Recommendation**: A10G provides best cost/performance ratio for this workload.