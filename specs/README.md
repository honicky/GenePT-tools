# Specifications and Configuration Files

This directory contains specifications and configuration files for the GenePT-tools project.

## Directory Structure

- **`examples/`** - Example configuration files for users
  - `tuning_minimal.yaml` - Simplest hyperparameter tuning configuration
  - `tuning_quick.yaml` - Quick exploration with 10-20 trials
  - `tuning_mixed.yaml` - Example with both fixed and optimized parameters
  - `tuning_full.yaml` - Comprehensive production-ready tuning
  
- **`*.md`** - Technical specifications
  - `cellxgene_mlp_training_spec.md` - MLP training implementation specification
  - `hierarchical_metrics_spec.md` - Cell Ontology hierarchical metrics specification
  - `hyperparameter_tuning_spec.md` - Optuna integration specification
  
- **`hyperparameter_tuning_config.yaml`** - Full reference configuration with all options

## Using Example Configurations

### Quick Start
```bash
# Use minimal config for testing
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_minimal.yaml \
  --local-data-dir data/training \
  --test-data-dir data/test
```

### Production Tuning
```bash
# Run comprehensive hyperparameter search
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_full.yaml \
  --tuning-n-trials 100 \
  --tuning-storage sqlite:///optuna_study.db \
  --local-data-dir data/training \
  --test-data-dir data/test
```

## Creating Custom Configurations

Copy one of the example files and modify as needed. Key sections:

1. **`optuna`** - Study settings (trials, metrics, pruning)
2. **`hyperparameters`** - Parameters to optimize with their ranges
3. **`fixed_params`** - Parameters that stay constant (not optimized)
4. **`best_configs`** - Warm-start from known good configurations

### Fixed vs Optimized Parameters

**To optimize a parameter**, put it in `hyperparameters`:
```yaml
hyperparameters:
  learning_rate:
    type: "float"
    low: 1e-5
    high: 1e-2
    log: true
```

**To keep a parameter fixed**, put it in `fixed_params`:
```yaml
fixed_params:
  learning_rate: 1e-4  # This value will be used for all trials
```

You can mix and match - optimize some parameters while keeping others fixed. See `tuning_mixed.yaml` for a complete example.

### Parameter Precedence

When using hyperparameter tuning, parameters are resolved in this order:

1. **Command-line arguments** (highest priority)
2. **Config file parameters** (`fixed_params` or optimized from `hyperparameters`)
3. **Default values** (lowest priority)

#### Override Behavior

**These command-line arguments always override config values:**
- `--local-data-dir` - Training data location
- `--test-data-dir` - Validation data location  
- `--checkpoint-dir` - Where to save checkpoints
- `--wandb-project` - W&B project name
- `--wandb-entity` - W&B team/organization

**These are used to control the tuning process itself:**
- `--tuning-n-trials` - Overrides `n_trials` in config
- `--tuning-timeout` - Sets maximum time for tuning
- `--tuning-storage` - Database for study persistence

**Example:**
```bash
# Config file specifies batch_size in hyperparameters (will be optimized)
# and epochs: 2 in fixed_params
# Command line overrides data paths only
python scripts/train_cellxgene_mlp.py \
  --tuning-config specs/examples/tuning_quick.yaml \
  --local-data-dir /my/custom/data \  # Overrides any data path in config
  --checkpoint-dir /my/checkpoints    # Overrides checkpoint location
```

This design allows you to:
- Use the same config file with different datasets
- Override paths for different environments (dev/staging/prod)
- Keep tuning configs generic and portable

See `hyperparameter_tuning_config.yaml` for all available options.