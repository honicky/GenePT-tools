# Hyperparameter Tuning Specification for CellXGene MLP Training

## Overview
Enhance the existing CellXGene MLP training script to support systematic hyperparameter optimization using Optuna. The system will use a YAML configuration file to define search spaces, optimization strategies, and starting points from previous best runs.

## Key Requirements

### 1. Configuration-Driven Tuning
- **YAML-based configuration**: Define hyperparameter search spaces and optimization settings
- **Warm starting**: Use previous best configurations as starting points
- **Flexible search spaces**: Support different parameter types (float, int, categorical)
- **Conditional parameters**: Allow parameters dependent on other choices (e.g., scheduler params)

### 2. Integration with Existing Training Pipeline
- **Non-invasive changes**: Existing training script should work unchanged
- **Optional tuning mode**: Presence of `--tuning-config` enables hyperparameter optimization
- **Config file path**: `--tuning-config` to specify YAML configuration (if provided, tuning is enabled)
- **Maintain compatibility**: All existing CLI arguments remain functional

## Architecture

### Configuration Structure
The YAML configuration file organizes settings into logical sections:

- **Optuna Settings**: Study configuration, optimization strategy, pruning and sampling methods
- **Hyperparameter Search Spaces**: Model architecture, training parameters, optimizer choices, learning rate scheduling
- **Fixed Parameters**: Data paths, evaluation settings, system configuration that remain constant
- **Best Configurations**: Previous successful runs used for warm-starting
- **Warm-Start Strategy**: Advanced settings for how to use previous configurations

### Core Components

#### OptunaManager
Central class that orchestrates the hyperparameter optimization process. It loads the YAML configuration, creates and manages the Optuna study, handles warm-start initialization, and coordinates trial execution. The manager integrates with the existing MLPTrainer to evaluate different hyperparameter combinations.

#### Enhanced TrainingConfig
Extended configuration class that can be created from Optuna trial suggestions. Supports new parameters for advanced optimization like optimizer type, learning rate scheduling, label smoothing, and gradient clipping. Provides seamless conversion between Optuna suggestions and training configuration.

#### Hyperparameter Suggestion System
Interprets YAML specifications to create appropriate Optuna distributions. Handles different parameter types (float with optional log scale, integers, categorical choices) and manages conditional parameters that depend on other selections.

## Implementation Strategy

### Phase 1: Core Optuna Integration

Create the foundation for Optuna-based optimization by implementing the OptunaManager class and YAML configuration parser. The manager will handle study creation, trial management, and integration with the existing training pipeline.

The configuration parser validates YAML files, creates search spaces from specifications, and applies conditional parameter rules. It ensures that invalid configurations are caught early and provides clear error messages.

### Phase 2: Training Script Enhancement

Modify the training script to support optional tuning mode triggered by the presence of `--tuning-config`. When in tuning mode, the script creates an OptunaManager instance and runs optimization trials instead of standard training.

Additional CLI arguments allow overriding the number of trials and setting timeouts for the optimization process. The script maintains full backward compatibility when tuning configuration is not provided.

### Phase 3: Warm-Start Strategies

Warm-starting accelerates optimization by leveraging previous successful configurations. The system supports multiple warm-start approaches:

#### Basic Warm-Start
Previous best configurations are loaded from the YAML file and added as completed trials to the Optuna study. These trials influence the optimization algorithm's sampling strategy, biasing it toward promising regions of the hyperparameter space. Warm start configurations will be tried in the order they are listed in the configuration file.  

#### Auto-Loading from External Sources
The system can automatically load configurations from:
- Weights & Biases runs.  These can be listed by URL instead of specifying the configuration parameters

### Phase 4: Monitoring and Reporting

Integration with Weights & Biases provides comprehensive experiment tracking. Each trial's hyperparameters, metrics, and training curves are logged for analysis. Special visualizations highlight the relationship between hyperparameters and performance.

Trial progress is reported in real-time, showing the current best configuration and recent trial results. The system tracks both optimization metrics and computational efficiency to help balance performance with training cost.

## Usage Patterns

### Basic Hyperparameter Tuning
Run optimization with a specified number of trials, using the search spaces defined in the YAML configuration. The optimizer will explore the space, guided by any warm-start configurations provided.

### Time-Bounded Optimization
Set a maximum time limit for optimization, useful for fitting tuning into specific time windows. The optimizer will run as many trials as possible within the time constraint.

## Success Criteria

### Functional Requirements
- Successfully run hyperparameter optimization with proper trial management
- Find configurations that improve upon baseline performance
- Support warm-starting from previous runs with measurable impact on convergence
- Proper pruning of unpromising trials to save computational resources

### Performance Goals
- Efficiently explore parameter space using Bayesian optimization
- Converge to good solutions within reasonable trial counts (typically 50-100 trials)

### Usability Requirements
- Clear, self-documenting YAML configuration format
- Informative progress reporting during optimization
- Easy modification of search spaces without code changes
- Seamless integration with existing tools (WandB, checkpointing)

## Testing Strategy

### Unit Tests
Verify correct parsing of YAML configurations, proper creation of Optuna distributions, warm-start trial addition, and configuration generation from trials. Test conditional parameter logic and edge cases in parameter ranges.  Utilize functional style (without side effects) for core functions to make unit testing without mocks easier.  These tests should run in a few seconds at most.

### Integration Tests
End-to-end optimization on small datasets to verify the complete pipeline. Test study persistence and recovery, warm-start effectiveness, and pruning behavior. Ensure backward compatibility with existing training scripts.  These tests should run quickly.

## Key Design Decisions

### Why Optuna?
Optuna provides state-of-the-art Bayesian optimization with Tree-structured Parzen Estimators, efficient pruning algorithms, and excellent study persistence. Its define-by-run API integrates naturally with PyTorch training loops.

### Why YAML Configuration?
YAML provides a human-readable format for complex nested configurations. It's version-control friendly, supports comments for documentation, and is familiar to ML practitioners from other tools.

### Why Warm-Starting Matters
In practice, hyperparameter optimization is iterative. Warm-starting dramatically reduces the time to find good configurations by leveraging institutional knowledge and previous experiments. This is especially valuable in production settings where training time is expensive.

## References

- Optuna: https://optuna.readthedocs.io/
- Tree-structured Parzen Estimator: Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NIPS 2011
- Hyperband Pruning: Li et al., "Hyperband: A Novel Bandit-Based Approach", JMLR 2017
- Previous best run (nwhis8xb): See `notebooks/cellxgene_v2_mlp.ipynb`