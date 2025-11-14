"""Optuna-based hyperparameter optimization manager for CellXGene MLP training."""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import numpy as np

try:
  import optuna
  from optuna.trial import Trial, FrozenTrial
  OPTUNA_AVAILABLE = True
except ImportError:
  OPTUNA_AVAILABLE = False
  Trial = Any
  FrozenTrial = Any

try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class OptunaManager:
  """Manages Optuna study and hyperparameter optimization."""
  
  def __init__(self, config_path: Path, storage: Optional[str] = None):
    """Initialize OptunaManager with configuration.
    
    Args:
      config_path: Path to YAML configuration file
      storage: Optional database URL for study persistence
    """
    if not OPTUNA_AVAILABLE:
      raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
    
    self.config_path = config_path
    self.config = self._load_yaml_config(config_path)
    self.storage = storage or self.config.get('optuna', {}).get('storage')
    
    # Create or load study
    self.study = self._create_study()
    
    # Add warm-start trials if configured
    self._add_warm_start_trials()
  
  def _load_yaml_config(self, path: Path) -> Dict:
    """Load and validate YAML configuration.
    
    Args:
      path: Path to YAML file
      
    Returns:
      Parsed configuration dictionary
    """
    with open(path, 'r') as f:
      config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['optuna', 'hyperparameters']
    for section in required_sections:
      if section not in config:
        raise ValueError(f"Missing required section '{section}' in config file")
    
    return config
  
  def _create_study(self) -> optuna.Study:
    """Create or load Optuna study.
    
    Returns:
      Optuna Study object
    """
    optuna_config = self.config.get('optuna', {})
    
    # Create sampler
    sampler_config = optuna_config.get('sampler', {})
    sampler_type = sampler_config.get('type', 'TPESampler')
    
    if sampler_type == 'TPESampler':
      sampler = optuna.samplers.TPESampler(
        n_startup_trials=sampler_config.get('n_startup_trials', 10),
        seed=sampler_config.get('seed', 42)
      )
    else:
      sampler = None  # Use default
    
    # Create pruner
    pruner_config = optuna_config.get('pruner', {})
    pruner_type = pruner_config.get('type', 'MedianPruner')
    
    if pruner_type == 'MedianPruner':
      pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_config.get('n_startup_trials', 5),
        n_warmup_steps=pruner_config.get('n_warmup_steps', 100),
        interval_steps=pruner_config.get('interval_steps', 10)
      )
    else:
      pruner = None  # No pruning
    
    # Create or load study
    study = optuna.create_study(
      study_name=optuna_config.get('study_name', 'cellxgene_mlp_optimization'),
      direction=optuna_config.get('direction', 'minimize'),
      storage=self.storage,
      load_if_exists=optuna_config.get('load_if_exists', True),
      sampler=sampler,
      pruner=pruner
    )
    
    # Count trial states
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    logger.info(f"Created/loaded study '{study.study_name}'")
    logger.info(f"  Existing trials: {len(study.trials)} total")
    logger.info(f"    - Completed: {completed_trials}")
    logger.info(f"    - Failed: {failed_trials}")
    logger.info(f"    - Pruned: {pruned_trials}")
    
    if failed_trials > 0:
      logger.warning(f"Found {failed_trials} failed trials. These will not be retried automatically.")
      logger.info("To retry failed trials, you can:")
      logger.info("  1. Delete them from the database manually")
      logger.info("  2. Or increase n_trials to run additional trials")
    
    return study
  
  def _add_warm_start_trials(self):
    """Add warm-start configurations as completed trials."""
    best_configs = self.config.get('best_configs', [])
    
    if not best_configs:
      return
    
    # Sort by priority if specified (lower priority = added first)
    best_configs = sorted(
      best_configs,
      key=lambda x: x.get('priority', float('inf'))
    )
    
    optuna_metric = self.config['optuna'].get('metric', 'val120k_logloss')
    
    for cfg in best_configs:
      # Check if we should add this config
      if self._should_add_warm_start(cfg):
        # Create distributions for parameters
        distributions = self._create_distributions(cfg['hyperparameters'])
        
        # Get metric value if available
        value = None
        if 'metrics' in cfg and optuna_metric in cfg['metrics']:
          value = cfg['metrics'][optuna_metric]
        
        # Create and add trial
        trial = optuna.trial.create_trial(
          params=cfg['hyperparameters'],
          distributions=distributions,
          value=value,
          state=optuna.trial.TrialState.COMPLETE if value is not None else optuna.trial.TrialState.WAITING
        )
        
        self.study.add_trial(trial)
        logger.info(f"Added warm-start trial '{cfg['name']}' with {len(cfg['hyperparameters'])} parameters")
    
    # Handle WandB warm-start if configured
    self._add_wandb_warm_start()
  
  def _should_add_warm_start(self, config: Dict) -> bool:
    """Determine if a warm-start config should be added.
    
    Args:
      config: Warm-start configuration
      
    Returns:
      True if config should be added
    """
    # Always add for now; could implement progressive warm-start logic here
    return True
  
  def _create_distributions(self, params: Dict) -> Dict:
    """Create Optuna distributions for parameters.
    
    Args:
      params: Parameter values
      
    Returns:
      Dictionary of Optuna distributions
    """
    distributions = {}
    hyperparams_spec = self.config['hyperparameters']
    
    for param_name, param_value in params.items():
      if param_name not in hyperparams_spec:
        continue
      
      spec = hyperparams_spec[param_name]
      param_type = spec.get('type')
      
      if param_type == 'categorical':
        distributions[param_name] = optuna.distributions.CategoricalDistribution(
          choices=spec['choices']
        )
      elif param_type == 'float':
        # Convert to float in case YAML parsed as string (e.g., scientific notation)
        low_val = float(spec['low'])
        high_val = float(spec['high'])
        if spec.get('log', False):
          distributions[param_name] = optuna.distributions.FloatDistribution(
            low=low_val,
            high=high_val,
            log=True
          )
        else:
          distributions[param_name] = optuna.distributions.FloatDistribution(
            low=low_val,
            high=high_val
          )
      elif param_type == 'int':
        # Convert to int in case YAML parsed as string
        low_val = int(spec['low'])
        high_val = int(spec['high'])
        distributions[param_name] = optuna.distributions.IntDistribution(
          low=low_val,
          high=high_val
        )
    
    return distributions
  
  def _add_wandb_warm_start(self):
    """Load warm-start configurations from Weights & Biases."""
    warm_start_config = self.config.get('warm_start_strategy', {})
    auto_load = warm_start_config.get('auto_load', {})
    
    if not auto_load.get('enabled', False) or not WANDB_AVAILABLE:
      return
    
    if auto_load.get('from_wandb'):
      # Parse WandB URLs from best_configs
      for cfg in self.config.get('best_configs', []):
        if 'wandb_url' in cfg:
          self._load_from_wandb_url(cfg['wandb_url'])
  
  def _load_from_wandb_url(self, url: str):
    """Load configuration from WandB URL.
    
    Args:
      url: WandB run URL
    """
    if not WANDB_AVAILABLE:
      logger.warning(f"WandB not available, skipping URL: {url}")
      return
    
    try:
      # Parse URL to get entity, project, run_id
      # Format: https://wandb.ai/entity/project/runs/run_id
      parts = url.split('/')
      entity = parts[-4]
      project = parts[-3]
      run_id = parts[-1]
      
      api = wandb.Api()
      run = api.run(f"{entity}/{project}/{run_id}")
      
      # Extract hyperparameters from config
      params = {}
      for key in self.config['hyperparameters'].keys():
        if key in run.config:
          params[key] = run.config[key]
      
      # Get metric value
      optuna_metric = self.config['optuna'].get('metric', 'val120k_logloss')
      value = run.summary.get(optuna_metric)
      
      # Create trial
      distributions = self._create_distributions(params)
      trial = optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        value=value
      )
      
      self.study.add_trial(trial)
      logger.info(f"Added warm-start trial from WandB run {run_id}")
      
    except Exception as e:
      logger.warning(f"Failed to load from WandB URL {url}: {e}")
  
  def suggest_hyperparameters(self, trial: Trial) -> TrainingConfig:
    """Suggest hyperparameters for a trial.
    
    Args:
      trial: Optuna trial object
      
    Returns:
      TrainingConfig with suggested hyperparameters
    """
    params = {}
    hyperparams_spec = self.config['hyperparameters']
    
    for param_name, spec in hyperparams_spec.items():
      # Check for conditional parameters
      if 'condition' in spec:
        if not self._evaluate_condition(spec['condition'], params):
          continue
      
      # Suggest parameter based on type
      param_type = spec.get('type')
      
      if param_type == 'categorical':
        params[param_name] = trial.suggest_categorical(param_name, spec['choices'])
      elif param_type == 'float':
        # Convert to float in case YAML parsed as string (e.g., scientific notation)
        low_val = float(spec['low'])
        high_val = float(spec['high'])
        params[param_name] = trial.suggest_float(
          param_name,
          low_val,
          high_val,
          log=spec.get('log', False)
        )
      elif param_type == 'int':
        # Convert to int in case YAML parsed as string
        low_val = int(spec['low'])
        high_val = int(spec['high'])
        params[param_name] = trial.suggest_int(
          param_name,
          low_val,
          high_val
        )
    
    # Merge with fixed parameters
    fixed_params = self.config.get('fixed_params', {})
    all_params = {**fixed_params, **params}
    
    # Create TrainingConfig
    return self._create_training_config(all_params)
  
  def _evaluate_condition(self, condition: str, params: Dict) -> bool:
    """Evaluate a conditional parameter expression.
    
    Args:
      condition: Condition string (e.g., "lr_scheduler == 'cosine'")
      params: Current parameters
      
    Returns:
      True if condition is met
    """
    try:
      # Simple evaluation - in production, use safer evaluation
      return eval(condition, {"__builtins__": {}}, params)
    except:
      return False
  
  def _create_training_config(self, params: Dict) -> TrainingConfig:
    """Create TrainingConfig from parameters.
    
    Args:
      params: All parameters (fixed + suggested)
      
    Returns:
      TrainingConfig instance
    """
    # Map parameters to TrainingConfig fields
    config_kwargs = {}
    
    # Direct mappings (best_model_metric and best_model_mode are automatically derived from Optuna config)
    direct_mappings = [
      'n_dims', 'n_hidden_layers', 'dropout', 'learning_rate',
      'weight_decay', 'optimizer_type', 'lr_scheduler', 'gradient_clip_val',
      'batch_size', 'mixed_precision',
      's3_bucket', 's3_prefix', 'aws_profile',
      'eval_every_n_batches', 'eval_full_every_n_batches',
      'checkpoint_every_n_batches', 'device', 'num_workers',
      'seed', 'shuffle_files_per_epoch', 'shuffle_within_files',
      'enable_hierarchical_metrics', 'ontology_cache_dir', 'checkpoint_dir', 'resume_from',
      'start_batch_file', 'end_batch_file', 'max_steps_per_epoch',
      'wandb_save_artifacts', 'local_checkpoints',
      'wandb_project', 'wandb_entity', 'wandb_run_name',
      # Composable dataset parameters
      'use_composable_dataset', 'base_data_dir', 'embedding_types', 'genept_dims',
      # Cell type filtering parameters
      'cell_count_threshold', 'cell_counts_file', 'track_invalid_embeddings',
      # Other config parameters
      'epochs', 'verbose', 'profile_timing'
    ]
    
    for key in direct_mappings:
      if key in params:
        config_kwargs[key] = params[key]
    
    # Handle Path conversions
    path_fields = [
      'local_data_dir', 'test_data_dir', 'checkpoint_dir', 'ontology_cache_dir',
      'base_data_dir', 'cell_counts_file'
    ]
    for field in path_fields:
      if field in params and params[field] is not None:
        config_kwargs[field] = Path(params[field])
    
    # Set epochs for quick evaluation during tuning
    # Only override if not specified in fixed_params
    if 'epochs' not in params:
      config_kwargs['epochs'] = self.config['optuna'].get('n_epochs_per_trial', 2)
    
    # Automatically set best model tracking to match Optuna optimization metric
    # This ensures consistency between hyperparameter optimization and model saving
    optuna_metric = self.config['optuna'].get('metric', 'val120k_logloss')
    optuna_direction = self.config['optuna'].get('direction', 'minimize')
    
    # Warn if manually specified metrics conflict with Optuna settings
    if 'best_model_metric' in params and params['best_model_metric'] != optuna_metric:
      logger.warning(
        f"Overriding manually specified best_model_metric '{params['best_model_metric']}' "
        f"with Optuna optimization metric '{optuna_metric}' for consistency"
      )
    
    config_kwargs['best_model_metric'] = optuna_metric
    config_kwargs['best_model_mode'] = 'max' if optuna_direction == 'maximize' else 'min'
    
    # Automatically disable local checkpoints if WandB artifacts are enabled
    # This prevents filesystem collisions during hyperparameter optimization
    if config_kwargs.get('wandb_save_artifacts', True) and 'wandb_project' in params:
      config_kwargs['local_checkpoints'] = False
      logger.info("Disabled local checkpoints since WandB artifacts are enabled for hyperparameter optimization")
    
    return TrainingConfig(**config_kwargs)
  
  def run_optimization(
    self,
    trainer_factory: Callable[[Trial], Any],
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None
  ):
    """Run hyperparameter optimization.
    
    Args:
      trainer_factory: Function that creates and runs a trainer given a trial
      n_trials: Number of trials to run (overrides config)
      timeout: Maximum time in seconds (overrides config)
    """
    n_trials = n_trials or self.config['optuna'].get('n_trials', 100)
    # Note: timeout parameter to optimize() is TOTAL timeout, not per-trial
    # If we have timeout_per_trial in config, we should either:
    # 1. Not use it (set timeout=None)
    # 2. Or multiply by n_trials for total timeout
    # For now, we'll use the timeout parameter as total timeout if explicitly provided
    timeout = timeout  # Use command-line provided timeout as total timeout
    if timeout is None:
      # Don't use timeout_per_trial as total timeout - that's a bug!
      # Could do: timeout = self.config.get('resources', {}).get('timeout_per_trial') * n_trials
      # But for now, just don't set a total timeout
      timeout = None
    
    # Check how many successful trials we already have
    completed_count = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    failed_count = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    # By default, ensure we get n_trials SUCCESSFUL trials (not counting failed ones)
    # This means if we want 50 successful trials and already have 2 complete + 1 failed,
    # we'll run up to 48 more trials to get to 50 successful (the failed one doesn't count)
    
    # Get the target number of successful trials
    target_successful_trials = n_trials
    
    # Check if we should ensure successful trials (default: True)
    ensure_successful = self.config['optuna'].get('ensure_successful_trials', True)
    
    if ensure_successful and completed_count > 0:
      # Calculate how many more trials we need to reach the target
      remaining_trials_needed = max(0, target_successful_trials - completed_count)
      
      if remaining_trials_needed == 0:
        logger.info(f"Already have {completed_count} successful trials, reached target of {target_successful_trials}")
        return
      
      # Optuna's n_trials counts ALL trials, so we need to account for existing trials
      # We'll request enough trials to likely get the remaining successful ones
      # Add a buffer for potential failures (10% extra or at least 1)
      buffer = max(1, int(remaining_trials_needed * 0.1))
      n_trials_to_run = remaining_trials_needed + buffer
      
      logger.info(f"Target: {target_successful_trials} successful trials")
      logger.info(f"Already have: {completed_count} successful, {failed_count} failed")
      logger.info(f"Will run up to {n_trials_to_run} more trials to reach target")
      
      n_trials = n_trials_to_run
    else:
      logger.info(f"Starting optimization: requesting {n_trials} trials")
      if completed_count > 0 or failed_count > 0:
        logger.info(f"Already have {completed_count} completed, {failed_count} failed")
    logger.info(f"Total timeout: {timeout}s" if timeout else "No timeout")
    
    # Create objective function
    def objective(trial: Trial) -> float:
      try:
        # Create and run trainer
        trainer = trainer_factory(trial)
        metrics = trainer.run()

        metric_name = self.config['optuna'].get('metric', 'val120k_logloss')

        # Try to get the metric value
        value = metrics.get(metric_name)

        # If not found and metric name starts with 'val_', try with 'val120k_' prefix
        if value is None and metric_name.startswith('val_'):
          base_metric = metric_name[4:]  # Remove 'val_' prefix
          val120k_metric = f'val120k_{base_metric}'
          value = metrics.get(val120k_metric)
          if value is not None:
            logger.info(f"Using metric '{val120k_metric}' instead of '{metric_name}'")
            metric_name = val120k_metric

        if value is None:
          available_metrics = ', '.join(sorted(metrics.keys()))
          raise ValueError(
            f"Metric '{metric_name}' not found in trainer results. "
            f"Available metrics: {available_metrics}"
          )
        
        return value
        
      except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise
    
    # Add progress callback
    def progress_callback(study: optuna.Study, trial: FrozenTrial):
      logger.info(f"Trial {trial.number} finished with value: {trial.value}")
      if study.best_trial:
        logger.info(f"Best value so far: {study.best_value} (trial {study.best_trial.number})")
    
    # Run optimization
    self.study.optimize(
      objective,
      n_trials=n_trials,
      timeout=timeout,
      callbacks=[progress_callback]
    )
  
  def get_best_config(self) -> TrainingConfig:
    """Get the best configuration found.
    
    Returns:
      TrainingConfig with best hyperparameters
    """
    if not self.study.trials:
      raise ValueError("No trials completed yet")
    
    best_trial = self.study.best_trial
    best_params = best_trial.params
    
    # Merge with fixed parameters
    fixed_params = self.config.get('fixed_params', {})
    all_params = {**fixed_params, **best_params}
    
    # Create config with full epochs for final training
    config = self._create_training_config(all_params)
    
    # Override with full epochs for final training
    if 'epochs' in self.config.get('fixed_params', {}):
      config.epochs = self.config['fixed_params']['epochs']
    else:
      config.epochs = 10  # Default full training epochs
    
    return config
  
  def save_results(self, output_path: Path):
    """Save optimization results to file.
    
    Args:
      output_path: Path to save results
    """
    results = {
      'best_trial': {
        'number': self.study.best_trial.number,
        'value': self.study.best_value,
        'params': self.study.best_params
      },
      'n_trials': len(self.study.trials),
      'study_name': self.study.study_name
    }
    
    with open(output_path, 'w') as f:
      json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved optimization results to {output_path}")