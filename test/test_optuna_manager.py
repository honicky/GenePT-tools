"""Unit tests for OptunaManager class."""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Check if optuna is available
try:
  import optuna
  from src.training.optuna_manager import OptunaManager
  from src.training.config import TrainingConfig
  OPTUNA_AVAILABLE = True
except ImportError:
  OPTUNA_AVAILABLE = False


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptunaManager:
  """Tests for OptunaManager functionality."""
  
  @pytest.fixture
  def basic_config(self):
    """Create a basic YAML configuration for testing."""
    config = {
      'optuna': {
        'study_name': 'test_study',
        'direction': 'minimize',
        'metric_to_optimize': 'val_loss',
        'n_trials': 5,
        'n_epochs_per_trial': 1,
        'sampler': {
          'type': 'TPESampler',
          'n_startup_trials': 2,
          'seed': 42
        },
        'pruner': {
          'type': 'MedianPruner',
          'n_startup_trials': 2,
          'n_warmup_steps': 5,
          'interval_steps': 2
        }
      },
      'hyperparameters': {
        'n_dims': {
          'type': 'categorical',
          'choices': [100, 200, 500],
          'default': 500
        },
        'n_hidden_layers': {
          'type': 'int',
          'low': 1,
          'high': 3,
          'default': 2
        },
        'dropout': {
          'type': 'float',
          'low': 0.0,
          'high': 0.5,
          'default': 0.1,
          'log': False
        },
        'learning_rate': {
          'type': 'float',
          'low': 1e-5,
          'high': 1e-2,
          'default': 1e-3,
          'log': True
        }
      },
      'fixed_params': {
        'batch_size': 128,
        'epochs': 2,
        'device': 'cpu'
      }
    }
    return config
  
  @pytest.fixture
  def config_with_warm_start(self, basic_config):
    """Create config with warm-start settings."""
    config = basic_config.copy()
    config['best_configs'] = [
      {
        'name': 'previous_best',
        'hyperparameters': {
          'n_dims': 500,
          'n_hidden_layers': 2,
          'dropout': 0.05,
          'learning_rate': 1e-4
        },
        'metrics': {
          'val_loss': 0.5,
          'val_accuracy': 0.85
        },
        'priority': 1
      }
    ]
    return config
  
  def test_load_yaml_config(self, basic_config):
    """Test loading and validating YAML configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(basic_config, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      assert manager.config == basic_config
      assert manager.study is not None
    finally:
      config_path.unlink()
  
  def test_missing_required_sections(self):
    """Test that missing required sections raise errors."""
    incomplete_config = {
      'optuna': {
        'study_name': 'test_study',
        'direction': 'minimize'
      }
      # Missing 'hyperparameters' section
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(incomplete_config, f)
      config_path = Path(f.name)
    
    try:
      with pytest.raises(ValueError, match="Missing required section 'hyperparameters'"):
        OptunaManager(config_path)
    finally:
      config_path.unlink()
  
  def test_create_study(self, basic_config):
    """Test study creation with correct settings."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(basic_config, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      assert manager.study.study_name == 'test_study'
      assert manager.study.direction == optuna.study.StudyDirection.MINIMIZE
      assert isinstance(manager.study.sampler, optuna.samplers.TPESampler)
      assert isinstance(manager.study.pruner, optuna.pruners.MedianPruner)
    finally:
      config_path.unlink()
  
  def test_suggest_hyperparameters(self, basic_config):
    """Test hyperparameter suggestion."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(basic_config, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      
      # Create a trial
      trial = manager.study.ask()
      
      # Suggest hyperparameters
      config = manager.suggest_hyperparameters(trial)
      
      # Check that we got a TrainingConfig
      assert isinstance(config, TrainingConfig)
      
      # Check that parameters are within specified ranges
      assert config.n_dims in [100, 200, 500]
      assert 1 <= config.n_hidden_layers <= 3
      assert 0.0 <= config.dropout <= 0.5
      assert 1e-5 <= config.learning_rate <= 1e-2
      
      # Check fixed parameters
      assert config.batch_size == 128
      assert config.epochs == 1  # n_epochs_per_trial
      assert config.device == 'cpu'
    finally:
      config_path.unlink()
  
  def test_warm_start_trials(self, config_with_warm_start):
    """Test adding warm-start trials."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(config_with_warm_start, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      
      # Check that warm-start trial was added
      assert len(manager.study.trials) == 1
      
      # Check warm-start trial parameters
      trial = manager.study.trials[0]
      assert trial.params['n_dims'] == 500
      assert trial.params['n_hidden_layers'] == 2
      assert trial.params['dropout'] == 0.05
      assert trial.params['learning_rate'] == 1e-4
      assert trial.value == 0.5  # val_loss from metrics
    finally:
      config_path.unlink()
  
  def test_create_distributions(self, basic_config):
    """Test creation of Optuna distributions."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(basic_config, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      
      params = {
        'n_dims': 500,
        'n_hidden_layers': 2,
        'dropout': 0.1,
        'learning_rate': 1e-3
      }
      
      distributions = manager._create_distributions(params)
      
      # Check distribution types
      assert isinstance(distributions['n_dims'], optuna.distributions.CategoricalDistribution)
      assert isinstance(distributions['n_hidden_layers'], optuna.distributions.IntDistribution)
      assert isinstance(distributions['dropout'], optuna.distributions.FloatDistribution)
      assert isinstance(distributions['learning_rate'], optuna.distributions.FloatDistribution)
      
      # Check log scale
      assert distributions['learning_rate'].log is True
      assert distributions['dropout'].log is False
    finally:
      config_path.unlink()
  
  def test_get_best_config(self, config_with_warm_start):
    """Test getting the best configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(config_with_warm_start, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      
      # Add a few more trials
      for _ in range(3):
        trial = manager.study.ask()
        manager.study.tell(trial, values=0.6)  # Worse than warm-start
      
      # Get best config
      best_config = manager.get_best_config()
      
      assert isinstance(best_config, TrainingConfig)
      # Should return warm-start config as it has the best value
      assert best_config.n_dims == 500
      assert best_config.n_hidden_layers == 2
      assert best_config.dropout == 0.05
      assert best_config.learning_rate == 1e-4
      # Should use full epochs for final training
      assert best_config.epochs == 2  # From fixed_params
    finally:
      config_path.unlink()
  
  def test_conditional_parameters(self):
    """Test conditional parameter handling."""
    config = {
      'optuna': {
        'study_name': 'test_study',
        'direction': 'minimize',
        'metric_to_optimize': 'val_loss'
      },
      'hyperparameters': {
        'lr_scheduler': {
          'type': 'categorical',
          'choices': ['none', 'cosine', 'step']
        },
        'step_size': {
          'type': 'int',
          'low': 100,
          'high': 1000,
          'condition': "lr_scheduler == 'step'"
        }
      },
      'fixed_params': {}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(config, f)
      config_path = Path(f.name)
    
    try:
      manager = OptunaManager(config_path)
      
      # Test that conditional parameters are only suggested when condition is met
      for _ in range(5):
        trial = manager.study.ask()
        config = manager.suggest_hyperparameters(trial)
        
        # If lr_scheduler is not 'step', step_size should not be in the trial params
        if trial.params.get('lr_scheduler') != 'step':
          assert 'step_size' not in trial.params
    finally:
      config_path.unlink()


def test_optuna_not_available():
  """Test behavior when Optuna is not available."""
  # This test runs regardless of Optuna availability
  if not OPTUNA_AVAILABLE:
    # Should not be able to import OptunaManager
    with pytest.raises(NameError):
      manager = OptunaManager(Path("dummy.yaml"))  # noqa: F821