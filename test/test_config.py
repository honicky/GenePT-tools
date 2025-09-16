"""Unit tests for configuration classes."""

import pytest
from pathlib import Path
import tempfile

from src.training.config import TrainingConfig, OptunaConfig


class TestTrainingConfig:
  """Test TrainingConfig dataclass."""
  
  def test_default_initialization(self):
    """Test TrainingConfig with default values."""
    config = TrainingConfig()
    
    # Check default values from notebook's best run
    assert config.n_dims == 500
    assert config.n_hidden_layers == 3
    assert config.dropout == 0.053
    assert config.learning_rate == 4.366e-5
    assert config.batch_size == 1024
    assert config.epochs == 10
    
    # Check paths are Path objects
    assert isinstance(config.data_dir, Path)
    assert isinstance(config.checkpoint_dir, Path)
    
    # Check defaults
    assert config.s3_bucket == "pythiomicsdata"
    assert config.s3_prefix == "cellxgene_v2/training_v1_suffled"
    assert config.aws_profile is None
    assert config.seed == 42
  
  def test_custom_initialization(self):
    """Test TrainingConfig with custom values."""
    config = TrainingConfig(
      n_dims=300,
      batch_size=512,
      learning_rate=1e-3,
      epochs=5,
      local_data_dir="/custom/path",
      wandb_project="test_project"
    )
    
    assert config.n_dims == 300
    assert config.batch_size == 512
    assert config.learning_rate == 1e-3
    assert config.epochs == 5
    assert config.local_data_dir == Path("/custom/path")
    assert config.wandb_project == "test_project"
  
  def test_path_conversion(self):
    """Test that string paths are converted to Path objects."""
    config = TrainingConfig(
      data_dir="./data",
      checkpoint_dir="./checkpoints",
      local_data_dir="./cache",
      test_data_dir="./test_data",
      resume_from="./checkpoint.pt"
    )
    
    assert isinstance(config.data_dir, Path)
    assert isinstance(config.checkpoint_dir, Path)
    assert isinstance(config.local_data_dir, Path)
    assert isinstance(config.test_data_dir, Path)
    assert isinstance(config.resume_from, Path)
  
  def test_device_auto_detection(self):
    """Test automatic device detection."""
    import torch
    
    config = TrainingConfig(device="auto")
    
    if torch.cuda.is_available():
      assert config.device == "cuda"
    elif torch.backends.mps.is_available():
      assert config.device == "mps"
    else:
      assert config.device == "cpu"
  
  def test_device_explicit(self):
    """Test explicit device setting."""
    config = TrainingConfig(device="cpu")
    assert config.device == "cpu"
    
    # Should not change even if auto would select something else
    config2 = TrainingConfig(device="cuda")
    assert config2.device == "cuda"
  
  def test_to_dict(self):
    """Test conversion to dictionary for logging."""
    config = TrainingConfig(
      n_dims=300,
      batch_size=512,
      checkpoint_dir=Path("/tmp/checkpoints"),
      wandb_tags=["test", "experiment"]
    )
    
    config_dict = config.to_dict()
    
    # Check values are preserved
    assert config_dict['n_dims'] == 300
    assert config_dict['batch_size'] == 512
    assert config_dict['wandb_tags'] == ["test", "experiment"]
    
    # Check paths are converted to strings
    assert config_dict['checkpoint_dir'] == "/tmp/checkpoints"
    assert isinstance(config_dict['checkpoint_dir'], str)
  
  def test_evaluation_parameters(self):
    """Test evaluation-related parameters."""
    config = TrainingConfig(
      eval_every_n_batches=20,
      eval_full_every_n_batches=500,
      checkpoint_every_n_batches=2000
    )
    
    assert config.eval_every_n_batches == 20
    assert config.eval_full_every_n_batches == 500
    assert config.checkpoint_every_n_batches == 2000
  
  def test_subset_parameters(self):
    """Test parameters for working with data subsets."""
    config = TrainingConfig(
      start_batch_file=10,
      end_batch_file=20
    )
    
    assert config.start_batch_file == 10
    assert config.end_batch_file == 20
  
  def test_shuffling_parameters(self):
    """Test shuffling configuration."""
    config = TrainingConfig(
      shuffle_files_per_epoch=False,
      shuffle_within_files=False
    )
    
    assert config.shuffle_files_per_epoch is False
    assert config.shuffle_within_files is False


class TestOptunaConfig:
  """Test OptunaConfig for hyperparameter optimization."""
  
  def test_default_initialization(self):
    """Test OptunaConfig with default values."""
    config = OptunaConfig()
    
    assert config.learning_rate_min == 1e-5
    assert config.learning_rate_max == 1e-2
    assert config.dropout_min == 0.0
    assert config.dropout_max == 0.5
    assert config.n_hidden_layers_min == 1
    assert config.n_hidden_layers_max == 4
    assert config.batch_sizes == [512, 1024, 2048]
    
    assert config.n_trials == 100
    assert config.n_epochs_per_trial == 2
    assert config.metric_to_optimize == "val_loss"
    assert config.direction == "minimize"
  
  def test_custom_initialization(self):
    """Test OptunaConfig with custom values."""
    config = OptunaConfig(
      learning_rate_min=1e-4,
      learning_rate_max=1e-3,
      n_trials=50,
      metric_to_optimize="macro_f1",
      direction="maximize",
      study_name="custom_study"
    )
    
    assert config.learning_rate_min == 1e-4
    assert config.learning_rate_max == 1e-3
    assert config.n_trials == 50
    assert config.metric_to_optimize == "macro_f1"
    assert config.direction == "maximize"
    assert config.study_name == "custom_study"
  
  def test_suggest_config(self):
    """Test config suggestion for Optuna trial."""
    # Mock trial object
    class MockTrial:
      def __init__(self):
        self.suggestions = {}
      
      def suggest_loguniform(self, name, low, high):
        self.suggestions[name] = (low + high) / 2
        return self.suggestions[name]
      
      def suggest_uniform(self, name, low, high):
        self.suggestions[name] = (low + high) / 2
        return self.suggestions[name]
      
      def suggest_int(self, name, low, high):
        self.suggestions[name] = (low + high) // 2
        return self.suggestions[name]
      
      def suggest_categorical(self, name, choices):
        self.suggestions[name] = choices[0]
        return self.suggestions[name]
    
    # Create base config
    base_config = TrainingConfig(
      epochs=10,
      n_dims=500
    )
    
    # Create Optuna config
    optuna_config = OptunaConfig(
      n_epochs_per_trial=3
    )
    
    # Create mock trial
    trial = MockTrial()
    
    # Suggest config
    suggested = optuna_config.suggest_config(trial, base_config)
    
    # Check that base values are preserved
    assert suggested.n_dims == 500
    
    # Check that trial values are used
    assert suggested.learning_rate > optuna_config.learning_rate_min
    assert suggested.learning_rate < optuna_config.learning_rate_max
    assert suggested.dropout >= optuna_config.dropout_min
    assert suggested.dropout <= optuna_config.dropout_max
    assert suggested.n_hidden_layers >= optuna_config.n_hidden_layers_min
    assert suggested.n_hidden_layers <= optuna_config.n_hidden_layers_max
    assert suggested.batch_size in optuna_config.batch_sizes
    
    # Check epochs overridden for optimization
    assert suggested.epochs == 3
  
  def test_storage_configuration(self):
    """Test storage configuration for Optuna studies."""
    config = OptunaConfig(
      storage="sqlite:///optuna.db",
      load_if_exists=False
    )
    
    assert config.storage == "sqlite:///optuna.db"
    assert config.load_if_exists is False