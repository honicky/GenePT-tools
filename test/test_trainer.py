"""Unit tests for the MLPTrainer class."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile

from src.training.trainer import MLPTrainer
from src.training.config import TrainingConfig
from src.models.mlp_classifier import MLPClassifier


class TestMLPTrainerInit:
  """Test MLPTrainer initialization."""
  
  def test_basic_initialization(self):
    """Test basic trainer initialization."""
    # Setup
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      n_dims=100,
      n_hidden_layers=2,
      dropout=0.1,
      device="cpu",
      load_validation_data=False
    )
    
    # Create trainer
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Verify initialization
    assert trainer.config == config
    assert trainer.cell_types == cell_types
    assert trainer.num_classes == 50  # len(cell_type_codes)
    assert isinstance(trainer.model, MLPClassifier)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.global_step == 0
    assert trainer.start_epoch == 0
    assert trainer.start_batch == 0
  
  def test_model_creation(self):
    """Test that model is created with correct architecture."""
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      n_dims=200,
      n_hidden_layers=3,
      dropout=0.05,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Check model architecture
    assert trainer.model.input_dim == 200
    assert trainer.model.num_classes == 50
    assert trainer.model.n_hidden_layers == 3
    assert trainer.model.dropout == 0.05
  
  def test_optimizer_creation(self):
    """Test optimizer initialization with weight decay."""
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      learning_rate=1e-3,
      weight_decay=1e-4,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Check optimizer configuration
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    # Check that we have two param groups (decay and no decay)
    assert len(trainer.optimizer.param_groups) == 2
    
    # Check learning rate
    for group in trainer.optimizer.param_groups:
      assert group['lr'] == 1e-3
    
    # Check weight decay settings
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 1e-4
    assert trainer.optimizer.param_groups[1]['weight_decay'] == 0.0


class TestTrainBatch:
  """Test single batch training."""
  
  def test_train_single_batch(self):
    """Test training on a single batch."""
    # Setup
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      n_dims=64,
      n_hidden_layers=1,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Create dummy batch
    batch_size = 32
    X = torch.randn(batch_size, 64)
    y = torch.randint(0, 50, (batch_size,))
    
    # Train on batch
    loss = trainer.train_batch(X, y)
    
    # Check that loss is returned and reasonable
    assert isinstance(loss, float)
    assert loss > 0
    assert loss < 10  # Reasonable range for cross-entropy
  
  def test_train_batch_updates_weights(self):
    """Test that training actually updates model weights."""
    # Setup
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      n_dims=64,
      n_hidden_layers=1,
      learning_rate=0.1,  # High learning rate for visible changes
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Save initial weights
    initial_weights = {}
    for name, param in trainer.model.named_parameters():
      initial_weights[name] = param.clone().detach()
    
    # Train on batch
    X = torch.randn(32, 64)
    y = torch.randint(0, 50, (32,))
    trainer.train_batch(X, y)
    
    # Check that weights changed
    weights_changed = False
    for name, param in trainer.model.named_parameters():
      if not torch.allclose(param, initial_weights[name]):
        weights_changed = True
        break
    
    assert weights_changed, "Model weights should change after training"
  
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  def test_train_batch_cuda(self):
    """Test training on CUDA device."""
    cell_types = ["type_" + str(i) for i in range(100)]
    cell_type_codes = pd.Series(range(50))
    
    config = TrainingConfig(
      n_dims=64,
      device="cuda"
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Create batch (on CPU, should be moved to CUDA internally)
    X = torch.randn(32, 64)
    y = torch.randint(0, 50, (32,))
    
    # Train on batch
    loss = trainer.train_batch(X, y)
    
    assert isinstance(loss, float)
    assert loss > 0


class TestValidationDataLoading:
  """Test validation data loading."""
  
  def test_process_validation_df(self):
    """Test processing validation dataframe."""
    # Create dummy validation data
    n_samples = 100
    n_dims = 50
    
    # Create DataFrame with embeddings and cell types
    data = {}
    for i in range(n_dims):
      data[str(i)] = np.random.randn(n_samples)
    
    cell_types = ["type_" + str(i % 10) for i in range(n_samples)]
    data['cell_type'] = cell_types
    
    df = pd.DataFrame(data)
    
    # Setup trainer
    all_cell_types = ["type_" + str(i) for i in range(20)]
    cell_type_codes = pd.Series(range(10))
    
    config = TrainingConfig(
      n_dims=n_dims,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=all_cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Process validation data
    X, y = trainer._process_validation_df(df)
    
    # Check outputs
    assert X.shape == (n_samples, n_dims)
    assert len(y) == n_samples
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert np.all(y >= 0)
    assert np.all(y < 10)
  
  def test_load_validation_data_from_test_dir(self, tmp_path):
    """Test loading validation data from test directory."""
    # Create dummy validation files
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    # Create validation data
    n_samples = 50
    n_dims = 30
    
    data = {}
    for i in range(n_dims):
      data[str(i)] = np.random.randn(n_samples)
    data['cell_type'] = ["type_" + str(i % 5) for i in range(n_samples)]
    
    df = pd.DataFrame(data)
    
    # Save as parquet files
    df.to_parquet(test_dir / "val_5k.parquet")
    df.to_parquet(test_dir / "val_120k.parquet")
    
    # Setup trainer
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(5))
    
    config = TrainingConfig(
      n_dims=n_dims,
      test_data_dir=test_dir,
      device="cpu",
      load_validation_data=True  # This test needs to load validation data
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Check that validation data was loaded
    assert trainer.X_val_5k is not None
    assert trainer.y_val_5k is not None
    assert trainer.X_val_120k is not None
    assert trainer.y_val_120k is not None
    
    assert trainer.X_val_5k.shape[0] == n_samples
    assert trainer.X_val_5k.shape[1] == n_dims


class TestEvaluation:
  """Test evaluation during training."""
  
  def test_evaluate_validation(self):
    """Test validation evaluation."""
    # Setup trainer with dummy validation data
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(5))
    
    config = TrainingConfig(
      n_dims=20,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Set dummy validation data
    trainer.X_val_5k = np.random.randn(50, 20).astype(np.float32)
    trainer.y_val_5k = np.random.randint(0, 5, 50)
    
    # Evaluate
    metrics = trainer.evaluate_validation(use_5k=True)
    
    # Check metrics
    assert 'logloss' in metrics
    assert 'macro_f1' in metrics
    assert 'recall_at_10' in metrics
    assert 'mrr_at_10' in metrics
    assert 'dcg_at_10' in metrics
    
    # Check metric values are reasonable
    assert metrics['logloss'] > 0
    assert 0 <= metrics['macro_f1'] <= 1
    assert 0 <= metrics['recall_at_10'] <= 1


class TestCheckpointing:
  """Test checkpoint save/load functionality."""
  
  def test_load_checkpoint(self, tmp_path):
    """Test loading from checkpoint."""
    # Create trainer
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(5))
    
    config = TrainingConfig(
      n_dims=20,
      device="cpu",
      load_validation_data=False
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Save a checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save({
      'model_state_dict': trainer.model.state_dict(),
      'optimizer_state_dict': trainer.optimizer.state_dict(),
      'epoch': 5,
      'batch_idx': 100,
      'global_step': 500
    }, checkpoint_path)
    
    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    
    # Check state was restored
    assert trainer.start_epoch == 5
    assert trainer.start_batch == 100
    assert trainer.global_step == 500


class TestWandBIntegration:
  """Test Weights & Biases integration."""
  
  @patch('src.training.trainer.wandb')
  def test_init_wandb(self, mock_wandb):
    """Test W&B initialization."""
    # Setup mock
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run
    
    # Create trainer with W&B config
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(5))
    
    config = TrainingConfig(
      n_dims=20,
      device="cpu",
      load_validation_data=False,
      wandb_project="test_project",
      wandb_entity="test_entity",
      wandb_run_name="test_run",
      wandb_tags=["test", "unit"]
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Check W&B was initialized
    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs['project'] == "test_project"
    assert call_kwargs['entity'] == "test_entity"
    assert call_kwargs['name'] == "test_run"
    assert call_kwargs['tags'] == ["test", "unit"]
    
    # Check model was watched
    mock_wandb.watch.assert_called_once()
  
  @patch('src.training.trainer.WANDB_AVAILABLE', False)
  @patch('src.training.trainer.wandb', None)
  def test_wandb_import_error(self):
    """Test graceful handling when wandb is not installed."""
    # Create trainer with W&B config
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(5))
    
    config = TrainingConfig(
      n_dims=20,
      device="cpu",
      load_validation_data=False,
      wandb_project="test_project",
      test_data_dir=Path("/nonexistent"),  # Avoid loading real test data
      verbose=False
    )
    
    # Should not raise error
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # W&B run should be None
    assert trainer.wandb_run is None