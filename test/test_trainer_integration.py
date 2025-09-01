"""Integration tests for full training loop."""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.training.trainer import MLPTrainer
from src.training.config import TrainingConfig


class TestFullTrainingLoop:
  """Test complete training loop integration."""
  
  @patch('src.training.trainer.S3ParquetStreamDataset')
  def test_training_loop_with_mock_data(self, mock_dataset_class, tmp_path):
    """Test full training loop with mocked dataset."""
    # Create mock dataset that returns batches
    mock_dataset = MagicMock()
    mock_dataset_class.return_value = mock_dataset
    
    # Create small batches of data
    n_batches = 5
    batch_size = 32
    n_dims = 20
    n_classes = 5
    
    def mock_iter():
      for _ in range(n_batches):
        X = torch.randn(batch_size, n_dims)
        y = torch.randint(0, n_classes, (batch_size,))
        yield X, y
    
    mock_dataset.__iter__ = mock_iter
    
    # Setup trainer
    cell_types = ["type_" + str(i) for i in range(10)]
    cell_type_codes = pd.Series(range(n_classes))
    
    config = TrainingConfig(
      n_dims=n_dims,
      epochs=2,
      batch_size=batch_size,
      checkpoint_dir=tmp_path,
      eval_every_n_batches=2,
      checkpoint_every_n_batches=3,
      device="cpu",
      verbose=False,
      wandb_project=None  # No W&B for testing
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Set dummy validation data
    trainer.X_val_5k = np.random.randn(50, n_dims).astype(np.float32)
    trainer.y_val_5k = np.random.randint(0, n_classes, 50)
    trainer.X_val_120k = np.random.randn(120, n_dims).astype(np.float32)
    trainer.y_val_120k = np.random.randint(0, n_classes, 120)
    
    # Run training
    final_metrics = trainer.run()
    
    # Check that training completed
    assert 'logloss' in final_metrics
    assert 'macro_f1' in final_metrics
    assert 'recall_at_10' in final_metrics
    
    # Check that final checkpoint was saved
    final_checkpoint = tmp_path / "final_checkpoint.pt"
    assert final_checkpoint.exists()
    
    # Check that checkpoints were created during training
    checkpoints = list(tmp_path.glob("checkpoint_*.pt"))
    assert len(checkpoints) > 0
  
  def test_training_with_real_small_dataset(self, tmp_path):
    """Test training with a small real dataset."""
    # Create small training data
    n_samples = 100
    n_dims = 30
    n_classes = 3
    
    # Create and save small parquet files
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    
    for i in range(3):  # 3 small batch files
      data = {}
      for j in range(n_dims):
        data[str(j)] = np.random.randn(n_samples)
      
      # Add cell type column
      data['cell_type'] = ["type_" + str(j % n_classes) for j in range(n_samples)]
      
      df = pd.DataFrame(data)
      df.to_parquet(train_dir / f"batch_{i:04d}.parquet")
    
    # Create validation data
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    
    val_data = {}
    for j in range(n_dims):
      val_data[str(j)] = np.random.randn(50)
    val_data['cell_type'] = ["type_" + str(j % n_classes) for j in range(50)]
    
    val_df = pd.DataFrame(val_data)
    val_df.to_parquet(test_dir / "val_5k.parquet")
    
    # Setup trainer
    cell_types = ["type_" + str(i) for i in range(n_classes)]
    cell_type_codes = pd.Series(range(n_classes))
    
    config = TrainingConfig(
      n_dims=n_dims,
      epochs=1,
      batch_size=32,
      local_data_dir=train_dir,
      test_data_dir=test_dir,
      checkpoint_dir=tmp_path / "checkpoints",
      download_if_missing=False,  # Use local files only
      eval_every_n_batches=2,
      checkpoint_every_n_batches=5,
      device="cpu",
      verbose=True,
      wandb_project=None,
      end_batch_file=3  # Only use our 3 files
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Run training
    final_metrics = trainer.run()
    
    # Check that training completed successfully
    assert isinstance(final_metrics, dict)
    assert 'logloss' in final_metrics
    assert final_metrics['logloss'] > 0
    
    # Check model can make predictions
    trainer.model.eval()
    with torch.no_grad():
      test_input = torch.randn(10, n_dims)
      output = trainer.model(test_input)
      assert output.shape == (10, n_classes)
  
  def test_resume_from_checkpoint(self, tmp_path):
    """Test resuming training from checkpoint."""
    # Setup initial training
    cell_types = ["type_" + str(i) for i in range(5)]
    cell_type_codes = pd.Series(range(3))
    
    config = TrainingConfig(
      n_dims=20,
      epochs=2,
      checkpoint_dir=tmp_path,
      device="cpu",
      verbose=False,
      wandb_project=None
    )
    
    trainer1 = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Save a checkpoint
    checkpoint_path = tmp_path / "checkpoint_epoch1.pt"
    torch.save({
      'model_state_dict': trainer1.model.state_dict(),
      'optimizer_state_dict': trainer1.optimizer.state_dict(),
      'epoch': 1,
      'batch_idx': 50,
      'global_step': 150
    }, checkpoint_path)
    
    # Create new trainer and resume
    config2 = TrainingConfig(
      n_dims=20,
      epochs=3,
      checkpoint_dir=tmp_path,
      resume_from=checkpoint_path,
      device="cpu",
      verbose=False,
      wandb_project=None
    )
    
    trainer2 = MLPTrainer(
      config=config2,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Check that state was restored
    assert trainer2.start_epoch == 1
    assert trainer2.start_batch == 50
    assert trainer2.global_step == 150
    
    # Check model weights were restored
    for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
      assert torch.allclose(p1, p2)


class TestCheckpointingDuringTraining:
  """Test checkpoint saving during training."""
  
  @patch('src.training.trainer.S3ParquetStreamDataset')
  def test_checkpoint_best_model(self, mock_dataset_class, tmp_path):
    """Test that best model is saved based on validation metrics."""
    # Setup mock dataset
    mock_dataset = MagicMock()
    mock_dataset_class.return_value = mock_dataset
    
    # Create batches with decreasing loss pattern
    def mock_iter():
      for i in range(10):
        X = torch.randn(16, 20)
        y = torch.randint(0, 3, (16,))
        # Make loss decrease over time (simulated)
        yield X, y
    
    mock_dataset.__iter__ = mock_iter
    
    # Setup trainer
    cell_types = ["type_" + str(i) for i in range(5)]
    cell_type_codes = pd.Series(range(3))
    
    config = TrainingConfig(
      n_dims=20,
      epochs=1,
      batch_size=16,
      checkpoint_dir=tmp_path,
      eval_every_n_batches=2,
      checkpoint_every_n_batches=3,
      device="cpu",
      verbose=False,
      wandb_project=None
    )
    
    trainer = MLPTrainer(
      config=config,
      cell_types=cell_types,
      cell_type_codes=cell_type_codes
    )
    
    # Mock validation data with improving metrics
    trainer.X_val_5k = np.random.randn(30, 20).astype(np.float32)
    trainer.y_val_5k = np.random.randint(0, 3, 30)
    
    # Patch evaluate_validation to return improving metrics
    original_eval = trainer.evaluate_validation
    eval_count = [0]
    
    def mock_eval(use_5k=True):
      eval_count[0] += 1
      # Return progressively better metrics
      return {
        'logloss': 1.0 - 0.1 * eval_count[0],
        'macro_f1': 0.5 + 0.05 * eval_count[0],
        'recall_at_10': 0.6 + 0.05 * eval_count[0],
        'mrr_at_10': 0.4 + 0.05 * eval_count[0],
        'dcg_at_10': 0.5 + 0.05 * eval_count[0]
      }
    
    trainer.evaluate_validation = mock_eval
    
    # Run training
    trainer.run()
    
    # Check that best model was saved
    best_model_path = tmp_path / "best_model.pt"
    assert best_model_path.exists()
    
    # Check that best metrics were saved
    best_metrics_path = tmp_path / "best_model_metrics.json"
    assert best_metrics_path.exists()