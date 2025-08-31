"""Unit tests for checkpoint utilities."""

import pytest
import torch
import torch.nn as nn
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.utils.checkpoint import (
  save_checkpoint, load_checkpoint, save_best_model, CheckpointManager
)
from src.training.config import TrainingConfig


class TestSaveLoadCheckpoint:
  """Test checkpoint saving and loading."""
  
  def test_save_and_load_basic(self, tmp_path):
    """Test basic checkpoint save and load."""
    # Create simple model and optimizer
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Save checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    saved_path = save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=5,
      batch_idx=100,
      global_step=500
    )
    
    assert saved_path.exists()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Verify contents
    assert checkpoint['epoch'] == 5
    assert checkpoint['batch_idx'] == 100
    assert checkpoint['global_step'] == 500
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'timestamp' in checkpoint
  
  def test_save_with_metrics(self, tmp_path):
    """Test saving checkpoint with metrics."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    best_metrics = {
      'val_loss': 0.123,
      'val_accuracy': 0.95
    }
    
    checkpoint_path = tmp_path / "checkpoint_with_metrics.pt"
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=10,
      batch_idx=200,
      global_step=1000,
      best_metrics=best_metrics
    )
    
    # Load and verify
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint['best_metrics'] == best_metrics
  
  def test_save_with_config(self, tmp_path):
    """Test saving checkpoint with configuration."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    config = TrainingConfig(
      learning_rate=0.001,
      batch_size=32,
      epochs=10
    )
    
    checkpoint_path = tmp_path / "checkpoint_with_config.pt"
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=1,
      batch_idx=0,
      global_step=0,
      config=config
    )
    
    # Check that config JSON was also saved
    config_path = tmp_path / "checkpoint_with_config_config.json"
    assert config_path.exists()
    
    with open(config_path) as f:
      saved_config = json.load(f)
    
    assert saved_config['learning_rate'] == 0.001
    assert saved_config['batch_size'] == 32
    assert saved_config['epochs'] == 10
  
  def test_load_into_model_and_optimizer(self, tmp_path):
    """Test loading checkpoint directly into model and optimizer."""
    # Create and save original model
    model1 = nn.Sequential(
      nn.Linear(10, 20),
      nn.ReLU(),
      nn.Linear(20, 5)
    )
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    
    # Modify model weights
    with torch.no_grad():
      for param in model1.parameters():
        param.add_(1.0)
    
    # Take an optimizer step to change its state
    loss = model1(torch.randn(2, 10)).sum()
    loss.backward()
    optimizer1.step()
    
    # Save checkpoint
    checkpoint_path = tmp_path / "model_checkpoint.pt"
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model1,
      optimizer=optimizer1,
      epoch=3,
      batch_idx=50,
      global_step=150
    )
    
    # Create new model and optimizer with different state
    model2 = nn.Sequential(
      nn.Linear(10, 20),
      nn.ReLU(),
      nn.Linear(20, 5)
    )
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    # Load checkpoint into new model and optimizer
    checkpoint = load_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model2,
      optimizer=optimizer2
    )
    
    # Verify model weights are restored
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
      assert torch.allclose(p1, p2)
    
    # Verify checkpoint data
    assert checkpoint['epoch'] == 3
    assert checkpoint['batch_idx'] == 50
    assert checkpoint['global_step'] == 150
  
  def test_load_with_device(self, tmp_path):
    """Test loading checkpoint with device mapping."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint_path = tmp_path / "device_checkpoint.pt"
    save_checkpoint(
      checkpoint_path=checkpoint_path,
      model=model,
      optimizer=optimizer,
      epoch=1,
      batch_idx=10,
      global_step=10
    )
    
    # Load to CPU explicitly
    checkpoint = load_checkpoint(
      checkpoint_path=checkpoint_path,
      device=torch.device("cpu")
    )
    
    assert 'model_state_dict' in checkpoint


class TestSaveBestModel:
  """Test best model saving functionality."""
  
  def test_save_best_model_min_mode(self, tmp_path):
    """Test saving best model with minimize mode."""
    model = nn.Linear(10, 5)
    
    # First save - should save as it's the first
    metrics1 = {'val_loss': 0.5, 'val_acc': 0.8}
    best = save_best_model(
      model=model,
      metrics=metrics1,
      metric_name='val_loss',
      checkpoint_dir=tmp_path,
      mode='min',
      current_best=None
    )
    
    assert best == 0.5
    assert (tmp_path / 'best_model.pt').exists()
    assert (tmp_path / 'best_model_metrics.json').exists()
    
    # Second save with worse metric - should not save
    metrics2 = {'val_loss': 0.6, 'val_acc': 0.85}
    best = save_best_model(
      model=model,
      metrics=metrics2,
      metric_name='val_loss',
      checkpoint_dir=tmp_path,
      mode='min',
      current_best=0.5
    )
    
    assert best == 0.5  # Should return previous best
    
    # Third save with better metric - should save
    metrics3 = {'val_loss': 0.4, 'val_acc': 0.9}
    best = save_best_model(
      model=model,
      metrics=metrics3,
      metric_name='val_loss',
      checkpoint_dir=tmp_path,
      mode='min',
      current_best=0.5
    )
    
    assert best == 0.4
    
    # Verify saved metrics
    with open(tmp_path / 'best_model_metrics.json') as f:
      saved_metrics = json.load(f)
    assert saved_metrics['val_loss'] == 0.4
    assert saved_metrics['val_acc'] == 0.9
  
  def test_save_best_model_max_mode(self, tmp_path):
    """Test saving best model with maximize mode."""
    model = nn.Linear(10, 5)
    
    # First save
    metrics1 = {'accuracy': 0.8}
    best = save_best_model(
      model=model,
      metrics=metrics1,
      metric_name='accuracy',
      checkpoint_dir=tmp_path,
      mode='max',
      current_best=None
    )
    
    assert best == 0.8
    
    # Better metric
    metrics2 = {'accuracy': 0.9}
    best = save_best_model(
      model=model,
      metrics=metrics2,
      metric_name='accuracy',
      checkpoint_dir=tmp_path,
      mode='max',
      current_best=0.8
    )
    
    assert best == 0.9
    
    # Worse metric
    metrics3 = {'accuracy': 0.85}
    best = save_best_model(
      model=model,
      metrics=metrics3,
      metric_name='accuracy',
      checkpoint_dir=tmp_path,
      mode='max',
      current_best=0.9
    )
    
    assert best == 0.9  # Should keep previous best
  
  def test_missing_metric(self, tmp_path):
    """Test handling of missing metric."""
    model = nn.Linear(10, 5)
    metrics = {'other_metric': 0.5}
    
    best = save_best_model(
      model=model,
      metrics=metrics,
      metric_name='val_loss',
      checkpoint_dir=tmp_path,
      mode='min',
      current_best=None
    )
    
    assert best is None
    assert not (tmp_path / 'best_model.pt').exists()


class TestCheckpointManager:
  """Test CheckpointManager class."""
  
  def test_initialization(self, tmp_path):
    """Test CheckpointManager initialization."""
    manager = CheckpointManager(
      checkpoint_dir=tmp_path,
      save_every_n_batches=100,
      keep_last_n=3,
      track_metric='val_loss',
      metric_mode='min'
    )
    
    assert manager.checkpoint_dir == tmp_path
    assert manager.save_every_n_batches == 100
    assert manager.keep_last_n == 3
    assert manager.track_metric == 'val_loss'
    assert manager.metric_mode == 'min'
    assert manager.best_metric is None
    assert manager.checkpoints == []
  
  def test_should_save(self):
    """Test should_save logic."""
    manager = CheckpointManager(
      checkpoint_dir=Path("/tmp"),
      save_every_n_batches=100
    )
    
    assert not manager.should_save(50)
    assert manager.should_save(100)
    assert not manager.should_save(150)
    assert manager.should_save(200)
  
  def test_save_and_cleanup(self, tmp_path):
    """Test saving checkpoints and cleanup of old ones."""
    manager = CheckpointManager(
      checkpoint_dir=tmp_path,
      save_every_n_batches=1,
      keep_last_n=2
    )
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save 3 checkpoints (should keep only last 2)
    paths = []
    for i in range(3):
      path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=i,
        batch_idx=i*10,
        global_step=i*10
      )
      paths.append(path)
    
    # First checkpoint should be deleted
    assert not paths[0].exists()
    # Last two should exist
    assert paths[1].exists()
    assert paths[2].exists()
    
    # Manager should track only last 2
    assert len(manager.checkpoints) == 2
  
  def test_best_model_tracking(self, tmp_path):
    """Test best model tracking in checkpoint manager."""
    manager = CheckpointManager(
      checkpoint_dir=tmp_path,
      save_every_n_batches=1,
      track_metric='val_loss',
      metric_mode='min'
    )
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save with improving metrics
    metrics1 = {'val_loss': 0.5}
    manager.save(
      model=model,
      optimizer=optimizer,
      epoch=0,
      batch_idx=0,
      global_step=0,
      metrics=metrics1
    )
    
    assert manager.best_metric == 0.5
    assert (tmp_path / 'best_model.pt').exists()
    
    # Save with worse metrics
    metrics2 = {'val_loss': 0.6}
    manager.save(
      model=model,
      optimizer=optimizer,
      epoch=1,
      batch_idx=10,
      global_step=10,
      metrics=metrics2
    )
    
    assert manager.best_metric == 0.5  # Should not update
    
    # Save with better metrics
    metrics3 = {'val_loss': 0.4}
    manager.save(
      model=model,
      optimizer=optimizer,
      epoch=2,
      batch_idx=20,
      global_step=20,
      metrics=metrics3
    )
    
    assert manager.best_metric == 0.4
  
  def test_save_final(self, tmp_path):
    """Test saving final checkpoint."""
    manager = CheckpointManager(
      checkpoint_dir=tmp_path,
      track_metric='accuracy',
      metric_mode='max'
    )
    
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Set best metric
    manager.best_metric = 0.95
    
    # Save final
    final_metrics = {'accuracy': 0.93, 'loss': 0.1}
    final_path = manager.save_final(
      model=model,
      optimizer=optimizer,
      epoch=10,
      batch_idx=1000,
      global_step=10000,
      metrics=final_metrics
    )
    
    assert final_path == tmp_path / "final_checkpoint.pt"
    assert final_path.exists()
    
    # Load and verify
    checkpoint = torch.load(final_path)
    assert checkpoint['epoch'] == 10
    assert checkpoint['batch_idx'] == 1000
    assert checkpoint['global_step'] == 10000
    assert checkpoint['best_metrics']['best_accuracy'] == 0.95
    assert checkpoint['final_metrics'] == final_metrics