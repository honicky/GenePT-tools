"""Unit tests for MLP classifier model."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.mlp_classifier import create_mlp, MLPClassifier


class TestCreateMLP:
  """Test the create_mlp function."""
  
  def test_create_mlp_basic(self):
    """Test basic MLP creation."""
    model = create_mlp(
      input_dim=100,
      num_classes=10,
      n_hidden_layers=2,
      dropout=0.1
    )
    
    # Check it's a Sequential model
    assert isinstance(model, nn.Sequential)
    
    # Count layer types
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 3  # 2 hidden + 1 output
    
    # Check first layer dimensions
    assert linear_layers[0].in_features == 100
    
    # Check output layer dimensions
    assert linear_layers[-1].out_features == 10
  
  def test_hidden_dimension_interpolation(self):
    """Test that hidden dimensions are interpolated correctly."""
    model = create_mlp(
      input_dim=500,
      num_classes=100,
      n_hidden_layers=3,
      dropout=0.0
    )
    
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    
    # Check interpolated dimensions
    # Should go from 500 -> ~400 -> ~300 -> ~200 -> 100
    assert linear_layers[0].in_features == 500
    assert linear_layers[0].out_features == 400  # 500 + (100-500) * 1/4
    assert linear_layers[1].out_features == 300  # 500 + (100-500) * 2/4
    assert linear_layers[2].out_features == 200  # 500 + (100-500) * 3/4
    assert linear_layers[3].out_features == 100
  
  def test_layer_ordering(self):
    """Test that layers are in correct order."""
    model = create_mlp(
      input_dim=100,
      num_classes=10,
      n_hidden_layers=1,
      dropout=0.1
    )
    
    # Expected order: Linear, BatchNorm, ReLU, Dropout, Linear
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.BatchNorm1d)
    assert isinstance(model[2], nn.ReLU)
    assert isinstance(model[3], nn.Dropout)
    assert isinstance(model[4], nn.Linear)
  
  def test_single_hidden_layer(self):
    """Test MLP with single hidden layer."""
    model = create_mlp(
      input_dim=50,
      num_classes=5,
      n_hidden_layers=1,
      dropout=0.2
    )
    
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2  # 1 hidden + 1 output
    
    # Check dimensions
    assert linear_layers[0].in_features == 50
    assert linear_layers[0].out_features == 27  # 50 + (5-50) * 1/2
    assert linear_layers[1].out_features == 5
  
  def test_dropout_probability(self):
    """Test that dropout layers have correct probability."""
    dropout_prob = 0.35
    model = create_mlp(
      input_dim=100,
      num_classes=10,
      n_hidden_layers=2,
      dropout=dropout_prob
    )
    
    dropout_layers = [m for m in model if isinstance(m, nn.Dropout)]
    assert len(dropout_layers) == 2  # One per hidden layer
    
    for layer in dropout_layers:
      assert layer.p == dropout_prob


class TestMLPClassifier:
  """Test the MLPClassifier wrapper class."""
  
  def test_initialization(self):
    """Test MLPClassifier initialization."""
    classifier = MLPClassifier(
      input_dim=500,
      num_classes=377,
      n_hidden_layers=3,
      dropout=0.053
    )
    
    assert classifier.input_dim == 500
    assert classifier.num_classes == 377
    assert classifier.n_hidden_layers == 3
    assert classifier.dropout == 0.053
    assert isinstance(classifier.model, nn.Sequential)
  
  def test_forward_pass(self):
    """Test forward pass through the model."""
    batch_size = 32
    input_dim = 100
    num_classes = 10
    
    classifier = MLPClassifier(
      input_dim=input_dim,
      num_classes=num_classes,
      n_hidden_layers=2,
      dropout=0.1
    )
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = classifier(x)
    
    # Check output shape
    assert output.shape == (batch_size, num_classes)
    
    # Check that output is not normalized (logits, not probabilities)
    assert not torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-3)
  
  def test_get_hidden_dims(self):
    """Test getting hidden layer dimensions."""
    classifier = MLPClassifier(
      input_dim=500,
      num_classes=100,
      n_hidden_layers=3,
      dropout=0.0
    )
    
    hidden_dims = classifier.get_hidden_dims()
    
    assert len(hidden_dims) == 3
    assert hidden_dims == [400, 300, 200]
  
  def test_count_parameters(self):
    """Test parameter counting."""
    classifier = MLPClassifier(
      input_dim=100,
      num_classes=10,
      n_hidden_layers=1,
      dropout=0.0
    )
    
    param_count = classifier.count_parameters()
    
    # Calculate expected parameters
    # Layer 1: 100 * 55 + 55 (weights + bias) = 5555
    # BatchNorm: 55 * 2 (gamma + beta) = 110
    # Layer 2: 55 * 10 + 10 (weights + bias) = 560
    # Total: 5555 + 110 + 560 = 6225
    expected = 6225
    
    assert param_count == expected
  
  def test_notebook_configuration(self):
    """Test with the exact configuration from the notebook."""
    classifier = MLPClassifier(
      input_dim=500,
      num_classes=377,
      n_hidden_layers=3,
      dropout=0.053
    )
    
    hidden_dims = classifier.get_hidden_dims()
    
    # Check dimensions match expected interpolation
    assert hidden_dims[0] == 469  # 500 + (377-500) * 1/4
    assert hidden_dims[1] == 438  # 500 + (377-500) * 2/4  
    assert hidden_dims[2] == 407  # 500 + (377-500) * 3/4
    
    # Test forward pass with notebook config
    batch_size = 1024
    x = torch.randn(batch_size, 500)
    output = classifier(x)
    
    assert output.shape == (batch_size, 377)
  
  def test_model_training_mode(self):
    """Test that model can be set to train/eval mode."""
    classifier = MLPClassifier(
      input_dim=100,
      num_classes=10,
      n_hidden_layers=1,
      dropout=0.5
    )
    
    # Set to training mode
    classifier.train()
    assert classifier.training
    
    # Forward pass should apply dropout
    x = torch.randn(10, 100)
    output1 = classifier(x)
    output2 = classifier(x)
    
    # Outputs should be different due to dropout
    assert not torch.allclose(output1, output2)
    
    # Set to eval mode
    classifier.eval()
    assert not classifier.training
    
    # Forward passes should be identical
    output3 = classifier(x)
    output4 = classifier(x)
    assert torch.allclose(output3, output4)
  
  @pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
      not torch.cuda.is_available(), reason="CUDA not available"
    ))
  ])
  def test_device_placement(self, device):
    """Test model can be placed on different devices."""
    classifier = MLPClassifier(
      input_dim=50,
      num_classes=10,
      n_hidden_layers=1,
      dropout=0.1
    )
    
    classifier = classifier.to(device)
    
    # Check all parameters are on correct device
    for param in classifier.parameters():
      assert param.device.type == device
    
    # Test forward pass on device
    x = torch.randn(5, 50).to(device)
    output = classifier(x)
    assert output.device.type == device