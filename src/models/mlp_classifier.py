"""MLP classifier architecture for CellXGene cell type prediction."""

import torch
import torch.nn as nn


def create_mlp(input_dim: int, num_classes: int, n_hidden_layers: int, dropout: float) -> nn.Module:
  """Create an MLP classifier with interpolated hidden dimensions.
  
  This matches the architecture from the notebook that achieved the best performance.
  
  Args:
    input_dim: Input dimension (embedding size)
    num_classes: Number of output classes
    n_hidden_layers: Number of hidden layers
    dropout: Dropout probability
    
  Returns:
    MLP model as nn.Sequential
  """
  layers = []
  
  # Calculate hidden layer dimensions using linear interpolation
  # This creates a smooth transition from input_dim to num_classes
  dims = [
    int(input_dim + (num_classes - input_dim) * (i + 1) / (n_hidden_layers + 1)) 
    for i in range(n_hidden_layers)
  ]
  
  prev_dim = input_dim
  for h_dim in dims:
    # Linear layer
    layers.append(nn.Linear(prev_dim, h_dim))
    # Batch normalization
    layers.append(nn.BatchNorm1d(h_dim))
    # Activation
    layers.append(nn.ReLU())
    # Dropout
    layers.append(nn.Dropout(dropout))
    prev_dim = h_dim
  
  # Output layer (no dropout or activation after this)
  layers.append(nn.Linear(prev_dim, num_classes))
  
  return nn.Sequential(*layers)


class MLPClassifier(nn.Module):
  """MLP classifier wrapper with additional functionality.
  
  This class wraps the MLP model and provides additional methods
  that might be useful for training and evaluation.
  """
  
  def __init__(self, input_dim: int, num_classes: int, n_hidden_layers: int = 3, dropout: float = 0.05):
    """Initialize the MLP classifier.
    
    Args:
      input_dim: Input dimension (embedding size)
      num_classes: Number of output classes
      n_hidden_layers: Number of hidden layers (default: 3, from best notebook run)
      dropout: Dropout probability (default: 0.053, from best notebook run)
    """
    super().__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.n_hidden_layers = n_hidden_layers
    self.dropout = dropout
    
    # Create the MLP
    self.model = create_mlp(input_dim, num_classes, n_hidden_layers, dropout)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model.
    
    Args:
      x: Input tensor of shape (batch_size, input_dim)
      
    Returns:
      Logits of shape (batch_size, num_classes)
    """
    return self.model(x)
  
  def get_hidden_dims(self) -> list:
    """Get the dimensions of hidden layers.
    
    Returns:
      List of hidden layer dimensions
    """
    dims = []
    for module in self.model:
      if isinstance(module, nn.Linear):
        dims.append(module.out_features)
    return dims[:-1]  # Exclude output layer
  
  def count_parameters(self) -> int:
    """Count the number of trainable parameters.
    
    Returns:
      Total number of trainable parameters
    """
    return sum(p.numel() for p in self.parameters() if p.requires_grad)