"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.training.metrics import (
  mrr_at_k, dcg_at_k, recall_at_k,
  inference_batch, inference_all, evaluate
)


class TestRankingMetrics:
  """Test ranking metrics (MRR, DCG, Recall)."""
  
  def test_recall_at_k_perfect(self):
    """Test recall@k with perfect predictions."""
    # 3 samples, 5 classes
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Correct: class 0
      [0.05, 0.9, 0.02, 0.02, 0.01],  # Correct: class 1
      [0.02, 0.02, 0.9, 0.05, 0.01],  # Correct: class 2
    ])
    
    # All predictions should be correct for any k >= 1
    assert recall_at_k(y_true, y_pred_probs, k=1) == 1.0
    assert recall_at_k(y_true, y_pred_probs, k=2) == 1.0
    assert recall_at_k(y_true, y_pred_probs, k=5) == 1.0
  
  def test_recall_at_k_partial(self):
    """Test recall@k with partial correct predictions."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Correct: class 0 at rank 1
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Wrong: predicts 0 but true is 1
      [0.02, 0.9, 0.05, 0.02, 0.01],  # Wrong: predicts 1 but true is 2
    ])
    
    # Only first sample is correct at k=1
    assert recall_at_k(y_true, y_pred_probs, k=1) == pytest.approx(1/3)
    
    # At k=2, all three are found (0 at rank 1, 1 at rank 2, 2 at rank 3)
    assert recall_at_k(y_true, y_pred_probs, k=2) == pytest.approx(1.0)
  
  def test_recall_at_k_zero(self):
    """Test recall@k when all predictions are wrong."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.01, 0.02, 0.02, 0.05, 0.9],  # Predicts 4, true is 0
      [0.01, 0.02, 0.02, 0.05, 0.9],  # Predicts 4, true is 1
      [0.01, 0.02, 0.02, 0.05, 0.9],  # Predicts 4, true is 2
    ])
    
    # True classes are not in top-2
    assert recall_at_k(y_true, y_pred_probs, k=1) == 0.0
    assert recall_at_k(y_true, y_pred_probs, k=2) == 0.0
  
  def test_mrr_at_k_perfect(self):
    """Test MRR@k with perfect predictions."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Rank 1
      [0.05, 0.9, 0.02, 0.02, 0.01],  # Rank 1
      [0.02, 0.02, 0.9, 0.05, 0.01],  # Rank 1
    ])
    
    # MRR should be 1.0 (all at rank 1)
    assert mrr_at_k(y_true, y_pred_probs, k=5) == 1.0
  
  def test_mrr_at_k_mixed_ranks(self):
    """Test MRR@k with mixed ranks."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 0 at rank 1: 1/1
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 1 at rank 2: 1/2
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 2 at rank 3: 1/3
    ])
    
    # MRR = (1/1 + 1/2 + 1/3) / 3
    expected = (1.0 + 0.5 + 1/3) / 3
    # Actual calculation for this specific array
    result = mrr_at_k(y_true, y_pred_probs, k=5)
    # Class 0 is at position 0 (rank 1): 1/1 = 1.0
    # Class 1 is at position 1 (rank 2): 1/2 = 0.5
    # Class 2 is at position 2 (rank 3): 1/3 = 0.333...
    # Average: (1.0 + 0.5 + 0.333...) / 3 ≈ 0.611
    assert result == pytest.approx(0.611, abs=0.01)
  
  def test_mrr_at_k_cutoff(self):
    """Test MRR@k respects k cutoff."""
    y_true = np.array([0, 1, 4])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Class 0 at rank 1
      [0.9, 0.05, 0.02, 0.02, 0.01],  # Class 1 at rank 2
      [0.1, 0.2, 0.3, 0.35, 0.05],    # Class 4 at rank 5
    ])
    
    # With k=2, only first two samples contribute
    assert mrr_at_k(y_true, y_pred_probs, k=2) == pytest.approx((1.0 + 0.5 + 0) / 3)
    
    # With k=5, all samples contribute
    assert mrr_at_k(y_true, y_pred_probs, k=5) == pytest.approx((1.0 + 0.5 + 0.2) / 3)
  
  def test_dcg_at_k_perfect(self):
    """Test DCG@k with perfect predictions."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.02, 0.02, 0.01],
      [0.05, 0.9, 0.02, 0.02, 0.01],
      [0.02, 0.02, 0.9, 0.05, 0.01],
    ])
    
    # DCG for all at rank 1: 1/log2(2) = 1.0
    assert dcg_at_k(y_true, y_pred_probs, k=5) == 1.0
  
  def test_dcg_at_k_mixed_ranks(self):
    """Test DCG@k with mixed ranks."""
    y_true = np.array([0, 1, 2])
    y_pred_probs = np.array([
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 0 at rank 1: 1/log2(2)
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 1 at rank 2: 1/log2(3)
      [0.9, 0.05, 0.03, 0.015, 0.005],  # Class 2 at rank 3: 1/log2(4)
    ])
    
    result = dcg_at_k(y_true, y_pred_probs, k=5)
    # Class 0 is at position 0 (rank 1): 1/log2(2) = 1.0
    # Class 1 is at position 1 (rank 2): 1/log2(3) ≈ 0.631
    # Class 2 is at position 2 (rank 3): 1/log2(4) = 0.5
    # Average: (1.0 + 0.631 + 0.5) / 3 ≈ 0.710
    assert result == pytest.approx(0.710, abs=0.01)


class TestInference:
  """Test inference functions."""
  
  def test_inference_batch(self):
    """Test single batch inference."""
    # Create a simple mock model
    model = MagicMock()
    model.eval = MagicMock()
    
    # Mock model output (logits)
    batch_size = 4
    num_classes = 3
    logits = torch.randn(batch_size, num_classes)
    model.return_value = logits
    
    # Input data
    X = np.random.randn(batch_size, 10).astype(np.float32)
    device = torch.device("cpu")
    
    # Run inference
    with torch.no_grad():
      probs = inference_batch(model, X, device)
    
    # Check output
    assert probs.shape == (batch_size, num_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
  
  def test_inference_all(self):
    """Test inference on multiple batches."""
    # Create mock model
    model = MagicMock()
    model.eval = MagicMock()
    
    # Setup data
    n_samples = 10
    n_features = 5
    num_classes = 3
    batch_size = 4
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    device = torch.device("cpu")
    
    # Mock model to return different logits for each batch
    def mock_forward(x):
      return torch.randn(len(x), num_classes)
    
    model.side_effect = mock_forward
    
    # Run inference
    y_pred, all_preds = inference_all(model, X, batch_size, device)
    
    # Check outputs
    assert y_pred.shape == (n_samples,)
    assert all_preds.shape == (n_samples, num_classes)
    assert np.all(y_pred >= 0) and np.all(y_pred < num_classes)
    assert np.allclose(all_preds.sum(axis=1), 1.0)


class TestEvaluate:
  """Test the main evaluate function."""
  
  def test_evaluate_basic(self):
    """Test basic evaluation functionality."""
    # Create a simple linear model for testing
    class SimpleModel(nn.Module):
      def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
      
      def forward(self, x):
        return self.linear(x)
    
    # Setup
    n_samples = 100
    input_dim = 10
    num_classes = 5
    
    model = SimpleModel(input_dim, num_classes)
    model.eval()
    
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, n_samples)
    
    # Evaluate
    metrics = evaluate(
      model=model,
      X=X,
      y=y,
      num_classes=num_classes,
      batch_size=32,
      device=torch.device("cpu"),
      k_values=(2, 5)
    )
    
    # Check that all expected metrics are present
    expected_metrics = [
      'logloss', 'macro_f1', 'macro_precision', 'macro_recall',
      'recall_at_2', 'mrr_at_2', 'dcg_at_2',
      'recall_at_5', 'mrr_at_5', 'dcg_at_5'
    ]
    
    for metric in expected_metrics:
      assert metric in metrics
      assert isinstance(metrics[metric], (float, np.floating))
      assert not np.isnan(metrics[metric])
    
    # Check metric ranges
    assert metrics['logloss'] > 0  # Log loss should be positive
    assert 0 <= metrics['macro_f1'] <= 1
    assert 0 <= metrics['macro_precision'] <= 1
    assert 0 <= metrics['macro_recall'] <= 1
    assert 0 <= metrics['recall_at_2'] <= 1
    assert 0 <= metrics['recall_at_5'] <= 1
  
  def test_evaluate_perfect_predictions(self):
    """Test evaluation with perfect predictions."""
    # Create a model that always predicts correctly
    class PerfectModel(nn.Module):
      def __init__(self, y_true, X_true):
        super().__init__()
        self.y_true = torch.tensor(y_true)
        self.X_true = torch.tensor(X_true, dtype=torch.float32)
      
      def forward(self, x):
        batch_size = x.shape[0]
        num_classes = self.y_true.max().item() + 1
        
        # Create one-hot encoding with some noise
        logits = torch.randn(batch_size, num_classes) * 0.1
        
        # Match input samples to find their true labels
        for i in range(batch_size):
          # Find which sample this is by comparing with X_true
          # Use sum of features as a simple identifier
          x_sum = x[i].sum()
          diffs = torch.abs(self.X_true.sum(dim=1) - x_sum)
          true_idx = torch.argmin(diffs).item()
          
          # Make true class have highest logit
          logits[i, self.y_true[true_idx]] = 10.0
        return logits
    
    # Setup
    n_samples = 50
    num_classes = 10
    X = np.random.randn(n_samples, 20).astype(np.float32)
    y = np.random.randint(0, num_classes, n_samples)
    
    model = PerfectModel(y, X)
    model.eval()
    
    # Evaluate
    metrics = evaluate(
      model=model,
      X=X,
      y=y,
      num_classes=num_classes,
      batch_size=16
    )
    
    # With perfect predictions, recall metrics should be 1.0
    assert metrics['recall_at_2'] == pytest.approx(1.0, abs=0.01)
    assert metrics['recall_at_5'] == pytest.approx(1.0, abs=0.01)
    assert metrics['recall_at_10'] == pytest.approx(1.0, abs=0.01)
    
    # MRR should also be 1.0 (all predictions at rank 1)
    assert metrics['mrr_at_2'] == pytest.approx(1.0, abs=0.01)
    assert metrics['mrr_at_5'] == pytest.approx(1.0, abs=0.01)
  
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
  def test_evaluate_cuda(self):
    """Test evaluation on CUDA device."""
    # Simple model
    model = nn.Linear(10, 5)
    model = model.cuda()
    model.eval()
    
    # Data
    X = np.random.randn(20, 10).astype(np.float32)
    y = np.random.randint(0, 5, 20)
    
    # Evaluate on CUDA
    metrics = evaluate(
      model=model,
      X=X,
      y=y,
      num_classes=5,
      batch_size=8,
      device=torch.device("cuda")
    )
    
    # Check metrics are computed
    assert 'logloss' in metrics
    assert 'macro_f1' in metrics
    assert not np.isnan(metrics['logloss'])