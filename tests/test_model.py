"""Tests for core OnlineNMF functionality."""

import pytest
import torch
import numpy as np

from src.onmf.model import OnlineNMF


class TestOnlineNMF:
    """Test cases for OnlineNMF class."""

    def test_initialization(self):
        """Test model initialization."""
        model = OnlineNMF(r=10, k=5, batch_size=20, N=50, max_iter=50)
        
        assert model.r == 10
        assert model.k == 5
        assert model.batch_size == 20
        assert model.N == 50
        assert model.max_iter == 50
        assert model.W_final is None

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            OnlineNMF(r=-1, k=5, batch_size=20, N=50, max_iter=50)
        
        with pytest.raises(ValueError):
            OnlineNMF(r=10, k=100, batch_size=20, N=50, max_iter=50)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        # Create synthetic data
        torch.manual_seed(42)
        T, d = 200, 6
        X = torch.abs(torch.randn(T, d)) + 1.0
        
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        model.fit(X, lambd=1.0, beta=2.0, lambd_min=2.0, beta_min=1.0)
        
        assert model.W_final is not None
        assert model.W_final.shape == (d, 3, 5)
        assert model.A_final.shape == (5, 5)
        assert model.B_final.shape == (5, d * 3)

    def test_predict_without_fit(self):
        """Test that prediction fails without fitting."""
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        X = torch.abs(torch.randn(50, 6))
        
        with pytest.raises(ValueError):
            model.predict(X, lambd_pred=0.5)

    def test_predict_basic(self):
        """Test basic prediction functionality."""
        torch.manual_seed(42)
        T, d = 150, 4
        X = torch.abs(torch.randn(T, d)) + 1.0
        
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        model.fit(X[:100], lambd=1.0, beta=2.0, lambd_min=2.0, beta_min=1.0)
        
        pred = model.predict(X[:100], lambd_pred=0.5)
        
        assert pred.shape == (d,)
        assert torch.all(pred >= 0)  # NMF ensures non-negativity

    def test_importance_scores(self):
        """Test importance score computation."""
        torch.manual_seed(42)
        X = torch.abs(torch.randn(150, 4)) + 1.0
        
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        model.fit(X, lambd=1.0, beta=2.0, lambd_min=2.0, beta_min=1.0)
        
        importance = model.get_importance_scores()
        
        assert len(importance) == 5
        assert np.isclose(importance.sum(), 1.0)
        assert np.all(importance >= 0)

    def test_sliding_window(self):
        """Test sliding window construction."""
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        
        X = torch.arange(30).reshape(10, 3).float()
        windows = model._sliding_window(X, k=3)

        assert windows.shape == (8, 3, 3)  # (N-k+1, k, d)

    def test_update_H(self):
        """Test sparse code update."""
        model = OnlineNMF(r=3, k=2, batch_size=10, N=20, max_iter=50)
        
        torch.manual_seed(42)
        X = torch.abs(torch.randn(6, 5))
        W = torch.abs(torch.randn(6, 3))
        H_init = torch.abs(torch.randn(3, 5))
        
        H = model._update_H(X, W, H_init, lambd=1.0)
        
        assert H.shape == (3, 5)
        assert torch.all(H >= 0)

    def test_loss_history(self):
        """Test that loss history is tracked."""
        torch.manual_seed(42)
        X = torch.abs(torch.randn(150, 4)) + 1.0
        
        model = OnlineNMF(r=5, k=3, batch_size=10, N=20, max_iter=20)
        model.fit(X, lambd=1.0, beta=2.0, lambd_min=2.0, beta_min=1.0)
        
        assert len(model.loss_history) > 0
        assert all(loss >= 0 for loss in model.loss_history)

    def test_device_handling(self):
        """Test CPU/GPU device handling."""
        model = OnlineNMF(
            r=5, k=3, batch_size=10, N=20, max_iter=20,
            device=torch.device('cpu')
        )
        
        assert model.device == torch.device('cpu')
        
        X = torch.abs(torch.randn(100, 4))
        model.fit(X, lambd=1.0, beta=2.0, lambd_min=2.0, beta_min=1.0)
        
        assert model.W_final.device.type == 'cpu'

if __name__ == "__main__":
    pytest.main([__file__])
  