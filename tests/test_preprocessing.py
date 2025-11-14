"""Tests for preprocessing utilities."""

import pytest
import pandas as pd
import torch
from datetime import datetime

from src.onmf.preprocessing import (
    FinancialDataLoader,
    FinancialPreprocessor,
)


class TestFinancialPreprocessor:
    """Test cases for FinancialPreprocessor."""

    def test_compute_returns_log(self):
        """Test log returns calculation."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'price': [100, 110, 105, 115],
        })
        
        returns = preprocessor.compute_returns(data, method='log')
        
        assert len(returns) == 4
        assert pd.isna(returns.iloc[0]['price'])
        assert returns.iloc[1]['price'] > 0  # Price increased

    def test_compute_returns_simple(self):
        """Test simple returns calculation."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'price': [100, 110, 105],
        })
        
        returns = preprocessor.compute_returns(data, method='simple')
        
        assert abs(returns.iloc[1]['price'] - 0.1) < 1e-6

    def test_handle_missing_values(self):
        """Test missing value handling."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'col1': [1, None, 3, None, 5],
            'col2': [10, 20, None, 40, 50],
        })
        
        filled = preprocessor.handle_missing_values(data, method='forward_fill')
        
        assert not filled.isna().any().any()

    def test_scale_data_standardize(self):
        """Test standardization scaling."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
        })
        
        scaled, params = preprocessor.scale_data(data, method='standardize')
        
        assert abs(scaled['col1'].mean()) < 1e-6
        assert abs(scaled['col1'].std() - 1.0) < 1e-6
        assert 'mean' in params
        assert 'std' in params

    def test_scale_data_minmax(self):
        """Test min-max scaling."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
        })
        
        scaled, params = preprocessor.scale_data(data, method='minmax')
        
        assert scaled['col1'].min() == 0.0
        assert abs(scaled['col1'].max() - 1.0) < 1e-6

    def test_make_non_negative_shift(self):
        """Test non-negative transformation with shift."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'col1': [-5, -2, 0, 3, 5],
        })
        
        nn_data, params = preprocessor.make_non_negative(data, method='shift')
        
        assert (nn_data >= 0).all().all()
        assert nn_data['col1'].min() == 0.0

    def test_to_tensor(self):
        """Test DataFrame to tensor conversion."""
        preprocessor = FinancialPreprocessor()
        
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
        })
        
        tensor = preprocessor.to_tensor(data, device=torch.device('cpu'))
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)
        assert tensor.device.type == 'cpu'

if __name__ == "__main__":
    pytest.main([__file__])
  