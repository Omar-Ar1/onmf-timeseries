"""
Data preprocessing utilities for financial time-series.

This module provides functions for loading, cleaning, and preprocessing
financial market data for use with Online NMF models.
"""

from typing import List, Optional, Tuple, Dict, Union
from datetime import datetime, timedelta
import logging

import torch
import numpy as np
import pandas as pd
import yfinance as yf
import re

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """
    Load and preprocess financial market data from Yahoo Finance.

    Example:
        >>> loader = FinancialDataLoader()
        >>> data, metadata = loader.load_multiple_assets(
        ...     ['AAPL', 'GOOGL', 'MSFT'],
        ...     start='2020-01-01',
        ...     features=['Close', 'Volume']
        ... )
    """

    VALID_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            cache_dir: Optional directory for caching downloaded data.
        """
        self.cache_dir = cache_dir

    def load_multiple_assets(
        self,
        tickers: List[str],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        features: Optional[List[str]] = None,
        interval: str = '1d',
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load data for multiple assets.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL']).
            start: Start date ('YYYY-MM-DD' or datetime object).
            end: End date (defaults to today).
            features: List of features to extract. Defaults to ['Close', 'Volume'].
            interval: Data interval ('1d', '1wk', '1mo', etc.).

        Returns:
            Tuple of (data_df, metadata):
                - data_df: DataFrame with multi-index columns (ticker, feature)
                - metadata: Dictionary with data information

        Raises:
            ValueError: If invalid tickers or features provided.
        """
        if not tickers:
            raise ValueError("Must provide at least one ticker")

        if features is None:
            features = ['Close', 'Volume']

        # Validate features
        invalid_features = set(features) - set(self.VALID_FEATURES)
        if invalid_features:
            raise ValueError(
                f"Invalid features: {invalid_features}. "
                f"Valid features: {self.VALID_FEATURES}"
            )

        # Set end date to today if not provided
        if end is None:
            end = datetime.now()

        logger.info(
            f"Downloading data for {len(tickers)} assets "
            f"from {start} to {end}"
        )

        # Download data
        try:
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download data: {e}")

        if data.empty:
            raise ValueError("No data downloaded. Check ticker symbols and dates.")

        # Handle single vs multiple tickers
        if len(tickers) == 1:
            # For single ticker, yfinance returns simple columns
            data.columns = pd.MultiIndex.from_product(
                [[tickers[0]], data.columns]
            )

        # Select requested features
        try:
            selected_data = data.loc[:, (features, slice(None))]
        except KeyError as e:
            raise ValueError(f"Feature selection failed: {e}")

        # Sort columns for consistency
        selected_data = selected_data.sort_index(axis=1)

        # Create metadata
        metadata = {
            'tickers': tickers,
            'features': features,
            'start_date': selected_data.index.min(),
            'end_date': selected_data.index.max(),
            'n_samples': len(selected_data),
            'interval': interval,
        }

        logger.info(
            f"Loaded {metadata['n_samples']} samples "
            f"for {len(tickers)} assets"
        )

        return selected_data, metadata

    def load_market_indices(
        self,
        indices: Optional[List[str]] = None,
        start: Union[str, datetime] = '2020-01-01',
        end: Optional[Union[str, datetime]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load major market indices.

        Args:
            indices: List of index symbols. Defaults to major US indices.
            start: Start date.
            end: End date.

        Returns:
            Tuple of (data_df, metadata).
        """
        if indices is None:
            indices = [
                '^GSPC',   # S&P 500
                '^DJI',    # Dow Jones
                '^IXIC',   # NASDAQ
                '^RUT',    # Russell 2000
                '^VIX',    # Volatility Index
            ]

        return self.load_multiple_assets(
            tickers=indices,
            start=start,
            end=end,
            features=['Close', 'Volume'],
        )


class FinancialPreprocessor:
    """
    Preprocess financial data for Online NMF.

    Handles missing values, scaling, returns calculation, and tensor conversion.
    """

    @staticmethod
    def compute_returns(
        data: pd.DataFrame,
        method: str = 'log',
        periods: int = 1,
    ) -> pd.DataFrame:
        """
        Compute returns from price data.

        Args:
            data: DataFrame with price data.
            method: 'log' for log returns or 'simple' for arithmetic returns.
            periods: Number of periods for return calculation.

        Returns:
            DataFrame with returns.
        """
        if method == 'log':
            returns = np.log(data / data.shift(periods))
        elif method == 'simple':
            returns = data.pct_change(periods=periods)
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        method: str = 'forward_fill',
        max_consecutive: int = 5,
    ) -> pd.DataFrame:
        """
        Handle missing values in data.

        Args:
            data: DataFrame with potential missing values.
            method: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'.
            max_consecutive: Maximum consecutive NaNs to fill.

        Returns:
            DataFrame with missing values handled.
        """
        df = data.copy()

        if method == 'forward_fill':
            df = df.ffill(limit=max_consecutive)
        elif method == 'backward_fill':
            df = df.fillna(method='bfill', limit=max_consecutive)
        elif method == 'interpolate':
            df = df.interpolate(method='linear', limit=max_consecutive)
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Drop any remaining NaNs
        df = df.dropna()

        return df

    @staticmethod
    def scale_data(
        data: pd.DataFrame,
        method: str = 'standardize',
        clip_std: float = 5.0,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale data for model training.

        Args:
            data: DataFrame to scale.
            method: 'standardize' (z-score) or 'minmax'.
            clip_std: Clip outliers beyond this many standard deviations.

        Returns:
            Tuple of (scaled_data, scaler_params) for inverse transform.
        """
        df = data.copy()

        if method == 'standardize':
            mean = df.mean()
            std = df.std()

            # Clip outliers
            df = df.clip(
                lower=mean - clip_std * std,
                upper=mean + clip_std * std,
                axis=1,
            )

            # Standardize
            scaled = (df - mean) / (std + 1e-9)

            scaler_params = {
                'method': 'standardize',
                'mean': mean,
                'std': std,
            }

        elif method == 'minmax':
            min_val = df.min()
            max_val = df.max()

            scaled = (df - min_val) / (max_val - min_val + 1e-9)

            scaler_params = {
                'method': 'minmax',
                'min': min_val,
                'max': max_val,
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        return scaled, scaler_params

    @staticmethod
    def inverse_scale(
        data: Union[pd.DataFrame, torch.Tensor, np.ndarray],
        scaler_params: Dict,
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Inverse transform scaled data.

        Args:
            data: Scaled data.
            scaler_params: Parameters from scale_data().

        Returns:
            Original scale data.
        """
        method = scaler_params['method']

        if method == 'standardize':
            mean = scaler_params['mean']
            std = scaler_params['std']

            if isinstance(data, pd.DataFrame):
                return data * std + mean
            elif isinstance(data, torch.Tensor):
                mean_t = torch.tensor(mean.values, device=data.device)
                std_t = torch.tensor(std.values, device=data.device)
                return data * std_t + mean_t
            else:  # numpy
                return data * std.values + mean.values

        elif method == 'minmax':
            min_val = scaler_params['min']
            max_val = scaler_params['max']

            if isinstance(data, pd.DataFrame):
                return data * (max_val - min_val) + min_val
            elif isinstance(data, torch.Tensor):
                min_t = torch.tensor(min_val.values, device=data.device)
                max_t = torch.tensor(max_val.values, device=data.device)
                return data * (max_t - min_t) + min_t
            else:  # numpy
                return data * (max_val.values - min_val.values) + min_val.values

    @staticmethod
    def make_non_negative(
        data: pd.DataFrame,
        method: str = 'shift',
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform data to be non-negative (required for NMF).

        Args:
            data: Input DataFrame.
            method: 'shift' (add minimum), 'log1p', or 'abs'.

        Returns:
            Tuple of (non_negative_data, transform_params).
        """
        df = data.copy()

        if method == 'shift':
            min_val = df.min()
            shift = -min_val.clip(upper=0)  # Shift negative values to 0
            df_nn = df + shift

            transform_params = {
                'method': 'shift',
                'shift': shift,
            }

        elif method == 'log1p':
            # Requires data to be non-negative already
            if (df < 0).any().any():
                raise ValueError("Cannot use log1p on negative values")

            df_nn = np.log1p(df)

            transform_params = {
                'method': 'log1p',
            }

        elif method == 'abs':
            df_nn = df.abs()

            transform_params = {
                'method': 'abs',
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        return df_nn, transform_params

    @staticmethod
    def to_tensor(
        data: pd.DataFrame,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert DataFrame to PyTorch tensor.

        Args:
            data: Input DataFrame.
            device: Target device (CPU/CUDA).

        Returns:
            PyTorch tensor of shape (T, d).
        """
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # Convert to numpy then tensor
        tensor = torch.tensor(
            data.values,
            dtype=torch.float32,
            device=device,
        )

        return tensor


def prepare_financial_data(
    tickers: List[str],
    start: str,
    end: str = None,
    features: List[str] = None,
    train_ratio: float = 0.8,
    compute_returns: bool = True,
    return_type: str = 'log',
    scaling_method: str = 'minmax',
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, Dict]:
    """
    Complete pipeline to prepare financial data for Online NMF.

    Args:
        tickers: List of ticker symbols.
        start: Start date.
        end: End date (optional).
        features: Features to extract.
        compute_returns: Whether to compute returns instead of raw prices.
        return_type: 'log' or 'simple' returns.
        device: PyTorch device.

    Returns:
        Tuple of (tensor, original_df, metadata).
    """
    loader = FinancialDataLoader()
    preprocessor = FinancialPreprocessor()
    
    # Load raw data
    data, metadata = loader.load_multiple_assets(
        tickers=tickers,
        start=start,
        end=end,
        features=features or ['Close', 'Volume'],
    )
    
    logger.info(f"Loaded data shape: {data.shape}")
    original_data = data.copy()
    
    # Compute returns if requested
    if compute_returns:
        price_cols = [col for col in data.columns if  re.search(r'\b(Close|Open|High)\b', col[0])]
        if price_cols:
            returns = preprocessor.compute_returns(
                data[price_cols],
                method=return_type,
            )
            data[price_cols] = returns
    
    # Handle missing values
    data = preprocessor.handle_missing_values(data, method='forward_fill')
    
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    logger.info(f"Train size: {train_size}, Test size: {len(test_data)}")
    
    # FIT SCALER ONLY ON TRAINING DATA 
    train_scaled, scaler_params = preprocessor.scale_data(
        train_data, 
        method=scaling_method
    )
    metadata.update({
        'scaler': scaler_params,
        'train_size': train_size,
        'test_size': len(test_data),
        'feature_names': [f"{col[0]}_{col[1]}" for col in data.columns],
    })
        
    # Apply same transformation to test data
    if scaling_method == 'standardize':
        test_scaled = (test_data - scaler_params['mean']) / (scaler_params['std'] + 1e-9)

        # Make non-negative (fit on train only)
        train_nn, nn_params = preprocessor.make_non_negative(train_scaled, method='shift')
        test_nn = test_scaled + nn_params['shift']
        
        # Clip to ensure non-negativity
        train_scaled = train_nn.clip(lower=0)
        test_scaled = test_nn.clip(lower=0)
        metadata.update({
            'nn_transform': nn_params,
        })

    elif scaling_method == 'minmax':
        test_scaled = (test_data - scaler_params['min']) / (scaler_params['max'] - scaler_params['min'] + 1e-9)

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = preprocessor.to_tensor(train_scaled, device=device)
    X_test = preprocessor.to_tensor(test_scaled, device=device)
    

    return X_train, X_test, data, metadata
