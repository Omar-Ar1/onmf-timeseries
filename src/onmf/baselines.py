import numpy as np
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple


class BaselineModels:
    """Baseline models for comparison."""
    
    @staticmethod
    def naive_persistence(X_train: np.ndarray, n_predictions: int) -> np.ndarray:
        """
        Naive baseline: predict last observed value.
        """
        predictions = np.repeat(X_train[-1:, :], n_predictions, axis=0)
        return predictions
    
    @staticmethod
    def moving_average(X_train: np.ndarray, n_predictions: int, window: int = 5) -> np.ndarray:
        """
        Moving average baseline.
        """
        predictions = []
        X = X_train.copy()
        
        for _ in range(n_predictions):
            pred = X[-window:].mean(axis=0)
            predictions.append(pred)
            X = np.vstack([X, pred])
        
        return np.array(predictions)
    
    @staticmethod
    def linear_autoregression(
        X_train: np.ndarray, 
        n_predictions: int,
        lag: int = 5
    ) -> np.ndarray:
        """
        Linear autoregression with Ridge regularization.
        """
        # Prepare lagged features
        def create_lagged_features(X, lag):
            n, d = X.shape
            X_lagged = []
            y = []
            for i in range(lag, n):
                X_lagged.append(X[i-lag:i].flatten())
                y.append(X[i])
            return np.array(X_lagged), np.array(y)
        
        X_train_lagged, y_train = create_lagged_features(X_train, lag)
        
        # Train model per feature
        models = []
        for i in range(X_train.shape[1]):
            model = Ridge(alpha=1.0)
            model.fit(X_train_lagged, y_train[:, i])
            models.append(model)
        
        # Make predictions
        predictions = []
        X_context = X_train.copy()
        
        for _ in range(n_predictions):
            X_lag = X_context[-lag:].flatten().reshape(1, -1)
            pred = np.array([model.predict(X_lag)[0] for model in models])
            predictions.append(pred)
            X_context = np.vstack([X_context, pred])
        
        return np.array(predictions)
    @staticmethod
    def arima_forecast(
        X_train: np.ndarray,
        n_predictions: int,
        order: Tuple[int, int, int] = (2, 1, 2),
        logger=None,
    ) -> np.ndarray:
        """
        ARIMA model (single feature only due to computational cost).
        """
        try:
            model = ARIMA(X_train, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_predictions)
            return forecast
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}, using naive instead")
            return np.repeat(X_train[-1, :], n_predictions)