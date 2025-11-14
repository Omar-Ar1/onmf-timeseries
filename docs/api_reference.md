# API Reference

This document summarizes the public interfaces exposed by the `onmf` package. All paths below are relative to `src/onmf/`.

---

## Package Initialization (`__init__.py`)

```python
from onmf import (
    OnlineNMF,
    FinancialDataLoader,
    FinancialPreprocessor,
    prepare_financial_data,
    ONMFVisualizer,
    plot_financial_overview,
)
```

Importing the package exposes the core model, preprocessing helpers, and visualization utilities. The canonical version is `0.1.0`.

---

## Model (`model.py`)

### `class OnlineNMF`

Implements Online Non-Negative Matrix Factorization with minibatch initialization and online updates.

**Constructor**

```python
OnlineNMF(
    r: int,
    k: int,
    batch_size: int,
    N: int,
    max_iter: int,
    *,
    tol: float = 1e-4,
    device: Optional[torch.device] = None,
)
```

- `r`: number of dictionary atoms (rank).
- `k`: sliding window length (temporal context).
- `batch_size`: minibatch size for initialization stage.
- `N`: history window size used by the online algorithm.
- `max_iter`: maximum iterations for internal subproblems.
- `tol`: tolerance for early stopping between iterations.
- `device`: optional torch device; defaults to CUDA when available.

**Public Methods**

| Method | Description |
| ------ | ----------- |
| `fit(X, lambda_, beta, lambda_minB, beta_minB)` | Learns dictionary atoms and sparse codes from tensor `X` shaped `(T, d)`. Runs minibatch learning followed by online updates with decay parameters `beta`/`beta_minB` and L1 regularization `lambda_`/`lambda_minB`. |
| `predict(X_context, lambda_pred)` | Returns a `(d,)` forecast using the latest `N` observations inside `X_context`. Performs sparse coding with `lambda_pred` then reconstructs the final timestep. |
| `get_importance_scores()` | Returns a normalized NumPy vector representing the relative activation of each dictionary atom measured across the training horizon. |
| `reset()` | Clears learned state (dictionary, aggregates, codes) without re-instantiating the object. |

**Notable Attributes After `fit`**

- `W_final`: learned dictionary of shape `(d, k, r)`.
- `A_final`, `B_final`: aggregate matrices used for online updates.
- `code`: accumulated sparse codes.
- `device`: computation device (`cpu` or `cuda`).

---

## Preprocessing (`preprocessing.py`)

### `class FinancialDataLoader`

Downloads and organizes market data from Yahoo Finance.

- `load_multiple_assets(tickers, start, end=None, features=None, interval='1d')` → `(DataFrame, metadata)`  
  Downloads specified tickers, keeps selected OHLCV features, and returns a DataFrame with multi-index columns `(feature, ticker)` plus metadata (download interval, observation counts, etc.).

### `class FinancialPreprocessor`

Provides transformations to prepare data for ONMF.

| Method | Purpose |
| ------ | ------- |
| `compute_returns(data, method='log')` | Converts price series to log or simple returns, preserving the input columns. |
| `handle_missing_values(data, method='forward_fill')` | Fills gaps via forward/backward fill or interpolation. |
| `scale_data(data, method='standardize'/'minmax')` | Fits scaling parameters on the provided frame and returns `(scaled_df, params)` where `params` contain means/std or min/max. |
| `make_non_negative(data, method='shift')` | Shifts/clips scaled data so all entries are ≥ 0. Returns `(non_negative_df, params)` with the applied shift. |
| `to_tensor(data, device=None)` | Converts the DataFrame to a `torch.float32` tensor. Device defaults to CUDA when available. |

### `prepare_financial_data(...)`

Convenience function wiring loader and preprocessor:

```python
prepare_financial_data(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    features: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    compute_returns: bool = True,
    return_type: str = 'log',
    scaling_method: str = 'minmax',
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, Dict]
```

Returns `(X_train, X_test, original_df, metadata)` ready for the model.

---

## Baselines (`baselines.py`)

`class BaselineModels` groups several deterministic baseline forecasters operating on NumPy arrays:

- `naive_persistence(X_train, n_predictions)` → repeats the last observed value.
- `moving_average(X_train, n_predictions, window=5)` → iteratively averages the trailing window and appends predictions.
- `linear_autoregression(X_train, n_predictions, lag=5)` → fits Ridge regression using flattened lag windows independently per feature.
- `arima_forecast(X_train, n_predictions, order=(2,1,2), logger=None)` → runs `statsmodels.ARIMA` on a single series; falls back to naive predictions if the solver fails.

Each baseline returns an array shaped `(n_predictions, d)` ready for comparison with ONMF.

---

## Visualization (`visualization.py`)

### `class ONMFVisualizer`

Helper for inspecting learned atoms and predictions.

- `__init__(model, feature_names=None)` ensures the supplied names match `model.d`.
- `plot_dictionary_atoms(top_n, save_path=None, title=..., feature_tick_step=2, timestep_tick_step=1)` displays a grid of heatmaps for the most important atoms.
- `plot_predictions(y_true, predictions_dict, asset_index=0, asset_name=None, feature_names=None, save_path=None)` overlays model predictions against truth.
- `plot_atom_importance(top_n=20, figsize=(10,6), save_path=None)` draws a bar chart of normalized importance scores.

Each method returns a Matplotlib `Figure`. Many also save PNGs when `save_path` is provided.

### `plot_financial_overview(data, title="Financial Data Overview", figsize=(14,8), save_path=None)`

Generates a two-panel plot showing closing prices and volumes for all tickers present in the MultiIndex DataFrame.

---

## Metrics & Utilities (`utils.py`)

- `compute_metrics(y_true, y_pred)` → `dict` with keys `MAE`, `RMSE`, `MAPE`, `Direction_Acc`.
- `compute_metrics_with_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, block_length=None, random_state=None)` → `(metrics_dict, ci_dict)` where each CI is a `(lower, upper)` tuple derived from block-bootstrap resampling.

Internal helpers (`_per_timestep_mean`, `_direction_accuracy_series`, `_block_bootstrap_means`, `_ci_from_samples`, `_resolve_block_length`, `_sample_block_indices`) manage temporal averaging and bootstrap sampling logic.

---

## Tests (`tests/`)

The pytest suite exercises:

- `tests/test_model.py`: initialization, parameter validation, fitting, prediction, sliding window behavior, sparse-code updates, loss history, device handling.
- `tests/test_preprocessing.py`: returns computation, missing values, scaling, non-negativity transform, tensor conversion.

Use these as executable examples of the APIs above.
