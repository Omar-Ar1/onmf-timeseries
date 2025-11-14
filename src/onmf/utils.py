from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute point estimates for standard regression metrics."""
    direction_acc, _ = _direction_accuracy_series(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Acc': direction_acc,
    }


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    block_length: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Compute metrics along with block-bootstrap confidence intervals.

    Args:
        y_true: Ground-truth values, shape (n_steps, n_features) or (n_steps,).
        y_pred: Predicted values with the same shape as y_true.
        n_bootstrap: Number of bootstrap replications.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        block_length: Optional block length for block bootstrap. If None, use
            an automatic rule-of-thumb based on the series length.
        random_state: Optional seed for reproducibility.

    Returns:
        Tuple where the first element is the point-estimate metrics dictionary
        (identical to compute_metrics) and the second element is a dictionary
        of (lower, upper) confidence interval bounds for each metric.
    """
    metrics = compute_metrics(y_true, y_pred)
    rng = np.random.default_rng(random_state)

    abs_err = _per_timestep_mean(np.abs(y_true - y_pred))
    sq_err = _per_timestep_mean((y_true - y_pred) ** 2)
    rel_err = _per_timestep_mean(
        np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))
    ) * 100
    _, direction_series = _direction_accuracy_series(y_true, y_pred)

    mae_samples = _block_bootstrap_means(
        abs_err, n_bootstrap, block_length, rng
    )
    rmse_samples = _block_bootstrap_means(
        sq_err, n_bootstrap, block_length, rng
    )
    mape_samples = _block_bootstrap_means(
        rel_err, n_bootstrap, block_length, rng
    )
    direction_samples = _block_bootstrap_means(
        direction_series, n_bootstrap, block_length, rng
    )

    mae_ci = _ci_from_samples(mae_samples, ci)
    rmse_ci = _ci_from_samples(
        np.sqrt(np.maximum(rmse_samples, 0.0)),
        ci,
    )
    mape_ci = _ci_from_samples(mape_samples, ci)
    direction_ci = _ci_from_samples(direction_samples, ci)

    return metrics, {
        'MAE': mae_ci,
        'RMSE': rmse_ci,
        'MAPE': mape_ci,
        'Direction_Acc': direction_ci,
    }


def _per_timestep_mean(values: np.ndarray) -> np.ndarray:
    """Aggregate values per time step while preserving temporal order."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.astype(float)
    return np.mean(arr, axis=1)


def _direction_accuracy_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return overall direction accuracy and per-step series (in %)."""
    true_diff = np.diff(y_true, axis=0)
    pred_diff = np.diff(y_pred, axis=0)

    if true_diff.size == 0:
        return np.nan, np.array([])

    if true_diff.ndim == 1:
        matches = (np.sign(true_diff) == np.sign(pred_diff)).astype(float)
    else:
        matches = np.mean(np.sign(true_diff) == np.sign(pred_diff), axis=1)

    series = matches * 100.0
    return float(series.mean()), series


def _block_bootstrap_means(
    series: np.ndarray,
    n_bootstrap: int,
    block_length: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap distribution of sample means using circular blocks."""
    series = np.asarray(series, dtype=float)
    if series.size == 0 or n_bootstrap <= 0:
        return np.array([])

    block_length = _resolve_block_length(series.size, block_length)
    samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = _sample_block_indices(series.size, block_length, rng)
        samples[i] = np.nanmean(series[idx])

    return samples


def _ci_from_samples(samples: np.ndarray, ci: float) -> Tuple[float, float]:
    """Convert bootstrap samples into percentile confidence intervals."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0 or not np.isfinite(samples).any():
        return (np.nan, np.nan)

    lower_q = (1.0 - ci) / 2.0 * 100.0
    upper_q = 100.0 - lower_q
    return (
        float(np.nanpercentile(samples, lower_q)),
        float(np.nanpercentile(samples, upper_q)),
    )


def _resolve_block_length(series_len: int, block_length: Optional[int]) -> int:
    """Pick a sensible block length if one is not provided."""
    if series_len <= 1:
        return series_len

    if block_length is None or block_length <= 0:
        # Rule-of-thumb: n^(1/3), at least 2 and at most n
        block_length = max(2, int(round(series_len ** (1 / 3))))

    return min(block_length, series_len)


def _sample_block_indices(
    series_len: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample contiguous circular blocks to preserve temporal structure."""
    if series_len == 0:
        return np.array([], dtype=int)

    indices = np.empty(series_len, dtype=int)
    pos = 0
    while pos < series_len:
        start = rng.integers(0, series_len)
        block = (start + np.arange(block_length)) % series_len
        take = min(block_length, series_len - pos)
        indices[pos:pos + take] = block[:take]
        pos += take
    return indices
