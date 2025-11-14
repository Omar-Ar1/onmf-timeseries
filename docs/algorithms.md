# Algorithm Notes

This document explains the learning strategy implemented in `onmf-timeseries`, highlighting how the code mirrors the procedure from “COVID-19 Time-series Prediction by Joint Dictionary Learning and Online NMF” (Lyu et al., 2020).

---

## Problem Setup

![Online NNMF Illustration](docs/03_onmf_illustration.png)

We observe a multi-dimensional time-series \( X \in \mathbb{R}_{+}^{T \times d} \) (e.g., closing prices across assets). The Online NMF objective is to factorize overlapping windows of length \( k \) into a dictionary \( W \in \mathbb{R}_{+}^{d \times k \times r} \) and sparse activation codes \( H \in \mathbb{R}_{+}^{r \times n} \) such that:

\[
\min_{W, H} \frac{1}{2} \| X_{\text{windows}} - W H \|_F^2 + \lambda \| H \|_1 \quad \text{s.t.} \quad W \ge 0, H \ge 0
\]

Here, \( r \) is the number of dictionary atoms (rank) and \( X_{\text{windows}} \) stacks flattened sliding windows of shape \( d \cdot k \).

---

## Sliding-Window Construction

`OnlineNMF._sliding_window` builds overlapping windows from the most recent \( N \) observations:

1. Extract the last \( N \) rows of \( X \).
2. For each position \( t \), stack \( k \) consecutive steps into a 3D tensor `(num_windows, k, d)`.
3. Flatten each window to `(d * k)` before feeding to the optimization routines.

This transformation mimics Algorithm 3 from the paper, ensuring consistent input shape for both minibatch and online phases.

---

## Minibatch Initialization (Algorithm 3)

Purpose: bootstrap a reasonable dictionary before switching to online updates.

Implementation highlights (`OnlineNMF._minibatch_learning`):

- Randomly sample `batch_size` windows from the initial portion of the series.
- Initialize `W` with non-negative random values.
- Alternately solve for `H` (sparse codes) via `_update_H` and for `W` via projected gradient updates.
- Maintain aggregate matrices \( A = H H^\top \) and \( B = H X^\top \) that serve as sufficient statistics for online updates.

Result: `W_init`, `A_init`, `B_init` that seed the online loop.

---

## Online Updates (Algorithm 1)

For each new window arriving after initialization:

1. **Sparse Coding**: solve for \( H_t \) using `_update_H`. This is a projected gradient method with step size derived from the Lipschitz constant \( \| W^\top W \|_2 \). After each iteration:
   - Gradient descent on \( H \)
   - Soft-threshold with non-negativity clamp: `H_new = clamp(H - step * grad - lambda * step, min=0)`
2. **Aggregate Update**:
   \[
   A_t = (1 - \gamma_t) A_{t-1} + \gamma_t H_t H_t^\top,\quad
   B_t = (1 - \gamma_t) B_{t-1} + \gamma_t H_t X_t^\top
   \]
   where \( \gamma_t = t^{-\beta} \) controls the decay (`beta` parameter).
3. **Dictionary Update**: treat `W_flat` as a `(d*k) × r` matrix and perform HALS-like steps:
   - Compute gradient `W_flat @ A_t - B_t.T`
   - Update `W_flat = clamp(W_flat - step * grad, min=0)`
   - Stop when Frobenius-norm change drops below `tol` or `max_iter` is reached.

The algorithm balances historical information (through `A`, `B`) with adaptability via the decay exponent `beta`.

---

## Prediction Mechanism

To forecast one step ahead, `OnlineNMF.predict`:

1. Takes the last `N` observations from the provided context tensor.
2. Builds windows and aggregates them into \(\overline{W}\), the mean dictionary over time to reduce noise.
3. Solves for codes with `lambda_pred` using `_update_H`.
4. Reconstructs the full window and returns the final column (the next timestep).

This follows the paper’s “online prediction with averaged dictionary” strategy.

---

## Regularization Parameters

- `lambda_minB`: L1 penalty during the minibatch initialization; enforces sparsity in early codes.
- `beta_minB`: decay exponent during initialization (usually 1.0).
- `lambda_`: L1 penalty during online updates.
- `beta`: decay exponent once running online; higher values emphasize recent windows.
- `lambda_pred`: optional L1 penalty during prediction to control code sparsity.

These settings trade off adaptiveness versus stability.

---

## Baselines

The repository pairs Online NMF with interpretable baselines:

- **Naive Persistence**: copies the last observation forward.
- **Moving Average**: recursively averages the last `window` outputs to produce smoother predictions.
- **Linear Autoregression**: fits Ridge regression on flattened lag windows per feature, approximating VAR behavior without cross-feature coefficients.
- **ARIMA**: classical single-series ARIMA for reference (computationally heavier).

Comparing against these baselines ensures improvements are not due to data leakage or scaling artifacts.

---

## Metrics & Confidence Intervals

Evaluation focuses on MAE, RMSE, MAPE, and Direction Accuracy (percentage of correctly predicted direction changes). To quantify uncertainty, `compute_metrics_with_ci` applies block bootstrap sampling:

1. Compute per-timestep error series.
2. Sample circular blocks of length \( L \approx n^{1/3} \) unless overridden.
3. Aggregate means across bootstrap replicates.
4. Use percentile intervals to obtain lower/upper bounds for each metric.

This preserves temporal dependence and provides statistically grounded comparisons.

---

## Visualization Strategy

`ONMFVisualizer` renders:

- **Dictionary atoms**: heatmaps of each atom (features vs. timesteps) sorted by importance.
- **Atom importance**: normalized bar charts showing contribution to reconstruction.
- **Prediction comparisons**: overlays of true vs predicted series for multiple models with confidence shading.

These plots align with the interpretability emphasis of Online NMF, helping diagnose how atoms relate to market regimes.

---

## Summary

The implementation mirrors the joint dictionary learning and online refinement described by Lyu et al., adapting it to PyTorch with:

- Sliding-window tensorization
- Dual-phase learning (minibatch + online)
- Projected gradient + HALS updates
- Averaged-dictionary forecasting
- Evaluation/visualization/CI tooling

This pipeline enables reproducible experiments on financial time-series and can be generalized to any multi-dimensional, non-negative data stream.
