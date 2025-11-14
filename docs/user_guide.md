# User Guide

This guide walks through the recommended workflow for running Online NMF on multi-dimensional time-series with the utilities bundled in this repository.

---

## 1. Prerequisites

- Python 3.8+
- Recommended packages: `torch`, `pandas`, `yfinance`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
- (Optional) CUDA-capable GPU. The model automatically uses `torch.cuda.is_available()` to select the computation device.

Install the project locally:

```bash
git clone https://github.com/Omar-Ar1/onmf-timeseries.git
cd onmf-timeseries
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 2. Data Preparation

Use `prepare_financial_data` to download data, engineer returns (optional), split train/test sets, scale features, and ensure non-negativity:

```python
from onmf.preprocessing import prepare_financial_data

X_train, X_test, df_raw, metadata = prepare_financial_data(
    tickers=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    start="2020-01-01",
    end=None,
    features=["Close", "High", "Low", "Open"],
    train_ratio=0.9,
    compute_returns=False,
    scaling_method="minmax",
)
```

Outputs:

- `X_train`, `X_test`: PyTorch tensors `(T, d)`
- `df_raw`: pandas DataFrame before scaling (MultiIndex columns)
- `metadata`: scaler parameters, train/test sizes, feature names, etc.

If you provide your own dataset, mimic the preprocessing pipeline: handle missing values, scale features on the training split only, and shift/clip to keep values non-negative.

---

## 3. Training Online NMF

Instantiate `OnlineNMF` and call `fit` with hyperparameters derived from the paper:

```python
from onmf import OnlineNMF

model = OnlineNMF(
    r=10,          # atoms
    k=6,           # temporal window length
    batch_size=20, # minibatch size for initialization
    N=100,         # history window for online stage
    max_iter=800,  # iterations per subproblem
    tol=1e-5,
)

model.fit(
    X_train,
    lambda_=1.0,
    beta=4.0,
    lambda_minB=3.0,
    beta_minB=1.0,
)
```

During fitting, the class performs:

1. Minibatch initialization via sliding windows (`_minibatch_learning`)
2. Alternating updates of sparse codes `H` and dictionary `W`
3. Online updates with exponentially decayed aggregates (`A`, `B`)

Successful training stores the learned dictionary in `model.W_final` and aggregated codes in `model.code`.

---

## 4. Forecasting

To make one-step-ahead predictions, provide the most recent `N` observations:

```python
context = torch.cat([X_train, X_test[:i]], dim=0)  # roll forward
y_pred = model.predict(context, lambda_pred=0.5)
```

`predict` reconstructs the `(d, k)` dictionary slices corresponding to the newest window and returns the last column as the forecast. Iterate this loop across the test horizon for rolling predictions.

---

## 5. Baselines and Metrics

The example pipeline compares ONMF to:

- Naive persistence (`BaselineModels.naive_persistence`)
- Moving average (`BaselineModels.moving_average`)
- Ridge-based linear autoregression (`BaselineModels.linear_autoregression`)
- ARIMA (single-feature, optional)

Evaluate performance via `compute_metrics` or `compute_metrics_with_ci`:

```python
from onmf.utils import compute_metrics_with_ci

metrics, bounds = compute_metrics_with_ci(
    y_true,
    y_pred,
    n_bootstrap=1000,
    ci=0.95,
    block_length=None,
    random_state=42,
)
```

Metrics include MAE, RMSE, MAPE, and Direction Accuracy. Confidence intervals rely on circular block bootstrap sampling to respect temporal ordering.

---

## 6. Visualization

`src/onmf/visualization.py` supplies:

- `ONMFVisualizer.plot_dictionary_atoms`: heatmaps of the most important atoms (features × timesteps)
- `ONMFVisualizer.plot_atom_importance`: bar chart of normalized activation weights
- `ONMFVisualizer.plot_predictions`: compare ground truth vs multiple models
- `plot_financial_overview`: quick look at raw prices and volumes

All plotting functions accept optional `save_path` parameters for automation within pipelines.

---

## 7. Full Example Workflow

The `examples/financial_markets/run_analysis.py` script encapsulates the entire workflow:

1. Load configuration (`config.yaml`)
2. Prepare data (`prepare_financial_data`)
3. Train Online NMF
4. Generate rolling predictions
5. Train/evaluate baseline models
6. Compute metrics + bootstrap CIs
7. Save figures and CSV outputs under `results/financial_markets/<run_name>/`

Run it from the project root:

```bash
cd examples/financial_markets
python run_analysis.py --config config.yaml
```

All artifacts (predictions, metrics, figures, configurations) are versioned inside the `results/` directory for reproducibility.

---

## 8. Troubleshooting

- **Exploding/NaN atoms**: check that inputs are non-negative and adjust `lambda_` or `lambda_minB`.
- **Slow convergence**: lower `max_iter`, reduce `r`, or run on GPU.
- **Poor direction accuracy**: increase `k` for more temporal context or blend ONMF with linear models for directionality.
- **Yahoo Finance download failures**: set `FinancialDataLoader(cache_dir=...)` and retry to avoid rate limits.

---

## 9. Next Steps

- Experiment with non-financial datasets by plugging in alternative loaders.
- Use learned dictionaries as features for other predictive models.
- Extend the evaluation suite with multi-step forecasting or regime classification.

For API specifics, consult `docs/api_reference.md` and the inline docstrings in `src/onmf/`.
