# Financial Markets Time-Series Forecasting

This example demonstrates Online NMF for multi-asset financial market prediction.

## Features

- **Multi-asset learning**: Learn correlation patterns across multiple stocks
- **Interpretable patterns**: Dictionary atoms reveal market regimes
- **Flexible features**: Support for prices, volumes, returns, volatility
- **Real-world data**: Uses Yahoo Finance API for open-source data

## Quick Start

### 1. Setup

```bash
# From project root
cd examples/financial_markets

# Install dependencies (if not already done)
pip install -r ../../requirements.txt
```

### 2. Run Analysis

```bash
# Use default configuration
python run_analysis.py

# Use custom configuration
python run_analysis.py --config my_config.yaml
```

### 3. View Results

Results are saved to `results/financial_markets/`:
- `figures/` - Visualizations
- `predictions/` - Prediction CSV files
- `models/` - Saved model states

## Configuration

Edit `config.yaml` to customize:

### Data Configuration

```yaml
data:
  tickers: [AAPL, GOOGL, MSFT]  # Asset symbols
  start_date: "2020-01-01"       # Start date
  features: [Close, Volume]      # Features to use
  compute_returns: true          # Use returns instead of prices
```

### Model Configuration

```yaml
model:
  r: 30        # Number of dictionary atoms
  k: 6         # Temporal window (days)
  N: 100       # History window
  lambd: 1.0   # L1 regularization
```

## Understanding the Results

### Dictionary Atoms

Each atom represents a temporal pattern learned from the data:
- **High importance atoms**: Common patterns across time
- **Shape interpretation**: 
  - Upward trend: Bullish pattern
  - Downward trend: Bearish pattern
  - Oscillation: Mean-reversion pattern

### Example Patterns

```
Atom 1 (15% importance):
- All assets moving up together
- Indicates bull market correlation

Atom 5 (8% importance):
- Tech stocks up, others flat
- Sector-specific movement

Atom 12 (5% importance):
- High volatility in Tesla
- Low correlation with others
```

### Predictions

The model makes one-step-ahead predictions by:
1. Fitting atoms to recent k-1 days
2. Reconstructing the full k-day window
3. Extracting the prediction from the last day

## Use Cases

### 1. Portfolio Risk Management
Monitor correlation patterns to identify market regime changes.

### 2. Anomaly Detection
Unusual atom activations may signal market disruptions.

### 3. Feature Engineering
Use atom coefficients as features for other ML models.

### 4. Market Regime Classification
Cluster atoms to identify bull/bear/sideways markets.

## Customization Examples

### Cryptocurrency Markets

```yaml
data:
  tickers: [BTC-USD, ETH-USD, BNB-USD, SOL-USD]
  features: [Close, Volume]
model:
  r: 40
  k: 7
  lambd: 0.3  # Lower regularization for high volatility
```

### Sector Analysis

```yaml
data:
  tickers: 
    - XLF  # Financial sector
    - XLK  # Technology sector
    - XLE  # Energy sector
    - XLV  # Healthcare sector
```

### High-Frequency Patterns

```yaml
data:
  start_date: "2024-01-01"
  interval: "1h"  # Hourly data
model:
  k: 12           # 12-hour window
  N: 168          # 1 week history
```

## Performance Notes

- **Training time**: ~2-5 minutes for 1000 samples on CPU
- **GPU acceleration**: Automatic if CUDA available
- **Memory usage**: ~500MB for typical configuration
- **Scalability**: Tested up to 50 assets × 2000 time steps

## Troubleshooting

### Issue: Poor predictions

**Solution**: Try:
- Increase `r` (more atoms)
- Increase `k` (longer temporal context)
- Adjust `lambd` (regularization)
- Use more training data

### Issue: Atoms not interpretable

**Solution**:
- Check data preprocessing (outliers, scaling)
- Reduce `r` (fewer, clearer patterns)
- Increase `lambd_min` (sparser initialization)

### Issue: Training too slow

**Solution**:
- Reduce `batch_size`
- Reduce `max_iter`
- Use fewer assets/features
- Enable GPU

## Advanced Topics

### Transfer Learning

Train on one set of assets, apply to another:

```python
# Train on large-cap tech
config1 = load_config('tech_config.yaml')
model = run_analysis(config1)

# Apply to small-cap tech
config2 = load_config('smallcap_config.yaml')
# Use model.W_final as initialization
```

### Ensemble Predictions

Combine multiple models with different hyperparameters:

```python
models = []
for config in config_list:
    model = OnlineNMF(**config['model'])
    model.fit(X_train, **config['training'])
    models.append(model)

# Average predictions
predictions = np.mean([m.predict(X_context) for m in models], axis=0)
```

## References

- Paper: Lyu et al. (2020). COVID-19 Time-series Prediction by Joint 
  Dictionary Learning and Online NMF. arXiv:2004.09112
- Yahoo Finance API: https://github.com/ranaroussi/yfinance

## Citation

If you use this code for research, please cite the original paper:

```bibtex
@article{lyu2020covid19,
  title={COVID-19 Time-series Prediction by Joint Dictionary Learning and Online NMF},
  author={Lyu, Hanbaek and Strohmeier, Christopher and Menz, Georg and Needell, Deanna},
  journal={arXiv preprint arXiv:2004.09112},
  year={2020}
}
```