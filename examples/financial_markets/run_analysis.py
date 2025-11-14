"""
Financial Markets Time-Series Forecasting with Online NMF.

This script demonstrates the application of Online NMF to multi-asset
financial market prediction, learning correlation patterns across assets
and features.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import yaml

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.onmf.model import OnlineNMF
from src.onmf.preprocessing import prepare_financial_data
from src.onmf.visualization import ONMFVisualizer, plot_financial_overview
from src.onmf.baselines import BaselineModels
from src.onmf.utils import compute_metrics, compute_metrics_with_ci



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_financial_analysis(config: dict) -> dict:
    """
    Run comprehensive analysis with proper evaluation and baselines.
    """
    logger.info("=" * 80)
    logger.info("Financial Time-Series Forecasting with Online NMF")
    logger.info("=" * 80)
    
    data_config = config['data']
    model_config = config['model']
    output_config = config.get('output', {})
    
    results_dir = Path(output_config.get('results_dir', 'results'))
    figures_dir = results_dir / 'figures'
    for d in [figures_dir, results_dir / 'models', results_dir / 'predictions']:
        d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 80)
    
    X_train, X_test, original_data, metadata = prepare_financial_data(
        tickers=data_config['tickers'],
        start=data_config['start_date'],
        end=data_config.get('end_date'),
        features=data_config.get('features', ['Close', 'Volume']),
        train_ratio=data_config.get('train_ratio', 0.8),
        compute_returns=data_config.get('compute_returns', False),
    )
    
    train_size = metadata['train_size']
    test_size = metadata['test_size']
    n_predictions = len(X_test)
    
    # =========================================================================
    # 2. Train Online NMF
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Training Online NMF")
    logger.info("=" * 80)
    
    model = OnlineNMF(
        r=model_config['r'],
        k=model_config['k'],
        batch_size=model_config['batch_size'],
        N=model_config['N'],
        max_iter=model_config['max_iter'],
    )
    start = time.time()
    model.fit(
        X_train,
        lambda_=model_config['lambda_'],
        beta=model_config['beta'],
        lambda_minB=model_config['lambda_minB'],
        beta_minB=model_config['beta_minB'],
    )
    
    # Make predictions
    onmf_predictions = []
    for i in range(n_predictions):
        context_end = train_size + i
        X_context = torch.cat([X_train, X_test[:i]]) if i > 0 else X_train
        pred = model.predict(X_context, lambda_pred=model_config.get('lambda_pred', 0.5))
        onmf_predictions.append(pred.cpu().numpy())
    elapsed = time.time() - start
    logger.info(f"Online NMF training and prediction completed in {elapsed:.2f} seconds.")
    onmf_predictions = np.array(onmf_predictions)
    
    # =========================================================================
    # 3. Run Baseline Models
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Running Baseline Models")
    logger.info("=" * 80)
    
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test[:n_predictions].cpu().numpy()
    
    baselines = BaselineModels()
    
    logger.info("Running Naive Persistence...")
    naive_pred = baselines.naive_persistence(X_train_np, n_predictions)
    
    logger.info("Running Moving Average...")
    ma_pred = baselines.moving_average(X_train_np, n_predictions, window=5)
    
    logger.info("Running Linear Autoregression...")
    start = time.time()
    ar_pred = baselines.linear_autoregression(X_train_np, n_predictions, lag=5)
    elapsed = time.time() - start
    logger.info(f"Linear Autoregression completed in {elapsed:.2f} seconds.")


    # =========================================================================
    # 4. Evaluate All Models
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Evaluation")
    logger.info("=" * 80)
    
    params = metadata["scaler"]  
    max_arr = params["max"].values
    min_arr = params['min'].values
    y_true = X_test_np * (max_arr - min_arr + 1e-9) + min_arr
    onmf_predictions = onmf_predictions * (max_arr - min_arr + 1e-9) + min_arr
    ma_pred = ma_pred * (max_arr - min_arr + 1e-9) + min_arr
    ar_pred = ar_pred * (max_arr - min_arr + 1e-9) + min_arr
    naive_pred = naive_pred * (max_arr - min_arr + 1e-9) + min_arr

    model_predictions = {
        'Online NMF': onmf_predictions,
        'Naive (Last Value)': naive_pred,
        'Moving Average': ma_pred,
        'Linear AR': ar_pred,
    }

    ci_config = output_config.get('confidence_intervals', {})
    ci_level = ci_config.get('level', 0.95)
    n_bootstrap = ci_config.get('n_bootstrap', 1000)
    block_length = ci_config.get('block_length')
    ci_random_state = ci_config.get('random_state')

    results = {}
    results_ci = {}
    for name, preds in model_predictions.items():
        metrics, cis = compute_metrics_with_ci(
            y_true,
            preds,
            n_bootstrap=n_bootstrap,
            ci=ci_level,
            block_length=block_length,
            random_state=ci_random_state,
        )
        results[name] = metrics
        results_ci[name] = cis
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 80)
    
    df_results = pd.DataFrame(results).T
    logger.info("\n" + df_results.to_string())

    logger.info(
        f"\nConfidence intervals (level={ci_level:.0%}, "
        f"block_length={block_length or 'auto'}, "
        f"n_bootstrap={n_bootstrap})"
    )
    for model_name in results:
        logger.info(f"{model_name}:")
        for metric_name, metric_value in results[model_name].items():
            lower, upper = results_ci[model_name][metric_name]
            if np.isfinite(lower) and np.isfinite(upper):
                logger.info(
                    f"  {metric_name}: {metric_value:.4f} "
                    f"[{lower:.4f}, {upper:.4f}]"
                )
            else:
                logger.info(
                    f"  {metric_name}: {metric_value:.4f} "
                    "(CI unavailable)"
                )
    
    # Determine best model
    best_model = df_results['MAE'].idxmin()
    logger.info(f"\n🏆 Best Model by MAE: {best_model}")
    
    # =========================================================================
    # 5. Visualizations
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Creating Visualizations")
    logger.info("=" * 80)
    
    feature_names = metadata['feature_names']
    
    # Plot 1: Model Comparison (first 4 features)
    fig, axes = plt.subplots(4, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    dates = original_data.index
    train_size = len(X_train_np)
    pred_dates = dates[train_size:train_size + n_predictions]
    train_dates = dates[:train_size + 1]
    print("Train Dates: ", len(train_dates))
    X_train_np = X_train_np * (max_arr - min_arr + 1e-9) + min_arr

    for idx in range(min(8, len(feature_names))):
        ax = axes[idx]
        
        # Plot true values
        ax.plot(dates[-3 * test_size:], np.array(original_data)[-3 * test_size:, idx], 
                label='Train', color='gray', alpha=0.5)
        ax.plot(pred_dates, y_true[:, idx], 
                label='True', color='black', linewidth=2)
        
        # Plot predictions
        ax.plot(pred_dates, onmf_predictions[:, idx], 
                label='Online NMF', linestyle='--', marker='o', markersize=3)
        ax.plot(pred_dates, naive_pred[:, idx], 
                label='Naive', linestyle='--', alpha=0.6)
        ax.plot(pred_dates, ma_pred[:, idx], 
                label='MA', linestyle='--', alpha=0.6)
        ax.plot(pred_dates, ar_pred[:, idx], 
                label='Linear AR', linestyle='--', alpha=0.6)
        
        ax.axvline(x=dates[train_size], color='red', linestyle=':', alpha=0.5)
        ax.set_title(f'{feature_names[idx]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.suptitle(
        f'ONMF Predictions over a {n_predictions} Days Horizon',
        fontsize=14,
        y=1.02 
        )
    plt.tight_layout()
    plt.savefig(figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()
    
    # Plot 2: Metrics Comparison with Confidence Intervals
    metric_names = ['MAE', 'RMSE', 'MAPE', 'Direction_Acc']
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        pairs = sorted(
            [(model, results[model][metric]) for model in results.keys()],
            key=lambda x: x[1],
        )[::-1]
        values = [p[1] for p in pairs]
        models = [p[0] for p in pairs]
        
        bounds = [results_ci[m][metric] for m in models]
        lower_err = [
            max(val - bound[0], 0.0) if np.isfinite(bound[0]) else 0.0
            for val, bound in zip(values, bounds)
        ]
        upper_err = [
            max(bound[1] - val, 0.0) if np.isfinite(bound[1]) else 0.0
            for val, bound in zip(values, bounds)
        ]
        yerr = np.vstack([lower_err, upper_err])
        
        if metric == 'Direction_Acc':
            colors = [
                '#2ecc71' if i == 0 else "#bce1e4"
                for i in range(len(models))
            ]
        else:
            colors = [
                '#2ecc71' if i == len(models) - 1 else "#bce1e4"
                for i in range(len(models))
            ]
        
        bars = ax.bar(
            range(len(models)),
            values,
            color=colors,
            yerr=yerr,
            capsize=4,
            edgecolor='#555555',
            linewidth=0.8,
        )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(
            f'{metric} Comparison (Lower is Better)'
            if metric != 'Direction_Acc'
            else f'{metric} Comparison (Higher is Better)'
        )
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    fig.suptitle(
        f'ONMF {n_predictions} Days Predictions Evaluation',
        fontsize=14,
        y=1.02 
        )
    plt.tight_layout()
    plt.savefig(figures_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Dictionary atoms (if ONMF is competitive)
    viz = ONMFVisualizer(model, feature_names=feature_names)
    viz.plot_dictionary_atoms(top_n=min(10, model_config['r']), 
                                save_path=figures_dir / 'dictionary_atoms.png', feature_tick_step=4)
    plt.close()
    
    viz.plot_atom_importance(top_n=min(10, model_config['r']),
                            save_path=figures_dir / 'atom_importance.png')
    plt.close()


    # =========================================================================
    # 6. Save Results
    # =========================================================================
    df_results.to_csv(results_dir / 'predictions' / 'metrics_comparison.csv')
    
    ci_rows = []
    for model_name, metric_bounds in results_ci.items():
        row = {}
        for metric_name, (lower, upper) in metric_bounds.items():
            row[f'{metric_name}_lower'] = lower
            row[f'{metric_name}_upper'] = upper
        ci_rows.append(pd.Series(row, name=model_name))
    
    if ci_rows:
        df_ci = pd.DataFrame(ci_rows)
        df_ci.to_csv(
            results_dir / 'predictions' / 'metrics_confidence_intervals.csv'
        )
    
    # Save predictions
    pred_df = pd.DataFrame({
        'True': y_true[:, 0],
        'ONMF': onmf_predictions[:, 0],
        'Naive': naive_pred[:, 0],
        'MA': ma_pred[:, 0],
        'AR': ar_pred[:, 0],
    }, index=pred_dates)
    pred_df.to_csv(results_dir / 'predictions' / 'predictions_comparison.csv')
    
    logger.info(f"\n✓ Results saved to {results_dir}")
    
    return {
        'results': results,
        'results_ci': results_ci,
        'best_model': best_model,
        'predictions': {
            'onmf': onmf_predictions,
            'naive': naive_pred,
            'ma': ma_pred,
            'ar': ar_pred,
        },
        'y_true': y_true,
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Financial Markets Analysis with Online NMF'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Run analysis
    try:
        results = run_financial_analysis(config)
        logger.info("\n✓ Analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
