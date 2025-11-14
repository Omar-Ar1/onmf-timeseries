"""
Visualization utilities for Online NMF analysis.

Provides functions for plotting dictionary atoms, predictions, and analysis results.
"""

from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ONMFVisualizer:
    """
    Visualization utilities for Online NMF models.
    
    Example:
        >>> viz = ONMFVisualizer(model, feature_names=['AAPL_Close', 'GOOGL_Close'])
        >>> viz.plot_dictionary_atoms(top_n=12)
        >>> viz.plot_predictions(X_true, X_pred, asset_name='AAPL')
    """

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize visualizer.

        Args:
            model: Fitted OnlineNMF model.
            feature_names: Names of features for labeling plots.
        """
        self.model = model
        self.feature_names = feature_names or [
            f"Feature {i}" for i in range(model.d)
        ]

        if len(self.feature_names) != model.d:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names)}) "
                f"must match model dimensions ({model.d})"
            )

    def _plot_dictionary_atoms(
        self,
        top_n,
        save_path=None,
        title="Joint ONMF Dictionary Atoms",
        feature_tick_step=2,     # show every 2nd feature label
        timestep_tick_step=1     # show every timestep label
    ):
        """
        visualization of NMF dictionary atoms.
        """
        d, k, n_atoms = self.model.W_final.shape

        # Flatten W into (d*k, r)
        W_flat = self.model.W_final.reshape(d * k, n_atoms)

        # Sort atoms by importance
        importance = self.model.get_importance_scores()
        atom_indices = np.argsort(-importance)[:top_n]
        print(np.isnan(self.model.W_final).sum(), np.isinf(self.model.W_final).sum())

        # Grid layout
        n_cols = int(np.ceil(np.sqrt(top_n)))
        n_rows = int(np.ceil(top_n / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            squeeze=False,
        )

        # Feature names and timesteps (BEFORE downsampling)
        feature_names = np.array(self.feature_names)
        timesteps = np.array([f"T{i}" for i in range(k)])
        
        # Downsampled labels
        feature_labels = feature_names[::feature_tick_step]
        timestep_labels = timesteps[::timestep_tick_step]

        # Accumulate all shown values for global colorbar
        all_vals = []

        # create an axis on the right for the colorbar
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # [left, bottom, width, height]

        for idx, atom_id in enumerate(atom_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Reshape this atom (k × d)
            atom_vec = W_flat[:, atom_id]
            atom_mat = atom_vec.reshape(k, d).T  # → (d, k)
            
            # Downsample the matrix
            atom_mat = atom_mat[::feature_tick_step, ::timestep_tick_step] 

            im = ax.imshow(atom_mat, cmap='viridis', interpolation='nearest', aspect='auto')
            all_vals.append(atom_mat)

            ax.set_title(f"Atom {atom_id}", fontsize=11)

            # Set ticks to match the DOWNSAMPLED dimensions
            ax.set_yticks(np.arange(len(feature_labels)))
            ax.set_yticklabels(feature_labels, fontsize=8)

            ax.set_xticks(np.arange(len(timestep_labels)))
            ax.set_xticklabels(timestep_labels, rotation=45, ha='right', fontsize=8)

            # Remove grid lines completely
            ax.grid(False)

        # Turn off any unused subplots
        for j in range(top_n, n_rows * n_cols):
            r = j // n_cols
            c = j % n_cols
            axes[r, c].axis("off")

        # --- Shared colorbar ---
        all_vals = np.concatenate([a.flatten() for a in all_vals])
        vmin, vmax = all_vals.min(), all_vals.max()

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax)),
            cax=cax,
        )
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 0.90, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved dictionary atoms plot to {save_path}")

        return fig

    def plot_dictionary_atoms(
        self,
        top_n: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        feature_tick_step: int = 1,
    ) -> plt.Figure:
        """
        Plot dictionary atoms with importance scores.

        Args:
            top_n: Number of top atoms to plot (by importance).
            figsize: Figure size (width, height).
            save_path: Path to save figure.

        Returns:
            Matplotlib figure object.
        """
        if self.model.W_final is None:
            raise ValueError("Model not fitted")

        # Get dictionary and importance
        W_flat = self.model.W_final.view(
            self.model.d * self.model.k, self.model.r
        )
        importance = self.model.get_importance_scores()

        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]

        if top_n is not None:
            sorted_indices = sorted_indices[:top_n]
            num_atoms = top_n
        else:
            num_atoms = self.model.r

        # Determine grid layout
        ncols = min(4, num_atoms)
        nrows = (num_atoms + ncols - 1) // ncols

        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, atom_idx in enumerate(sorted_indices):
            ax = axes[i]

            # Extract atom: (d*k,) -> (d, k) -> (k, d)
            atom_flat = W_flat[:, atom_idx].cpu().numpy()
            atom = atom_flat.reshape(self.model.d, self.model.k).T

            # Plot each feature's temporal pattern
            for feat_idx, feat_name in enumerate(self.feature_names[::feature_tick_step]):
                ax.plot(
                    range(self.model.k),
                    atom[:, feat_idx],
                    marker='o',
                    label=feat_name,
                    alpha=0.7,
                )

            ax.set_title(
                f"Atom {atom_idx + 1}\n"
                f"Importance: {importance[atom_idx]:.2%}",
                fontsize=10,
            )
            ax.set_xlabel("Time Step")
            ax.set_xticks(np.arange(atom.shape[0]))
            ax.set_ylabel("Value")
            #ax.legend(fontsize=8, loc='best', frameon=True)
            ax.grid(True, alpha=0.3)

        # Collect all handles and labels from the first axis (assuming all share the same)
        handles, labels = axes[0].get_legend_handles_labels()

        # Add a common legend below all subplots
        fig.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(-0.1, 0.99),  
            ncol=1,
            fontsize=10,
            frameon=True,
        )

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()  # Leave space at bottom
        fig.suptitle(
            f'Joint Dictionary Atoms of {self.model.k} Timesteps Evolution',
                fontsize=14,
                y=1.02  # Push title slightly above the top
        )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dictionary atoms plot to {save_path}")

        return fig

    def plot_predictions(
        self,
        dates: pd.DatetimeIndex,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_indices: Optional[List[int]] = None,
        prediction_start_idx: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot true vs predicted values.

        Args:
            dates: DatetimeIndex for x-axis.
            y_true: True values, shape (T, d).
            y_pred: Predicted values, shape (T_pred, d).
            feature_indices: Which features to plot (default: all).
            prediction_start_idx: Index where predictions start.
            figsize: Figure size.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure object.
        """
        if feature_indices is None:
            feature_indices = list(range(len(self.feature_names)))

        n_features = len(feature_indices)
        ncols = min(2, n_features)
        nrows = (n_features + ncols - 1) // ncols

        if figsize is None:
            figsize = (8 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]

            # Plot true values
            ax.plot(
                dates[:len(y_true)],
                y_true[:, feat_idx],
                label='True',
                color='blue',
                alpha=0.7,
            )

            # Plot predictions
            if prediction_start_idx is not None:
                pred_dates = dates[prediction_start_idx:prediction_start_idx + len(y_pred)]
            else:
                pred_dates = dates[-len(y_pred):]

            ax.plot(
                pred_dates,
                y_pred[:, feat_idx],
                label='Predicted',
                color='red',
                linestyle='--',
                alpha=0.7,
            )

            # Add vertical line at prediction start
            if prediction_start_idx is not None:
                ax.axvline(
                    x=dates[prediction_start_idx],
                    color='green',
                    linestyle=':',
                    label='Prediction Start',
                    alpha=0.5,
                )

            ax.set_title(self.feature_names[feat_idx])
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")

        return fig

    def plot_atom_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot bar chart of atom importance scores.

        Args:
            top_n: Number of top atoms to show.
            figsize: Figure size.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure object.
        """
        importance = self.model.get_importance_scores()

        # Get top N
        top_indices = np.argsort(importance)[::-1][:top_n]
        top_importance = importance[top_indices]

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
        bars = ax.bar(
            range(len(top_indices)),
            top_importance,
            color=colors,
        )

        ax.set_xlabel('Atom Index')
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Top {top_n} Dictionary Atoms by Importance')
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels([f"{idx+1}" for idx in top_indices])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, top_importance):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.1%}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved importance plot to {save_path}")

        return fig


def plot_financial_overview(
    data: pd.DataFrame,
    title: str = "Financial Data Overview",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create overview plot of financial data.

    Args:
        data: DataFrame with financial data.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot prices
    ax = axes[0]
    for col in data.columns:
        if 'Close' in col[0]:
            ticker = col[1]
            ax.plot(data.index, data[col], label=ticker, alpha=0.7)

    ax.set_title(f"{title} - Closing Prices")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot volumes
    ax = axes[1]
    for col in data.columns:
        if 'Volume' in col[0]:
            ticker = col[1]
            ax.plot(data.index, data[col], label=ticker, alpha=0.7)

    ax.set_title("Trading Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved overview plot to {save_path}")

    return fig