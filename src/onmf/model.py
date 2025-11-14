"""
Online Non-Negative Matrix Factorization for Time-Series Forecasting.

This module implements Online NMF as described in:
    Lyu, H., et al. (2020). COVID-19 Time-series Prediction by Joint Dictionary 
    Learning and Online NMF. arXiv:2004.09112v1

Classes:
    OnlineNMF: Main class for online dictionary learning and prediction
"""

from typing import Optional, Tuple, List
import logging

import torch
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineNMF:
    """
    Online Non-Negative Matrix Factorization for multi-dimensional time-series.

    This implementation learns temporal dictionary patterns from correlated time-series
    data using a combination of minibatch and online learning approaches.

    Attributes:
        r (int): Number of dictionary atoms (rank).
        k (int): Sliding window size (temporal context).
        batch_size (int): Number of minibatches for initialization.
        N (int): History window size for online learning.
        max_iter (int): Maximum iterations for optimization sub-problems.
        tol (float): Convergence tolerance.
        device (torch.device): Computation device (CPU/CUDA).
        
    Example:
        >>> model = OnlineNMF(r=20, k=6, batch_size=50, N=100, max_iter=100)
        >>> model.fit(X, lambda_=1.0, beta=4.0, lambda_minB=3.0, beta_minB=1.0)
        >>> prediction = model.predict(X[-100:], lambda_pred=0.5)
    """

    def __init__(
        self,
        r: int,
        k: int,
        batch_size: int,
        N: int,
        max_iter: int,
        tol: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Online NMF model.

        Args:
            r: Number of dictionary atoms (rank of decomposition).
            k: Sliding window size (number of time steps in each sample).
            batch_size: Number of minibatches for dictionary initialization.
            N: History window size for online learning.
            max_iter: Maximum number of iterations for sub-problems.
            tol: Tolerance for early stopping during optimization.
            device: Torch device (CPU/CUDA). If None, automatically selects.

        Raises:
            ValueError: If any parameter is non-positive.
        """
        if r <= 0 or k <= 0 or batch_size <= 0 or N <= 0 or max_iter <= 0:
            raise ValueError("All parameters must be positive integers")
        if k > N:
            raise ValueError(f"Window size k={k} cannot exceed history size N={N}")

        self.r = r
        self.k = k
        self.batch_size = batch_size
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Model state (initialized during fit)
        self.d: int = -1  # Feature dimension
        self.W_final: Optional[torch.Tensor] = None  # Dictionary (d, k, r)
        self.A_final: Optional[torch.Tensor] = None  # Aggregate matrix (r, r)
        self.B_final: Optional[torch.Tensor] = None  # Aggregate matrix (r, d*k)
        self.code: Optional[torch.Tensor] = None  # Accumulated sparse codes

        logger.info(
            f"Initialized OnlineNMF: r={r}, k={k}, N={N}, device={self.device}"
        )

    def fit(
        self,
        X: torch.Tensor,
        lambda_: float,
        beta: float,
        lambda_minB: float,
        beta_minB: float,
    ) -> "OnlineNMF":
        """
        Fit the Online NMF model to time-series data.

        Combines minibatch learning (Algorithm 3) for initialization and
        online learning (Algorithm 1) for adaptive dictionary updates.

        Args:
            X: Input data matrix of shape (T, d) where T is time steps
               and d is feature dimension.
            lambda_: L1 regularization parameter for online phase.
            beta: Decay rate for dictionary updates (online phase).
            lambda_minB: L1 regularization for minibatch phase.
            beta_minB: Decay rate for minibatch phase.

        Returns:
            self: Fitted model instance.

        Raises:
            ValueError: If X has insufficient time steps or invalid shape.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

        T_total, self.d = X.shape

        if T_total < self.N:
            raise ValueError(
                f"Need at least N={self.N} time steps, got {T_total}"
            )

        X = X.to(self.device)
        logger.info(f"Fitting model on data with shape {X.shape}")

        # Phase 1: Minibatch initialization
        logger.info("Phase 1: Minibatch learning for initialization...")
        W, A, B = self._minibatch_learning(X, lambda_minB, beta_minB, T_total)
        logger.info("Minibatch learning complete.")

        # Phase 2: Online learning
        logger.info("Phase 2: Online learning with temporal adaptation...")
        H = torch.abs(torch.rand(self.r, self.N - self.k + 1)).to(self.device)
        all_codes = []

        for t in range(self.N, T_total):
            if t % 100 == 0 or t == T_total - 1:
                logger.info(f"Online step {t}/{T_total}")

            # Get current data window [t-N+1, t]
            X_window = X[t - self.N + 1 : t + 1]

            # Form Hankel-like tensor and flatten
            X_t_tensor = self._sliding_window(X_window, self.k).permute(
                2, 1, 0
            )
            X_t_flat = X_t_tensor.reshape(self.d * self.k, self.N - self.k + 1)

            # Flatten dictionary
            W_flat = W.view(self.d * self.k, self.r)

            # Update sparse codes H (Eq. 3)
            H = self._update_H(X_t_flat, W_flat, H, lambda_)
            all_codes.append(H)

            # Update dictionary W, aggregates A, B (Eq. 5-7)
            W, A, B = self._update_W(t, X_t_flat, W_flat, H, A, B, beta)

        logger.info("Online learning complete.")

        # Store final state
        self.W_final = W
        self.A_final = A
        self.B_final = B
        self.code = torch.sum(torch.stack(all_codes), dim=0)

        return self

    def predict(
        self, X_context: torch.Tensor, lambda_pred: float
    ) -> torch.Tensor:
        """
        Predict the next time step using learned dictionary (Algorithm 2).

        Args:
            X_context: Historical data of shape (T, d) where T >= k.
            lambda_pred: L1 regularization parameter for prediction.

        Returns:
            Predicted values for next time step, shape (d,).

        Raises:
            ValueError: If model not fitted or X_context too short.
            RuntimeError: If prediction fails.
        """
        if self.W_final is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if X_context.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X_context.shape}")

        t, d_context = X_context.shape

        if d_context != self.d:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.d}, got {d_context}"
            )

        if t < self.k:
            raise ValueError(
                f"Need at least k={self.k} time steps for prediction, got {t}"
            )

        X_context = X_context.to(self.device)

        # Get last k-1 data points
        X_last_k_minus_1 = X_context[-(self.k - 1) :, :]  # Shape (k-1, d)

        # Form flattened vector (k-1)*d
        X_last_flat = X_last_k_minus_1.T.reshape(self.d * (self.k - 1), 1)

        # Initialize sparse code
        H = torch.abs(torch.rand(self.r, 1)).to(self.device)

        # Get dictionary without last time step
        W_bar = self.W_final[:, : (self.k - 1), :]  # Shape (d, k-1, r)
        W_bar_flat = W_bar.reshape(self.d * (self.k - 1), self.r)

        # Solve for optimal H
        H = self._update_H(X_last_flat, W_bar_flat, H, lambda_pred)

        # Reconstruct full window
        W_final_flat = self.W_final.view(self.d * self.k, self.r)
        X_hat_flat = W_final_flat @ H

        # Reshape and extract prediction
        X_hat_tensor = (
            X_hat_flat.reshape(self.d, self.k, 1).permute(1, 0, 2).squeeze(-1)
        )
        prediction = X_hat_tensor[-1]  # Last time step

        return prediction

    def _minibatch_learning(
        self,
        X: torch.Tensor,
        lambda_minB: float,
        beta_minB: float,
        T_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Minibatch learning for dictionary initialization (Algorithm 3).

        Args:
            X: Input data matrix (T, d).
            lambda_minB: L1 regularization for sparse coding.
            beta_minB: Decay rate for dictionary updates.
            T_total: Total number of time steps.

        Returns:
            Tuple of (W, A, B):
                - W: Initial dictionary (d, k, r)
                - A: Aggregate matrix (r, r)
                - B: Aggregate matrix (r, d*k)
        """
        # Initialize
        W = torch.abs(torch.rand(self.d, self.k, self.r)).to(self.device)
        A = torch.zeros(self.r, self.r).to(self.device)
        B = torch.zeros(self.r, self.d * self.k).to(self.device)
        H_batch = torch.abs(torch.rand(self.r, self.N - self.k + 1)).to(
            self.device
        )

        for j in range(self.batch_size):
            # Sample random time point t >= N
            t = np.random.randint(self.N, T_total)

            # Get data window
            X_window = X[t - self.N + 1 : t + 1]

            # Form Hankel tensor and flatten
            X_t_tensor = self._sliding_window(X_window, self.k).permute(
                2, 1, 0
            )
            X_batch_flat = X_t_tensor.reshape(self.d * self.k, self.N - self.k + 1)

            W_flat = W.view(self.d * self.k, self.r)

            # Update H and dictionary
            H_batch = self._update_H(X_batch_flat, W_flat, H_batch, lambda_minB)
            W, A, B = self._update_W(
                j + 1, X_batch_flat, W_flat, H_batch, A, B, beta_minB
            )

        return W, A, B

    def _sliding_window(
        self, array: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        Generate sliding windows (Hankel matrix construction).

        Creates the X_t tensor from the paper (Eq. 4) using unfold operation.

        Args:
            array: Input time-series data of shape (N, d).
            k: Size of each sliding window.

        Returns:
            Sliding window tensor of shape (N-k+1, k, d).
        """
        # unfold(dimension, size, step) creates sliding windows
        # Result: (N-k+1, k, d)
        return array.unfold(0, k, 1).contiguous()

    def _update_H(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        H_init: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """
        Update sparse codes H using ISTA (Iterative Shrinkage-Thresholding).

        Solves: argmin_H ||X - WH||_F^2 + lambda_a * ||H||_1
        subject to H >= 0 (non-negativity constraint for NMF).

        Args:
            X: Data matrix (d*k, n).
            W: Dictionary matrix (d*k, r).
            H_init: Initial sparse codes (r, n) for warm start.
            lambda_: L1 regularization parameter.

        Returns:
            Optimized sparse codes H (r, n).
        """
        H = H_init.clone()

        # Precompute for efficiency
        WtW = W.T @ W
        WtX = W.T @ X

        # Lipschitz constant (largest eigenvalue of W^T W)
        L = torch.linalg.norm(WtW, ord=2)
        step_size = 1.0 / (2 * L + 1e-9)

        for _ in range(self.max_iter):
            H_old = H.clone()

            # Gradient: W^T(WH - X)
            grad = WtW @ H - WtX

            # Gradient descent step
            H_new = H - step_size * grad

            # Soft-thresholding with non-negativity
            H_new = torch.clamp(H_new - lambda_ * step_size, min=0)

            # Check convergence
            if torch.norm(H_new - H_old, p="fro") < self.tol:
                break

            H = H_new

        return H

    def _update_W(
        self,
        t: int,
        X_flat: torch.Tensor,
        W_flat: torch.Tensor,
        H: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update dictionary W using aggregate matrices A and B (Eq. 5-7).

        Uses HALS-like block coordinate descent for the W subproblem.

        Args:
            t: Current time step (for decay calculation).
            X_flat: Flattened data matrix (d*k, N-k+1).
            W_flat: Flattened dictionary (d*k, r).
            H: Sparse codes (r, N-k+1).
            A: Aggregate matrix (r, r).
            B: Aggregate matrix (r, d*k).
            beta: Decay rate exponent.

        Returns:
            Tuple of (W, A, B) with updated values.
        """
        t = max(t, 1)  # Prevent division by zero
        decay = t ** (-beta)

        # Update aggregates (Eq. 5, 6)
        A = (1 - decay) * A + decay * (H @ H.T)
        B = (1 - decay) * B + decay * (H @ X_flat.T)

        W_old = W_flat.clone()

        # Lipschitz constant (largest eigenvalue of W^T W)
        L = torch.linalg.norm(A, ord=2)
        step_size = 1.0 / (2 * L + 1e-9)

        # Column-wise dictionary update (Eq. 7)
        for _ in range(self.max_iter):
            #for j in range(self.r):
            # Gradient for column w_j
            grad = W_flat @ A - B.T

            # HALS update with non-negativity
            W_flat = torch.clamp(
                W_flat - step_size * grad, min=0
            )

            # Check convergence
            if torch.norm(W_flat - W_old, p="fro") < self.tol:
                break
            W_old = W_flat.clone()

        # Reshape back to tensor
        W_reshaped = W_flat.view(self.d, self.k, self.r)

        return W_reshaped, A, B

    def get_importance_scores(self) -> np.ndarray:
        """
        Compute importance metric for each dictionary atom.

        Returns:
            Normalized importance scores for each atom (length r).

        Raises:
            ValueError: If model not fitted.
        """
        if self.code is None:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.code.sum(dim=1).cpu().numpy()
        importance_normalized = importance / (importance.sum() + 1e-9)

        return importance_normalized

    def reset(self) -> None:
        """Reset model state (useful for re-training)."""
        self.W_final = None
        self.A_final = None
        self.B_final = None
        self.code = None
        logger.info("Model state reset")