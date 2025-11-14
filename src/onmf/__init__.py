"""
Online NMF for Time-Series Forecasting.

A Python package for multi-dimensional time-series prediction using
Online Non-Negative Matrix Factorization (Online NMF).
"""

__version__ = "0.1.0"
__author__ = "Omar Arbi"

from .model import OnlineNMF
from .preprocessing import (
    FinancialDataLoader,
    FinancialPreprocessor,
    prepare_financial_data,
)
from .visualization import ONMFVisualizer, plot_financial_overview

__all__ = [
    "OnlineNMF",
    "FinancialDataLoader",
    "FinancialPreprocessor",
    "prepare_financial_data",
    "ONMFVisualizer",
    "plot_financial_overview",
]
