"""
Analysis module for wind and ocean current data.

This module provides tools for analyzing wind and ocean current patterns,
including statistical analysis, correlation studies, and physical computations.
"""

from .wind_analysis import WindAnalyzer
from .current_analysis import CurrentAnalyzer

__all__ = ["WindAnalyzer", "CurrentAnalyzer"]
