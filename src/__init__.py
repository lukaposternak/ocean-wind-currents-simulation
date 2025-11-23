"""
Ocean-Wind-Currents Simulation Package

A comprehensive Python package for simulating, analyzing, and visualizing
wind and superficial ocean currents.
"""

__version__ = "0.1.0"
__author__ = "Ocean Simulation Team"

from . import data
from . import analysis
from . import visualization
from . import simulation

__all__ = ["data", "analysis", "visualization", "simulation"]
