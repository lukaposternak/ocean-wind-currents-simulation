"""
Data acquisition module for ocean and wind data.

This module provides clients for accessing data from various sources:
- Copernicus Marine Service
- HYCOM (Hybrid Coordinate Ocean Model)
"""

from .copernicus_client import CopernicusClient
from .hycom_client import HYCOMClient

__all__ = ["CopernicusClient", "HYCOMClient"]
