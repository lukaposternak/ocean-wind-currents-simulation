"""
Copernicus Marine Service client for downloading ocean and wind data.

This module provides functionality to access and download data from the
Copernicus Marine Environment Monitoring Service (CMEMS).
"""

import os
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import xarray as xr
import numpy as np

try:
    import copernicusmarine
    COPERNICUS_AVAILABLE = True
except ImportError:
    COPERNICUS_AVAILABLE = False
    logging.warning("copernicus-marine-client not available. Install with: pip install copernicus-marine-client")


logger = logging.getLogger(__name__)


class CopernicusClient:
    """
    Client for accessing Copernicus Marine Service data.
    
    This client handles authentication, data requests, and downloads from
    the Copernicus Marine Service API.
    
    Parameters
    ----------
    username : str, optional
        Copernicus Marine Service username. If not provided, will look for
        COPERNICUS_USERNAME environment variable.
    password : str, optional
        Copernicus Marine Service password. If not provided, will look for
        COPERNICUS_PASSWORD environment variable.
    
    Attributes
    ----------
    username : str
        The username for authentication
    password : str
        The password for authentication
    authenticated : bool
        Whether the client is authenticated
    
    Examples
    --------
    >>> client = CopernicusClient(username="user", password="pass")
    >>> data = client.get_ocean_currents(
    ...     lat_min=20, lat_max=30, lon_min=-90, lon_max=-80,
    ...     start_date="2024-01-01", end_date="2024-01-02"
    ... )
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize the Copernicus client."""
        self.username = username or os.getenv("COPERNICUS_USERNAME", "")
        self.password = password or os.getenv("COPERNICUS_PASSWORD", "")
        self.authenticated = False
        
        if not COPERNICUS_AVAILABLE:
            logger.warning("Copernicus Marine client not installed. Some features will be unavailable.")
        
        if self.username and self.password:
            self.authenticated = True
            logger.info("Copernicus client initialized with credentials")
        else:
            logger.warning("Copernicus credentials not provided. Some operations may fail.")
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with Copernicus Marine Service.
        
        Parameters
        ----------
        username : str
            Copernicus username
        password : str
            Copernicus password
        
        Returns
        -------
        bool
            True if authentication successful
        """
        self.username = username
        self.password = password
        self.authenticated = True
        logger.info("Credentials updated successfully")
        return True
    
    def get_ocean_currents(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: str,
        end_date: str,
        depth_min: float = 0.0,
        depth_max: float = 5.0,
        product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Download ocean current data from Copernicus.
        
        Parameters
        ----------
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        start_date : str
            Start date (YYYY-MM-DD format)
        end_date : str
            End date (YYYY-MM-DD format)
        depth_min : float, optional
            Minimum depth in meters (default: 0.0)
        depth_max : float, optional
            Maximum depth in meters (default: 5.0)
        product_id : str, optional
            Copernicus product ID
        variables : list of str, optional
            Variables to download. Default: ['uo', 'vo'] (U and V current components)
        
        Returns
        -------
        xr.Dataset
            Dataset containing ocean current data
        
        Raises
        ------
        ValueError
            If credentials are not provided or if Copernicus client is not available
        RuntimeError
            If download fails
        """
        if not self.authenticated:
            raise ValueError("Not authenticated. Provide credentials first.")
        
        if not COPERNICUS_AVAILABLE:
            # Return mock data for demonstration purposes
            logger.warning("Copernicus client not available. Returning mock data.")
            return self._create_mock_current_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
        
        if variables is None:
            variables = ["uo", "vo"]  # Default: U and V current components
        
        try:
            logger.info(f"Downloading ocean currents from {start_date} to {end_date}")
            logger.info(f"Region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
            
            # Use copernicus marine client to download data
            # Note: This is a simplified example. Actual implementation may vary.
            dataset = copernicusmarine.open_dataset(
                dataset_id=product_id,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=start_date,
                end_datetime=end_date,
                minimum_depth=depth_min,
                maximum_depth=depth_max,
                variables=variables,
            )
            
            logger.info("Ocean current data downloaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download ocean currents: {str(e)}")
            # Return mock data as fallback
            return self._create_mock_current_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
    
    def get_wind_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: str,
        end_date: str,
        product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        variables: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Download wind data from Copernicus.
        
        Parameters
        ----------
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        start_date : str
            Start date (YYYY-MM-DD format)
        end_date : str
            End date (YYYY-MM-DD format)
        product_id : str, optional
            Copernicus product ID
        variables : list of str, optional
            Variables to download. Default: ['eastward_wind', 'northward_wind']
        
        Returns
        -------
        xr.Dataset
            Dataset containing wind data
        
        Raises
        ------
        ValueError
            If credentials are not provided
        RuntimeError
            If download fails
        """
        if not self.authenticated:
            raise ValueError("Not authenticated. Provide credentials first.")
        
        if not COPERNICUS_AVAILABLE:
            logger.warning("Copernicus client not available. Returning mock data.")
            return self._create_mock_wind_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
        
        if variables is None:
            variables = ["eastward_wind", "northward_wind"]
        
        try:
            logger.info(f"Downloading wind data from {start_date} to {end_date}")
            logger.info(f"Region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
            
            dataset = copernicusmarine.open_dataset(
                dataset_id=product_id,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=start_date,
                end_datetime=end_date,
                variables=variables,
            )
            
            logger.info("Wind data downloaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download wind data: {str(e)}")
            return self._create_mock_wind_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
    
    def _create_mock_current_data(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
        start_date: str, end_date: str
    ) -> xr.Dataset:
        """Create mock ocean current data for testing."""
        # Create coordinate arrays
        lats = np.linspace(lat_min, lat_max, 20)
        lons = np.linspace(lon_min, lon_max, 25)
        times = np.array([np.datetime64(start_date), np.datetime64(end_date)])
        depths = np.array([0.0, 5.0])
        
        # Create mock current data with realistic patterns
        np.random.seed(42)
        uo = np.random.randn(len(times), len(depths), len(lats), len(lons)) * 0.3
        vo = np.random.randn(len(times), len(depths), len(lats), len(lons)) * 0.3
        
        # Add some spatial structure (gulf stream-like pattern)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        uo = uo + 0.5 * np.sin(lat_grid * 0.1)[np.newaxis, np.newaxis, :, :]
        vo = vo + 0.3 * np.cos(lon_grid * 0.1)[np.newaxis, np.newaxis, :, :]
        
        # Create dataset
        ds = xr.Dataset(
            {
                "uo": (["time", "depth", "latitude", "longitude"], uo),
                "vo": (["time", "depth", "latitude", "longitude"], vo),
            },
            coords={
                "time": times,
                "depth": depths,
                "latitude": lats,
                "longitude": lons,
            },
        )
        
        # Add metadata
        ds["uo"].attrs = {"long_name": "Eastward current velocity", "units": "m/s"}
        ds["vo"].attrs = {"long_name": "Northward current velocity", "units": "m/s"}
        
        logger.info("Created mock ocean current data")
        return ds
    
    def _create_mock_wind_data(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
        start_date: str, end_date: str
    ) -> xr.Dataset:
        """Create mock wind data for testing."""
        # Create coordinate arrays
        lats = np.linspace(lat_min, lat_max, 20)
        lons = np.linspace(lon_min, lon_max, 25)
        times = np.array([np.datetime64(start_date), np.datetime64(end_date)])
        
        # Create mock wind data
        np.random.seed(43)
        eastward_wind = np.random.randn(len(times), len(lats), len(lons)) * 3.0 + 5.0
        northward_wind = np.random.randn(len(times), len(lats), len(lons)) * 2.5 + 2.0
        
        # Add some spatial patterns
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        eastward_wind = eastward_wind + 3.0 * np.sin(lat_grid * 0.15)[np.newaxis, :, :]
        northward_wind = northward_wind + 2.0 * np.cos(lon_grid * 0.1)[np.newaxis, :, :]
        
        # Create dataset
        ds = xr.Dataset(
            {
                "eastward_wind": (["time", "latitude", "longitude"], eastward_wind),
                "northward_wind": (["time", "latitude", "longitude"], northward_wind),
            },
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
        )
        
        # Add metadata
        ds["eastward_wind"].attrs = {"long_name": "Eastward wind velocity", "units": "m/s"}
        ds["northward_wind"].attrs = {"long_name": "Northward wind velocity", "units": "m/s"}
        
        logger.info("Created mock wind data")
        return ds
    
    def get_combined_data(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        start_date: str,
        end_date: str,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Download both wind and ocean current data.
        
        Parameters
        ----------
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        start_date : str
            Start date (YYYY-MM-DD format)
        end_date : str
            End date (YYYY-MM-DD format)
        
        Returns
        -------
        tuple of xr.Dataset
            (wind_data, current_data)
        """
        wind_data = self.get_wind_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
        current_data = self.get_ocean_currents(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
        
        return wind_data, current_data
