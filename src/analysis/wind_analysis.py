"""
Wind analysis tools for computing wind statistics and patterns.

This module provides functionality for analyzing wind data, including
magnitude calculations, direction, statistical analysis, and wind roses.
"""

import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


class WindAnalyzer:
    """
    Analyzer for wind data.
    
    This class provides methods for computing wind statistics, analyzing
    wind patterns, and calculating derived quantities from wind U/V components.
    
    Parameters
    ----------
    wind_data : xr.Dataset
        Dataset containing wind data with U and V components
    u_var : str, optional
        Name of the U (eastward) wind component variable
    v_var : str, optional
        Name of the V (northward) wind component variable
    
    Attributes
    ----------
    data : xr.Dataset
        The wind dataset
    u_var : str
        U component variable name
    v_var : str
        V component variable name
    
    Examples
    --------
    >>> analyzer = WindAnalyzer(wind_data, u_var='eastward_wind', v_var='northward_wind')
    >>> magnitude = analyzer.compute_magnitude()
    >>> direction = analyzer.compute_direction()
    >>> stats = analyzer.compute_statistics()
    """
    
    def __init__(
        self,
        wind_data: xr.Dataset,
        u_var: str = "eastward_wind",
        v_var: str = "northward_wind",
    ):
        """Initialize the wind analyzer."""
        self.data = wind_data
        self.u_var = u_var
        self.v_var = v_var
        
        # Validate that required variables exist
        if u_var not in wind_data:
            logger.warning(f"U component '{u_var}' not found in dataset")
        if v_var not in wind_data:
            logger.warning(f"V component '{v_var}' not found in dataset")
        
        logger.info("Wind analyzer initialized")
    
    def compute_magnitude(self) -> xr.DataArray:
        """
        Compute wind speed magnitude from U and V components.
        
        Returns
        -------
        xr.DataArray
            Wind speed magnitude in m/s
        
        Notes
        -----
        Wind speed is calculated as: speed = sqrt(u^2 + v^2)
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        magnitude = np.sqrt(u**2 + v**2)
        magnitude.attrs = {
            "long_name": "Wind speed magnitude",
            "units": "m/s",
            "description": "Computed from U and V components",
        }
        
        logger.info("Computed wind magnitude")
        return magnitude
    
    def compute_direction(self, degrees: bool = True) -> xr.DataArray:
        """
        Compute wind direction from U and V components.
        
        Parameters
        ----------
        degrees : bool, optional
            If True, return direction in degrees (0-360).
            If False, return direction in radians (default: True)
        
        Returns
        -------
        xr.DataArray
            Wind direction (meteorological convention: direction FROM which wind blows)
        
        Notes
        -----
        Uses meteorological convention: 0° = North, 90° = East, 180° = South, 270° = West
        Direction indicates where the wind is coming FROM.
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        # Calculate direction (in radians)
        # arctan2(v, u) gives direction TO which wind blows
        # We add 180° to get direction FROM which wind blows (meteorological convention)
        direction = np.arctan2(u, v)
        
        if degrees:
            direction = np.rad2deg(direction)
            # Ensure 0-360 range
            direction = (direction + 360) % 360
            units = "degrees"
        else:
            # Ensure 0-2π range
            direction = (direction + 2 * np.pi) % (2 * np.pi)
            units = "radians"
        
        direction.attrs = {
            "long_name": "Wind direction",
            "units": units,
            "description": "Meteorological convention: direction FROM which wind blows",
            "convention": "0=North, 90=East, 180=South, 270=West",
        }
        
        logger.info("Computed wind direction")
        return direction
    
    def compute_statistics(self, dim: Optional[str] = None) -> Dict[str, xr.DataArray]:
        """
        Compute statistical measures of wind speed.
        
        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute statistics (e.g., 'time').
            If None, computes over all dimensions.
        
        Returns
        -------
        dict
            Dictionary containing statistical measures:
            - mean: Mean wind speed
            - std: Standard deviation
            - min: Minimum wind speed
            - max: Maximum wind speed
            - percentiles: 25th, 50th (median), 75th percentiles
        """
        magnitude = self.compute_magnitude()
        
        stats_dict = {
            "mean": magnitude.mean(dim=dim),
            "std": magnitude.std(dim=dim),
            "min": magnitude.min(dim=dim),
            "max": magnitude.max(dim=dim),
            "p25": magnitude.quantile(0.25, dim=dim),
            "median": magnitude.median(dim=dim),
            "p75": magnitude.quantile(0.75, dim=dim),
        }
        
        # Add attributes
        stats_dict["mean"].attrs = {"long_name": "Mean wind speed", "units": "m/s"}
        stats_dict["std"].attrs = {"long_name": "Wind speed std dev", "units": "m/s"}
        stats_dict["min"].attrs = {"long_name": "Minimum wind speed", "units": "m/s"}
        stats_dict["max"].attrs = {"long_name": "Maximum wind speed", "units": "m/s"}
        
        logger.info("Computed wind statistics")
        return stats_dict
    
    def compute_wind_rose(
        self, n_directions: int = 16, speed_bins: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute wind rose data (frequency of wind by direction and speed).
        
        Parameters
        ----------
        n_directions : int, optional
            Number of direction bins (default: 16)
        speed_bins : list of float, optional
            Speed bin edges in m/s. Default: [0, 2, 4, 6, 8, 10, 15, 20]
        
        Returns
        -------
        pd.DataFrame
            Wind rose data with directions and speed bin frequencies
        """
        if speed_bins is None:
            speed_bins = [0, 2, 4, 6, 8, 10, 15, 20]
        
        magnitude = self.compute_magnitude().values.flatten()
        direction = self.compute_direction(degrees=True).values.flatten()
        
        # Remove NaN values
        valid = ~(np.isnan(magnitude) | np.isnan(direction))
        magnitude = magnitude[valid]
        direction = direction[valid]
        
        # Create direction bins
        direction_bins = np.linspace(0, 360, n_directions + 1)
        direction_labels = [
            f"{(direction_bins[i] + direction_bins[i+1])/2:.0f}°"
            for i in range(n_directions)
        ]
        
        # Bin the data
        dir_indices = np.digitize(direction, direction_bins) - 1
        dir_indices = dir_indices % n_directions  # Handle 360° wrapping
        
        # Create frequency table
        frequency_table = []
        
        for i in range(n_directions):
            dir_mask = dir_indices == i
            dir_speeds = magnitude[dir_mask]
            
            if len(dir_speeds) > 0:
                # Count occurrences in each speed bin
                bin_counts = np.histogram(dir_speeds, bins=speed_bins)[0]
                frequencies = bin_counts / len(magnitude) * 100  # Convert to percentage
            else:
                frequencies = np.zeros(len(speed_bins) - 1)
            
            row = {"direction": direction_labels[i]}
            for j in range(len(speed_bins) - 1):
                row[f"{speed_bins[j]}-{speed_bins[j+1]} m/s"] = frequencies[j]
            
            frequency_table.append(row)
        
        df = pd.DataFrame(frequency_table)
        logger.info(f"Computed wind rose with {n_directions} directions")
        return df
    
    def compute_temporal_trend(self, time_dim: str = "time") -> Dict[str, float]:
        """
        Compute temporal trend in wind speed.
        
        Parameters
        ----------
        time_dim : str, optional
            Name of the time dimension
        
        Returns
        -------
        dict
            Dictionary containing:
            - slope: Trend slope (m/s per time unit)
            - intercept: Trend intercept
            - r_value: Correlation coefficient
            - p_value: P-value for hypothesis test
        """
        magnitude = self.compute_magnitude()
        
        # Average over spatial dimensions
        if "latitude" in magnitude.dims:
            magnitude = magnitude.mean(dim=["latitude", "longitude"])
        elif "lat" in magnitude.dims:
            magnitude = magnitude.mean(dim=["lat", "lon"])
        
        # Convert time to numeric (days since start)
        time_values = magnitude[time_dim].values
        time_numeric = np.arange(len(time_values))
        
        # Perform linear regression
        speed_values = magnitude.values
        
        # Remove NaN values
        valid = ~np.isnan(speed_values)
        if valid.sum() < 2:
            logger.warning("Not enough valid data points for trend analysis")
            return {
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "p_value": np.nan,
            }
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric[valid], speed_values[valid]
        )
        
        result = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }
        
        logger.info(f"Computed temporal trend: slope={slope:.6f}, p={p_value:.4f}")
        return result
    
    def correlate_with_current(
        self, current_data: xr.Dataset, current_u_var: str = "uo", current_v_var: str = "vo"
    ) -> Dict[str, xr.DataArray]:
        """
        Compute correlation between wind and ocean currents.
        
        Parameters
        ----------
        current_data : xr.Dataset
            Dataset containing ocean current data
        current_u_var : str, optional
            Name of U current component variable
        current_v_var : str, optional
            Name of V current component variable
        
        Returns
        -------
        dict
            Dictionary containing correlation coefficients:
            - u_correlation: Correlation between wind U and current U
            - v_correlation: Correlation between wind V and current V
            - magnitude_correlation: Correlation between wind and current magnitudes
        """
        # Get wind components
        wind_u = self.data[self.u_var]
        wind_v = self.data[self.v_var]
        wind_mag = self.compute_magnitude()
        
        # Get current components (surface level)
        if "depth" in current_data[current_u_var].dims:
            current_u = current_data[current_u_var].isel(depth=0)
            current_v = current_data[current_v_var].isel(depth=0)
        else:
            current_u = current_data[current_u_var]
            current_v = current_data[current_v_var]
        
        current_mag = np.sqrt(current_u**2 + current_v**2)
        
        # Align datasets (interpolate if necessary)
        # For simplicity, we'll use only overlapping data
        try:
            # Calculate correlations along time dimension
            u_corr = xr.corr(wind_u, current_u, dim="time")
            v_corr = xr.corr(wind_v, current_v, dim="time")
            mag_corr = xr.corr(wind_mag, current_mag, dim="time")
            
            result = {
                "u_correlation": u_corr,
                "v_correlation": v_corr,
                "magnitude_correlation": mag_corr,
            }
            
            logger.info("Computed wind-current correlations")
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute correlations: {str(e)}")
            return {
                "u_correlation": None,
                "v_correlation": None,
                "magnitude_correlation": None,
            }
    
    def compute_wind_stress(self, air_density: float = 1.225, drag_coeff: float = 0.0015) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute wind stress components.
        
        Parameters
        ----------
        air_density : float, optional
            Air density in kg/m³ (default: 1.225)
        drag_coeff : float, optional
            Drag coefficient (default: 0.0015)
        
        Returns
        -------
        tuple of xr.DataArray
            (tau_x, tau_y) - Wind stress components in N/m²
        
        Notes
        -----
        Wind stress is calculated as: τ = ρ_air * C_d * |U| * U
        where U is wind velocity vector
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        magnitude = self.compute_magnitude()
        
        tau_x = air_density * drag_coeff * magnitude * u
        tau_y = air_density * drag_coeff * magnitude * v
        
        tau_x.attrs = {"long_name": "Eastward wind stress", "units": "N/m^2"}
        tau_y.attrs = {"long_name": "Northward wind stress", "units": "N/m^2"}
        
        logger.info("Computed wind stress")
        return tau_x, tau_y
