"""
Ocean current analysis tools for computing current statistics and patterns.

This module provides functionality for analyzing ocean current data, including
magnitude calculations, direction, vorticity, divergence, and geostrophic calculations.
"""

import logging
from typing import Optional, Dict, Tuple
import numpy as np
import xarray as xr
from scipy import stats


logger = logging.getLogger(__name__)


class CurrentAnalyzer:
    """
    Analyzer for ocean current data.
    
    This class provides methods for computing current statistics, analyzing
    current patterns, and calculating derived quantities from current U/V components.
    
    Parameters
    ----------
    current_data : xr.Dataset
        Dataset containing current data with U and V components
    u_var : str, optional
        Name of the U (eastward) current component variable
    v_var : str, optional
        Name of the V (northward) current component variable
    
    Attributes
    ----------
    data : xr.Dataset
        The current dataset
    u_var : str
        U component variable name
    v_var : str
        V component variable name
    
    Examples
    --------
    >>> analyzer = CurrentAnalyzer(current_data, u_var='uo', v_var='vo')
    >>> magnitude = analyzer.compute_magnitude()
    >>> direction = analyzer.compute_direction()
    >>> vorticity = analyzer.compute_vorticity()
    """
    
    def __init__(
        self,
        current_data: xr.Dataset,
        u_var: str = "uo",
        v_var: str = "vo",
    ):
        """Initialize the current analyzer."""
        self.data = current_data
        self.u_var = u_var
        self.v_var = v_var
        
        # Validate that required variables exist
        if u_var not in current_data:
            logger.warning(f"U component '{u_var}' not found in dataset")
        if v_var not in current_data:
            logger.warning(f"V component '{v_var}' not found in dataset")
        
        logger.info("Current analyzer initialized")
    
    def compute_magnitude(self, depth_level: Optional[int] = 0) -> xr.DataArray:
        """
        Compute current speed magnitude from U and V components.
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0 for surface)
        
        Returns
        -------
        xr.DataArray
            Current speed magnitude in m/s
        
        Notes
        -----
        Current speed is calculated as: speed = sqrt(u^2 + v^2)
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        # Extract depth level if present
        if "depth" in u.dims:
            u = u.isel(depth=depth_level)
            v = v.isel(depth=depth_level)
        
        magnitude = np.sqrt(u**2 + v**2)
        magnitude.attrs = {
            "long_name": "Current speed magnitude",
            "units": "m/s",
            "description": "Computed from U and V components",
        }
        
        logger.info("Computed current magnitude")
        return magnitude
    
    def compute_direction(self, depth_level: Optional[int] = 0, degrees: bool = True) -> xr.DataArray:
        """
        Compute current direction from U and V components.
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0 for surface)
        degrees : bool, optional
            If True, return direction in degrees (0-360).
            If False, return direction in radians (default: True)
        
        Returns
        -------
        xr.DataArray
            Current direction (oceanographic convention: direction TO which current flows)
        
        Notes
        -----
        Uses oceanographic convention: 0° = North, 90° = East, 180° = South, 270° = West
        Direction indicates where the current is flowing TO.
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        # Extract depth level if present
        if "depth" in u.dims:
            u = u.isel(depth=depth_level)
            v = v.isel(depth=depth_level)
        
        # Calculate direction (oceanographic convention: direction TO)
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
            "long_name": "Current direction",
            "units": units,
            "description": "Oceanographic convention: direction TO which current flows",
            "convention": "0=North, 90=East, 180=South, 270=West",
        }
        
        logger.info("Computed current direction")
        return direction
    
    def compute_statistics(
        self, depth_level: Optional[int] = 0, dim: Optional[str] = None
    ) -> Dict[str, xr.DataArray]:
        """
        Compute statistical measures of current speed.
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0 for surface)
        dim : str, optional
            Dimension along which to compute statistics (e.g., 'time').
            If None, computes over all dimensions.
        
        Returns
        -------
        dict
            Dictionary containing statistical measures
        """
        magnitude = self.compute_magnitude(depth_level=depth_level)
        
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
        stats_dict["mean"].attrs = {"long_name": "Mean current speed", "units": "m/s"}
        stats_dict["std"].attrs = {"long_name": "Current speed std dev", "units": "m/s"}
        stats_dict["min"].attrs = {"long_name": "Minimum current speed", "units": "m/s"}
        stats_dict["max"].attrs = {"long_name": "Maximum current speed", "units": "m/s"}
        
        logger.info("Computed current statistics")
        return stats_dict
    
    def compute_vorticity(self, depth_level: Optional[int] = 0) -> xr.DataArray:
        """
        Compute relative vorticity (vertical component of curl).
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0 for surface)
        
        Returns
        -------
        xr.DataArray
            Relative vorticity in 1/s
        
        Notes
        -----
        Vorticity is calculated as: ζ = ∂v/∂x - ∂u/∂y
        Uses finite differences for spatial derivatives.
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        # Extract depth level if present
        if "depth" in u.dims:
            u = u.isel(depth=depth_level)
            v = v.isel(depth=depth_level)
        
        # Get coordinate names
        lat_coord = "latitude" if "latitude" in u.dims else "lat"
        lon_coord = "longitude" if "longitude" in u.dims else "lon"
        
        # Compute derivatives using central differences
        # Convert degrees to meters for proper units
        earth_radius = 6371000  # meters
        
        # Get coordinates
        lat = u[lat_coord]
        lon = u[lon_coord]
        
        # Compute grid spacing in meters
        dlat = np.gradient(lat.values)
        dlon = np.gradient(lon.values)
        
        # Convert to meters
        dy = dlat * (np.pi / 180) * earth_radius
        dx = dlon * (np.pi / 180) * earth_radius * np.cos(np.deg2rad(lat.values))
        
        # Compute derivatives
        dv_dx = np.gradient(v.values, axis=v.dims.index(lon_coord)) / dx[np.newaxis, :, np.newaxis]
        du_dy = np.gradient(u.values, axis=u.dims.index(lat_coord)) / dy[np.newaxis, :, np.newaxis]
        
        # Handle different dimension orders
        if lat_coord == u.dims[-1]:
            du_dy = du_dy.transpose()
        if lon_coord == v.dims[-1]:
            dv_dx = dv_dx.transpose()
        
        vorticity = xr.DataArray(
            dv_dx - du_dy,
            coords=u.coords,
            dims=u.dims,
            attrs={
                "long_name": "Relative vorticity",
                "units": "1/s",
                "description": "Vertical component of curl (∂v/∂x - ∂u/∂y)",
            },
        )
        
        logger.info("Computed vorticity")
        return vorticity
    
    def compute_divergence(self, depth_level: Optional[int] = 0) -> xr.DataArray:
        """
        Compute horizontal divergence.
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0 for surface)
        
        Returns
        -------
        xr.DataArray
            Horizontal divergence in 1/s
        
        Notes
        -----
        Divergence is calculated as: div = ∂u/∂x + ∂v/∂y
        Uses finite differences for spatial derivatives.
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        # Extract depth level if present
        if "depth" in u.dims:
            u = u.isel(depth=depth_level)
            v = v.isel(depth=depth_level)
        
        # Get coordinate names
        lat_coord = "latitude" if "latitude" in u.dims else "lat"
        lon_coord = "longitude" if "longitude" in u.dims else "lon"
        
        # Compute derivatives
        earth_radius = 6371000
        lat = u[lat_coord]
        lon = u[lon_coord]
        
        dlat = np.gradient(lat.values)
        dlon = np.gradient(lon.values)
        
        dy = dlat * (np.pi / 180) * earth_radius
        dx = dlon * (np.pi / 180) * earth_radius * np.cos(np.deg2rad(lat.values))
        
        du_dx = np.gradient(u.values, axis=u.dims.index(lon_coord)) / dx[np.newaxis, :, np.newaxis]
        dv_dy = np.gradient(v.values, axis=v.dims.index(lat_coord)) / dy[np.newaxis, :, np.newaxis]
        
        if lat_coord == v.dims[-1]:
            dv_dy = dv_dy.transpose()
        if lon_coord == u.dims[-1]:
            du_dx = du_dx.transpose()
        
        divergence = xr.DataArray(
            du_dx + dv_dy,
            coords=u.coords,
            dims=u.dims,
            attrs={
                "long_name": "Horizontal divergence",
                "units": "1/s",
                "description": "∂u/∂x + ∂v/∂y",
            },
        )
        
        logger.info("Computed divergence")
        return divergence
    
    def compute_geostrophic_current(
        self,
        ssh: xr.DataArray,
        latitude: Optional[xr.DataArray] = None,
        g: float = 9.81,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute geostrophic current from sea surface height.
        
        Parameters
        ----------
        ssh : xr.DataArray
            Sea surface height in meters
        latitude : xr.DataArray, optional
            Latitude coordinate. If None, extracted from ssh
        g : float, optional
            Gravitational acceleration (default: 9.81 m/s²)
        
        Returns
        -------
        tuple of xr.DataArray
            (u_geo, v_geo) - Geostrophic current components in m/s
        
        Notes
        -----
        Geostrophic balance: fv = -g∂η/∂x, fu = g∂η/∂y
        where f is the Coriolis parameter and η is sea surface height
        """
        if latitude is None:
            lat_coord = "latitude" if "latitude" in ssh.dims else "lat"
            latitude = ssh[lat_coord]
        
        # Compute Coriolis parameter
        omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        f = 2 * omega * np.sin(np.deg2rad(latitude))
        
        # Compute SSH gradients
        earth_radius = 6371000
        lon_coord = "longitude" if "longitude" in ssh.dims else "lon"
        lat_coord = "latitude" if "latitude" in ssh.dims else "lat"
        
        lat_vals = ssh[lat_coord].values
        lon_vals = ssh[lon_coord].values
        
        dlat = np.gradient(lat_vals)
        dlon = np.gradient(lon_vals)
        
        dy = dlat * (np.pi / 180) * earth_radius
        dx = dlon * (np.pi / 180) * earth_radius * np.cos(np.deg2rad(lat_vals))
        
        deta_dx = np.gradient(ssh.values, axis=ssh.dims.index(lon_coord)) / dx
        deta_dy = np.gradient(ssh.values, axis=ssh.dims.index(lat_coord)) / dy
        
        # Compute geostrophic velocities
        # u_geo = -g/f * ∂η/∂y
        # v_geo = g/f * ∂η/∂x
        u_geo = xr.DataArray(
            -g * deta_dy / f.values,
            coords=ssh.coords,
            dims=ssh.dims,
            attrs={"long_name": "Geostrophic eastward velocity", "units": "m/s"},
        )
        
        v_geo = xr.DataArray(
            g * deta_dx / f.values,
            coords=ssh.coords,
            dims=ssh.dims,
            attrs={"long_name": "Geostrophic northward velocity", "units": "m/s"},
        )
        
        logger.info("Computed geostrophic currents")
        return u_geo, v_geo
    
    def compute_kinetic_energy(self, depth_level: Optional[int] = 0, density: float = 1025) -> xr.DataArray:
        """
        Compute kinetic energy per unit volume.
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0)
        density : float, optional
            Water density in kg/m³ (default: 1025)
        
        Returns
        -------
        xr.DataArray
            Kinetic energy in J/m³
        
        Notes
        -----
        KE = 0.5 * ρ * (u² + v²)
        """
        u = self.data[self.u_var]
        v = self.data[self.v_var]
        
        if "depth" in u.dims:
            u = u.isel(depth=depth_level)
            v = v.isel(depth=depth_level)
        
        ke = 0.5 * density * (u**2 + v**2)
        ke.attrs = {
            "long_name": "Kinetic energy per unit volume",
            "units": "J/m^3",
            "description": "0.5 * ρ * (u² + v²)",
        }
        
        logger.info("Computed kinetic energy")
        return ke
    
    def compute_enstrophy(self, depth_level: Optional[int] = 0) -> xr.DataArray:
        """
        Compute enstrophy (squared vorticity).
        
        Parameters
        ----------
        depth_level : int, optional
            Depth level to extract (default: 0)
        
        Returns
        -------
        xr.DataArray
            Enstrophy in 1/s²
        
        Notes
        -----
        Enstrophy = 0.5 * ζ²
        where ζ is relative vorticity
        """
        vorticity = self.compute_vorticity(depth_level=depth_level)
        enstrophy = 0.5 * vorticity**2
        enstrophy.attrs = {
            "long_name": "Enstrophy",
            "units": "1/s^2",
            "description": "0.5 * vorticity²",
        }
        
        logger.info("Computed enstrophy")
        return enstrophy
