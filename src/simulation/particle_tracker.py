"""
Lagrangian particle tracking for ocean current simulations.

This module implements particle tracking using ocean current and wind fields
with various numerical integration schemes.
"""

import logging
from typing import Optional, Tuple, List, Union
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


logger = logging.getLogger(__name__)


class ParticleTracker:
    """
    Lagrangian particle tracker for ocean currents.
    
    This class implements particle tracking using ocean current fields
    with optional wind drift effects. Supports multiple numerical
    integration schemes (Euler, RK2, RK4).
    
    Parameters
    ----------
    current_u : xr.DataArray
        Eastward current component (m/s)
    current_v : xr.DataArray
        Northward current component (m/s)
    wind_u : xr.DataArray, optional
        Eastward wind component (m/s)
    wind_v : xr.DataArray, optional
        Northward wind component (m/s)
    wind_drift_coefficient : float, optional
        Wind drift coefficient (0-1, typically 0.02-0.04). Default: 0.03
    integration_method : str, optional
        Numerical integration method: 'euler', 'rk2', 'rk4'. Default: 'rk4'
    apply_land_mask : bool, optional
        Whether to stop particles at land boundaries. Default: True
    
    Attributes
    ----------
    current_u : xr.DataArray
        Current U component
    current_v : xr.DataArray
        Current V component
    wind_u : xr.DataArray or None
        Wind U component
    wind_v : xr.DataArray or None
        Wind V component
    wind_drift_coef : float
        Wind drift coefficient
    method : str
        Integration method
    
    Examples
    --------
    >>> tracker = ParticleTracker(current_u, current_v, integration_method='rk4')
    >>> initial_positions = np.array([[lon1, lat1], [lon2, lat2]])
    >>> trajectories = tracker.track_particles(initial_positions, n_steps=100, dt=3600)
    """
    
    def __init__(
        self,
        current_u: xr.DataArray,
        current_v: xr.DataArray,
        wind_u: Optional[xr.DataArray] = None,
        wind_v: Optional[xr.DataArray] = None,
        wind_drift_coefficient: float = 0.03,
        integration_method: str = "rk4",
        apply_land_mask: bool = True,
    ):
        """Initialize the particle tracker."""
        self.current_u = current_u
        self.current_v = current_v
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.wind_drift_coef = wind_drift_coefficient
        self.method = integration_method.lower()
        self.apply_land_mask = apply_land_mask
        
        # Get coordinates
        self.lat_coord = "latitude" if "latitude" in current_u.dims else "lat"
        self.lon_coord = "longitude" if "longitude" in current_u.dims else "lon"
        
        # Extract surface level if depth dimension exists
        if "depth" in current_u.dims:
            self.current_u = current_u.isel(depth=0)
            self.current_v = current_v.isel(depth=0)
        
        # Get coordinate arrays
        self.lats = self.current_u[self.lat_coord].values
        self.lons = self.current_u[self.lon_coord].values
        
        # Earth radius for coordinate conversion
        self.earth_radius = 6371000.0  # meters
        
        # Validate method
        if self.method not in ["euler", "rk2", "rk4"]:
            logger.warning(f"Unknown method '{self.method}', using 'rk4'")
            self.method = "rk4"
        
        logger.info(f"Particle tracker initialized with method: {self.method}")
        logger.info(f"Wind drift coefficient: {self.wind_drift_coef}")
    
    def _get_velocity_at_point(
        self,
        lon: float,
        lat: float,
        time_index: int = 0,
    ) -> Tuple[float, float]:
        """
        Get velocity at a specific point using interpolation.
        
        Parameters
        ----------
        lon : float
            Longitude
        lat : float
            Latitude
        time_index : int
            Time index
        
        Returns
        -------
        tuple
            (u, v) velocity components in m/s
        """
        # Check bounds
        if (
            lon < self.lons.min()
            or lon > self.lons.max()
            or lat < self.lats.min()
            or lat > self.lats.max()
        ):
            return np.nan, np.nan
        
        # Get data for this time step
        if "time" in self.current_u.dims:
            u_data = self.current_u.isel(time=time_index).values
            v_data = self.current_v.isel(time=time_index).values
        else:
            u_data = self.current_u.values
            v_data = self.current_v.values
        
        # Create interpolators
        try:
            u_interp = RegularGridInterpolator(
                (self.lats, self.lons),
                u_data,
                bounds_error=False,
                fill_value=np.nan,
            )
            v_interp = RegularGridInterpolator(
                (self.lats, self.lons),
                v_data,
                bounds_error=False,
                fill_value=np.nan,
            )
            
            # Interpolate
            u = u_interp([lat, lon])[0]
            v = v_interp([lat, lon])[0]
            
            # Add wind drift if available
            if self.wind_u is not None and self.wind_v is not None:
                if "time" in self.wind_u.dims:
                    wind_u_data = self.wind_u.isel(time=time_index).values
                    wind_v_data = self.wind_v.isel(time=time_index).values
                else:
                    wind_u_data = self.wind_u.values
                    wind_v_data = self.wind_v.values
                
                wind_u_interp = RegularGridInterpolator(
                    (self.lats, self.lons),
                    wind_u_data,
                    bounds_error=False,
                    fill_value=0.0,
                )
                wind_v_interp = RegularGridInterpolator(
                    (self.lats, self.lons),
                    wind_v_data,
                    bounds_error=False,
                    fill_value=0.0,
                )
                
                wind_u = wind_u_interp([lat, lon])[0]
                wind_v = wind_v_interp([lat, lon])[0]
                
                u += self.wind_drift_coef * wind_u
                v += self.wind_drift_coef * wind_v
            
            return u, v
            
        except Exception as e:
            logger.debug(f"Interpolation error at ({lon}, {lat}): {str(e)}")
            return np.nan, np.nan
    
    def _velocity_to_displacement(
        self,
        u: float,
        v: float,
        lat: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Convert velocity to displacement in lon/lat coordinates.
        
        Parameters
        ----------
        u : float
            Eastward velocity (m/s)
        v : float
            Northward velocity (m/s)
        lat : float
            Current latitude
        dt : float
            Time step (seconds)
        
        Returns
        -------
        tuple
            (dlon, dlat) displacement in degrees
        """
        if np.isnan(u) or np.isnan(v):
            return np.nan, np.nan
        
        # Convert meters to degrees
        # dlat = v * dt / (R * pi/180)
        # dlon = u * dt / (R * cos(lat) * pi/180)
        
        meters_per_degree_lat = self.earth_radius * np.pi / 180.0
        meters_per_degree_lon = meters_per_degree_lat * np.cos(np.deg2rad(lat))
        
        dlat = (v * dt) / meters_per_degree_lat
        dlon = (u * dt) / meters_per_degree_lon
        
        return dlon, dlat
    
    def _euler_step(
        self,
        lon: float,
        lat: float,
        time_index: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Perform one Euler integration step.
        
        Parameters
        ----------
        lon : float
            Current longitude
        lat : float
            Current latitude
        time_index : int
            Current time index
        dt : float
            Time step (seconds)
        
        Returns
        -------
        tuple
            (new_lon, new_lat)
        """
        u, v = self._get_velocity_at_point(lon, lat, time_index)
        dlon, dlat = self._velocity_to_displacement(u, v, lat, dt)
        
        return lon + dlon, lat + dlat
    
    def _rk2_step(
        self,
        lon: float,
        lat: float,
        time_index: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Perform one RK2 (midpoint method) integration step.
        
        Parameters
        ----------
        lon : float
            Current longitude
        lat : float
            Current latitude
        time_index : int
            Current time index
        dt : float
            Time step (seconds)
        
        Returns
        -------
        tuple
            (new_lon, new_lat)
        """
        # First step (half time step)
        u1, v1 = self._get_velocity_at_point(lon, lat, time_index)
        dlon1, dlat1 = self._velocity_to_displacement(u1, v1, lat, dt / 2)
        
        lon_mid = lon + dlon1
        lat_mid = lat + dlat1
        
        # Second step (full time step using midpoint velocity)
        u2, v2 = self._get_velocity_at_point(lon_mid, lat_mid, time_index)
        dlon2, dlat2 = self._velocity_to_displacement(u2, v2, lat_mid, dt)
        
        return lon + dlon2, lat + dlat2
    
    def _rk4_step(
        self,
        lon: float,
        lat: float,
        time_index: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Perform one RK4 integration step.
        
        Parameters
        ----------
        lon : float
            Current longitude
        lat : float
            Current latitude
        time_index : int
            Current time index
        dt : float
            Time step (seconds)
        
        Returns
        -------
        tuple
            (new_lon, new_lat)
        """
        # k1
        u1, v1 = self._get_velocity_at_point(lon, lat, time_index)
        dlon1, dlat1 = self._velocity_to_displacement(u1, v1, lat, dt)
        
        # k2
        lon2 = lon + dlon1 / 2
        lat2 = lat + dlat1 / 2
        u2, v2 = self._get_velocity_at_point(lon2, lat2, time_index)
        dlon2, dlat2 = self._velocity_to_displacement(u2, v2, lat2, dt)
        
        # k3
        lon3 = lon + dlon2 / 2
        lat3 = lat + dlat2 / 2
        u3, v3 = self._get_velocity_at_point(lon3, lat3, time_index)
        dlon3, dlat3 = self._velocity_to_displacement(u3, v3, lat3, dt)
        
        # k4
        lon4 = lon + dlon3
        lat4 = lat + dlat3
        u4, v4 = self._get_velocity_at_point(lon4, lat4, time_index)
        dlon4, dlat4 = self._velocity_to_displacement(u4, v4, lat4, dt)
        
        # Combine
        dlon = (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4) / 6
        dlat = (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4) / 6
        
        return lon + dlon, lat + dlat
    
    def track_particles(
        self,
        initial_positions: np.ndarray,
        n_steps: int,
        dt: float = 3600.0,
        time_start_index: int = 0,
    ) -> np.ndarray:
        """
        Track particles over time.
        
        Parameters
        ----------
        initial_positions : np.ndarray
            Array of shape (n_particles, 2) containing (lon, lat) positions
        n_steps : int
            Number of time steps to simulate
        dt : float, optional
            Time step in seconds (default: 3600 = 1 hour)
        time_start_index : int, optional
            Starting time index in the data
        
        Returns
        -------
        np.ndarray
            Array of shape (n_particles, n_steps, 2) containing trajectories
        """
        n_particles = initial_positions.shape[0]
        trajectories = np.zeros((n_particles, n_steps, 2))
        
        # Set initial positions
        trajectories[:, 0, :] = initial_positions
        
        # Select integration method
        if self.method == "euler":
            step_func = self._euler_step
        elif self.method == "rk2":
            step_func = self._rk2_step
        else:  # rk4
            step_func = self._rk4_step
        
        # Integrate
        logger.info(f"Tracking {n_particles} particles for {n_steps} steps")
        
        for step in range(1, n_steps):
            # Determine time index (cycle through available times if necessary)
            if "time" in self.current_u.dims:
                n_times = len(self.current_u["time"])
                time_idx = (time_start_index + step) % n_times
            else:
                time_idx = 0
            
            for i in range(n_particles):
                lon, lat = trajectories[i, step - 1, :]
                
                # Skip if particle is already NaN (stopped)
                if np.isnan(lon) or np.isnan(lat):
                    trajectories[i, step, :] = [np.nan, np.nan]
                    continue
                
                # Perform integration step
                new_lon, new_lat = step_func(lon, lat, time_idx, dt)
                
                # Check if particle is still in bounds
                if (
                    np.isnan(new_lon)
                    or np.isnan(new_lat)
                    or new_lon < self.lons.min()
                    or new_lon > self.lons.max()
                    or new_lat < self.lats.min()
                    or new_lat > self.lats.max()
                ):
                    # Particle left domain
                    trajectories[i, step, :] = [np.nan, np.nan]
                else:
                    trajectories[i, step, :] = [new_lon, new_lat]
        
        logger.info("Particle tracking complete")
        return trajectories
    
    def create_particle_grid(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        n_lon: int = 10,
        n_lat: int = 10,
    ) -> np.ndarray:
        """
        Create a regular grid of particle initial positions.
        
        Parameters
        ----------
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        n_lon : int, optional
            Number of particles in longitude direction
        n_lat : int, optional
            Number of particles in latitude direction
        
        Returns
        -------
        np.ndarray
            Array of shape (n_lon * n_lat, 2) containing (lon, lat) positions
        """
        lons = np.linspace(lon_min, lon_max, n_lon)
        lats = np.linspace(lat_min, lat_max, n_lat)
        
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        positions = np.column_stack([lon_grid.flatten(), lat_grid.flatten()])
        
        logger.info(f"Created particle grid with {len(positions)} particles")
        return positions
    
    def create_random_particles(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        n_particles: int = 100,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create randomly distributed particle initial positions.
        
        Parameters
        ----------
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        n_particles : int, optional
            Number of particles
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        np.ndarray
            Array of shape (n_particles, 2) containing (lon, lat) positions
        """
        if seed is not None:
            np.random.seed(seed)
        
        lons = np.random.uniform(lon_min, lon_max, n_particles)
        lats = np.random.uniform(lat_min, lat_max, n_particles)
        
        positions = np.column_stack([lons, lats])
        
        logger.info(f"Created {n_particles} random particles")
        return positions
