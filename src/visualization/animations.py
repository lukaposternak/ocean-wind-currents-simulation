"""
Animation tools for wind and ocean current time series.

This module provides functionality for creating animations of wind and ocean
current fields over time using matplotlib.animation.
"""

import logging
from typing import Optional, Tuple, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


logger = logging.getLogger(__name__)


class FieldAnimator:
    """
    Create animations of wind and ocean current fields.
    
    This class provides methods for creating time-series animations
    of vector fields with customizable styling and output formats.
    
    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default: (12, 8)
    dpi : int, optional
        Figure resolution. Default: 100 (lower for faster animation)
    projection : str, optional
        Map projection. Default: 'PlateCarree'
    
    Attributes
    ----------
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    projection : cartopy CRS or None
        Map projection
    
    Examples
    --------
    >>> animator = FieldAnimator(figsize=(12, 8), dpi=100)
    >>> anim = animator.animate_wind_field(wind_u, wind_v, interval=100)
    >>> animator.save_animation(anim, "wind_animation.mp4", fps=10)
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 100,
        projection: str = "PlateCarree",
    ):
        """Initialize the field animator."""
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up projection
        if CARTOPY_AVAILABLE:
            if projection == "PlateCarree":
                self.projection = ccrs.PlateCarree()
            elif projection == "Mercator":
                self.projection = ccrs.Mercator()
            else:
                self.projection = ccrs.PlateCarree()
        else:
            self.projection = None
        
        logger.info(f"Field animator initialized with projection: {projection}")
    
    def animate_wind_field(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        title: str = "Wind Field Animation",
        scale: float = 50,
        subsample: int = 2,
        colormap: str = "viridis",
        interval: int = 100,
        blit: bool = False,
    ) -> animation.FuncAnimation:
        """
        Create animation of wind field over time.
        
        Parameters
        ----------
        u : xr.DataArray
            Eastward wind component (must have time dimension)
        v : xr.DataArray
            Northward wind component (must have time dimension)
        title : str, optional
            Animation title
        scale : float, optional
            Arrow scale factor
        subsample : int, optional
            Spatial subsampling factor
        colormap : str, optional
            Colormap for magnitude
        interval : int, optional
            Delay between frames in milliseconds
        blit : bool, optional
            Use blitting for faster rendering
        
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object
        """
        if "time" not in u.dims:
            raise ValueError("Data must have time dimension for animation")
        
        # Get coordinates
        lat_coord = "latitude" if "latitude" in u.dims else "lat"
        lon_coord = "longitude" if "longitude" in u.dims else "lon"
        
        lat = u[lat_coord].values[::subsample]
        lon = u[lon_coord].values[::subsample]
        times = u["time"].values
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Set up figure
        if CARTOPY_AVAILABLE and self.projection is not None:
            fig, ax = plt.subplots(
                figsize=self.figsize,
                dpi=self.dpi,
                subplot_kw={"projection": self.projection},
            )
            
            # Add map features
            ax.coastlines(resolution="50m", linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.1)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            
            # Set extent
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
            
            transform = ccrs.PlateCarree()
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.grid(True, alpha=0.3)
            transform = None
        
        # Initial empty quiver plot
        u_init = u.isel(time=0)[::subsample, ::subsample].values
        v_init = v.isel(time=0)[::subsample, ::subsample].values
        magnitude_init = np.sqrt(u_init**2 + v_init**2)
        
        if transform is not None:
            quiver = ax.quiver(
                lon_grid,
                lat_grid,
                u_init,
                v_init,
                magnitude_init,
                cmap=colormap,
                scale=scale,
                scale_units="inches",
                width=0.003,
                alpha=0.8,
                transform=transform,
            )
        else:
            quiver = ax.quiver(
                lon_grid,
                lat_grid,
                u_init,
                v_init,
                magnitude_init,
                cmap=colormap,
                scale=scale,
                scale_units="inches",
                width=0.003,
                alpha=0.8,
            )
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
        cbar.set_label("Wind Speed (m/s)", fontsize=10)
        
        # Add quiver key
        ax.quiverkey(
            quiver, 0.9, 1.05, 5, "5 m/s", labelpos="E", coordinates="axes"
        )
        
        # Title with time
        title_text = ax.set_title(
            f"{title}\nTime: {times[0]}",
            fontsize=12,
            fontweight="bold",
        )
        
        def update(frame):
            """Update function for animation."""
            # Get data for this frame
            u_frame = u.isel(time=frame)[::subsample, ::subsample].values
            v_frame = v.isel(time=frame)[::subsample, ::subsample].values
            magnitude = np.sqrt(u_frame**2 + v_frame**2)
            
            # Update quiver
            quiver.set_UVC(u_frame, v_frame, magnitude)
            
            # Update title with current time
            title_text.set_text(f"{title}\nTime: {times[frame]}")
            
            return quiver, title_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=interval,
            blit=False,  # blit doesn't work well with cartopy
        )
        
        logger.info(f"Created wind field animation with {len(times)} frames")
        return anim
    
    def animate_current_field(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        title: str = "Ocean Current Animation",
        depth_index: int = 0,
        scale: float = 25,
        subsample: int = 2,
        colormap: str = "plasma",
        interval: int = 100,
    ) -> animation.FuncAnimation:
        """
        Create animation of ocean current field over time.
        
        Parameters
        ----------
        u : xr.DataArray
            Eastward current component
        v : xr.DataArray
            Northward current component
        title : str, optional
            Animation title
        depth_index : int, optional
            Depth level to animate
        scale : float, optional
            Arrow scale factor
        subsample : int, optional
            Spatial subsampling
        colormap : str, optional
            Colormap
        interval : int, optional
            Frame interval in ms
        
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object
        """
        # Extract depth level if present
        if "depth" in u.dims:
            u = u.isel(depth=depth_index)
            v = v.isel(depth=depth_index)
        
        # Use wind animation function
        return self.animate_wind_field(
            u,
            v,
            title=title,
            scale=scale,
            subsample=subsample,
            colormap=colormap,
            interval=interval,
        )
    
    def animate_particle_trajectories(
        self,
        trajectories: np.ndarray,
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        title: str = "Particle Trajectories",
        interval: int = 50,
        trail_length: int = 20,
    ) -> animation.FuncAnimation:
        """
        Animate particle trajectories.
        
        Parameters
        ----------
        trajectories : np.ndarray
            Array of shape (n_particles, n_steps, 2) containing (lon, lat) positions
        lat_bounds : tuple
            (min_lat, max_lat)
        lon_bounds : tuple
            (min_lon, max_lon)
        title : str, optional
            Animation title
        interval : int, optional
            Frame interval in ms
        trail_length : int, optional
            Number of previous positions to show as trail
        
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object
        """
        n_particles, n_steps, _ = trajectories.shape
        
        # Set up figure
        if CARTOPY_AVAILABLE and self.projection is not None:
            fig, ax = plt.subplots(
                figsize=self.figsize,
                dpi=self.dpi,
                subplot_kw={"projection": self.projection},
            )
            
            ax.coastlines(resolution="50m", linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
            ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]])
            
            transform = ccrs.PlateCarree()
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.set_xlim(lon_bounds)
            ax.set_ylim(lat_bounds)
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.grid(True, alpha=0.3)
            transform = None
        
        # Initialize particle positions
        if transform is not None:
            particles = ax.scatter(
                [], [], c="red", s=20, alpha=0.8, transform=transform, zorder=5
            )
            trails = [ax.plot([], [], "r-", alpha=0.3, linewidth=0.5, transform=transform)[0]
                     for _ in range(n_particles)]
        else:
            particles = ax.scatter([], [], c="red", s=20, alpha=0.8, zorder=5)
            trails = [ax.plot([], [], "r-", alpha=0.3, linewidth=0.5)[0]
                     for _ in range(n_particles)]
        
        title_text = ax.set_title(f"{title}\nStep: 0/{n_steps}", fontsize=12, fontweight="bold")
        
        def update(frame):
            """Update function for animation."""
            # Current positions
            current_pos = trajectories[:, frame, :]
            
            # Remove NaN particles
            valid = ~np.isnan(current_pos[:, 0])
            
            if transform is not None:
                particles.set_offsets(current_pos[valid])
            else:
                particles.set_offsets(current_pos[valid])
            
            # Update trails
            for i in range(n_particles):
                if valid[i]:
                    start_idx = max(0, frame - trail_length)
                    trail_pos = trajectories[i, start_idx:frame+1, :]
                    # Remove NaN from trail
                    trail_valid = ~np.isnan(trail_pos[:, 0])
                    trails[i].set_data(trail_pos[trail_valid, 0], trail_pos[trail_valid, 1])
                else:
                    trails[i].set_data([], [])
            
            title_text.set_text(f"{title}\nStep: {frame}/{n_steps}")
            
            return particles, *trails, title_text
        
        anim = animation.FuncAnimation(
            fig, update, frames=n_steps, interval=interval, blit=False
        )
        
        logger.info(f"Created particle trajectory animation with {n_steps} steps")
        return anim
    
    def save_animation(
        self,
        anim: animation.FuncAnimation,
        filename: str,
        fps: int = 10,
        bitrate: int = 1800,
        codec: str = "h264",
        progress_callback: Optional[Callable] = None,
    ):
        """
        Save animation to file.
        
        Parameters
        ----------
        anim : matplotlib.animation.FuncAnimation
            Animation to save
        filename : str
            Output filename (extension determines format: .mp4, .gif, etc.)
        fps : int, optional
            Frames per second
        bitrate : int, optional
            Video bitrate (for video formats)
        codec : str, optional
            Video codec (for video formats)
        progress_callback : callable, optional
            Callback function for progress updates
        """
        # Determine writer based on file extension
        if filename.endswith(".mp4"):
            writer = animation.FFMpegWriter(
                fps=fps, bitrate=bitrate, codec=codec
            )
        elif filename.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            # Default to mp4
            writer = animation.FFMpegWriter(
                fps=fps, bitrate=bitrate, codec=codec
            )
        
        try:
            logger.info(f"Saving animation to: {filename}")
            anim.save(filename, writer=writer, progress_callback=progress_callback)
            logger.info(f"Animation saved successfully to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save animation: {str(e)}")
            logger.info("Trying alternative method with PillowWriter...")
            try:
                # Fallback to GIF with Pillow
                gif_filename = filename.rsplit(".", 1)[0] + ".gif"
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_filename, writer=writer)
                logger.info(f"Animation saved as GIF: {gif_filename}")
            except Exception as e2:
                logger.error(f"Failed to save with fallback method: {str(e2)}")
                raise
