"""
Quiver plot visualization for wind and ocean currents.

This module provides functionality for creating static quiver (vector) plots
of wind and ocean current fields using matplotlib and cartopy.
"""

import logging
from typing import Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    logging.warning("cartopy not available. Some features will be limited.")


logger = logging.getLogger(__name__)


class QuiverPlotter:
    """
    Create quiver plots for wind and ocean current data.
    
    This class provides methods for creating professional quiver (vector)
    plots on geographic maps with customizable styling.
    
    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default: (12, 8)
    dpi : int, optional
        Figure resolution in dots per inch. Default: 300
    projection : str, optional
        Map projection to use. Options: 'PlateCarree', 'Mercator', 'LambertConformal'
        Default: 'PlateCarree'
    
    Attributes
    ----------
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    projection : cartopy CRS
        Map projection
    
    Examples
    --------
    >>> plotter = QuiverPlotter(figsize=(14, 10), dpi=300)
    >>> fig, ax = plotter.plot_wind_field(wind_u, wind_v, title="Wind Field")
    >>> plt.savefig("wind_field.png")
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (12, 8),
        dpi: int = 300,
        projection: str = "PlateCarree",
    ):
        """Initialize the quiver plotter."""
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up projection
        if CARTOPY_AVAILABLE:
            if projection == "PlateCarree":
                self.projection = ccrs.PlateCarree()
            elif projection == "Mercator":
                self.projection = ccrs.Mercator()
            elif projection == "LambertConformal":
                self.projection = ccrs.LambertConformal()
            else:
                logger.warning(f"Unknown projection '{projection}', using PlateCarree")
                self.projection = ccrs.PlateCarree()
        else:
            self.projection = None
            logger.warning("Cartopy not available, using basic plotting")
        
        logger.info(f"Quiver plotter initialized with projection: {projection}")
    
    def plot_wind_field(
        self,
        u: Union[xr.DataArray, np.ndarray],
        v: Union[xr.DataArray, np.ndarray],
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        title: str = "Wind Field",
        time_index: int = 0,
        scale: float = 50,
        subsample: int = 1,
        colormap: str = "viridis",
        add_magnitude_colorbar: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a quiver plot of wind field.
        
        Parameters
        ----------
        u : xr.DataArray or np.ndarray
            Eastward wind component
        v : xr.DataArray or np.ndarray
            Northward wind component
        lat : np.ndarray, optional
            Latitude coordinates (required if u, v are numpy arrays)
        lon : np.ndarray, optional
            Longitude coordinates (required if u, v are numpy arrays)
        title : str, optional
            Plot title
        time_index : int, optional
            Time index to plot if data is time-dependent
        scale : float, optional
            Arrow scale factor
        subsample : int, optional
            Subsample every Nth point for clearer visualization
        colormap : str, optional
            Colormap for magnitude
        add_magnitude_colorbar : bool, optional
            Whether to add colorbar showing magnitude
        ax : plt.Axes, optional
            Existing axes to plot on
        
        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        # Extract data from xarray if necessary
        if isinstance(u, xr.DataArray):
            # Extract time slice if present
            if "time" in u.dims:
                u_data = u.isel(time=time_index)
                v_data = v.isel(time=time_index)
            else:
                u_data = u
                v_data = v
            
            # Get coordinates
            lat_coord = "latitude" if "latitude" in u_data.dims else "lat"
            lon_coord = "longitude" if "longitude" in u_data.dims else "lon"
            lat = u_data[lat_coord].values
            lon = u_data[lon_coord].values
            
            u_data = u_data.values
            v_data = v_data.values
        else:
            u_data = u
            v_data = v
            if lat is None or lon is None:
                raise ValueError("lat and lon must be provided for numpy arrays")
        
        # Subsample
        u_sub = u_data[::subsample, ::subsample]
        v_sub = v_data[::subsample, ::subsample]
        lat_sub = lat[::subsample]
        lon_sub = lon[::subsample]
        
        # Compute magnitude
        magnitude = np.sqrt(u_sub**2 + v_sub**2)
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
        
        # Create figure and axes
        if ax is None:
            if CARTOPY_AVAILABLE and self.projection is not None:
                fig, ax = plt.subplots(
                    figsize=self.figsize,
                    dpi=self.dpi,
                    subplot_kw={"projection": self.projection},
                )
            else:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        # Plot quiver
        if CARTOPY_AVAILABLE and self.projection is not None:
            quiver = ax.quiver(
                lon_grid,
                lat_grid,
                u_sub,
                v_sub,
                magnitude,
                cmap=colormap,
                scale=scale,
                scale_units="inches",
                width=0.003,
                headwidth=3,
                headlength=4,
                headaxislength=3.5,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
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
            ax.set_extent([lon_sub.min(), lon_sub.max(), lat_sub.min(), lat_sub.max()])
        else:
            quiver = ax.quiver(
                lon_grid,
                lat_grid,
                u_sub,
                v_sub,
                magnitude,
                cmap=colormap,
                scale=scale,
                scale_units="inches",
                width=0.003,
                alpha=0.8,
            )
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.grid(True, alpha=0.3)
        
        # Add colorbar
        if add_magnitude_colorbar:
            cbar = plt.colorbar(quiver, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
            cbar.set_label("Wind Speed (m/s)", fontsize=10)
        
        # Add title
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Add quiver key (scale reference)
        ax.quiverkey(
            quiver, 0.9, 1.05, 5, "5 m/s", labelpos="E", coordinates="axes", fontproperties={"size": 9}
        )
        
        plt.tight_layout()
        logger.info(f"Created wind field quiver plot: {title}")
        
        return fig, ax
    
    def plot_current_field(
        self,
        u: Union[xr.DataArray, np.ndarray],
        v: Union[xr.DataArray, np.ndarray],
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        title: str = "Ocean Current Field",
        time_index: int = 0,
        depth_index: int = 0,
        scale: float = 25,
        subsample: int = 1,
        colormap: str = "plasma",
        add_magnitude_colorbar: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a quiver plot of ocean current field.
        
        Parameters
        ----------
        u : xr.DataArray or np.ndarray
            Eastward current component
        v : xr.DataArray or np.ndarray
            Northward current component
        lat : np.ndarray, optional
            Latitude coordinates
        lon : np.ndarray, optional
            Longitude coordinates
        title : str, optional
            Plot title
        time_index : int, optional
            Time index to plot
        depth_index : int, optional
            Depth index to plot
        scale : float, optional
            Arrow scale factor
        subsample : int, optional
            Subsample factor
        colormap : str, optional
            Colormap for magnitude
        add_magnitude_colorbar : bool, optional
            Whether to add colorbar
        ax : plt.Axes, optional
            Existing axes
        
        Returns
        -------
        tuple
            (fig, ax) - matplotlib figure and axes objects
        """
        # Extract data
        if isinstance(u, xr.DataArray):
            # Extract time and depth slices if present
            if "time" in u.dims:
                u_data = u.isel(time=time_index)
                v_data = v.isel(time=time_index)
            else:
                u_data = u
                v_data = v
            
            if "depth" in u_data.dims:
                u_data = u_data.isel(depth=depth_index)
                v_data = v_data.isel(depth=depth_index)
            
            # Get coordinates
            lat_coord = "latitude" if "latitude" in u_data.dims else "lat"
            lon_coord = "longitude" if "longitude" in u_data.dims else "lon"
            lat = u_data[lat_coord].values
            lon = u_data[lon_coord].values
            
            u_data = u_data.values
            v_data = v_data.values
        else:
            u_data = u
            v_data = v
        
        # Use the wind plotting function with different defaults
        return self.plot_wind_field(
            u_data,
            v_data,
            lat=lat,
            lon=lon,
            title=title,
            time_index=0,  # Already extracted
            scale=scale,
            subsample=subsample,
            colormap=colormap,
            add_magnitude_colorbar=add_magnitude_colorbar,
            ax=ax,
        )
    
    def plot_combined_wind_current(
        self,
        wind_u: xr.DataArray,
        wind_v: xr.DataArray,
        current_u: xr.DataArray,
        current_v: xr.DataArray,
        title: str = "Wind and Ocean Currents",
        time_index: int = 0,
        depth_index: int = 0,
        wind_scale: float = 50,
        current_scale: float = 25,
        subsample: int = 1,
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Create side-by-side plots of wind and ocean currents.
        
        Parameters
        ----------
        wind_u : xr.DataArray
            Eastward wind component
        wind_v : xr.DataArray
            Northward wind component
        current_u : xr.DataArray
            Eastward current component
        current_v : xr.DataArray
            Northward current component
        title : str, optional
            Overall plot title
        time_index : int, optional
            Time index to plot
        depth_index : int, optional
            Depth index for currents
        wind_scale : float, optional
            Wind arrow scale
        current_scale : float, optional
            Current arrow scale
        subsample : int, optional
            Subsample factor
        
        Returns
        -------
        tuple
            (fig, (ax1, ax2)) - figure and axes objects
        """
        if CARTOPY_AVAILABLE and self.projection is not None:
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(self.figsize[0] * 1.8, self.figsize[1]),
                dpi=self.dpi,
                subplot_kw={"projection": self.projection},
            )
        else:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]), dpi=self.dpi
            )
        
        # Plot wind
        self.plot_wind_field(
            wind_u,
            wind_v,
            title="Wind Field",
            time_index=time_index,
            scale=wind_scale,
            subsample=subsample,
            colormap="viridis",
            ax=ax1,
        )
        
        # Plot current
        self.plot_current_field(
            current_u,
            current_v,
            title="Ocean Currents",
            time_index=time_index,
            depth_index=depth_index,
            scale=current_scale,
            subsample=subsample,
            colormap="plasma",
            ax=ax2,
        )
        
        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        
        plt.tight_layout()
        logger.info(f"Created combined wind-current plot: {title}")
        
        return fig, (ax1, ax2)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = "png",
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
    ):
        """
        Save figure to file.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        filename : str
            Output filename
        format : str, optional
            File format (png, pdf, svg, etc.)
        dpi : int, optional
            Resolution. If None, uses figure dpi.
        bbox_inches : str, optional
            Bounding box setting
        """
        if dpi is None:
            dpi = self.dpi
        
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved figure to: {filename}")
