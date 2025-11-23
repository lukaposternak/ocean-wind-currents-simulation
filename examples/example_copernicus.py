#!/usr/bin/env python3
"""
Example script demonstrating Copernicus Marine Service data access.

This script shows how to:
1. Connect to Copernicus Marine Service
2. Download wind and ocean current data
3. Perform basic analysis
4. Create visualizations
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.copernicus_client import CopernicusClient
from src.analysis.wind_analysis import WindAnalyzer
from src.analysis.current_analysis import CurrentAnalyzer
from src.visualization.quiver_plots import QuiverPlotter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating Copernicus data access and analysis."""
    
    logger.info("=" * 80)
    logger.info("Copernicus Marine Service Example")
    logger.info("=" * 80)
    
    # 1. Initialize Copernicus client
    logger.info("\n1. Initializing Copernicus client...")
    client = CopernicusClient()
    # Note: In production, provide credentials:
    # client = CopernicusClient(username="your_username", password="your_password")
    
    # 2. Define region of interest (Gulf of Mexico)
    logger.info("\n2. Defining region of interest (Gulf of Mexico)...")
    lat_min, lat_max = 20.0, 30.0
    lon_min, lon_max = -95.0, -85.0
    start_date = "2024-01-01"
    end_date = "2024-01-03"
    
    logger.info(f"   Region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
    logger.info(f"   Period: {start_date} to {end_date}")
    
    # 3. Download wind data
    logger.info("\n3. Downloading wind data...")
    try:
        wind_data = client.get_wind_data(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"   Wind data shape: {wind_data.dims}")
        logger.info(f"   Variables: {list(wind_data.data_vars)}")
    except Exception as e:
        logger.error(f"   Failed to download wind data: {e}")
        logger.info("   Using mock data for demonstration")
        wind_data = client.get_wind_data(
            lat_min, lat_max, lon_min, lon_max, start_date, end_date
        )
    
    # 4. Download ocean current data
    logger.info("\n4. Downloading ocean current data...")
    try:
        current_data = client.get_ocean_currents(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"   Current data shape: {current_data.dims}")
        logger.info(f"   Variables: {list(current_data.data_vars)}")
    except Exception as e:
        logger.error(f"   Failed to download current data: {e}")
        logger.info("   Using mock data for demonstration")
        current_data = client.get_ocean_currents(
            lat_min, lat_max, lon_min, lon_max, start_date, end_date
        )
    
    # 5. Analyze wind data
    logger.info("\n5. Analyzing wind data...")
    wind_analyzer = WindAnalyzer(wind_data)
    
    wind_magnitude = wind_analyzer.compute_magnitude()
    wind_direction = wind_analyzer.compute_direction()
    wind_stats = wind_analyzer.compute_statistics(dim="time")
    
    logger.info(f"   Mean wind speed: {wind_stats['mean'].mean().values:.2f} m/s")
    logger.info(f"   Max wind speed: {wind_stats['max'].max().values:.2f} m/s")
    logger.info(f"   Min wind speed: {wind_stats['min'].min().values:.2f} m/s")
    
    # 6. Analyze current data
    logger.info("\n6. Analyzing ocean current data...")
    current_analyzer = CurrentAnalyzer(current_data)
    
    current_magnitude = current_analyzer.compute_magnitude()
    current_direction = current_analyzer.compute_direction()
    current_stats = current_analyzer.compute_statistics(dim="time")
    
    logger.info(f"   Mean current speed: {current_stats['mean'].mean().values:.2f} m/s")
    logger.info(f"   Max current speed: {current_stats['max'].max().values:.2f} m/s")
    
    # 7. Compute derived quantities
    logger.info("\n7. Computing derived quantities...")
    
    # Wind stress
    tau_x, tau_y = wind_analyzer.compute_wind_stress()
    logger.info(f"   Mean wind stress magnitude: {(tau_x**2 + tau_y**2)**0.5.mean().values:.4f} N/m²")
    
    # Current vorticity
    try:
        vorticity = current_analyzer.compute_vorticity()
        logger.info(f"   Mean vorticity: {vorticity.mean().values:.2e} 1/s")
    except Exception as e:
        logger.warning(f"   Could not compute vorticity: {e}")
    
    # 8. Create visualizations
    logger.info("\n8. Creating visualizations...")
    plotter = QuiverPlotter(figsize=(14, 10), dpi=150)
    
    # Create output directory
    output_dir = Path("output/copernicus_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot wind field
    logger.info("   Creating wind field plot...")
    try:
        fig, ax = plotter.plot_wind_field(
            wind_data["eastward_wind"],
            wind_data["northward_wind"],
            title="Wind Field - Gulf of Mexico",
            time_index=0,
            subsample=2,
        )
        output_file = output_dir / "wind_field.png"
        plotter.save_figure(fig, str(output_file))
        logger.info(f"   Saved: {output_file}")
    except Exception as e:
        logger.error(f"   Failed to create wind plot: {e}")
    
    # Plot current field
    logger.info("   Creating current field plot...")
    try:
        fig, ax = plotter.plot_current_field(
            current_data["uo"],
            current_data["vo"],
            title="Ocean Currents - Gulf of Mexico",
            time_index=0,
            depth_index=0,
            subsample=2,
        )
        output_file = output_dir / "current_field.png"
        plotter.save_figure(fig, str(output_file))
        logger.info(f"   Saved: {output_file}")
    except Exception as e:
        logger.error(f"   Failed to create current plot: {e}")
    
    # Combined plot
    logger.info("   Creating combined wind-current plot...")
    try:
        fig, axes = plotter.plot_combined_wind_current(
            wind_data["eastward_wind"],
            wind_data["northward_wind"],
            current_data["uo"],
            current_data["vo"],
            title="Wind and Ocean Currents - Gulf of Mexico",
            subsample=2,
        )
        output_file = output_dir / "combined_field.png"
        plotter.save_figure(fig, str(output_file))
        logger.info(f"   Saved: {output_file}")
    except Exception as e:
        logger.error(f"   Failed to create combined plot: {e}")
    
    # 9. Compute wind-current correlation
    logger.info("\n9. Computing wind-current correlation...")
    try:
        correlations = wind_analyzer.correlate_with_current(current_data)
        if correlations["magnitude_correlation"] is not None:
            mag_corr = correlations["magnitude_correlation"].mean().values
            logger.info(f"   Mean magnitude correlation: {mag_corr:.3f}")
    except Exception as e:
        logger.warning(f"   Could not compute correlation: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
