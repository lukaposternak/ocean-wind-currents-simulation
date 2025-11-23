#!/usr/bin/env python3
"""
Example script demonstrating HYCOM data access.

This script shows how to:
1. Connect to HYCOM data server
2. Download ocean current data
3. Analyze the data
4. Create quiver plot visualizations
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hycom_client import HYCOMClient
from src.analysis.current_analysis import CurrentAnalyzer
from src.visualization.quiver_plots import QuiverPlotter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating HYCOM data access and analysis."""
    
    logger.info("=" * 80)
    logger.info("HYCOM Ocean Model Example")
    logger.info("=" * 80)
    
    # 1. Initialize HYCOM client
    logger.info("\n1. Initializing HYCOM client...")
    client = HYCOMClient(product="GLBv0.08/expt_93.0")
    
    # Show available products
    products = client.list_available_products()
    logger.info("   Available HYCOM products:")
    for product in products:
        logger.info(f"     - {product}")
    
    # Show current product info
    info = client.get_product_info()
    logger.info(f"\n   Using product: {info.get('name', 'Unknown')}")
    logger.info(f"   Resolution: {info.get('resolution', 'Unknown')}")
    logger.info(f"   Variables: {info.get('variables', [])}")
    
    # 2. Define region of interest (North Atlantic)
    logger.info("\n2. Defining region of interest (North Atlantic)...")
    lat_min, lat_max = 30.0, 45.0
    lon_min, lon_max = -75.0, -55.0
    start_date = "2024-01-01"
    end_date = "2024-01-03"
    
    logger.info(f"   Region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
    logger.info(f"   Period: {start_date} to {end_date}")
    
    # 3. Download ocean current data
    logger.info("\n3. Downloading ocean current data from HYCOM...")
    try:
        current_data = client.get_ocean_currents(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"   Data shape: {current_data.dims}")
        logger.info(f"   Variables: {list(current_data.data_vars)}")
        logger.info(f"   Time steps: {len(current_data.time)}")
    except Exception as e:
        logger.error(f"   Failed to download data: {e}")
        logger.info("   Using mock data for demonstration")
        current_data = client.get_ocean_currents(
            lat_min, lat_max, lon_min, lon_max, start_date, end_date
        )
    
    # 4. Download temperature data
    logger.info("\n4. Downloading temperature data...")
    try:
        temp_data = client.get_temperature(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"   Temperature data shape: {temp_data.dims}")
        mean_temp = temp_data["water_temp"].mean().values
        logger.info(f"   Mean SST: {mean_temp:.2f} °C")
    except Exception as e:
        logger.warning(f"   Could not download temperature: {e}")
    
    # 5. Analyze current data
    logger.info("\n5. Analyzing ocean current data...")
    analyzer = CurrentAnalyzer(current_data, u_var="water_u", v_var="water_v")
    
    # Compute magnitude and direction
    magnitude = analyzer.compute_magnitude(depth_level=0)
    direction = analyzer.compute_direction(depth_level=0)
    
    # Compute statistics
    stats = analyzer.compute_statistics(depth_level=0, dim="time")
    
    logger.info(f"   Mean current speed: {stats['mean'].mean().values:.3f} m/s")
    logger.info(f"   Std dev: {stats['std'].mean().values:.3f} m/s")
    logger.info(f"   Max current speed: {stats['max'].max().values:.3f} m/s")
    logger.info(f"   Min current speed: {stats['min'].min().values:.3f} m/s")
    
    # 6. Compute derived quantities
    logger.info("\n6. Computing derived quantities...")
    
    # Vorticity
    try:
        vorticity = analyzer.compute_vorticity(depth_level=0)
        logger.info(f"   Mean vorticity: {vorticity.mean().values:.2e} 1/s")
        logger.info(f"   Max vorticity: {vorticity.max().values:.2e} 1/s")
    except Exception as e:
        logger.warning(f"   Could not compute vorticity: {e}")
    
    # Divergence
    try:
        divergence = analyzer.compute_divergence(depth_level=0)
        logger.info(f"   Mean divergence: {divergence.mean().values:.2e} 1/s")
    except Exception as e:
        logger.warning(f"   Could not compute divergence: {e}")
    
    # Kinetic energy
    try:
        ke = analyzer.compute_kinetic_energy(depth_level=0)
        logger.info(f"   Mean kinetic energy: {ke.mean().values:.2f} J/m³")
    except Exception as e:
        logger.warning(f"   Could not compute kinetic energy: {e}")
    
    # 7. Create visualizations
    logger.info("\n7. Creating visualizations...")
    plotter = QuiverPlotter(figsize=(14, 10), dpi=150)
    
    # Create output directory
    output_dir = Path("output/hycom_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot surface currents at different time steps
    logger.info("   Creating current field plots...")
    for time_idx in [0, len(current_data.time) // 2]:
        try:
            fig, ax = plotter.plot_current_field(
                current_data["water_u"],
                current_data["water_v"],
                title=f"HYCOM Ocean Currents - North Atlantic (t={time_idx})",
                time_index=time_idx,
                depth_index=0,
                subsample=2,
                scale=30,
            )
            output_file = output_dir / f"current_field_t{time_idx}.png"
            plotter.save_figure(fig, str(output_file))
            logger.info(f"   Saved: {output_file}")
        except Exception as e:
            logger.error(f"   Failed to create plot for t={time_idx}: {e}")
    
    # Plot currents at different depths
    if "depth" in current_data["water_u"].dims and len(current_data.depth) > 1:
        logger.info("   Creating depth comparison plots...")
        try:
            fig, ax = plotter.plot_current_field(
                current_data["water_u"],
                current_data["water_v"],
                title="HYCOM Ocean Currents - Subsurface (5m depth)",
                time_index=0,
                depth_index=2,  # ~5m depth
                subsample=2,
                scale=30,
            )
            output_file = output_dir / "current_field_5m.png"
            plotter.save_figure(fig, str(output_file))
            logger.info(f"   Saved: {output_file}")
        except Exception as e:
            logger.error(f"   Failed to create depth plot: {e}")
    
    # 8. Temporal analysis
    logger.info("\n8. Performing temporal analysis...")
    
    # Compute time series of spatially-averaged current speed
    magnitude_mean = magnitude.mean(dim=["lat", "lon"])
    
    logger.info("   Current speed time series (spatially averaged):")
    for i, time in enumerate(current_data.time.values):
        logger.info(f"     {time}: {magnitude_mean.values[i]:.3f} m/s")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
