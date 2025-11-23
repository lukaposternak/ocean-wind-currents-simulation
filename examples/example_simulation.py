#!/usr/bin/env python3
"""
Example script demonstrating particle tracking simulation.

This script shows how to:
1. Load wind and current data
2. Initialize particle tracker
3. Run Lagrangian simulation
4. Visualize particle trajectories
5. Create animations
"""

import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.copernicus_client import CopernicusClient
from src.data.hycom_client import HYCOMClient
from src.simulation.particle_tracker import ParticleTracker
from src.visualization.quiver_plots import QuiverPlotter
from src.visualization.animations import FieldAnimator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating particle tracking simulation."""
    
    logger.info("=" * 80)
    logger.info("Particle Tracking Simulation Example")
    logger.info("=" * 80)
    
    # 1. Load data
    logger.info("\n1. Loading ocean current and wind data...")
    
    # Define region (Caribbean Sea)
    lat_min, lat_max = 15.0, 25.0
    lon_min, lon_max = -85.0, -75.0
    start_date = "2024-01-01"
    end_date = "2024-01-05"
    
    logger.info(f"   Region: lat[{lat_min}, {lat_max}], lon[{lon_min}, {lon_max}]")
    logger.info(f"   Period: {start_date} to {end_date}")
    
    # Load current data (using HYCOM client for this example)
    hycom_client = HYCOMClient()
    current_data = hycom_client.get_ocean_currents(
        lat_min, lat_max, lon_min, lon_max, start_date, end_date
    )
    logger.info(f"   Current data loaded: {current_data.dims}")
    
    # Load wind data (using Copernicus client)
    copernicus_client = CopernicusClient()
    wind_data = copernicus_client.get_wind_data(
        lat_min, lat_max, lon_min, lon_max, start_date, end_date
    )
    logger.info(f"   Wind data loaded: {wind_data.dims}")
    
    # 2. Initialize particle tracker
    logger.info("\n2. Initializing particle tracker...")
    
    # Test different integration methods
    methods = ["euler", "rk2", "rk4"]
    
    for method in methods:
        logger.info(f"\n   Testing {method.upper()} method...")
        
        tracker = ParticleTracker(
            current_u=current_data["water_u"],
            current_v=current_data["water_v"],
            wind_u=wind_data["eastward_wind"],
            wind_v=wind_data["northward_wind"],
            wind_drift_coefficient=0.03,
            integration_method=method,
        )
        logger.info(f"   Tracker initialized with {method} integration")
        
        # 3. Create initial particle positions
        logger.info(f"\n3. Creating initial particle positions ({method})...")
        
        # Option 1: Grid of particles
        initial_grid = tracker.create_particle_grid(
            lon_min=-82.0,
            lon_max=-78.0,
            lat_min=18.0,
            lat_max=22.0,
            n_lon=5,
            n_lat=5,
        )
        logger.info(f"   Created grid with {len(initial_grid)} particles")
        
        # Option 2: Random particles
        initial_random = tracker.create_random_particles(
            lon_min=-82.0,
            lon_max=-78.0,
            lat_min=18.0,
            lat_max=22.0,
            n_particles=20,
            seed=42,
        )
        logger.info(f"   Created {len(initial_random)} random particles")
        
        # Use grid for this example
        initial_positions = initial_grid
        
        # 4. Run simulation
        logger.info(f"\n4. Running particle tracking simulation ({method})...")
        
        n_steps = 50  # Number of time steps
        dt = 3600 * 3  # Time step: 3 hours in seconds
        
        logger.info(f"   Number of steps: {n_steps}")
        logger.info(f"   Time step: {dt/3600:.1f} hours")
        logger.info(f"   Total simulation time: {n_steps * dt / 3600:.1f} hours")
        
        trajectories = tracker.track_particles(
            initial_positions=initial_positions,
            n_steps=n_steps,
            dt=dt,
        )
        
        logger.info(f"   Simulation complete!")
        logger.info(f"   Trajectories shape: {trajectories.shape}")
        
        # Count active particles at end
        final_positions = trajectories[:, -1, :]
        n_active = np.sum(~np.isnan(final_positions[:, 0]))
        logger.info(f"   Active particles at end: {n_active}/{len(initial_positions)}")
        
        # 5. Analyze trajectories
        logger.info(f"\n5. Analyzing trajectories ({method})...")
        
        # Compute total distance traveled
        distances = np.zeros(len(initial_positions))
        for i in range(len(initial_positions)):
            traj = trajectories[i]
            # Remove NaN values
            valid = ~np.isnan(traj[:, 0])
            if valid.sum() > 1:
                traj_valid = traj[valid]
                # Approximate distance (simple lat/lon difference)
                diffs = np.diff(traj_valid, axis=0)
                distances[i] = np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2))
        
        mean_distance = np.mean(distances[distances > 0])
        max_distance = np.max(distances)
        
        logger.info(f"   Mean distance traveled: {mean_distance:.3f} degrees")
        logger.info(f"   Max distance traveled: {max_distance:.3f} degrees")
        
        # 6. Create visualizations
        logger.info(f"\n6. Creating visualizations ({method})...")
        
        # Create output directory
        output_dir = Path(f"output/simulation_example/{method}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot initial and final positions
        logger.info("   Creating trajectory plot...")
        try:
            fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
            
            # Plot all trajectories
            for i in range(len(initial_positions)):
                traj = trajectories[i]
                valid = ~np.isnan(traj[:, 0])
                if valid.sum() > 1:
                    ax.plot(traj[valid, 0], traj[valid, 1], 'b-', alpha=0.3, linewidth=0.5)
            
            # Plot initial positions
            ax.scatter(
                initial_positions[:, 0],
                initial_positions[:, 1],
                c='green',
                s=50,
                marker='o',
                label='Initial',
                zorder=5,
            )
            
            # Plot final positions
            final_valid = ~np.isnan(final_positions[:, 0])
            if final_valid.sum() > 0:
                ax.scatter(
                    final_positions[final_valid, 0],
                    final_positions[final_valid, 1],
                    c='red',
                    s=50,
                    marker='x',
                    label='Final',
                    zorder=5,
                )
            
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(f"Particle Trajectories ({method.upper()} integration)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([lon_min, lon_max])
            ax.set_ylim([lat_min, lat_max])
            
            output_file = output_dir / "trajectories.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"   Saved: {output_file}")
        except Exception as e:
            logger.error(f"   Failed to create trajectory plot: {e}")
        
        # 7. Create animation (only for RK4 to save time)
        if method == "rk4":
            logger.info("\n7. Creating particle animation...")
            try:
                animator = FieldAnimator(figsize=(12, 10), dpi=100)
                
                anim = animator.animate_particle_trajectories(
                    trajectories=trajectories,
                    lat_bounds=(lat_min, lat_max),
                    lon_bounds=(lon_min, lon_max),
                    title="Particle Drift Simulation",
                    interval=100,
                    trail_length=10,
                )
                
                output_file = output_dir / "particle_animation.gif"
                logger.info(f"   Saving animation to: {output_file}")
                animator.save_animation(anim, str(output_file), fps=10)
                logger.info(f"   Animation saved!")
            except Exception as e:
                logger.error(f"   Failed to create animation: {e}")
                logger.info("   This may be due to missing FFmpeg or Pillow")
        
        # 8. Create combined visualization with current field
        logger.info(f"\n8. Creating combined current-trajectory plot ({method})...")
        try:
            plotter = QuiverPlotter(figsize=(14, 10), dpi=150)
            
            fig, ax = plotter.plot_current_field(
                current_data["water_u"],
                current_data["water_v"],
                title=f"Ocean Currents with Particle Trajectories ({method.upper()})",
                time_index=0,
                depth_index=0,
                subsample=3,
                scale=25,
            )
            
            # Overlay trajectories
            for i in range(len(initial_positions)):
                traj = trajectories[i]
                valid = ~np.isnan(traj[:, 0])
                if valid.sum() > 1:
                    ax.plot(traj[valid, 0], traj[valid, 1], 'r-', alpha=0.5, linewidth=1)
            
            # Mark initial positions
            ax.scatter(
                initial_positions[:, 0],
                initial_positions[:, 1],
                c='lime',
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=1,
                label='Initial positions',
                zorder=10,
            )
            
            ax.legend(loc='upper right')
            
            output_file = output_dir / "combined_currents_trajectories.png"
            plotter.save_figure(fig, str(output_file))
            logger.info(f"   Saved: {output_file}")
        except Exception as e:
            logger.error(f"   Failed to create combined plot: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Simulation example completed successfully!")
    logger.info(f"Output files saved to: output/simulation_example/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
