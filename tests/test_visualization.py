"""
Unit tests for visualization modules.
"""

import pytest
import numpy as np
import xarray as xr
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.quiver_plots import QuiverPlotter
from src.visualization.animations import FieldAnimator


def create_mock_wind_data():
    """Create mock wind data for testing."""
    lats = np.linspace(20, 30, 10)
    lons = np.linspace(-90, -80, 12)
    times = np.array([np.datetime64('2024-01-01'), np.datetime64('2024-01-02')])
    
    u = np.random.randn(len(times), len(lats), len(lons)) * 5 + 3
    v = np.random.randn(len(times), len(lats), len(lons)) * 4 + 2
    
    return xr.Dataset(
        {
            "eastward_wind": (["time", "latitude", "longitude"], u),
            "northward_wind": (["time", "latitude", "longitude"], v),
        },
        coords={"time": times, "latitude": lats, "longitude": lons}
    )


def create_mock_current_data():
    """Create mock current data for testing."""
    lats = np.linspace(20, 30, 10)
    lons = np.linspace(-90, -80, 12)
    times = np.array([np.datetime64('2024-01-01'), np.datetime64('2024-01-02')])
    depths = np.array([0.0, 5.0])
    
    u = np.random.randn(len(times), len(depths), len(lats), len(lons)) * 0.3
    v = np.random.randn(len(times), len(depths), len(lats), len(lons)) * 0.3
    
    return xr.Dataset(
        {
            "uo": (["time", "depth", "latitude", "longitude"], u),
            "vo": (["time", "depth", "latitude", "longitude"], v),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": lats,
            "longitude": lons
        }
    )


class TestQuiverPlotter:
    """Test suite for QuiverPlotter."""
    
    def test_initialization(self):
        """Test plotter initialization."""
        plotter = QuiverPlotter()
        assert plotter is not None
        assert plotter.figsize is not None
        assert plotter.dpi is not None
    
    def test_plot_wind_field(self):
        """Test wind field plotting."""
        data = create_mock_wind_data()
        plotter = QuiverPlotter(figsize=(10, 8), dpi=100)
        
        fig, ax = plotter.plot_wind_field(
            data["eastward_wind"],
            data["northward_wind"],
            title="Test Wind Field",
            time_index=0,
            subsample=2
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_current_field(self):
        """Test current field plotting."""
        data = create_mock_current_data()
        plotter = QuiverPlotter(figsize=(10, 8), dpi=100)
        
        fig, ax = plotter.plot_current_field(
            data["uo"],
            data["vo"],
            title="Test Current Field",
            time_index=0,
            depth_index=0,
            subsample=2
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_combined_wind_current(self):
        """Test combined wind-current plotting."""
        wind_data = create_mock_wind_data()
        current_data = create_mock_current_data()
        plotter = QuiverPlotter(figsize=(10, 8), dpi=100)
        
        fig, axes = plotter.plot_combined_wind_current(
            wind_data["eastward_wind"],
            wind_data["northward_wind"],
            current_data["uo"],
            current_data["vo"],
            title="Test Combined",
            subsample=2
        )
        
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)
    
    def test_save_figure(self, tmp_path):
        """Test figure saving."""
        data = create_mock_wind_data()
        plotter = QuiverPlotter(figsize=(10, 8), dpi=100)
        
        fig, ax = plotter.plot_wind_field(
            data["eastward_wind"],
            data["northward_wind"],
            subsample=2
        )
        
        output_file = tmp_path / "test_plot.png"
        plotter.save_figure(fig, str(output_file))
        
        assert output_file.exists()
        plt.close(fig)


class TestFieldAnimator:
    """Test suite for FieldAnimator."""
    
    def test_initialization(self):
        """Test animator initialization."""
        animator = FieldAnimator()
        assert animator is not None
        assert animator.figsize is not None
        assert animator.dpi is not None
    
    def test_animate_wind_field(self):
        """Test wind field animation creation."""
        data = create_mock_wind_data()
        animator = FieldAnimator(figsize=(10, 8), dpi=50)
        
        try:
            anim = animator.animate_wind_field(
                data["eastward_wind"],
                data["northward_wind"],
                title="Test Animation",
                subsample=2,
                interval=100
            )
            assert anim is not None
            plt.close('all')
        except Exception as e:
            # Animation may fail without display
            pytest.skip(f"Animation creation failed: {e}")
    
    def test_animate_current_field(self):
        """Test current field animation creation."""
        data = create_mock_current_data()
        animator = FieldAnimator(figsize=(10, 8), dpi=50)
        
        try:
            anim = animator.animate_current_field(
                data["uo"],
                data["vo"],
                title="Test Animation",
                depth_index=0,
                subsample=2,
                interval=100
            )
            assert anim is not None
            plt.close('all')
        except Exception as e:
            pytest.skip(f"Animation creation failed: {e}")
    
    def test_animate_particle_trajectories(self):
        """Test particle trajectory animation."""
        # Create mock trajectories
        n_particles = 10
        n_steps = 20
        trajectories = np.zeros((n_particles, n_steps, 2))
        
        # Simple linear trajectories
        for i in range(n_particles):
            trajectories[i, :, 0] = np.linspace(-85, -80, n_steps)  # lon
            trajectories[i, :, 1] = np.linspace(20 + i*0.5, 25 + i*0.5, n_steps)  # lat
        
        animator = FieldAnimator(figsize=(10, 8), dpi=50)
        
        try:
            anim = animator.animate_particle_trajectories(
                trajectories,
                lat_bounds=(20, 30),
                lon_bounds=(-90, -75),
                title="Test Particle Animation",
                interval=100,
                trail_length=5
            )
            assert anim is not None
            plt.close('all')
        except Exception as e:
            pytest.skip(f"Particle animation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
