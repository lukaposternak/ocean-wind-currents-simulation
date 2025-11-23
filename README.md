# Ocean-Wind-Currents Simulation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for simulating, analyzing, and visualizing wind and superficial ocean currents. This package provides tools for accessing oceanographic data, performing analysis, creating visualizations, and running Lagrangian particle tracking simulations.

## Features

- **Data Acquisition**: Access data from Copernicus Marine Service and HYCOM ocean models
- **Analysis Tools**: Compute wind/current statistics, vorticity, divergence, and correlations
- **Visualization**: Create professional quiver plots and animations with cartopy
- **Particle Tracking**: Lagrangian particle simulation with multiple integration schemes (Euler, RK2, RK4)
- **Flexible Configuration**: YAML-based configuration system
- **Production Ready**: Comprehensive error handling, logging, and documentation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/lukaposternak/ocean-wind-currents-simulation.git
cd ocean-wind-currents-simulation

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Optional Dependencies

For full functionality, you may need additional system packages:

```bash
# Ubuntu/Debian
sudo apt-get install libgeos-dev libproj-dev

# macOS
brew install geos proj
```

## Quick Start

```python
from src.data.copernicus_client import CopernicusClient
from src.analysis.wind_analysis import WindAnalyzer
from src.visualization.quiver_plots import QuiverPlotter

# Initialize client
client = CopernicusClient()

# Download data
wind_data = client.get_wind_data(
    lat_min=20, lat_max=30,
    lon_min=-95, lon_max=-85,
    start_date="2024-01-01",
    end_date="2024-01-02"
)

# Analyze
analyzer = WindAnalyzer(wind_data)
magnitude = analyzer.compute_magnitude()
stats = analyzer.compute_statistics()

# Visualize
plotter = QuiverPlotter()
fig, ax = plotter.plot_wind_field(
    wind_data["eastward_wind"],
    wind_data["northward_wind"],
    title="Wind Field - Gulf of Mexico"
)
plotter.save_figure(fig, "wind_field.png")
```

## Data Sources

### Copernicus Marine Service

The Copernicus Marine Environment Monitoring Service (CMEMS) provides global ocean data.

**Registration**: https://marine.copernicus.eu/

**Setup**:
```python
from src.data.copernicus_client import CopernicusClient

client = CopernicusClient(
    username="your_username",
    password="your_password"
)

# Or use environment variables
# export COPERNICUS_USERNAME="your_username"
# export COPERNICUS_PASSWORD="your_password"
```

**Available Products**:
- `GLOBAL_ANALYSISFORECAST_PHY_001_024`: Global ocean physics analysis and forecast

### HYCOM

The Hybrid Coordinate Ocean Model (HYCOM) provides global ocean predictions.

**Access**: https://www.hycom.org/

**Setup**:
```python
from src.data.hycom_client import HYCOMClient

client = HYCOMClient(product="GLBv0.08/expt_93.0")
```

**Available Products**:
- `GLBv0.08/expt_93.0`: Global 1/12° Analysis
- `GLBy0.08/expt_93.0`: Global 1/12° Forecast

## Usage Examples

### Data Acquisition

```python
# Download wind data
wind_data = copernicus_client.get_wind_data(
    lat_min=30, lat_max=45,
    lon_min=-75, lon_max=-55,
    start_date="2024-01-01",
    end_date="2024-01-07"
)

# Download ocean current data
current_data = hycom_client.get_ocean_currents(
    lat_min=30, lat_max=45,
    lon_min=-75, lon_max=-55,
    start_date="2024-01-01",
    end_date="2024-01-07"
)
```

### Analysis

```python
from src.analysis.wind_analysis import WindAnalyzer
from src.analysis.current_analysis import CurrentAnalyzer

# Wind analysis
wind_analyzer = WindAnalyzer(wind_data)
magnitude = wind_analyzer.compute_magnitude()
direction = wind_analyzer.compute_direction()
stats = wind_analyzer.compute_statistics(dim="time")
wind_rose = wind_analyzer.compute_wind_rose(n_directions=16)

# Current analysis
current_analyzer = CurrentAnalyzer(current_data)
vorticity = current_analyzer.compute_vorticity()
divergence = current_analyzer.compute_divergence()
kinetic_energy = current_analyzer.compute_kinetic_energy()

# Correlation analysis
correlations = wind_analyzer.correlate_with_current(current_data)
```

### Visualization

```python
from src.visualization.quiver_plots import QuiverPlotter
from src.visualization.animations import FieldAnimator

# Static plots
plotter = QuiverPlotter(figsize=(14, 10), dpi=300)

# Wind field
fig, ax = plotter.plot_wind_field(
    wind_u, wind_v,
    title="Wind Field",
    subsample=2
)

# Current field
fig, ax = plotter.plot_current_field(
    current_u, current_v,
    title="Ocean Currents",
    subsample=2
)

# Combined plot
fig, axes = plotter.plot_combined_wind_current(
    wind_u, wind_v,
    current_u, current_v,
    title="Wind and Currents"
)

# Animations
animator = FieldAnimator(figsize=(12, 8), dpi=100)

# Animate wind field
anim = animator.animate_wind_field(wind_u, wind_v, interval=100)
animator.save_animation(anim, "wind_animation.mp4", fps=10)

# Animate currents
anim = animator.animate_current_field(current_u, current_v, interval=100)
animator.save_animation(anim, "current_animation.mp4", fps=10)
```

### Particle Tracking

```python
from src.simulation.particle_tracker import ParticleTracker

# Initialize tracker
tracker = ParticleTracker(
    current_u=current_data["uo"],
    current_v=current_data["vo"],
    wind_u=wind_data["eastward_wind"],
    wind_v=wind_data["northward_wind"],
    wind_drift_coefficient=0.03,
    integration_method="rk4"
)

# Create initial particle positions
initial_positions = tracker.create_particle_grid(
    lon_min=-80, lon_max=-70,
    lat_min=25, lat_max=35,
    n_lon=10, n_lat=10
)

# Run simulation
trajectories = tracker.track_particles(
    initial_positions=initial_positions,
    n_steps=100,
    dt=3600  # 1 hour time step
)

# Animate trajectories
animator = FieldAnimator()
anim = animator.animate_particle_trajectories(
    trajectories,
    lat_bounds=(20, 40),
    lon_bounds=(-85, -65),
    title="Particle Drift"
)
animator.save_animation(anim, "particles.mp4")
```

## API Documentation

### Data Module

#### CopernicusClient

```python
client = CopernicusClient(username=None, password=None)

# Methods
client.get_wind_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
client.get_ocean_currents(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
client.get_combined_data(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
```

#### HYCOMClient

```python
client = HYCOMClient(base_url=..., product=...)

# Methods
client.get_ocean_currents(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
client.get_temperature(lat_min, lat_max, lon_min, lon_max, start_date, end_date)
client.list_available_products()
client.get_product_info(product=None)
```

### Analysis Module

#### WindAnalyzer

```python
analyzer = WindAnalyzer(wind_data, u_var="eastward_wind", v_var="northward_wind")

# Methods
analyzer.compute_magnitude()
analyzer.compute_direction(degrees=True)
analyzer.compute_statistics(dim=None)
analyzer.compute_wind_rose(n_directions=16, speed_bins=None)
analyzer.compute_temporal_trend(time_dim="time")
analyzer.correlate_with_current(current_data)
analyzer.compute_wind_stress(air_density=1.225, drag_coeff=0.0015)
```

#### CurrentAnalyzer

```python
analyzer = CurrentAnalyzer(current_data, u_var="uo", v_var="vo")

# Methods
analyzer.compute_magnitude(depth_level=0)
analyzer.compute_direction(depth_level=0, degrees=True)
analyzer.compute_statistics(depth_level=0, dim=None)
analyzer.compute_vorticity(depth_level=0)
analyzer.compute_divergence(depth_level=0)
analyzer.compute_geostrophic_current(ssh, latitude=None, g=9.81)
analyzer.compute_kinetic_energy(depth_level=0, density=1025)
analyzer.compute_enstrophy(depth_level=0)
```

### Visualization Module

#### QuiverPlotter

```python
plotter = QuiverPlotter(figsize=(12, 8), dpi=300, projection="PlateCarree")

# Methods
plotter.plot_wind_field(u, v, lat=None, lon=None, title="...", ...)
plotter.plot_current_field(u, v, lat=None, lon=None, title="...", ...)
plotter.plot_combined_wind_current(wind_u, wind_v, current_u, current_v, ...)
plotter.save_figure(fig, filename, format="png", dpi=None)
```

#### FieldAnimator

```python
animator = FieldAnimator(figsize=(12, 8), dpi=100, projection="PlateCarree")

# Methods
animator.animate_wind_field(u, v, title="...", interval=100, ...)
animator.animate_current_field(u, v, title="...", interval=100, ...)
animator.animate_particle_trajectories(trajectories, lat_bounds, lon_bounds, ...)
animator.save_animation(anim, filename, fps=10, bitrate=1800)
```

### Simulation Module

#### ParticleTracker

```python
tracker = ParticleTracker(
    current_u, current_v,
    wind_u=None, wind_v=None,
    wind_drift_coefficient=0.03,
    integration_method="rk4"
)

# Methods
tracker.track_particles(initial_positions, n_steps, dt=3600)
tracker.create_particle_grid(lon_min, lon_max, lat_min, lat_max, n_lon=10, n_lat=10)
tracker.create_random_particles(lon_min, lon_max, lat_min, lat_max, n_particles=100)
```

## Configuration

Edit `config/config.yaml` to customize:

- Data source credentials and endpoints
- Default geographic regions
- Visualization parameters
- Simulation settings
- Output directories

Example:
```yaml
data_sources:
  copernicus:
    username: "your_username"
    password: "your_password"

regions:
  gulf_of_mexico:
    lat_min: 18.0
    lat_max: 31.0
    lon_min: -98.0
    lon_max: -80.0

simulation:
  particle_tracker:
    integration_method: "rk4"
    dt: 3600
    wind_drift_coefficient: 0.03
```

## Examples

Run the example scripts:

```bash
# Copernicus Marine Service example
python examples/example_copernicus.py

# HYCOM example
python examples/example_hycom.py

# Particle tracking simulation
python examples/example_simulation.py
```

Or explore the Jupyter notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

This project uses:
- Black for code formatting
- Flake8 for linting
- NumPy-style docstrings

Run formatters:
```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ocean_wind_currents_sim,
  author = {Ocean Simulation Team},
  title = {Ocean-Wind-Currents Simulation},
  year = {2024},
  url = {https://github.com/lukaposternak/ocean-wind-currents-simulation}
}
```

## Data Attribution

- **Copernicus Marine Service**: E.U. Copernicus Marine Service Information
- **HYCOM**: Hybrid Coordinate Ocean Model Consortium

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Copernicus Marine Environment Monitoring Service (CMEMS)
- HYCOM Consortium
- Cartopy development team
- Xarray development team

## Contact

For questions or support, please open an issue on GitHub.

---

**Project Status**: Active Development

**Last Updated**: November 2024
