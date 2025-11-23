"""Setup script for ocean-wind-currents-simulation package."""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    """Read the long description from README.md."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ocean-wind-currents-simulation",
    version="0.1.0",
    author="Ocean Simulation Team",
    author_email="",
    description="A comprehensive Python package for simulating, analyzing, and visualizing wind and superficial ocean currents",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/lukaposternak/ocean-wind-currents-simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "ocean-sim=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
