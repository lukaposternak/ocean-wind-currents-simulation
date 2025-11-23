"""
Unit tests for data acquisition modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.copernicus_client import CopernicusClient
from src.data.hycom_client import HYCOMClient


class TestCopernicusClient:
    """Test suite for CopernicusClient."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = CopernicusClient()
        assert client is not None
        assert hasattr(client, 'username')
        assert hasattr(client, 'password')
    
    def test_authentication(self):
        """Test authentication method."""
        client = CopernicusClient()
        result = client.authenticate("test_user", "test_pass")
        assert result is True
        assert client.username == "test_user"
        assert client.password == "test_pass"
        assert client.authenticated is True
    
    def test_get_wind_data(self):
        """Test wind data retrieval (mock data)."""
        client = CopernicusClient()
        
        data = client.get_wind_data(
            lat_min=20.0,
            lat_max=25.0,
            lon_min=-95.0,
            lon_max=-90.0,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert data is not None
        assert "eastward_wind" in data
        assert "northward_wind" in data
        assert "latitude" in data.dims or "lat" in data.dims
        assert "longitude" in data.dims or "lon" in data.dims
    
    def test_get_ocean_currents(self):
        """Test ocean current data retrieval (mock data)."""
        client = CopernicusClient()
        
        data = client.get_ocean_currents(
            lat_min=20.0,
            lat_max=25.0,
            lon_min=-95.0,
            lon_max=-90.0,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert data is not None
        assert "uo" in data
        assert "vo" in data
        assert "depth" in data.dims
    
    def test_get_combined_data(self):
        """Test combined data retrieval."""
        client = CopernicusClient()
        
        wind_data, current_data = client.get_combined_data(
            lat_min=20.0,
            lat_max=25.0,
            lon_min=-95.0,
            lon_max=-90.0,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert wind_data is not None
        assert current_data is not None


class TestHYCOMClient:
    """Test suite for HYCOMClient."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = HYCOMClient()
        assert client is not None
        assert hasattr(client, 'base_url')
        assert hasattr(client, 'product')
    
    def test_list_products(self):
        """Test listing available products."""
        client = HYCOMClient()
        products = client.list_available_products()
        assert isinstance(products, list)
        assert len(products) > 0
    
    def test_get_product_info(self):
        """Test getting product information."""
        client = HYCOMClient()
        info = client.get_product_info()
        assert isinstance(info, dict)
        assert "name" in info or len(info) > 0
    
    def test_get_ocean_currents(self):
        """Test ocean current data retrieval (mock data)."""
        client = HYCOMClient()
        
        data = client.get_ocean_currents(
            lat_min=30.0,
            lat_max=35.0,
            lon_min=-75.0,
            lon_max=-70.0,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert data is not None
        assert "water_u" in data
        assert "water_v" in data
        assert "time" in data.dims
    
    def test_get_temperature(self):
        """Test temperature data retrieval (mock data)."""
        client = HYCOMClient()
        
        data = client.get_temperature(
            lat_min=30.0,
            lat_max=35.0,
            lon_min=-75.0,
            lon_max=-70.0,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert data is not None
        assert "water_temp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
