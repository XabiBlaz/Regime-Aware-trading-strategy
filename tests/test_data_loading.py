import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from signals.volatility import load_prices_and_vix

def test_data_loading():
    """Test that data loading works correctly."""
    try:
        prices, vix = load_prices_and_vix(prefer_download=False)
        
        # Basic validation
        assert not prices.empty, "Price data should not be empty"
        assert not vix.empty, "VIX data should not be empty"
        assert len(prices) == len(vix), "Price and VIX data should have same length"
        
        # Data quality checks
        assert prices.isnull().sum().sum() == 0, "Price data should not contain NaN"
        assert vix.isnull().sum() == 0, "VIX data should not contain NaN"
        
        print(f"✓ Data loading test passed: {len(prices)} observations, {len(prices.columns)} assets")
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
