import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from signals.volatility import load_prices_and_vix

def test_data_shape():
    """Test data loading and basic properties."""
    prices, vix = load_prices_and_vix()
    
    # VIX should be SEPARATED, not included in prices
    assert "^VIX" not in prices.columns, "VIX should be separated from prices"
    assert len(prices) > 2500, f"Expected >2500 observations, got {len(prices)}"
    assert prices.index[0].year == 2014, f"Expected data to start in 2014, got {prices.index[0].year}"
    assert prices.isna().sum().sum() < 0.01 * prices.size, "Too much missing data in prices"
    
    # Test VIX separately
    assert len(vix) == len(prices), "VIX and prices should have same length"
    assert vix.isna().sum() < 0.01 * len(vix), "Too much missing data in VIX"
    
    print(f"✓ Data shape test passed: {len(prices)} obs, {len(prices.columns)} assets")

if __name__ == "__main__":
    test_data_shape()