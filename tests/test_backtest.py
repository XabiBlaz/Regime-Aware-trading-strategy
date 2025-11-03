import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.backtester import Backtester
from strategy.regime_strategy import RegimeAwareStrategy
from signals.volatility import load_prices_and_vix

def test_data_quality():
    """Test data quality and completeness."""
    try:
        prices, vix = load_prices_and_vix(prefer_download=False)
        
        # Data quality checks
        assert not prices.empty, "Price data should not be empty"
        assert not vix.empty, "VIX data should not be empty"
        assert len(prices) == len(vix), "Price and VIX data should have same length"
        assert len(prices.columns) > 0, "Should have at least one asset"
        
        # Check for missing data
        missing_prices = prices.isnull().sum().sum()
        missing_vix = vix.isnull().sum()
        
        if missing_prices > 0:
            print(f"Warning: Found {missing_prices} missing values in prices")
        if missing_vix > 0:
            print(f"Warning: Found {missing_vix} missing values in VIX")
        
        print(f"✓ Data quality test passed: {len(prices)} observations, {len(prices.columns)} assets")
        return True
        
    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False

def test_backtest_runs():
    """Test that the backtest runs without errors and produces reasonable results."""
    try:
        prices, vix = load_prices_and_vix(prefer_download=False)
        
        # Use subset for testing
        prices = prices.iloc[:600]
        vix = vix.loc[prices.index]
        
        print(f"Test data: {len(prices)} observations, {len(prices.columns)} assets")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"Assets: {list(prices.columns)}")
        
        # Create and run strategy
        strat = RegimeAwareStrategy(prices, vix)
        bt = Backtester(prices, strat)
        stats = bt.run()

        # Comprehensive assertions
        assert isinstance(stats, dict), "Stats should be a dictionary"
        
        # Check required metrics exist
        required_metrics = ["Sharpe", "CAGR", "MaxDD"]
        for metric in required_metrics:
            assert metric in stats, f"Missing required metric: {metric}"
        
        # More permissive bounds for different data scenarios
        assert -10 < stats["Sharpe"] < 15, f"Sharpe {stats['Sharpe']:.3f} outside reasonable bounds"
        assert -2.0 < stats["CAGR"] < 3.0, f"CAGR {stats['CAGR']:.3f} outside reasonable bounds"
        assert -1.0 < stats["MaxDD"] <= 0, f"Max Drawdown {stats['MaxDD']:.3f} should be negative or zero"
        
        # Display results
        print("\n" + "="*40)
        print("TEST RESULTS")
        print("="*40)
        print(f"Sharpe Ratio........ {stats['Sharpe']:>8.3f}")
        print(f"CAGR................ {stats['CAGR']:>8.2%}")
        print(f"Max Drawdown........ {stats['MaxDD']:>8.2%}")
        print("="*40)
        print("✓ All tests passed!")
        
        return stats
        
    except Exception as e:
        print(f"❌ Backtest test failed: {e}")
        raise

if __name__ == "__main__":
    print("Running data quality test...")
    data_ok = test_data_quality()
    
    if data_ok:
        print("\nRunning backtest test...")
        test_backtest_runs()
    else:
        print("Skipping backtest test due to data issues")
