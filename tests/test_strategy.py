import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategy.regime_strategy import RegimeAwareStrategy
from signals.volatility import load_prices_and_vix

def test_strategy_positions():
    """Test that strategy generates valid position weights."""
    prices, vix = load_prices_and_vix()
    
    # Use subset for testing
    prices = prices.iloc[:400]
    vix = vix.loc[prices.index]
    
    print(f"Testing strategy with {len(prices)} observations")
    
    # Create strategy and get positions
    strat = RegimeAwareStrategy(prices, vix)
    w = strat.positions()

    # Validate position weights
    assert w.shape == prices.shape, f"Weight shape {w.shape} doesn't match prices {prices.shape}"
    assert w.abs().max().max() < 5.0, f"Weights too large: {w.abs().max().max()}"
    assert w.isna().sum().sum() == 0, "Weights should not contain NaN"
    
    print(f"✓ Strategy test passed - Max weight: {w.abs().max().max():.3f}")

if __name__ == "__main__":
    test_strategy_positions()
