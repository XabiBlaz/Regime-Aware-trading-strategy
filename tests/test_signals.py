import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from signals.momentum import momentum_zscores, momentum_positions
from signals.pairs import pairs_positions
from signals.volatility import load_prices_and_vix

def test_momentum():
    """Test momentum signal generation."""
    prices, _ = load_prices_and_vix()
    prices = prices.iloc[:300]  # VIX already separated - no need to drop
    
    z = momentum_zscores(prices)
    w = momentum_positions(z)
    assert (abs(w.sum(axis=1)) < 1e-6).all(), "Momentum positions should be net zero"
    print("✓ Momentum test passed")

def test_pairs():
    """Test pairs trading signals."""
    prices, _ = load_prices_and_vix()
    prices = prices.iloc[:300]  # VIX already separated
    
    pos = pairs_positions(prices)
    s = pos.sum(axis=1).abs().max()
    assert s < 1e-6, "Pairs positions should be dollar-neutral"
    print("✓ Pairs test passed")

if __name__ == "__main__":
    test_momentum()
    test_pairs()
    print("✓ All signal tests passed!")