"""
One-shot back-test runner.
"""
from pathlib import Path
from backtest.backtester import Backtester
from strategy.regime_strategy import RegimeAwareStrategy
from signals.volatility import load_prices_and_vix

def main():
    # Load data (VIX is already separated)
    prices, vix = load_prices_and_vix()
    
    # REMOVE THIS LINE - VIX is already separated:
    # prices = prices.drop(columns="^VIX")
    
    # Use subset for faster testing (optional)
    prices = prices.iloc[:600]  # First 600 observations
    vix = vix.loc[prices.index]  # Align VIX with price data
    
    print(f"Using {len(prices)} observations from {prices.index[0]} to {prices.index[-1]}")
    print(f"Assets: {list(prices.columns)}")
    print(f"VIX range: {vix.min():.2f} to {vix.max():.2f}")
    
    # Create strategy and backtester
    strategy = RegimeAwareStrategy(prices, vix)
    backtester = Backtester(prices, strategy)
    
    # Run backtest
    print("\nRunning backtest...")
    stats = backtester.run()
    
    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if key in ['Sharpe', 'Sortino', 'Calmar']:
                print(f"{key:.<20} {value:>8.3f}")
            elif key in ['CAGR', 'Volatility', 'Max_Drawdown']:
                print(f"{key:.<20} {value:>8.2%}")
            else:
                print(f"{key:.<20} {value:>8.4f}")
        else:
            print(f"{key:.<20} {value}")
    
    # Plot equity curve
    print("\nGenerating equity curve...")
    backtester.plot_equity_curve()

if __name__ == "__main__":
    main()
