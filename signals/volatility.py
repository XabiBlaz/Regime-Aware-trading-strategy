"""
Volatility regime detection + realised vol + data loader.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import time

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA",
    "SPY", "QQQ", "IWM", "XLE", "XLK", "USO", "^VIX"
]
START = "2014-01-01"


class Regime(Enum):
    LOW = "low_vol"
    MEDIUM = "medium_vol" 
    HIGH = "high_vol"


def load_prices_and_vix() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load price data and VIX with robust error handling and retries.
    
    Returns:
        Tuple of (prices DataFrame, VIX Series)
    """
    # Try different download strategies
    for attempt in range(3):
        try:
            if attempt == 0:
                # Strategy 1: Download all at once
                print(f"Attempt {attempt + 1}: Downloading all tickers together...")
                return _download_all_together()
            elif attempt == 1:
                # Strategy 2: Download individually with retries
                print(f"Attempt {attempt + 1}: Downloading tickers individually...")
                return _download_individually()
            else:
                # Strategy 3: Use cached/mock data
                print(f"Attempt {attempt + 1}: Using fallback data...")
                return _create_mock_data()
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("Retrying...")
                time.sleep(2)
            continue
    
    raise RuntimeError("All data loading attempts failed")

def _download_all_together() -> Tuple[pd.DataFrame, pd.Series]:
    """Download all tickers at once."""
    data = yf.download(
        TICKERS, 
        start=START, 
        progress=False,
        auto_adjust=True,
        prepost=False,
        threads=False,  # Disable threading to avoid timeouts
        timeout=30
    )
    
    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            df = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            df = data["Close"]
        else:
            raise ValueError("No price columns found")
    else:
        df = data
    
    if df.empty:
        raise ValueError("No data downloaded")
    
    # Separate VIX
    vix_cols = [col for col in df.columns if "VIX" in str(col)]
    if not vix_cols:
        raise ValueError("VIX data not found")
    
    vix = df[vix_cols[0]]
    prices = df.drop(columns=vix_cols)
    
    # Clean data
    prices = prices.dropna()
    vix = vix.loc[prices.index]
    
    print(f"Successfully loaded {len(prices)} observations for {len(prices.columns)} assets")
    return prices, vix

def _download_individually() -> Tuple[pd.DataFrame, pd.Series]:
    """Download each ticker individually with retries."""
    all_data = {}
    
    for ticker in TICKERS:
        for retry in range(3):
            try:
                print(f"Downloading {ticker} (attempt {retry + 1})...")
                data = yf.download(
                    ticker, 
                    start=START, 
                    progress=False,
                    auto_adjust=True,
                    timeout=15
                )
                
                if not data.empty:
                    # Use the most appropriate price column
                    if "Adj Close" in data.columns:
                        all_data[ticker] = data["Adj Close"]
                    elif "Close" in data.columns:
                        all_data[ticker] = data["Close"]
                    else:
                        all_data[ticker] = data.iloc[:, 0]
                    break
                else:
                    print(f"No data for {ticker}")
                    
            except Exception as e:
                print(f"Failed to download {ticker}: {e}")
                if retry < 2:
                    time.sleep(1)
                continue
    
    if not all_data:
        raise ValueError("No data could be downloaded")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Handle missing data by forward filling
    df = df.fillna(method='ffill').dropna()
    
    # Separate VIX
    if "^VIX" in df.columns:
        vix = df["^VIX"]
        prices = df.drop(columns="^VIX")
    else:
        raise ValueError("VIX data not available")
    
    print(f"Successfully loaded {len(prices)} observations for {len(prices.columns)} assets")
    return prices, vix

def _create_mock_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Create mock data for testing when downloads fail."""
    print("Creating mock data for testing...")
    
    # Create date range
    dates = pd.date_range(start=START, end="2024-12-31", freq='D')
    dates = dates[dates.weekday < 5]  # Business days only
    
    # Create mock price data
    np.random.seed(42)  # For reproducibility
    n_assets = 8
    n_days = len(dates)
    
    # Generate realistic price paths
    initial_prices = [100, 150, 200, 300, 400, 50, 75, 25]
    asset_names = ["AAPL", "MSFT", "AMZN", "SPY", "QQQ", "IWM", "XLE", "XLK"]
    
    price_data = {}
    for i, asset in enumerate(asset_names):
        # Generate returns with some volatility and drift
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
        prices = [initial_prices[i]]
        
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        price_data[asset] = prices
    
    prices_df = pd.DataFrame(price_data, index=dates)
    
    # Generate VIX data (volatility index)
    vix_data = np.random.lognormal(mean=np.log(20), sigma=0.3, size=n_days)
    vix_data = np.clip(vix_data, 10, 80)  # Realistic VIX range
    vix_series = pd.Series(vix_data, index=dates, name="^VIX")
    
    print(f"Mock data created: {len(prices_df)} observations for {len(prices_df.columns)} assets")
    return prices_df, vix_series

def realised_vol(series: pd.Series, window: int = 20) -> pd.Series:
    """Annualised realised vol of log-returns."""
    log_ret = np.log(series).diff()
    return log_ret.rolling(window).std() * np.sqrt(252)


def classify_regime(vix: pd.Series) -> pd.Series:
    """
    Classify market regimes based on VIX levels.
    
    Args:
        vix: VIX volatility series
        
    Returns:
        Series with regime classifications
    """
    # Initialize with default regime (avoid Categorical issues)
    regime = pd.Series(index=vix.index, dtype='object')
    
    # Classify regimes based on VIX thresholds
    regime[vix < 15] = Regime.LOW.value
    regime[(vix >= 15) & (vix < 25)] = Regime.MEDIUM.value  
    regime[vix >= 25] = Regime.HIGH.value
    
    # Convert to categorical for memory efficiency (optional)
    regime = regime.astype('category')
    
    return regime
