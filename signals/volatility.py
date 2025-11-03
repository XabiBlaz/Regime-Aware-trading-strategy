"""
Volatility regime detection + realised vol + data loader.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import numpy as np
import time

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover - offline fallback
    yf = None

DATA_DIR: Optional[Path]
_data_dir_candidate = Path(__file__).resolve().parents[1] / "data" / "cache"
try:
    _data_dir_candidate.mkdir(parents=True, exist_ok=True)
except OSError:
    DATA_DIR = None
else:
    DATA_DIR = _data_dir_candidate

CACHE_PRICES = "prices.csv"
CACHE_VIX = "vix.csv"

TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA",
    "SPY", "QQQ", "IWM", "XLE", "XLK", "USO",
    "TLT", "GLD",
    "^VIX"
]
START = "2014-01-01"


class Regime(Enum):
    LOW = "low_vol"
    MEDIUM = "medium_vol" 
    HIGH = "high_vol"


def _cache_paths() -> Tuple[Optional[Path], Optional[Path]]:
    if DATA_DIR is None:
        return None, None
    return DATA_DIR / CACHE_PRICES, DATA_DIR / CACHE_VIX


def _load_from_cache() -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    prices_path, vix_path = _cache_paths()
    if not prices_path or not vix_path:
        return None
    if not prices_path.exists() or not vix_path.exists():
        return None

    try:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
    except Exception:
        return None

    if prices.empty or vix_df.empty:
        return None

    vix = vix_df.iloc[:, 0]
    vix.name = "^VIX"

    prices = prices.sort_index()
    vix = vix.sort_index()

    aligned_index = prices.index.intersection(vix.index)
    if aligned_index.empty:
        return None

    prices = prices.loc[aligned_index]
    vix = vix.loc[aligned_index]
    return prices, vix


def _save_to_cache(prices: pd.DataFrame, vix: pd.Series) -> None:
    prices_path, vix_path = _cache_paths()
    if not prices_path or not vix_path:
        return

    try:
        prices.to_csv(prices_path)
        vix.to_frame(name="^VIX").to_csv(vix_path)
    except OSError:
        # Read-only environments are acceptable; simply skip caching.
        pass


def load_prices_and_vix(force_refresh: bool = False, prefer_download: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load price data and VIX with robust error handling, caching, and retries.

    Args:
        force_refresh: Skip cached files and re-download data.
        prefer_download: If False, skip network calls and fall back to cached or
            synthetic data immediately.

    Returns:
        Tuple of (prices DataFrame, VIX Series)
    """
    if not force_refresh:
        cached = _load_from_cache()
        if cached:
            return cached

    loaders: list[tuple[str, Callable[[], Tuple[pd.DataFrame, pd.Series]]]] = []
    if prefer_download and yf is not None:
        loaders.extend([
            ("Downloading all tickers together", _download_all_together),
            ("Downloading tickers individually", _download_individually),
        ])
    loaders.append(("Using fallback data", _create_mock_data))

    last_error: Optional[Exception] = None
    for attempt, (label, loader) in enumerate(loaders, start=1):
        try:
            print(f"Attempt {attempt}: {label}...")
            prices, vix = loader()
            _save_to_cache(prices, vix)
            return prices, vix
        except Exception as exc:  # pragma: no cover - defensive path
            last_error = exc
            if attempt < len(loaders) and loader is not _create_mock_data:
                print(f"Attempt {attempt} failed: {exc}")
                time.sleep(2)
            else:
                print(f"Attempt {attempt} failed: {exc}")

    raise RuntimeError("All data loading attempts failed") from last_error

def _download_all_together() -> Tuple[pd.DataFrame, pd.Series]:
    """Download all tickers at once."""
    if yf is None:
        raise RuntimeError("yfinance is not available")
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
    if yf is None:
        raise RuntimeError("yfinance is not available")
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
    dates = pd.date_range(start=START, end="2024-12-31", freq="B")
    np.random.seed(42)  # For reproducibility
    n_days = len(dates)

    asset_names = [ticker for ticker in TICKERS if ticker != "^VIX"]
    base_prices = np.linspace(40, 220, len(asset_names))

    price_data = {}
    for initial_price, asset in zip(base_prices, asset_names):
        # Generate returns with some volatility and drift
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% daily return, 2% volatility
        prices = [initial_price]

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
