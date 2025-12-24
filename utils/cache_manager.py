"""
Caching strategies for Streamlit application
"""

import streamlit as st
import pandas as pd
from typing import Optional
from utils.data_loader import DataManager


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_main_dataframe(use_cached: bool = True) -> Optional[pd.DataFrame]:
    """
    Load and cache the main dividend dataset

    Args:
        use_cached: Whether to use cached CSV file

    Returns:
        DataFrame with dividend data
    """
    manager = DataManager()
    df = manager.get_main_dataframe(use_cached=use_cached)

    if df is not None:
        # Data type optimization
        if 'Symbol' in df.columns:
            df['Symbol'] = df['Symbol'].astype('category')
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype('category')

    return df


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_historical_prices(symbol: str, period: str = "max", start_date: str = None, end_date: str = None):
    """
    Cache historical price data per symbol

    Args:
        symbol: Stock symbol
        period: Time period (default: "max")
        start_date: Start date (YYYY-MM-DD) - overrides period if provided
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with historical price data including Close and Dividends columns
    """
    import yfinance as yf
    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        hist = ticker.history(start=start_date, end=end_date)
    else:
        hist = ticker.history(period=period)

    # Ensure Dividends column exists
    if 'Dividends' not in hist.columns:
        hist['Dividends'] = 0.0

    return hist


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_benchmark_data(symbol: str = 'SPY', start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load benchmark (S&P 500) historical data with caching.

    Args:
        symbol: Benchmark ticker (default: SPY)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with Date index, Close and Dividends columns
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)

    if start_date and end_date:
        hist = ticker.history(start=start_date, end=end_date)
    else:
        hist = ticker.history(period='max')

    # Ensure Dividends column exists
    if 'Dividends' not in hist.columns:
        hist['Dividends'] = 0.0

    return hist[['Close', 'Dividends']] if not hist.empty else pd.DataFrame()


@st.cache_resource
def get_yfinance_session():
    """Reuse yfinance session across calls"""
    import yfinance as yf
    return yf.Session()


def clear_all_caches():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
