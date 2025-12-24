"""
Data processing module for filtering, normalization, and scoring
Based on logic from dividend_stockanalysis.ipynb
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from config import AppConfig


def normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to a 0-1 scale.
    If the series has no variation, returns a neutral score of 0.5 for all entries.

    Args:
        series: Pandas Series to normalize

    Returns:
        Normalized series
    """
    min_val = series.min()
    max_val = series.max()

    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def filter_stocks(
    df: pd.DataFrame,
    min_yield: float = AppConfig.DEFAULT_MIN_YIELD,
    payout_min: float = AppConfig.DEFAULT_PAYOUT_MIN,
    payout_max: float = AppConfig.DEFAULT_PAYOUT_MAX,
    min_years: int = AppConfig.DEFAULT_MIN_YEARS,
    min_growth: float = AppConfig.DEFAULT_MIN_GROWTH,
    min_growth_5y: float = AppConfig.DEFAULT_MIN_GROWTH_5Y,
    sectors: list = None,
    mkt_cap_tiers: list = None
) -> pd.DataFrame:
    """
    Filter stocks based on dividend criteria
    Based on notebook filtering logic from cell-12

    Args:
        df: Input dataframe
        min_yield: Minimum dividend yield (default: 3.5%)
        payout_min: Minimum payout ratio (default: 20%)
        payout_max: Maximum payout ratio (default: 80%)
        min_years: Minimum consecutive dividend years (default: 5)
        min_growth: Minimum 1-year dividend growth (default: 4%)
        min_growth_5y: Minimum 5-year dividend growth (default: 4%)
        sectors: List of sectors to include (default: None = all)

    Returns:
        Filtered dataframe
    """
    filtered = df.copy()

    # Apply filters
    if 'Div. Yield' in filtered.columns:
        filtered = filtered[filtered['Div. Yield'] >= min_yield]

    if 'Payout Ratio' in filtered.columns:
        filtered = filtered[
            (filtered['Payout Ratio'] >= payout_min) &
            (filtered['Payout Ratio'] <= payout_max)
        ]

    if 'Years' in filtered.columns:
        filtered = filtered[filtered['Years'] >= min_years]

    if 'Div. Growth' in filtered.columns:
        filtered = filtered[filtered['Div. Growth'] >= min_growth]

    if 'Div. Growth 5Y' in filtered.columns:
        filtered = filtered[filtered['Div. Growth 5Y'] >= min_growth_5y]

    # Sector filter
    if sectors and len(sectors) > 0 and 'Sector' in filtered.columns:
        filtered = filtered[filtered['Sector'].isin(sectors)]

    # Market Cap Tier filter
    if mkt_cap_tiers and len(mkt_cap_tiers) > 0 and 'mkt_cap_tier' in filtered.columns:
        filtered = filtered[filtered['mkt_cap_tier'].isin(mkt_cap_tiers)]

    return filtered


def calculate_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized metrics for scoring

    Args:
        df: Input dataframe with dividend data

    Returns:
        DataFrame with additional normalized columns
    """
    result = df.copy()

    # Normalize metrics (0-1 scale)
    if 'Div. Growth' in result.columns:
        result['norm_div_growth'] = normalize(result['Div. Growth'])

    if 'Div. Growth 5Y' in result.columns:
        result['norm_cagr'] = normalize(result['Div. Growth 5Y'])

    if 'Div. Yield' in result.columns:
        result['norm_yield'] = normalize(result['Div. Yield'])

    if 'Years' in result.columns:
        result['norm_years'] = normalize(result['Years'])

    # Inverted normalization for Payout Ratio (lower is better)
    if 'Payout Ratio' in result.columns:
        # Using fixed bounds from filtering criteria
        result['norm_payout'] = (0.8 - result['Payout Ratio']) / (0.8 - 0.2)

    return result


def calculate_composite_score(
    df: pd.DataFrame,
    weights: Dict[str, float] = None,
    score_type: str = 'high_dividend'
) -> pd.DataFrame:
    """
    Calculate composite score based on normalized metrics and weights

    Args:
        df: DataFrame with normalized metrics
        weights: Dictionary of weights for each metric
        score_type: 'high_dividend' or 'dividend_growth'

    Returns:
        DataFrame with composite score column added
    """
    result = df.copy()

    # Use default weights if not provided
    if weights is None:
        if score_type == 'high_dividend':
            weights = AppConfig.HIGH_DIV_WEIGHTS
        else:  # dividend_growth
            weights = AppConfig.DIV_GROWTH_WEIGHTS

    # Ensure normalized metrics exist
    if 'norm_div_growth' not in result.columns:
        result = calculate_normalized_metrics(result)

    # Calculate composite score with proper column naming
    score_column_map = {
        'high_dividend': 'high_div_composite',
        'dividend_growth': 'dividend_growth_composite'
    }
    score_column = score_column_map.get(score_type, f'{score_type}_composite')

    result[score_column] = (
        weights.get('yield', 0) * result.get('norm_yield', 0) +
        weights.get('years', 0) * result.get('norm_years', 0) +
        weights.get('cagr', 0) * result.get('norm_cagr', 0) +
        weights.get('growth', 0) * result.get('norm_div_growth', 0) +
        weights.get('payout', 0) * result.get('norm_payout', 0)
    )

    return result


def get_top_stocks(
    df: pd.DataFrame,
    score_column: str = 'high_div_composite',
    n: int = 20
) -> pd.DataFrame:
    """
    Get top N stocks by composite score

    Args:
        df: DataFrame with score column
        score_column: Name of the score column
        n: Number of top stocks to return

    Returns:
        DataFrame with top N stocks sorted by score
    """
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataframe")

    return df.nlargest(n, score_column)


def categorize_market_cap(market_cap_value):
    """
    Categorize market cap into tiers based on Russell Index criteria

    Args:
        market_cap_value: Market cap as string (e.g., "911.47B") or numeric in billions

    Returns:
        String representing the market cap tier
    """
    if pd.isna(market_cap_value):
        return 'Unknown'

    if isinstance(market_cap_value, str):
        value_str = market_cap_value.strip().upper()
        if not value_str or value_str == '-':
            return 'Unknown'

        value_str = value_str.replace('$', '')

        multiplier = 1
        if value_str.endswith('T'):
            multiplier = 1e12
            value_str = value_str[:-1]
        elif value_str.endswith('B'):
            multiplier = 1e9
            value_str = value_str[:-1]
        elif value_str.endswith('M'):
            multiplier = 1e6
            value_str = value_str[:-1]
        elif value_str.endswith('K'):
            multiplier = 1e3
            value_str = value_str[:-1]

        try:
            numeric_value = float(value_str)
            market_cap_millions = (numeric_value * multiplier) / 1e6
        except ValueError:
            return 'Unknown'
    elif isinstance(market_cap_value, (int, float)):
        market_cap_millions = market_cap_value * 1000
    else:
        return 'Unknown'

    if market_cap_millions >= 200000:
        return 'Mega-cap'
    elif market_cap_millions >= 10000:
        return 'Large-cap'
    elif market_cap_millions >= 2000:
        return 'Mid-cap'
    elif market_cap_millions >= 300:
        return 'Small-cap'
    elif market_cap_millions >= 50:
        return 'Micro-cap'
    else:
        return 'Nano-cap'


def add_market_cap_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market cap tier column to dataframe

    Args:
        df: DataFrame with 'Market Cap' column (in billions)

    Returns:
        DataFrame with 'mkt_cap_tier' column added
    """
    result = df.copy()

    if 'Market Cap' in result.columns:
        result['mkt_cap_tier'] = result['Market Cap'].apply(categorize_market_cap)

    return result


def prepare_display_dataframe(
    df: pd.DataFrame,
    display_columns: list = None
) -> pd.DataFrame:
    """
    Prepare dataframe for display with formatted values

    Args:
        df: Input dataframe
        display_columns: List of columns to display

    Returns:
        Formatted dataframe
    """
    result = df.copy()

    # Format percentage columns
    pct_columns = ['Div. Yield', 'Payout Ratio', 'Div. Growth', 'Div. Growth 5Y']
    for col in pct_columns:
        if col in result.columns:
            result[col] = (result[col] * 100).round(2)

    # Select display columns if specified
    if display_columns:
        available_cols = [col for col in display_columns if col in result.columns]
        result = result[available_cols]

    return result
