"""
Plotly visualization templates for dividend stock analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


def create_top_stocks_bar_chart(
    df: pd.DataFrame,
    score_column: str,
    title: str = "Top 10 Stocks by Composite Score",
    n: int = 10
) -> go.Figure:
    """
    Create bar chart showing top N stocks by score

    Args:
        df: DataFrame with stock data
        score_column: Column name for the score
        title: Chart title
        n: Number of top stocks to display

    Returns:
        Plotly Figure object
    """
    top_stocks = df.nlargest(n, score_column)

    fig = go.Figure(data=[
        go.Bar(
            x=top_stocks['Symbol'],
            y=top_stocks[score_column],
            text=top_stocks[score_column].round(3),
            textposition='outside',
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Symbol",
        yaxis_title="Composite Score",
        height=400,
        showlegend=False
    )

    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    title: str = "Dividend Yield vs Years",
    hover_data: list = None
) -> go.Figure:
    """
    Create scatter plot with bubble sizes

    Args:
        df: DataFrame with stock data
        x_col: Column for x-axis
        y_col: Column for y-axis
        size_col: Column for bubble size
        title: Chart title
        hover_data: Additional columns to show on hover

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        hover_name='Symbol',
        hover_data=hover_data,
        title=title,
        color=size_col,
        color_continuous_scale='Blues'
    )

    fig.update_layout(height=500)

    return fig


def create_distribution_histogram(
    df: pd.DataFrame,
    column: str,
    title: str = "Distribution",
    bins: int = 30
) -> go.Figure:
    """
    Create histogram showing distribution

    Args:
        df: DataFrame with data
        column: Column to plot
        title: Chart title
        bins: Number of bins

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=df[column],
            nbinsx=bins,
            marker_color='lightgreen'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig


def create_dual_axis_chart(
    price_data: pd.DataFrame,
    yield_data: pd.Series,
    title: str = "Stock Price & Dividend Yield"
) -> go.Figure:
    """
    Create dual-axis chart with price (candlestick) and yield (line)

    Args:
        price_data: DataFrame with OHLC data
        yield_data: Series with dividend yield values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Candlestick for price
    if all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name="Price"
        ))
    else:
        # If OHLC not available, use line chart
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            name="Price",
            line=dict(color='blue')
        ))

    # Line for yield on secondary y-axis
    fig.add_trace(go.Scatter(
        x=yield_data.index,
        y=yield_data.values * 100,  # Convert to percentage
        name="Dividend Yield (%)",
        yaxis="y2",
        line=dict(color='orange', width=2)
    ))

    # Layout with dual y-axis
    fig.update_layout(
        title=title,
        yaxis=dict(title="Price ($)"),
        yaxis2=dict(
            title="Dividend Yield (%)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        height=600
    )

    return fig


def create_yield_gauge(
    current_yield: float,
    min_5y: float,
    avg_5y: float,
    max_5y: float
) -> go.Figure:
    """
    Create gauge chart showing current yield vs historical range

    Args:
        current_yield: Current dividend yield
        min_5y: 5-year minimum
        avg_5y: 5-year average
        max_5y: 5-year maximum

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_yield * 100,  # Convert to percentage
        delta={'reference': avg_5y * 100},
        title={'text': "Current Yield vs 5Y Average (%)"},
        gauge={
            'axis': {'range': [min_5y * 100, max_5y * 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_5y * 100, avg_5y * 100], 'color': "lightgray"},
                {'range': [avg_5y * 100, max_5y * 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_5y * 100
            }
        }
    ))

    fig.update_layout(height=400)

    return fig


def create_dividend_history_bar(
    dividends: pd.Series,
    title: str = "Annual Dividend History"
) -> go.Figure:
    """
    Create bar chart showing dividend history

    Args:
        dividends: Series with dividend amounts (indexed by date)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Group by year and sum
    annual_divs = dividends.groupby(dividends.index.year).sum()

    fig = go.Figure(data=[
        go.Bar(
            x=annual_divs.index,
            y=annual_divs.values,
            marker_color='green',
            text=annual_divs.values.round(2),
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Annual Dividend ($)",
        height=400,
        showlegend=False
    )

    return fig


# --- Portfolio Backtest Visualization Functions ---

def create_portfolio_growth_chart(
    portfolio_values: pd.DataFrame,
    portfolio_no_drip: pd.DataFrame,
    benchmark_values: pd.DataFrame,
    buyhold_values: pd.DataFrame = None,
    schd_values: pd.DataFrame = None
) -> go.Figure:
    """
    Create multi-line chart showing portfolio growth comparison.

    Args:
        portfolio_values: DataFrame with Date index and 'Value' column (DRIP enabled)
        portfolio_no_drip: DataFrame with Date index and 'Value' column (DRIP disabled)
        benchmark_values: DataFrame with Date index and 'Value' column (benchmark)
        buyhold_values: DataFrame with Date index and 'Value' column (buy & hold strategy)
        schd_values: DataFrame with Date index and 'Value' column (SCHD ETF)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Portfolio with DRIP
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values['Value'],
        mode='lines',
        name='Portfolio (DRIP)',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Portfolio (DRIP)</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))

    # Portfolio without DRIP
    fig.add_trace(go.Scatter(
        x=portfolio_no_drip.index,
        y=portfolio_no_drip['Value'],
        mode='lines',
        name='Portfolio (No DRIP)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>Portfolio (No DRIP)</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))

    # Buy & Hold Benchmark
    if buyhold_values is not None and not buyhold_values.empty:
        fig.add_trace(go.Scatter(
            x=buyhold_values.index,
            y=buyhold_values['Value'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#9467bd', width=2, dash='dot'),
            hovertemplate='<b>Buy & Hold</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    # Benchmark (S&P 500)
    if not benchmark_values.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values['Value'],
            mode='lines',
            name='S&P 500 (SPY)',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    # SCHD ETF Benchmark
    if schd_values is not None and not schd_values.empty:
        fig.add_trace(go.Scatter(
            x=schd_values.index,
            y=schd_values['Value'],
            mode='lines',
            name='SCHD ETF',
            line=dict(color='#d62728', width=2),
            hovertemplate='<b>SCHD ETF</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Portfolio Growth vs Benchmarks",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_dividend_income_chart(dividend_history: pd.DataFrame) -> go.Figure:
    """
    Create stacked bar chart showing dividend income by stock.

    Args:
        dividend_history: DataFrame with columns: Date, Symbol, Amount

    Returns:
        Plotly Figure object
    """
    if dividend_history.empty:
        return go.Figure()

    # Group by year and symbol
    dividend_history['Year'] = pd.to_datetime(dividend_history['Date']).dt.year
    annual_dividends = dividend_history.groupby(['Year', 'Symbol'])['Amount'].sum().reset_index()

    # Pivot for stacked bar
    pivot_df = annual_dividends.pivot(index='Year', columns='Symbol', values='Amount').fillna(0)

    fig = go.Figure()

    # Add trace for each stock
    for symbol in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[symbol],
            name=symbol,
            hovertemplate=f'<b>{symbol}</b><br>Year: %{{x}}<br>Dividends: $%{{y:,.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title="Annual Dividend Income by Stock",
        xaxis_title="Year",
        yaxis_title="Dividend Income ($)",
        barmode='stack',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_cumulative_dividend_chart(dividend_history: pd.DataFrame) -> go.Figure:
    """
    Create line chart showing cumulative dividend income over time.

    Args:
        dividend_history: DataFrame with columns: Date, Amount

    Returns:
        Plotly Figure object
    """
    if dividend_history.empty:
        return go.Figure()

    # Sort by date and calculate cumulative sum
    df = dividend_history.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Cumulative'] = df['Amount'].cumsum()

    fig = go.Figure()

    # Cumulative dividends
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative'],
        mode='lines',
        name='Cumulative Dividends',
        line=dict(color='green', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 128, 0, 0.1)',
        hovertemplate='<b>Cumulative Dividends</b><br>Date: %{x}<br>Total: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Cumulative Dividend Income Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Dividends ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_underwater_chart(daily_values: pd.DataFrame) -> go.Figure:
    """
    Create underwater chart showing drawdown over time.

    Args:
        daily_values: DataFrame with Date index and 'Value' column

    Returns:
        Plotly Figure object
    """
    if daily_values.empty:
        return go.Figure()

    # Calculate running maximum
    running_max = daily_values['Value'].expanding().max()

    # Calculate drawdown percentage
    drawdown = (daily_values['Value'] - running_max) / running_max * 100

    fig = go.Figure()

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=0),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Portfolio Drawdown Over Time (Underwater Chart)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def create_return_distribution_chart(daily_values: pd.DataFrame) -> go.Figure:
    """
    Create histogram showing daily return distribution with normal curve overlay.

    Args:
        daily_values: DataFrame with Date index and 'Value' column

    Returns:
        Plotly Figure object
    """
    if daily_values.empty or len(daily_values) < 2:
        return go.Figure()

    # Calculate daily returns
    returns = daily_values['Value'].pct_change().dropna() * 100  # Convert to percentage

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=50,
        name='Daily Returns',
        marker_color='lightblue',
        opacity=0.7,
        histnorm='probability density'
    ))

    # Normal distribution overlay
    import numpy as np
    from scipy import stats

    mean = returns.mean()
    std = returns.std()
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = stats.norm.pdf(x_range, mean, std)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))

    # VaR 95% line
    var_95 = returns.quantile(0.05)
    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.2f}%",
        annotation_position="top"
    )

    fig.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Probability Density",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig


def create_tax_payment_chart(tax_payments: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing tax payments over time.

    Args:
        tax_payments: DataFrame with columns: Date, Type, Amount

    Returns:
        Plotly Figure object
    """
    if tax_payments.empty:
        return go.Figure()

    # Group by year and type
    tax_payments['Year'] = pd.to_datetime(tax_payments['Date']).dt.year
    annual_taxes = tax_payments.groupby(['Year', 'Type'])['Amount'].sum().reset_index()

    # Pivot for stacked bar
    pivot_df = annual_taxes.pivot(index='Year', columns='Type', values='Amount').fillna(0)

    fig = go.Figure()

    # Add trace for each tax type
    for tax_type in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[tax_type],
            name=tax_type,
            hovertemplate=f'<b>{tax_type}</b><br>Year: %{{x}}<br>Tax: $%{{y:,.2f}}<extra></extra>'
        ))

    fig.update_layout(
        title="Annual Tax Payments by Type",
        xaxis_title="Year",
        yaxis_title="Tax Paid ($)",
        barmode='stack',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def create_pre_post_tax_comparison(results: dict) -> go.Figure:
    """
    Create dual-line chart comparing pre-tax and post-tax portfolio values.

    Args:
        results: Dictionary containing backtest results with 'daily_values' and 'tax_payments'

    Returns:
        Plotly Figure object
    """
    daily_values = results.get('daily_values', pd.DataFrame())
    tax_payments = results.get('tax_payments', pd.DataFrame())

    if daily_values.empty:
        return go.Figure()

    fig = go.Figure()

    # Pre-tax (portfolio values as-is)
    fig.add_trace(go.Scatter(
        x=daily_values.index,
        y=daily_values['Value'],
        mode='lines',
        name='Pre-Tax Value',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Pre-Tax</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))

    # Post-tax (subtract cumulative taxes)
    if not tax_payments.empty:
        tax_payments['Date'] = pd.to_datetime(tax_payments['Date'])
        cumulative_taxes = tax_payments.groupby('Date')['Amount'].sum().cumsum()

        # Align with daily values
        post_tax_values = daily_values.copy()
        for date, cum_tax in cumulative_taxes.items():
            if date in post_tax_values.index:
                post_tax_values.loc[date:, 'Value'] -= cum_tax

        fig.add_trace(go.Scatter(
            x=post_tax_values.index,
            y=post_tax_values['Value'],
            mode='lines',
            name='Post-Tax Value',
            line=dict(color='orange', width=2, dash='dash'),
            hovertemplate='<b>Post-Tax</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title="Pre-Tax vs Post-Tax Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_price_chart_with_ema(
    price_data: pd.DataFrame,
    title: str = "Stock Price with EMA"
) -> go.Figure:
    """
    Create price chart with EMA lines (5, 10, 20, 40, 120, 200)

    Args:
        price_data: DataFrame with Close prices
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add candlestick chart
    if all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Price',
            showlegend=True
        ))
    else:
        # Fallback to line chart if OHLC data not available
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data['Close'],
            name='Price',
            line=dict(color='black', width=1.5)
        ))

    # Calculate and add EMA lines
    ema_periods = [
        (5, 'pink', 'EMA5'),
        (10, 'lightgreen', 'EMA10'),
        (20, 'blue', 'EMA20'),
        (40, 'red', 'EMA40'),
        (120, 'yellow', 'EMA120'),
        (200, 'green', 'EMA200')
    ]

    for period, color, name in ema_periods:
        ema = price_data['Close'].ewm(span=period, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=ema,
            name=name,
            line=dict(color=color, width=1.5),
            hoverinfo='y'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis=dict(
            rangeslider=dict(visible=False)
        )
    )

    return fig


def create_yield_chart_with_stats(
    yield_data: pd.Series,
    title: str = "Dividend Yield History"
) -> go.Figure:
    """
    Create dividend yield chart with statistical reference lines

    Args:
        yield_data: Series with dividend yield values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add yield line
    fig.add_trace(go.Scatter(
        x=yield_data.index,
        y=yield_data.values * 100,
        name='Dividend Yield',
        line=dict(color='orange', width=2),
        hovertemplate='<b>Yield</b><br>Date: %{x}<br>Yield: %{y:.2f}%<extra></extra>'
    ))

    # Calculate statistics
    valid_yields = yield_data[yield_data > 0]
    if len(valid_yields) > 0:
        mean_yield = valid_yields.mean() * 100
        median_yield = valid_yields.median() * 100
        min_yield = valid_yields.min() * 100
        max_yield = valid_yields.max() * 100

        # Add horizontal lines for statistics
        fig.add_hline(
            y=mean_yield,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Average: {mean_yield:.2f}%",
            annotation_position="right"
        )

        fig.add_hline(
            y=median_yield,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {median_yield:.2f}%",
            annotation_position="right"
        )

        fig.add_hline(
            y=min_yield,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Min: {min_yield:.2f}%",
            annotation_position="right"
        )

        fig.add_hline(
            y=max_yield,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Max: {max_yield:.2f}%",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Dividend Yield (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True
    )

    return fig
