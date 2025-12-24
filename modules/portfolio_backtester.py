"""
Portfolio Backtest Simulator Module

This module provides comprehensive portfolio backtesting functionality including:
- Historical performance simulation
- DRIP (Dividend Reinvestment Plan) modeling
- Tax impact calculation
- Advanced risk metrics (Sharpe, Sortino, MDD, Beta, Alpha, VaR)
- Benchmark comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yfinance as yf
import streamlit as st


class PortfolioBacktester:
    """
    Comprehensive portfolio backtesting engine with DRIP and tax modeling.
    """

    def __init__(
        self,
        stocks: List[str],
        weights: Dict[str, float],
        start_date: str,
        end_date: str,
        initial_investment: float,
        monthly_contribution: float = 0
    ):
        """
        Initialize the portfolio backtester.

        Args:
            stocks: List of stock symbols
            weights: Dictionary mapping symbols to portfolio weights (must sum to 1.0)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_investment: Initial investment amount in dollars
            monthly_contribution: Monthly contribution amount in dollars
        """
        self.stocks = stocks
        self.weights = weights
        self.start_date = start_date
        self.end_date = end_date
        self.initial_investment = initial_investment
        self.monthly_contribution = monthly_contribution

        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.001):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.buyhold_data: Optional[pd.DataFrame] = None
        self.schd_data: Optional[pd.DataFrame] = None

    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price and dividend data for all stocks.

        Returns:
            Dictionary mapping symbols to DataFrames with Date index,
            columns: Close, Dividends
        """
        for symbol in self.stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=self.start_date, end=self.end_date)

                if hist.empty:
                    st.warning(f"No data available for {symbol}")
                    continue

                # Get dividend data
                dividends = ticker.dividends
                hist['Dividends'] = 0.0

                # Merge dividends into history
                for div_date, div_amount in dividends.items():
                    if div_date in hist.index:
                        hist.loc[div_date, 'Dividends'] = div_amount

                self.historical_data[symbol] = hist[['Close', 'Dividends']]

            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                continue

        return self.historical_data

    def fetch_benchmark_data(self, benchmark: str = 'SPY') -> pd.DataFrame:
        """
        Fetch benchmark (S&P 500) historical data.

        Args:
            benchmark: Benchmark ticker symbol (default: SPY)

        Returns:
            DataFrame with Date index and Close, Dividends columns
        """
        try:
            ticker = yf.Ticker(benchmark)
            hist = ticker.history(start=self.start_date, end=self.end_date)

            # Get dividend data
            dividends = ticker.dividends
            hist['Dividends'] = 0.0

            # Merge dividends
            for div_date, div_amount in dividends.items():
                if div_date in hist.index:
                    hist.loc[div_date, 'Dividends'] = div_amount

            self.benchmark_data = hist[['Close', 'Dividends']]

        except Exception as e:
            st.error(f"Error fetching benchmark data: {str(e)}")
            self.benchmark_data = pd.DataFrame()

        return self.benchmark_data

    def fetch_schd_data(self) -> pd.DataFrame:
        """
        Fetch SCHD ETF historical data.

        Returns:
            DataFrame with Date index and Close, Dividends columns
        """
        try:
            ticker = yf.Ticker('SCHD')
            hist = ticker.history(start=self.start_date, end=self.end_date)

            # Get dividend data
            dividends = ticker.dividends
            hist['Dividends'] = 0.0

            # Merge dividends
            for div_date, div_amount in dividends.items():
                if div_date in hist.index:
                    hist.loc[div_date, 'Dividends'] = div_amount

            self.schd_data = hist[['Close', 'Dividends']]

        except Exception as e:
            st.warning(f"SCHD data unavailable: {str(e)}")
            self.schd_data = pd.DataFrame()

        return self.schd_data

    def run_backtest(
        self,
        drip_enabled: bool = True,
        drip_fee: float = 0.0,
        tax_config: Optional[Dict[str, float]] = None,
        rebalancing_frequency: str = "No Rebalancing",
        rebalancing_fee: float = 0.0
    ) -> Dict:
        """
        Run comprehensive backtest simulation.

        Args:
            drip_enabled: Enable dividend reinvestment
            drip_fee: DRIP fee percentage (e.g., 0.01 for 1%)
            tax_config: Dictionary with tax rates:
                - qualified_dividend_rate
                - ordinary_dividend_rate
                - long_term_capital_gains_rate
                - short_term_capital_gains_rate
            rebalancing_frequency: Rebalancing frequency
                ("No Rebalancing", "Monthly", "Quarterly", "Semi-Annually", "Annually")
            rebalancing_fee: Trading fee percentage for rebalancing (e.g., 0.001 for 0.1%)

        Returns:
            Dictionary containing:
                - daily_values: DataFrame with portfolio values over time
                - daily_values_no_drip: DataFrame without DRIP for comparison
                - benchmark_values: DataFrame with benchmark values
                - dividend_history: DataFrame with dividend payments
                - tax_payments: DataFrame with tax payment history
                - holdings: DataFrame with final holdings
                - metrics: Dictionary of performance metrics
        """
        if not self.historical_data:
            self.fetch_historical_data()

        if self.benchmark_data is None:
            self.fetch_benchmark_data()

        # Initialize holdings (shares owned)
        holdings = {symbol: 0.0 for symbol in self.stocks}
        holdings_no_drip = {symbol: 0.0 for symbol in self.stocks}

        # Tax-lot tracking for capital gains
        tax_lots = {symbol: [] for symbol in self.stocks}

        # Get all trading days (union of all stock trading days)
        all_dates = set()
        for df in self.historical_data.values():
            all_dates.update(df.index)
        trading_days = sorted(list(all_dates))

        # Results storage
        daily_values = []
        daily_values_no_drip = []
        dividend_history = []
        tax_payments = []

        # Track cash available
        cash = self.initial_investment
        cash_no_drip = self.initial_investment

        # Initial purchase on first day
        first_day = True
        last_month = None
        last_rebalance_date = None
        rebalancing_history = []

        for date in trading_days:
            # Monthly contribution (on first trading day of each month)
            current_month = date.tz_localize(None).to_period('M')
            if last_month is not None and current_month != last_month:
                cash += self.monthly_contribution
                cash_no_drip += self.monthly_contribution
            last_month = current_month

            # Initial purchase or monthly rebalancing
            if first_day:
                # Buy initial positions
                for symbol in self.stocks:
                    if symbol in self.historical_data and date in self.historical_data[symbol].index:
                        price = self.historical_data[symbol].loc[date, 'Close']
                        allocation = self.initial_investment * self.weights[symbol]
                        shares = allocation / price
                        holdings[symbol] = shares
                        holdings_no_drip[symbol] = shares

                        # Record tax lot
                        tax_lots[symbol].append({
                            'date': date,
                            'shares': shares,
                            'cost_basis': price
                        })

                cash = 0
                cash_no_drip = 0
                first_day = False
                last_rebalance_date = date  # Track initial purchase as baseline

            # Check if rebalancing is needed
            if self._should_rebalance(date, last_rebalance_date, rebalancing_frequency):
                # Get current prices
                current_prices = {}
                for symbol in self.stocks:
                    if symbol in self.historical_data and date in self.historical_data[symbol].index:
                        current_prices[symbol] = self.historical_data[symbol].loc[date, 'Close']

                # Rebalance portfolio
                holdings, rebal_fees, rebal_taxes = self._rebalance_portfolio(
                    holdings,
                    current_prices,
                    self.weights,
                    rebalancing_fee,
                    tax_config,
                    tax_lots
                )

                # Deduct fees and taxes from cash
                cash -= (rebal_fees + rebal_taxes)

                # Record rebalancing event
                rebalancing_history.append({
                    'Date': date,
                    'Fees': rebal_fees,
                    'Taxes': rebal_taxes
                })

                if tax_config and rebal_taxes > 0:
                    tax_payments.append({
                        'Date': date,
                        'Symbol': 'Portfolio',
                        'Type': 'Capital Gains (Rebalancing)',
                        'Amount': rebal_taxes
                    })

                last_rebalance_date = date

            # Process dividends for the day
            total_dividends = 0
            total_dividends_no_drip = 0

            for symbol in self.stocks:
                if symbol not in self.historical_data:
                    continue

                if date not in self.historical_data[symbol].index:
                    continue

                dividend_per_share = self.historical_data[symbol].loc[date, 'Dividends']

                if dividend_per_share > 0:
                    # Calculate dividend payment
                    dividend_amount = holdings[symbol] * dividend_per_share
                    dividend_amount_no_drip = holdings_no_drip[symbol] * dividend_per_share

                    # Apply taxes if configured
                    dividend_after_tax = dividend_amount
                    dividend_after_tax_no_drip = dividend_amount_no_drip

                    if tax_config:
                        tax_amount = self._calculate_taxes(
                            dividend_amount,
                            holding_period=365,  # Assume qualified
                            tax_config=tax_config
                        )
                        dividend_after_tax = dividend_amount - tax_amount

                        tax_amount_no_drip = self._calculate_taxes(
                            dividend_amount_no_drip,
                            holding_period=365,
                            tax_config=tax_config
                        )
                        dividend_after_tax_no_drip = dividend_amount_no_drip - tax_amount_no_drip

                        # Record tax payment
                        tax_payments.append({
                            'Date': date,
                            'Symbol': symbol,
                            'Type': 'Dividend',
                            'Amount': tax_amount
                        })

                    # DRIP: Reinvest dividends
                    if drip_enabled:
                        current_price = self.historical_data[symbol].loc[date, 'Close']
                        new_shares = self._apply_drip(
                            dividend_after_tax,
                            current_price,
                            drip_fee
                        )
                        holdings[symbol] += new_shares

                        # Record tax lot for DRIP purchase
                        if new_shares > 0:
                            tax_lots[symbol].append({
                                'date': date,
                                'shares': new_shares,
                                'cost_basis': current_price
                            })
                    else:
                        cash += dividend_after_tax

                    # No DRIP version
                    cash_no_drip += dividend_after_tax_no_drip

                    total_dividends += dividend_amount
                    total_dividends_no_drip += dividend_amount_no_drip

                    # Record dividend
                    dividend_history.append({
                        'Date': date,
                        'Symbol': symbol,
                        'Amount': dividend_amount,
                        'Shares': holdings[symbol]
                    })

            # Calculate portfolio value for the day
            portfolio_value = cash
            portfolio_value_no_drip = cash_no_drip

            for symbol in self.stocks:
                if symbol in self.historical_data and date in self.historical_data[symbol].index:
                    price = self.historical_data[symbol].loc[date, 'Close']
                    portfolio_value += holdings[symbol] * price
                    portfolio_value_no_drip += holdings_no_drip[symbol] * price

            daily_values.append({
                'Date': date,
                'Value': portfolio_value,
                'Cash': cash
            })

            daily_values_no_drip.append({
                'Date': date,
                'Value': portfolio_value_no_drip,
                'Cash': cash_no_drip
            })

        # Calculate benchmark values
        benchmark_values = self._calculate_benchmark_returns(
            self.benchmark_data,
            self.initial_investment,
            self.monthly_contribution,
            drip_enabled
        )

        # Calculate Buy & Hold benchmark (no rebalancing)
        buyhold_values = self._calculate_buyhold_returns(
            holdings_no_drip if not drip_enabled else holdings,
            self.initial_investment,
            self.monthly_contribution,
            drip_enabled
        )

        # Calculate SCHD benchmark
        schd_values = self._calculate_benchmark_returns(
            self.schd_data if self.schd_data is not None else pd.DataFrame(),
            self.initial_investment,
            self.monthly_contribution,
            drip_enabled
        )

        # Convert to DataFrames
        daily_values_df = pd.DataFrame(daily_values).set_index('Date')
        daily_values_no_drip_df = pd.DataFrame(daily_values_no_drip).set_index('Date')
        dividend_history_df = pd.DataFrame(dividend_history)
        tax_payments_df = pd.DataFrame(tax_payments) if tax_payments else pd.DataFrame()
        rebalancing_history_df = pd.DataFrame(rebalancing_history) if rebalancing_history else pd.DataFrame()

        # Calculate final holdings
        final_holdings = []
        for symbol in self.stocks:
            if symbol in self.historical_data:
                last_date = self.historical_data[symbol].index[-1]
                last_price = self.historical_data[symbol].loc[last_date, 'Close']
                total_dividends_received = dividend_history_df[
                    dividend_history_df['Symbol'] == symbol
                ]['Amount'].sum() if not dividend_history_df.empty else 0

                final_holdings.append({
                    'Symbol': symbol,
                    'Shares': holdings[symbol],
                    'Current Price': last_price,
                    'Market Value': holdings[symbol] * last_price,
                    'Total Dividends': total_dividends_received
                })

        holdings_df = pd.DataFrame(final_holdings)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(
            daily_values_df,
            benchmark_values,
            dividend_history_df
        )

        return {
            'daily_values': daily_values_df,
            'daily_values_no_drip': daily_values_no_drip_df,
            'benchmark_values': benchmark_values,
            'buyhold_values': buyhold_values,
            'schd_values': schd_values,
            'dividend_history': dividend_history_df,
            'tax_payments': tax_payments_df,
            'rebalancing_history': rebalancing_history_df,
            'holdings': holdings_df,
            'metrics': metrics
        }

    def _apply_drip(
        self,
        dividend_amount: float,
        current_price: float,
        drip_fee: float
    ) -> float:
        """
        Apply DRIP to reinvest dividends.

        Args:
            dividend_amount: Dividend amount to reinvest
            current_price: Current stock price
            drip_fee: DRIP fee percentage

        Returns:
            Number of shares purchased (fractional shares supported)
        """
        if current_price <= 0:
            return 0.0

        # Deduct fee
        net_dividend = dividend_amount * (1 - drip_fee)

        # Calculate shares (fractional shares allowed)
        shares = net_dividend / current_price

        return shares

    def _calculate_taxes(
        self,
        dividend_amount: float,
        holding_period: int,
        tax_config: Dict[str, float]
    ) -> float:
        """
        Calculate taxes on dividends.

        Args:
            dividend_amount: Dividend amount
            holding_period: Days held (>365 = qualified)
            tax_config: Tax configuration dictionary

        Returns:
            Tax amount
        """
        # Assume qualified if held > 60 days
        if holding_period >= 60:
            tax_rate = tax_config.get('qualified_dividend_rate', 0.15)
        else:
            tax_rate = tax_config.get('ordinary_dividend_rate', 0.22)

        return dividend_amount * tax_rate

    def _calculate_benchmark_returns(
        self,
        benchmark_data: pd.DataFrame,
        initial_investment: float,
        monthly_contribution: float,
        drip_enabled: bool
    ) -> pd.DataFrame:
        """
        Calculate benchmark portfolio returns.

        Args:
            benchmark_data: Benchmark price/dividend data
            initial_investment: Initial investment
            monthly_contribution: Monthly contribution
            drip_enabled: Enable DRIP for benchmark

        Returns:
            DataFrame with Date index and Value column
        """
        if benchmark_data.empty:
            return pd.DataFrame()

        shares = 0.0
        cash = initial_investment
        values = []
        last_month = None

        for date in benchmark_data.index:
            # Monthly contribution
            current_month = date.tz_localize(None).to_period('M')
            if last_month is not None and current_month != last_month:
                cash += monthly_contribution
            last_month = current_month

            # Initial purchase
            if shares == 0:
                price = benchmark_data.loc[date, 'Close']
                shares = initial_investment / price
                cash = 0

            # Process dividends
            dividend_per_share = benchmark_data.loc[date, 'Dividends']
            if dividend_per_share > 0:
                dividend_amount = shares * dividend_per_share

                if drip_enabled:
                    price = benchmark_data.loc[date, 'Close']
                    new_shares = dividend_amount / price
                    shares += new_shares
                else:
                    cash += dividend_amount

            # Calculate value
            price = benchmark_data.loc[date, 'Close']
            value = shares * price + cash

            values.append({'Date': date, 'Value': value})

        return pd.DataFrame(values).set_index('Date')

    def _calculate_buyhold_returns(
        self,
        initial_holdings: Dict[str, float],
        initial_investment: float,
        monthly_contribution: float,
        drip_enabled: bool
    ) -> pd.DataFrame:
        """
        Calculate buy and hold portfolio returns (initial allocation, no rebalancing).

        Args:
            initial_holdings: Initial holdings dictionary
            initial_investment: Initial investment
            monthly_contribution: Monthly contribution
            drip_enabled: Enable DRIP for buy & hold

        Returns:
            DataFrame with Date index and Value column
        """
        if not self.historical_data:
            return pd.DataFrame()

        # Get all trading days
        all_dates = set()
        for df in self.historical_data.values():
            all_dates.update(df.index)
        trading_days = sorted(list(all_dates))

        # Initialize holdings with same initial allocation as main portfolio
        holdings = {symbol: 0.0 for symbol in self.stocks}
        cash = initial_investment

        # Buy initial positions on first day
        first_day = True
        values = []
        last_month = None

        for date in trading_days:
            # Monthly contribution
            current_month = date.tz_localize(None).to_period('M')
            if last_month is not None and current_month != last_month:
                cash += monthly_contribution
            last_month = current_month

            # Initial purchase on first day
            if first_day:
                for symbol in self.stocks:
                    if symbol in self.historical_data and date in self.historical_data[symbol].index:
                        price = self.historical_data[symbol].loc[date, 'Close']
                        allocation = initial_investment * self.weights[symbol]
                        shares = allocation / price
                        holdings[symbol] = shares

                cash = 0
                first_day = False

            # Process dividends (reinvest if DRIP enabled)
            for symbol in self.stocks:
                if symbol not in self.historical_data:
                    continue

                if date not in self.historical_data[symbol].index:
                    continue

                dividend_per_share = self.historical_data[symbol].loc[date, 'Dividends']

                if dividend_per_share > 0:
                    dividend_amount = holdings[symbol] * dividend_per_share

                    if drip_enabled:
                        # Reinvest dividends
                        current_price = self.historical_data[symbol].loc[date, 'Close']
                        new_shares = dividend_amount / current_price
                        holdings[symbol] += new_shares
                    else:
                        cash += dividend_amount

            # Calculate portfolio value
            portfolio_value = cash
            for symbol in self.stocks:
                if symbol in self.historical_data and date in self.historical_data[symbol].index:
                    price = self.historical_data[symbol].loc[date, 'Close']
                    portfolio_value += holdings[symbol] * price

            values.append({'Date': date, 'Value': portfolio_value})

        return pd.DataFrame(values).set_index('Date')

    def calculate_performance_metrics(
        self,
        daily_values: pd.DataFrame,
        benchmark_values: pd.DataFrame,
        dividend_history: pd.DataFrame = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            daily_values: Portfolio values DataFrame
            benchmark_values: Benchmark values DataFrame
            dividend_history: DataFrame with dividend payment history

        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        returns = daily_values['Value'].pct_change().dropna()

        # Basic metrics
        final_value = daily_values['Value'].iloc[-1]
        total_return = (final_value / self.initial_investment - 1) * 100

        # CAGR
        years = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days / 365.25
        cagr = (pow(final_value / self.initial_investment, 1 / years) - 1) * 100 if years > 0 else 0

        # Annualized return
        annualized_return = returns.mean() * 252 * 100

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_mdd(daily_values['Value'])
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # VaR
        var_95 = self._calculate_var(returns)

        # Beta and Alpha
        beta, alpha = 0, 0
        if not benchmark_values.empty:
            benchmark_returns = benchmark_values['Value'].pct_change().dropna()
            # Align dates
            aligned_returns = pd.concat([returns, benchmark_returns], axis=1, join='inner')
            if len(aligned_returns) > 0:
                beta, alpha = self._calculate_beta_alpha(
                    aligned_returns.iloc[:, 0],
                    aligned_returns.iloc[:, 1]
                )

        # Benchmark comparison
        benchmark_return = 0
        if not benchmark_values.empty:
            benchmark_final = benchmark_values['Value'].iloc[-1]
            benchmark_return = (benchmark_final / self.initial_investment - 1) * 100

        outperformance = total_return - benchmark_return

        # Win rate (percentage of positive return days)
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

        # Dividend metrics - calculate from dividend_history
        total_dividends = 0
        annual_dividend_income = 0

        if dividend_history is not None and not dividend_history.empty:
            # Total dividends received over entire period
            total_dividends = dividend_history['Amount'].sum()

            # Annual dividend income (annualized from total period)
            if years > 0:
                annual_dividend_income = total_dividends / years
            else:
                annual_dividend_income = 0

        return {
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'var_95': var_95,
            'beta': beta,
            'alpha': alpha,
            'benchmark_return': benchmark_return,
            'outperformance': outperformance,
            'win_rate': win_rate,
            'total_dividends': total_dividends,
            'annual_dividend_income': annual_dividend_income
        }

    def _calculate_mdd(self, values: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            values: Series of portfolio values

        Returns:
            Maximum drawdown as percentage
        """
        # Calculate running maximum
        running_max = values.expanding().max()

        # Calculate drawdown
        drawdown = (values - running_max) / running_max * 100

        # Return maximum drawdown (most negative)
        return drawdown.min()

    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe Ratio
        """
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        # Excess returns
        excess_returns = returns - daily_rf

        # Sharpe ratio
        if excess_returns.std() == 0:
            return 0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        return sharpe

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino Ratio (downside deviation only).

        Args:
            returns: Series of daily returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino Ratio
        """
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252

        # Excess returns
        excess_returns = returns - daily_rf

        # Downside returns (negative only)
        downside_returns = excess_returns[excess_returns < 0]

        # Downside deviation
        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0

        sortino = excess_returns.mean() / downside_std * np.sqrt(252)

        return sortino

    def _calculate_beta_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Beta and Alpha.

        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns

        Returns:
            Tuple of (beta, alpha)
        """
        # Covariance and variance
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        # Beta
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

        # Alpha (annualized)
        risk_free_rate = 0.02
        portfolio_return = portfolio_returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252

        alpha = (portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))) * 100

        return beta, alpha

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of daily returns
            confidence: Confidence level (default 95%)

        Returns:
            VaR as percentage
        """
        # Calculate percentile
        var = returns.quantile(1 - confidence) * 100

        return var

    def _should_rebalance(
        self,
        date: pd.Timestamp,
        last_rebalance_date: Optional[pd.Timestamp],
        rebalancing_frequency: str
    ) -> bool:
        """
        Determine if portfolio should be rebalanced on given date.

        Args:
            date: Current date
            last_rebalance_date: Last rebalancing date (None if never rebalanced)
            rebalancing_frequency: Rebalancing frequency setting

        Returns:
            True if should rebalance, False otherwise
        """
        if rebalancing_frequency == "No Rebalancing":
            return False

        if last_rebalance_date is None:
            return False  # Initial purchase is not a rebalance

        if rebalancing_frequency == "Monthly":
            # Rebalance on first trading day of each month
            return date.month != last_rebalance_date.month

        elif rebalancing_frequency == "Quarterly":
            # Rebalance every 3 months
            months_diff = (date.year - last_rebalance_date.year) * 12 + (date.month - last_rebalance_date.month)
            return months_diff >= 3

        elif rebalancing_frequency == "Semi-Annually":
            # Rebalance every 6 months
            months_diff = (date.year - last_rebalance_date.year) * 12 + (date.month - last_rebalance_date.month)
            return months_diff >= 6

        elif rebalancing_frequency == "Annually":
            # Rebalance once a year
            return date.year != last_rebalance_date.year

        return False

    def _rebalance_portfolio(
        self,
        holdings: Dict[str, float],
        current_prices: Dict[str, float],
        target_weights: Dict[str, float],
        rebalancing_fee: float,
        tax_config: Optional[Dict[str, float]],
        tax_lots: Dict[str, list]
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Rebalance portfolio to target weights.

        Args:
            holdings: Current holdings (shares per symbol)
            current_prices: Current prices per symbol
            target_weights: Target allocation weights
            rebalancing_fee: Fee percentage per trade
            tax_config: Tax configuration
            tax_lots: Tax lot tracking for capital gains

        Returns:
            Tuple of (new_holdings, total_fees, total_taxes)
        """
        # Calculate current portfolio value
        total_value = sum(holdings[symbol] * current_prices.get(symbol, 0)
                         for symbol in holdings.keys())

        if total_value <= 0:
            return holdings, 0.0, 0.0

        # Calculate target values and current values
        target_values = {symbol: total_value * target_weights.get(symbol, 0)
                        for symbol in holdings.keys()}
        current_values = {symbol: holdings[symbol] * current_prices.get(symbol, 0)
                         for symbol in holdings.keys()}

        # Determine trades needed
        new_holdings = holdings.copy()
        total_fees = 0.0
        total_taxes = 0.0

        for symbol in holdings.keys():
            current_value = current_values[symbol]
            target_value = target_values[symbol]
            price = current_prices.get(symbol, 0)

            if price <= 0:
                continue

            value_diff = target_value - current_value

            if abs(value_diff) > 1:  # Only trade if difference > $1
                # Calculate shares to trade
                shares_to_trade = value_diff / price

                if shares_to_trade > 0:
                    # Buying
                    fee = abs(value_diff) * rebalancing_fee
                    new_holdings[symbol] += shares_to_trade
                    total_fees += fee

                else:
                    # Selling
                    shares_to_sell = abs(shares_to_trade)
                    fee = abs(value_diff) * rebalancing_fee
                    total_fees += fee

                    # Calculate capital gains tax if applicable
                    if tax_config and symbol in tax_lots:
                        capital_gains_tax = self._calculate_capital_gains_tax(
                            symbol,
                            shares_to_sell,
                            price,
                            tax_lots[symbol],
                            tax_config
                        )
                        total_taxes += capital_gains_tax

                    new_holdings[symbol] -= shares_to_sell

        return new_holdings, total_fees, total_taxes

    def _calculate_capital_gains_tax(
        self,
        symbol: str,
        shares_sold: float,
        sale_price: float,
        tax_lots: list,
        tax_config: Dict[str, float]
    ) -> float:
        """
        Calculate capital gains tax using FIFO method.

        Args:
            symbol: Stock symbol
            shares_sold: Number of shares sold
            sale_price: Sale price per share
            tax_lots: List of tax lots for this symbol
            tax_config: Tax configuration

        Returns:
            Total capital gains tax
        """
        if not tax_lots:
            return 0.0

        total_tax = 0.0
        shares_remaining = shares_sold

        for lot in tax_lots:
            if shares_remaining <= 0:
                break

            lot_shares = lot['shares']
            lot_cost_basis = lot['cost_basis']

            # Determine how many shares from this lot
            shares_from_lot = min(shares_remaining, lot_shares)

            # Calculate gain/loss
            proceeds = shares_from_lot * sale_price
            cost = shares_from_lot * lot_cost_basis
            capital_gain = proceeds - cost

            if capital_gain > 0:
                # Assume long-term if held > 365 days (simplified)
                tax_rate = tax_config.get('long_term_capital_gains_rate', 0.15)
                total_tax += capital_gain * tax_rate

            shares_remaining -= shares_from_lot

        return total_tax
