"""
Stock Details Page (Korean Version)
Deep dive analysis for individual stocks with visualizations
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from utils.cache_manager import load_main_dataframe, load_historical_prices
from modules.visualization import (
    create_price_chart_with_ema,
    create_yield_chart_with_stats,
    create_dividend_history_bar
)

st.set_page_config(page_title="종목 상세 분석", page_icon="🔍", layout="wide")

st.title("🔍 종목 상세 분석")
st.markdown("개별 종목의 배당 분석을 심층 탐구하고 과거 데이터와 시각화를 제공합니다.")

# Load data
df = load_main_dataframe(use_cached=True)

if df is None:
    st.error("데이터가 없습니다. 홈페이지로 돌아가 데이터를 로드하세요.")
    st.stop()

# Stock selector
st.subheader("종목 선택")

# Get available symbols
available_symbols = sorted(df['Symbol'].unique().tolist()) if 'Symbol' in df.columns else []

if not available_symbols:
    st.error("사용 가능한 종목이 없습니다")
    st.stop()

selected_symbol = st.selectbox(
    "분석할 종목 선택",
    options=available_symbols,
    index=0
)

# Get stock data
stock_data = df[df['Symbol'] == selected_symbol].iloc[0] if len(df[df['Symbol'] == selected_symbol]) > 0 else None

if stock_data is None:
    st.error(f"{selected_symbol} 데이터를 찾을 수 없습니다")
    st.stop()

st.divider()

# Display key metrics
st.subheader(f"{selected_symbol} - {stock_data.get('Company Name', 'N/A')}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("배당률", f"{stock_data.get('Div. Yield', 0) * 100:.2f}%")
    st.metric("연간 배당금", f"${stock_data.get('Div. ($)', 0):.2f}")

with col2:
    st.metric("배당성향", f"{stock_data.get('Payout Ratio', 0) * 100:.1f}%")
    st.metric("배당 지급 연수", f"{int(stock_data.get('Years', 0))}")

with col3:
    st.metric("1년 성장률", f"{stock_data.get('Div. Growth', 0) * 100:.2f}%")
    st.metric("5년 CAGR", f"{stock_data.get('Div. Growth 5Y', 0) * 100:.2f}%")

# Financial Health Metrics
st.markdown("---")
st.subheader("재무 건전성 지표")

col4, col5, col6 = st.columns(3)

with col4:
    fcf_ratio = stock_data.get('FCF_Dividend_Ratio', 0)
    if fcf_ratio > 0:
        st.metric(
            "FCF/배당 비율",
            f"{fcf_ratio:.2f}x",
            help="잉여현금흐름을 총 배당금으로 나눈 값. >1.0이면 FCF가 배당을 완전히 커버함."
        )
    else:
        st.metric("FCF/배당 비율", "N/A", help="데이터 없음")

with col5:
    debt_to_equity = stock_data.get('Debt_to_Equity', 0)
    if debt_to_equity >= 0:
        st.metric(
            "부채비율 (D/E)",
            f"{debt_to_equity:.2f}",
            help="총 부채를 주주 자본으로 나눈 값. 낮을수록 재무 레버리지가 적음."
        )
    else:
        st.metric("부채비율 (D/E)", "N/A", help="데이터 없음")

with col6:
    roe = stock_data.get('ROE', 0)
    if roe != 0:
        st.metric(
            "ROE",
            f"{roe:.2f}%",
            help="자기자본이익률. 수익성 측정 - 자본 1달러당 얼마나 많은 이익을 창출하는지."
        )
    else:
        st.metric("ROE", "N/A", help="데이터 없음")

st.divider()

# Tabs for different analyses
tab1, tab2, tab3 = st.tabs([
    "📈 주가 & 배당률 히스토리",
    "💰 배당 히스토리",
    "ℹ️ 회사 정보"
])

# Fetch historical data
with st.spinner(f"{selected_symbol} 과거 데이터 로드 중..."):
    try:
        ticker = yf.Ticker(selected_symbol)
        hist_data = ticker.history(period="5y")
        dividends = ticker.dividends
        calendar = ticker.calendar
        info = ticker.info
    except Exception as e:
        st.error(f"데이터 가져오기 오류: {str(e)}")
        hist_data = None
        dividends = None
        calendar = None
        info = {}

with tab1:
    st.subheader("주가 & 배당 수익률 히스토리")

    # Period selector
    period = st.radio(
        "기간 선택",
        options=["1Y", "3Y", "5Y", "10Y", "Max"],
        index=2,
        horizontal=True
    )

    period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y", "Max": "max"}

    with st.spinner(f"{period} 데이터 로드 중..."):
        try:
            period_data = load_historical_prices(selected_symbol, period=period_map[period])

            if period_data is not None and len(period_data) > 0:
                # Price chart with EMA
                st.markdown("### 주가 및 EMA")
                price_fig = create_price_chart_with_ema(
                    period_data,
                    title=f"{selected_symbol} - 주가 및 EMA ({period})"
                )
                st.plotly_chart(price_fig, width='stretch')

                # Calculate dividend yield
                if len(dividends) > 0:
                    # Align dividends with price data
                    yield_series = pd.Series(index=period_data.index, dtype=float)

                    for date in period_data.index:
                        # Get last known dividend
                        recent_divs = dividends[dividends.index <= date]
                        if len(recent_divs) > 0:
                            last_div = recent_divs.iloc[-1]
                            # Annualize (assuming quarterly)
                            annual_div = last_div * 4
                            yield_series[date] = annual_div / period_data.loc[date, 'Close']
                        else:
                            yield_series[date] = 0

                    # Dividend yield chart with statistics
                    st.markdown("### 배당률 및 통계")
                    yield_fig = create_yield_chart_with_stats(
                        yield_series,
                        title=f"{selected_symbol} - 배당률 ({period})"
                    )
                    st.plotly_chart(yield_fig, width='stretch')
                else:
                    st.info("배당률 계산을 위한 배당 데이터가 없습니다")
            else:
                st.warning(f"{period} 주가 데이터가 없습니다")

        except Exception as e:
            st.error(f"기간 데이터 로드 오류: {str(e)}")

with tab2:
    st.subheader("배당금 지급 히스토리")

    # Show upcoming dividend dates if available
    if calendar and isinstance(calendar, dict):
        ex_div_date = calendar.get('Ex-Dividend Date')
        div_date = calendar.get('Dividend Date')

        if ex_div_date or div_date:
            st.markdown("#### 예정된 배당 정보")
            col1, col2 = st.columns(2)

            with col1:
                if ex_div_date:
                    st.metric("다음 배당락일", ex_div_date.strftime('%Y-%m-%d'))

            with col2:
                if div_date:
                    st.metric("다음 지급일", div_date.strftime('%Y-%m-%d'))

            st.divider()

    if dividends is not None and len(dividends) > 0:
        # Bar chart of annual dividends
        fig = create_dividend_history_bar(dividends)
        st.plotly_chart(fig, width='stretch')

        # Dividend payment table
        st.subheader("과거 배당금 지급 내역")
        recent_divs = dividends.tail(20).sort_index(ascending=False)
        div_df = pd.DataFrame({
            'Date': recent_divs.index.strftime('%Y-%m-%d'),
            'Dividend ($)': recent_divs.values.round(4)
        })
        st.dataframe(div_df, width='stretch', hide_index=True)
    else:
        st.info("배당금 지급 내역이 없습니다")

with tab3:
    st.subheader("회사 정보")

    if info:
        # Display company details
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**섹터:**")
            st.write(info.get('sector', 'N/A'))

            st.markdown("**산업:**")
            st.write(info.get('industry', 'N/A'))

            st.markdown("**웹사이트:**")
            website = info.get('website', '')
            if website:
                st.markdown(f"[{website}]({website})")
            else:
                st.write("N/A")

        with col2:
            st.markdown("**시가총액:**")
            market_cap = info.get('marketCap', 0)
            if market_cap > 1e9:
                st.write(f"${market_cap / 1e9:.2f}B")
            elif market_cap > 1e6:
                st.write(f"${market_cap / 1e6:.2f}M")
            else:
                st.write("N/A")

            st.markdown("**직원 수:**")
            st.write(f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "N/A")

            st.markdown("**거래소:**")
            st.write(info.get('exchange', 'N/A'))

        # Company description
        st.divider()
        st.markdown("**사업 설명:**")
        description = info.get('longBusinessSummary', info.get('description', '설명 없음'))
        st.write(description)

    else:
        st.info("회사 정보가 없습니다")

# Footer note
st.divider()
st.caption("데이터 제공: Yahoo Finance. 정보가 지연될 수 있습니다.")
