"""
Portfolio Backtest Simulator Page (Korean Version)

Comprehensive portfolio performance analysis with:
- DRIP (Dividend Reinvestment Plan) simulation
- Tax impact modeling
- Advanced risk metrics
- Benchmark comparison (S&P 500)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.portfolio_backtester import PortfolioBacktester
from modules.visualization import (
    create_portfolio_growth_chart,
    create_dividend_income_chart,
    create_cumulative_dividend_chart,
    create_underwater_chart,
    create_return_distribution_chart,
    create_tax_payment_chart,
    create_pre_post_tax_comparison
)
from utils.cache_manager import load_main_dataframe
from config import BacktestConfig

# Page configuration
st.set_page_config(
    page_title="포트폴리오 백테스트",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 포트폴리오 백테스트 시뮬레이터")
st.markdown("""
종합 포트폴리오 성과 분석 기능:
- **DRIP 시뮬레이션**: 소수점 주식으로 자동 배당 재투자
- **세금 모델링**: 적격 배당금 및 자본 이득 세금 계산
- **리스크 지표**: Sharpe, Sortino, MDD, Beta, Alpha, VaR 등
- **벤치마크 비교**: S&P 500 (SPY) 대비 비교
""")

st.divider()

# Load main dataframe
df = load_main_dataframe()

if df is None or df.empty:
    st.error("배당 데이터 로드 실패. 데이터 파일을 확인하세요.")
    st.stop()

# --- SIDEBAR: Portfolio Configuration ---
st.sidebar.header("1. 포트폴리오 구성")

# Stock selection
selected_stocks = st.sidebar.multiselect(
    "종목 선택 (최대 20개)",
    options=sorted(df['Symbol'].unique().tolist()),
    default=[],
    help="포트폴리오에 포함할 최대 20개 종목 선택"
)

if len(selected_stocks) > BacktestConfig.MAX_PORTFOLIO_STOCKS:
    st.sidebar.error(f"최대 {BacktestConfig.MAX_PORTFOLIO_STOCKS}개 종목만 허용됩니다!")
    selected_stocks = selected_stocks[:BacktestConfig.MAX_PORTFOLIO_STOCKS]

# Allocation method
allocation_method = st.sidebar.radio(
    "할당 방식",
    BacktestConfig.ALLOCATION_METHODS,
    help="선택한 종목에 투자금을 할당하는 방식 선택"
)

# Custom weights (if selected)
weights = {}
if allocation_method == "Custom Weight" and len(selected_stocks) > 0:
    st.sidebar.subheader("사용자 지정 할당")

    for stock in selected_stocks:
        weights[stock] = st.sidebar.slider(
            f"{stock} 비중 (%)",
            min_value=0,
            max_value=100,
            value=100 // len(selected_stocks),
            step=1
        ) / 100

    # Validate total weight
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=0.01):
        st.sidebar.warning(f"⚠️ 총 비중: {total_weight*100:.1f}% (100%여야 함)")
    else:
        st.sidebar.success(f"✓ 총 비중: {total_weight*100:.1f}%")

st.sidebar.divider()

# --- SIDEBAR: Backtest Settings ---
st.sidebar.header("2. 백테스트 설정")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "시작일",
        value=pd.to_datetime(BacktestConfig.DEFAULT_START_DATE),
        min_value=pd.to_datetime("2000-01-01"),
        max_value=pd.to_datetime("today")
    )

with col2:
    end_date = st.date_input(
        "종료일",
        value=pd.to_datetime("today"),
        min_value=start_date
    )

# Investment amounts
initial_investment = st.sidebar.number_input(
    "초기 투자금 ($)",
    min_value=1000,
    max_value=10000000,
    value=BacktestConfig.DEFAULT_INITIAL_INVESTMENT,
    step=1000,
    help="최초 일시불 투자금"
)

monthly_contribution = st.sidebar.number_input(
    "월간 적립금 ($)",
    min_value=0,
    max_value=100000,
    value=BacktestConfig.DEFAULT_MONTHLY_CONTRIBUTION,
    step=100,
    help="매월 투자할 금액"
)

# DRIP settings
drip_enabled = st.sidebar.checkbox(
    "DRIP 활성화",
    value=True,
    help="배당금을 자동으로 재투자하여 추가 주식 매수"
)

drip_fee = 0.0
if drip_enabled:
    drip_fee = st.sidebar.number_input(
        "DRIP 수수료 (%)",
        min_value=0.0,
        max_value=5.0,
        value=BacktestConfig.DEFAULT_DRIP_FEE,
        step=0.1,
        help="배당 재투자 시 부과되는 수수료 (일반적으로 0%)"
    ) / 100

# Tax settings
tax_enabled = st.sidebar.checkbox(
    "세금 영향 포함",
    value=False,
    help="배당 소득 및 자본 이득 세금의 영향 모델링"
)

tax_config = None
if tax_enabled:
    with st.sidebar.expander("⚙️ 세금 설정", expanded=False):
        tax_config = {
            'qualified_dividend_rate': st.number_input(
                "적격 배당금 세율 (%)",
                min_value=0.0,
                max_value=50.0,
                value=BacktestConfig.DEFAULT_QUALIFIED_DIVIDEND_TAX * 100,
                step=0.5,
                help="적격 배당금 세율 (60일 초과 보유)"
            ) / 100,
            'ordinary_dividend_rate': st.number_input(
                "일반 배당금 세율 (%)",
                min_value=0.0,
                max_value=50.0,
                value=BacktestConfig.DEFAULT_ORDINARY_DIVIDEND_TAX * 100,
                step=0.5,
                help="일반 배당금 세율"
            ) / 100,
            'long_term_capital_gains_rate': st.number_input(
                "장기 자본 이득 세율 (%)",
                min_value=0.0,
                max_value=50.0,
                value=BacktestConfig.DEFAULT_LONG_TERM_CAPITAL_GAINS_TAX * 100,
                step=0.5,
                help="1년 초과 보유 자산 세율"
            ) / 100
        }

st.sidebar.divider()

# --- REBALANCING SETTINGS ---
st.sidebar.header("3. 리밸런싱 전략")

rebalancing_frequency = st.sidebar.selectbox(
    "리밸런싱 빈도",
    options=BacktestConfig.REBALANCING_FREQUENCIES,
    index=0,
    help="목표 비중으로 포트폴리오를 재조정하는 빈도"
)

rebalancing_fee = 0.0
if rebalancing_frequency != "No Rebalancing":
    rebalancing_fee = st.sidebar.number_input(
        "리밸런싱 수수료 (%)",
        min_value=0.0,
        max_value=2.0,
        value=BacktestConfig.DEFAULT_REBALANCING_FEE * 100,
        step=0.01,
        help="리밸런싱 거래당 수수료 비율"
    ) / 100

    st.sidebar.info(f"📊 목표 할당 비중을 유지하기 위해 **{rebalancing_frequency.lower()}** 리밸런싱이 발생합니다.")

st.sidebar.divider()

# --- RUN BACKTEST BUTTON ---
run_backtest = st.sidebar.button(
    "🚀 백테스트 실행",
    type="primary",
    width='stretch'
)

# --- MAIN CONTENT AREA ---
if run_backtest:
    if len(selected_stocks) == 0:
        st.error("⚠️ 백테스트할 종목을 최소 1개 선택하세요.")
        st.stop()

    # Calculate weights based on allocation method
    if allocation_method == "Equal Weight":
        weights = {stock: 1/len(selected_stocks) for stock in selected_stocks}

    elif allocation_method == "Yield Weight":
        # Weight by dividend yield
        stock_data = df[df['Symbol'].isin(selected_stocks)].set_index('Symbol')
        yields = stock_data['Div. Yield'].fillna(0)
        total_yield = yields.sum()

        if total_yield > 0:
            weights = {stock: yields[stock]/total_yield for stock in selected_stocks}
        else:
            st.warning("배당률 데이터가 없어 동일 비중 사용")
            weights = {stock: 1/len(selected_stocks) for stock in selected_stocks}

    elif allocation_method == "Market Cap Weight":
        st.info("시가총액 비중은 추가 데이터가 필요합니다. 현재는 동일 비중 사용.")
        weights = {stock: 1/len(selected_stocks) for stock in selected_stocks}

    elif allocation_method == "Custom Weight":
        # Weights already defined above
        pass

    # Validate weights
    if not np.isclose(sum(weights.values()), 1.0, atol=0.01):
        st.error("비중 합계가 100%여야 합니다. 사용자 지정 비중을 조정하세요.")
        st.stop()

    # Run backtest
    with st.spinner("🔄 백테스트 실행 중... 대규모 포트폴리오는 10-30초 소요될 수 있습니다."):
        try:
            # Initialize backtester
            backtester = PortfolioBacktester(
                stocks=selected_stocks,
                weights=weights,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_investment=initial_investment,
                monthly_contribution=monthly_contribution
            )

            # Fetch data
            st.info("📥 Yahoo Finance에서 과거 데이터 가져오는 중...")
            backtester.fetch_historical_data()
            backtester.fetch_benchmark_data()
            backtester.fetch_schd_data()

            # Run backtest
            st.info("⚙️ 포트폴리오 성과 시뮬레이션 중...")
            results = backtester.run_backtest(
                drip_enabled=drip_enabled,
                drip_fee=drip_fee,
                tax_config=tax_config,
                rebalancing_frequency=rebalancing_frequency,
                rebalancing_fee=rebalancing_fee
            )

            # Store results in session state
            st.session_state['backtest_results'] = results
            st.session_state['backtest_params'] = {
                'stocks': selected_stocks,
                'weights': weights,
                'start_date': start_date,
                'end_date': end_date,
                'initial_investment': initial_investment,
                'monthly_contribution': monthly_contribution,
                'drip_enabled': drip_enabled,
                'tax_enabled': tax_enabled,
                'rebalancing_frequency': rebalancing_frequency,
                'rebalancing_fee': rebalancing_fee
            }

            st.success("✅ 백테스트 완료!")

        except Exception as e:
            st.error(f"❌ 백테스트 실행 오류: {str(e)}")
            st.exception(e)
            st.stop()

# --- DISPLAY RESULTS ---
if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    params = st.session_state.get('backtest_params', {})
    metrics = results['metrics']

    st.header("📈 성과 요약")

    # --- KEY METRICS (4 columns) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "최종 포트폴리오 가치",
            f"${metrics['final_value']:,.2f}",
            delta=f"{metrics['total_return']:.2f}%",
            help="백테스트 기간 종료 시점의 포트폴리오 총 가치. 델타는 초기 투자 대비 총 수익률을 표시합니다."
        )
        st.metric(
            "총 수령 배당금",
            f"${metrics['total_dividends']:,.2f}",
            help="전체 백테스트 기간 동안 받은 모든 배당금의 합계. DRIP가 활성화된 경우 이 배당금은 자동으로 재투자되었습니다."
        )

    with col2:
        st.metric(
            "연환산 수익률",
            f"{metrics['annualized_return']:.2f}%",
            help="일일 평균 수익률을 연간 기준으로 환산 (일일 수익률 × 252 거래일). 매년 평균적으로 기대할 수 있는 수익률을 보여줍니다."
        )
        st.metric(
            "연간 배당 소득",
            f"${metrics['annual_dividend_income']:,.2f}",
            help="연간 평균 배당 소득 (총 배당금 ÷ 연수). 배당금으로부터의 꾸준한 현금 흐름을 나타냅니다."
        )

    with col3:
        st.metric(
            "CAGR",
            f"{metrics['cagr']:.2f}%",
            help="복리 연평균 성장률: 투자가 안정적인 비율로 성장했다면 매년 성장하는 비율. 단순 연환산 수익률보다 더 정확합니다."
        )
        st.metric(
            "샤프 비율",
            f"{metrics['sharpe_ratio']:.2f}",
            help="위험 조정 수익률 지표 (수익률 ÷ 변동성). 높을수록 좋음. >1은 양호, >2는 매우 우수. 위험 단위당 초과 수익을 측정합니다."
        )

    with col4:
        st.metric(
            "최대 낙폭",
            f"{metrics['max_drawdown']:.2f}%",
            delta=None,
            delta_color="inverse",
            help="기간 중 최고점에서 최저점까지의 최대 하락폭. 경험했을 최악의 손실을 보여줍니다. 낮을수록 (덜 마이너스) 좋습니다."
        )
        st.metric(
            "소르티노 비율",
            f"{metrics['sortino_ratio']:.2f}",
            help="샤프 비율과 유사하지만 하방 변동성만 고려. 더 나은 위험 조정 수익률 측정치. 높은 값은 더 나은 리스크 관리를 나타냅니다."
        )

    # --- ADVANCED RISK METRICS (Expandable) ---
    with st.expander("📊 고급 리스크 지표", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "칼마 비율",
                f"{metrics['calmar_ratio']:.2f}",
                help="CAGR을 절대 최대 낙폭으로 나눈 값. 최악의 손실 대비 수익을 측정. >0.5는 양호, >1.0은 우수. 높은 수익이 리스크를 정당화하는지 보여줍니다."
            )
            st.metric(
                "베타 (vs SPY)",
                f"{metrics['beta']:.2f}",
                help="S&P 500 대비 변동성 측정. β<1은 덜 변동적 (방어적), β=1은 시장과 동일, β>1은 더 변동적 (공격적). 포트폴리오의 시장 움직임에 대한 민감도."
            )

        with col2:
            st.metric(
                "알파",
                f"{metrics['alpha']:.2f}%",
                help="베타가 예측하는 것 이상의 초과 수익. 양수 알파는 리스크 조정 후 시장을 능가함을 의미. 관리자의 능력이나 전략의 효과성을 측정."
            )
            st.metric(
                "연간 변동성",
                f"{metrics['volatility']:.2f}%",
                help="연환산된 수익률의 표준편차. 수익률이 얼마나 변동하는지 보여줌. 낮을수록 더 안정적. 일반적인 주식 포트폴리오: 15-25%. 가격 변동에 대한 편안함."
            )

        with col3:
            st.metric(
                "VaR (95%)",
                f"{metrics['var_95']:.2f}%",
                help="위험 가치: 95% 신뢰수준에서 예상되는 최대 일일 손실. 하루에 이보다 더 손실을 볼 확률은 5%만 있음. 리스크 관리 지표."
            )
            st.metric(
                "벤치마크 수익률",
                f"{metrics['benchmark_return']:.2f}%",
                help="동일 기간 동안 동일 설정으로 S&P 500(SPY)의 총 수익률. 포트폴리오 성과를 시장과 비교하는 데 사용."
            )

        with col4:
            st.metric(
                "초과 성과",
                f"{metrics['outperformance']:.2f}%",
                help="포트폴리오 수익률에서 벤치마크 수익률을 뺀 값. 양수는 시장을 이겼음을 의미. 단순히 SPY를 사는 것 대비 적극적 관리가 가치를 더했는지 보여줌."
            )
            st.metric(
                "승률",
                f"{metrics['win_rate']:.2f}%",
                help="양수 수익률을 가진 거래일의 비율. >50%는 상승일이 하락일보다 많음을 의미. 이익의 일관성을 반영하지만 규모는 측정하지 않음."
            )

    st.divider()

    # --- CHARTS (4 Tabs) ---
    st.header("📊 시각적 분석")

    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
        "📈 포트폴리오 성장",
        "💰 배당 소득",
        "📉 낙폭 분석",
        "💸 세금 영향"
    ])

    with chart_tab1:
        st.subheader("시간별 포트폴리오 가치")

        try:
            fig = create_portfolio_growth_chart(
                results['daily_values'],
                results['daily_values_no_drip'],
                results['benchmark_values'],
                results.get('buyhold_values'),
                results.get('schd_values')
            )
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"포트폴리오 성장 차트 생성 오류: {str(e)}")
            st.info("차트 시각화 기능이 구현 중입니다. 플레이스홀더를 사용합니다.")

    with chart_tab2:
        st.subheader("배당 소득 분석")

        if not results['dividend_history'].empty:
            try:
                # Annual dividend income chart
                fig1 = create_dividend_income_chart(results['dividend_history'])
                st.plotly_chart(fig1, width='stretch')

                # Cumulative dividend chart
                fig2 = create_cumulative_dividend_chart(results['dividend_history'])
                st.plotly_chart(fig2, width='stretch')
            except Exception as e:
                st.error(f"배당 차트 생성 오류: {str(e)}")
                st.info("차트 시각화 기능이 구현 중입니다.")
        else:
            st.info("선택한 기간에 배당 데이터가 없습니다.")

    with chart_tab3:
        st.subheader("리스크 및 낙폭 분석")

        try:
            # Underwater chart
            fig1 = create_underwater_chart(results['daily_values'])
            st.plotly_chart(fig1, width='stretch')

            # Return distribution
            fig2 = create_return_distribution_chart(results['daily_values'])
            st.plotly_chart(fig2, width='stretch')
        except Exception as e:
            st.error(f"낙폭 차트 생성 오류: {str(e)}")
            st.info("차트 시각화 기능이 구현 중입니다.")

    with chart_tab4:
        if tax_enabled and not results['tax_payments'].empty:
            st.subheader("세금 영향 분석")

            try:
                # Tax payment timeline
                fig1 = create_tax_payment_chart(results['tax_payments'])
                st.plotly_chart(fig1, width='stretch')

                # Pre vs post-tax comparison
                fig2 = create_pre_post_tax_comparison(results)
                st.plotly_chart(fig2, width='stretch')
            except Exception as e:
                st.error(f"세금 차트 생성 오류: {str(e)}")
                st.info("차트 시각화 기능이 구현 중입니다.")
        else:
            st.info("💡 세금 영향 분석이 비활성화되어 있습니다. 사이드바에서 활성화하여 세금 관련 차트를 확인하세요.")

    st.divider()

    # --- DETAILED HOLDINGS TABLE ---
    with st.expander("📋 상세 보유 내역 보기", expanded=False):
        if not results['holdings'].empty:
            st.dataframe(
                results['holdings'],
                column_config={
                    "Symbol": st.column_config.TextColumn("Ticker", width="small"),
                    "Shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "Current Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Market Value": st.column_config.NumberColumn("Value", format="$%.2f"),
                    "Total Dividends": st.column_config.NumberColumn("Dividends Received", format="$%.2f")
                },
                width='stretch',
                hide_index=True
            )

            # Download CSV button
            csv = results['holdings'].to_csv(index=False)
            st.download_button(
                label="📥 보유 내역 CSV 다운로드",
                data=csv,
                file_name=f"portfolio_holdings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("보유 내역 데이터가 없습니다.")

    # --- REBALANCING HISTORY ---
    if not results.get('rebalancing_history', pd.DataFrame()).empty:
        with st.expander("🔄 리밸런싱 히스토리 보기", expanded=False):
            rebalancing_df = results['rebalancing_history']

            st.markdown(f"**총 리밸런싱 횟수:** {len(rebalancing_df)}")

            if len(rebalancing_df) > 0:
                total_rebal_fees = rebalancing_df['Fees'].sum()
                total_rebal_taxes = rebalancing_df['Taxes'].sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 리밸런싱 횟수", len(rebalancing_df))
                with col2:
                    st.metric("총 납부 수수료", f"${total_rebal_fees:,.2f}")
                with col3:
                    st.metric("총 납부 세금", f"${total_rebal_taxes:,.2f}")

                st.dataframe(
                    rebalancing_df,
                    column_config={
                        "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                        "Fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
                        "Taxes": st.column_config.NumberColumn("Taxes", format="$%.2f")
                    },
                    width='stretch',
                    hide_index=True
                )

                # Download CSV button
                csv = rebalancing_df.to_csv(index=False)
                st.download_button(
                    label="📥 리밸런싱 히스토리 CSV 다운로드",
                    data=csv,
                    file_name=f"rebalancing_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

else:
    # Initial state - show instructions
    st.info("""
    👈 **시작하기:**

    1. 사이드바에서 종목 선택 (최대 20개)
    2. 할당 방식 선택
    3. 백테스트 설정 구성 (날짜, 투자금)
    4. 선택적으로 DRIP 및 세금 모델링 활성화
    5. 리밸런싱 전략 선택 (없음은 매수 후 보유, 또는 월간/분기별 등)
    6. "🚀 백테스트 실행" 클릭하여 결과 확인

    **참고:** 이 시뮬레이션은 정확한 DRIP 모델링을 위해 소수점 주식을 사용합니다.

    **리밸런싱:** 빈도를 선택하면 목표 비중으로 포트폴리오를 자동 조정합니다.
    이 과정에서 거래 수수료와 자본 이득 세금이 발생할 수 있습니다.
    """)

    # Show sample portfolio suggestion
    if not df.empty:
        st.subheader("💡 샘플 포트폴리오 아이디어")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**고배당 수익률**")
            top_yield = df.nlargest(5, 'Div. Yield')[['Symbol', 'Company Name', 'Div. Yield']].head(5)
            st.dataframe(top_yield, hide_index=True, width='stretch')

        with col2:
            st.markdown("**배당 귀족 (25년 이상)**")
            aristocrats = df[df['Years'] >= 25].nlargest(5, 'Years')[['Symbol', 'Company Name', 'Years']].head(5)
            if not aristocrats.empty:
                st.dataframe(aristocrats, hide_index=True, width='stretch')
            else:
                st.info("배당 귀족을 보려면 데이터를 필터링하세요")
