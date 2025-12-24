"""
Dividend Stock Analysis Platform - Main Application (Korean Version)
Streamlit-based web application for dividend stock analysis
"""

import streamlit as st
import os
from datetime import datetime
from utils.cache_manager import load_main_dataframe, clear_all_caches
from utils.data_loader import DataManager, check_data_file_exists
from config import AppConfig

# Page configuration
st.set_page_config(
    page_title="배당주 분석 플랫폼",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables at the top
if 'update_in_progress' not in st.session_state:
    st.session_state['update_in_progress'] = False
if 'update_completed' not in st.session_state:
    st.session_state['update_completed'] = False
if 'data_source_mode' not in st.session_state:
    st.session_state['data_source_mode'] = 'cached'

# Sidebar - Data Source Selection
st.sidebar.header("⚙️ 데이터 설정")

data_source = st.sidebar.radio(
    "데이터 소스 선택",
    options=["📁 기존 데이터 사용 (빠름)", "🔄 최신 데이터 수집 (3-5분)"],
    index=0,
    help="기존 데이터는 즉시 로드됩니다. 최신 데이터 수집은 가장 최근 정보를 제공합니다."
)

# Update data_source_mode based on radio selection (for UI display only)
crawl_mode_selected = "🔄 최신 데이터 수집" in data_source

# Display data source information
if not crawl_mode_selected:
    # Use Existing Data mode
    if check_data_file_exists():
        data_info = DataManager.get_data_info()
        if data_info['exists']:
            st.sidebar.success("✅ 기존 데이터 사용 중")
            st.sidebar.info(
                f"📅 마지막 업데이트: {data_info['last_modified'].strftime('%Y-%m-%d %H:%M')}"
            )
            st.sidebar.metric("총 종목 수", f"{data_info.get('row_count', 'N/A'):,}")
    else:
        st.sidebar.error("❌ 기존 데이터 파일을 찾을 수 없습니다!")
        st.sidebar.warning("'최신 데이터 수집'을 선택해주세요")
        st.stop()
else:
    # Crawl Latest Data mode selected
    st.sidebar.warning("⚠️ '데이터 업데이트 시작' 버튼을 눌러 수집을 시작하세요 (3-5분)")

    # Show completion message if update just finished
    if st.session_state['update_completed']:
        st.sidebar.success("✅ 데이터 업데이트 완료!")
        if 'last_update' in st.session_state:
            st.sidebar.info(f"🕐 업데이트 시각: {st.session_state['last_update']}")
        if 'update_stats' in st.session_state:
            st.sidebar.markdown("### 📊 업데이트 요약")
            stats = st.session_state['update_stats']
            st.sidebar.markdown(f"- 총 종목 수: **{stats.get('total', 0):,}**")
            st.sidebar.markdown(f"- 평균 배당률: **{stats.get('avg_yield', 'N/A')}**")

        # Show button to switch back to existing data mode
        if st.sidebar.button("업데이트된 데이터 사용"):
            st.session_state['update_completed'] = False
            st.session_state['data_source_mode'] = 'cached'
            st.rerun()

    # Only show the update button if not in progress and not just completed
    if not st.session_state['update_completed']:
        # Button to trigger update
        button_clicked = st.sidebar.button(
            "🚀 데이터 업데이트 시작",
            type="primary",
            disabled=st.session_state['update_in_progress']
        )

        # Only execute update when button is clicked AND not already in progress
        if button_clicked and not st.session_state['update_in_progress']:
            try:
                # Set update flag to prevent re-entry
                st.session_state['update_in_progress'] = True
                st.session_state['update_completed'] = False

                from datetime import datetime
                from modules.data_collector import DividendDataCollector

                # Progress tracking UI elements
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()

                # Initialize collector
                collector = DividendDataCollector()

                # Progress callback for scraping
                def update_scraping_progress(current, total):
                    progress = int((current / total) * 50)  # 0-50% for scraping
                    progress_bar.progress(progress)
                    status_text.text(f"📊 페이지 수집 중 {current}/{total}...")

                # Stage 1: Scraping & Validation (0-50%)
                status_text.text("🚀 웹 크롤러 시작 중...")

                df = collector.update_all_data(
                    use_scraping=True,
                    progress_callback=update_scraping_progress
                )

                # Stage 2: Processing complete (50-70%)
                progress_bar.progress(70)
                status_text.text("✓ 데이터 처리 완료")

                # Stage 3: Clear caches (70-100%)
                status_text.text("🧹 캐시 삭제 중...")
                st.cache_data.clear()
                progress_bar.progress(100)

                # Store completion info in session state
                update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state['last_update'] = update_time
                st.session_state['data_source'] = 'crawled'

                # Store stats for display
                stats = {'total': len(df)}
                if 'Div. Yield' in df.columns:
                    avg_yield = df['Div. Yield'].mean()
                    stats['avg_yield'] = f"{avg_yield:.2%}"
                st.session_state['update_stats'] = stats

                # Mark as completed BEFORE rerun
                st.session_state['update_in_progress'] = False
                st.session_state['update_completed'] = True
                st.session_state['data_source_mode'] = 'cached'

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Trigger rerun to load new data and show completion message
                st.rerun()

            except Exception as e:
                st.session_state['update_in_progress'] = False
                st.session_state['update_completed'] = False
                st.sidebar.error(f"❌ 업데이트 실패: {str(e)}")
                st.sidebar.info("💡 팁: 기존 데이터를 사용하세요")

                # Show error details in expander
                with st.sidebar.expander("오류 상세 정보 보기"):
                    import traceback
                    st.code(traceback.format_exc())

                # Clear progress indicators
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()

# Load data based on session state mode (not radio button selection)
# This prevents automatic crawling when radio is changed
try:
    use_cached = st.session_state['data_source_mode'] == 'cached'
    df = load_main_dataframe(use_cached=use_cached)

    if df is None:
        st.error("데이터 로드 실패. 데이터 파일을 확인해주세요.")
        st.stop()

except Exception as e:
    st.error(f"데이터 로드 오류: {str(e)}")
    st.stop()

# Main content
st.markdown('<p class="main-header">💰 요약</p>', unsafe_allow_html=True)

st.markdown("""
종합 배당주 분석 플랫폼에 오신 것을 환영합니다. 이 도구는 다음을 지원합니다:
- 아래의 전체 배당주 데이터셋 탐색
- 맞춤 기준으로 필터링된 분석을 위한 스크리너 페이지 사용
- 상세한 지표와 시각화를 통한 개별 종목 분석
- DRIP 및 세금 고려사항이 포함된 포트폴리오 백테스트

사이드바 메뉴를 사용하여 전문 분석 도구에 접근하세요.
""")

st.divider()

# Interactive Dataset Display
st.subheader("배당주 데이터셋")

# Configure column display formatting
column_config = {
    "Symbol": st.column_config.TextColumn(
        "티커",
        help="주식 티커 심볼 (예: AAPL, MSFT)"
    ),
    "Company Name": st.column_config.TextColumn(
        "회사명",
        help="회사의 정식 법인명"
    ),
    "Category": st.column_config.TextColumn(
        "등급",
        help="배당 달성 등급 (Aristocrats: 25년 이상, Kings: 50년 이상, Champions: 연속 증가)"
    ),
    "Div. Yield": st.column_config.NumberColumn(
        "배당률",
        format="%.2f%%",
        help="연간 배당 수익률 - 주당 연간 배당금을 현재 주가로 나눈 값"
    ),
    "Div. Growth 5Y": st.column_config.NumberColumn(
        "5년 배당 성장률",
        format="%.2f%%",
        help="5년 배당 성장률 (CAGR) - 지난 5년간 배당금의 연평균 복리 성장률"
    ),
    "Years": st.column_config.NumberColumn(
        "배당 지급 연수",
        help="중단 없이 배당금을 지급한 연속 연수"
    ),
    "Payout Ratio": st.column_config.NumberColumn(
        "배당성향",
        format="%.2f%%",
        help="배당성향 - 순이익 대비 배당금 지급 비율 (낮을수록 지속가능)"
    ),
    "Market Cap": st.column_config.NumberColumn(
        "시가총액",
        format="$%.2fB",
        help="시가총액 (십억 달러 단위) - 발행 주식 전체의 시장 가치 (주가 × 총 발행 주식 수)"
    ),
    "Sector": st.column_config.TextColumn(
        "섹터",
        help="주요 산업 섹터 (예: Technology, Healthcare, Financials)"
    ),
    "Industry": st.column_config.TextColumn(
        "산업군",
        help="섹터 내 세부 산업 분류 (예: Software, Biotechnology, Banks)"
    ),
    "Five_y_DividendYield_diff": st.column_config.NumberColumn(
        "5년 배당률 차이",
        format="%.2f%%",
        help="5년 평균 배당률 대비 차이 - 양수는 현재 배당률이 역사적 평균보다 높음을 의미 (저평가 가능성)"
    ),
    "Ten_y_DividendYield_diff": st.column_config.NumberColumn(
        "10년 배당률 차이",
        format="%.2f%%",
        help="10년 평균 배당률 대비 차이 - 양수는 현재 배당률이 역사적 평균보다 높음을 의미 (저평가 가능성)"
    ),
}

# Prepare dataframe for display - convert decimal to percentage for display
display_df = df.copy()

# Convert decimal columns to percentage for proper display
import pandas as pd
percentage_cols = ['Div. Yield', 'Div. Growth 5Y', 'Payout Ratio',
                   'Five_y_DividendYield_diff', 'Ten_y_DividendYield_diff']
for col in percentage_cols:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce') * 100

# Convert Market Cap to numeric (handle string values like "911.47B")
if 'Market Cap' in display_df.columns:
    def parse_market_cap(value):
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return value / 1e9

        value_str = str(value).strip().upper()
        if not value_str or value_str == '-':
            return None

        # Remove $ if present
        value_str = value_str.replace('$', '')

        # Extract multiplier
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
            return (numeric_value * multiplier) / 1e9
        except ValueError:
            return None

    display_df['Market Cap'] = display_df['Market Cap'].apply(parse_market_cap)

# Select columns to display
display_columns = ['Symbol', 'Company Name', 'Category', 'Div. Yield', 'Div. Growth 5Y',
                   'Years', 'Payout Ratio', 'Market Cap', 'Sector', 'Industry',
                   'Five_y_DividendYield_diff', 'Ten_y_DividendYield_diff']

# Filter to only existing columns
available_columns = [col for col in display_columns if col in display_df.columns]

# Display interactive dataframe
st.dataframe(
    display_df[available_columns],
    column_config=column_config,
    width='stretch',
    hide_index=True,
    height=600
)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>💡 사이드바 메뉴를 사용하여 스크리닝, 분석, 백테스팅 도구에 접근하세요</p>
    <p style='font-size: 0.8rem;'>데이터 출처: StockAnalysis.com 및 Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
