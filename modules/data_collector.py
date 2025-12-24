"""
Data collection module
Handles web scraping and data processing
"""

import pandas as pd
import os
import time
import shutil
import warnings
from typing import Optional, Callable
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import yfinance as yf

from config import AppConfig, get_data_path, get_main_data_path, get_raw_data_path


class DividendDataCollector:
    """
    Data collection and processing class
    """

    def __init__(self):
        self.data_dir = AppConfig.DATA_DIR

    def update_all_data(self, use_scraping: bool = True, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Main data update function
        Complete workflow: scraping → validation → processing → enrichment → filtering

        Args:
            use_scraping: If True, scrape new data. If False, process existing raw data.
            progress_callback: Optional callback for progress updates

        Returns:
            Processed DataFrame
        """
        # Stage 0: Backup existing data
        print("=" * 60)
        print("STAGE 0: Backing up existing data...")
        print("=" * 60)
        self.backup_existing_data()

        # Stage 1: Data Collection
        print("\n" + "=" * 60)
        print("STAGE 1: Data Collection")
        print("=" * 60)

        if use_scraping:
            # Scrape new data from StockAnalysis.com
            raw_df = self.collect_stockanalysis_data(progress_callback=progress_callback)

            # Validate scraped data
            print("\nValidating scraped data...")
            validation_report = self.validate_scraped_data(raw_df)

            if not validation_report['valid']:
                error_msg = "Data validation failed:\n" + "\n".join(validation_report['errors'])
                raise ValueError(error_msg)

            print("✓ Data validation passed")

            # Save raw data
            raw_data_path = get_raw_data_path()
            raw_df.to_csv(raw_data_path, index=False)
            print(f"✓ Raw data saved to {raw_data_path}")

        else:
            # Load existing raw data
            raw_data_path = get_raw_data_path()

            if not os.path.exists(raw_data_path):
                raise FileNotFoundError(
                    f"Raw data file not found: {raw_data_path}\n"
                    "Please enable scraping or ensure dividend_from_stockanalysis.csv exists."
                )

            raw_df = pd.read_csv(raw_data_path)
            print(f"✓ Loaded existing raw data from {raw_data_path}")

        # Stage 2: Data Processing & Filtering
        print("\n" + "=" * 60)
        print("STAGE 2: Data Processing & Filtering")
        print("=" * 60)

        filtered_df = self.process_raw_data_from_df(raw_df)
        print(f"✓ Applied initial filters: {len(filtered_df)} stocks remaining")

        # Stage 2.5: Add Premium Stock Categories
        print("\n" + "=" * 60)
        print("STAGE 2.5: Adding Premium Stock Categories")
        print("=" * 60)

        aristocrats_set, kings_set, schd_set = self.load_premium_stock_lists()

        filtered_df = self.add_missing_premium_stocks(
            filtered_df, aristocrats_set, kings_set, schd_set
        )
        print(f"✓ Total stocks after adding missing premium stocks: {len(filtered_df)}")

        filtered_df = self.categorize_stocks(
            filtered_df, aristocrats_set, kings_set, schd_set
        )
        print(f"✓ Categorization complete")

        category_counts = filtered_df['Category'].value_counts()
        print("\nCategory breakdown:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

        # Stage 3: Yahoo Finance Enrichment
        print("\n" + "=" * 60)
        print("STAGE 3: Yahoo Finance Enrichment")
        print("=" * 60)

        enriched_df = self.enrich_with_yfinance(filtered_df)

        # Stage 4: Second-Level Filtering
        print("\n" + "=" * 60)
        print("STAGE 4: Final Filtering")
        print("=" * 60)

        final_df = self.apply_yield_comparison_filter(enriched_df)

        # Stage 5: Save Final Data
        print("\n" + "=" * 60)
        print("STAGE 5: Saving Results")
        print("=" * 60)

        output_path = get_main_data_path()
        final_df.to_csv(output_path, index=False)
        print(f"✓ Final data saved to {output_path}")
        print(f"✓ Total stocks in final dataset: {len(final_df)}")

        print("\n" + "=" * 60)
        print("DATA UPDATE COMPLETE")
        print("=" * 60)

        return final_df

    def process_raw_data_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw DataFrame (alternative to process_raw_data that takes a file path)

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame
        """
        # Data preprocessing
        df = self.convert_percentage_columns(df)
        df = self.convert_numeric_columns(df)
        df = self.apply_initial_filters(df)

        # Apply 5-criteria filtering from notebook
        df = self.apply_dividend_criteria_filters(df)

        return df

    def apply_dividend_criteria_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the 5 dividend quality criteria filters
        Premium stocks (Aristocrats/Kings/SCHD) bypass these filters

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        result = df.copy()

        aristocrats_set, kings_set, schd_set = self.load_premium_stock_lists()
        all_premium_stocks = aristocrats_set | kings_set | schd_set

        if 'Symbol' not in result.columns:
            return result

        premium_mask = result['Symbol'].str.upper().isin(all_premium_stocks)
        premium_stocks = result[premium_mask].copy()
        regular_stocks = result[~premium_mask].copy()

        print(f"  Premium stocks (bypass filters): {len(premium_stocks)}")
        print(f"  Regular stocks (apply filters): {len(regular_stocks)}")

        if 'Div. Yield' in regular_stocks.columns:
            regular_stocks = regular_stocks[regular_stocks['Div. Yield'] >= AppConfig.DEFAULT_MIN_YIELD]

        if 'Payout Ratio' in regular_stocks.columns:
            regular_stocks = regular_stocks[
                (regular_stocks['Payout Ratio'] >= AppConfig.DEFAULT_PAYOUT_MIN) &
                (regular_stocks['Payout Ratio'] <= AppConfig.DEFAULT_PAYOUT_MAX)
            ]

        if 'Years' in regular_stocks.columns:
            regular_stocks = regular_stocks[regular_stocks['Years'] >= AppConfig.DEFAULT_MIN_YEARS]

        if 'Div. Growth' in regular_stocks.columns:
            regular_stocks = regular_stocks[regular_stocks['Div. Growth'] >= AppConfig.DEFAULT_MIN_GROWTH]

        if 'Div. Growth 5Y' in regular_stocks.columns:
            regular_stocks = regular_stocks[regular_stocks['Div. Growth 5Y'] >= AppConfig.DEFAULT_MIN_GROWTH_5Y]

        result = pd.concat([premium_stocks, regular_stocks], ignore_index=True)

        print(f"  After filtering: {len(result)} total stocks")

        return result

    def load_premium_stock_lists(self) -> tuple:
        """
        Load premium stock lists from CSV files

        Returns:
            tuple: (aristocrats_set, kings_set, schd_set) as sets of ticker symbols
        """
        aristocrats_set = set()
        kings_set = set()
        schd_set = set()

        try:
            aristocrats_path = get_data_path(AppConfig.DIVIDEND_ARISTOCRATS_FILE)
            if os.path.exists(aristocrats_path):
                aristocrats_df = pd.read_csv(aristocrats_path)
                aristocrats_set = set(aristocrats_df['Symbol'].str.upper())
                print(f"  Loaded {len(aristocrats_set)} Dividend Aristocrats")
            else:
                print(f"  Warning: {aristocrats_path} not found")
        except Exception as e:
            print(f"  Warning: Could not load Aristocrats list: {e}")

        try:
            kings_path = get_data_path(AppConfig.DIVIDEND_KINGS_FILE)
            if os.path.exists(kings_path):
                kings_df = pd.read_csv(kings_path)
                kings_set = set(kings_df['Symbol'].str.upper())
                print(f"  Loaded {len(kings_set)} Dividend Kings")
            else:
                print(f"  Warning: {kings_path} not found")
        except Exception as e:
            print(f"  Warning: Could not load Kings list: {e}")

        try:
            schd_path = get_data_path(AppConfig.SCHD_HOLDINGS_FILE)
            if os.path.exists(schd_path):
                schd_df = pd.read_csv(schd_path)
                schd_set = set(schd_df['Symbol'].str.upper())
                print(f"  Loaded {len(schd_set)} SCHD holdings")
            else:
                print(f"  Warning: {schd_path} not found")
        except Exception as e:
            print(f"  Warning: Could not load SCHD list: {e}")

        return aristocrats_set, kings_set, schd_set

    def categorize_stocks(self, df: pd.DataFrame, aristocrats_set: set,
                         kings_set: set, schd_set: set) -> pd.DataFrame:
        """
        Add Category column to DataFrame based on premium stock membership

        Args:
            df: Input DataFrame with Symbol column
            aristocrats_set: Set of Dividend Aristocrat tickers
            kings_set: Set of Dividend King tickers
            schd_set: Set of SCHD holdings tickers

        Returns:
            DataFrame with new Category column
        """
        result = df.copy()

        def get_category(symbol):
            categories = []

            symbol_upper = str(symbol).upper()

            if symbol_upper in aristocrats_set:
                categories.append('Aristocrat')
            if symbol_upper in kings_set:
                categories.append('King')
            if symbol_upper in schd_set:
                categories.append('SCHD')

            if categories:
                return '+'.join(categories)
            else:
                return 'Others'

        result['Category'] = result['Symbol'].apply(get_category)

        return result

    def add_missing_premium_stocks(self, df: pd.DataFrame, aristocrats_set: set,
                                   kings_set: set, schd_set: set) -> pd.DataFrame:
        """
        Add premium stocks that are missing from the DataFrame

        Args:
            df: Current DataFrame
            aristocrats_set: Set of Dividend Aristocrat tickers
            kings_set: Set of Dividend King tickers
            schd_set: Set of SCHD holdings tickers

        Returns:
            DataFrame with missing premium stocks added
        """
        result = df.copy()

        all_premium_stocks = aristocrats_set | kings_set | schd_set

        if 'Symbol' in result.columns:
            existing_symbols = set(result['Symbol'].str.upper())
        else:
            existing_symbols = set()

        missing_symbols = all_premium_stocks - existing_symbols

        if not missing_symbols:
            print(f"  All premium stocks already present in dataset")
            return result

        print(f"  Adding {len(missing_symbols)} missing premium stocks: {sorted(missing_symbols)}")

        missing_rows = []
        for symbol in missing_symbols:
            row_data = {
                'Symbol': symbol,
                'Company Name': f"{symbol} (Premium Stock)",
            }

            for col in result.columns:
                if col not in row_data:
                    if result[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        row_data[col] = 0.0
                    else:
                        row_data[col] = ''

            missing_rows.append(row_data)

        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            result = pd.concat([result, missing_df], ignore_index=True)
            result = result.drop_duplicates(subset=['Symbol'], keep='first')

        return result

    def process_raw_data(self, raw_data_path: str) -> pd.DataFrame:
        """
        Process raw data from StockAnalysis.com

        Args:
            raw_data_path: Path to raw CSV file

        Returns:
            Processed DataFrame
        """
        # Load raw data
        df = pd.read_csv(raw_data_path)

        # Data preprocessing
        df = self.convert_percentage_columns(df)
        df = self.convert_numeric_columns(df)
        df = self.apply_initial_filters(df)

        return df

    def convert_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert percentage string columns to float

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with converted percentages
        """
        result = df.copy()

        # Percentage columns to convert
        pct_columns = ['Div. Yield', 'Payout Ratio', 'Div. Growth', 'Div. Growth 5Y', 'Div. Growth 3Y']

        for col in pct_columns:
            if col in result.columns:
                result[col] = result[col].apply(self._convert_percentage)

        return result

    @staticmethod
    def _convert_percentage(value):
        """Convert percentage string to float"""
        try:
            if pd.isna(value) or value == '-':
                return None
            # Remove % and convert to decimal
            return float(str(value).replace('%', '')) / 100
        except:
            return None

    def convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric string columns"""
        result = df.copy()

        # Convert Years column
        if 'Years' in result.columns:
            result['Years'] = pd.to_numeric(result['Years'], errors='coerce')

        return result

    def apply_initial_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply initial data quality filters

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        result = df.copy()

        # Remove rows with missing dividend yield
        if 'Div. Yield' in result.columns:
            result = result[result['Div. Yield'].notna()]
            result = result[result['Div. Yield'] > 0]

        return result

    def get_chrome_driver(self):
        """
        Initialize Chrome WebDriver in headless mode

        Returns:
            WebDriver instance
        """
        options = webdriver.ChromeOptions()

        # Always run in headless mode (user requirement)
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    def collect_stockanalysis_data(self, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Web scraping from StockAnalysis.com

        Args:
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            DataFrame with scraped dividend stock data
        """
        print("Initializing web scraper...")
        driver = self.get_chrome_driver()

        try:
            # Navigate to StockAnalysis.com screener
            print(f"Accessing {AppConfig.STOCKANALYSIS_URL}")
            driver.get(AppConfig.STOCKANALYSIS_URL)
            time.sleep(AppConfig.PAGE_LOAD_WAIT)

            # Configure UI - click through XPaths to set up table
            print("Configuring screener UI...")
            for i, xpath in enumerate(AppConfig.SCRAPER_XPATHS):
                # Try to close popup before each action
                try:
                    popup = driver.find_element(By.XPATH, AppConfig.POPUP_XPATH)
                    popup.click()
                    time.sleep(0.5)
                except:
                    pass  # Popup not present

                # Click the element
                try:
                    element = driver.find_element(By.XPATH, xpath)
                    element.click()
                    time.sleep(AppConfig.PAGE_LOAD_WAIT)
                    print(f"  ✓ Clicked element {i+1}/{len(AppConfig.SCRAPER_XPATHS)}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not click element {i+1}: {e}")

            # Scrape all pages
            print(f"\nScraping {AppConfig.MAX_SCRAPE_PAGES} pages...")
            df_list = []

            for page_num in tqdm(range(AppConfig.MAX_SCRAPE_PAGES), desc="Scraping pages"):
                # Try to close popup
                try:
                    popup = driver.find_element(By.XPATH, AppConfig.POPUP_XPATH)
                    popup.click()
                except:
                    pass

                # Extract table HTML
                try:
                    table_html = driver.find_element(By.ID, "main-table").get_attribute('outerHTML')

                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(table_html, 'html.parser')

                    # Extract headers
                    headers = [header.text.strip() for header in soup.find_all('th')]

                    # Extract rows
                    rows = []
                    for row in soup.find_all('tr')[1:]:  # Skip header row
                        rows.append([cell.text.strip() for cell in row.find_all('td')])

                    # Create DataFrame
                    if rows and headers:
                        page_df = pd.DataFrame(rows, columns=headers)
                        df_list.append(page_df)

                    # Update progress
                    if progress_callback:
                        progress_callback(page_num + 1, AppConfig.MAX_SCRAPE_PAGES)

                except Exception as e:
                    print(f"\n⚠ Error scraping page {page_num + 1}: {e}")

                # Navigate to next page
                if page_num < AppConfig.MAX_SCRAPE_PAGES - 1:
                    try:
                        next_button = driver.find_element(By.XPATH, AppConfig.NEXT_BUTTON_XPATH)
                        next_button.click()
                        time.sleep(AppConfig.PAGE_LOAD_WAIT)
                    except Exception as e:
                        print(f"\n⚠ Could not navigate to next page: {e}")
                        break

            # Concatenate all DataFrames
            if df_list:
                final_df = pd.concat(df_list, ignore_index=True)
                print(f"\n✓ Successfully scraped {len(final_df)} stocks")
                return final_df
            else:
                raise ValueError("No data was scraped")

        finally:
            driver.quit()
            print("WebDriver closed")

    def backup_existing_data(self):
        """
        Create backup of existing data (no timestamp - single backup file)
        Only maintains 2 files: current data + one backup
        """
        raw_path = os.path.join(self.data_dir, 'dividend_from_stockanalysis.csv')
        backup_path = os.path.join(self.data_dir, 'dividend_from_stockanalysis_backup.csv')
        final_path = os.path.join(self.data_dir, 'final_df2.csv')
        final_backup_path = os.path.join(self.data_dir, 'final_df2_backup.csv')

        # Backup raw data
        if os.path.exists(raw_path):
            shutil.copy2(raw_path, backup_path)
            print(f"✓ Backed up raw data to {backup_path}")

        # Backup processed data
        if os.path.exists(final_path):
            shutil.copy2(final_path, final_backup_path)
            print(f"✓ Backed up processed data to {final_backup_path}")

    def validate_scraped_data(self, df: pd.DataFrame) -> dict:
        """
        Validate scraped data and return quality report

        Args:
            df: DataFrame to validate

        Returns:
            dict with validation results and statistics
        """
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Required columns from StockAnalysis.com
        required_columns = [
            'Symbol', 'Company Name', 'Market Cap',
            'Div. ($)', 'Div. Yield', 'Payout Ratio',
            'Div. Growth', 'Payout Freq.', 'Years',
            'Div. Growth 3Y', 'Div. Growth 5Y'
        ]

        # 1. Check DataFrame not empty
        if df.empty:
            validation_report['valid'] = False
            validation_report['errors'].append("Scraped data is empty")
            return validation_report

        # 2. Check minimum row count
        if len(df) < AppConfig.MIN_EXPECTED_STOCKS:
            validation_report['valid'] = False
            validation_report['errors'].append(
                f"Too few stocks scraped: {len(df)} (expected >{AppConfig.MIN_EXPECTED_STOCKS})"
            )

        # 3. Validate column schema
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_report['valid'] = False
            validation_report['errors'].append(f"Missing columns: {missing_cols}")

        # 4. Data quality checks
        validation_report['stats']['total_rows'] = len(df)
        validation_report['stats']['total_columns'] = len(df.columns)

        # Check for missing/invalid dividend yields
        if 'Div. Yield' in df.columns:
            invalid_yields = df[df['Div. Yield'] == '-']
            validation_report['stats']['stocks_with_dividends'] = len(df) - len(invalid_yields)
            validation_report['stats']['stocks_without_dividends'] = len(invalid_yields)

        # Check for duplicate symbols
        if 'Symbol' in df.columns:
            duplicates = df[df.duplicated('Symbol', keep=False)]
            if len(duplicates) > 0:
                validation_report['warnings'].append(f"Found {len(duplicates)} duplicate symbols")
                validation_report['stats']['duplicate_count'] = len(duplicates)

        return validation_report

    def enrich_with_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich filtered stocks with Yahoo Finance data

        Args:
            df: DataFrame with Symbol column

        Returns:
            DataFrame with additional YF columns
        """
        print("\nEnriching data with Yahoo Finance metrics...")
        result = df.copy()

        # Initialize new columns
        yf_columns = [
            'fiveYearAvgDivdendYield',
            'Trainling_5Y_avg_dividend_yield',
            'Trainling_5Y_min_dividend_yield',
            'Trainling_5Y_max_dividend_yield',
            'Trainling_10Y_avg_dividend_yield',
            'Trainling_10Y_min_dividend_yield',
            'Trainling_10Y_max_dividend_yield',
            'Sector',
            'Industry',
            'FCF_Dividend_Ratio',
            'Debt_to_Equity',
            'ROE'
        ]
        for col in yf_columns:
            if 'dividend' in col.lower() or col in ['FCF_Dividend_Ratio', 'Debt_to_Equity', 'ROE']:
                result[col] = 0.0
            else:
                result[col] = ''

        # Process each ticker
        for ticker_symbol in tqdm(result['Symbol'], desc="Fetching Yahoo Finance data"):
            try:
                ticker_obj = yf.Ticker(ticker_symbol)

                # Get 5-year average from info
                try:
                    avg_yield = ticker_obj.info.get('fiveYearAvgDividendYield', 0)
                    if avg_yield:
                        result.loc[result['Symbol'] == ticker_symbol, 'fiveYearAvgDivdendYield'] = round(avg_yield / 100, 4)
                except:
                    pass

                # Get Sector and Industry
                try:
                    result.loc[result['Symbol'] == ticker_symbol, 'Sector'] = ticker_obj.info.get('sector', '')
                    result.loc[result['Symbol'] == ticker_symbol, 'Industry'] = ticker_obj.info.get('industry', '')
                except:
                    pass

                # Get financial metrics (FCF/Dividend Ratio, Debt-to-Equity, ROE)
                try:
                    # FCF/Dividend Ratio calculation
                    # free_cash_flow = ticker_obj.info.get('freeCashflow', 0)
                    # total_dividend_paid = ticker_obj.info.get('dividendsPaid', 0)
                    fcf = ticker_obj.info.get('freeCashflow', 0)
                    shares = ticker_obj.info.get('sharesOutstanding', 0)
                    div_rate = ticker_obj.info.get('dividendRate', 0)

                    if fcf !=0 and shares > 0 and div_rate > 0:
                        # 1. 총 배당금 지급액 추정 (주식 수 * 주당 배당금)
                        total_dividend_estimated = shares * div_rate
                        # dividendsPaid is negative, so use abs()
                        fcf_dividend_ratio = fcf / total_dividend_estimated
                        result.loc[result['Symbol'] == ticker_symbol, 'FCF_Dividend_Ratio'] = round(fcf_dividend_ratio, 2)

                    # Debt-to-Equity Ratio
                    debt_to_equity = ticker_obj.info.get('debtToEquity', 0)
                    if debt_to_equity:
                        result.loc[result['Symbol'] == ticker_symbol, 'Debt_to_Equity'] = round(debt_to_equity, 2)

                    # Return on Equity
                    roe = ticker_obj.info.get('returnOnEquity', 0)
                    if roe:
                        # ROE comes as decimal (e.g., 0.15 for 15%), convert to percentage
                        result.loc[result['Symbol'] == ticker_symbol, 'ROE'] = round(roe * 100, 2)
                except:
                    pass

                # Calculate rolling metrics from 11-year history
                try:
                    hist_df = ticker_obj.history(period="11y")[['Dividends', 'Close']]

                    if not hist_df.empty:
                        # Forward-fill dividends and calculate yield
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        hist_df['Dividends'] = hist_df['Dividends'].replace(0, method='ffill')
                        hist_df['Annual_Dividends'] = hist_df['Dividends'] * 4  # Quarterly assumption
                        hist_df['dividend_yield'] = hist_df['Annual_Dividends'] / hist_df['Close']

                        # Rolling windows (1260 days ≈ 5 years, 2520 days ≈ 10 years)
                        hist_df['T5Y_avg'] = hist_df['dividend_yield'].rolling(window=1260).mean()
                        hist_df['T5Y_min'] = hist_df['dividend_yield'].rolling(window=1260).min()
                        hist_df['T5Y_max'] = hist_df['dividend_yield'].rolling(window=1260).max()
                        hist_df['T10Y_avg'] = hist_df['dividend_yield'].rolling(window=2520).mean()
                        hist_df['T10Y_min'] = hist_df['dividend_yield'].rolling(window=2520).min()
                        hist_df['T10Y_max'] = hist_df['dividend_yield'].rolling(window=2520).max()

                        # Get latest values
                        if len(hist_df) > 0:
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_5Y_avg_dividend_yield'] = hist_df['T5Y_avg'].iloc[-1] if not pd.isna(hist_df['T5Y_avg'].iloc[-1]) else 0.0
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_5Y_min_dividend_yield'] = hist_df['T5Y_min'].iloc[-1] if not pd.isna(hist_df['T5Y_min'].iloc[-1]) else 0.0
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_5Y_max_dividend_yield'] = hist_df['T5Y_max'].iloc[-1] if not pd.isna(hist_df['T5Y_max'].iloc[-1]) else 0.0
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_10Y_avg_dividend_yield'] = hist_df['T10Y_avg'].iloc[-1] if not pd.isna(hist_df['T10Y_avg'].iloc[-1]) else 0.0
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_10Y_min_dividend_yield'] = hist_df['T10Y_min'].iloc[-1] if not pd.isna(hist_df['T10Y_min'].iloc[-1]) else 0.0
                            result.loc[result['Symbol'] == ticker_symbol, 'Trainling_10Y_max_dividend_yield'] = hist_df['T10Y_max'].iloc[-1] if not pd.isna(hist_df['T10Y_max'].iloc[-1]) else 0.0

                except Exception as e:
                    # Handle delisted stocks, 404 errors
                    if "404" in str(e) or "delisted" in str(e).lower():
                        pass  # Silently skip delisted stocks
                    else:
                        pass  # Skip other errors

            except Exception as e:
                # Skip errors for individual stocks
                continue

        print(f"✓ Enrichment complete")
        return result

    def apply_yield_comparison_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply second-level filtering: current yield > historical avg
        Premium stocks (Aristocrats/Kings/SCHD) bypass this filter

        Args:
            df: DataFrame with Yahoo Finance enrichment

        Returns:
            Filtered DataFrame
        """
        result = df.copy()

        # Ensure numeric types for yield columns
        result['fiveYearAvgDivdendYield'] = pd.to_numeric(result['fiveYearAvgDivdendYield'], errors='coerce').fillna(0)
        result['Trainling_10Y_avg_dividend_yield'] = pd.to_numeric(result['Trainling_10Y_avg_dividend_yield'], errors='coerce').fillna(0)

        # Calculate yield differences (avoid division by zero)
        result['Five_y_DividendYield_diff'] = result.apply(
            lambda row: (row['Div. Yield'] - row['fiveYearAvgDivdendYield']) / row['fiveYearAvgDivdendYield']
            if row['fiveYearAvgDivdendYield'] > 0 else 0,
            axis=1
        )
        result['Ten_y_DividendYield_diff'] = result.apply(
            lambda row: (row['Div. Yield'] - row['Trainling_10Y_avg_dividend_yield']) / row['Trainling_10Y_avg_dividend_yield']
            if row['Trainling_10Y_avg_dividend_yield'] > 0 else 0,
            axis=1
        )

        if 'Category' in result.columns:
            premium_mask = result['Category'] != 'Others'
            premium_stocks = result[premium_mask].copy()
            regular_stocks = result[~premium_mask].copy()

            regular_stocks = regular_stocks[
                (regular_stocks['Five_y_DividendYield_diff'] >= 0) |
                (regular_stocks['Ten_y_DividendYield_diff'] >= 0)
            ]

            result = pd.concat([premium_stocks, regular_stocks], ignore_index=True)
            print(f"✓ Premium stocks (bypass yield filter): {len(premium_stocks)}")
            print(f"✓ Regular stocks (after yield filter): {len(regular_stocks)}")
        else:
            result = result[
                (result['Five_y_DividendYield_diff'] >= 0) |
                (result['Ten_y_DividendYield_diff'] >= 0)
            ]

        print(f"✓ Applied yield comparison filter: {len(result)} stocks remaining")
        return result
