import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging
from .models import FinancialData
from pypfopt import risk_models

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_data():
    logger.info("Attempting to update stock data...")

    # Path to the CSV file where data is saved
    csv_file_path = 'tickers_prices.csv'

    try:
        # Read the existing data to find the most recent date
        existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=['Date'])
        most_recent_date = existing_data.index.max()
        logger.info(f"Most recent date in the CSV: {most_recent_date}")
    except FileNotFoundError:
        logger.info("CSV file not found. A new file will be created.")
        existing_data = pd.DataFrame()
        most_recent_date = None
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        return

    # Check if today's date is the most recent date in the CSV
    date_today = datetime.now().date()
    if most_recent_date:
        date_diff = date_today - most_recent_date.date()
        if date_diff.days <= 1:
            logger.info("The stock data is already up to date or within one day of the most recent date.")
            return
        else:
            logger.info("Updating stock data...")
    else:
        # If no most recent date, set a start date
        most_recent_date = datetime.now() - timedelta(days=365 * 3)  # Approximately 3 years ago

    # Fetch tickers
    tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    # Convert the filtered DataFrame to a list of symbols
    tickers = tickers_df['Symbol'].to_list()

    # Remove specific tickers
    tickers_to_remove = ['BF.B', 'BRK.B']
    tickers = [ticker for ticker in tickers if ticker not in tickers_to_remove]

    # Define the start date for fetching new data
    start_date = most_recent_date + timedelta(days=1)  # Day after the most recent date in the CSV

    # Download new stock data
    new_data = yf.download(tickers, start=start_date, end=datetime.now(), auto_adjust=True)['Close']

    # Debug: print shape and head of the new data
    logger.info(f"New data shape: {new_data.shape}")
    logger.info(f"New data head: \n{new_data.head()}")

    # Handle missing data by filling or dropping NaNs
    new_data = new_data.dropna()

    # Concatenate new data with existing data
    if not existing_data.empty:
        updated_data = pd.concat([existing_data, new_data])
    else:
        updated_data = new_data

    # Debug: print shape and head of the updated data
    logger.info(f"Updated data shape: {updated_data.shape}")
    logger.info(f"Updated data head: \n{updated_data.head()}")

    # Save the updated DataFrame to the CSV file
    updated_data.to_csv(csv_file_path)
    logger.info("Stock data has been successfully updated.")

    # Ensure the data is passed correctly to the risk model
    try:
        S2 = risk_models.exp_cov(updated_data, frequency=updated_data.shape[1], span=updated_data.shape[1], log_returns=True)
        logger.info("Covariance matrix calculated successfully.")
    except Exception as e:
        logger.error(f"Error calculating covariance matrix: {e}")

def get_financial_data(symbol):
    stock = yf.Ticker(symbol)

    # Fetching financial data
    market_cap = stock.info.get('marketCap')
    enterprise_value = stock.info.get('enterpriseValue')
    trailing_pe = stock.info.get('trailingPE')
    forward_pe = stock.info.get('forwardPE')
    peg_ratio = stock.info.get('pegRatio')
    price_sales = stock.info.get('priceToSalesTrailing12Months')
    price_book = stock.info.get('priceToBook')
    ev_revenue = stock.info.get('enterpriseToRevenue')
    ev_ebitda = stock.info.get('enterpriseToEbitda')
    profit_margin = stock.info.get('profitMargins')
    return_on_assets = stock.info.get('returnOnAssets')
    return_on_equity = stock.info.get('returnOnEquity')
    revenue = stock.info.get('totalRevenue')
    net_income = stock.info.get('netIncomeToCommon')
    diluted_eps = stock.info.get('trailingEps')
    total_cash = stock.info.get('totalCash')
    debt_to_equity = stock.info.get('debtToEquity')
    levered_free_cash_flow = stock.info.get('freeCashflow')
    earnings_per_share = stock.info.get('trailingEps')
    price_to_earnings_ratio = stock.info.get('trailingPE')
    dividend_yield = stock.info.get('dividendYield')
    book_value = stock.info.get('bookValue')
    debt_to_equity_ratio = stock.info.get('debtToEquity')
    revenue_growth = stock.info.get('revenueGrowth')
    free_cash_flow = stock.info.get('freeCashflow')
    
    FinancialData.objects.update_or_create(
        ticker=symbol,
        defaults={
            'created_at': datetime.now().date(),
            'market_cap': market_cap,
            'enterprise_value': enterprise_value,
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_sales': price_sales,
            'price_book': price_book,
            'ev_revenue': ev_revenue,
            'ev_ebitda': ev_ebitda,
            'profit_margin': profit_margin,
            'return_on_assets': return_on_assets,
            'return_on_equity': return_on_equity,
            'revenue': revenue,
            'net_income': net_income,
            'diluted_eps': diluted_eps,
            'total_cash': total_cash,
            'debt_to_equity': debt_to_equity,
            'levered_free_cash_flow': levered_free_cash_flow,
            'earnings_per_share': earnings_per_share,
            'price_to_earnings_ratio': price_to_earnings_ratio,
            'dividend_yield': dividend_yield,
            'book_value': book_value,
            'debt_to_equity_ratio': debt_to_equity_ratio,
            'revenue_growth': revenue_growth,
            'free_cash_flow': free_cash_flow
        }
    )

def save_all_sp500_metrics():

    most_recent_date = FinancialData.objects.all().order_by('id').first().created_at
    # Check if today's date is the most recent date in the CSV
    date_today = datetime.now().date()
    try:
        if most_recent_date:
            date_diff = date_today - most_recent_date.date()
            if date_diff.days <= 1:
                logger.info("The financial data is already up to date or within one day of the most recent date.")
                return
            else:
                logger.info("Updating stock data...")
    except AttributeError:
        pass
    
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.Symbol.to_list()

    # Remove specific tickers
    tickers_to_remove = ['BF.B', 'BRK.B', 'BF-B', 'BRK-B']
    tickers = [ticker for ticker in tickers if ticker not in tickers_to_remove]

    for ticker in tickers:
        try:
            get_financial_data(ticker)
        except Exception as e:
            print(f"Could not retrieve data for {ticker}: {e}")