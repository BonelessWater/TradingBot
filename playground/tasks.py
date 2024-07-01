import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging

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
        most_recent_date = None
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        return

    # Check if today's date is the most recent date in the CSV
    date_today = datetime.now().date()
    print(most_recent_date and most_recent_date.date() == date_today)
    if most_recent_date and most_recent_date.date() == date_today:
        logger.info("The stock data is already up to date for today.")
        return
    else:
        logger.info("Updating stock data...")

    # Fetch tickers
    tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers_df.Symbol.to_list()

    # Define the past date from which to fetch data
    past_date = datetime.now() - timedelta(days=365 * 3)  # Approximately 3 years ago

# Download stock data
    tickers_price_df = yf.download(tickers, start=past_date, end=datetime.now(), auto_adjust=True)['Close']

    # Save or update the DataFrame to the CSV file
    tickers_price_df.to_csv(csv_file_path)
    logger.info("Stock data has been successfully updated.")