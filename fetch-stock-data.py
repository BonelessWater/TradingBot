import yfinance as yf
import pandas as pd
import django
import os

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tradingbot.settings')
django.setup()

from playground.models import StockData

def fetch_and_save_stock_data(symbol):
    # Fetch the historical data
    stock = yf.Ticker(symbol)
    hist = stock.history(start='2024-01-01')

    # Save the data to the database
    for index, row in hist.iterrows():
        stock_data, created = StockData.objects.get_or_create(
            symbol=symbol,
            date=index.date(),
            defaults={
                'close_price': row['Close'],
                'volume': row['Volume']
            }
        )
        if not created:
            # Update existing record if necessary
            stock_data.close_price = row['Close']
            stock_data.volume = row['Volume']
            stock_data.save()

def get_sp500_tickers():
    # Scrape the S&P 500 list from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    
    # Get the ticker symbols
    sp500_tickers = sp500_table['Symbol'].tolist()
    return sp500_tickers

if __name__ == "__main__":
    symbols = get_sp500_tickers()  # Add more stock symbols as needed
    print(symbols)
    for symbol in symbols:
        print(symbol)
        fetch_and_save_stock_data(symbol)