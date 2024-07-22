from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .forms import ParametersForm, ResearchForm
from .models import CovarianceData, SP500Ticker, FinancialData, IncomeStatement, BalanceSheet, CashFlow, Earnings
import requests
from pathlib import Path
import os
from dotenv import load_dotenv # Retrieving api key
import logging # Logger for debugging
import yfinance as yf # Retrieving stock data
import time as t
from datetime import datetime, timedelta, date
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import ast # For non-post requests
import decimal
from tqdm import tqdm # Monte Carlo simulation

# Efficient frontier
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov

# Sentiment analysis
from sentiment import finsent
import nltk

def robots_txt(request):
    lines = [
        "User-agent: *",
        "Disallow: /private/",
        "Disallow: /tmp/",
        "Disallow: /admin/",
        "Sitemap: https://stockportfoliobuilder.azurewebsites.net/sitemap.xml",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / '.env')
ALPHA_KEY = os.getenv('ALPHA_KEY')

logger = logging.getLogger(__name__)

def main(request):
    select_low_correlation_stocks(0)
    CovarianceData.flush()
    return render(request, 'main.html')

def order_tickers_by_returns(amount_of_tickers, tickers=None):
    # Read the CSV file
    tickers_price_df = pd.read_csv('tickers_prices.csv', index_col='Date', parse_dates=['Date'])
    
    if tickers and isinstance(tickers, list):
        return tickers_price_df[tickers]
    
    # Ensure the DataFrame is sorted by date
    tickers_price_df.sort_index(inplace=True)
    
    # Calculate the returns for each ticker
    returns = tickers_price_df.pct_change().iloc[1:].sum()
    
    # Sort tickers based on the returns
    sorted_returns = returns.sort_values(ascending=False)
    
    # Get the top 'amount_of_tickers' tickers
    top_returns = sorted_returns.head(amount_of_tickers)
    
    # Filter the original DataFrame to only include the top tickers
    tickers_price_df = tickers_price_df[top_returns.index]
    
    return tickers_price_df

def select_low_correlation_stocks(number_of_stocks, correlation_threshold=0.2):
    min_return = 0.10
    # Read the CSV file
    tickers_price_df = pd.read_csv('tickers_prices.csv', index_col='Date', parse_dates=['Date'])
    
    # Ensure the DataFrame is sorted by date
    tickers_price_df.sort_index(inplace=True)

    # Fetch S&P 500 data for the past 3 years
    sp500_data = yf.download('^GSPC', start=tickers_price_df.index.min(), end=tickers_price_df.index.max())
    sp500_returns = np.log(sp500_data['Adj Close'] / sp500_data['Adj Close'].shift(1)).dropna()

    # Join S&P 500 returns to the tickers data
    tickers_price_df['SP500'] = sp500_returns.reindex(tickers_price_df.index)

    # Calculate logarithmic returns for each ticker
    log_returns = np.log(tickers_price_df / tickers_price_df.shift(1)).dropna()

    # Calculate average annual log returns assuming 252 trading days
    annual_returns = log_returns.mean() * 252

    # Filter stocks with returns greater than 10%
    high_return_stocks = annual_returns[annual_returns > min_return]

    # Select data for high-return stocks including SP500
    high_return_data = log_returns[high_return_stocks.index.tolist() + ['SP500']]

    # Calculate the correlation matrix for these high-return stocks with SP500
    correlation_matrix = high_return_data.corr()

    # Extract the correlation of each stock with the SP500 and filter for negative correlations below the threshold
    sp500_correlations = correlation_matrix['SP500'].drop('SP500')
    print("Correlation with S&P 500:", sp500_correlations)

    # Select stocks based on the negative correlation threshold
    selected_stocks = sp500_correlations[sp500_correlations < correlation_threshold].index.tolist()

    # Filter the original DataFrame to only include the selected tickers
    filtered_tickers_df = tickers_price_df[selected_stocks]

    # Ensure you do not select more stocks than the number requested
    selected_stocks = selected_stocks
    filtered_tickers_df = filtered_tickers_df[selected_stocks]
    print(filtered_tickers_df.head())
    print(filtered_tickers_df)
    print(selected_stocks)
    print(len(selected_stocks))

def get_financial_data(symbol, investment_amount = -1, weight = 0.0):
    """Gets the financial data relavent to a specified stock ticker

    Parameters
    ----------
    symbol : string
        the stock ticker
    investment_amount : int, optional
        the amount in units, that the user wants to invest, by default -1
    weight : float, optional
        the calculated percentage that the user should invest in the specific stock (given by efficient frontier), by default 0

    Returns
    -------
    dict, dict, dict
        dictionaries of a stock's financial information
    """

    # Query the database for the specified ticker
    financial_data = FinancialData.objects.get(ticker=symbol)
    
    # Populate the 'valuation' dictionary
    valuation = {
        'Ticker': financial_data.ticker,
        'Market Cap': financial_data.market_cap,
        'Enterprise Value': financial_data.enterprise_value,
        'Trailing P/E': financial_data.trailing_pe,
        'Forward P/E': financial_data.forward_pe,
        'PEG Ratio': financial_data.peg_ratio,
        'Price Sales': financial_data.price_sales,
        'Price Book': financial_data.price_book,
        'Enterprise Revenue': financial_data.ev_revenue,
        'Enterprise EBITDA': financial_data.ev_ebitda,
    }
    
    # Populate the 'finance' dictionary
    finance = {
        'Profit Margin': financial_data.profit_margin,
        'Return on Assets (TTM)': financial_data.return_on_assets,
        'Return on Equity (TTM)': financial_data.return_on_equity,
        'Revenue (TTM)': financial_data.revenue,
        'Net Income AVI to Common (TTM)': financial_data.net_income,
        'Diluted EPS (TTM)': financial_data.diluted_eps,
        'Total Cash (MRQ)': financial_data.total_cash,
        'Total Debt/Equity (MRQ)': financial_data.debt_to_equity,
        'Levered Free Cash Flow (MRQ)': financial_data.levered_free_cash_flow,
    }
    
    # Populate the 'data' dictionary
    data = {
        'ticker': financial_data.ticker,
        'marketcap': financial_data.market_cap,
        'enterprisevalue': financial_data.enterprise_value,
        'trailingpe': financial_data.trailing_pe,
        'forwardpe': financial_data.forward_pe,
        'pegratio': financial_data.peg_ratio,
        'pricesales': financial_data.price_sales,
    }
    
    # Populate the 'other' dictionary
    other = {
        'earnings per share': financial_data.earnings_per_share,
        'price to earnings ratio': financial_data.price_to_earnings_ratio,
        'dividend yield': financial_data.dividend_yield,
        'book value': financial_data.book_value,
        'debt to equity ratio': financial_data.debt_to_equity_ratio,
        'revenue growth': financial_data.revenue_growth,
        'free cash flow': financial_data.free_cash_flow,
        'return on equity': financial_data.return_on_equity,
    }

    if investment_amount != -1:
        data['percentage'] = f"{weight*100:.2f}%"
        data['investmentamount'] = f"{(int(investment_amount) * weight):.3f}"

    return valuation, finance, data

def get_sp500_data():
    """Gets sp500 data to be shown in the research tab by default

    Returns
    -------
    dict
        sp500 closing prices and moving average
    """
    
    # Define the ticker symbol for the S&P 500
    ticker_symbol = '^GSPC'
    
    # Calculate the start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)  # roughly 3 years
    
    try:
        # Fetch the data using yfinance
        sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Calculate the moving average and handle NaN values
    sp500_data['50_MA'] = sp500_data['Close'].rolling(window=50).mean()
    sp500_data['50_MA'] = sp500_data['50_MA'].fillna(sp500_data['Close'])

    # Ensure the index is of datetime type
    sp500_data.index = pd.to_datetime(sp500_data.index)

    # Prepare the dictionary
    data_dict = {
        'date': [date.strftime('%Y-%m-%d') for date in sp500_data.index],
        'Close Price': sp500_data['Close'].tolist(), 
        'Moving Average': sp500_data['50_MA'].tolist() 
    }
    return json.dumps(data_dict)

def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

def get_covariance(amount_of_tickers, build_tickers=None):
    """Gets or calculates the covarainces matrix of the sp500's closing prices. if already calculated, it is fetched from a database

    Returns
    -------
    pd.dataframe
        returns the covaraince matrix in the form of a datafram

    Raises
    ------
    ValueError
        raises value errors for debugging
    """
    
    # If tickers is a list then it will likely not be saved in the database so we can skip all this error checking and efficiency and jump straight into the calculation
    if build_tickers and isinstance(build_tickers, list):
        tickers_price_df = order_tickers_by_returns(amount_of_tickers, build_tickers)
        S2 = exp_cov(tickers_price_df, frequency=252, span=60, log_returns=True)  # Adjust frequency and span as needed
        S2 = (S2 + S2.T) / 2  # Ensure symmetry
        return S2
    
    try:
        # Query all symbols from the database
        tickers = list(SP500Ticker.objects.all().order_by('id').values_list('symbol', flat=True))
        
        if not tickers:
            raise ValueError("No tickers available after filtering.")

        tickers_str = ','.join(sorted(tickers))
        today = date.today()

        # Check if the data already exists for today
        covariance_entry = CovarianceData.objects.filter(tickers=tickers_str).order_by('-calculation_date').first()

        today = date.today()

        try:
            calculation_day = covariance_entry.calculation_date
            if today == calculation_day:
                try:
                    S2 = covariance_entry.deserialize_matrix(covariance_entry.covariance_matrix)
                    return S2
                except AttributeError:
                    logger.error("Error deserializing covariance matrix from the database.")
        except:
            pass

        # If not during the restricted time and no entry was found or deserialization failed
        csv_file_path = 'tickers_prices.csv'
        try:
            tickers_price_df = order_tickers_by_returns(amount_of_tickers, tickers)
            #tickers, tickers_price_df = select_low_correlation_stocks(amount_of_tickers)
            logger.info(f"CSV file loaded successfully from {csv_file_path}")
        except FileNotFoundError:
            logger.error(f"CSV file not found at path: {csv_file_path}")
            raise ValueError("CSV file not found. Please ensure the file path is correct.")
        except Exception as e:
            logger.error(f"Error reading the CSV file: {e}")
            raise ValueError(f"Error reading the CSV file: {e}")

        # Ensure DataFrame is not empty
        if tickers_price_df.empty:
            logger.error("Tickers price DataFrame is empty.")
            raise ValueError("Tickers price DataFrame is empty.")

        # Handle missing values by filling with 0
        tickers_price_df.fillna(value=0, inplace=True)

        # Log the size of the DataFrame
        logger.info(f"Size of DataFrame: {tickers_price_df.shape}")
        logger.info(f"Missing values {tickers_price_df.isnull().sum()}")

        # Calculate covariance matrix
        try:
            logger.info('Calculated covariance matrix!!!')
            S2 = exp_cov(tickers_price_df, frequency=252, span=60, log_returns=True)  # Adjust frequency and span as needed
            S2 = (S2 + S2.T) / 2  # Ensure symmetry
            if S2.empty:
                logger.error("Calculated covariance matrix is empty.")
                raise ValueError("Calculated covariance matrix is empty.")
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            raise ValueError(f"Error calculating covariance matrix: {e}")

        # Save the new result in the database
        CovarianceData.objects.update_or_create(
            tickers=tickers_str,
            calculation_date=today,
            defaults={'covariance_matrix': S2.to_json()}
        )
        logger.info("Covariance matrix calculated and saved successfully.")

        return S2
    except Exception as e:
        logger.error(f"Unexpected error in get_covariance: {e}")
        raise

def get_portfolio(investment_amount=0, number_of_stocks=0, min_var=100, build_tickers=None, build=False):
    """returns the efficient portfolio of the number of stocks the user specifies

    Parameters
    ----------
    investment_amount : int
        the amount in units, that the user would like to invest
    number_of_stocks : int
        the number of stocks the user whats to consider
    min_var : float
        the minimun value at risk that the user is willing to incur
    tickers : int / list
        if variables is an array it computes the portfolio with the given tickers
    Returns
    -------
    if tickers == 0:
        return error, error message, best protfolio by sharpe ratio, sharpe ratio stock info, best portfolio by returns, returns stock info
    else:
        return error, error message, best protfolio by sharpe ratio, sharpe ratio stock info
    """
    # Error checkers
    error = False
    message = ''
    if build_tickers == None:
        if not isinstance(investment_amount, decimal.Decimal):
            message = 'Your investment amount should be a number'
            return True, message, 0,0,0,0
        elif type(number_of_stocks) != int:
            message = 'Your number of stocks should be a whole number'
            return True, message, 0,0,0,0

        if investment_amount < 0:
            message = 'You must invest more than one unit'
            return True, message, 0,0,0,0
        elif number_of_stocks <= 0:
            message = 'You must build your portfolio with more than one stock'
            return True, message, 0,0,0,0
        elif number_of_stocks > 50:
            message = 'You have chosen too many stocks, just invest in the sp500 at that point'
            return True, message, 0,0,0,0

    amount_of_tickers = 50
    if build_tickers:
        amount_of_tickers = number_of_stocks = len(build_tickers)

    tickers_price_df = order_tickers_by_returns(amount_of_tickers, build_tickers)
    tickers = tickers_price_df.columns.tolist() # Gives list of top tickers by returns
    mu2 = ema_historical_return(tickers_price_df, compounding=True, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)

    S2 = get_covariance(amount_of_tickers, build_tickers)

    mu2.name = None
    mu2 = mu2.fillna(0)

    exp_ret_sort = mu2.sort_values(ascending=False)
    exp_ret_sort.drop(exp_ret_sort.tail(len(exp_ret_sort) - number_of_stocks).index, inplace=True)

    tickers_return = exp_ret_sort.index.tolist()

    return_cov = exp_cov(tickers_price_df[tickers_return], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    
    variance_expon_list = []

    for ticker in tickers:
        variance_expon_list.append(S2[ticker][ticker])
    stdev_expon_list = pd.Series(data = np.sqrt(variance_expon_list), index = tickers)

    variance_expon_df = pd.DataFrame(data = variance_expon_list, index = tickers, columns = ['Deviation'])
    stdev_expon_df = np.sqrt(variance_expon_df)

    stdev_expon_sort = stdev_expon_list.sort_values(ascending=True)
    stdev_expon_sort.drop(stdev_expon_sort.tail(len(stdev_expon_sort)-number_of_stocks).index,inplace = True)

    sharpe = mu2/stdev_expon_df['Deviation']

    sharpe_expon_sort = sharpe.sort_values(ascending = False)
    sharpe_expon_sort.drop(sharpe_expon_sort.tail(len(sharpe_expon_sort)-number_of_stocks).index,inplace = True)

    tickers_sharpe = sharpe_expon_sort.index.tolist()

    sharpe_cov = exp_cov(tickers_price_df[tickers_sharpe], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    # Maximum Sharpe ratio
    ef_sharpe = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
    weights_sharpe = ef_sharpe.max_sharpe()
    weights_sharpe = ef_sharpe.clean_weights()
    # Maximum return for a given risk
    ef_return = EfficientFrontier(mu2[tickers_return], return_cov)
    
    try: 
        weights_return = ef_return.efficient_risk(target_volatility=float(min_var))
        weights_return = ef_return.clean_weights()
    except ValueError as e:
        return True, e, 0,0,0,0

    ef_sharpe.portfolio_performance(verbose=True)
    ef_return.portfolio_performance(verbose=True)

    ef_sharpe_plot = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
    ef_return_plot = EfficientFrontier(mu2[tickers_return], return_cov)

    risk_free_rate = 0.105 # defined by the avg return of the sp500
  
    n_samples = 500    
    # Generate random portfolios
    w = np.random.dirichlet(np.ones(ef_sharpe_plot.n_assets), n_samples)
    rets = w.dot(ef_sharpe_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_sharpe_plot.cov_matrix @ w.T))
    sharpes = (rets - 0.105) / stds
    # Prepare the dictionary
    data_dict_sharpe = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist(),
        'weights': w.tolist()
    }

    # Generate random portfolios
    w = np.random.dirichlet(np.ones(ef_return_plot.n_assets), n_samples)
    rets = w.dot(ef_return_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_return_plot.cov_matrix @ w.T))
    sharpes = (rets - risk_free_rate) / stds
    # Prepare the dictionary
    data_dict_return = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist(),
        'weights': w.tolist()
    }

    financial_data_sharpe = {}
    financial_data_return = {}
    for symbol in weights_sharpe:
        valuation_sharpe, finance_sharpe, financial_data_sharpe[symbol] = get_financial_data(symbol, investment_amount, weights_sharpe[symbol])
    for symbol in weights_return:
        valuation_return, finance_return, financial_data_return[symbol] = get_financial_data(symbol, investment_amount, weights_return[symbol])

    sharpe_dict = {'valuation_sharpe': valuation_sharpe, 
                   'finance_sharpe': finance_sharpe, 
                   'financial_data': financial_data_sharpe, 
                   'tickers': json.dumps(tickers_sharpe), 
                   'weights': str(dict(weights_sharpe))}
    return_dict = {'valuation_return': valuation_return, 
                   'finance_return': finance_return, 
                   'financial_data': financial_data_return, 
                   'tickers': json.dumps(tickers_return), 
                   'weights': str(dict(weights_return))}

    sharpe_data = json.dumps(data_dict_sharpe)
    return_data = json.dumps(data_dict_return)

    if build == True:
        return sharpe_data, sharpe_dict
    return error, message, sharpe_data, sharpe_dict, return_data, return_dict

def monte_carlo(ticker_or_weights):
    historical_data = None
    
    if isinstance(ticker_or_weights, dict):
        # Create an empty DataFrame to store weighted prices
        df = pd.DataFrame()
        for ticker, weight in ticker_or_weights.items():
            # Load ticker data; rename column; remove last row because it is sometimes NaN
            temp_df = pd.read_csv('tickers_prices.csv', usecols=['Date', ticker], parse_dates=['Date'])
            temp_df.rename(columns={ticker: 'Close'}, inplace=True)
            temp_df = temp_df[:-1]  # Remove the last row in case it's NaN
            
            # Handle NaN values in the temporary DataFrame
            temp_df['Close'].fillna(method='ffill', inplace=True)
            temp_df['Close'].fillna(method='bfill', inplace=True)
            
            if df.empty:
                df['Date'] = temp_df['Date']
                df['Weighted Close'] = temp_df['Close'] * weight
            else:
                df = pd.merge(df, temp_df[['Date', 'Close']], on='Date', suffixes=('', f'_{ticker}'))
                df['Weighted Close'] += temp_df['Close'] * weight

        # Handle NaN values by forward filling and then backward filling to handle any edge cases
        df['Weighted Close'].fillna(method='ffill', inplace=True)
        df['Weighted Close'].fillna(method='bfill', inplace=True)

        # Extract all historical data excluding the last day
        historical_data = df.iloc[:-1]
    else:
        # Load ticker data; rename column; remove last row because it is sometimes NaN
        df = pd.read_csv('tickers_prices.csv', usecols=['Date', ticker_or_weights], parse_dates=['Date'])
        df.rename(columns={ticker_or_weights: 'Close'}, inplace=True)
        df = df[:-1]  # Remove the last row in case it's NaN

    if df.empty:
        raise ValueError("The DataFrame is empty. Please check the input data.")

    number_simulation = 100
    predict_day = 30
    returns = df['Weighted Close'].pct_change() if 'Weighted Close' in df.columns else df['Close'].pct_change()
    volatility = returns.std()
    results = pd.DataFrame()
    last_date = df['Date'].iloc[-1]

    # Generate simulations
    for i in tqdm(range(number_simulation)):
        prices = [df['Weighted Close'].iloc[-1]] if 'Weighted Close' in df.columns else [df['Close'].iloc[-1]]
        for d in range(predict_day):
            price_today = prices[d] * (1 + np.random.normal(0, volatility))
            prices.append(price_today)
        results[i] = prices

    # Identify significant simulations based on final values
    final_values = results.iloc[-1]  # Gets the last row (final values of each simulation)
    sorted_indices = list(final_values.argsort())

    indices_to_keep = OrderedDict({
        'Lowest Value': sorted_indices[0],  # Lowest
        'Highest Value': sorted_indices[-1],  # Highest
        '90th Percentile': sorted_indices[int(len(sorted_indices) * 0.90)],  # 90th percentile
        '75th Percentile': sorted_indices[int(len(sorted_indices) * 0.75)],  # 75th percentile
        '50th Percentile': sorted_indices[int(len(sorted_indices) * 0.50)],  # 50th percentile
        '25th Percentile': sorted_indices[int(len(sorted_indices) * 0.25)]   # 25th percentile
    })

    # Convert OrderedDict to dict
    indices_to_keep = dict(indices_to_keep)

    # Preparing data for JSON conversion
    json_data = {'datasets': []}
    
    for label, index in indices_to_keep.items():
        dataset = {
            'label': label,
            'data': [{'x': (last_date + timedelta(days=d)).strftime('%Y-%m-%d'), 'y': results[index].iloc[d]}
                     for d in range(predict_day + 1)]  # ensure we include all days
        }
        json_data['datasets'].append(dataset)
    
    # Add historical data to JSON if applicable
    if historical_data is not None:
        historical_dataset = {
            'label': 'Weighted Portfolio',
            'data': [{'x': row['Date'].strftime('%Y-%m-%d'), 'y': row['Weighted Close']} for _, row in historical_data.iterrows()]
        }
        json_data['datasets'].append(historical_dataset)

    return json_data

def sentiment(tickers):
    nltk.download('vader_lexicon')

    sentiment_companies = finsent.get_all_stocks(tickers)

    return sentiment_companies.to_json(orient='records')
    
def build(request):   
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if request.GET.get('weights') is not None:
            weights = ast.literal_eval(request.GET.get('weights'))
            data = monte_carlo(weights)
            return JsonResponse(data)
        
        elif request.GET.get('tickers') is not None:
            tickers = ast.literal_eval(request.GET.get('tickers'))
            sentiment_companies = sentiment(tickers)
            return JsonResponse(sentiment_companies, safe=False)
        
        elif request.GET.get('build_tickers') is not None:
            build_tickers = request.GET.get('build_tickers')
            build_tickers = build_tickers.split(',')
            sharpe_data, sharpe_dict = get_portfolio(build_tickers=build_tickers, build=True)
            data = {'build_data': sharpe_data, 'build_dict': sharpe_dict}
            return JsonResponse(data)

    if request.method == 'POST':
        parameters = ParametersForm(request.POST)

        tickers_and_names = list(SP500Ticker.objects.all().values_list('symbol', 'name'))
        tickers = json.dumps([item[0] for item in tickers_and_names])
        names = json.dumps([item[1] for item in tickers_and_names])

        if parameters.is_valid():
            investment_amount = parameters.cleaned_data['amount']
            number_of_stocks = parameters.cleaned_data['amount_stocks']
            min_var = parameters.cleaned_data['min_var']

            error, error_message, sharpe_data, sharpe_dict, return_data, return_dict = get_portfolio(
                investment_amount=investment_amount, 
                number_of_stocks=number_of_stocks, 
                min_var=min_var)
            
            if error == True:
                return render(request, 'build.html', 
                              {'is_post': False, 
                               'error': True, 
                               'message': error_message})
            return render(request, 'build.html', 
                          {'is_post': True, 
                           'error': False, 
                           'message': '', 
                           'tickers': tickers,
                           'names': names,
                           'title': 'Efficient Frontier', 
                           'chart_type': 'scatter', 
                           'sharpe_data': sharpe_data, 
                           'sharpe_dict': sharpe_dict, 
                           'return_data': return_data, 
                           'return_dict': return_dict})
    
    error_message = ''
    return render(request, 'build.html', 
                  {'is_post': False, 
                   'error': False, 
                   'message': error_message})

def get_stock_data(ticker, sma, ema, rsi, bollinger_bands, macd, stochastic_oscillator):
    df = pd.read_csv('tickers_prices.csv', usecols=['Date', ticker], parse_dates=['Date'])

    # Rename the ticker column to 'close_price' for clarity
    df.rename(columns={ticker: 'close_price'}, inplace=True)
    
    # Drop the last day because it is sometimes NaN
    df = df[:-1]

    # Calculate Simple Moving Average (SMA)
    try:
        df['SMA'] = df['close_price'].rolling(window=sma).mean().fillna(df['close_price'])
    except Exception as e:
        df['SMA'] = 0
        print(f"Error calculating SMA: {e}")

    # Calculate Exponential Moving Average (EMA)
    try:
        df['EMA'] = df['close_price'].ewm(span=ema if isinstance(ema, int) else 0, adjust=False).mean().fillna(df['close_price'])
    except Exception as e:
        df['EMA'] = 0
        print(f"Error calculating EMA: {e}")

    # Calculate Relative Strength Index (RSI)
    try:
        delta = df['close_price'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi if isinstance(rsi, int) else 0).mean().fillna(0)
        avg_loss = loss.rolling(window=rsi if isinstance(rsi, int) else 0).mean().fillna(0)
        rs = avg_gain / avg_loss
        df['RSI'] = (100 - (100 / (1 + rs))).fillna(0)
    except Exception as e:
        df['RSI'] = 0
        print(f"Error calculating RSI: {e}")

    # Calculate MACD
    try:
        if isinstance(macd, str) and len(macd.split(',')) == 3 and all(num.isdigit() for num in macd.split(',')):
            fast, slow, signal = map(int, macd.split(','))
            fast_ema = df['close_price'].ewm(span=fast, adjust=False).mean()
            slow_ema = df['close_price'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = fast_ema - slow_ema
            df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        else:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
    except Exception as e:
        df['MACD'] = 0
        df['MACD_Signal'] = 0
        print(f"Error calculating MACD: {e}")

    # Calculate Bollinger Bands
    try:
        middle_band = df['close_price'].rolling(window=bollinger_bands if isinstance(bollinger_bands, int) else 0).mean()
        df['Middle_Band'] = middle_band.fillna(0)
        df['Upper_Band'] = (middle_band + 2 * df['close_price'].rolling(window=bollinger_bands if isinstance(bollinger_bands, int) else 0).std()).fillna(0)
        df['Lower_Band'] = (middle_band - 2 * df['close_price'].rolling(window=bollinger_bands if isinstance(bollinger_bands, int) else 0).std()).fillna(0)
    except Exception as e:
        df['Middle_Band'] = 0
        df['Upper_Band'] = 0
        df['Lower_Band'] = 0
        print(f"Error calculating Bollinger Bands: {e}")

    # Calculate Stochastic Oscillator
    try:
        if isinstance(stochastic_oscillator, str) and len(stochastic_oscillator.split(',')) == 3 and all(num.isdigit() for num in stochastic_oscillator.split(',')):
            k_period, d_period, smooth = map(int, stochastic_oscillator.split(','))
            low = df['low'].rolling(window=k_period).min()
            high = df['high'].rolling(window=d_period).max()
            df['%K'] = 100 * (df['close_price'] - low) / (high - low)
            df['%D'] = df['%K'].rolling(window=smooth).mean()
        else:
            df['%K'] = 0
            df['%D'] = 0
    except Exception as e:
        df['%K'] = 0
        df['%D'] = 0
        print(f"Error calculating Stochastic Oscillator: {e}")

    data = {
        'date': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'Close Price': df['close_price'].tolist(),
        'Simple Moving Average': [float(x) for x in df['SMA'].tolist()],
        'Exponential Moving Average': [float(x) for x in df['EMA'].tolist()],
        'Relative Strength Index': [float(x) for x in df['RSI'].tolist()],
        'MACD': [float(x) for x in df['MACD'].tolist()],
        'MACD Signal': [float(x) for x in df['MACD_Signal'].tolist()],
        'Middle Band': [float(x) for x in df['Middle_Band'].tolist()],
        'Upper Band': [float(x) for x in df['Upper_Band'].tolist()],
        'Lower Band': [float(x) for x in df['Lower_Band'].tolist()],
        'Fast Stochastic Indicator': [float(x) for x in df['%K'].tolist()],
        'Slow Stochastic Indicator': [float(x) for x in df['%D'].tolist()],
    }

    return json.dumps(data)

def data_output(ticker, data_type):
    if data_type == 'INCOME_STATEMENT':
        exist = IncomeStatement.objects.filter(symbol=ticker).exists()
        if not exist:
            response = requests.get(f'https://www.alphavantage.co/query?function={data_type}&symbol={ticker}&apikey={ALPHA_KEY}')
            data = response.json().get('annualReports', [])

            options = ['fiscalDateEnding', 'reportedCurrency', 'grossProfit', 'totalRevenue', 'costOfRevenue', 'operatingIncome', 'operatingExpenses', 'depreciation', 'netIncome']
            with open('output.csv', 'w') as f:
                f.write(f"{','.join(options)}\n")
                for entry in data:
                    line = ",".join(str(entry.get(key) if entry.get(key) is not None else 0) for key in options)
                    f.write(line + "\n")

            # Reading the CSV
            file = pd.read_csv('output.csv')
            
            # Ensure no NaN values slip through
            file.fillna(0, inplace=True)

            # Create IncomeStatement entries
            for row in file.itertuples():
                IncomeStatement.objects.create(
                    symbol=ticker,
                    fiscal_Date_Ending=datetime.strptime(row.fiscalDateEnding, '%Y-%m-%d'),
                    reported_Currency=row.reportedCurrency,
                    gross_Profit=row.grossProfit,
                    total_Revenue=row.totalRevenue,
                    cost_Of_Revenue=row.costOfRevenue,
                    operating_Income=row.operatingIncome,
                    operating_Expenses=row.operatingExpenses,
                    Depreciation=row.depreciation,
                    net_Income=row.netIncome
                )

        # Gathering data for visualization or further processing
        condition = {
            "symbol": f'{ticker}',
        }

        queryset = IncomeStatement.objects.filter(**condition)

        function = 'Income Statement'
        net_income_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.net_Income} for obj in queryset][::-1]     
        total_revenue_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.total_Revenue} for obj in queryset][::-1]
        cost_of_revenue_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.cost_Of_Revenue} for obj in queryset][::-1]
        operating_income_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.operating_Income} for obj in queryset][::-1]
        gross_profit_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.gross_Profit} for obj in queryset][::-1]
        operating_expenses_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.operating_Expenses} for obj in queryset][::-1]
        depreciation_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.Depreciation} for obj in queryset][::-1]

        return [function, ticker, json.dumps(net_income_data), json.dumps(total_revenue_data), json.dumps(cost_of_revenue_data), json.dumps(operating_income_data), json.dumps(gross_profit_data), json.dumps(operating_expenses_data), json.dumps(depreciation_data) ]
    elif data_type == 'BALANCE_SHEET':
        exist = BalanceSheet.objects.filter(symbol=ticker).exists()
        if not exist:
            response = requests.get(f'https://www.alphavantage.co/query?function={data_type}&symbol={ticker}&apikey={ALPHA_KEY}')
            data = response.json().get('annualReports', [])

            options = ['fiscalDateEnding', 'reportedCurrency', 'totalAssets', 'totalCurrentAssets', 'investments', 'currentDebt', 'treasuryStock', 'commonStock']
            with open('output.csv', 'w') as f:
                f.write(f"{','.join(options)}\n")
                for entry in data:
                    line = ",".join(str(entry.get(key, 0) if entry.get(key) is not None else 0) for key in options)
                    f.write(line + "\n")

            file = pd.read_csv('output.csv')
            file.fillna(0, inplace=True)  # Ensures no NaN values

            for row in file.itertuples():
                BalanceSheet.objects.create(
                    symbol=ticker,
                    fiscal_Date_Ending=datetime.strptime(row.fiscalDateEnding, '%Y-%m-%d'),
                    reported_Currency=row.reportedCurrency,
                    total_Assets=row.totalAssets or 0,
                    total_Current_Assets=row.totalCurrentAssets or 0,
                    Investments=row.investments or 0,
                    current_Debt=row.currentDebt or 0,
                    treasury_Stock=row.treasuryStock or 0,
                    common_Stock=row.commonStock or 0
                )

        condition = {
            "symbol": f'{ticker}',
        }

        queryset = BalanceSheet.objects.filter(**condition)

        function = 'Balance Sheet'
        total_assets_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.total_Assets} for obj in queryset][::-1]     
        total_current_assets_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.total_Current_Assets} for obj in queryset][::-1]
        investment_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.Investments} for obj in queryset][::-1]
        current_debt_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.current_Debt} for obj in queryset][::-1]
        treasury_stock_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.treasury_Stock} for obj in queryset][::-1]
        common_stock_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.common_Stock} for obj in queryset][::-1]

        return [function, ticker, json.dumps(total_assets_data), json.dumps(total_current_assets_data), json.dumps(investment_data), json.dumps(current_debt_data), json.dumps(treasury_stock_data), json.dumps(common_stock_data)]
    elif data_type == 'CASH_FLOW':
        exist = CashFlow.objects.filter(symbol=ticker).exists()
        if not exist:
            response = requests.get(f'https://www.alphavantage.co/query?function={data_type}&symbol={ticker}&apikey={ALPHA_KEY}')
            data = response.json().get('annualReports', [])

            # Prepare CSV file
            options = ['fiscalDateEnding', 'operatingCashflow', 'capitalExpenditures', 'changeInInventory', 'profitLoss', 'cashflowFromInvestment', 'cashflowFromFinancing', 'dividendPayout']
            with open('output.csv', 'w') as f:
                f.write(f"{','.join(options)}\n")
                for entry in data:
                    line = ",".join(str(entry.get(key, 0) if entry.get(key) is not None else 0) for key in options)
                    f.write(line + "\n")

            # Read the CSV file
            file = pd.read_csv('output.csv')
            file.fillna(0, inplace=True)  # Ensure no NaN values

            # Create CashFlow entries
            for row in file.itertuples():
                CashFlow.objects.create(
                    symbol=ticker,
                    fiscal_Date_Ending=datetime.strptime(row.fiscalDateEnding, '%Y-%m-%d'),
                    operating_Cashflow=row.operatingCashflow or 0,
                    capital_Expenditures=row.capitalExpenditures or 0,
                    change_In_Inventory=row.changeInInventory or 0,
                    profit_Loss=row.profitLoss or 0,
                    cashflow_From_Investment=row.cashflowFromInvestment or 0,
                    cashflow_From_Financing=row.cashflowFromFinancing or 0,
                    dividend_Payout=row.dividendPayout or 0
                )

        condition = {
            "symbol": f'{ticker}',
        }

        queryset = CashFlow.objects.filter(**condition)

        function = 'Cash Flow'
        operating_cashflow_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.operating_Cashflow} for obj in queryset][::-1]     
        capital_expenditures_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.capital_Expenditures} for obj in queryset][::-1]
        change_in_inventory_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.change_In_Inventory} for obj in queryset][::-1]
        profit_loss_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.profit_Loss} for obj in queryset][::-1]
        cashflow_from_investment_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.cashflow_From_Investment} for obj in queryset][::-1]
        cashflow_from_financing_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.cashflow_From_Financing} for obj in queryset][::-1]
        dividend_payout_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.dividend_Payout} for obj in queryset][::-1]
        return [function, ticker, json.dumps(operating_cashflow_data), json.dumps(capital_expenditures_data), json.dumps(change_in_inventory_data), json.dumps(profit_loss_data), json.dumps(cashflow_from_investment_data), json.dumps(cashflow_from_financing_data), json.dumps(dividend_payout_data)]
    elif data_type == 'EARNINGS':
        exist = Earnings.objects.filter(symbol=ticker).exists()
        if not exist:
            response = requests.get(f'https://www.alphavantage.co/query?function={data_type}&symbol={ticker}&apikey={ALPHA_KEY}')
            data = response.json().get('quarterlyEarnings', [])

            options = ['fiscalDateEnding', 'reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']
            with open('output.csv', 'w') as f:
                f.write(f"{','.join(options)}\n")
                for entry in data:
                    line = ",".join(str(entry.get(key, 0) if entry.get(key) is not None else 0) for key in options)
                    f.write(line + "\n")

            # Read the CSV file
            file = pd.read_csv('output.csv')
            file.fillna(0, inplace=True)  # Ensure no NaN values

            # Create Earnings entries
            for row in file.itertuples():
                Earnings.objects.create(
                    symbol=ticker,
                    fiscal_Date_Ending=datetime.strptime(row.fiscalDateEnding, '%Y-%m-%d'),
                    reported_eps=row.reportedEPS or 0,
                    estimated_eps=row.estimatedEPS or 0,
                    surprise=row.surprise or 0,
                    surprise_percentage=row.surprisePercentage or 0
                )

        condition = {
            "symbol": f'{ticker}',
        }

        queryset = Earnings.objects.filter(**condition)

        function = 'Earnings'
        reported_eps_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.reported_eps} for obj in queryset][::-1]     
        estimated_eps_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.estimated_eps} for obj in queryset][::-1]
        surprise_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.Surprise} for obj in queryset][::-1]
        surprise_percentage_data = [{'x': obj.fiscal_Date_Ending.strftime("%Y-%m-%d"), 'y': obj.surprise_percentage} for obj in queryset][::-1]
        return [function, ticker, json.dumps(reported_eps_data), json.dumps(estimated_eps_data), json.dumps(surprise_data), json.dumps(surprise_percentage_data)]

def trade(request):
    return render(request, 'trade.html')

def research(request):
    tickers_and_names = list(SP500Ticker.objects.all().values_list('symbol', 'name'))
    tickers = json.dumps([item[0] for item in tickers_and_names])
    names = json.dumps([item[1] for item in tickers_and_names])
    selected_ticker = None
    selected_ticker = request.GET.get('ticker', None)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':

        ticker = request.GET.get('ticker')
        if request.GET.get('action') == 'simulate':
            data = monte_carlo(ticker)
            return JsonResponse(data)
        else:
            data_type = request.GET.get('action')
            data_values = data_output(ticker=ticker, data_type=data_type)

            if data_type == 'INCOME_STATEMENT':
                keys = ['data_type','ticker','net_income_data','total_revenue_data','cost_of_revenue_data','operating_income_data','gross_profit_data','operating_expenses_data','depreciation_data']
            if data_type == 'BALANCE_SHEET':
                keys = ['data_type','ticker','total_assets_data','total_current_assets_data','investment_data','current_debt_data','treasury_stock_data','common_stock_data']
            elif data_type == 'CASH_FLOW':
                keys = ['data_type','ticker','operating_cashflow_data','capital_expenditures_data','change_in_inventory_data','profit_loss_data','cashflow_from_investments_data','cashflow_from_financing_data','dividend_payout_data']
            elif data_type == 'EARNINGS':
                keys = ['data_type','ticker','reported_eps_data','estimated_eps_data','surprise_data','surprise_percentage_data']

            data_response = {k: v for k, v in zip(keys, data_values)}

            return JsonResponse(data_response)

    # Handle form submissions
    if request.method == 'POST':
        research_form = ResearchForm(request.POST)
        if research_form.is_valid():
            selected_ticker = research_form.cleaned_data['ticker']
            industry = list(SP500Ticker.objects.filter(symbol=selected_ticker).values_list('industry', flat=True))[0]

            sma_value = 200 if research_form.cleaned_data.get('SMAValue', 50) is None else research_form.cleaned_data.get('SMAValue', 50)
            ema_value = 50 if research_form.cleaned_data.get('EMAValue', 50) is None else research_form.cleaned_data.get('EMAValue', 50)
            bollinger_bands_value = research_form.cleaned_data.get('BollingerBandsValue', 0)
            rsi_value = research_form.cleaned_data.get('RSIValue', 0)
            macd_value = research_form.cleaned_data.get('MACDValue', 0)
            stochastic_value = research_form.cleaned_data.get('StochasticValue', 0)

            chart_data = get_stock_data(selected_ticker, sma_value, ema_value, rsi_value, bollinger_bands_value, macd_value, stochastic_value)
            valuation, finance, financial_data = get_financial_data(selected_ticker)

            return render(request, 'research.html', {
                'is_post': True,
                'tickers': tickers,
                'names': names,
                'chart_data': chart_data,
                'title': selected_ticker,
                'industry': industry,
                'financial_data': financial_data,
                'valuation': valuation,
                'finance': finance,
            })
        else:
            print(research_form.errors)
    elif request.method == "GET" and selected_ticker is not None:
        sma_value = 200
        ema_value = 50 
        bollinger_bands_value = 0
        rsi_value = 0
        macd_value = 0
        stochastic_value = 0

        chart_data = get_stock_data(selected_ticker, sma_value, ema_value, rsi_value, bollinger_bands_value, macd_value, stochastic_value)
        valuation, finance, financial_data = get_financial_data(selected_ticker)

        return render(request, 'research.html', {
            'is_post': True,
            'tickers': tickers,
            'names': names,
            'chart_data': chart_data,
            'title': selected_ticker,
            'financial_data': financial_data,
            'valuation': valuation,
            'finance': finance,
        })
    
    # Default GET request
    chart_data = get_sp500_data() if not selected_ticker else get_stock_data(selected_ticker, 50, 50, 0, 0, 0, 0)
    valuation, finance, financial_data = get_financial_data(selected_ticker) if selected_ticker else ({}, {}, 'none')

    return render(request, 'research.html', {
        'is_post': False,
        'tickers': tickers,
        'names': names,
        'chart_data': chart_data,
        'title': selected_ticker or 'S&P 500',
        'financial_data': financial_data,
        'valuation': valuation,
        'finance': finance
    })

def pct_change(x,period=1):
    x = np.array(x)
    return ((x[period:] - x[:-period]) / x[:-period])

def indicator(request):
    return render(request, 'indicator.html')

