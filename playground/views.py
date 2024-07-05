from django.shortcuts import render
from django.http import HttpResponse
from .forms import ParametersForm, ResearchForm
from .models import CovarianceData, SP500Ticker, FinancialData
import logging
import yfinance as yf
import time as t
import json
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from datetime import datetime, timedelta, date, time
from statistics import NormalDist
from datetime import datetime, timedelta
import decimal

logger = logging.getLogger(__name__)

def main(request):
    #CovarianceData.flush()
    #SP500Ticker.flush()
    #store_tickers_in_db()
    return render(request, 'main.html')

def get_financial_data(symbol, investment_amount = -1, weight = 0):

    # Query the database for the specified ticker
    financial_data = FinancialData.objects.get(ticker=symbol)
    
    # Populate the 'valuation' dictionary
    valuation = {
        'ticker': financial_data.ticker,
        'marketcap': financial_data.market_cap,
        'enterprisevalue': financial_data.enterprise_value,
        'trailingpe': financial_data.trailing_pe,
        'forwardpe': financial_data.forward_pe,
        'pegratio': financial_data.peg_ratio,
        'pricesales': financial_data.price_sales,
        'pricebook': financial_data.price_book,
        'enterpriserevenue': financial_data.ev_revenue,
        'enterpriseebitda': financial_data.ev_ebitda,
    }
    
    # Populate the 'finance' dictionary
    finance = {
        'profit margin': financial_data.profit_margin,
        'return on assets (ttm)': financial_data.return_on_assets,
        'return on equity (ttm)': financial_data.return_on_equity,
        'revenue (ttm)': financial_data.revenue,
        'net income avi to common (ttm)': financial_data.net_income,
        'diluted eps (ttm)': financial_data.diluted_eps,
        'total cash (mrq)': financial_data.total_cash,
        'total debt/equity (mrq)': financial_data.debt_to_equity,
        'levered free cash flow (ttm)': financial_data.levered_free_cash_flow,
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

def get_covariance():
    try:
        current_time = datetime.now().time()
        start_time = time(0, 0)  # 12:00 AM
        end_time = time(0, 5)  # 12:05 AM

        # Query all symbols from the database
        tickers = list(SP500Ticker.objects.all().order_by('id').values_list('symbol', flat=True))
        
        if not tickers:
            raise ValueError("No tickers available after filtering.")

        tickers_str = ','.join(sorted(tickers))
        today = date.today()

        # Check if current time is between 12:00 AM and 12:05 AM
        if start_time <= current_time <= end_time:
            return HttpResponse("Daily calculations are currently under construction. Please try again after 12:05 AM UTC")

        # Check if the data already exists for today
        covariance_entry = CovarianceData.objects.filter(tickers=tickers_str).first()
        print("didnt have to do allat")

        if covariance_entry:
            try:
                S2 = covariance_entry.deserialize_matrix(covariance_entry.covariance_matrix)
                return S2
            except AttributeError:
                logger.error("Error deserializing covariance matrix from the database.")

        # If not during the restricted time and no entry was found or deserialization failed
        csv_file_path = 'tickers_prices.csv'
        try:
            tickers_price_df = pd.read_csv(csv_file_path, index_col='Date', parse_dates=['Date'])
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

def get_portfolio(investment_amount, number_of_stocks, horizon, min_var):
    confidence_level = 0.999
    # Error checkers
    error = False
    message = ''

    if not isinstance(investment_amount, decimal.Decimal):
        message = 'Your investment amount should be a number'
        return True, message, 0,0,0,0
    elif type(number_of_stocks) != int:
        message = 'Your number of stocks should be a whole number'
        return True, message, 0,0,0,0
    elif type(horizon) != int or type(horizon) == float:
        message = 'Your investment horizon should be a number'
        return True, message, 0,0,0,0

    horizon = round(horizon)

    if confidence_level < 0 or 1 < confidence_level:
        message = 'Your confidence level should be a number between 0 and 1'
        return True, message, 0,0,0,0
    elif investment_amount < 0:
        message = 'You must invest more than one unit'
        return True, message, 0,0,0,0
    elif number_of_stocks <= 0:
        message = 'You must build your portfolio with more than one stock'
        return True, message, 0,0,0,0
    elif number_of_stocks > 50:
        message = 'You have chosen too many stocks, just invest in the sp500 at that point'
        return True, message, 0,0,0,0
    elif horizon <= 0:
        message = 'What are you trying to look into the past? Choose a larger horizon'
        return True, message, 0,0,0,0
    
    confidence_level = float(confidence_level)
    
    horizon = np.sqrt(horizon)
    
    z = NormalDist().inv_cdf(confidence_level)

    date_today = date.today()
    past_date = date_today - timedelta(days=3 * 365)

    tickers = list(SP500Ticker.objects.all().order_by('id').values_list('symbol', flat=True))

    csv_file_path = 'tickers_prices.csv'

    # Read the DataFrame back from the CSV file
    tickers_price_df = pd.read_csv(csv_file_path, index_col='Date', parse_dates=['Date'])

    mu2 = ema_historical_return(tickers_price_df, compounding=True, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)

    start = t.time()
    S2 = get_covariance()
    end = t.time()    
    print(end-start)

    mu2.name = None
    mu2 = mu2.fillna(0)
    
    tickers_daily_volatility_df = np.log(1 + tickers_price_df.pct_change(fill_method=None))
    tickers_individual_volatility_df = pd.DataFrame(data=np.std(tickers_daily_volatility_df, axis=0), columns=['Individual Volatility'])

    avg_individual_volatility_df = pd.DataFrame(data=np.mean(tickers_daily_volatility_df, axis=0), columns=['Avg Individual Volatility'])
    var_individual_df = pd.DataFrame((avg_individual_volatility_df['Avg Individual Volatility'].mul(horizon)).sub((tickers_individual_volatility_df['Individual Volatility'].mul(z)).mul(horizon)), columns=['Individual VaR'])
    
    exp_ret = pd.DataFrame(data=mu2, index=tickers, columns=['Exponential Returns'])

    exp_ret_sort = mu2.sort_values(ascending=False)
    exp_ret_sort.drop(exp_ret_sort.tail(len(exp_ret_sort) - number_of_stocks).index, inplace=True)

    tickers_return = exp_ret_sort.index.tolist()

    return_cov = exp_cov(tickers_price_df[tickers_return], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    
    variance_expon_list = []

    for i in tickers:
        variance_expon_list.append(S2[i][i])
    stdev_expon_list = pd.Series(data = np.sqrt(variance_expon_list), index = tickers)

    variance_expon_df = pd.DataFrame(data = variance_expon_list, index = tickers, columns = ['Deviation'])
    stdev_expon_df = np.sqrt(variance_expon_df)

    stdev_expon_sort = stdev_expon_list.sort_values(ascending=True)
    stdev_expon_sort.drop(stdev_expon_sort.tail(len(stdev_expon_sort)-number_of_stocks).index,inplace = True)

    sharpe = mu2/stdev_expon_df['Deviation']
    sharpe_expon_df = pd.DataFrame(sharpe, columns = ['Sharpe Ratio'])

    sharpe_expon_sort = sharpe.sort_values(ascending = False)
    sharpe_expon_sort.drop(sharpe_expon_sort.tail(len(sharpe_expon_sort)-number_of_stocks).index,inplace = True)

    tickers_sharpe = sharpe_expon_sort.index.tolist()

    sharpe_cov = exp_cov(tickers_price_df[tickers_sharpe], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    # Maximum Sharpe ratio
    ef_sharpe = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
    weights_sharpe = ef_sharpe.max_sharpe()
    ef_sharpe.clean_weights()

    # Maximum return for a given risk
    ef_return = EfficientFrontier(mu2[tickers_return], return_cov)
    
    try: 
        weights_return = ef_return.efficient_risk(target_volatility=float(min_var))
    except ValueError as e:
        return True, e, 0,0,0,0
    ef_return.clean_weights()

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

    sharpe_dict = {'valuation_sharpe': valuation_sharpe, 'finance_sharpe': finance_sharpe, 'financial_data_sharpe': financial_data_sharpe, 'tickers': json.dumps(tickers_sharpe)}
    return_dict = {'valuation_return': valuation_return, 'finance_return': finance_return, 'financial_data_return': financial_data_return, 'tickers': json.dumps(tickers_return)}

    sharpe_data = json.dumps(data_dict_sharpe)
    return_data = json.dumps(data_dict_return)
    
    return error, message, sharpe_data, sharpe_dict, return_data, return_dict

def parameters(request):    
    if request.method == 'POST':

        parameters = ParametersForm(request.POST)
        if parameters.is_valid():
            investment_amount = parameters.cleaned_data['amount']
            number_of_stocks = parameters.cleaned_data['amount_stocks']
            horizon = parameters.cleaned_data['horizon']
            min_var = parameters.cleaned_data['min_var']

            error, error_message, sharpe_data, sharpe_dict, return_data, return_dict = get_portfolio(investment_amount, number_of_stocks, horizon, min_var)
            if error == True:
                return render(request, 'parameters.html', {'is_post': False, 'error': True, 'message': error_message})
            return render(request, 'parameters.html', {'is_post': True, 'error': False, 'message': '', 'title': 'Efficient Frontier', 'chart_type': 'scatter', 'sharpe_data': sharpe_data, 'sharpe_dict': sharpe_dict, 'return_data': return_data, 'return_dict': return_dict})
    error_message = ''
    return render(request, 'parameters.html', {'is_post': False, 'error': False, 'message': error_message})

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

def research(request):
    
    tickers_and_names = list(SP500Ticker.objects.all().values_list('symbol', 'name')) # Query them together to not lose ordering
    
    tickers = json.dumps([item[0] for item in tickers_and_names])
    names = json.dumps([item[1] for item in tickers_and_names])

    if request.method == 'POST':
        
        research_form = ResearchForm(request.POST)
        if research_form.is_valid():
            selected_ticker = research_form.cleaned_data['ticker']
            sma_value = research_form.cleaned_data.get('SMAValue')
            ema_value = research_form.cleaned_data.get('EMAValue')
            bollinger_bands_value = research_form.cleaned_data.get('BollingerBandsValue')
            rsi_value = research_form.cleaned_data.get('RSIValue')
            macd_value = research_form.cleaned_data.get('MACDValue')
            stochastic_value = research_form.cleaned_data.get('StochasticValue')

            chart_data = get_stock_data(selected_ticker, sma_value, ema_value, rsi_value, bollinger_bands_value, macd_value, stochastic_value)

            valuation, finance, financial_data = get_financial_data(selected_ticker)


            return render(request, 'research.html', {'tickers': tickers, 'names': names, 'chart_data': chart_data, 'title': selected_ticker, 'financial_data': financial_data, 'valuation': valuation, 'finance': finance})
        else:
            print(research_form.errors) 

    return render(request, 'research.html', {'tickers': tickers, 'names': names, 'chart_data': get_sp500_data(), 'title': 'S&P 500', 'financial_data': 'none'})

def indicator(request):
    return render(request, 'indicator.html')