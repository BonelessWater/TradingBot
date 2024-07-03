from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
import logging
from .forms import ParametersForm, ResearchForm
from .models import CovarianceData, SP500Ticker, FinancialData
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

def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

def get_covariance():
    current_time = datetime.now().time()
    start_time = time(0, 0)  # 12:00 AM
    end_time = time(0, 30)  # 12:30 AM

    # Query all symbols from the database
    tickers = SP500Ticker.objects.all().values_list('symbol', flat=True)

    # Convert the QuerySet to a list and remove specific tickers
    tickers = list(tickers)
    
    if not tickers:
        raise ValueError("No tickers available after filtering.")

    tickers_str = ','.join(sorted(tickers))
    today = date.today()

    # Check if current time is between 12:00 AM and 12:30 AM
    if current_time >= start_time and current_time <= end_time:
        return HttpResponse("Daily calculations are currently under construction. Please try again after 12:30 AM UTC")

    # Check if the data already exists for today
    covariance_entry = CovarianceData.objects.filter(tickers=tickers_str, calculation_date=today).first()

    if covariance_entry:
        try:
            S2 = covariance_entry.deserialize_matrix(covariance_entry.covariance_matrix)
            return S2
        except AttributeError:
            pass

    # If not during the restricted time and no entry was found or deserialization failed
    csv_file_path = 'tickers_prices.csv'
    tickers_price_df = pd.read_csv(csv_file_path, index_col='Date', parse_dates=['Date'])

    # Ensure DataFrame is not empty
    if tickers_price_df.empty:
        raise ValueError("Tickers price DataFrame is empty.")

    # Handle missing values by dropping rows with any missing values
    tickers_price_df.dropna(inplace=True)

    # Ensure there are no missing values after dropping
    if tickers_price_df.isnull().values.any():
        raise ValueError("Tickers price DataFrame contains missing values even after dropping.")

    # Calculate covariance matrix
    try:
        S2 = exp_cov(tickers_price_df, frequency=tickers_price_df.shape[1], span=tickers_price_df.shape[1], log_returns=True)
        if S2.empty:
            raise ValueError("Calculated covariance matrix is empty.")
    except Exception as e:
        raise ValueError(f"Error calculating covariance matrix: {e}")

    # Save the new result in the database
    CovarianceData.objects.update_or_create(
        tickers=tickers_str,
        calculation_date=today,
        defaults={'covariance_matrix': S2.to_json()}
    )

    return S2

def get_portfolio(investment_amount, number_of_stocks, horizon, confidence_level, min_var):
    min_return = 0.0
    min_var = 0.8
    horizon = np.sqrt(horizon)

    z = NormalDist().inv_cdf(confidence_level)

    date_today = date.today()
    past_date = date_today - timedelta(days=3 * 365)
    
    # Retrieve distinct and ordered ticker symbols from your Django model
    #tickers = list(StockData.objects.values_list('symbol', flat=True).distinct().order_by('symbol'))
    #df = yf.download(tickers, past_date, date_today, auto_adjust=True)['Close']

    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.Symbol.to_list()

    # Remove specific tickers
    tickers_to_remove = ['BF.B', 'BRK.B', 'BF-B', 'BRK-B']
    tickers = [ticker for ticker in tickers if ticker not in tickers_to_remove]

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

    S2_symmetric = (S2 + S2.T) / 2
    
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

    tickers_risk = stdev_expon_sort.index.tolist()

    risk_cov = exp_cov(tickers_price_df[tickers_risk], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
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

    # Minimum risk for a given return
    ef_risk = EfficientFrontier(mu2[tickers_risk], risk_cov)
    weights_risk = ef_risk.efficient_return(target_return=min_return)
    ef_risk.clean_weights()

    # Maximum return for a given risk
    ef_return = EfficientFrontier(mu2[tickers_return], return_cov)
    
    weights_return = ef_return.efficient_risk(target_volatility=min_var)
    ef_return.clean_weights()

    ef_sharpe.portfolio_performance(verbose=True)
    ef_risk.portfolio_performance(verbose=True)
    ef_return.portfolio_performance(verbose=True)

    ef_sharpe_plot = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
    ef_risk_plot = EfficientFrontier(mu2[tickers_risk], risk_cov)
    ef_return_plot = EfficientFrontier(mu2[tickers_return], return_cov)

  
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef_sharpe_plot.n_assets), n_samples)
    rets = w.dot(ef_sharpe_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_sharpe_plot.cov_matrix @ w.T))
    sharpes = rets / stds
    # Prepare the dictionary
    data_dict_sharpe = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist() 
    }
    
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef_risk_plot.n_assets), n_samples)
    rets = w.dot(ef_risk_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_risk_plot.cov_matrix @ w.T))
    sharpes = rets / stds
    # Prepare the dictionary
    data_dict_risk = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist() 
    }
    
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef_return_plot.n_assets), n_samples)
    rets = w.dot(ef_return_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_return_plot.cov_matrix @ w.T))
    sharpes = rets / stds
    # Prepare the dictionary
    data_dict_return = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist() 
    }

    financial_data_sharpe = {}
    financial_data_risk = {}
    financial_data_return = {}
    for symbol in weights_sharpe:
        valuation_sharpe, finance_sharpe, financial_data_sharpe[symbol] = get_financial_data(symbol, investment_amount, weights_sharpe[symbol])
    for symbol in weights_risk:
        valuation_risk, finance_risk, financial_data_risk[symbol] = get_financial_data(symbol, investment_amount, weights_risk[symbol])
    for symbol in weights_return:
        valuation_return, finance_return, financial_data_return[symbol] = get_financial_data(symbol, investment_amount, weights_return[symbol])

    sharpe_dict = {'valuation_sharpe': valuation_sharpe, 'finance_sharpe': finance_sharpe, 'financial_data_sharpe': financial_data_sharpe}
    risk_dict = {'valuation_risk': valuation_risk, 'finance_risk': finance_risk, 'financial_data_risk': financial_data_risk}
    return_dict = {'valuation_return': valuation_return, 'finance_return': finance_return, 'financial_data_return': financial_data_return}

    sharpe_data = json.dumps(data_dict_sharpe)
    risk_data = json.dumps(data_dict_risk)
    return_data = json.dumps(data_dict_return)
    
    return sharpe_data, sharpe_dict, risk_data, risk_dict, return_data, return_dict

def parameters(request):    
    if request.method == 'POST':

        parameters = ParametersForm(request.POST)
        if parameters.is_valid():
            investment_amount = parameters.cleaned_data['amount']
            number_of_stocks = parameters.cleaned_data['amount_stocks']
            horizon = parameters.cleaned_data['horizon']
            confidence_level = parameters.cleaned_data['confidence']
            min_var = parameters.cleaned_data['min_var']
            
            sharpe_data, sharpe_dict, risk_data, risk_dict, return_data, return_dict = get_portfolio(investment_amount, number_of_stocks, horizon, confidence_level, min_var)

            return render(request, 'parameters.html', {'is_post': True, 'title': 'Efficient Frontier', 'chart_type': 'scatter', 'sharpe_data': sharpe_data, 'sharpe_dict': sharpe_dict, 'risk_data': risk_data, 'risk_dict': risk_dict, 'return_data': return_data, 'return_dict': return_dict})

    return render(request, 'parameters.html', {'is_post': False})

def get_stock_data(ticker, sma, ema, rsi, bollinger_bands, macd, stochastic_oscillator):
    df = pd.read_csv('tickers_prices.csv', usecols=['Date', ticker], parse_dates=['Date'])

    # Rename the ticker column to 'close_price' for clarity
    df.rename(columns={ticker: 'close_price'}, inplace=True)
    
    # Drop the last day because it is sometimes NaN
    df = df[:-1]

    # Calculate Simple Moving Average (SMA)
    try:
        df['SMA'] = df['close_price'].rolling(window=sma if isinstance(sma, int) else 0).mean().fillna(df['close_price'])
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
    
    tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers_df.Symbol.to_list()
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
            return render(request, 'research.html', {'tickers': tickers, 'chart_data': chart_data, 'title': selected_ticker, 'financial_data': financial_data, 'valuation': valuation, 'finance': finance})
        else:
            print(research_form.errors) 

    return render(request, 'research.html', {'tickers': tickers, 'chart_data': get_sp500_data(), 'title': 'S&P 500', 'financial_data': 'none'})

def indicator(request):
    return render(request, 'indicator.html')