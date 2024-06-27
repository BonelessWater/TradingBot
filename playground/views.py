from django.shortcuts import render
from .forms import ParametersForm
from .models import StockData
import yfinance as yf
import datetime
import json
import numpy as np
import plotly.express as px
import pandas as pd
import math
import statistics as stat
import datetime as dt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import date
from dateutil.relativedelta import relativedelta
from statistics import NormalDist

def main(request):
    return render(request, 'main.html')

def get_sp500_data():
    # Define the ticker symbol for S&P 500
    ticker_symbol = '^GSPC'
    
    # Calculate the start and end dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=3*365)  # roughly 3 years
    
    # Fetch the data using yfinance
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    sp500_data['50_MA'] = sp500_data['Close'].rolling(window=50).mean()
    
    # Drop NaN values
    sp500_data['50_MA'].fillna(sp500_data['Close'], inplace=True)

    # Prepare the dictionary
    data_dict = {
        'x': sp500_data.index.strftime('%Y-%m-%d').tolist(),  # Convert index to string for readability
        'y': sp500_data['Close'].tolist(),  # Use the 'Close' prices for y values
        'avg': sp500_data['50_MA'].tolist()  # Include the 50-day moving average
    }
    return json.dumps(data_dict)

def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

def get_portfolio(investment_amount, number_of_stocks, timeframe, horizon, confidence_level, min_var):
    horizon = np.sqrt(horizon)

    date_today = date.today()
    past_date = timeframe
    z = NormalDist().inv_cdf(confidence_level)

    tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

    tickers = tickers[~tickers['Symbol'].isin(['BF.B', 'BRK.B'])]['Symbol']

    # Query the database
    stock_prices = StockData.objects.filter(date__range=[past_date, date_today]).values('symbol', 'date', 'close_price')

    # Convert to DataFrame
    df = pd.DataFrame(list(stock_prices))

    # Pivot the DataFrame to match the format from Yahoo Finance
    tickers_price_df = df.pivot(index='date', columns='symbol', values='close_price')

    mu2 = ema_historical_return(tickers_price_df, compounding = True, frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    S2 = exp_cov(tickers_price_df, frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    mu2.name = None
    mu2 = mu2.fillna(0)

    S2_symmetric = (S2 + S2.T) / 2
    
    tickers_daily_volatility_df = np.log(1+tickers_price_df.pct_change(fill_method=None))
    tickers_individual_volatility_df = pd.DataFrame(data = np.std(tickers_daily_volatility_df, axis = 0), columns = ['Individual Volatility'])

    avg_individual_volatility_df = pd.DataFrame(data = np.mean(tickers_daily_volatility_df, axis = 0), columns = ['Avg Individual Volatility'])
    var_individual_df = pd.DataFrame((avg_individual_volatility_df['Avg Individual Volatility'].mul(horizon)).sub((tickers_individual_volatility_df['Individual Volatility'].mul(z)).mul(horizon)), columns = ['Individual VaR'])
    
    exp_ret = pd.DataFrame(data = mu2, index = tickers, columns = ['Exponential Returns'])

    exp_ret_sort = mu2.sort_values(ascending=False)
    exp_ret_sort.drop(exp_ret_sort.tail(len(exp_ret_sort)-number_of_stocks).index,inplace = True)

    tickers_returns = exp_ret_sort.index.tolist()

    cov_returns = exp_cov(tickers_price_df[tickers_returns], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    variance_expon_list = []
    for i in tickers:
        variance_expon_list.append(S2[i][i])
    stdev_expon_list = pd.Series(data = np.sqrt(variance_expon_list), index = tickers)


    variance_expon_df = pd.DataFrame(data = variance_expon_list, index = tickers, columns = ['Deviation'])
    stdev_expon_df = np.sqrt(variance_expon_df)

    stdev_expon_sort = stdev_expon_list.sort_values(ascending=True)
    stdev_expon_sort.drop(stdev_expon_sort.tail(len(stdev_expon_sort)-number_of_stocks).index,inplace = True)

    tickers_risk = stdev_expon_sort.index.tolist()

    cov_risk = exp_cov(tickers_price_df[tickers_risk], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    sharpe = mu2/stdev_expon_df['Deviation']
    sharpe_expon_df = pd.DataFrame(sharpe, columns = ['Sharpe Ratio'])

    sharpe_expon_sort = sharpe.sort_values(ascending = False)
    sharpe_expon_sort.drop(sharpe_expon_sort.tail(len(sharpe_expon_sort)-number_of_stocks).index,inplace = True)

    tickers_sharpe = sharpe_expon_sort.index.tolist()

    sharpe_cov = exp_cov(tickers_price_df[tickers_sharpe], frequency = len(tickers_price_df), span = len(tickers_price_df), log_returns = True)
    
    print('Maximizar Ratio de Sharpe')
    ##Optimizar portafolio para maximizar ratio de sharpe (retornos exponenciales)
    ef_sharpe = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
    weights = ef_sharpe.max_sharpe()
    cleaned_weights = ef_sharpe.clean_weights()
    ef_sharpe.save_weights_to_file("weights.txt")  # saves to file
    ef_sharpe.portfolio_performance(verbose=True)

    ef_sharpe_plot = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)

    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef_sharpe_plot.n_assets), n_samples)
    rets = w.dot(ef_sharpe_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_sharpe_plot.cov_matrix @ w.T))
    sharpes = rets / stds
    
  
    # Prepare the dictionary
    data_dict = {
        'x': stds.tolist(), 
        'y': rets.tolist(), 
        'sharpe': sharpes.tolist() 
    }
    print(data_dict['sharpe'])
    return json.dumps(data_dict)

def parameters(request):    
    if request.method == 'POST':

        parameters = ParametersForm(request.POST)
        if parameters.is_valid():
            investment_amount = parameters.cleaned_data['amount']
            number_of_stocks = parameters.cleaned_data['amount_stocks']
            timeframe = parameters.cleaned_data['time_frame']
            horizon = parameters.cleaned_data['horizon']
            confidence_level = parameters.cleaned_data['confidence']
            min_var = parameters.cleaned_data['min_var']
            
            chart_data = get_portfolio(investment_amount, number_of_stocks, timeframe, horizon, confidence_level, min_var)

            return render(request, 'parameters.html', {'title': 'Efficient Frontier','chart_data': chart_data, 'chart_type': 'scatter'})
    chart_data = get_sp500_data()
    return render(request, 'parameters.html', {'title': 'S&P 500','chart_data': chart_data, 'chart_type': 'line'})
        