from django.shortcuts import render
from .forms import ParametersForm
from .models import StockData
import yfinance as yf
import datetime
import json
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from datetime import date
from statistics import NormalDist

def main(request):
    return render(request, 'main.html')

def get_financial_data(symbol):
    stock = yf.Ticker(symbol)

    # Valuation Measures
    market_cap = stock.info.get('marketCap')
    enterprise_value = stock.info.get('enterpriseValue')
    trailing_pe = stock.info.get('trailingPE')
    forward_pe = stock.info.get('forwardPE')
    peg_ratio = stock.info.get('pegRatio')
    price_sales = stock.info.get('priceToSalesTrailing12Months')
    price_book = stock.info.get('priceToBook')
    ev_revenue = stock.info.get('enterpriseToRevenue')
    ev_ebitda = stock.info.get('enterpriseToEbitda')

    # Financial Highlights - Profitability and Income Statement
    profit_margin = stock.info.get('profitMargins')
    return_on_assets = stock.info.get('returnOnAssets')
    return_on_equity = stock.info.get('returnOnEquity')
    revenue = stock.info.get('totalRevenue')
    net_income = stock.info.get('netIncomeToCommon')
    diluted_eps = stock.info.get('trailingEps')

    # Financial Highlights - Balance Sheet and Cash Flow
    total_cash = stock.info.get('totalCash')
    debt_to_equity = stock.info.get('debtToEquity')
    levered_free_cash_flow = stock.info.get('freeCashflow')

    # Custom Financial Metrics
    earnings_per_share = stock.info.get('trailingEps')
    price_to_earnings_ratio = stock.info.get('trailingPE')
    dividend_yield = stock.info.get('dividendYield')
    book_value = stock.info.get('bookValue')
    debt_to_equity_ratio = stock.info.get('debtToEquity')
    revenue_growth = stock.info.get('revenueGrowth')
    free_cash_flow = stock.info.get('freeCashflow')
    return_on_equity = stock.info.get('returnOnEquity')

    data = {
        'Ticker': symbol,
        'MarketCap': market_cap,
        'EnterpriseValue': enterprise_value,
        'TrailingPE': trailing_pe,
        #'Forward P/E': forward_pe,
        #'PEG Ratio (5yr expected)': peg_ratio,
        #'Price/Sales (ttm)': price_sales,
        #'Price/Book (mrq)': price_book,
        #'Enterprise Value/Revenue': ev_revenue,
        #'Enterprise Value/EBITDA': ev_ebitda,
        #'Profit Margin': profit_margin,
        #'Return on Assets (ttm)': return_on_assets,
        #'Return on Equity (ttm)': return_on_equity,
        #'Revenue (ttm)': revenue,
        #'Net Income Avi to Common (ttm)': net_income,
        #'Diluted EPS (ttm)': diluted_eps,
        #'Total Cash (mrq)': total_cash,
        #'Total Debt/Equity (mrq)': debt_to_equity,
        #'Levered Free Cash Flow (ttm)': levered_free_cash_flow,
        #'Earnings Per Share': earnings_per_share,
        #'Price to Earnings Ratio': price_to_earnings_ratio,
        #'Dividend Yield': dividend_yield,
        #'Book Value': book_value,
        #'Debt to Equity Ratio': debt_to_equity_ratio,
        #'Revenue Growth': revenue_growth,
        #'Free Cash Flow': free_cash_flow,
        #'Return on Equity': return_on_equity
    }

    return data

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

    # Ensure the index is of datetime type
    sp500_data.index = pd.to_datetime(sp500_data.index)

    # Prepare the dictionary
    data_dict = {
        'x': [date.strftime('%Y-%m-%d') for date in sp500_data.index],  # Convert index to string for readability
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

    # Read tickers from Wikipedia
    tickers = StockData.objects.values_list('symbol', flat=True).distinct().order_by('symbol')

    # Query the database
    stock_prices = StockData.objects.filter(date__range=[past_date, date_today]).values('symbol', 'date', 'close_price')

    # Convert to DataFrame
    df = pd.DataFrame(list(stock_prices))

    # Ensure symbols are in uppercase
    df['symbol'] = df['symbol'].str.upper()


    # Pivot the DataFrame to match the format from Yahoo Finance
    tickers_price_df = df.pivot(index='date', columns='symbol', values='close_price')

    # Ensure we only consider tickers present in both the sources
    valid_tickers = list(set(tickers) & set(tickers_price_df.columns))

    # Recalculate the returns and covariance matrix
    tickers_price_df = tickers_price_df[valid_tickers]

    mu2 = ema_historical_return(tickers_price_df, compounding=True, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    S2 = exp_cov(tickers_price_df, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    
    mu2.name = None
    mu2 = mu2.fillna(0)

    S2_symmetric = (S2 + S2.T) / 2
    
    tickers_daily_volatility_df = np.log(1 + tickers_price_df.pct_change(fill_method=None))
    tickers_individual_volatility_df = pd.DataFrame(data=np.std(tickers_daily_volatility_df, axis=0), columns=['Individual Volatility'])

    avg_individual_volatility_df = pd.DataFrame(data=np.mean(tickers_daily_volatility_df, axis=0), columns=['Avg Individual Volatility'])
    var_individual_df = pd.DataFrame((avg_individual_volatility_df['Avg Individual Volatility'].mul(horizon)).sub((tickers_individual_volatility_df['Individual Volatility'].mul(z)).mul(horizon)), columns=['Individual VaR'])
    
    exp_ret = pd.DataFrame(data=mu2, index=valid_tickers, columns=['Exponential Returns'])

    exp_ret_sort = mu2.sort_values(ascending=False)
    exp_ret_sort.drop(exp_ret_sort.tail(len(exp_ret_sort) - number_of_stocks).index, inplace=True)

    tickers_returns = exp_ret_sort.index.tolist()

    cov_returns = exp_cov(tickers_price_df[tickers_returns], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    
    variance_expon_list = []

    for i in valid_tickers:
        if i in S2:
            variance_expon_list.append(S2[i][i])
        else:
            print(f"Ticker {i} not found in covariance matrix S2")
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


    financial_data = {}
    for symbol in weights:
        financial_data[symbol] = get_financial_data(symbol)

    return json.dumps(data_dict), financial_data

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
            
            chart_data, financial_data = get_portfolio(investment_amount, number_of_stocks, timeframe, horizon, confidence_level, min_var)
            print(financial_data)
            return render(request, 'parameters.html', {'title': 'Efficient Frontier', 'chart_data': chart_data, 'chart_type': 'scatter', 'financial_data': financial_data})
    chart_data = get_sp500_data()
    return render(request, 'parameters.html', {'title': 'S&P 500','chart_data': chart_data, 'chart_type': 'line'})

def get_stock_data(ticker):
    stock_data = StockData.objects.filter(symbol=ticker).order_by('date').values('date', 'close_price')
    
    # Arrange data into the desired format
    data = {'x': [], 'y': []}
    for entry in stock_data:
        data['x'].append(entry['date'].strftime('%Y-%m-%d'))
        data['y'].append(entry['close_price'])
    return json.dumps(data)

def research(request):
    tickers = StockData.objects.values_list('symbol', flat=True).distinct().order_by('symbol')
    if request.method == 'POST':
        selected_ticker = request.POST.get('ticker')
        chart_data = get_stock_data(selected_ticker)
        financial_data = get_financial_data(selected_ticker)
        print(financial_data)
        return render(request, 'research.html', {'tickers': tickers, 'chart_data': chart_data, 'title': selected_ticker, 'financial_data': financial_data})
    return render(request, 'research.html', {'tickers': tickers, 'chart_data': get_sp500_data(), 'title': 'S&P 500', 'financial_data': 'none'})
    