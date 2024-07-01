from django.shortcuts import render
from .forms import ParametersForm, ResearchForm
from .models import StockData, CovarianceData
import yfinance as yf
import datetime
import time
import json
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov
from datetime import datetime, timedelta, date
from statistics import NormalDist

def main(request):
    return render(request, 'main.html')

def get_financial_data(symbol, investment_amount = -1, weight = 0):
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

    valuation = {
        'ticker': symbol,
        'marketcap': market_cap,
        'enterprisevalue': enterprise_value,
        'trailingpe': trailing_pe,
        'forwardpe': forward_pe,
        'pegratio': peg_ratio,
        'pricesales': price_sales,
        'pricebook': price_book,
        'enterpriserevenue': ev_revenue,
        'enterpriseebitda': ev_ebitda,
    }
    
    finance = {
        'profit margin': profit_margin,
        'return on assets (ttm)': return_on_assets,
        'return on equity (ttm)': return_on_equity,
        'revenue (ttm)': revenue,
        'net income avi to common (ttm)': net_income,
        'diluted eps (ttm)': diluted_eps,
        'total cash (mrq)': total_cash,
        'total debt/equity (mrq)': debt_to_equity,
        'levered free cash flow (ttm)': levered_free_cash_flow,
    }
    
    data = {
        'ticker': symbol,
        'marketcap': market_cap,
        'enterprisevalue': enterprise_value,
        'trailingpe': trailing_pe,
        'forwardpe': forward_pe,
        'pegratio': peg_ratio,
        'pricesales': price_sales,
    }

    other = {
        'earnings per share': earnings_per_share,
        'price to earnings ratio': price_to_earnings_ratio,
        'dividend yield': dividend_yield,
        'book value': book_value,
        'debt to equity ratio': debt_to_equity_ratio,
        'revenue growth': revenue_growth,
        'free cash flow': free_cash_flow,
        'return on equity': return_on_equity
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
    start_date = end_date - timedelta(days=3*365)  # roughly 3 years
    
    try:
        # Fetch the data using yfinance
        sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        # Check if the 'Close' column exists
        if 'Close' in sp500_data.columns:
            print(sp500_data['Close'])
        else:
            print("Data retrieval was successful but 'Close' column is not present.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    sp500_data['50_MA'] = sp500_data['Close'].rolling(window=50).mean()
    
    # Drop NaN values
    sp500_data['50_MA'].fillna(sp500_data['Close'], inplace=True)

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

def get_covariance(tickers_price_df, tickers, calculation_date=None):
    if tickers_price_df.empty:
        raise ValueError("The DataFrame tickers_price_df is empty.")

    # Use the current date if no specific date is provided
    if calculation_date is None:
        calculation_date = date.today()

    # Convert tickers list to a sorted, comma-separated string
    tickers_str = ','.join(sorted(tickers))
    
    # Check if the data already exists for the specific date
    covariance_entry = CovarianceData.objects.filter(tickers=tickers_str, calculation_date=calculation_date).first()
 
    if covariance_entry:
        try:
            # Deserialize the matrix
            S2 = covariance_entry.deserialize_matrix(covariance_entry.covariance_matrix)
            print(f"Deserialized matrix shape: {S2.shape}")
        except ValueError as e:
            print(f"Deserialization error: {e}")
            covariance_entry.delete()
            covariance_entry = None

    if not covariance_entry:
        # Calculate the covariance matrix
        S2 = exp_cov(tickers_price_df, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
        
        # Ensure the calculated matrix is not empty before saving
        if S2.empty:
            raise ValueError("Calculated covariance matrix is empty.")
        
        # Save the result in the database
        covariance_entry = CovarianceData(
            tickers=tickers_str,
            covariance_matrix=S2,
            calculation_date=calculation_date
        )
        covariance_entry.save()
        print(f"Saved covariance matrix: {covariance_entry.covariance_matrix[:100]}... (truncated)")

    return S2

def get_portfolio(investment_amount, number_of_stocks, horizon, confidence_level, min_var):
    horizon = np.sqrt(horizon)

    z = NormalDist().inv_cdf(confidence_level)

    date_today = date.today()
    past_date = date_today - timedelta(days=3 * 365)
    
    # Retrieve distinct and ordered ticker symbols from your Django model
    #tickers = list(StockData.objects.values_list('symbol', flat=True).distinct().order_by('symbol'))
    #df = yf.download(tickers, past_date, date_today, auto_adjust=True)['Close']

    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.Symbol.to_list()

    tickers_price_df = yf.download(tickers, past_date, date_today, auto_adjust=True)['Close']

    

    mu2 = ema_historical_return(tickers_price_df, compounding=True, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)

    start = time.time()
    S2 = get_covariance(tickers_price_df, tickers)
    end = time.time()    
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

    tickers_returns = exp_ret_sort.index.tolist()

    cov_returns = exp_cov(tickers_price_df[tickers_returns], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
    
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
        valuation, finance, financial_data[symbol] = get_financial_data(symbol, investment_amount, weights[symbol])
    

    return json.dumps(data_dict), valuation, finance, financial_data

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
            
            chart_data, valuation, finance, financial_data = get_portfolio(investment_amount, number_of_stocks, horizon, confidence_level, min_var)
            print(financial_data)
            return render(request, 'parameters.html', {'title': 'Efficient Frontier', 'chart_data': chart_data, 'chart_type': 'scatter', 'financial_data': financial_data, 'valuation': valuation, 'finance': finance})
    chart_data = get_sp500_data()
    return render(request, 'parameters.html', {'title': 'S&P 500','chart_data': chart_data, 'chart_type': 'line'})

def get_stock_data(ticker, sma, ema, rsi, bollinger_bands, macd, stochastic_oscillator):
    stock_data = StockData.objects.filter(symbol=ticker).order_by('date').values('date', 'close_price', 'volume')
    df = pd.DataFrame(list(stock_data))

    # Ensure the date is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date')

    # Calculate Simple Moving Average (SMA)
    try:
        if isinstance(sma, int):
            df['SMA'] = df['close_price'].rolling(window=sma).mean().fillna(df['close_price'])
        else:
            df['SMA'] = 0
    except Exception as e:
        df['SMA'] = 0
        print(f"Error calculating SMA: {e}")

    # Calculate Exponential Moving Average (EMA)
    try:
        if isinstance(ema, int):
            df['EMA'] = df['close_price'].ewm(span=ema, adjust=False).mean().fillna(df['close_price'])
        else:
            df['EMA'] = 0
    except Exception as e:
        df['EMA'] = 0
        print(f"Error calculating EMA: {e}")

    # Calculate Relative Strength Index (RSI)
    try:
        if isinstance(rsi, int):
            delta = df['close_price'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi).mean().fillna(0)
            avg_loss = loss.rolling(window=rsi).mean().fillna(0)
            rs = avg_gain / avg_loss
            df['RSI'] = (100 - (100 / (1 + rs))).fillna(0)
        else:
            df['RSI'] = 0
    except Exception as e:
        df['RSI'] = 0
        print(f"Error calculating RSI: {e}")    

    # Calculate Moving Average Convergence Divergence (MACD)
    try:
        if macd and all(isinstance(int(number), int) for number in macd.split(',')):
            macd_values = [int(number) for number in macd.split(',')]
            ema1 = df['close_price'].ewm(span=macd_values[0], adjust=False).mean()
            ema2 = df['close_price'].ewm(span=macd_values[1], adjust=False).mean()
            df['MACD'] = ema1 - ema2
            df['MACD_Signal'] = df['MACD'].ewm(span=macd_values[2], adjust=False).mean()
        else:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
    except Exception as e:
        df['MACD'] = 0
        df['MACD_Signal'] = 0
        print(f"Error calculating MACD: {e}")

    # Calculate Bollinger Bands
    try:
        if isinstance(bollinger_bands, int):
            df['Middle_Band'] = df['close_price'].rolling(window=bollinger_bands).mean().fillna(0)
            df['Upper_Band'] = (df['Middle_Band'] + 2 * df['close_price'].rolling(window=bollinger_bands).std()).fillna(0)
            df['Lower_Band'] = (df['Middle_Band'] - 2 * df['close_price'].rolling(window=bollinger_bands).std()).fillna(0)
        else:
            df['Middle_Band'] = 0
            df['Upper_Band'] = 0
            df['Lower_Band'] = 0
    except Exception as e:
        df['Middle_Band'] = 0
        df['Upper_Band'] = 0
        df['Lower_Band'] = 0
        print(f"Error calculating Bollinger Bands: {e}")

    # Calculate Stochastic Oscillator
    try:
        if stochastic_oscillator and all(isinstance(int(number), int) for number in stochastic_oscillator.split(',')):
            so_values = [int(number) for number in stochastic_oscillator.split(',')]
            low = df['low'].rolling(window=so_values[0]).min()
            high = df['high'].rolling(window=so_values[1]).max()
            df['%K'] = (df['close_price'] - low) / (high - low) * 100
            df['%D'] = df['%K'].rolling(window=so_values[2]).mean()
        else:
            df['%K'] = 0
            df['%D'] = 0
    except Exception as e:
        df['%K'] = 0
        df['%D'] = 0
        print(f"Error calculating Stochastic Oscillator: {e}")

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close_price'].diff()) * df['volume']).fillna(0).cumsum()

    data = {
        'date': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'Close Price': df['close_price'].tolist(),
        'Simple Moving Average': df['SMA'].tolist(),
        'Exponential Moving Average': df['EMA'].tolist(),
        'Relative Strength Index': df['RSI'].tolist(),
        'MACD': df['MACD'].tolist(),
        'MACD Signal': df['MACD_Signal'].tolist(),
        'Middle Band': df['Middle_Band'].tolist(),
        'Upper Band': df['Upper_Band'].tolist(),
        'Lower Band': df['Lower_Band'].tolist(),
        'Fast Stochastic Indicator': df['%K'].tolist(),
        'Slow Stochastic Indicator': df['%D'].tolist(),
    }

    return json.dumps(data)

def research(request):
    
    tickers = StockData.objects.values_list('symbol', flat=True).distinct().order_by('symbol')
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
    print('invalid')
    return render(request, 'research.html', {'tickers': tickers, 'chart_data': get_sp500_data(), 'title': 'S&P 500', 'financial_data': 'none'})

def indicator(request):
    return render(request, 'indicator.html')