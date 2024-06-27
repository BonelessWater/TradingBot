from django.shortcuts import render
from .forms import ParametersForm
import yfinance as yf
import datetime
import json

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

def get_portfolio():
    parameters = ParametersForm()

def parameters(request):    
    if request.method == 'POST':

        chart_data = get_portfolio()

        return render(request, 'parameters.html', {'chart_data': chart_data, 'chart_type': 'scatter'})
    else: 
        chart_data = get_sp500_data()
        return render(request, 'parameters.html', {'chart_data': chart_data, 'chart_type': 'line'})
        