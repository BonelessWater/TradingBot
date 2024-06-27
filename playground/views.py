from django.shortcuts import render
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
    
    # Prepare the dictionary
    data_dict = {
        'x': sp500_data.index.strftime('%Y-%m-%d').tolist(),  # Convert index to string for readability
        'y': sp500_data['Close'].tolist()  # Use the 'Close' prices for y values
    }
    return json.dumps(data_dict)

def parameters(request):
    chart_data = get_sp500_data()
    #print(chart_data)
    return render(request, 'parameters.html', {'chart_data': chart_data})