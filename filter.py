import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt.risk_models import exp_cov
from scipy.optimize import linprog

# Set up date range and fetch S&P 500 data
end_date = datetime.today()
start_date = end_date - timedelta(days=3 * 365)
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
sp500_returns = sp500_data.pct_change()  # Use percentage change for correlation calculation

# Process all stock data
stock_data = pd.read_csv('tickers_prices.csv', index_col='Date', parse_dates=['Date'])
stock_data = stock_data.loc[~stock_data.index.duplicated(keep='first')].sort_index()

# Calculate overall returns from start_date to end_date
total_returns = (stock_data.iloc[-1] / stock_data.iloc[0] - 1)

# Filter stocks that made more than 4% returns
highest_stock_data = stock_data.loc[:, total_returns > 0.04]

# Calculate daily returns for the filtered stocks and include S&P 500
daily_returns = highest_stock_data.pct_change().dropna()
daily_returns['SP500'] = sp500_returns.loc[daily_returns.index]  # Align indices and include S&P 500 returns

# Calculate the exponential covariance matrix
exp_cov_matrix = exp_cov(daily_returns, span=180)  # Using 180 days as the half-life

# Convert covariance matrix to correlation matrix
correlation_matrix = pd.DataFrame(np.corrcoef(exp_cov_matrix, rowvar=False),
                                  index=daily_returns.columns, columns=daily_returns.columns)

# Extract correlations of each stock with the S&P 500
sp500_correlations = correlation_matrix['SP500'].drop('SP500')

# Filter to get only stocks with negative correlations
negative_correlations_only = sp500_correlations[sp500_correlations < 0]
positive_correlations_only = sp500_correlations[sp500_correlations > 0.2]

print("Negative Correlations:")
print(negative_correlations_only.sort_values())

print("Positive Correlations:")
print(positive_correlations_only.sort_values())


#print(negative_correlations)
#print(positive_correlations)
# Combine positive and negative correlation dataframes
#
## Optimization setup
#n = 80
#correlation_array = top_80_correlations['Correlation'].values
#c = correlation_array  # Minimize the sum of weighted correlations
#bounds = [(1/(2*n), 5/(2*n)) for _ in range(n)]
#A_eq = [np.ones(n)]
#b_eq = [1]
#
#result = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
#
#if result.success:
#    print("Weights:", result.x)
#    print("Sum of weighted correlations:", np.dot(correlation_array, result.x))
#else:
#    print("Optimization failed:", result.message)