import json
import numpy as np
import pandas as pd
from datetime import date
from statistics import NormalDist
import time
from dateutil.relativedelta import relativedelta
import sqlite3
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import exp_cov

# Constants (these should be defined elsewhere or passed as parameters)
confidence_level = 0.95
number_of_stocks = 10
horizon = 10

# Connect to the SQLite database
conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

start_time = time.time()
horizon = np.sqrt(horizon)

date_today = date.today()
past_date = date_today - relativedelta(months=4)
z = NormalDist().inv_cdf(confidence_level)

print(f"Initialization Time: {time.time() - start_time} seconds")

# Query the database to get stock symbols and their close prices
start_time = time.time()
query = """
SELECT symbol, date, close_price
FROM playground_stockdata
WHERE date BETWEEN ? AND ?
"""
cursor.execute(query, (past_date, date_today))
rows = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=['symbol', 'date', 'close_price'])

# Get the unique tickers from the database
tickers = df['symbol'].unique()
print(f"Fetching Tickers from Database Time: {time.time() - start_time} seconds")

# Pivot the DataFrame to match the format from Yahoo Finance
start_time = time.time()
tickers_price_df = df.pivot(index='date', columns='symbol', values='close_price')
print(f"Database Query and DataFrame Creation Time: {time.time() - start_time} seconds")

# Calculating returns and covariances
start_time = time.time()
mu2 = ema_historical_return(tickers_price_df, compounding=True, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
S2 = exp_cov(tickers_price_df, frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
mu2.name = None
mu2 = mu2.fillna(0)
S2_symmetric = (S2 + S2.T) / 2
print(f"Calculating Returns and Covariances Time: {time.time() - start_time} seconds")

# Individual volatility
start_time = time.time()
tickers_daily_volatility_df = np.log(1 + tickers_price_df.pct_change(fill_method=None))
tickers_individual_volatility_df = pd.DataFrame(data=np.std(tickers_daily_volatility_df, axis=0), columns=['Individual Volatility'])
avg_individual_volatility_df = pd.DataFrame(data=np.mean(tickers_daily_volatility_df, axis=0), columns=['Avg Individual Volatility'])
var_individual_df = pd.DataFrame((avg_individual_volatility_df['Avg Individual Volatility'].mul(horizon)).sub((tickers_individual_volatility_df['Individual Volatility'].mul(z)).mul(horizon)), columns=['Individual VaR'])
print(f"Calculating Individual Volatility Time: {time.time() - start_time} seconds")

# Exponential returns and sorting
start_time = time.time()
exp_ret = pd.DataFrame(data=mu2, index=tickers, columns=['Exponential Returns'])
exp_ret_sort = mu2.sort_values(ascending=False)
exp_ret_sort.drop(exp_ret_sort.tail(len(exp_ret_sort) - number_of_stocks).index, inplace=True)
tickers_returns = exp_ret_sort.index.tolist()
cov_returns = exp_cov(tickers_price_df[tickers_returns], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
print(f"Calculating and Sorting Exponential Returns Time: {time.time() - start_time} seconds")

# Standard deviations and sorting
start_time = time.time()
variance_expon_list = [S2[i][i] for i in tickers]
stdev_expon_list = pd.Series(data=np.sqrt(variance_expon_list), index=tickers)
variance_expon_df = pd.DataFrame(data=variance_expon_list, index=tickers, columns=['Deviation'])
stdev_expon_df = np.sqrt(variance_expon_df)
stdev_expon_sort = stdev_expon_list.sort_values(ascending=True)
stdev_expon_sort.drop(stdev_expon_sort.tail(len(stdev_expon_sort) - number_of_stocks).index, inplace=True)
tickers_risk = stdev_expon_sort.index.tolist()
cov_risk = exp_cov(tickers_price_df[tickers_risk], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
print(f"Calculating and Sorting Standard Deviations Time: {time.time() - start_time} seconds")

# Sharpe ratio and optimization
start_time = time.time()
sharpe = mu2 / stdev_expon_df['Deviation']
sharpe_expon_df = pd.DataFrame(sharpe, columns=['Sharpe Ratio'])
sharpe_expon_sort = sharpe.sort_values(ascending=False)
sharpe_expon_sort.drop(sharpe_expon_sort.tail(len(sharpe_expon_sort) - number_of_stocks).index, inplace=True)
tickers_sharpe = sharpe_expon_sort.index.tolist()
sharpe_cov = exp_cov(tickers_price_df[tickers_sharpe], frequency=len(tickers_price_df), span=len(tickers_price_df), log_returns=True)
print(f"Calculating Sharpe Ratio and Optimization Time: {time.time() - start_time} seconds")

# Optimizing portfolio
start_time = time.time()
ef_sharpe = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
weights = ef_sharpe.max_sharpe()
cleaned_weights = ef_sharpe.clean_weights()
ef_sharpe.save_weights_to_file("weights.txt")
ef_sharpe.portfolio_performance(verbose=True)

ef_sharpe_plot = EfficientFrontier(mu2[tickers_sharpe], sharpe_cov)
print(f"Portfolio Optimization Time: {time.time() - start_time} seconds")

# Generating random portfolios
start_time = time.time()
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
print(f"Generating Random Portfolios Time: {time.time() - start_time} seconds")

# Close the database connection
conn.close()

# Return the result as JSON
json_result = json.dumps(data_dict)
