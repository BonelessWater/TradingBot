import requests
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas import json_normalize

class finsent:
    def __init__(self, ticker):
        self.ticker = ticker

    # URL generating function:
    def ticker_feed(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = "https://finviz.com/quote.ashx?t=" + str(self.ticker) + '&p=d'
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to retrieve data for {self.ticker}. Status code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        news_table = soup.find(id='news-table')
        headlines = []
        
        if news_table:
            for row in news_table.find_all('tr'):
                headline = row.a.get_text() if row.a else None
                if headline:
                    headlines.append(headline)
        else:
            print(f"No news table found for {self.ticker}")

        return headlines
    
    # We define a DataFrame of headlines and we assess the sentiment for each
    def news_sent(self):
        headlines = self.ticker_feed()
        if not headlines:  # Handle case where there are no headlines
            return pd.DataFrame(columns=["Ticker", "Headline", "neg", "neu", "pos", "compound"])
        
        df = pd.DataFrame(data={"Ticker": self.ticker, "Headline": headlines})
        sia = SentimentIntensityAnalyzer()
        output = []
        for i in headlines:
            pol_score = sia.polarity_scores(i)
            output.append(pol_score)
        output = json_normalize(output)
        df_hdln = pd.DataFrame(data={"Ticker": self.ticker, "Headline": headlines})
        df_output = pd.concat([df_hdln, output], axis=1, sort=False)
        return df_output
    
    # This method returns the aggregated sentiment value of the reference company:
    def get_averages(self):
        base_df = self.news_sent()
        if base_df.empty:
            return pd.DataFrame(columns=["Ticker", "neg", "neu", "pos", "compound"])
        
        averages = base_df.mean(numeric_only=True)
        avg_df = pd.DataFrame(data=averages).T
        return avg_df
    
    # Method to build the df of a company's aggregated sentiment:
    def sentiment(self):
        ticker = self.ticker
        sentiment = self.get_averages()
        if 'Ticker' not in sentiment.columns:
            sentiment.insert(0, 'Ticker', ticker)
        else:
            sentiment['Ticker'] = ticker
        return sentiment
    
    # Method to build a df of all the companies in a specified list of tickers:
    @staticmethod
    def get_all_stocks(tickers):
        main = pd.DataFrame()
        for i in tickers:
            x = finsent(i)
            y = x.sentiment()
            main = pd.concat([main, y], ignore_index=True)

        return main