from django.test import TestCase
from .models import CovarianceData, SP500Ticker
import logging
from datetime import date
from pypfopt.risk_models import exp_cov
import pandas as pd

logger = logging.getLogger(__name__)

class CovTest(TestCase):
    def setUp(self):
        

        try:
            # Query all symbols from the database
            tickers = list(SP500Ticker.objects.all().order_by('id').values_list('symbol', flat=True))
            print(SP500Ticker.objects.count())

            if not tickers:
                raise ValueError("No tickers available after filtering.")

            tickers_str = ','.join(sorted(tickers))
            today = date.today()

            # If not during the restricted time and no entry was found or deserialization failed
            csv_file_path = 'tickers_prices.csv'
            try:
                tickers_price_df = pd.read_csv(csv_file_path, index_col='date', parse_dates=['date'])
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

        except Exception as e:
            logger.error(f"Unexpected error in get_covariance: {e}")
            raise 
    
    def test_cov(self):
        tickers = list(SP500Ticker.objects.all().order_by('id').values_list('symbol', flat=True))
        tickers_str = ','.join(sorted(tickers))

        covariance_entry = CovarianceData.objects.filter(tickers=tickers_str).order_by('-calculation_date').first()
        test_S2 = covariance_entry.deserialize_matrix(covariance_entry.covariance_matrix)
        
        csv_file_path = 'tickers_prices.csv'

        tickers_price_df = pd.read_csv(csv_file_path, index_col='date', parse_dates=['date'])
        tickers_price_df.fillna(value=0, inplace=True)
        
        S2 = exp_cov(tickers_price_df, frequency=252, span=60, log_returns=True)  # Adjust frequency and span as needed
        S2 = (S2 + S2.T) / 2  # Ensure symmetry

        self.assertEqual(S2, test_S2)