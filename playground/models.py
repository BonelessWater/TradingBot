from django.db import models
import numpy as np
from datetime import date
import pandas as pd
from django.utils import timezone

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    close_price = models.DecimalField(max_digits=10, decimal_places=3)
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('symbol', 'date')

    @staticmethod
    def flush():
        StockData.objects.all().delete()
        
class CovarianceData(models.Model):
    tickers = models.CharField(max_length=255)
    covariance_matrix = models.TextField()  # Store as a serialized string
    calculation_date = models.DateField(default=date.today)  # Store the date of the calculation with a default value
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if isinstance(self.covariance_matrix, pd.DataFrame):
            self.covariance_matrix = self.serialize_matrix(self.covariance_matrix)
        super().save(*args, **kwargs)

    @staticmethod
    def serialize_matrix(matrix):
        return matrix.to_json()

    @staticmethod
    def deserialize_matrix(matrix_str):
        return pd.read_json(matrix_str)

    @staticmethod
    def flush():
        CovarianceData.objects.all().delete()

class SP500Ticker(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=255)
    industry = models.CharField(max_length=255)

    def __str__(self):
        return self.symbol

    @staticmethod
    def flush():
        SP500Ticker.objects.all().delete()

class FinancialData(models.Model):
    created_at = models.DateTimeField(null=True, blank=True, default=timezone.now)
    ticker = models.CharField(max_length=10, unique=True)
    market_cap = models.BigIntegerField(null=True, blank=True)
    enterprise_value = models.BigIntegerField(null=True, blank=True)
    trailing_pe = models.FloatField(null=True, blank=True)
    forward_pe = models.FloatField(null=True, blank=True)
    peg_ratio = models.FloatField(null=True, blank=True)
    price_sales = models.FloatField(null=True, blank=True)
    price_book = models.FloatField(null=True, blank=True)
    ev_revenue = models.FloatField(null=True, blank=True)
    ev_ebitda = models.FloatField(null=True, blank=True)
    profit_margin = models.FloatField(null=True, blank=True)
    return_on_assets = models.FloatField(null=True, blank=True)
    return_on_equity = models.FloatField(null=True, blank=True)
    revenue = models.BigIntegerField(null=True, blank=True)
    net_income = models.BigIntegerField(null=True, blank=True)
    diluted_eps = models.FloatField(null=True, blank=True)
    total_cash = models.BigIntegerField(null=True, blank=True)
    debt_to_equity = models.FloatField(null=True, blank=True)
    levered_free_cash_flow = models.BigIntegerField(null=True, blank=True)
    earnings_per_share = models.FloatField(null=True, blank=True)
    price_to_earnings_ratio = models.FloatField(null=True, blank=True)
    dividend_yield = models.FloatField(null=True, blank=True)
    book_value = models.FloatField(null=True, blank=True)
    debt_to_equity_ratio = models.FloatField(null=True, blank=True)
    revenue_growth = models.FloatField(null=True, blank=True)
    free_cash_flow = models.BigIntegerField(null=True, blank=True)

    def __str__(self):
        return self.ticker
    
    @staticmethod
    def flush():
        FinancialData.objects.all().delete()