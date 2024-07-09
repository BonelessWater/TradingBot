from django.db import models
import numpy as np
from datetime import date
import pandas as pd
from django.utils import timezone

symbol_values = [
    ("VZ", "Verizon Communications"), ("KO", "The Coca-Cola Company"), ("NFLX", "Netflix"), ("ADBE", "Adobe Inc."), ("CSCO", "Cisco Systems"), ("XOM", "Exxon Mobil Corporation"), ("CMG", "Chipotle Mexican Grill"), ("SBUX", "Starbucks Corporation"), ("PFE", "Pfizer Inc."), ("CRM", "Salesforce"), 
    ("WMT", "Walmart Inc."), ("HD", "The Home Depot"), ("GE", "General Electric Company"), ("PEP", "PepsiCo, Inc."), ("T", "AT&T Inc."), ("FDX", "FedEx Corporation"), ("INTC", "Intel Corporation"), ("MU", "Micron Technology, Inc."), ("GM", "General Motors Company"), ("COST", "Costco Wholesale Corporation"), 
    ("TWTR", "Twitter, Inc."), ("MS", "Morgan Stanley"), ("CAT", "Caterpillar Inc."), ("MMM", "3M Company"), ("UPS", "United Parcel Service"), ("BKNG", "Booking Holdings Inc."), ("MCD", "McDonald's Corporation"), ("ABT", "Abbott Laboratories"), ("BMY", "Bristol Myers Squibb"), ("WBA", "Walgreens Boots Alliance, Inc."),
    ("IBM", "International Business Machines Corporation"), ("NVDA", "NVIDIA Corporation"), ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"), ("AMZN", "Amazon.com, Inc."), ("GOOGL", "Alphabet Inc."), ("FB", "Meta Platforms, Inc."), ("TSLA", "Tesla, Inc."), ("JPM", "JPMorgan Chase & Co."), ("JNJ", "Johnson & Johnson"), 
    ("V", "Visa Inc."), ("PG", "Procter & Gamble Company"), ("HD", "The Home Depot, Inc."), ("UNH", "UnitedHealth Group Incorporated"), ("MA", "Mastercard Incorporated"), ("BAC", "Bank of America Corp."), ("DIS", "The Walt Disney Company"), ("PYPL", "PayPal Holdings, Inc."), ("INTC", "Intel Corporation"), ("CMCSA", "Comcast Corporation"), 
    ("VZ", "Verizon Communications Inc."), ("KO", "The Coca-Cola Company"), ("NFLX", "Netflix, Inc."), ("ADBE", "Adobe Inc."), ("CSCO", "Cisco Systems, Inc."), ("XOM", "Exxon Mobil Corporation"), ("CMG", "Chipotle Mexican Grill, Inc."), ("SBUX", "Starbucks Corporation"), ("PFE", "Pfizer Inc."), ("CRM", "salesforce.com, inc."), 
    ("WMT", "Walmart Inc."), ("HD", "The Home Depot, Inc."), ("GE", "General Electric Company"), ("PEP", "PepsiCo, Inc."), ("T", "AT&T Inc."), ("FDX", "FedEx Corporation"), ("INTC", "Intel Corporation"), ("MU", "Micron Technology, Inc."), ("GM", "General Motors Company"), ("COST", "Costco Wholesale Corporation"), 
    ("TWTR", "Twitter, Inc."), ("MS", "Morgan Stanley"), ("CAT", "Caterpillar Inc."), ("MMM", "3M Company"), ("UPS", "United Parcel Service, Inc."), ("BKNG", "Booking Holdings Inc."), ("MCD", "McDonald's Corporation"), ("ABT", "Abbott Laboratories"), ("BMY", "Bristol Myers Squibb Company"), ("WBA", "Walgreens Boots Alliance, Inc."),
    ("NKE", "NIKE, Inc."), ("MO", "Altria Group, Inc."), ("BLK", "BlackRock, Inc."), ("IBM", "International Business Machines Corporation"), ("NVDA", "NVIDIA Corporation"), ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"), ("AMZN", "Amazon.com, Inc."), ("GOOGL", "Alphabet Inc."), ("FB", "Meta Platforms, Inc."), 
    ("TSLA", "Tesla, Inc."), ("JPM", "JPMorgan Chase & Co."), ("JNJ", "Johnson & Johnson"), ("V", "Visa Inc."), ("PG", "Procter & Gamble Company"), ("HD", "The Home Depot, Inc."), ("UNH", "UnitedHealth Group Incorporated"), ("MA", "Mastercard Incorporated"), ("BAC", "Bank of America Corp."), ("DIS", "The Walt Disney Company"), 
    ("PYPL", "PayPal Holdings, Inc."), ("INTC", "Intel Corporation"), ("CMCSA", "Comcast Corporation"), ("VZ", "Verizon Communications Inc."), ("KO", "The Coca-Cola Company"), ("NFLX", "Netflix, Inc."), ("ADBE", "Adobe Inc."), ("CSCO", "Cisco Systems, Inc."), ("XOM", "Exxon Mobil Corporation"), ("CMG", "Chipotle Mexican Grill, Inc."), 
    ("SBUX", "Starbucks Corporation"), ("PFE", "Pfizer Inc."), ("CRM", "salesforce.com, inc."), ("WMT", "Walmart Inc."), ("HD", "The Home Depot, Inc."), ("GE", "General Electric Company"), ("PEP", "PepsiCo, Inc."), ("T", "AT&T Inc."), ("FDX", "FedEx Corporation"), ("INTC", "Intel Corporation"), ("MU", "Micron Technology, Inc."), 
    ("GM", "General Motors Company"), ("COST", "Costco Wholesale Corporation"), ("TWTR", "Twitter, Inc."), ("MS", "Morgan Stanley"), ("CAT", "Caterpillar Inc."), ("MMM", "3M Company"), ("UPS", "United Parcel Service, Inc."), ("BKNG", "Booking Holdings Inc."), ("MCD", "McDonald's Corporation"), ("ABT", "Abbott Laboratories"), 
    ("BMY", "Bristol Myers Squibb Company"), ("WBA", "Walgreens Boots Alliance, Inc."), ("NKE", "NIKE, Inc."), ("MO", "Altria Group, Inc."), ("BLK", "BlackRock, Inc.")
]

class IncomeStatement(models.Model):
    symbol = models.CharField(max_length=5, choices=symbol_values, blank=True, null=True)
    fiscal_Date_Ending = models.DateField(blank=True, null=True)
    reported_Currency = models.CharField(max_length=4, blank=True, null=True)
    gross_Profit = models.IntegerField(blank=True, null=True)
    total_Revenue = models.IntegerField(blank=True, null=True)
    cost_Of_Revenue = models.IntegerField(blank=True, null=True)
    operating_Income = models.IntegerField(blank=True, null=True)
    operating_Expenses = models.IntegerField(blank=True, null=True)
    Depreciation = models.IntegerField(blank=True, null=True)
    net_Income = models.IntegerField(blank=True, null=True)
    
class BalanceSheet(models.Model):
    symbol = models.CharField(max_length=5, choices=symbol_values, blank=True, null=True)
    reported_Currency = models.CharField(max_length=4, blank=True, null=True)
    fiscal_Date_Ending = models.DateField(blank=True, null=True)
    total_Assets = models.IntegerField(blank=True, null=True)
    total_Current_Assets = models.IntegerField(blank=True, null=True)
    Investments = models.IntegerField(blank=True, null=True)
    current_Debt = models.IntegerField(blank=True, null=True)
    treasury_Stock = models.IntegerField(blank=True, null=True)
    common_Stock = models.IntegerField(blank=True, null=True)

class CashFlow(models.Model):
    symbol = models.CharField(max_length=5, choices=symbol_values, blank=True, null=True)
    fiscal_Date_Ending = models.DateField(blank=True, null=True)
    operating_Cashflow = models.IntegerField(blank=True, null=True)
    capital_Expenditures = models.IntegerField(blank=True, null=True)
    change_In_Inventory = models.IntegerField(blank=True, null=True)
    profit_Loss = models.IntegerField(blank=True, null=True)
    cashflow_From_Investment = models.IntegerField(blank=True, null=True)
    cashflow_From_Financing = models.IntegerField(blank=True, null=True)
    dividend_Payout = models.IntegerField(blank=True, null=True)
    
class Earnings(models.Model):
    symbol = models.CharField(max_length=5, choices=symbol_values, blank=True, null=True)
    fiscal_Date_Ending = models.DateField(blank=True, null=True)
    reported_eps = models.FloatField(blank=True, null=True)
    estimated_eps = models.FloatField(blank=True, null=True)
    Surprise = models.FloatField(blank=True, null=True)
    surprise_percentage = models.FloatField(blank=True, null=True)
    
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