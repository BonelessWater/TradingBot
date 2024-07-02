from django.db import models
import numpy as np
from datetime import date
import pandas as pd
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