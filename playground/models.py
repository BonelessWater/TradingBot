from django.db import models
import numpy as np
class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    close_price = models.DecimalField(max_digits=10, decimal_places=3)
    volume = models.BigIntegerField()

    class Meta:
        unique_together = ('symbol', 'date')

class CovarianceData(models.Model):
    tickers = models.CharField(max_length=255)
    covariance_matrix = models.TextField()  # Store as a serialized string
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if isinstance(self.covariance_matrix, np.ndarray):
            self.covariance_matrix = self.serialize_matrix(self.covariance_matrix)
        super().save(*args, **kwargs)

    @staticmethod
    def serialize_matrix(matrix):
        return np.array2string(matrix, separator=',')

    @staticmethod
    def deserialize_matrix(matrix_str, size):
        return np.fromstring(matrix_str.strip('[]'), sep=',').reshape((size, size))