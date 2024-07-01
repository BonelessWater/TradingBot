from django import forms

class ParametersForm(forms.Form):
    amount = forms.DecimalField(label='Investment amount')
    amount_stocks = forms.IntegerField(label='Number of Stocks')
    horizon = forms.IntegerField(label='Time horizon (months)')
    confidence = forms.DecimalField(label='VaR confidence level')
    min_var = forms.DecimalField(label='Min VaR')

class ResearchForm(forms.Form):
    ticker = forms.CharField()
    SMA = forms.BooleanField(required=False)
    SMAValue = forms.IntegerField(required=False)
    EMA = forms.BooleanField(required=False)
    EMAValue = forms.IntegerField(required=False)
    BollingerBands = forms.BooleanField(required=False)
    BollingerBandsValue = forms.IntegerField(required=False)
    RSI = forms.BooleanField(required=False)
    RSIValue = forms.IntegerField(required=False)
    MACD = forms.BooleanField(required=False)
    MACDValue = forms.CharField(required=False)
    StochasticOscillator = forms.BooleanField(required=False)
    StochasticValue = forms.CharField(required=False)