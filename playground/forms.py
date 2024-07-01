from django import forms

class ParametersForm(forms.Form):
    amount = forms.DecimalField(label='Investment amount')
    amount_stocks = forms.IntegerField(label='Number of Stocks')
    time_frame = forms.DateField(label='Date time frame', widget=forms.DateInput(attrs={'type': 'date'}))
    horizon = forms.IntegerField(label='Time horizon (months)')
    confidence = forms.DecimalField(label='VaR confidence level')
    min_var = forms.DecimalField(label='Min VaR')

class ResearchForm(forms.Form):
    ticker = forms.CharField()
    SMA = forms.IntegerField(required=False)
    EMA = forms.IntegerField(required=False)
    BollingerBands = forms.BooleanField(required=False)
    RSI = forms.BooleanField(required=False)
    MACD = forms.BooleanField(required=False)
    StochasticOscillator = forms.BooleanField(required=False)