from django import forms

class ParametersForm(forms.Form):
    investment_amount = forms.IntegerField()
    number_of_stocks = forms.IntegerField()
    timeframe = forms.DateField()
    horizon = forms.IntegerField()
    confidence_level = forms.FloatField()
    min_var = forms.FloatField()

    