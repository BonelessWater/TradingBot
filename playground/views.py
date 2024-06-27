from django.shortcuts import render

def main(request):
    return render(request, 'main.html')

def parameters(request):
    return render(request, 'parameters.html')