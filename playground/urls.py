from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

# URL CONFIG
urlpatterns = [
    
    # menu links
    path('', main, name='main'),
    path('parameters', parameters, name='parameters'),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)