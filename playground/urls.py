from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

# URL CONFIG
urlpatterns = [
    
    # Robot.txt file for web crawlers
    path('robots933456.txt', robots_txt, name='robots_txt'),

    # menu links
    path('', main, name='main'),

    # Redirects
    path('parameters', parameters, name='parameters'),    
    path('research', research, name='research'),
    path('indicator', indicator, name='indicator'),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)