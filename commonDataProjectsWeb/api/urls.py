"""
API URL configuration.
"""
from django.urls import path
from . import views

app_name = 'api'  # Namespace

urlpatterns = [
    path('health/', views.health_check, name='health'),
    path('hello/', views.hello, name='hello'),
]

