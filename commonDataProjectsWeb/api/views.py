"""
API views for microservice.
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.shortcuts import render


def home(request):
    """Home page view."""
    return render(request, 'home.html')


@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'service': 'django-microservice'
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
def hello(request):
    """Simple hello endpoint."""
    return Response({
        'message': 'Hello from Django Microservice!',
        'version': '1.0.0'
    }, status=status.HTTP_200_OK)

