"""
Tests for API endpoints.
"""
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status


class HealthCheckTest(TestCase):
    """Test health check endpoint."""
    
    def setUp(self):
        self.client = APIClient()
    
    def test_health_check(self):
        """Test health check returns 200."""
        url = reverse('health')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'healthy')

