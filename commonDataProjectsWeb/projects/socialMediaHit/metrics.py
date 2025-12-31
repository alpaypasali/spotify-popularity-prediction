"""
Prometheus metrics for Social Media Hit Prediction service.
"""
from prometheus_client import Counter, Histogram, Gauge
import time

# ML Model Performance Metrics
spotify_hit_prediction_requests = Counter(
    'spotify_hit_prediction_requests_total',
    'Total number of Spotify hit prediction requests',
    ['status']  # success, error
)

spotify_hit_prediction_duration = Histogram(
    'spotify_hit_prediction_duration_seconds',
    'Duration of Spotify hit prediction requests in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

model_loading_status = Gauge(
    'spotify_model_loading_status',
    'Spotify model loading status (1=loaded, 0=not_loaded)'
)

model_loading_duration = Histogram(
    'spotify_model_loading_duration_seconds',
    'Duration of model loading in seconds'
)

# Business Metrics
playlist_requests = Counter(
    'spotify_playlist_requests_total',
    'Total number of playlist requests',
    ['status']
)

similar_songs_requests = Counter(
    'spotify_similar_songs_requests_total',
    'Total number of similar songs search requests',
    ['status']
)

similar_songs_duration = Histogram(
    'spotify_similar_songs_duration_seconds',
    'Duration of similar songs search in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

platform_stats_requests = Counter(
    'spotify_platform_stats_requests_total',
    'Total number of platform statistics requests'
)

# Dataset Metrics
dataset_size = Gauge(
    'spotify_dataset_size',
    'Number of songs in the dataset'
)

# Endpoint Latency Metrics (will be tracked via middleware)
endpoint_latency = Histogram(
    'spotify_endpoint_latency_seconds',
    'Endpoint latency in seconds',
    ['endpoint', 'method'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Prediction Score Metrics
prediction_scores = Histogram(
    'spotify_prediction_scores',
    'Distribution of prediction scores',
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

# Error Metrics
prediction_errors = Counter(
    'spotify_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

