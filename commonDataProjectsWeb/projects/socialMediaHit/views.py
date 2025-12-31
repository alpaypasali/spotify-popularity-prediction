from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
import json
import time
try:
    from .metrics import (
        spotify_hit_prediction_requests,
        spotify_hit_prediction_duration,
        playlist_requests,
        similar_songs_requests,
        similar_songs_duration,
        platform_stats_requests,
        endpoint_latency,
        prediction_errors
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    # Dummy metrics to prevent errors
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    spotify_hit_prediction_requests = DummyMetric()
    spotify_hit_prediction_duration = DummyMetric()
    playlist_requests = DummyMetric()
    similar_songs_requests = DummyMetric()
    similar_songs_duration = DummyMetric()
    platform_stats_requests = DummyMetric()
    endpoint_latency = DummyMetric()
    prediction_errors = DummyMetric()


def home(request):
    """Project home page - About section."""
    return render(request, 'socialMediaHit/home.html')


def spotify_prediction(request):
    """Spotify Hit Prediction interactive page."""
    return render(request, 'socialMediaHit/spotify_prediction.html')


def instagram_picture_prediction(request):
    """Instagram Reels Picture Hit Prediction page."""
    return render(request, 'socialMediaHit/instagram_picture.html')


def instagram_caption_prediction(request):
    """Instagram Reels Caption/Hashtag Hit Prediction page."""
    return render(request, 'socialMediaHit/instagram_caption.html')


def final_prediction(request):
    """Final combination hit score prediction page."""
    return render(request, 'socialMediaHit/final_prediction.html')


# API Endpoints
@api_view(['GET'])
def health(request):
    """Health check endpoint."""
    return Response({
        'status': 'healthy',
        'project': 'socialMediaHit'
    }, status=status.HTTP_200_OK)


# Singleton service instance (shared across all requests)
_spotify_service = None

def get_spotify_service():
    """Get singleton SpotifyPredictionService instance."""
    global _spotify_service
    if _spotify_service is None:
        from .services import SpotifyPredictionService
        _spotify_service = SpotifyPredictionService()
    return _spotify_service


@api_view(['GET'])
def get_platform_stats(request):
    """Get platform statistics for homepage."""
    start_time = time.time()
    try:
        if METRICS_AVAILABLE:
            platform_stats_requests.inc()
        service = get_spotify_service()
        
        # Get actual track count from dataset
        track_count = len(service.df) if service.df is not None and not service.df.empty else 0
        
        # ML Models count (XGBoost, Random Forest, GBM, LightGBM)
        ml_models_count = 4  # XGBoost, RandomForest, GBM, LightGBM
        
        # Markets count - get from services.py _calculate_market_factor method
        # The actual markets supported are defined in base_factors dict
        markets = ['US', 'GB', 'DE', 'FR', 'JP', 'BR']  # From services.py line 530
        markets_count = len(markets)
        
        response = Response({
            'tracks_analyzed': track_count,
            'ml_models': ml_models_count,
            'markets': markets_count,
            'markets_list': markets  # Also return the list for reference
        }, status=status.HTTP_200_OK)
        
        if METRICS_AVAILABLE:
            endpoint_latency.labels(endpoint='get_platform_stats', method='GET').observe(time.time() - start_time)
        return response
    except Exception as e:
        if METRICS_AVAILABLE:
            endpoint_latency.labels(endpoint='get_platform_stats', method='GET').observe(time.time() - start_time)
        # Fallback values if service fails
        return Response({
            'tracks_analyzed': 953,
            'ml_models': 4,
            'markets': 6,
            'markets_list': ['US', 'GB', 'DE', 'FR', 'JP', 'BR']
        }, status=status.HTTP_200_OK)


@api_view(['POST'])
def predict_spotify_hit(request):
    """Predict Spotify hit probability for selected song and markets."""
    start_time = time.time()
    try:
        data = json.loads(request.body) if isinstance(request.body, bytes) else request.data
        service = get_spotify_service()
        result = service.predict_hit(data)
        
        # Record metrics
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            spotify_hit_prediction_duration.observe(duration)
            spotify_hit_prediction_requests.labels(status='success').inc()
            endpoint_latency.labels(endpoint='predict_spotify_hit', method='POST').observe(duration)
        
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            spotify_hit_prediction_requests.labels(status='error').inc()
            prediction_errors.labels(error_type=type(e).__name__).inc()
            endpoint_latency.labels(endpoint='predict_spotify_hit', method='POST').observe(duration)
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def get_similar_songs(request):
    """Get similar songs based on audio features."""
    start_time = time.time()
    try:
        data = json.loads(request.body) if isinstance(request.body, bytes) else request.data
        service = get_spotify_service()
        result = service.get_similar_songs(data)
        
        # Record metrics
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            similar_songs_duration.observe(duration)
            similar_songs_requests.labels(status='success').inc()
            endpoint_latency.labels(endpoint='get_similar_songs', method='POST').observe(duration)
        
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            similar_songs_requests.labels(status='error').inc()
            prediction_errors.labels(error_type=type(e).__name__).inc()
            endpoint_latency.labels(endpoint='get_similar_songs', method='POST').observe(duration)
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict_instagram_picture_hit(request):
    """Predict Instagram Reels picture hit probability."""
    from .services import InstagramPredictionService
    
    try:
        data = json.loads(request.body) if isinstance(request.body, bytes) else request.data
        service = InstagramPredictionService()
        result = service.predict_picture_hit(data)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict_instagram_caption_hit(request):
    """Predict Instagram Reels caption/hashtag hit probability."""
    from .services import InstagramPredictionService
    
    try:
        data = json.loads(request.body) if isinstance(request.body, bytes) else request.data
        service = InstagramPredictionService()
        result = service.predict_caption_hit(data)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def predict_final_combination(request):
    """Predict final combination hit score."""
    from .services import FinalPredictionService
    
    try:
        data = json.loads(request.body) if isinstance(request.body, bytes) else request.data
        service = FinalPredictionService()
        result = service.predict_combination(data)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_spotify_playlist(request):
    """Get Spotify playlist with songs from dataset."""
    start_time = time.time()
    try:
        service = get_spotify_service()
        playlist = service.get_playlist()
        
        # Record metrics
        if METRICS_AVAILABLE:
            playlist_requests.labels(status='success').inc()
            endpoint_latency.labels(endpoint='get_spotify_playlist', method='GET').observe(time.time() - start_time)
        
        return Response(playlist, status=status.HTTP_200_OK)
    except Exception as e:
        if METRICS_AVAILABLE:
            playlist_requests.labels(status='error').inc()
            endpoint_latency.labels(endpoint='get_spotify_playlist', method='GET').observe(time.time() - start_time)
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_spotify_track_id(request):
    """Get Spotify track ID for embedding."""
    import requests
    import base64
    from django.conf import settings
    
    track_name = request.GET.get('track', '')
    artist_name = request.GET.get('artist', '')
    
    if not track_name or not artist_name:
        return Response({
            'error': 'Track name and artist name required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Get Spotify Client ID and Secret from settings (loaded from .env)
        # If not set, will use public search (limited functionality)
        client_id = settings.SPOTIFY_CLIENT_ID
        client_secret = settings.SPOTIFY_CLIENT_SECRET
        
        # Get access token if credentials are available
        access_token = None
        if client_id and client_secret:
            try:
                # Get access token using client credentials
                auth_url = 'https://accounts.spotify.com/api/token'
                auth_header = base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()
                auth_response = requests.post(
                    auth_url,
                    headers={'Authorization': f'Basic {auth_header}'},
                    data={'grant_type': 'client_credentials'},
                    timeout=5
                )
                if auth_response.status_code == 200:
                    access_token = auth_response.json().get('access_token')
            except Exception as e:
                print(f"Error getting Spotify token: {e}")
        
        # Search for track
        search_query = f"track:{track_name} artist:{artist_name}"
        search_url = "https://api.spotify.com/v1/search"
        params = {
            'q': search_query,
            'type': 'track',
            'limit': 1
        }
        
        headers = {}
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        
        response = requests.get(search_url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('tracks', {}).get('items'):
                track_id = data['tracks']['items'][0]['id']
                return Response({
                    'track_id': track_id,
                    'embed_url': f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
                }, status=status.HTTP_200_OK)
            else:
                # No tracks found in search results
                print(f"Spotify search returned no results for: {track_name} by {artist_name}")
                return Response({
                    'error': f'Track "{track_name}" by "{artist_name}" not found on Spotify.',
                    'track_id': None
                }, status=status.HTTP_404_NOT_FOUND)
        else:
            # API returned error
            print(f"Spotify API error: {response.status_code} - {response.text}")
            return Response({
                'error': f'Spotify API error: {response.status_code}',
                'track_id': None
            }, status=response.status_code if response.status_code < 500 else status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
