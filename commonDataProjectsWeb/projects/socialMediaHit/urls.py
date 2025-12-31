"""
URL configuration for socialMediaHit.
"""
from django.urls import path
from . import views

app_name = 'socialMediaHit'  # Namespace

urlpatterns = [
    # Pages
    path('', views.home, name='home'),
    path('spotify/', views.spotify_prediction, name='spotify'),
    path('instagram-picture/', views.instagram_picture_prediction, name='instagram_picture'),
    path('instagram-caption/', views.instagram_caption_prediction, name='instagram_caption'),
    path('final/', views.final_prediction, name='final'),
    
    # API Endpoints
    path('api/health/', views.health, name='health'),
    path('api/stats/', views.get_platform_stats, name='platform_stats'),
    path('api/playlist/', views.get_spotify_playlist, name='get_playlist'),
    path('api/track-id/', views.get_spotify_track_id, name='get_track_id'),
    path('api/predict/spotify/', views.predict_spotify_hit, name='predict_spotify'),
    path('api/similar-songs/', views.get_similar_songs, name='similar_songs'),
    path('api/predict/instagram-picture/', views.predict_instagram_picture_hit, name='predict_instagram_picture'),
    path('api/predict/instagram-caption/', views.predict_instagram_caption_hit, name='predict_instagram_caption'),
    path('api/predict/final/', views.predict_final_combination, name='predict_final'),
]
