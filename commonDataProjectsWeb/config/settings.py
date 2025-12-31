"""
Django settings for microservice project.
AWS Free Tier optimized configuration.
"""
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='django-insecure-change-in-production')
DEBUG = config('DEBUG', default=False, cast=bool)
# ALLOWED_HOSTS: Support Docker internal hostname 'web' for Prometheus scraping
allowed_hosts_str = config('ALLOWED_HOSTS', default='*')
if allowed_hosts_str == '*':
    ALLOWED_HOSTS = ['*']  # Allow all hosts in development
else:
    ALLOWED_HOSTS = allowed_hosts_str.split(',')
    # Always add 'web' for Docker internal communication (Prometheus)
    if 'web' not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append('web')

# Spotify API (Optional - for music player embedding)
# Get credentials from: https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID = config('SPOTIFY_CLIENT_ID', default=None)
SPOTIFY_CLIENT_SECRET = config('SPOTIFY_CLIENT_SECRET', default=None)

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_prometheus',
    'api',
]

# Auto-discover modular projects in projects/ directory
PROJECTS_DIR = BASE_DIR / 'projects'
if PROJECTS_DIR.exists():
    for project_dir in PROJECTS_DIR.iterdir():
        if project_dir.is_dir() and not project_dir.name.startswith('.'):
            project_name = project_dir.name
            app_name = f'projects.{project_name}'
            # Check if it's a valid Django app (has apps.py or __init__.py)
            # and not already in INSTALLED_APPS (avoid duplicates)
            if ((project_dir / 'apps.py').exists() or (project_dir / '__init__.py').exists()) and app_name not in INSTALLED_APPS:
                INSTALLED_APPS.append(app_name)

MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database - AWS RDS Free Tier compatible
DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.postgresql',
        'NAME': config('DB_NAME', default='microservice_db'),
        'USER': config('DB_USER', default='postgres'),
        'PASSWORD': config('DB_PASSWORD', default='postgres'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
        'CONN_MAX_AGE': 600,
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (WhiteNoise for AWS)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

