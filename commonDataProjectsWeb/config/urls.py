"""
URL configuration for microservice project.
Auto-discovers projects in projects/ directory.
"""
from django.contrib import admin
from django.urls import path, include
from django_prometheus import exports
from api import views
from pathlib import Path
import json

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls),
    path('api/', include('api.urls', namespace='api')),
    path('metrics/', exports.ExportToDjangoView, name='prometheus-django-metrics'),
]

# Auto-discover modular projects
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / 'projects'

if PROJECTS_DIR.exists():
    for project_dir in PROJECTS_DIR.iterdir():
        if project_dir.is_dir() and not project_dir.name.startswith('.'):
            project_name = project_dir.name
            
            # Check if urls.py exists
            if (project_dir / 'urls.py').exists():
                # Read config.json for URL prefix and namespace, or use project name
                config_file = project_dir / 'config.json'
                url_prefix = project_name.replace('_', '-')
                namespace = project_name
                
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                        url_prefix = config.get('url_prefix', url_prefix)
                        namespace = config.get('namespace', namespace)
                    except:
                        pass
                
                # Add project URLs with namespace
                urlpatterns.append(
                    path(f'{url_prefix}/', include(f'projects.{project_name}.urls', namespace=namespace))
                )

