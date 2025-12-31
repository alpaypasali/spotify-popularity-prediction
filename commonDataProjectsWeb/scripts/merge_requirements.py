#!/usr/bin/env python
"""
Merge project requirements.txt files into main requirements.txt.

Usage:
    python scripts/merge_requirements.py
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / 'projects'
MAIN_REQUIREMENTS = BASE_DIR / 'requirements.txt'


def merge_requirements():
    """Merge all project requirements.txt files into main requirements.txt."""
    
    # Read main requirements
    main_content = MAIN_REQUIREMENTS.read_text().strip()
    
    # Find all project requirements.txt files
    project_requirements = []
    if PROJECTS_DIR.exists():
        for req_file in PROJECTS_DIR.rglob('requirements.txt'):
            if req_file.parent.name != 'projects':  # Skip if directly in projects/
                project_requirements.append(req_file)
    
    # Merge project requirements
    merged_content = main_content
    if project_requirements:
        merged_content += "\n\n# Auto-detected project dependencies\n"
        for req_file in sorted(project_requirements):
            project_name = req_file.parent.name
            merged_content += f"\n# From: projects/{project_name}/requirements.txt\n"
            merged_content += req_file.read_text()
            merged_content += "\n"
    
    # Write merged requirements
    MAIN_REQUIREMENTS.write_text(merged_content)
    
    print("✅ Requirements merged successfully!")
    print(f"📦 Found {len(project_requirements)} project requirements files:")
    for req_file in sorted(project_requirements):
        print(f"   - {req_file.relative_to(BASE_DIR)}")


if __name__ == '__main__':
    merge_requirements()
