#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Architecture Diagram for Final Presentation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from pathlib import Path

# Set style
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#1ED760'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'eda_visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
SPOTIFY_GREEN = '#1ED760'
DARK_GREEN = '#1DB954'
PINK = '#FF6B9D'
LIGHT_PINK = '#FFB3D1'
BLUE = '#4A90E2'
DARK_BLUE = '#2E5C8A'
WHITE = '#FFFFFF'
YELLOW = '#FFD700'

# Define box style
def create_box(ax, x, y, width, height, text, color, text_color=WHITE, fontsize=10, bold=False):
    """Create a rounded rectangle box with text"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1", 
                        facecolor=color, 
                        edgecolor=WHITE,
                        linewidth=1.5,
                        alpha=0.9)
    ax.add_patch(box)
    
    # Add text
    fontweight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, color=text_color,
           fontweight=fontweight,
           wrap=True)
    return box

# Define arrow style
def create_arrow(ax, x1, y1, x2, y2, color=SPOTIFY_GREEN, linewidth=2):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=20,
                           color=color,
                           linewidth=linewidth,
                           alpha=0.8)
    ax.add_patch(arrow)
    return arrow

# ============================================================================
# LEVEL 1: Data Sources (Top)
# ============================================================================
# Left: Spotify API
spotify_box = create_box(ax, 1, 12, 4, 1.2, 
                        'Spotify API\nMüzik Verileri', 
                        SPOTIFY_GREEN, fontsize=12, bold=True)

# Right: Kaggle Datasets
kaggle_box = create_box(ax, 15, 12, 4, 1.2,
                       'Kaggle Datasets\nInstagram Meta Verileri',
                       PINK, fontsize=12, bold=True)

# ============================================================================
# LEVEL 2: Feature Extraction (Middle-Top)
# ============================================================================
# From Spotify
spotify_hit = create_box(ax, 1.5, 10, 3, 0.8,
                        'Spotify Hit',
                        DARK_GREEN, fontsize=10)

# From Instagram (5 boxes)
caption_nlp = create_box(ax, 13, 10.5, 2.5, 0.7,
                        'Caption NLP',
                        LIGHT_PINK, fontsize=9)

image_cnn = create_box(ax, 13, 9.5, 2.5, 0.7,
                      'Image CNN',
                      LIGHT_PINK, fontsize=9)

ses_mfcc = create_box(ax, 13, 8.5, 2.5, 0.7,
                     'Ses MFCC',
                     LIGHT_PINK, fontsize=9)

video_analytics = create_box(ax, 13, 7.5, 2.5, 0.7,
                            'Video Analytics',
                            LIGHT_PINK, fontsize=9)

reels_multimodal = create_box(ax, 13, 6.5, 2.5, 0.7,
                             'Reels Multi-Modal',
                             LIGHT_PINK, fontsize=9)

# ============================================================================
# LEVEL 3: Hit Prediction Modules (Middle)
# ============================================================================
# Spotify Hit Prediction
spotify_pred = create_box(ax, 1.5, 8.5, 3, 0.8,
                         'Spotify Hit\nTahmini',
                         DARK_GREEN, fontsize=10, bold=True)

# Instagram predictions (5 boxes)
insta_caption_pred = create_box(ax, 16, 10.5, 2.5, 0.7,
                               'Insta Caption\nHit Tahmini',
                               DARK_BLUE, fontsize=9, bold=True)

insta_image_pred = create_box(ax, 16, 9.5, 2.5, 0.7,
                              'Insta Image\nHit Tahmini',
                              DARK_BLUE, fontsize=9, bold=True)

insta_ses_pred = create_box(ax, 16, 8.5, 2.5, 0.7,
                           'Insta Ses\nHit Tahmini',
                           DARK_BLUE, fontsize=9, bold=True)

insta_video_pred = create_box(ax, 16, 7.5, 2.5, 0.7,
                             'Insta Video\nHit Tahmini',
                             DARK_BLUE, fontsize=9, bold=True)

insta_reels_pred = create_box(ax, 16, 6.5, 2.5, 0.7,
                             'Insta Reels\nHit Tahmini',
                             DARK_BLUE, fontsize=9, bold=True)

# ============================================================================
# LEVEL 4: System Integration (Middle-Bottom)
# ============================================================================
integration_box = create_box(ax, 7, 5, 6, 1,
                            'Sistem Entegrasyonu & API',
                            BLUE, fontsize=14, bold=True)

# ============================================================================
# LEVEL 5: Final Output (Bottom)
# ============================================================================
final_box = create_box(ax, 7, 2.5, 6, 1,
                      'Reels İçerik Hit Tahmini / Üreticisi',
                      DARK_GREEN, fontsize=14, bold=True)

# ============================================================================
# ARROWS: Connect the boxes
# ============================================================================
# From data sources to feature extraction
create_arrow(ax, 3, 12, 3, 10.8)  # Spotify API -> Spotify Hit
create_arrow(ax, 17, 12, 15.25, 10.85)  # Kaggle -> Caption NLP
create_arrow(ax, 17, 12, 15.25, 9.85)   # Kaggle -> Image CNN
create_arrow(ax, 17, 12, 15.25, 8.85)   # Kaggle -> Ses MFCC
create_arrow(ax, 17, 12, 15.25, 7.85)   # Kaggle -> Video Analytics
create_arrow(ax, 17, 12, 15.25, 6.85)   # Kaggle -> Reels Multi-Modal

# From feature extraction to predictions
create_arrow(ax, 3, 10, 3, 9.3)  # Spotify Hit -> Spotify Hit Tahmini
create_arrow(ax, 15.5, 10.5, 16, 10.85)  # Caption NLP -> Insta Caption Hit
create_arrow(ax, 15.5, 9.5, 16, 9.85)    # Image CNN -> Insta Image Hit
create_arrow(ax, 15.5, 8.5, 16, 8.85)    # Ses MFCC -> Insta Ses Hit
create_arrow(ax, 15.5, 7.5, 16, 7.85)    # Video Analytics -> Insta Video Hit
create_arrow(ax, 15.5, 6.5, 16, 6.85)    # Reels Multi-Modal -> Insta Reels Hit

# From predictions to integration
create_arrow(ax, 3, 8.5, 7, 5.5)  # Spotify Hit Tahmini -> Integration
create_arrow(ax, 18.5, 10.5, 13, 5.5)  # Insta Caption Hit -> Integration
create_arrow(ax, 18.5, 9.5, 13, 5.5)   # Insta Image Hit -> Integration
create_arrow(ax, 18.5, 8.5, 13, 5.5)   # Insta Ses Hit -> Integration
create_arrow(ax, 18.5, 7.5, 13, 5.5)   # Insta Video Hit -> Integration
create_arrow(ax, 18.5, 6.5, 13, 5.5)   # Insta Reels Hit -> Integration

# From integration to final output
create_arrow(ax, 10, 5, 10, 3.5, linewidth=3)  # Integration -> Final Output

# ============================================================================
# Add title
# ============================================================================
ax.text(10, 13.5, 'SİSTEM MİMARİSİ - HIT TAHMİN PLATFORMU',
       ha='center', va='center',
       fontsize=18, color=SPOTIFY_GREEN,
       fontweight='bold')

# Save
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'system_architecture_diagram.png', 
           dpi=300, facecolor='#121212', bbox_inches='tight')
print(f"System architecture diagram saved to: {OUTPUT_DIR / 'system_architecture_diagram.png'}")
plt.close()

