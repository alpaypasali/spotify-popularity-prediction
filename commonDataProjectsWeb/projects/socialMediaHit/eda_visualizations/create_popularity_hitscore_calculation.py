#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How Popularity and Hit Score are Calculated - Single Page Explanation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

# Colors
SPOTIFY_GREEN = '#1ED760'
DARK_GREEN = '#1DB954'
BLUE = '#4A90E2'
PURPLE = '#9B59B6'
YELLOW = '#FFD700'
RED = '#FF6B6B'
WHITE = '#FFFFFF'
LIGHT_GRAY = '#1a1a2e'
ORANGE = '#FFA500'

def create_box(ax, x, y, width, height, title, content_lines, color, title_color=WHITE):
    """Create a box with title and content lines"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.2", 
                        facecolor=color, 
                        edgecolor=WHITE,
                        linewidth=2.5,
                        alpha=0.95)
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.25, title,
           ha='center', va='top',
           fontsize=16, color=title_color,
           fontweight='bold')
    
    # Content
    y_start = y + height - 0.6
    line_height = 0.4
    for i, line in enumerate(content_lines):
        if line.strip():  # Skip empty lines
            ax.text(x + 0.25, y_start - i * line_height, line,
                   ha='left', va='top',
                   fontsize=12, color=WHITE, weight='normal')
    return box

def create_arrow(ax, x1, y1, x2, y2, color=SPOTIFY_GREEN, linewidth=2, label=None):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=25,
                           color=color,
                           linewidth=linewidth,
                           alpha=0.8)
    ax.add_patch(arrow)
    
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
               fontsize=9, color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#121212', 
                        edgecolor=color, alpha=0.9))
    return arrow

# Create figure - larger for better readability
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Main Title
ax.text(10, 13.5, 'POPULARITY vs HIT SCORE: NASIL HESAPLANIR?', 
       ha='center', va='center',
       fontsize=24, color=SPOTIFY_GREEN, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#121212', edgecolor=SPOTIFY_GREEN, linewidth=2))

# ============================================================================
# LEFT SIDE: POPULARITY CALCULATION
# ============================================================================

# Popularity Section Title
ax.text(5, 12.5, 'POPULARITY', ha='center', va='center',
       fontsize=20, color=BLUE, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#121212', edgecolor=BLUE, linewidth=3))

# Popularity Inputs
create_box(ax, 0.5, 9.5, 4.5, 2.5, '1. GİRDİLER (Inputs)',
          ['• Kullanıcı Davranışları:',
           '  - Stream sayısı',
           '  - Playlist eklemeleri',
           '  - Skip oranı',
           '  - Dinleme süresi',
           '  - Sosyal sinyaller',
           '',
           '• Spotify API verisi'],
          LIGHT_GRAY)

# Arrow 1
create_arrow(ax, 2.75, 9.5, 2.75, 8.5, BLUE, 4, 'Spotify\nAlgorithm')

# Popularity Calculation
create_box(ax, 0.5, 7, 4.5, 1.8, '2. HESAPLAMA (Calculation)',
          ['• Spotify\'ın proprietary algorithm',
           '• Weighted combination of signals',
           '• Time-based decay factor',
           '• User engagement metrics'],
          BLUE)

# Arrow 2
create_arrow(ax, 2.75, 7, 2.75, 6.2, BLUE, 4, 'Output')

# Popularity Output
create_box(ax, 0.5, 4.5, 4.5, 1.8, '3. ÇIKTI (Output)',
          ['POPULARITY SKORU',
           'Range: 0-100',
           'Örnek: Popularity = 75',
           'Gerçek popülerlik metrik'],
          DARK_GREEN, title_color=WHITE)

# Popularity Characteristics
create_box(ax, 0.5, 2, 4.5, 1.8, 'POPULARITY ÖZELLİKLERİ',
          ['• Spotify\'dan gelen hazır metrik',
           '• Kullanıcı davranışı bazlı',
           '• Gerçek popülerlik ölçüsü',
           '• Descriptive metric'],
          '#2E5C8A', title_color=WHITE)

# ============================================================================
# MIDDLE: DIVIDER
# ============================================================================
ax.plot([10, 10], [2, 12.5], color=WHITE, linewidth=4, linestyle='--', alpha=0.6)
ax.text(10, 7, 'VS', ha='center', va='center',
       fontsize=32, color=YELLOW, fontweight='bold',
       bbox=dict(boxstyle='round,pad=1', facecolor='#121212', edgecolor=YELLOW, linewidth=3))

# ============================================================================
# RIGHT SIDE: HIT SCORE CALCULATION
# ============================================================================

# Hit Score Section Title
ax.text(15, 12.5, 'HIT SCORE', ha='center', va='center',
       fontsize=20, color=PURPLE, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#121212', edgecolor=PURPLE, linewidth=3))

# Hit Score Inputs
create_box(ax, 15, 9.5, 4.5, 2.5, '1. GİRDİLER (Inputs)',
          ['• Audio Features:',
           '  - Danceability',
           '  - Energy, Valence',
           '  - Loudness, Tempo',
           '  - Speechiness, etc.',
           '',
           '• ML Model (Voting Regressor)'],
          LIGHT_GRAY)

# Arrow 3
create_arrow(ax, 17.25, 9.5, 17.25, 8.5, PURPLE, 4, 'Model\nPrediction')

# Hit Score Calculation Step 1
create_box(ax, 15, 7, 4.5, 1.5, '2. MODEL TAHMİNİ',
          ['Voting Regressor',
           'XGBoost + LightGBM',
           'Raw prediction value'],
          ORANGE)

# Arrow 4
create_arrow(ax, 17.25, 7, 17.25, 6.2, PURPLE, 3, 'Percentile\nMapping')

# Hit Score Calculation Step 2
create_box(ax, 15, 5, 4.5, 1.5, '3. NORMALİZASYON',
          ['Percentile-based mapping',
           'P10, P25, P50, P75, P90, P95, P99',
           'Linear interpolation'],
          YELLOW)

# Arrow 5
create_arrow(ax, 17.25, 5, 17.25, 4.5, PURPLE, 4, 'Normalize\nScore')

# Hit Score Output
create_box(ax, 15, 3, 4.5, 1.8, '4. ÇIKTI (Output)',
          ['HIT SCORE',
           'Range: 0-100',
           'Örnek: Hit Score = 82.4',
           'Tahminsel hit potansiyeli'],
          DARK_GREEN, title_color=WHITE)

# Hit Score Characteristics
create_box(ax, 15, 0.5, 4.5, 1.8, 'HIT SCORE ÖZELLİKLERİ',
          ['• ML model ile tahmin',
           '• Audio features bazlı',
           '• Tahminsel metrik',
           '• Predictive metric'],
          '#6A1B9A', title_color=WHITE)

# ============================================================================
# BOTTOM: KEY DIFFERENCES
# ============================================================================
diff_box = FancyBboxPatch((5.5, 0.5), 9, 1.5,
                         boxstyle="round,pad=0.3", 
                         facecolor='#282828', 
                         edgecolor=SPOTIFY_GREEN,
                         linewidth=3,
                         alpha=0.95)
ax.add_patch(diff_box)

ax.text(10, 1.9, 'ÖNEMLİ FARKLAR', ha='center', va='center',
       fontsize=16, color=SPOTIFY_GREEN, fontweight='bold')

diff_text = """• Popularity → Spotify'ın hazır metrik (kullanıcı davranışlarından)
• Hit Score → ML model tahmini (audio features'dan)
• Popularity → Gerçek popülerlik ölçüsü (Descriptive)
• Hit Score → Tahminsel hit potansiyeli (Predictive)
• İkisi farklı kaynaklardan ve yöntemlerle hesaplanır!"""

ax.text(10, 1.1, diff_text,
       ha='center', va='center',
       fontsize=13, color=WHITE, fontweight='normal',
       family='sans-serif')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '24_how_popularity_and_hitscore_calculated.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 24_how_popularity_and_hitscore_calculated.png")
plt.close()

print(f"=== POPULARITY vs HIT SCORE CALCULATION EXPLANATION COMPLETE ===")
print(f"Visualization saved to: {OUTPUT_DIR / '24_how_popularity_and_hitscore_calculated.png'}")

