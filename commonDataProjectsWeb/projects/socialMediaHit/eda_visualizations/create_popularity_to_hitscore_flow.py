#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Popularity to Hit Score Transformation Flow Diagram
Shows how popularity is transformed to hit score
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from pathlib import Path
import numpy as np

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

def create_box(ax, x, y, width, height, text, color, text_color=WHITE, fontsize=10, bold=False, title=None):
    """Create a rounded rectangle box with text"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1", 
                        facecolor=color, 
                        edgecolor=WHITE,
                        linewidth=2,
                        alpha=0.9)
    ax.add_patch(box)
    
    # Add title if provided
    if title:
        ax.text(x + width/2, y + height - 0.15, title,
               ha='center', va='top',
               fontsize=fontsize-2, color=text_color,
               fontweight='bold')
        # Main text
        ax.text(x + width/2, y + height/2 - 0.1, text,
               ha='center', va='center',
               fontsize=fontsize-1, color=text_color,
               fontweight=fontweight if 'fontweight' in locals() else 'normal',
               wrap=True)
    else:
        # Add text
        fontweight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center',
               fontsize=fontsize, color=text_color,
               fontweight=fontweight,
               wrap=True)
    return box

def create_arrow(ax, x1, y1, x2, y2, color=SPOTIFY_GREEN, linewidth=2, label=None):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           mutation_scale=20,
                           color=color,
                           linewidth=linewidth,
                           alpha=0.8)
    ax.add_patch(arrow)
    
    if label:
        # Add label in the middle
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
               fontsize=9, color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#121212', 
                        edgecolor=color, alpha=0.8))
    return arrow

# Create figure
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(9, 11.5, 'POPULARITY → HIT SCORE TRANSFORMATION FLOW', 
       ha='center', va='center',
       fontsize=20, color=SPOTIFY_GREEN, fontweight='bold')

# ============================================================================
# LEFT SIDE: POPULARITY
# ============================================================================

# Popularity Input Section
create_box(ax, 1, 8.5, 3.5, 1.5, 
          'Audio Features:\n• Danceability\n• Energy\n• Valence\n• Loudness\n• Tempo\n• Speechiness\n• Acousticness\n• ...', 
          LIGHT_GRAY, title='INPUT: Audio Features')

# Popularity Calculation
create_box(ax, 1, 6.5, 3.5, 1.5,
          'Spotify Algorithm:\n• User streams\n• Playlist adds\n• Skip rate\n• Listening time\n• Social signals',
          BLUE, title='CALCULATION: Spotify Algorithm')

# Popularity Output
create_box(ax, 1, 4.5, 3.5, 1.5,
          'POPULARITY\n(0-100 scale)\n\nExample: 75',
          SPOTIFY_GREEN, title='OUTPUT: Popularity Score', bold=True)

# Popularity Characteristics
create_box(ax, 1, 2, 3.5, 2,
          'CHARACTERISTICS:\n• Raw metric from Spotify\n• Based on actual user behavior\n• Range: 0-100\n• Real-world popularity measure',
          DARK_GREEN, fontsize=9)

# ============================================================================
# MIDDLE: TRANSFORMATION PROCESS
# ============================================================================

# Arrow from Popularity to Percentile Calculation
create_arrow(ax, 4.5, 5.25, 7, 5.25, SPOTIFY_GREEN, 3, 'Percentile\nMapping')

# Percentile Calculation Box
create_box(ax, 7, 4.5, 4, 1.5,
          'Calculate Percentile Position:\nPercentiles: [10, 25, 50, 75, 90, 95, 99]\n\nIf popularity = 75:\n→ Between P75 (50) and P90 (80)\n→ Position in range',
          YELLOW, title='STEP 1: Percentile Analysis', fontsize=9)

# Arrow to Mapping
create_arrow(ax, 9, 4.5, 9, 3.5, YELLOW, 2, 'Map to\nScore')

# Mapping Function Box
create_box(ax, 7, 2, 4, 1.3,
          'Percentile-Based Mapping:\nP99+ → 95-100\nP95-P99 → 90-95\nP90-P95 → 80-90\nP75-P90 → 65-80\nP50-P75 → 50-65\n...',
          PURPLE, title='STEP 2: Score Mapping', fontsize=9)

# ============================================================================
# RIGHT SIDE: HIT SCORE
# ============================================================================

# Arrow to Hit Score
create_arrow(ax, 11, 3.25, 13.5, 5.25, PURPLE, 3, 'Normalize\nto 0-100')

# Hit Score Output
create_box(ax, 13.5, 4.5, 3.5, 1.5,
          'HIT SCORE\n(0-100 scale)\n\nExample: 82.4',
          SPOTIFY_GREEN, title='OUTPUT: Hit Score', bold=True)

# Hit Score Characteristics
create_box(ax, 13.5, 2, 3.5, 2,
          'CHARACTERISTICS:\n• Normalized from popularity\n• Percentile-based transformation\n• Range: 0-100\n• Comparable across datasets\n• Used for predictions',
          DARK_GREEN, fontsize=9)

# ============================================================================
# BOTTOM: DETAILED EXAMPLE
# ============================================================================

# Example Section
example_box = FancyBboxPatch((1, 0), 16, 1.5,
                            boxstyle="round,pad=0.2", 
                            facecolor='#282828', 
                            edgecolor=SPOTIFY_GREEN,
                            linewidth=2,
                            alpha=0.9)
ax.add_patch(example_box)

example_text = """
EXAMPLE TRANSFORMATION:
Popularity = 75 → Calculate percentile position → Find position between P75 (50) and P90 (80) 
→ Apply mapping formula: 65 + ((75-50)/(80-50)) × 15 = 65 + 12.5 = 77.5 → Hit Score = 77.5

FORMULA: base_score + ((popularity - lower_bound) / (upper_bound - lower_bound)) × range
"""

ax.text(9, 0.75, example_text,
       ha='center', va='center',
       fontsize=10, color=WHITE, fontweight='normal',
       family='monospace')

# ============================================================================
# ADD LABELS FOR CLARITY
# ============================================================================

# Label for Popularity section
ax.text(2.75, 9.8, 'POPULARITY', ha='center', va='center',
       fontsize=12, color=BLUE, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='#121212', edgecolor=BLUE, pad=0.5))

# Label for Transformation section
ax.text(9, 9.8, 'TRANSFORMATION', ha='center', va='center',
       fontsize=12, color=YELLOW, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='#121212', edgecolor=YELLOW, pad=0.5))

# Label for Hit Score section
ax.text(15.25, 9.8, 'HIT SCORE', ha='center', va='center',
       fontsize=12, color=PURPLE, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='#121212', edgecolor=PURPLE, pad=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '22_popularity_to_hitscore_flow.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 22_popularity_to_hitscore_flow.png")
plt.close()

# ============================================================================
# DETAILED MAPPING TABLE VISUALIZATION
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'PERCENTILE-BASED MAPPING: POPULARITY → HIT SCORE', 
       ha='center', va='center',
       fontsize=18, color=SPOTIFY_GREEN, fontweight='bold')

# Create mapping table
mappings = [
    ('P99+', '≥ P99', '95-100', 'Very High Hit Potential'),
    ('P95-P99', 'P95 to P99', '90-95', 'High Hit Potential'),
    ('P90-P95', 'P90 to P95', '80-90', 'Strong Hit Potential'),
    ('P75-P90', 'P75 to P90', '65-80', 'Good Hit Potential'),
    ('P50-P75', 'P50 to P75', '50-65', 'Moderate Hit Potential'),
    ('P25-P50', 'P25 to P50', '35-50', 'Low Hit Potential'),
    ('P10-P25', 'P10 to P25', '20-35', 'Very Low Hit Potential'),
    ('< P10', '< P10', '0-20', 'Minimal Hit Potential'),
]

# Table header
headers = ['Percentile Range', 'Popularity Position', 'Hit Score Range', 'Interpretation']
col_widths = [3, 3, 2.5, 5]
x_start = 1

# Draw header
y_pos = 8
for i, (header, width) in enumerate(zip(headers, col_widths)):
    x_pos = x_start + sum(col_widths[:i])
    create_box(ax, x_pos, y_pos, width, 0.6, header, SPOTIFY_GREEN, bold=True, fontsize=11)

# Draw rows
for row_idx, (perc_range, pop_pos, hit_range, interpretation) in enumerate(mappings):
    y_pos = 7.2 - row_idx * 0.85
    colors = [LIGHT_GRAY, BLUE, YELLOW, PURPLE]
    
    for col_idx, (value, width, color) in enumerate(zip([perc_range, pop_pos, hit_range, interpretation], 
                                                         col_widths, colors)):
        x_pos = x_start + sum(col_widths[:col_idx])
        create_box(ax, x_pos, y_pos, width, 0.75, value, color, fontsize=10)

# Add explanation box
explanation_box = FancyBboxPatch((1, 0.5), 14, 1.5,
                                boxstyle="round,pad=0.2", 
                                facecolor='#282828', 
                                edgecolor=SPOTIFY_GREEN,
                                linewidth=2,
                                alpha=0.9)
ax.add_patch(explanation_box)

explanation_text = """
HOW IT WORKS:
1. Calculate percentiles of popularity values in dataset: [10, 25, 50, 75, 90, 95, 99]
2. Determine which percentile range the popularity value falls into
3. Apply linear interpolation within that range to map to hit score scale
4. Result: Normalized hit score (0-100) that represents relative popularity position
5. This normalization makes scores comparable across different datasets and time periods
"""

ax.text(8, 1.25, explanation_text,
       ha='center', va='center',
       fontsize=10, color=WHITE,
       family='monospace')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '23_hitscore_mapping_table.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 23_hitscore_mapping_table.png")
plt.close()

print(f"\n=== POPULARITY TO HIT SCORE FLOW DIAGRAMS COMPLETE ===")
print(f"All visualizations saved to: {OUTPUT_DIR}")

