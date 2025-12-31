#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Popularity vs Hit Score Analysis for Final Presentation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['axes.edgecolor'] = '#1ED760'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['xtick.color'] = '#FFFFFF'
plt.rcParams['ytick.color'] = '#FFFFFF'

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'eda_visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
SPOTIFY_GREEN = '#1ED760'
DARK_GREEN = '#1DB954'
RED = '#FF6B6B'
BLUE = '#4A90E2'
WHITE = '#FFFFFF'

# Load data
train_path = BASE_DIR / 'data' / 'processed' / 'spotify_emotion_train.csv'
df = pd.read_csv(train_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Find popularity column
pop_col = None
for col in ['popularity', 'POPULARITY', 'streams', 'STREAMS']:
    if col in df.columns:
        pop_col = col
        break

if pop_col is None:
    print("Warning: Popularity column not found!")
    sys.exit(1)

print(f"Using popularity column: {pop_col}")

# Calculate "hit score" based on model predictions
# For visualization purposes, we'll simulate hit scores based on popularity percentiles
# In real scenario, these would come from model predictions

# Normalize popularity to 0-100 scale (hit score)
df['popularity_norm'] = pd.to_numeric(df[pop_col], errors='coerce').dropna()
df = df[df['popularity_norm'].notna()].copy()

# Calculate percentile-based hit score (similar to model logic)
percentiles = np.percentile(df['popularity_norm'], [10, 25, 50, 75, 90, 95, 99])

def calculate_hit_score(popularity):
    """Calculate hit score from popularity using percentile-based mapping"""
    if popularity >= percentiles[6]:  # P99+
        return min(100, 95 + ((popularity - percentiles[6]) / (df['popularity_norm'].max() - percentiles[6])) * 5)
    elif popularity >= percentiles[5]:  # P95-P99
        return 90 + ((popularity - percentiles[5]) / (percentiles[6] - percentiles[5])) * 5
    elif popularity >= percentiles[4]:  # P90-P95
        return 80 + ((popularity - percentiles[4]) / (percentiles[5] - percentiles[4])) * 10
    elif popularity >= percentiles[3]:  # P75-P90
        return 65 + ((popularity - percentiles[3]) / (percentiles[4] - percentiles[3])) * 15
    elif popularity >= percentiles[2]:  # P50-P75
        return 50 + ((popularity - percentiles[2]) / (percentiles[3] - percentiles[2])) * 15
    elif popularity >= percentiles[1]:  # P25-P50
        return 35 + ((popularity - percentiles[1]) / (percentiles[2] - percentiles[1])) * 15
    elif popularity >= percentiles[0]:  # P10-P25
        return 20 + ((popularity - percentiles[0]) / (percentiles[1] - percentiles[0])) * 15
    else:  # < P10
        return max(0, (popularity / percentiles[0]) * 20)

df['hit_score'] = df['popularity_norm'].apply(calculate_hit_score)

# Sample data for visualization (if too large)
if len(df) > 10000:
    df_sample = df.sample(n=10000, random_state=42)
else:
    df_sample = df.copy()

print(f"Sample size: {len(df_sample)}")
print(f"Popularity range: {df_sample['popularity_norm'].min():.2f} - {df_sample['popularity_norm'].max():.2f}")
print(f"Hit score range: {df_sample['hit_score'].min():.2f} - {df_sample['hit_score'].max():.2f}")

# ============================================================================
# VISUALIZATION 1: Popularity vs Hit Score Scatter
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Popularity vs Hit Score Analysis', 
            fontsize=16, color='#1ED760', fontweight='bold', y=0.98)

# Plot 1: Scatter plot with regression line
ax1 = axes[0, 0]
scatter = ax1.scatter(df_sample['popularity_norm'], df_sample['hit_score'], 
                     alpha=0.3, s=10, c=df_sample['hit_score'], 
                     cmap='RdYlGn', edgecolors='none')
ax1.set_xlabel('Popularity (Original Scale)', fontsize=12, color='#FFFFFF', fontweight='bold')
ax1.set_ylabel('Hit Score (0-100)', fontsize=12, color='#FFFFFF', fontweight='bold')
ax1.set_title('Popularity vs Hit Score Scatter Plot', fontsize=13, color='#1ED760', fontweight='bold')
ax1.grid(alpha=0.3, color='#1ED760')

# Add regression line
z = np.polyfit(df_sample['popularity_norm'], df_sample['hit_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_sample['popularity_norm'].min(), df_sample['popularity_norm'].max(), 100)
ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={np.corrcoef(df_sample["popularity_norm"], df_sample["hit_score"])[0,1]:.3f})')
ax1.legend(fontsize=10)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Hit Score', fontsize=10, color='#FFFFFF')
cbar.ax.tick_params(colors='#FFFFFF')

# Plot 2: Distribution comparison
ax2 = axes[0, 1]
ax2.hist(df_sample['popularity_norm'], bins=50, alpha=0.6, label='Popularity', 
        color='#1ED760', edgecolor='white', density=True)
ax2_twin = ax2.twinx()
ax2_twin.hist(df_sample['hit_score'], bins=50, alpha=0.6, label='Hit Score', 
             color='#FF6B6B', edgecolor='white', density=True)
ax2.set_xlabel('Popularity / Hit Score', fontsize=12, color='#FFFFFF', fontweight='bold')
ax2.set_ylabel('Density (Popularity)', fontsize=11, color='#1ED760', fontweight='bold')
ax2_twin.set_ylabel('Density (Hit Score)', fontsize=11, color='#FF6B6B', fontweight='bold')
ax2.set_title('Distribution Comparison: Popularity vs Hit Score', 
             fontsize=13, color='#1ED760', fontweight='bold')
ax2.legend(loc='upper left', fontsize=10, framealpha=0.8)
ax2_twin.legend(loc='upper right', fontsize=10, framealpha=0.8)
ax2.grid(alpha=0.3, color='#1ED760', axis='x')
ax2_twin.grid(alpha=0.3, color='#FF6B6B', axis='y')

# Plot 3: Hit Score by Popularity Categories
ax3 = axes[1, 0]
# Create popularity categories
df_sample['popularity_category'] = pd.cut(df_sample['popularity_norm'], 
                                         bins=[0, 25, 50, 75, 100],
                                         labels=['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Very High (75-100)'])
box_data = [df_sample[df_sample['popularity_category'] == cat]['hit_score'].values 
           for cat in df_sample['popularity_category'].cat.categories]

bp = ax3.boxplot(box_data, tick_labels=df_sample['popularity_category'].cat.categories,
                patch_artist=True,
                boxprops=dict(facecolor='#1ED760', alpha=0.7),
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#1ED760'),
                capprops=dict(color='#1ED760'))
ax3.set_xlabel('Popularity Category', fontsize=12, color='#FFFFFF', fontweight='bold')
ax3.set_ylabel('Hit Score', fontsize=12, color='#FFFFFF', fontweight='bold')
ax3.set_title('Hit Score Distribution by Popularity Category', 
             fontsize=13, color='#1ED760', fontweight='bold')
ax3.tick_params(colors='#FFFFFF', rotation=15)
ax3.grid(alpha=0.3, color='#1ED760', axis='y')

# Plot 4: Correlation Heatmap
ax4 = axes[1, 1]
# Select numeric features for correlation
numeric_cols = ['popularity_norm', 'hit_score']
if 'danceability' in df_sample.columns:
    numeric_cols.extend(['danceability', 'energy', 'valence'])
elif 'DANCEABILITY' in df_sample.columns:
    numeric_cols.extend(['DANCEABILITY', 'ENERGY', 'VALENCE'])

available_cols = [col for col in numeric_cols if col in df_sample.columns]
corr_data = df_sample[available_cols].corr()

im = ax4.imshow(corr_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_data.columns)))
ax4.set_yticks(range(len(corr_data.columns)))
ax4.set_xticklabels(corr_data.columns, rotation=45, ha='right', fontsize=9, color='#FFFFFF')
ax4.set_yticklabels(corr_data.columns, fontsize=9, color='#FFFFFF')
ax4.set_title('Correlation: Popularity & Hit Score with Audio Features', 
             fontsize=13, color='#1ED760', fontweight='bold')

# Add correlation values
for i in range(len(corr_data.columns)):
    for j in range(len(corr_data.columns)):
        text = ax4.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="white", fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax4, label='Correlation', shrink=0.8)
ax4.grid(alpha=0.3, color='#1ED760')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_DIR / '20_popularity_vs_hit_score_analysis.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 20_popularity_vs_hit_score_analysis.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Popularity vs Hit Score Relationship Details
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Popularity vs Hit Score: Detailed Relationship Analysis', 
            fontsize=16, color='#1ED760', fontweight='bold')

# Plot 1: Hexbin plot
ax1 = axes[0]
hb = ax1.hexbin(df_sample['popularity_norm'], df_sample['hit_score'], 
               gridsize=30, cmap='RdYlGn', mincnt=1)
ax1.set_xlabel('Popularity (Original Scale)', fontsize=12, color='#FFFFFF', fontweight='bold')
ax1.set_ylabel('Hit Score (0-100)', fontsize=12, color='#FFFFFF', fontweight='bold')
ax1.set_title('Popularity vs Hit Score Density', fontsize=13, color='#1ED760', fontweight='bold')
ax1.grid(alpha=0.3, color='#1ED760')
cb = plt.colorbar(hb, ax=ax1)
cb.set_label('Count', fontsize=10, color='#FFFFFF')
cb.ax.tick_params(colors='#FFFFFF')

# Plot 2: Hit Score Percentiles by Popularity
ax2 = axes[1]
popularity_bins = np.linspace(df_sample['popularity_norm'].min(), df_sample['popularity_norm'].max(), 20)
bin_centers = (popularity_bins[:-1] + popularity_bins[1:]) / 2
percentiles_data = []
for i in range(len(popularity_bins)-1):
    mask = (df_sample['popularity_norm'] >= popularity_bins[i]) & (df_sample['popularity_norm'] < popularity_bins[i+1])
    if mask.sum() > 0:
        hit_scores_in_bin = df_sample[mask]['hit_score']
        percentiles_data.append({
            'popularity': bin_centers[i],
            'p25': np.percentile(hit_scores_in_bin, 25),
            'p50': np.percentile(hit_scores_in_bin, 50),
            'p75': np.percentile(hit_scores_in_bin, 75)
        })

percentiles_df = pd.DataFrame(percentiles_data)
ax2.fill_between(percentiles_df['popularity'], percentiles_df['p25'], percentiles_df['p75'], 
                alpha=0.3, color='#1ED760', label='25th-75th Percentile')
ax2.plot(percentiles_df['popularity'], percentiles_df['p50'], 'o-', color='#1ED760', 
        linewidth=2, markersize=4, label='Median Hit Score')
ax2.set_xlabel('Popularity (Original Scale)', fontsize=12, color='#FFFFFF', fontweight='bold')
ax2.set_ylabel('Hit Score', fontsize=12, color='#FFFFFF', fontweight='bold')
ax2.set_title('Hit Score Percentiles by Popularity', fontsize=13, color='#1ED760', fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.8)
ax2.grid(alpha=0.3, color='#1ED760')

# Plot 3: Statistics Summary
ax3 = axes[2]
ax3.axis('off')

# Calculate statistics
correlation = np.corrcoef(df_sample['popularity_norm'], df_sample['hit_score'])[0, 1]
spearman_corr, spearman_p = stats.spearmanr(df_sample['popularity_norm'], df_sample['hit_score'])

stats_text = f"""
POPULARITY vs HIT SCORE
STATISTICS SUMMARY

Sample Size: {len(df_sample):,} songs

Popularity Statistics:
  Mean: {df_sample['popularity_norm'].mean():.2f}
  Median: {df_sample['popularity_norm'].median():.2f}
  Std Dev: {df_sample['popularity_norm'].std():.2f}
  Min: {df_sample['popularity_norm'].min():.2f}
  Max: {df_sample['popularity_norm'].max():.2f}

Hit Score Statistics:
  Mean: {df_sample['hit_score'].mean():.2f}
  Median: {df_sample['hit_score'].median():.2f}
  Std Dev: {df_sample['hit_score'].std():.2f}
  Min: {df_sample['hit_score'].min():.2f}
  Max: {df_sample['hit_score'].max():.2f}

Correlation Analysis:
  Pearson Correlation: {correlation:.4f}
  Spearman Correlation: {spearman_corr:.4f}
  P-value: {spearman_p:.2e}

Relationship Interpretation:
  Hit Score is derived from Popularity
  using percentile-based mapping (0-100 scale).
  
  Higher popularity → Higher hit score
  Strong positive correlation indicates
  effective normalization and scaling.
"""

ax3.text(0.1, 0.5, stats_text, fontsize=11, color='#FFFFFF', 
        verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#282828', alpha=0.9, edgecolor=SPOTIFY_GREEN, linewidth=2))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '21_popularity_hit_relationship_details.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 21_popularity_hit_relationship_details.png")
plt.close()

print(f"\n=== POPULARITY vs HIT SCORE ANALYSIS COMPLETE ===")
print(f"Correlation: {correlation:.4f}")
print(f"All visualizations saved to: {OUTPUT_DIR}")

