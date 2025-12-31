#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete EDA Visualizations for Final Presentation
Includes: Basic EDA, Speechiness Analysis, Loudness Filtering Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

# Load data
train_path = BASE_DIR / 'data' / 'processed' / 'spotify_emotion_train.csv'
df = pd.read_csv(train_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# PART 1: BASIC EDA VISUALIZATIONS
# ============================================================================

# 1. Audio Features Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Audio Features Distribution', fontsize=16, color='#1ED760', fontweight='bold')

features = ['danceability', 'energy', 'valence', 'loudness', 'speechiness', 'acousticness']
for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    if feature == 'loudness':
        df[feature].hist(bins=50, ax=ax, color='#1ED760', alpha=0.7, edgecolor='white')
    else:
        df[feature].hist(bins=50, ax=ax, color='#1ED760', alpha=0.7, edgecolor='white')
    
    ax.set_title(f'{feature.capitalize()}', fontsize=12, color='#1ED760', fontweight='bold')
    ax.set_xlabel(feature.capitalize(), color='#FFFFFF')
    ax.set_ylabel('Frequency', color='#FFFFFF')
    ax.grid(alpha=0.3, color='#1ED760')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_audio_features_distribution.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 1_audio_features_distribution.png")
plt.close()

# 2. Correlation Heatmap
numeric_cols = ['danceability', 'energy', 'valence', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'tempo']
if 'popularity' in df.columns:
    numeric_cols.append('popularity')
elif 'POPULARITY' in df.columns:
    numeric_cols.append('POPULARITY')

corr_data = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, annot_kws={'color': 'white', 'fontsize': 9})
plt.title('Audio Features Correlation Matrix', fontsize=16, color='#1ED760', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_correlation_heatmap.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 2_correlation_heatmap.png")
plt.close()

# 3. Feature vs Popularity (if available)
pop_col = None
for col in ['popularity', 'POPULARITY', 'streams', 'STREAMS']:
    if col in df.columns:
        pop_col = col
        break

if pop_col:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Feature vs {pop_col.capitalize()}', fontsize=16, color='#1ED760', fontweight='bold')
    
    for i, feature in enumerate(features[:6]):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Scatter plot with regression line
        sample = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        ax.scatter(sample[feature], sample[pop_col], alpha=0.3, color='#1ED760', s=10)
        
        # Regression line
        z = np.polyfit(sample[feature].dropna(), sample[pop_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(sample[feature].sort_values(), p(sample[feature].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_title(f'{feature.capitalize()} vs {pop_col.capitalize()}', 
                    fontsize=12, color='#1ED760', fontweight='bold')
        ax.set_xlabel(feature.capitalize(), color='#FFFFFF')
        ax.set_ylabel(pop_col.capitalize(), color='#FFFFFF')
        ax.legend()
        ax.grid(alpha=0.3, color='#1ED760')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_feature_vs_popularity.png', dpi=300, facecolor='#121212', bbox_inches='tight')
    print("Saved: 3_feature_vs_popularity.png")
    plt.close()

# 4. Genre Analysis (if available)
if 'GENRE' in df.columns or 'genre' in df.columns:
    genre_col = 'GENRE' if 'GENRE' in df.columns else 'genre'
    genre_counts = df[genre_col].value_counts().head(15)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(genre_counts)), genre_counts.values, color='#1ED760', alpha=0.8)
    plt.yticks(range(len(genre_counts)), genre_counts.index, color='#FFFFFF')
    plt.xlabel('Count', color='#FFFFFF', fontsize=12)
    plt.title('Top 15 Genres Distribution', fontsize=16, color='#1ED760', fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, color='#1ED760')
    
    # Add value labels
    for i, (idx, val) in enumerate(genre_counts.items()):
        plt.text(val + max(genre_counts) * 0.01, i, f'{val:,}', 
                va='center', color='#FFFFFF', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_genre_distribution.png', dpi=300, facecolor='#121212', bbox_inches='tight')
    print("Saved: 4_genre_distribution.png")
    plt.close()

# 5. Year Analysis (if available)
year_cols = [col for col in df.columns if 'year' in col.lower() or 'release' in col.lower()]
if year_cols:
    year_col = year_cols[0]
    df_year = df.dropna(subset=[year_col])
    
    if len(df_year) > 0:
        # Year distribution
        plt.figure(figsize=(14, 6))
        year_counts = df_year[year_col].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, color='#1ED760', linewidth=2, marker='o', markersize=4)
        plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#1ED760')
        plt.title('Songs Distribution by Release Year', fontsize=16, color='#1ED760', fontweight='bold', pad=20)
        plt.xlabel('Year', color='#FFFFFF', fontsize=12)
        plt.ylabel('Number of Songs', color='#FFFFFF', fontsize=12)
        plt.grid(alpha=0.3, color='#1ED760')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '5_year_distribution.png', dpi=300, facecolor='#121212', bbox_inches='tight')
        print("Saved: 5_year_distribution.png")
        plt.close()
        
        # Feature evolution over time
        if pop_col:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Audio Features Evolution Over Time', fontsize=16, color='#1ED760', fontweight='bold')
            
            key_features = ['danceability', 'energy', 'valence', 'tempo']
            for i, feature in enumerate(key_features):
                if feature in df_year.columns:
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col]
                    
                    yearly_avg = df_year.groupby(year_col)[feature].mean().rolling(window=3).mean()
                    ax.plot(yearly_avg.index, yearly_avg.values, color='#1ED760', linewidth=2, marker='o', markersize=3)
                    ax.fill_between(yearly_avg.index, yearly_avg.values, alpha=0.3, color='#1ED760')
                    ax.set_title(f'{feature.capitalize()} Over Time', fontsize=12, color='#1ED760', fontweight='bold')
                    ax.set_xlabel('Year', color='#FFFFFF')
                    ax.set_ylabel(feature.capitalize(), color='#FFFFFF')
                    ax.grid(alpha=0.3, color='#1ED760')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / '6_feature_evolution.png', dpi=300, facecolor='#121212', bbox_inches='tight')
            print("Saved: 6_feature_evolution.png")
            plt.close()

# 6. Box plots for key features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Audio Features Box Plots', fontsize=16, color='#1ED760', fontweight='bold')

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    bp = ax.boxplot(df[feature].dropna(), patch_artist=True, 
                    boxprops=dict(facecolor='#1ED760', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#1ED760'),
                    capprops=dict(color='#1ED760'))
    
    ax.set_title(f'{feature.capitalize()}', fontsize=12, color='#1ED760', fontweight='bold')
    ax.set_ylabel('Value', color='#FFFFFF')
    ax.grid(alpha=0.3, color='#1ED760', axis='y')
    ax.set_xticklabels([''])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7_boxplots.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 7_boxplots.png")
plt.close()

# ============================================================================
# PART 2: SPEECHINESS ANALYSIS
# ============================================================================

print("\n=== SPEECHINESS ANALYSIS ===")

# Genre column
genre_col = 'genre' if 'genre' in df.columns else 'GENRE' if 'GENRE' in df.columns else None

if genre_col is None:
    print("Warning: Genre column not found! Skipping speechiness analysis.")
else:
    # 7. Speechiness Distribution by Genre (High speechiness songs)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Speechiness Analysis: High Speechiness Songs by Genre', 
                 fontsize=16, color='#1ED760', fontweight='bold', y=0.98)
    
    # High speechiness threshold
    high_speechiness = df[df['speechiness'] >= 0.70]
    
    # Plot 1: High speechiness songs count by genre
    ax1 = axes[0, 0]
    genre_counts = high_speechiness[genre_col].value_counts().head(15)
    bars = ax1.barh(range(len(genre_counts)), genre_counts.values, color='#1ED760', alpha=0.8)
    ax1.set_yticks(range(len(genre_counts)))
    ax1.set_yticklabels(genre_counts.index, color='#FFFFFF', fontsize=10)
    ax1.set_xlabel('Number of Songs (speechiness >= 0.70)', color='#FFFFFF', fontsize=11)
    ax1.set_title('High Speechiness Songs by Genre', fontsize=12, color='#1ED760', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, color='#1ED760')
    
    # Add value labels
    for i, (idx, val) in enumerate(genre_counts.items()):
        ax1.text(val + max(genre_counts) * 0.01, i, f'{val:,}', 
                va='center', color='#FFFFFF', fontsize=9)
    
    # Plot 2: Speechiness distribution by top genres
    ax2 = axes[0, 1]
    top_genres = high_speechiness[genre_col].value_counts().head(8).index
    df_top_genres = df[df[genre_col].isin(top_genres)]
    
    # Box plot
    box_data = [df_top_genres[df_top_genres[genre_col] == genre]['speechiness'].values 
                for genre in top_genres]
    bp = ax2.boxplot(box_data, tick_labels=top_genres, patch_artist=True,
                     boxprops=dict(facecolor='#1ED760', alpha=0.7),
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color='#1ED760'),
                     capprops=dict(color='#1ED760'))
    ax2.set_title('Speechiness Distribution by Top Genres', fontsize=12, color='#1ED760', fontweight='bold')
    ax2.set_ylabel('Speechiness', color='#FFFFFF', fontsize=11)
    ax2.set_xlabel('Genre', color='#FFFFFF', fontsize=11)
    ax2.tick_params(axis='x', rotation=45, labelsize=9, colors='#FFFFFF')
    ax2.tick_params(axis='y', colors='#FFFFFF')
    ax2.grid(alpha=0.3, color='#1ED760', axis='y')
    
    # Plot 3: Speechiness vs Popularity by Genre (scatter)
    ax3 = axes[1, 0]
    sample_size = min(3000, len(df))
    df_sample = df.sample(sample_size)
    
    # Color by genre for top genres
    top_genres_list = df[genre_col].value_counts().head(5).index
    colors_map = {genre: f'C{i}' for i, genre in enumerate(top_genres_list)}
    
    for genre in top_genres_list:
        genre_data = df_sample[df_sample[genre_col] == genre]
        if len(genre_data) > 0:
            ax3.scatter(genre_data['speechiness'], genre_data['popularity'], 
                       alpha=0.4, s=20, label=genre, color=colors_map.get(genre, '#1ED760'))
    
    ax3.set_xlabel('Speechiness', color='#FFFFFF', fontsize=11)
    ax3.set_ylabel('Popularity', color='#FFFFFF', fontsize=11)
    ax3.set_title('Speechiness vs Popularity by Genre', fontsize=12, color='#1ED760', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax3.grid(alpha=0.3, color='#1ED760')
    
    # Plot 4: Average speechiness by genre (bar chart)
    ax4 = axes[1, 1]
    avg_speechiness = df.groupby(genre_col)['speechiness'].mean().sort_values(ascending=False).head(15)
    bars = ax4.barh(range(len(avg_speechiness)), avg_speechiness.values, color='#1ED760', alpha=0.8)
    ax4.set_yticks(range(len(avg_speechiness)))
    ax4.set_yticklabels(avg_speechiness.index, color='#FFFFFF', fontsize=10)
    ax4.set_xlabel('Average Speechiness', color='#FFFFFF', fontsize=11)
    ax4.set_title('Average Speechiness by Genre', fontsize=12, color='#1ED760', fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3, color='#1ED760')
    
    # Add value labels
    for i, (idx, val) in enumerate(avg_speechiness.items()):
        ax4.text(val + max(avg_speechiness) * 0.01, i, f'{val:.3f}', 
                va='center', color='#FFFFFF', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / '8_speechiness_genre_analysis.png', dpi=300, facecolor='#121212', bbox_inches='tight')
    print("Saved: 8_speechiness_genre_analysis.png")
    plt.close()
    
    # 8. Detailed analysis: High speechiness songs (>= 0.90)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Very High Speechiness Analysis (speechiness >= 0.90)', 
                 fontsize=16, color='#1ED760', fontweight='bold')
    
    very_high_speechiness = df[df['speechiness'] >= 0.90]
    
    # Plot 1: Genre distribution of very high speechiness songs
    ax1 = axes[0]
    genre_counts_vh = very_high_speechiness[genre_col].value_counts().head(10)
    bars = ax1.bar(range(len(genre_counts_vh)), genre_counts_vh.values, color='#1ED760', alpha=0.8)
    ax1.set_xticks(range(len(genre_counts_vh)))
    ax1.set_xticklabels(genre_counts_vh.index, rotation=45, ha='right', fontsize=10, color='#FFFFFF')
    ax1.set_ylabel('Number of Songs', color='#FFFFFF', fontsize=11)
    ax1.set_title('Genre Distribution (speechiness >= 0.90)', fontsize=12, color='#1ED760', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, color='#1ED760')
    
    # Add value labels
    for i, (idx, val) in enumerate(genre_counts_vh.items()):
        ax1.text(i, val + max(genre_counts_vh) * 0.01, f'{val}', 
                ha='center', va='bottom', color='#FFFFFF', fontsize=9)
    
    # Plot 2: Speechiness distribution comparison
    ax2 = axes[1]
    ax2.hist(df[df['speechiness'] < 0.90]['speechiness'], bins=50, alpha=0.6, 
             label='Normal Speechiness (< 0.90)', color='#1ED760', edgecolor='white')
    ax2.hist(very_high_speechiness['speechiness'], bins=30, alpha=0.8, 
             label='Very High Speechiness (>= 0.90)', color='#FF6B6B', edgecolor='white')
    ax2.set_xlabel('Speechiness', color='#FFFFFF', fontsize=11)
    ax2.set_ylabel('Frequency', color='#FFFFFF', fontsize=11)
    ax2.set_title('Speechiness Distribution Comparison', fontsize=12, color='#1ED760', fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.grid(alpha=0.3, color='#1ED760')
    
    # Plot 3: Popularity comparison
    ax3 = axes[2]
    normal_pop = df[df['speechiness'] < 0.90]['popularity'].dropna()
    high_pop = very_high_speechiness['popularity'].dropna()
    
    box_data_comp = [normal_pop, high_pop]
    bp = ax3.boxplot(box_data_comp, tick_labels=['Normal\n(< 0.90)', 'Very High\n(>= 0.90)'], 
                     patch_artist=True,
                     boxprops=dict(facecolor='#1ED760', alpha=0.7),
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color='#1ED760'),
                     capprops=dict(color='#1ED760'))
    ax3.set_ylabel('Popularity', color='#FFFFFF', fontsize=11)
    ax3.set_title('Popularity Comparison', fontsize=12, color='#1ED760', fontweight='bold')
    ax3.tick_params(colors='#FFFFFF')
    ax3.grid(alpha=0.3, color='#1ED760', axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '9_very_high_speechiness_analysis.png', dpi=300, facecolor='#121212', bbox_inches='tight')
    print("Saved: 9_very_high_speechiness_analysis.png")
    plt.close()
    
    # Print statistics
    print(f"Total songs: {len(df):,}")
    print(f"Songs with speechiness >= 0.70: {len(high_speechiness):,} ({len(high_speechiness)/len(df)*100:.2f}%)")
    print(f"Songs with speechiness >= 0.90: {len(very_high_speechiness):,} ({len(very_high_speechiness)/len(df)*100:.2f}%)")
    print(f"\nTop genres for speechiness >= 0.90:")
    for genre, count in very_high_speechiness[genre_col].value_counts().head(10).items():
        print(f"  {genre}: {count:,} songs")
    
    print(f"\nAverage speechiness by top genres:")
    for genre in top_genres_list:
        avg = df[df[genre_col] == genre]['speechiness'].mean()
        print(f"  {genre}: {avg:.3f}")

# ============================================================================
# PART 3: LOUDNESS FILTERING ANALYSIS
# ============================================================================

print("\n=== LOUDNESS FILTERING ANALYSIS ===")

# Threshold
LOUDNESS_THRESHOLD = -40.0

# Filter data
df_before = df.copy()
df_after = df[df['loudness'] >= LOUDNESS_THRESHOLD].copy()

print(f"Before filtering: {len(df_before):,} songs")
print(f"After filtering (loudness >= -40 dB): {len(df_after):,} songs")
print(f"Removed: {len(df_before) - len(df_after):,} songs ({(len(df_before) - len(df_after))/len(df_before)*100:.2f}%)")

# 9. Loudness Distribution with Threshold
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Loudness Analysis: -40 dB Threshold Filtering', 
             fontsize=16, color='#1ED760', fontweight='bold', y=0.98)

# Plot 1: Loudness distribution before and after
ax1 = axes[0, 0]
ax1.hist(df_before['loudness'], bins=80, alpha=0.6, label='Before Filtering', 
         color='#1ED760', edgecolor='white', density=True)
ax1.hist(df_after['loudness'], bins=60, alpha=0.8, label='After Filtering (>= -40 dB)', 
         color='#FF6B6B', edgecolor='white', density=True)
ax1.axvline(LOUDNESS_THRESHOLD, color='yellow', linestyle='--', linewidth=2, 
            label=f'Threshold (-40 dB)')
ax1.set_xlabel('Loudness (dB)', color='#FFFFFF', fontsize=11)
ax1.set_ylabel('Density', color='#FFFFFF', fontsize=11)
ax1.set_title('Loudness Distribution: Before vs After Filtering', 
              fontsize=12, color='#1ED760', fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.8)
ax1.grid(alpha=0.3, color='#1ED760')

# Plot 2: Removed songs by loudness range
ax2 = axes[0, 1]
removed = df_before[df_before['loudness'] < LOUDNESS_THRESHOLD]
bins = np.arange(-60, -35, 2.5)
counts, edges = np.histogram(removed['loudness'], bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
bars = ax2.bar(centers, counts, width=2, color='#FF6B6B', alpha=0.8, edgecolor='white')
ax2.axvline(LOUDNESS_THRESHOLD, color='yellow', linestyle='--', linewidth=2, 
            label=f'Threshold (-40 dB)')
ax2.set_xlabel('Loudness (dB)', color='#FFFFFF', fontsize=11)
ax2.set_ylabel('Number of Removed Songs', color='#FFFFFF', fontsize=11)
ax2.set_title('Removed Songs by Loudness Range', 
              fontsize=12, color='#1ED760', fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.8)
ax2.grid(alpha=0.3, color='#1ED760', axis='y')

# Add value labels on bars
for i, (center, count) in enumerate(zip(centers, counts)):
    if count > 0:
        ax2.text(center, count + max(counts) * 0.01, f'{int(count)}', 
                ha='center', va='bottom', color='#FFFFFF', fontsize=8)

# Plot 3: Popularity comparison
ax3 = axes[1, 0]
box_data = [
    df_before['popularity'].dropna(),
    df_after['popularity'].dropna(),
    removed['popularity'].dropna() if len(removed) > 0 else []
]
bp = ax3.boxplot([d for d in box_data if len(d) > 0], 
                 tick_labels=['Before\nFiltering', 'After\nFiltering', 'Removed\nSongs'],
                 patch_artist=True,
                 boxprops=dict(facecolor='#1ED760', alpha=0.7),
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(color='#1ED760'),
                 capprops=dict(color='#1ED760'))
ax3.set_ylabel('Popularity', color='#FFFFFF', fontsize=11)
ax3.set_title('Popularity Comparison', fontsize=12, color='#1ED760', fontweight='bold')
ax3.tick_params(colors='#FFFFFF')
ax3.grid(alpha=0.3, color='#1ED760', axis='y')

# Plot 4: Statistics summary
ax4 = axes[1, 1]
ax4.axis('off')

stats_text = f"""
FILTERING STATISTICS

Total Songs (Before): {len(df_before):,}
Songs Removed: {len(df_before) - len(df_after):,}
  ({((len(df_before) - len(df_after))/len(df_before)*100):.2f}%)

Remaining Songs: {len(df_after):,}
  ({(len(df_after)/len(df_before)*100):.2f}%)

Loudness Range (Removed):
  Min: {removed['loudness'].min():.2f} dB
  Max: {removed['loudness'].max():.2f} dB
  Mean: {removed['loudness'].mean():.2f} dB

Loudness Range (Remaining):
  Min: {df_after['loudness'].min():.2f} dB
  Max: {df_after['loudness'].max():.2f} dB
  Mean: {df_after['loudness'].mean():.2f} dB

REASON:
Songs with loudness < -40 dB are
considered 'silent' or have weak
signal content, unsuitable for
popularity modeling.
"""

ax4.text(0.1, 0.5, stats_text, fontsize=11, color='#FFFFFF', 
         verticalalignment='center', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#282828', alpha=0.8, edgecolor='#1ED760'))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_DIR / '10_loudness_filtering_analysis.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 10_loudness_filtering_analysis.png")
plt.close()

# 10. Detailed loudness analysis
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Loudness Filtering Impact Analysis', 
             fontsize=16, color='#1ED760', fontweight='bold')

# Plot 1: Loudness distribution with threshold line
ax1 = axes[0]
ax1.hist(df_before['loudness'], bins=100, alpha=0.7, color='#1ED760', edgecolor='white', density=True)
ax1.axvline(LOUDNESS_THRESHOLD, color='red', linestyle='--', linewidth=3, 
            label=f'Threshold: -40 dB')
ax1.axvspan(-60, LOUDNESS_THRESHOLD, alpha=0.3, color='red', label='Removed Region')
ax1.set_xlabel('Loudness (dB)', color='#FFFFFF', fontsize=11)
ax1.set_ylabel('Density', color='#FFFFFF', fontsize=11)
ax1.set_title('Loudness Distribution with Threshold', 
              fontsize=12, color='#1ED760', fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.8)
ax1.grid(alpha=0.3, color='#1ED760')

# Plot 2: Loudness vs Popularity (scatter)
ax2 = axes[1]
sample_before = df_before.sample(min(5000, len(df_before)))
sample_after = df_after.sample(min(5000, len(df_after)))
sample_removed = removed.sample(min(500, len(removed))) if len(removed) > 0 else pd.DataFrame()

ax2.scatter(sample_after['loudness'], sample_after['popularity'], 
           alpha=0.4, s=15, color='#1ED760', label='Kept (>= -40 dB)')
if len(sample_removed) > 0:
    ax2.scatter(sample_removed['loudness'], sample_removed['popularity'], 
               alpha=0.6, s=15, color='#FF6B6B', label='Removed (< -40 dB)')
ax2.axvline(LOUDNESS_THRESHOLD, color='yellow', linestyle='--', linewidth=2, 
            label='Threshold')
ax2.set_xlabel('Loudness (dB)', color='#FFFFFF', fontsize=11)
ax2.set_ylabel('Popularity', color='#FFFFFF', fontsize=11)
ax2.set_title('Loudness vs Popularity', fontsize=12, color='#1ED760', fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.8)
ax2.grid(alpha=0.3, color='#1ED760')

# Plot 3: Count comparison
ax3 = axes[2]
categories = ['Before\nFiltering', 'After\nFiltering\n(>= -40 dB)', 'Removed\n(< -40 dB)']
counts = [len(df_before), len(df_after), len(df_before) - len(df_after)]
colors = ['#1ED760', '#1ED760', '#FF6B6B']
bars = ax3.bar(categories, counts, color=colors, alpha=0.8, edgecolor='white')
ax3.set_ylabel('Number of Songs', color='#FFFFFF', fontsize=11)
ax3.set_title('Song Count Comparison', fontsize=12, color='#1ED760', fontweight='bold')
ax3.tick_params(colors='#FFFFFF')
ax3.grid(alpha=0.3, color='#1ED760', axis='y')

# Add value labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
            f'{count:,}\n({count/len(df_before)*100:.2f}%)',
            ha='center', va='bottom', color='#FFFFFF', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / '11_loudness_impact_analysis.png', dpi=300, facecolor='#121212', bbox_inches='tight')
print("Saved: 11_loudness_impact_analysis.png")
plt.close()

# Print summary
print(f"\nThreshold: {LOUDNESS_THRESHOLD} dB")
print(f"Total songs before: {len(df_before):,}")
print(f"Songs removed: {len(df_before) - len(df_after):,} ({(len(df_before) - len(df_after))/len(df_before)*100:.2f}%)")
print(f"Songs remaining: {len(df_after):,} ({len(df_after)/len(df_before)*100:.2f}%)")
if len(removed) > 0:
    print(f"\nRemoved songs loudness range:")
    print(f"  Min: {removed['loudness'].min():.2f} dB")
    print(f"  Max: {removed['loudness'].max():.2f} dB")
    print(f"  Mean: {removed['loudness'].mean():.2f} dB")
    print(f"  Median: {removed['loudness'].median():.2f} dB")

print(f"\n=== ALL VISUALIZATIONS COMPLETE ===")
print(f"All visualizations saved to: {OUTPUT_DIR}")
