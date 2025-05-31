#!/usr/bin/env python3
"""
Random Forest Model Results - Simplified Figures
===============================================

Creates simplified versions of the showcase figures with only the requested plots:
1. Speed distribution analysis (no runner names)
2. Feature importance plot only
3. Route comparison heatmap with least cost paths (all runners, no names)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Fallback configuration
COLORS = {
    'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01'
}
ORIENTEERING_COLORS = {
    'fast_terrain': '#90EE90', 'medium_terrain': '#FFD700', 
    'slow_terrain': '#FF6347', 'impassible': '#8B0000',
    'paths': '#8B4513', 'controls': '#FF1493'
}
COST_COLORMAP = 'YlOrRd'

# Simple figure manager fallback
class FigureManager:
    def register_figure(self, **kwargs):
        base_dir = Path("output/figures") / kwargs['category'] / kwargs['subcategory']
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = kwargs['filename']
        formats = kwargs.get('formats', ['png', 'pdf'])
        return {fmt: base_dir / f"{filename}.{fmt}" for fmt in formats}

# Initialize figure manager
fig_manager = FigureManager()

def load_random_forest_data():
    """Load all Random Forest model data and outputs."""
    print("Loading Random Forest model data...")
    
    data = {}
    
    # Load cost surface
    try:
        cost_surface_path = "output/cost_surfaces/random_forest_cost_surface.tif"
        with rasterio.open(cost_surface_path) as src:
            data['cost_surface'] = src.read(1)
            data['cost_transform'] = src.transform
            data['cost_crs'] = src.crs
            data['cost_bounds'] = src.bounds
        print(f"✓ Loaded cost surface: {data['cost_surface'].shape}")
    except Exception as e:
        print(f"✗ Could not load cost surface: {e}")
        data['cost_surface'] = None
    
    # Load least cost paths
    try:
        paths_path = "output/cost_surfaces/least_cost_paths_rf.geojson"
        data['optimal_paths'] = gpd.read_file(paths_path)
        print(f"✓ Loaded {len(data['optimal_paths'])} optimal path segments")
    except Exception as e:
        print(f"✗ Could not load optimal paths: {e}")
        data['optimal_paths'] = None
    
    # Load GPS tracks for actual paths
    try:
        gps_data_path = "data/derived/csv/track_features.csv"
        data['gps_tracks'] = pd.read_csv(gps_data_path)
        from shapely import wkt
        data['gps_tracks']['geometry'] = data['gps_tracks']['geometry'].apply(wkt.loads)
        data['gps_tracks'] = gpd.GeoDataFrame(data['gps_tracks'])
        print(f"✓ Loaded {len(data['gps_tracks'])} GPS track points")
    except Exception as e:
        print(f"✗ Could not load GPS tracks: {e}")
        data['gps_tracks'] = None
    
    # Load control points
    try:
        controls_path = "data/derived/map/control_points_race_3006.shp"
        data['control_points'] = gpd.read_file(controls_path)
        print(f"✓ Loaded {len(data['control_points'])} control points")
    except Exception as e:
        print(f"✗ Could not load control points: {e}")
        data['control_points'] = None
    
    # Load feature importance
    try:
        importance_path = "output/model_trace/feature_importance.csv"
        data['feature_importance'] = pd.read_csv(importance_path)
        print(f"✓ Loaded feature importance for {len(data['feature_importance'])} features")
    except Exception as e:
        print(f"✗ Could not load feature importance: {e}")
        data['feature_importance'] = None
    
    return data

def create_speed_distribution_plot(data):
    """
    Create simplified speed distribution plot only
    """
    print("Creating speed distribution plot...")
    
    if data['gps_tracks'] is None or 'speed' not in data['gps_tracks'].columns:
        print("✗ Cannot create speed distribution plot - no speed data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    gps_data = data['gps_tracks']
    speeds = gps_data['speed']
    speeds_clean = speeds[(speeds >= 0.5) & (speeds <= 25)]
    
    # Create histogram
    ax.hist(speeds_clean, bins=50, alpha=0.7, color=COLORS['primary'], 
            edgecolor='black')
    
    # Add mean and median lines
    ax.axvline(speeds_clean.mean(), color='red', linestyle='--', 
               linewidth=3, label=f'Mean: {speeds_clean.mean():.1f} km/h')
    ax.axvline(speeds_clean.median(), color='orange', linestyle='--', 
               linewidth=3, label=f'Median: {speeds_clean.median():.1f} km/h')
    
    # Add percentiles
    p25 = np.percentile(speeds_clean, 25)
    p75 = np.percentile(speeds_clean, 75)
    ax.axvline(p25, color='green', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'25th percentile: {p25:.1f} km/h')
    ax.axvline(p75, color='green', linestyle=':', 
               linewidth=2, alpha=0.7, label=f'75th percentile: {p75:.1f} km/h')
    
    ax.set_xlabel('Speed (km/h)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Speed Distribution Analysis (n={len(speeds_clean):,} GPS points)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Count: {len(speeds_clean):,}
Mean: {speeds_clean.mean():.2f} km/h
Std: {speeds_clean.std():.2f} km/h
Min: {speeds_clean.min():.1f} km/h
Max: {speeds_clean.max():.1f} km/h"""
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_simplified",
        subcategory="speed_analysis",
        filename="speed_distribution_simplified",
        description="Simplified speed distribution analysis",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved speed distribution plot to: {paths['png']}")

def create_feature_importance_plot(data):
    """
    Create feature importance plot only
    """
    print("Creating feature importance plot...")
    
    if data['feature_importance'] is None:
        print("✗ Cannot create feature importance plot - no data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    importance_df = data['feature_importance'].copy()
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Take all features or top 20 if more than 20
    if len(importance_df) > 20:
        top_features = importance_df.tail(20)
        title_suffix = " (Top 20)"
    else:
        top_features = importance_df
        title_suffix = ""
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color=COLORS['primary'], alpha=0.7, edgecolor='black')
    
    # Customize feature names for better readability
    feature_names = []
    for name in top_features['predictor']:
        # Clean up feature names
        clean_name = name.replace('_', ' ').replace('landcover ', '').title()
        if len(clean_name) > 25:
            clean_name = clean_name[:22] + '...'
        feature_names.append(clean_name)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_xlabel('Feature Importance Score', fontsize=14)
    ax.set_title(f'Random Forest Feature Importance{title_suffix}', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add importance values on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax.text(bar.get_width() + max(top_features['importance']) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', fontsize=9)
    
    # Color bars by importance level
    max_importance = top_features['importance'].max()
    for i, bar in enumerate(bars):
        importance_ratio = top_features['importance'].iloc[i] / max_importance
        if importance_ratio > 0.8:
            bar.set_color('#d73027')  # High importance - red
        elif importance_ratio > 0.5:
            bar.set_color('#fc8d59')  # Medium importance - orange
        else:
            bar.set_color('#91bfdb')  # Low importance - blue
        bar.set_alpha(0.8)
    
    # Add legend for color coding
    high_patch = mpatches.Patch(color='#d73027', alpha=0.8, label='High Importance (>80%)')
    med_patch = mpatches.Patch(color='#fc8d59', alpha=0.8, label='Medium Importance (50-80%)')
    low_patch = mpatches.Patch(color='#91bfdb', alpha=0.8, label='Lower Importance (<50%)')
    ax.legend(handles=[high_patch, med_patch, low_patch], loc='lower right')
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_simplified",
        subcategory="model_performance",
        filename="feature_importance_simplified",
        description="Simplified feature importance analysis",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature importance plot to: {paths['png']}")

def create_terrain_difficulty_plot(data):
    """
    Create terrain difficulty classification plot with optimal paths
    """
    print("Creating terrain difficulty classification plot...")
    
    if data['cost_surface'] is None:
        print("✗ Cannot create terrain difficulty plot - no cost surface data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Get cost surface data
    cost_surface = data['cost_surface']
    transform = data['cost_transform']
    
    # Create terrain difficulty classification based on cost values
    # Use percentile-based thresholds for simplified classification
    valid_costs = cost_surface[cost_surface > 0]  # Exclude no-data values
    
    if len(valid_costs) == 0:
        print("✗ No valid cost data for terrain classification")
        return
    
    # Define simplified thresholds based on cost distribution (3 categories only)
    p33 = np.percentile(valid_costs, 33)
    p67 = np.percentile(valid_costs, 67)
    
    # Create classified terrain map with only 3 meaningful categories
    terrain_classified = np.zeros_like(cost_surface, dtype=np.uint8)
    terrain_classified[cost_surface <= p33] = 1  # Fast terrain (low cost)
    terrain_classified[(cost_surface > p33) & (cost_surface <= p67)] = 2  # Medium terrain
    terrain_classified[cost_surface > p67] = 3  # Slow terrain (high cost)
    terrain_classified[cost_surface <= 0] = 0  # No data/impassable
    
    # Create custom colormap for simplified terrain difficulty
    colors = ['#000000',  # No data - black
              '#90EE90',  # Fast - light green
              '#FFD700',  # Medium - gold
              '#FF6347']  # Slow - tomato
    
    from matplotlib.colors import ListedColormap
    terrain_cmap = ListedColormap(colors)
    
    # Display terrain classification
    im = ax.imshow(terrain_classified, extent=[
        data['cost_bounds'].left, data['cost_bounds'].right,
        data['cost_bounds'].bottom, data['cost_bounds'].top
    ], cmap=terrain_cmap, alpha=0.8, aspect='auto')
    
    # Add impassable areas as very dark red if available
    impassable_mask = cost_surface <= 0
    if np.any(impassable_mask):
        impassable_display = np.zeros_like(cost_surface)
        impassable_display[impassable_mask] = 1
        ax.imshow(impassable_display, extent=[
            data['cost_bounds'].left, data['cost_bounds'].right,
            data['cost_bounds'].bottom, data['cost_bounds'].top
        ], cmap=ListedColormap(['#00000000', '#8B0000']), alpha=0.9, aspect='auto')
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(['No Data', 'Fast Terrain\n(Low Cost)', 
                        'Medium Terrain', 'Slow Terrain\n(High Cost)'])
    cbar.set_label('Terrain Difficulty Classification', fontsize=12, fontweight='bold')
    
    # Overlay optimal paths with thick, visible lines
    if data['optimal_paths'] is not None:
        # Black outline for visibility
        data['optimal_paths'].plot(ax=ax, color='black', linewidth=8, alpha=0.9, zorder=9)
        # Bright cyan main line
        data['optimal_paths'].plot(ax=ax, color='cyan', linewidth=6, alpha=1.0, 
                                  label='Optimal Least-Cost Path', zorder=10)
    
    # Overlay control points with better visibility for numbers
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax, color='blue', markersize=200, 
                                   edgecolor='white', linewidth=4, zorder=15,
                                   label='Control Points')
        
        # Add control point numbers with better visibility (white text with black outline, positioned down-right)
        if 'cont_point' in data['control_points'].columns:
            for idx, row in data['control_points'].iterrows():
                if pd.notna(row['cont_point']):
                    ax.annotate(f"{int(row['cont_point'])}", 
                               (row.geometry.x, row.geometry.y),
                               xytext=(10, -10), textcoords='offset points',
                               fontsize=14, fontweight='bold', color='white',
                               ha='left', va='top', zorder=20,
                               path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    
    ax.set_title('Terrain Difficulty Classification with Optimal Least-Cost Path', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=14)
    ax.set_ylabel('Northing (m)', fontsize=14)
    
    # Create legend
    legend_elements = []
    if data['optimal_paths'] is not None:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', linewidth=4, 
                                        label='Optimal Least-Cost Path'))
    if data['control_points'] is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='blue', markersize=15,
                                        markeredgecolor='white', markeredgewidth=3,
                                        label='Control Points', linestyle='None'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                  framealpha=0.9, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3)
    
    # Add classification statistics
    total_pixels = np.sum(terrain_classified > 0)
    if total_pixels > 0:
        stats_text = f"""Classification Statistics:
Fast Terrain: {np.sum(terrain_classified == 1)/total_pixels*100:.1f}%
Medium Terrain: {np.sum(terrain_classified == 2)/total_pixels*100:.1f}%
Slow Terrain: {np.sum(terrain_classified == 3)/total_pixels*100:.1f}%

Cost Thresholds:
Fast: ≤{p33:.1f}
Medium: {p33:.1f}-{p67:.1f}
Slow: >{p67:.1f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_simplified",
        subcategory="terrain_analysis",
        filename="terrain_difficulty_simplified",
        description="Terrain difficulty classification with optimal paths",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved terrain difficulty plot to: {paths['png']}")

def create_route_heatmap_plot(data):
    """
    Create Strava-style speed visualization with all runners and least cost paths
    """
    print("Creating Strava-style speed visualization...")
    
    if data['gps_tracks'] is None or data['optimal_paths'] is None:
        print("✗ Cannot create speed visualization - missing GPS tracks or optimal paths")
        return
    
    if 'speed' not in data['gps_tracks'].columns:
        print("✗ Cannot create speed visualization - no speed data")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Create a subtle background from the cost surface
    if data['cost_surface'] is not None:
        # Use a very light grayscale background
        im1 = show(data['cost_surface'], transform=data['cost_transform'], 
                  cmap='gray', ax=ax, alpha=0.15, vmin=0, vmax=np.percentile(data['cost_surface'][data['cost_surface'] > 0], 95))
    
    # Get GPS data with speed information
    gps_data = data['gps_tracks'].copy()
    
    # Clean speed data
    gps_data = gps_data[(gps_data['speed'] >= 0.5) & (gps_data['speed'] <= 25)]
    
    if len(gps_data) == 0:
        print("✗ No valid speed data for visualization")
        return
    
    # Extract coordinates and speeds
    x_coords = gps_data.geometry.x.values
    y_coords = gps_data.geometry.y.values
    speeds = gps_data['speed'].values
    
    # Create Strava-style speed colormap (slow = red, medium = yellow, fast = green)
    import matplotlib.colors as mcolors
    strava_colors = ['#FF0000', '#FF4500', '#FFA500', '#FFFF00', '#ADFF2F', '#00FF00']
    strava_cmap = mcolors.LinearSegmentedColormap.from_list('strava_speed', strava_colors, N=256)
    
    # Create scatter plot with speed-based colors
    scatter = ax.scatter(x_coords, y_coords, c=speeds, cmap=strava_cmap, 
                        s=8, alpha=0.7, edgecolors='none', zorder=5)
    
    # Add colorbar for speed
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Speed (km/h)', fontsize=12)
    
    # Add speed ranges to colorbar
    speed_min, speed_max = speeds.min(), speeds.max()
    cbar.set_ticks(np.linspace(speed_min, speed_max, 6))
    cbar.set_ticklabels([f'{x:.1f}' for x in np.linspace(speed_min, speed_max, 6)])
    
    # Overlay optimal paths with thick lines
    if data['optimal_paths'] is not None:
        data['optimal_paths'].plot(ax=ax, color='cyan', linewidth=6, alpha=0.9, 
                                  label='Optimal Least-Cost Path', zorder=10)
        
        # Also add a black outline for better visibility
        data['optimal_paths'].plot(ax=ax, color='black', linewidth=8, alpha=0.7, 
                                  zorder=9)
    
    # Overlay control points
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax, color='blue', markersize=150, 
                                   edgecolor='white', linewidth=3, zorder=15,
                                   label='Control Points')
        
        # Add control point numbers if available
        if 'cont_point' in data['control_points'].columns:
            for idx, row in data['control_points'].iterrows():
                if pd.notna(row['cont_point']):
                    ax.annotate(f"{int(row['cont_point'])}", 
                               (row.geometry.x, row.geometry.y),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='white',
                               ha='left', va='bottom', zorder=20,
                               path_effects=[path_effects.withStroke(linewidth=1, foreground='black')])
    
    # Add impassable areas if available
    if 'impassable_areas' in data and data['impassable_areas'] is not None:
        data['impassable_areas'].plot(ax=ax, color='red', alpha=0.6, 
                                     edgecolor='darkred', linewidth=2,
                                     label='Impassable Areas', zorder=5)
    
    ax.set_title('Speed Visualization vs Optimal Path', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=14)
    ax.set_ylabel('Northing (m)', fontsize=14)
    
    # Create custom legend
    legend_elements = []
    if data['optimal_paths'] is not None:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', linewidth=4, 
                                        label='Optimal Least-Cost Path'))
    if data['control_points'] is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='blue', markersize=12,
                                        markeredgecolor='white', markeredgewidth=2,
                                        label='Control Points', linestyle='None'))
    
    # Add speed explanation
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#00FF00', alpha=0.8,
                                       label='High Speed'))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#FFFF00', alpha=0.8,
                                       label='Medium Speed'))
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='#FF0000', alpha=0.8,
                                       label='Low Speed'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
              framealpha=0.9, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    unique_runners = gps_data['runner_id'].nunique() if 'runner_id' in gps_data.columns else 'N/A'
    optimal_length = data['optimal_paths']['length_m'].sum() if 'length_m' in data['optimal_paths'].columns else 'N/A'
    
    stats_text = f"""Speed Statistics:
Total GPS Points: {len(gps_data):,}
Unique Runners: {unique_runners}
Speed Range: {speeds.min():.1f} - {speeds.max():.1f} km/h
Mean Speed: {speeds.mean():.1f} km/h
Optimal Path Length: {optimal_length:.0f}m"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_simplified",
        subcategory="route_comparison",
        filename="route_heatmap_simplified",
        description="Strava-style speed visualization with optimal paths",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Strava-style speed plot to: {paths['png']}")

def main():
    """Generate simplified Random Forest showcase figures."""
    print("=== RANDOM FOREST SIMPLIFIED FIGURES ===")
    print("Generating simplified versions of Random Forest results...")
    
    # Load all data
    data = load_random_forest_data()
    
    # Generate simplified plots
    print("\nGenerating simplified plots...")
    
    try:
        create_terrain_difficulty_plot(data)
    except Exception as e:
        print(f"✗ Error creating terrain difficulty plot: {e}")
    
    try:
        create_speed_distribution_plot(data)
    except Exception as e:
        print(f"✗ Error creating speed distribution plot: {e}")
    
    try:
        create_feature_importance_plot(data)
    except Exception as e:
        print(f"✗ Error creating feature importance plot: {e}")
    
    try:
        create_route_heatmap_plot(data)
    except Exception as e:
        print(f"✗ Error creating route heatmap plot: {e}")
    
    print("\n=== SIMPLIFIED FIGURES COMPLETE ===")
    print("All simplified Random Forest figures have been generated!")
    print("Check the output/figures/random_forest_simplified/ directory for results.")

if __name__ == "__main__":
    main()
