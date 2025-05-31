#!/usr/bin/env python3
"""
Random Forest Model Results Showcase
====================================

Generates comprehensive figures showcasing Random Forest model results:
1. Terrain difficulty classification with least cost paths
2. Route comparison: optimal vs actual paths  
3. Data coverage and speed distribution analysis
4. Model performance and predictor importance
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Try to import visualization config, create fallback if not available
try:
    from visualizations.viz_config import *
    from visualizations.figure_manager import FigureManager
except ImportError:
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
    
    # Load model summary
    try:
        summary_path = "output/stats_analysis/random_forest_summary.csv"
        data['model_summary'] = pd.read_csv(summary_path)
        print(f"✓ Loaded model performance summary")
    except Exception as e:
        print(f"✗ Could not load model summary: {e}")
        data['model_summary'] = None
    
    # Load impassable areas
    try:
        impassable_path = "data/derived/map/impassible_areas.shp"
        data['impassable_areas'] = gpd.read_file(impassable_path)
        print(f"✓ Loaded {len(data['impassable_areas'])} impassable areas")
    except Exception as e:
        print(f"✗ Could not load impassable areas: {e}")
        data['impassable_areas'] = None
    
    return data

def create_terrain_difficulty_plot(data):
    """
    Plot 1: Terrain difficulty classification with least cost paths
    """
    print("Creating terrain difficulty classification plot...")
    
    if data['cost_surface'] is None:
        print("✗ Cannot create terrain plot - no cost surface data")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    cost_surface = data['cost_surface']
    cost_transform = data['cost_transform']
    
    # Plot 1: Terrain difficulty classification
    cost_percentiles = np.nanpercentile(cost_surface, [33, 66])
    difficulty_zones = np.zeros_like(cost_surface)
    difficulty_zones[cost_surface <= cost_percentiles[0]] = 1  # Easy
    difficulty_zones[(cost_surface > cost_percentiles[0]) & 
                    (cost_surface <= cost_percentiles[1])] = 2  # Medium
    difficulty_zones[cost_surface > cost_percentiles[1]] = 3  # Hard
    difficulty_zones[np.isnan(cost_surface)] = 0  # No data
    
    colors = ['white', ORIENTEERING_COLORS['fast_terrain'], 
              ORIENTEERING_COLORS['medium_terrain'], ORIENTEERING_COLORS['slow_terrain']]
    
    im1 = ax1.imshow(difficulty_zones, extent=[cost_transform[2], 
                     cost_transform[2] + cost_surface.shape[1] * cost_transform[0],
                     cost_transform[5] + cost_surface.shape[0] * cost_transform[4], 
                     cost_transform[5]], 
                     cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=3)
    
    # Overlay optimal paths
    if data['optimal_paths'] is not None:
        data['optimal_paths'].plot(ax=ax1, color='black', linewidth=4, alpha=0.9, 
                                  label='Optimal Paths', zorder=10)
    
    # Overlay control points
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax1, color='blue', markersize=120, 
                                   edgecolor='white', linewidth=3, zorder=15,
                                   label='Control Points')
    
    # Overlay impassable areas
    if data['impassable_areas'] is not None:
        data['impassable_areas'].plot(ax=ax1, color='red', alpha=0.7, 
                                     edgecolor='darkred', linewidth=2,
                                     label='Impassable Areas')
    
    ax1.set_title('Terrain Difficulty Classification with Optimal Paths', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Easting (m)', fontsize=12)
    ax1.set_ylabel('Northing (m)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Create custom legend for difficulty zones
    easy_patch = mpatches.Patch(color=ORIENTEERING_COLORS['fast_terrain'], label='Easy Terrain (Fast)')
    medium_patch = mpatches.Patch(color=ORIENTEERING_COLORS['medium_terrain'], label='Medium Terrain')
    hard_patch = mpatches.Patch(color=ORIENTEERING_COLORS['slow_terrain'], label='Hard Terrain (Slow)')
    ax1.legend(handles=[easy_patch, medium_patch, hard_patch], loc='upper left')
    
    # Plot 2: Cost distribution
    valid_costs = cost_surface[~np.isnan(cost_surface)]
    valid_costs = valid_costs[valid_costs < np.percentile(valid_costs, 99)]
    
    ax2.hist(valid_costs, bins=50, alpha=0.7, color=COLORS['primary'], edgecolor='black')
    ax2.axvline(cost_percentiles[0], color=ORIENTEERING_COLORS['fast_terrain'], 
               linestyle='--', linewidth=3, label=f'Easy/Medium threshold: {cost_percentiles[0]:.3f}')
    ax2.axvline(cost_percentiles[1], color=ORIENTEERING_COLORS['slow_terrain'], 
               linestyle='--', linewidth=3, label=f'Medium/Hard threshold: {cost_percentiles[1]:.3f}')
    ax2.set_xlabel('Movement Cost (seconds/meter)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Terrain Cost Distribution', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cost gradient analysis
    grad_y, grad_x = np.gradient(cost_surface)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude[np.isnan(cost_surface)] = np.nan
    
    im3 = show(gradient_magnitude, transform=cost_transform, cmap='hot', ax=ax3, alpha=0.8)
    
    if data['optimal_paths'] is not None:
        data['optimal_paths'].plot(ax=ax3, color='cyan', linewidth=3, alpha=0.9)
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax3, color='blue', markersize=80, 
                                   edgecolor='white', linewidth=2, zorder=10)
    
    ax3.set_title('Cost Gradient Magnitude', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Easting (m)', fontsize=12)
    ax3.set_ylabel('Northing (m)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Path analysis
    if data['optimal_paths'] is not None:
        path_lengths = data['optimal_paths']['length_m']
        path_costs = data['optimal_paths']['total_cost']
        
        scatter = ax4.scatter(path_lengths, path_costs, s=100, alpha=0.7, 
                             c=range(len(path_lengths)), cmap='viridis')
        
        # Add trend line
        if len(path_lengths) > 1:
            z = np.polyfit(path_lengths, path_costs, 1)
            p = np.poly1d(z)
            ax4.plot(path_lengths, p(path_lengths), 'r--', linewidth=2, 
                    alpha=0.8, label=f'Trend (R² = {np.corrcoef(path_lengths, path_costs)[0,1]**2:.3f})')
        
        ax4.set_xlabel('Path Length (meters)', fontsize=12)
        ax4.set_ylabel('Total Path Cost', fontsize=12)
        ax4.set_title('Path Length vs Cost Analysis', fontsize=16, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax4, label='Path Segment')
    else:
        ax4.text(0.5, 0.5, 'No optimal path data available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=14)
        ax4.set_title('Path Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_showcase",
        subcategory="terrain_analysis",
        filename="terrain_difficulty_classification",
        description="Terrain difficulty classification with optimal paths",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved terrain difficulty plot to: {paths['png']}")

def create_route_comparison_plot(data):
    """
    Plot 2: Route comparison - optimal vs actual paths
    """
    print("Creating route comparison plot...")
    
    if data['gps_tracks'] is None or data['optimal_paths'] is None:
        print("✗ Cannot create route comparison - missing GPS tracks or optimal paths")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Route overlay on cost surface
    if data['cost_surface'] is not None:
        im1 = show(data['cost_surface'], transform=data['cost_transform'], 
                  cmap=COST_COLORMAP, ax=ax1, alpha=0.6)
        plt.colorbar(im1.get_images()[0], ax=ax1, label='Cost (s/m)', shrink=0.8)
    
    # Plot actual GPS tracks by runner
    if 'runner_id' in data['gps_tracks'].columns:
        unique_runners = data['gps_tracks']['runner_id'].unique()[:10]  # Limit to 10 for visibility
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runners)))
        
        for i, runner_id in enumerate(unique_runners):
            runner_data = data['gps_tracks'][data['gps_tracks']['runner_id'] == runner_id]
            ax1.scatter(runner_data.geometry.x, runner_data.geometry.y, 
                       c=[colors[i]], s=1, alpha=0.6, label=f'Runner {runner_id}')
    
    # Plot optimal paths
    data['optimal_paths'].plot(ax=ax1, color='red', linewidth=4, alpha=0.9, 
                              label='Optimal Path', zorder=10)
    
    # Plot control points
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax1, color='blue', markersize=100, 
                                   edgecolor='white', linewidth=2, zorder=15)
    
    ax1.set_title('Route Comparison: Optimal vs Actual Paths', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Easting (m)', fontsize=12)
    ax1.set_ylabel('Northing (m)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speed distribution comparison
    if 'speed' in data['gps_tracks'].columns:
        speeds = data['gps_tracks']['speed']
        speeds_clean = speeds[(speeds >= 0.5) & (speeds <= 25)]
        
        ax2.hist(speeds_clean, bins=50, alpha=0.7, color=COLORS['primary'], 
                edgecolor='black', label=f'Actual Speeds (n={len(speeds_clean):,})')
        
        # Calculate predicted speeds from optimal paths if possible
        if data['optimal_paths'] is not None and 'length_m' in data['optimal_paths'].columns:
            # Estimate speeds from optimal path segments
            path_speeds = []
            for _, path in data['optimal_paths'].iterrows():
                if path['total_cost'] > 0:
                    speed_ms = path['length_m'] / path['total_cost']
                    speed_kmh = speed_ms * 3.6
                    if 0.5 <= speed_kmh <= 25:
                        path_speeds.append(speed_kmh)
            
            if path_speeds:
                ax2.axvline(np.mean(path_speeds), color='red', linestyle='--', 
                           linewidth=3, label=f'Optimal Avg: {np.mean(path_speeds):.1f} km/h')
        
        ax2.axvline(np.mean(speeds_clean), color=COLORS['primary'], linestyle='--', 
                   linewidth=3, label=f'Actual Avg: {np.mean(speeds_clean):.1f} km/h')
        
        ax2.set_xlabel('Speed (km/h)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Speed Distribution: Actual vs Optimal', fontsize=16, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Path efficiency analysis
    if data['optimal_paths'] is not None:
        # Calculate efficiency metrics
        total_optimal_length = data['optimal_paths']['length_m'].sum()
        total_optimal_cost = data['optimal_paths']['total_cost'].sum()
        
        # Estimate actual path metrics per runner
        if 'runner_id' in data['gps_tracks'].columns:
            runner_stats = []
            for runner_id in data['gps_tracks']['runner_id'].unique():
                runner_data = data['gps_tracks'][data['gps_tracks']['runner_id'] == runner_id]
                if len(runner_data) > 1:
                    # Approximate total distance
                    coords = list(zip(runner_data.geometry.x, runner_data.geometry.y))
                    total_distance = sum([
                        np.sqrt((coords[i+1][0] - coords[i][0])**2 + 
                               (coords[i+1][1] - coords[i][1])**2)
                        for i in range(len(coords)-1)
                    ])
                    
                    # Calculate efficiency (optimal/actual)
                    efficiency = total_optimal_length / total_distance if total_distance > 0 else 0
                    avg_speed = runner_data['speed'].mean() if 'speed' in runner_data.columns else 0
                    
                    runner_stats.append({
                        'runner_id': runner_id,
                        'efficiency': efficiency,
                        'avg_speed': avg_speed,
                        'total_distance': total_distance
                    })
            
            if runner_stats:
                runner_df = pd.DataFrame(runner_stats)
                runner_df = runner_df[runner_df['efficiency'] <= 2]  # Remove outliers
                
                scatter = ax3.scatter(runner_df['efficiency'], runner_df['avg_speed'], 
                                    s=100, alpha=0.7, c=runner_df['total_distance'], 
                                    cmap='viridis')
                plt.colorbar(scatter, ax=ax3, label='Total Distance (m)')
                
                ax3.set_xlabel('Path Efficiency (Optimal/Actual)', fontsize=12)
                ax3.set_ylabel('Average Speed (km/h)', fontsize=12)
                ax3.set_title('Runner Efficiency vs Speed', fontsize=16, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add efficiency reference line
                ax3.axvline(1.0, color='red', linestyle='--', linewidth=2, 
                           label='Perfect Efficiency', alpha=0.7)
                ax3.legend()
    
    # Plot 4: Route deviation analysis
    if data['control_points'] is not None and len(data['control_points']) > 1:
        # Calculate distances between consecutive control points
        controls_sorted = data['control_points'].copy()
        if 'cont_point' in controls_sorted.columns:
            controls_sorted = controls_sorted.sort_values('cont_point')
        
        control_distances = []
        optimal_costs = []
        
        for i in range(len(controls_sorted) - 1):
            p1 = controls_sorted.iloc[i].geometry
            p2 = controls_sorted.iloc[i + 1].geometry
            euclidean_dist = p1.distance(p2)
            control_distances.append(euclidean_dist)
            
            # Find corresponding optimal path segment
            if data['optimal_paths'] is not None and i < len(data['optimal_paths']):
                optimal_costs.append(data['optimal_paths'].iloc[i]['total_cost'])
            else:
                optimal_costs.append(euclidean_dist / 4.0)  # Estimate 4 m/s average
        
        if control_distances and optimal_costs:
            ax4.scatter(control_distances, optimal_costs, s=100, alpha=0.7, 
                       color=COLORS['secondary'])
            
            # Add trend line
            if len(control_distances) > 1:
                z = np.polyfit(control_distances, optimal_costs, 1)
                p = np.poly1d(z)
                ax4.plot(control_distances, p(control_distances), 'r--', 
                        linewidth=2, alpha=0.8)
            
            ax4.set_xlabel('Euclidean Distance (m)', fontsize=12)
            ax4.set_ylabel('Optimal Path Cost (s)', fontsize=12)
            ax4.set_title('Control Point Segments: Distance vs Cost', fontsize=16, fontweight='bold')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_showcase",
        subcategory="route_comparison",
        filename="route_comparison_analysis",
        description="Optimal vs actual route comparison analysis",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved route comparison plot to: {paths['png']}")

def create_data_coverage_plot(data):
    """
    Plot 3: Data coverage and speed distribution analysis
    """
    print("Creating data coverage and speed distribution plot...")
    
    if data['gps_tracks'] is None:
        print("✗ Cannot create data coverage plot - no GPS data")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    gps_data = data['gps_tracks']
    
    # Plot 1: GPS data coverage density
    if data['cost_surface'] is not None:
        # Show cost surface as background
        im1 = show(data['cost_surface'], transform=data['cost_transform'], 
                  cmap='gray', ax=ax1, alpha=0.3)
    
    # Create density plot of GPS points
    x_coords = gps_data.geometry.x
    y_coords = gps_data.geometry.y
    
    # Create 2D histogram for density
    hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im_density = ax1.imshow(hist.T, extent=extent, origin='lower', 
                           cmap='Reds', alpha=0.7, aspect='auto')
    plt.colorbar(im_density, ax=ax1, label='GPS Point Density')
    
    # Overlay control points
    if data['control_points'] is not None:
        data['control_points'].plot(ax=ax1, color='blue', markersize=100, 
                                   edgecolor='white', linewidth=2, zorder=10,
                                   label='Control Points')
    
    ax1.set_title(f'GPS Data Coverage Density (n={len(gps_data):,} points)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Easting (m)', fontsize=12)
    ax1.set_ylabel('Northing (m)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speed distribution by runner
    if 'speed' in gps_data.columns and 'runner_id' in gps_data.columns:
        # Filter reasonable speeds
        speed_data = gps_data[(gps_data['speed'] >= 0.5) & (gps_data['speed'] <= 25)].copy()
        
        # Box plot by runner (show top 10 runners)
        top_runners = speed_data['runner_id'].value_counts().head(10).index
        runner_speeds = [speed_data[speed_data['runner_id'] == rid]['speed'].values 
                        for rid in top_runners]
        
        box_plot = ax2.boxplot(runner_speeds, labels=[f'R{rid}' for rid in top_runners], 
                              patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Runner ID', fontsize=12)
        ax2.set_ylabel('Speed (km/h)', fontsize=12)
        ax2.set_title('Speed Distribution by Runner (Top 10)', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Speed vs terrain characteristics
    if 'speed' in gps_data.columns:
        # Try to get terrain features if available
        terrain_features = ['elevation', 'slope', 'ndvi']
        available_features = [f for f in terrain_features if f in gps_data.columns]
        
        if available_features:
            feature = available_features[0]  # Use first available feature
            speeds = gps_data['speed']
            feature_values = gps_data[feature]
            
            # Filter data
            valid_mask = ((speeds >= 0.5) & (speeds <= 25) & 
                         (feature_values.notna()) & (np.isfinite(feature_values)))
            speeds_clean = speeds[valid_mask]
            features_clean = feature_values[valid_mask]
            
            # Create scatter plot with density coloring
            scatter = ax3.scatter(features_clean, speeds_clean, alpha=0.5, 
                                s=1, c=speeds_clean, cmap='viridis')
            plt.colorbar(scatter, ax=ax3, label='Speed (km/h)')
            
            # Add trend line
            if len(speeds_clean) > 100:
                z = np.polyfit(features_clean, speeds_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(features_clean.min(), features_clean.max(), 100)
                ax3.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8,
                        label=f'Trend (R²={np.corrcoef(features_clean, speeds_clean)[0,1]**2:.3f})')
                ax3.legend()
            
            ax3.set_xlabel(f'{feature.capitalize()}', fontsize=12)
            ax3.set_ylabel('Speed (km/h)', fontsize=12)
            ax3.set_title(f'Speed vs {feature.capitalize()}', fontsize=16, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            # Fallback: simple speed histogram
            speeds = gps_data['speed']
            speeds_clean = speeds[(speeds >= 0.5) & (speeds <= 25)]
            
            ax3.hist(speeds_clean, bins=50, alpha=0.7, color=COLORS['primary'], 
                    edgecolor='black')
            ax3.axvline(speeds_clean.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {speeds_clean.mean():.1f} km/h')
            ax3.axvline(speeds_clean.median(), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {speeds_clean.median():.1f} km/h')
            ax3.set_xlabel('Speed (km/h)', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Overall Speed Distribution', fontsize=16, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temporal analysis or track statistics
    if 'runner_id' in gps_data.columns:
        # Calculate statistics per track
        track_stats = []
        for runner_id in gps_data['runner_id'].unique():
            runner_data = gps_data[gps_data['runner_id'] == runner_id]
            
            stats = {
                'runner_id': runner_id,
                'n_points': len(runner_data),
                'avg_speed': runner_data['speed'].mean() if 'speed' in runner_data.columns else 0,
                'speed_std': runner_data['speed'].std() if 'speed' in runner_data.columns else 0,
                'coverage_area': 0  # Could calculate convex hull area
            }
            
            # Calculate approximate coverage area (convex hull)
            if len(runner_data) >= 3:
                try:
                    from shapely.geometry import MultiPoint
                    points = MultiPoint(list(zip(runner_data.geometry.x, runner_data.geometry.y)))
                    stats['coverage_area'] = points.convex_hull.area / 1000000  # km²
                except:
                    stats['coverage_area'] = 0
            
            track_stats.append(stats)
        
        track_df = pd.DataFrame(track_stats)
        
        # Plot number of points vs average speed
        scatter = ax4.scatter(track_df['n_points'], track_df['avg_speed'], 
                            s=100, alpha=0.7, c=track_df['coverage_area'], 
                            cmap='plasma')
        plt.colorbar(scatter, ax=ax4, label='Coverage Area (km²)')
        
        ax4.set_xlabel('Number of GPS Points', fontsize=12)
        ax4.set_ylabel('Average Speed (km/h)', fontsize=12)
        ax4.set_title('Track Quality: Points vs Speed', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add summary statistics
        ax4.text(0.02, 0.98, f'Total Tracks: {len(track_df)}\n'
                           f'Total Points: {track_df["n_points"].sum():,}\n'
                           f'Avg Points/Track: {track_df["n_points"].mean():.0f}',
                transform=ax4.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_showcase",
        subcategory="data_coverage",
        filename="data_coverage_speed_analysis",
        description="GPS data coverage and speed distribution analysis",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved data coverage plot to: {paths['png']}")

def create_model_performance_plot(data):
    """
    Plot 4: Model performance and predictor importance
    """
    print("Creating model performance and predictor importance plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Feature importance
    if data['feature_importance'] is not None:
        importance_df = data['feature_importance'].copy()
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Take top 15 features for readability
        top_features = importance_df.tail(15)
        
        bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                       color=COLORS['primary'], alpha=0.7)
        
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['predictor'], fontsize=10)  # Changed from 'feature' to 'predictor'
        ax1.set_xlabel('Importance Score', fontsize=12)
        ax1.set_title('Random Forest Feature Importance (Top 15)', 
                     fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add importance values on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', va='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Feature importance data not available', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=14)
        ax1.set_title('Feature Importance', fontsize=16, fontweight='bold')
    
    # Plot 2: Model performance metrics
    if data['model_summary'] is not None:
        summary = data['model_summary'].iloc[0]
        
        # Create performance metrics
        metrics = {
            'Cross-validation R²': summary.get('cv_r2_mean', 0),
            'Test R²': summary.get('test_r2', 0),
            'Cross-validation RMSE': summary.get('cv_rmse_mean', 0),
            'Test RMSE': summary.get('test_rmse', 0),
            'Test MAE': summary.get('test_mae', 0),
            'OOB Score': summary.get('oob_score_final', 0)
        }
        
        # Split into R² metrics and error metrics
        r2_metrics = {k: v for k, v in metrics.items() if 'R²' in k or 'OOB' in k}
        error_metrics = {k: v for k, v in metrics.items() if 'RMSE' in k or 'MAE' in k}
        
        # Plot R² metrics
        r2_names = list(r2_metrics.keys())
        r2_values = list(r2_metrics.values())
        
        bars_r2 = ax2.bar(range(len(r2_names)), r2_values, 
                         color=COLORS['secondary'], alpha=0.7)
        
        ax2.set_xticks(range(len(r2_names)))
        ax2.set_xticklabels(r2_names, rotation=45, ha='right')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('Model Performance: R² Metrics', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars_r2, r2_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line at 0.6 (good performance threshold)
        ax2.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, 
                   label='Good Performance (0.6)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Model performance data not available', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('Model Performance', fontsize=16, fontweight='bold')
    
    # Plot 3: Error metrics and uncertainty
    if data['model_summary'] is not None:
        summary = data['model_summary'].iloc[0]
        
        # Create error comparison
        error_data = {
            'Cross-validation': {
                'RMSE': summary.get('cv_rmse_mean', 0),
                'std': summary.get('cv_rmse_std', 0) if 'cv_rmse_std' in summary else 0
            },
            'Test Set': {
                'RMSE': summary.get('test_rmse', 0),
                'MAE': summary.get('test_mae', 0)
            }
        }
        
        # Plot cross-validation RMSE with error bars
        cv_rmse = error_data['Cross-validation']['RMSE']
        cv_std = error_data['Cross-validation']['std']
        test_rmse = error_data['Test Set']['RMSE']
        test_mae = error_data['Test Set']['MAE']
        
        x_pos = [0, 1, 2]
        values = [cv_rmse, test_rmse, test_mae]
        labels = ['CV RMSE', 'Test RMSE', 'Test MAE']
        errors = [cv_std, 0, 0]
        
        bars = ax3.bar(x_pos, values, yerr=errors, capsize=5, 
                      color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], 
                      alpha=0.7)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Error (log speed units)', fontsize=12)
        ax3.set_title('Model Error Metrics', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
            y_pos = bar.get_height() + error + 0.01
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Model complexity and training details
    if data['model_summary'] is not None:
        summary = data['model_summary'].iloc[0]
        
        # Create training summary
        training_info = {
            'Number of Estimators': summary.get('n_estimators', 0),
            'Max Depth': summary.get('max_depth', 0),
            'Training Time (s)': summary.get('cv_time_seconds', 0),
            'Total Time (s)': summary.get('total_time_seconds', 0),
            'CPU Cores Used': summary.get('cores_used', 0),
            'Features Used': len(data['feature_importance']) if data['feature_importance'] is not None else 0
        }
        
        # Create text summary
        text_summary = []
        for key, value in training_info.items():
            if 'Time' in key:
                text_summary.append(f'{key}: {value:.1f}')
            else:
                text_summary.append(f'{key}: {value}')
        
        ax4.text(0.1, 0.9, '\n'.join(text_summary), transform=ax4.transAxes, 
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        # Add performance summary
        if all(k in summary for k in ['cv_r2_mean', 'test_r2', 'cv_rmse_mean']):
            perf_summary = [
                f"Cross-validation R²: {summary['cv_r2_mean']:.3f} ± {summary.get('cv_r2_std', 0):.3f}",
                f"Test R²: {summary['test_r2']:.3f}",
                f"Cross-validation RMSE: {summary['cv_rmse_mean']:.3f}",
                f"Model Status: {'Excellent' if summary['test_r2'] > 0.6 else 'Good' if summary['test_r2'] > 0.4 else 'Moderate'}"
            ]
            
            ax4.text(0.1, 0.4, '\n'.join(perf_summary), transform=ax4.transAxes, 
                    fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Training Summary', fontsize=16, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Model training data not available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=14)
        ax4.set_title('Training Summary', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    paths = fig_manager.register_figure(
        category="random_forest_showcase",
        subcategory="model_performance",
        filename="model_performance_analysis",
        description="Random Forest model performance and feature importance",
        figure_type="working",
        formats=["png", "pdf"]
    )
    
    plt.savefig(paths['png'], dpi=300, bbox_inches='tight')
    plt.savefig(paths['pdf'], dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved model performance plot to: {paths['png']}")

def main():
    """Generate all Random Forest showcase figures."""
    print("=== RANDOM FOREST MODEL SHOWCASE ===")
    print("Generating comprehensive visualization of Random Forest results...")
    
    # Load all data
    data = load_random_forest_data()
    
    # Generate all plots
    print("\nGenerating showcase plots...")
    
    try:
        create_terrain_difficulty_plot(data)
    except Exception as e:
        print(f"✗ Error creating terrain difficulty plot: {e}")
    
    try:
        create_route_comparison_plot(data)
    except Exception as e:
        print(f"✗ Error creating route comparison plot: {e}")
    
    try:
        create_data_coverage_plot(data)
    except Exception as e:
        print(f"✗ Error creating data coverage plot: {e}")
    
    try:
        create_model_performance_plot(data)
    except Exception as e:
        print(f"✗ Error creating model performance plot: {e}")
    
    print("\n=== SHOWCASE GENERATION COMPLETE ===")
    print("All Random Forest showcase figures have been generated!")
    print("Check the output/figures/random_forest_showcase/ directory for results.")

if __name__ == "__main__":
    main()
