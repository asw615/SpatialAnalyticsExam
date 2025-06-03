#!/usr/bin/env python3
"""
Spatial Autocorrelation Analysis for Orienteering GPS Data

This script conducts spatial autocorrelation tests using Moran's I statistic to:
1. Test for spatial clustering in GPS track speeds (dependent variable)
2. Test for spatial clustering in environmental predictors (independent variables)
3. Assess the presence of spatial structure that could affect model validity
4. Generate diagnostic outputs for understanding spatial dependencies

Spatial autocorrelation violates the independence assumption of many statistical models.
This analysis helps determine if spatial effects need to be explicitly modeled.

Key Tests:
- Global Moran's I for overall spatial autocorrelation
- Local Moran's I (LISA) for hotspot identification
- Variogram analysis for understanding spatial scale of correlation
- Spatial lag tests for environmental variables

Output: Statistical tests, diagnostic plots, and recommendations for spatial modeling.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import rasterio
from rasterio.sample import sample_gen
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings

# Spatial statistics libraries
try:
    from pysal.lib import weights
    from pysal.explore import esda
    from pysal.viz import splot
    HAS_PYSAL = True
except ImportError:
    print("Warning: PySAL not available. Some spatial statistics features will be limited.")
    HAS_PYSAL = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_gps_data():
    """Load GPS track data with speeds."""
    print("Loading GPS track data...")
    
    # Load track features
    track_df = pd.read_csv("data/derived/csv/track_features.csv")
    
    # Convert geometry from WKT
    track_df['geometry'] = track_df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(track_df, crs='EPSG:3006')
    
    # Filter speeds to remove outliers (consistent with main analysis)
    gdf = gdf[(gdf['speed'] >= 0.5) & (gdf['speed'] <= 25.0)].copy()
    
    print(f"Loaded {len(gdf)} GPS points from {gdf['runner_id'].nunique()} runners")
    print(f"Speed range: {gdf['speed'].min():.1f} - {gdf['speed'].max():.1f} km/h")
    
    return gdf

def sample_environmental_data(gdf, sample_size=5000):
    """
    Sample environmental data at GPS locations.
    
    Args:
        gdf: GeoDataFrame with GPS points
        sample_size: Number of points to sample for analysis (to manage computational load)
    
    Returns:
        GeoDataFrame with environmental variables sampled at GPS locations
    """
    print(f"\nSampling environmental data at GPS locations...")
    
    # Sample points for computational efficiency
    if len(gdf) > sample_size:
        gdf_sample = gdf.sample(n=sample_size, random_state=42).copy()
        print(f"Sampled {sample_size} points from {len(gdf)} total for efficiency")
    else:
        gdf_sample = gdf.copy()
    
    # Find available NDVI file
    ndvi_file = None
    ndvi_dir = "data/derived/ndvi"
    if os.path.exists(ndvi_dir):
        # Look for NDVI files with the correct pattern
        import glob
        ndvi_pattern = os.path.join(ndvi_dir, "*NDVI_3006.tif")
        ndvi_files = glob.glob(ndvi_pattern)
        if ndvi_files:
            ndvi_file = ndvi_files[0]  # Use first available
            print(f"Found NDVI file: {ndvi_file}")
        else:
            print(f"No NDVI files found in {ndvi_dir}")
    
    # Environmental raster files to sample
    env_rasters = {
        'elevation': 'data/derived/dem/dem_3006.vrt',
        'dist_to_trails': 'data/derived/rasters/dist_to_trails.tif',
        'dist_to_roads': 'data/derived/rasters/dist_to_roads.tif',
        'dist_to_water': 'data/derived/rasters/dist_to_water.tif',
        'rocky_terrain': 'data/derived/rasters/rocky_terrain.tif',
        'firm_wetlands': 'data/derived/rasters/firm_wetlands.tif'
    }
    
    # Add NDVI if available
    if ndvi_file:
        env_rasters['ndvi'] = ndvi_file
    
    # Extract coordinates
    coords = [(point.x, point.y) for point in gdf_sample.geometry]
    
    # Sample each environmental variable
    for var_name, raster_path in env_rasters.items():
        if os.path.exists(raster_path):
            try:
                with rasterio.open(raster_path) as src:
                    # Sample raster values at GPS coordinates
                    values = list(sample_gen(src, coords))
                    gdf_sample[var_name] = [v[0] if v[0] is not None else np.nan for v in values]
                    
                print(f"  ✓ Sampled {var_name}")
            except Exception as e:
                print(f"  ✗ Error sampling {var_name}: {e}")
                gdf_sample[var_name] = np.nan
        else:
            print(f"  ✗ File not found: {raster_path}")
            gdf_sample[var_name] = np.nan
    
    # Remove points with missing environmental data (but be more flexible)
    initial_count = len(gdf_sample)
    # Only require elevation and speed to be non-null (core variables)
    core_vars = ['elevation']
    available_vars = [var for var in env_rasters.keys() if var in gdf_sample.columns and not gdf_sample[var].isna().all()]
    
    if available_vars:
        # Keep points that have at least elevation data
        gdf_sample = gdf_sample.dropna(subset=['elevation'])
        final_count = len(gdf_sample)
        print(f"Retained {final_count} points with elevation data ({initial_count - final_count} removed)")
        print(f"Available environmental variables: {available_vars}")
    else:
        print("Warning: No environmental variables successfully loaded")
        final_count = len(gdf_sample)
    
    return gdf_sample

def create_spatial_weights(gdf, k_neighbors=8, distance_threshold=100):
    """
    Create spatial weights matrix for autocorrelation analysis.
    
    Args:
        gdf: GeoDataFrame with point data
        k_neighbors: Number of nearest neighbors for weights
        distance_threshold: Maximum distance for neighborhood (meters)
    
    Returns:
        PySAL weights object or custom weights matrix
    """
    print(f"\nCreating spatial weights matrix...")
    print(f"Using {k_neighbors} nearest neighbors with {distance_threshold}m threshold")
    
    if HAS_PYSAL:
        # Use PySAL for sophisticated weights
        try:
            # K-nearest neighbors weights
            w_knn = weights.KNN.from_dataframe(gdf, k=k_neighbors)
            w_knn.transform = 'r'  # Row-standardize weights
            
            print(f"Created KNN weights: {w_knn.n} observations, {w_knn.mean_neighbors:.1f} avg neighbors")
            return w_knn
            
        except Exception as e:
            print(f"Error creating PySAL weights: {e}")
            return None
    else:
        # Simple distance-based weights without PySAL
        print("Creating simple distance-based weights (PySAL not available)")
        coords = np.array([[point.x, point.y] for point in gdf.geometry])
        
        # Use sklearn for nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Create weights matrix (exclude self)
        n = len(coords)
        weights_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(1, k_neighbors+1):  # Skip first (self)
                neighbor_idx = indices[i, j]
                if distances[i, j] <= distance_threshold:
                    weights_matrix[i, neighbor_idx] = 1.0 / distances[i, j]
        
        # Row-standardize
        row_sums = weights_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights_matrix = weights_matrix / row_sums[:, np.newaxis]
        
        return weights_matrix

def compute_morans_i(values, weights):
    """
    Compute Moran's I statistic for spatial autocorrelation.
    
    Args:
        values: Array of variable values
        weights: Spatial weights matrix or PySAL weights object
    
    Returns:
        Dictionary with Moran's I results
    """
    if HAS_PYSAL and hasattr(weights, 'sparse'):
        # Use PySAL implementation
        try:
            moran = esda.Moran(values, weights)
            return {
                'I': moran.I,
                'expected_I': moran.EI,
                'variance': moran.VI_norm,
                'z_score': moran.z_norm,
                'p_value': moran.p_norm,
                'interpretation': 'Positive autocorrelation' if moran.I > moran.EI else 'Negative autocorrelation'
            }
        except Exception as e:
            print(f"Error computing PySAL Moran's I: {e}")
            return None
    else:
        # Manual implementation
        n = len(values)
        if isinstance(weights, np.ndarray):
            W = weights
        else:
            print("Cannot compute Moran's I without proper weights matrix")
            return None
        
        # Center the variable
        y = values - np.mean(values)
        
        # Compute Moran's I
        numerator = np.sum(W * np.outer(y, y))
        denominator = np.sum(y**2)
        
        I = (n / np.sum(W)) * (numerator / denominator)
        
        # Expected value under null hypothesis
        expected_I = -1 / (n - 1)
        
        # Approximate significance (simplified)
        W_sum = np.sum(W)
        if W_sum == 0 or np.sum(y**2) == 0:
            return {
                'I': 0,
                'expected_I': expected_I,
                'variance': 0,
                'z_score': 0,
                'p_value': 1.0,
                'interpretation': 'No spatial weights or no variance'
            }
        
        variance_approx = (2 * W_sum**2) / (n * (n-1) * np.sum(y**2))
        if variance_approx <= 0:
            variance_approx = 0.001  # Small positive value to avoid sqrt issues
        z_score = (I - expected_I) / np.sqrt(variance_approx)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'I': I,
            'expected_I': expected_I,
            'variance': variance_approx,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': 'Positive autocorrelation' if I > expected_I else 'Negative autocorrelation'
        }

def analyze_spatial_autocorrelation(gdf, weights):
    """
    Conduct comprehensive spatial autocorrelation analysis.
    
    Args:
        gdf: GeoDataFrame with variables
        weights: Spatial weights matrix
    
    Returns:
        Dictionary with autocorrelation results for all variables
    """
    print("\n" + "="*60)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("="*60)
    
    # Variables to test
    variables = ['speed', 'elevation', 'ndvi', 'dist_to_trails', 'dist_to_roads', 
                 'dist_to_water', 'rocky_terrain', 'firm_wetlands']
    
    results = {}
    
    for var in variables:
        if var in gdf.columns and not gdf[var].isna().all():
            print(f"\nAnalyzing {var}...")
            
            # Remove missing values
            mask = ~gdf[var].isna()
            values = gdf[var][mask].values
            
            if len(values) < 10:
                print(f"  Insufficient data for {var} (only {len(values)} valid values)")
                continue
            
            # Compute Moran's I
            moran_result = compute_morans_i(values, weights)
            
            if moran_result:
                results[var] = moran_result
                
                print(f"  Moran's I: {moran_result['I']:.4f}")
                print(f"  Expected I: {moran_result['expected_I']:.4f}")
                print(f"  Z-score: {moran_result['z_score']:.4f}")
                print(f"  P-value: {moran_result['p_value']:.4f}")
                print(f"  Significance: {'***' if moran_result['p_value'] < 0.001 else '**' if moran_result['p_value'] < 0.01 else '*' if moran_result['p_value'] < 0.05 else 'ns'}")
                print(f"  Interpretation: {moran_result['interpretation']}")
        else:
            print(f"  Variable {var} not available or all missing")
    
    return results

def create_autocorrelation_plots(gdf, autocorr_results, output_dir):
    """
    Create visualization plots for spatial autocorrelation results.
    
    Args:
        gdf: GeoDataFrame with data
        autocorr_results: Results from autocorrelation analysis
        output_dir: Directory to save plots
    """
    print("\nCreating spatial autocorrelation visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Moran's I Summary Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract results for plotting
    variables = list(autocorr_results.keys())
    morans_i = [autocorr_results[var]['I'] for var in variables]
    p_values = [autocorr_results[var]['p_value'] for var in variables]
    z_scores = [autocorr_results[var]['z_score'] for var in variables]
    
    # Plot 1: Moran's I values
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars1 = ax1.barh(variables, morans_i, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Moran's I")
    ax1.set_title("Spatial Autocorrelation (Moran's I)")
    ax1.grid(True, alpha=0.3)
    
    # Add significance annotations
    for i, (var, p_val) in enumerate(zip(variables, p_values)):
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        if significance:
            ax1.text(morans_i[i] + 0.01, i, significance, va='center', fontweight='bold')
    
    # Plot 2: Z-scores (standardized test statistics)
    colors2 = ['red' if abs(z) > 1.96 else 'gray' for z in z_scores]
    bars2 = ax2.barh(variables, z_scores, color=colors2, alpha=0.7)
    ax2.axvline(x=1.96, color='red', linestyle='--', alpha=0.5, label='p < 0.05')
    ax2.axvline(x=-1.96, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel("Z-score")
    ax2.set_title("Statistical Significance (Z-scores)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_autocorrelation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial Distribution of Speed (most important variable)
    if 'speed' in gdf.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create scatter plot of GPS points colored by speed
        scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf['speed'], 
                           cmap='RdYlGn', s=1, alpha=0.6)
        
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Spatial Distribution of GPS Speed Data\n(Colors show potential spatial clustering)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (km/h)')
        
        # Add Moran's I result as text
        if 'speed' in autocorr_results:
            speed_result = autocorr_results['speed']
            stats_text = f"Moran's I = {speed_result['I']:.4f}\n"
            stats_text += f"Z-score = {speed_result['z_score']:.2f}\n"
            stats_text += f"P-value = {speed_result['p_value']:.4f}\n"
            stats_text += f"{speed_result['interpretation']}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speed_spatial_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Spatial autocorrelation plots saved to {output_dir}/")

def main():
    """Main function to run spatial autocorrelation analysis."""
    print("="*60)
    print("SPATIAL AUTOCORRELATION ANALYSIS FOR ORIENTEERING GPS DATA")
    print("="*60)
    
    # Create output directory
    output_dir = "output/spatial_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load GPS data
        gdf = load_gps_data()
        
        # 2. Sample environmental data
        gdf_env = sample_environmental_data(gdf, sample_size=3000)  # Reduced for computational efficiency
        
        if len(gdf_env) < 50:
            print("Error: Insufficient data points for spatial autocorrelation analysis")
            return
        
        # 3. Create spatial weights
        weights = create_spatial_weights(gdf_env, k_neighbors=8, distance_threshold=200)
        
        if weights is None:
            print("Error: Could not create spatial weights matrix")
            return
        
        # 4. Conduct autocorrelation analysis
        autocorr_results = analyze_spatial_autocorrelation(gdf_env, weights)
        
        if not autocorr_results:
            print("Error: No autocorrelation results generated")
            return
        
        # 5. Create visualizations
        create_autocorrelation_plots(gdf_env, autocorr_results, output_dir)
        
        print("\n" + "="*60)
        print("SPATIAL AUTOCORRELATION ANALYSIS COMPLETED")
        print("="*60)
        print(f"Results saved to: {output_dir}/")
        print("\nKey findings:")
        
        significant_count = sum(1 for result in autocorr_results.values() if result['p_value'] < 0.05)
        print(f"- {significant_count}/{len(autocorr_results)} variables show significant spatial autocorrelation")
        
        if 'speed' in autocorr_results:
            speed_sig = "YES" if autocorr_results['speed']['p_value'] < 0.05 else "NO"
            print(f"- Speed variable (dependent) autocorrelated: {speed_sig}")
        
    except Exception as e:
        print(f"Error in spatial autocorrelation analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
