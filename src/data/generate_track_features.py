#!/usr/bin/env python3
"""
Script to generate track_features.csv from GPS data files.
Converts LineString GPS tracks to individual points with speed calculations.
"""

import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import re
from scipy.spatial.distance import cdist

def extract_runner_name(filename):
    """Extract runner name from GPS filename."""
    # Pattern: "..., H21 Elit, [Runner Name]_track_3006.gpkg"
    match = re.search(r'H21 Elit, ([^_]+)_track_\d+\.gpkg$', filename)
    if match:
        return match.group(1).strip()
    else:
        # Fallback: use the part before _track_
        basename = os.path.basename(filename)
        return basename.split('_track_')[0].split(', ')[-1]

def load_control_points():
    """Load all control points for proper race section detection."""
    try:
        controls_gdf = gpd.read_file("data/derived/map/control_points_race_3006.shp")
        print(f"Loaded {len(controls_gdf)} control points")
        return controls_gdf
        
    except Exception as e:
        print(f"Error loading control points: {e}")
        return None

def assign_control_id(pt, controls, buffer=10.0):
    """Assign control ID if point is within buffer distance of a control."""
    for idx, row in controls.iterrows():
        if pt.distance(row.geometry) <= buffer:
            # Return a unique identifier for this control
            if row.get('start') == 1.0:
                return 'start'
            elif row.get('stop') == 1.0:
                return 'finish'
            elif pd.notna(row.get('cont_point')):
                return f"control_{int(row['cont_point']):02d}"
    return None

def trim_track_to_race_section(coords, controls):
    """
    Trim GPS track to only include points between first and last control hit.
    This follows the same logic as the original comprehensive script.
    
    Args:
        coords: List of (x, y) coordinate tuples
        controls: GeoDataFrame of control points
    
    Returns:
        Trimmed list of coordinates with control assignments
    """
    if len(coords) < 2 or controls is None:
        return coords, [None] * len(coords)
    
    # Create points and assign control IDs
    points = [Point(x, y) for x, y in coords]
    control_ids = [assign_control_id(pt, controls) for pt in points]
    
    # Find first and last control hit
    control_indices = [i for i, cid in enumerate(control_ids) if cid is not None]
    
    if not control_indices:
        print("Warning: No control points hit in track")
        return coords, control_ids
    
    first_control_idx = min(control_indices)
    last_control_idx = max(control_indices)
    
    # Trim to race section
    trimmed_coords = coords[first_control_idx:last_control_idx + 1]
    trimmed_control_ids = control_ids[first_control_idx:last_control_idx + 1]
    
    # Forward-fill control IDs (maintain continuity between controls)
    last_cid = None
    for i, cid in enumerate(trimmed_control_ids):
        if cid is not None:
            last_cid = cid
        else:
            trimmed_control_ids[i] = last_cid
    
    print(f"  Trimmed track: {len(coords)} -> {len(trimmed_coords)} points "
          f"(from control at idx {first_control_idx} to control at idx {last_control_idx})")
    
    return trimmed_coords, trimmed_control_ids

def calculate_speed_from_coords(coords, time_interval=1.0):
    """
    Calculate speeds between consecutive coordinate points.
    
    Args:
        coords: List of (x, y) coordinate tuples
        time_interval: Assumed time interval between points in seconds
    
    Returns:
        List of speeds in km/h
    """
    speeds = []
    
    for i in range(1, len(coords)):
        # Calculate distance between consecutive points (in meters)
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Convert to speed in km/h
        # distance (m) / time_interval (s) * 3600 (s/h) / 1000 (m/km)
        speed_kmh = (distance / time_interval) * 3.6
        speeds.append(speed_kmh)
    
    return speeds

def process_gps_file(file_path, controls):
    """
    Process a single GPS file and return DataFrame with points and speeds.
    
    Args:
        file_path: Path to the GPS file
        controls: GeoDataFrame of control points for trimming
        
    Returns:
        DataFrame with columns: geometry (WKT), runner_id, speed
    """
    try:
        # Extract runner name from filename
        runner_name = extract_runner_name(file_path)
        
        # Read GPS file
        gdf = gpd.read_file(file_path)
        
        if gdf.empty or gdf.geometry.iloc[0] is None:
            print(f"Warning: Empty or invalid geometry in {file_path}")
            return pd.DataFrame()
        
        # Get the LineString geometry
        linestring = gdf.geometry.iloc[0]
        
        # Extract coordinates
        coords = list(linestring.coords)
        
        if len(coords) < 2:
            print(f"Warning: Insufficient points in {file_path} (only {len(coords)} points)")
            return pd.DataFrame()
        
        # Trim track to race section (between first and last control hits)
        trimmed_coords, control_ids = trim_track_to_race_section(coords, controls)
        
        if len(trimmed_coords) < 2:
            print(f"Warning: Insufficient points after trimming in {file_path} (only {len(trimmed_coords)} points)")
            return pd.DataFrame()
        
        # Calculate speeds between consecutive points
        speeds = calculate_speed_from_coords(trimmed_coords)
        
        # Create points from coordinates (skip first point since it has no speed)
        points = [Point(x, y) for x, y in trimmed_coords[1:]]
        
        # Create DataFrame
        data = {
            'geometry': [point.wkt for point in points],
            'runner_id': [runner_name] * len(points),
            'speed': speeds
        }
        
        df = pd.DataFrame(data)
        
        print(f"Processed {runner_name}: {len(df)} points, speed range: {df['speed'].min():.2f}-{df['speed'].max():.2f} km/h")
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to process all GPS files and generate track_features.csv"""
    
    # Set up paths
    gps_dir = "data/derived/gps"
    output_file = "data/derived/csv/track_features.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("data/derived/csv", exist_ok=True)
    
    # Load control points
    print("Loading control points...")
    controls = load_control_points()
    
    if controls is None:
        print("Failed to load control points. Exiting.")
        return
    
    # Find all GPS files
    gps_files = glob.glob(os.path.join(gps_dir, "*.gpkg"))
    
    if not gps_files:
        print(f"No GPS files found in {gps_dir}")
        return
    
    print(f"Found {len(gps_files)} GPS files to process")
    
    # Process all files
    all_dataframes = []
    
    for i, file_path in enumerate(gps_files):
        print(f"Processing file {i+1}/{len(gps_files)}: {os.path.basename(file_path)}")
        df = process_gps_file(file_path, controls)
        if not df.empty:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("No valid data extracted from GPS files")
        return
    
    # Combine all dataframes
    print("\nCombining all track data...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Apply speed filtering as expected by the model (0.5-25.0 km/h)
    print(f"Total points before speed filtering: {len(combined_df)}")
    
    # Filter speeds
    filtered_df = combined_df[
        (combined_df["speed"] >= 0.5) & 
        (combined_df["speed"] <= 25.0)
    ].copy()
    
    print(f"Total points after speed filtering: {len(filtered_df)}")
    print(f"Speed range in filtered data: {filtered_df['speed'].min():.2f}-{filtered_df['speed'].max():.2f} km/h")
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Successfully generated {output_file}")
    print(f"Final dataset: {len(filtered_df)} points from {filtered_df['runner_id'].nunique()} runners")
    print("All tracks have been trimmed to only include the race section between start and finish controls.")
    
    # Display sample of the data
    print("\nSample of generated data:")
    print(filtered_df.head())
    
    # Display runner statistics
    print("\nRunner statistics:")
    runner_stats = filtered_df.groupby('runner_id').agg({
        'speed': ['count', 'mean', 'std'],
        'geometry': 'count'
    }).round(2)
    runner_stats.columns = ['point_count', 'avg_speed', 'speed_std', 'geometry_count']
    print(runner_stats.head(10))

if __name__ == "__main__":
    main()
