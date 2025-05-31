#!/usr/bin/env python3
"""
Least-cost path analysis using skimage or gdal_calc.
"""

import rasterio
import numpy as np
import pandas as pd
from skimage.graph import route_through_array
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, Point

# --- CONFIG ---
COST_RASTER = Path("output/cost_surfaces/random_forest_cost_surface.tif")
PATHS_OUTPUT = Path("output/cost_surfaces/least_cost_paths_rf.geojson")

def load_control_points():
    """Load control points for path analysis."""
    # Load actual orienteering control points from the race data
    try:
        print("Loading actual orienteering control points...")
        controls_gdf = gpd.read_file("data/derived/map/control_points_race_3006.shp")
        
        print(f"Loaded {len(controls_gdf)} actual control points from orienteering race")
        print(f"Control points extent: {controls_gdf.total_bounds}")
        
        # Create the proper orienteering course sequence
        course_sequence = []
        
        # 1. Add start point
        start_point = controls_gdf[controls_gdf['start'] == 1.0].iloc[0:1].copy()
        start_point['course_id'] = 'start'
        course_sequence.append(start_point)
        
        # 2. Add all control points in numerical order
        control_points = controls_gdf[controls_gdf['cont_point'].notna()].copy()
        control_points = control_points.sort_values('cont_point')
        control_points['course_id'] = ['control_' + str(int(x)).zfill(2) for x in control_points['cont_point']]
        course_sequence.append(control_points)
        
        # 3. Add finish point
        finish_point = controls_gdf[controls_gdf['stop'] == 1.0].iloc[0:1].copy()
        finish_point['course_id'] = 'finish'
        course_sequence.append(finish_point)
        
        # Combine all into proper sequence
        course_gdf = pd.concat(course_sequence, ignore_index=True)
        course_gdf['id'] = course_gdf['course_id']  # Use course_id as the main identifier
        
        print(f"Using full orienteering course: {len(course_gdf)} controls (start + {len(control_points)} controls + finish)")
        print("Course sequence:", course_gdf['id'].tolist())
        
        return course_gdf
        
    except FileNotFoundError:
        print("Actual control points not found, creating sample points...")
        # Fallback to sample points if actual controls not available
        sample_points = [
            {"id": "start", "x": 586500, "y": 6348500},
            {"id": "control1", "x": 587500, "y": 6349500},
            {"id": "control2", "x": 588500, "y": 6350500},
            {"id": "finish", "x": 588000, "y": 6350000}
        ]
        controls_gdf = gpd.GeoDataFrame(
            sample_points,
            geometry=[Point(p["x"], p["y"]) for p in sample_points],
            crs="EPSG:3006"
        )
        return controls_gdf

def compute_least_cost_paths(cost_raster_path, control_points):
    """Compute least-cost paths between control points."""
    
    print("Loading cost surface...")
    with rasterio.open(cost_raster_path) as src:
        cost_array = src.read(1)
        transform = src.transform
        crs = src.crs
    
    # Handle negative values and no-data
    cost_array = np.where(cost_array == -9999, np.nan, cost_array)  # Handle no-data
    cost_array = np.where(cost_array <= 0, 0.001, cost_array)  # Ensure positive costs
    cost_array = np.where(np.isnan(cost_array), 1.0, cost_array)  # Fill NaN with neutral cost
    
    print(f"Cost surface shape: {cost_array.shape}")
    print(f"Cost range: {np.nanmin(cost_array):.3f} to {np.nanmax(cost_array):.3f}")
    
    # Convert control points to pixel coordinates
    control_pixels = []
    for idx, point in control_points.iterrows():
        geom = point.geometry
        # Convert world coordinates to pixel coordinates
        col, row = ~transform * (geom.x, geom.y)
        col, row = int(col), int(row)
        
        # Check if point is within raster bounds
        if 0 <= row < cost_array.shape[0] and 0 <= col < cost_array.shape[1]:
            control_pixels.append({
                'id': point['id'],
                'row': row, 
                'col': col,
                'x': geom.x,
                'y': geom.y
            })
            print(f"Control point {point['id']}: pixel ({row}, {col}), world ({geom.x:.0f}, {geom.y:.0f})")
        else:
            print(f"Warning: Control point {point['id']} is outside raster bounds!")
    
    if len(control_pixels) < 2:
        print("Error: Need at least 2 valid control points for path computation!")
        return None
    
    # Compute paths between consecutive control points
    paths = []
    
    for i in range(len(control_pixels) - 1):
        start_pt = control_pixels[i]
        end_pt = control_pixels[i + 1]
        
        print(f"Computing path from {start_pt['id']} to {end_pt['id']}...")
        
        try:
            # Use skimage route_through_array for least-cost path
            indices, weight = route_through_array(
                cost_array, 
                (start_pt['row'], start_pt['col']), 
                (end_pt['row'], end_pt['col']),
                fully_connected=True  # Allow diagonal movement
            )
            
            # Convert pixel coordinates back to world coordinates
            path_coords = []
            for row, col in indices:
                x, y = transform * (col, row)
                path_coords.append((x, y))
            
            # Create path geometry
            path_geom = LineString(path_coords)
            
            paths.append({
                'from_id': start_pt['id'],
                'to_id': end_pt['id'],
                'geometry': path_geom,
                'total_cost': weight,
                'length_m': path_geom.length
            })
            
            print(f"  Path length: {path_geom.length:.0f}m, Total cost: {weight:.2f}")
            
        except Exception as e:
            print(f"Error computing path from {start_pt['id']} to {end_pt['id']}: {e}")
            continue
    
    if paths:
        # Create GeoDataFrame with paths
        paths_gdf = gpd.GeoDataFrame(paths, crs=crs)
        
        # Save to file
        paths_gdf.to_file(PATHS_OUTPUT, driver="GeoJSON")
        print(f"Saved {len(paths)} least-cost paths to: {PATHS_OUTPUT}")
        
        # Print summary
        total_length = paths_gdf['length_m'].sum()
        total_cost = paths_gdf['total_cost'].sum()
        print(f"Total route length: {total_length:.0f}m")
        print(f"Total route cost: {total_cost:.2f}")
        
        return paths_gdf
    else:
        print("No valid paths computed!")
        return None

def main():
    """Main function to run least-cost path analysis."""
    
    # Check if cost surface exists
    if not COST_RASTER.exists():
        print(f"Cost surface not found at {COST_RASTER}")
        print("Run enhanced_bayesian_model_optimized.py first to create the cost surface.")
        return
    
    print(f"Using cost surface: {COST_RASTER}")
    
    # Load control points
    control_points = load_control_points()
    print(f"Loaded {len(control_points)} control points")
    
    # Compute least-cost paths
    paths = compute_least_cost_paths(COST_RASTER, control_points)
    
    if paths is not None:
        print(f"Least-cost path analysis completed successfully!")
        print(f"View results in QGIS by loading: {PATHS_OUTPUT}")
    else:
        print("Least-cost path analysis failed!")

if __name__ == "__main__":
    main()
