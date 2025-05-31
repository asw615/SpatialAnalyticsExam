#!/usr/bin/env python3
"""
Extract valuable features from Lantmäteriet Topografi 50 data for orienteering speed prediction.

This script processes Swedish national topographic data to create raster features
that are highly relevant for predicting movement speeds in orienteering terrain.

Key features extracted:
1. Terrain accessibility (markframkomlighet) - Rocky/blocked terrain
2. Wetland types (sankmark) - Wet vs firm wetlands
3. Enhanced land cover (mark) - Detailed vegetation types
4. Trail network (ovrig_vag) - Hiking trails, forest roads
5. Enhanced water features (hydrolinje) - Stream network
6. Slope from contour lines (hojdlinje) - Detailed elevation derivatives

Focus on features that significantly impact orienteering movement speed.
"""

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
LANTMATERIET_DIR = Path("data/raw/lantmateriet")
OUTPUT_DIR = Path("data/derived/lantmateriet")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Reference raster for consistent projection and resolution
REFERENCE_RASTER = "data/derived/map/oringen_e4_2024_h21elit_REFERENCED_3006.tif"

def get_reference_params():
    """Get reference projection parameters from the orienteering map."""
    with rasterio.open(REFERENCE_RASTER) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds
        }

def extract_terrain_accessibility():
    """
    Extract terrain accessibility features - crucial for orienteering.
    
    Markframkomlighet indicates areas that are difficult to traverse:
    - 'Blockig mark' - Rocky, boulder-strewn terrain (slow movement)
    - 'Berg i dagen' - Exposed bedrock (very slow/impassable)
    """
    print("Extracting terrain accessibility features...")
    
    gdf = gpd.read_file(LANTMATERIET_DIR / "mark_ln08.gpkg", layer="markframkomlighet")
    ref_params = get_reference_params()
    
    if not gdf.empty:
        # Reproject to match reference
        if gdf.crs != ref_params['crs']:
            gdf = gdf.to_crs(ref_params['crs'])
        
        # Save as GeoPackage
        output_path = OUTPUT_DIR / "terrain_accessibility.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        
        # Show terrain types
        terrain_counts = gdf['objekttyp'].value_counts()
        print(f"  Saved terrain accessibility features: {len(gdf)} polygons")
        for terrain_type, count in terrain_counts.items():
            print(f"    {terrain_type}: {count} features")
    else:
        print("  No terrain accessibility features found")

def extract_wetland_features():
    """
    Extract wetland features - critical for orienteering navigation.
    
    Sankmark (wetlands) significantly impact movement speed:
    - Different wetland types have different traversability
    """
    print("Extracting wetland features...")
    
    gdf = gpd.read_file(LANTMATERIET_DIR / "mark_ln08.gpkg", layer="sankmark")
    ref_params = get_reference_params()
    
    if not gdf.empty:
        # Reproject to match reference
        if gdf.crs != ref_params['crs']:
            gdf = gdf.to_crs(ref_params['crs'])
        
        # Save as GeoPackage
        output_path = OUTPUT_DIR / "wetlands.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        
        # Show wetland types
        wetland_counts = gdf['objekttyp'].value_counts()
        print(f"  Saved wetland features: {len(gdf)} polygons")
        for wetland_type, count in wetland_counts.items():
            print(f"    {wetland_type}: {count} features")
    else:
        print("  No wetland features found")

def extract_building_features():
    """
    Extract building features from Lantmäteriet data.
    
    Buildings are significant obstacles for orienteering:
    - Must be navigated around, not through
    - Create no-go areas that affect route choice
    """
    print("Extracting building features...")
    
    gdf = gpd.read_file(LANTMATERIET_DIR / "byggnadsverk_ln08.gpkg", layer="byggnad")
    ref_params = get_reference_params()
    
    if not gdf.empty:
        # Reproject to match reference
        if gdf.crs != ref_params['crs']:
            gdf = gdf.to_crs(ref_params['crs'])
        
        # Save as GeoPackage
        output_path = OUTPUT_DIR / "buildings.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        
        # Show building types if available
        if 'objekttyp' in gdf.columns:
            building_counts = gdf['objekttyp'].value_counts()
            print(f"  Saved building features: {len(gdf)} polygons")
            for building_type, count in building_counts.items():
                print(f"    {building_type}: {count} features")
        else:
            print(f"  Saved building features: {len(gdf)} polygons")
    else:
        print("  No building features found")

def extract_enhanced_landcover():
    """
    Extract enhanced land cover features from official Swedish data.
    
    Provides more accurate and detailed land cover than OSM-derived data:
    - Forest types (coniferous/mixed vs deciduous)
    - Agricultural areas
    - Built environments
    - Open areas
    """
    print("Extracting enhanced land cover features...")
    
    gdf = gpd.read_file(LANTMATERIET_DIR / "mark_ln08.gpkg", layer="mark")
    ref_params = get_reference_params()
    
    if not gdf.empty:
        # Reproject to match reference
        if gdf.crs != ref_params['crs']:
            gdf = gdf.to_crs(ref_params['crs'])
        
        # Save as GeoPackage
        output_path = OUTPUT_DIR / "landcover.gpkg"
        gdf.to_file(output_path, driver="GPKG")
        
        # Show land cover types
        landcover_counts = gdf['objekttyp'].value_counts()
        print(f"  Saved land cover features: {len(gdf)} polygons")
        for landcover_type, count in landcover_counts.items():
            print(f"    {landcover_type}: {count} features")
    else:
        print("  No land cover features found")

def extract_trail_network():
    """
    Extract detailed trail and path network.
    
    Provides official trail data that's often more accurate than OSM:
    - Gångstig - Hiking trails
    - Vandringsled - Marked hiking trails  
    - Traktorväg - Forest roads
    - Cykelväg - Bicycle paths
    """
    print("Extracting trail network features...")
    
    # Main roads
    roads_gdf = gpd.read_file(LANTMATERIET_DIR / "kommunikation_ln08.gpkg", layer="vaglinje")
    # Other paths/trails
    trails_gdf = gpd.read_file(LANTMATERIET_DIR / "kommunikation_ln08.gpkg", layer="ovrig_vag")
    
    ref_params = get_reference_params()
    
    # Save as GeoPackages
    if not roads_gdf.empty:
        if roads_gdf.crs != ref_params['crs']:
            roads_gdf = roads_gdf.to_crs(ref_params['crs'])
        roads_output = OUTPUT_DIR / "roads.gpkg"
        roads_gdf.to_file(roads_output, driver="GPKG")
        print(f"  Saved roads: {len(roads_gdf)} features")
    
    if not trails_gdf.empty:
        if trails_gdf.crs != ref_params['crs']:
            trails_gdf = trails_gdf.to_crs(ref_params['crs'])
        trails_output = OUTPUT_DIR / "trails.gpkg"
        trails_gdf.to_file(trails_output, driver="GPKG")
        
        # Show trail types
        trail_counts = trails_gdf['objekttyp'].value_counts()
        print(f"  Saved trails: {len(trails_gdf)} features")
        for trail_type, count in trail_counts.items():
            print(f"    {trail_type}: {count} features")

def extract_water_features():
    """
    Extract water features that impact orienteering routes.
    
    Water features are significant barriers in orienteering:
    - Streams and rivers (must be crossed at specific points)
    - Create distance-to-water features
    """
    print("Extracting water features...")
    
    water_gdf = gpd.read_file(LANTMATERIET_DIR / "hydrografi_ln08.gpkg", layer="hydrolinje")
    ref_params = get_reference_params()
    
    if not water_gdf.empty:
        if water_gdf.crs != ref_params['crs']:
            water_gdf = water_gdf.to_crs(ref_params['crs'])
        
        # Save as GeoPackage
        output_path = OUTPUT_DIR / "water_features.gpkg"
        water_gdf.to_file(output_path, driver="GPKG")
        
        total_length = water_gdf.geometry.length.sum() / 1000  # km
        print(f"  Saved water features: {total_length:.1f} km total length")
    else:
        print("  No water features found")


def create_distance_rasters():
    """
    Create distance-to-feature rasters for key Lantmäteriet features.
    
    Distance features are often more predictive than binary presence/absence.
    This will be handled by the main rasterization script.
    """
    print("Distance rasters will be created by the main rasterization script...")
    pass

def main():
    """Extract all Lantmäteriet features for orienteering analysis."""
    print("=== EXTRACTING LANTMÄTERIET FEATURES FOR ORIENTEERING ===")
    print(f"Source data: {LANTMATERIET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check if reference raster exists
    if not Path(REFERENCE_RASTER).exists():
        print(f"ERROR: Reference raster not found: {REFERENCE_RASTER}")
        return
    
    try:
        # Extract terrain features
        extract_terrain_accessibility()
        extract_wetland_features()
        extract_enhanced_landcover()
        extract_building_features()
        extract_trail_network()
        extract_water_features()
        create_distance_rasters()
        
        print("\n=== FEATURE EXTRACTION COMPLETE ===")
        print(f"Features saved to: {OUTPUT_DIR}")
        print("\nKey features for orienteering model:")
        print("- rocky_terrain, exposed_rock: Terrain difficulty")
        print("- wetland_firm, wetland_wet: Wetland impact")
        print("- lm_forest_*: Enhanced forest types")
        print("- lm_buildings: Building obstacles")
        print("- lm_hiking_trails, lm_forest_roads: Trail network")
        print("- lm_water_features: Water barriers")
        print("- dist_to_*: Distance-based features")
        print("\nReady for rasterization!")

    except Exception as e:
        print(f"ERROR during feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
