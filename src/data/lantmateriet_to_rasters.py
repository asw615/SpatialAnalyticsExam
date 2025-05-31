#!/usr/bin/env python3
"""
Convert Lantmäteriet vector features to rasters for orienteering speed prediction.
Focus on high-quality Swedish topographic data, using an orienteering map as a reference grid.
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import os
from scipy.ndimage import distance_transform_edt

# --- CONFIG ---
REFERENCE_RASTER = "data/derived/map/oringen_e4_2024_h21elit_REFERENCED_3006.tif"
LANTMATERIET_DIR = "data/derived/lantmateriet"
RASTER_OUTDIR = "data/derived/rasters"
os.makedirs(RASTER_OUTDIR, exist_ok=True)

print("=== LANTMÄTERIET VECTOR TO RASTER CONVERSION (FOCUSED) ===")
print("Converting high-quality Swedish topographic data to raster format")
print("Using orienteering map as reference for optimal performance and focus")

# --- LOAD ORIENTEERING MAP AS REFERENCE GRID ---
print("Loading orienteering map reference grid...")
ref = rasterio.open(REFERENCE_RASTER)
ref_arr = ref.read(1)
meta = ref.meta.copy()
print(f"Reference shape: {ref_arr.shape}, Resolution: {ref.res[0]:.1f}m")
print(f"Reference bounds: {ref.bounds}")
print(f"Reference CRS: {ref.crs}")

# --- TERRAIN ACCESSIBILITY (Rocky terrain, exposed bedrock) ---
print("\n--- Processing Terrain Accessibility ---")
terrain_path = os.path.join(LANTMATERIET_DIR, "terrain_accessibility.gpkg")
if os.path.exists(terrain_path):
    terrain_gdf = gpd.read_file(terrain_path).to_crs(ref.crs)
    print(f"Loaded {len(terrain_gdf)} terrain accessibility features")
    
    # Create rocky terrain raster (both types combined)
    shapes = zip(terrain_gdf.geometry, [1]*len(terrain_gdf))
    rocky_raster = rasterize(
        shapes,
        out_shape=ref_arr.shape,
        transform=ref.transform,
        fill=0,
        dtype=np.uint8
    )
    
    # Save rocky terrain presence raster
    out_path = os.path.join(RASTER_OUTDIR, "rocky_terrain.tif")
    meta_uint8 = meta.copy()
    meta_uint8.update({"driver": "GTiff", "dtype": "uint8", "count": 1, "nodata": 0})
    with rasterio.open(out_path, "w", **meta_uint8) as dst:
        dst.write(rocky_raster, 1)
    
    coverage = np.sum(rocky_raster > 0) / rocky_raster.size
    print(f"Rocky terrain coverage: {coverage:.2%} of area")
    print(f"Saved: {out_path}")
    
    # Create distance-to-rocky-terrain raster
    dist = distance_transform_edt(rocky_raster == 0) * ref.res[0]
    dist_path = os.path.join(RASTER_OUTDIR, "dist_to_rocky_terrain.tif")
    meta_float32 = meta.copy()
    meta_float32.update({"driver": "GTiff", "dtype": "float32", "count": 1, "nodata": -9999.0})
    with rasterio.open(dist_path, "w", **meta_float32) as dst:
        dst.write(dist.astype(np.float32), 1)
    print(f"Saved distance raster: {dist_path}")
else:
    print(f"Terrain accessibility file not found: {terrain_path}")

# --- WETLANDS (Firm vs wet wetlands) ---
print("\n--- Processing Wetlands ---")
wetland_path = os.path.join(LANTMATERIET_DIR, "wetlands.gpkg")
if os.path.exists(wetland_path):
    wetlands_gdf = gpd.read_file(wetland_path).to_crs(ref.crs)
    print(f"Loaded {len(wetlands_gdf)} wetland features")
    
    # Separate firm and wet wetlands
    firm_wetlands = wetlands_gdf[wetlands_gdf['objekttyp'] == 'Sankmark, fast']
    wet_wetlands = wetlands_gdf[wetlands_gdf['objekttyp'] == 'Sankmark, våt']
    
    print(f"  - Firm wetlands: {len(firm_wetlands)}")
    print(f"  - Wet wetlands: {len(wet_wetlands)}")
    
    # Process firm wetlands
    if len(firm_wetlands) > 0:
        shapes = zip(firm_wetlands.geometry, [1]*len(firm_wetlands))
        firm_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
        
        out_path = os.path.join(RASTER_OUTDIR, "firm_wetlands.tif")
        with rasterio.open(out_path, "w", **meta_uint8) as dst:
            dst.write(firm_raster, 1)
        
        coverage = np.sum(firm_raster > 0) / firm_raster.size
        print(f"Firm wetlands coverage: {coverage:.2%} of area")
        print(f"Saved: {out_path}")
        
        # Distance to firm wetlands
        dist = distance_transform_edt(firm_raster == 0) * ref.res[0]
        dist_path = os.path.join(RASTER_OUTDIR, "dist_to_firm_wetlands.tif")
        with rasterio.open(dist_path, "w", **meta_float32) as dst:
            dst.write(dist.astype(np.float32), 1)
        print(f"Saved distance raster: {dist_path}")
    
    # Process wet wetlands
    if len(wet_wetlands) > 0:
        shapes = zip(wet_wetlands.geometry, [1]*len(wet_wetlands))
        wet_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
        
        out_path = os.path.join(RASTER_OUTDIR, "wet_wetlands.tif")
        with rasterio.open(out_path, "w", **meta_uint8) as dst:
            dst.write(wet_raster, 1)
        
        coverage = np.sum(wet_raster > 0) / wet_raster.size
        print(f"Wet wetlands coverage: {coverage:.2%} of area")
        print(f"Saved: {out_path}")
        
        # Distance to wet wetlands
        dist = distance_transform_edt(wet_raster == 0) * ref.res[0]
        dist_path = os.path.join(RASTER_OUTDIR, "dist_to_wet_wetlands.tif")
        with rasterio.open(dist_path, "w", **meta_float32) as dst:
            dst.write(dist.astype(np.float32), 1)
        print(f"Saved distance raster: {dist_path}")
else:
    print(f"Wetlands file not found: {wetland_path}")

# --- ENHANCED LAND COVER ---
print("\n--- Processing Enhanced Land Cover ---")
landcover_path = os.path.join(LANTMATERIET_DIR, "landcover.gpkg")
if os.path.exists(landcover_path):
    landcover_gdf = gpd.read_file(landcover_path).to_crs(ref.crs)
    print(f"Loaded {len(landcover_gdf)} land cover features")
    
    # Get unique land cover types
    landcover_types = landcover_gdf['objekttyp'].unique()
    print(f"Land cover types: {landcover_types}")
    
    for lc_type in landcover_types:
        lc_features = landcover_gdf[landcover_gdf['objekttyp'] == lc_type]
        
        shapes = zip(lc_features.geometry, [1]*len(lc_features))
        lc_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
        
        # Clean filename
        clean_name = lc_type.replace(' ', '_').replace('/', '_').replace('-', '_').replace(',', '').lower()
        out_path = os.path.join(RASTER_OUTDIR, f"landcover_{clean_name}.tif")
        
        with rasterio.open(out_path, "w", **meta_uint8) as dst:
            dst.write(lc_raster, 1)
        
        coverage = np.sum(lc_raster > 0) / lc_raster.size
        print(f"{lc_type}: {len(lc_features)} features, {coverage:.2%} coverage")
        print(f"Saved: {out_path}")
else:
    print(f"Land cover file not found: {landcover_path}")

# --- TRAIL NETWORKS WITH BUFFERS ---
print("\n--- Processing Trail Networks ---")
trail_path = os.path.join(LANTMATERIET_DIR, "trails.gpkg")
if os.path.exists(trail_path):
    trails_gdf = gpd.read_file(trail_path).to_crs(ref.crs)
    print(f"Loaded {len(trails_gdf)} trail features")
    
    # Get unique trail types for reporting
    trail_types = trails_gdf['objekttyp'].unique()
    print(f"Trail types: {trail_types}")
    
    # Create combined trails raster with 2m buffer for all trails
    all_trail_geom = trails_gdf.geometry.buffer(2.0)  # Standard 2m buffer for all trails
    shapes = zip(all_trail_geom, [1]*len(trails_gdf))
    all_trails_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    out_path = os.path.join(RASTER_OUTDIR, "on_trails.tif")
    with rasterio.open(out_path, "w", **meta_uint8) as dst:
        dst.write(all_trails_raster, 1)
    
    coverage = np.sum(all_trails_raster > 0) / all_trails_raster.size
    print(f"All trails combined: {len(trails_gdf)} features, 2.0m buffer, {coverage:.2%} coverage")
    print(f"Saved: {out_path}")
    
    # Distance to any trail (using original unbuffered geometry)
    original_shapes = zip(trails_gdf.geometry, [1]*len(trails_gdf))
    original_raster = rasterize(original_shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    dist = distance_transform_edt(original_raster == 0) * ref.res[0]
    dist_path = os.path.join(RASTER_OUTDIR, "dist_to_trails.tif")
    with rasterio.open(dist_path, "w", **meta_float32) as dst:
        dst.write(dist.astype(np.float32), 1)
    print(f"Saved distance raster: {dist_path}")
else:
    print(f"Trails file not found: {trail_path}")

# --- ROADS WITH BUFFERS ---
print("\n--- Processing Roads ---")
roads_path = os.path.join(LANTMATERIET_DIR, "roads.gpkg")
if os.path.exists(roads_path):
    roads_gdf = gpd.read_file(roads_path).to_crs(ref.crs)
    print(f"Loaded {len(roads_gdf)} road features")
    
    # Apply buffer for roads (3m standard, matching original script)
    buffered_roads = roads_gdf.geometry.buffer(3.0)
    shapes = zip(buffered_roads, [1]*len(roads_gdf))
    roads_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    out_path = os.path.join(RASTER_OUTDIR, "on_roads.tif")
    with rasterio.open(out_path, "w", **meta_uint8) as dst:
        dst.write(roads_raster, 1)
    
    coverage = np.sum(roads_raster > 0) / roads_raster.size
    print(f"Roads: 3.0m buffer, {coverage:.2%} coverage")
    print(f"Saved: {out_path}")
    
    # Distance to roads
    original_shapes = zip(roads_gdf.geometry, [1]*len(roads_gdf))
    original_raster = rasterize(original_shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    dist = distance_transform_edt(original_raster == 0) * ref.res[0]
    dist_path = os.path.join(RASTER_OUTDIR, "dist_to_roads.tif")
    with rasterio.open(dist_path, "w", **meta_float32) as dst:
        dst.write(dist.astype(np.float32), 1)
    print(f"Saved distance raster: {dist_path}")
else:
    print(f"Roads file not found: {roads_path}")

# --- WATER FEATURES ---
print("\n--- Processing Water Features ---")
water_path = os.path.join(LANTMATERIET_DIR, "water_features.gpkg")
if os.path.exists(water_path):
    water_gdf = gpd.read_file(water_path).to_crs(ref.crs)
    print(f"Loaded {len(water_gdf)} water features")
    
    # Apply small buffer for water features (1m to account for banks, matching original approach)
    buffered_water = water_gdf.geometry.buffer(1.0)
    shapes = zip(buffered_water, [1]*len(water_gdf))
    water_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    out_path = os.path.join(RASTER_OUTDIR, "on_water.tif")
    with rasterio.open(out_path, "w", **meta_uint8) as dst:
        dst.write(water_raster, 1)
    
    coverage = np.sum(water_raster > 0) / water_raster.size
    print(f"Water features: 1.0m buffer, {coverage:.2%} coverage")
    print(f"Saved: {out_path}")
    
    # Distance to water
    original_shapes = zip(water_gdf.geometry, [1]*len(water_gdf))
    original_raster = rasterize(original_shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    dist = distance_transform_edt(original_raster == 0) * ref.res[0]
    dist_path = os.path.join(RASTER_OUTDIR, "dist_to_water.tif")
    with rasterio.open(dist_path, "w", **meta_float32) as dst:
        dst.write(dist.astype(np.float32), 1)
    print(f"Saved distance raster: {dist_path}")
else:
    print(f"Water features file not found: {water_path}")

# --- BUILDINGS ---
print("\n--- Processing Buildings ---")
buildings_path = os.path.join(LANTMATERIET_DIR, "buildings.gpkg")
if os.path.exists(buildings_path):
    buildings_gdf = gpd.read_file(buildings_path).to_crs(ref.crs)
    print(f"Loaded {len(buildings_gdf)} building features")
    
    # No buffer for buildings (already polygons, matching original approach)
    shapes = zip(buildings_gdf.geometry, [1]*len(buildings_gdf))
    buildings_raster = rasterize(shapes, out_shape=ref_arr.shape, transform=ref.transform, fill=0, dtype=np.uint8)
    
    out_path = os.path.join(RASTER_OUTDIR, "on_buildings.tif")
    with rasterio.open(out_path, "w", **meta_uint8) as dst:
        dst.write(buildings_raster, 1)
    
    coverage = np.sum(buildings_raster > 0) / buildings_raster.size
    print(f"Buildings: {coverage:.2%} coverage")
    print(f"Saved: {out_path}")
    
    # Distance to buildings
    dist = distance_transform_edt(buildings_raster == 0) * ref.res[0]
    dist_path = os.path.join(RASTER_OUTDIR, "dist_to_buildings.tif")
    with rasterio.open(dist_path, "w", **meta_float32) as dst:
        dst.write(dist.astype(np.float32), 1)
    print(f"Saved distance raster: {dist_path}")
else:
    print(f"Buildings file not found: {buildings_path}")

# --- RASTERIZE IMPASSABLE AREAS (keep from original) ---
IMPASSABLE_SHP = "data/derived/map/impassible_areas.shp"
if os.path.exists(IMPASSABLE_SHP):
    print("\n--- Processing Impassable Areas ---")
    imp_gdf = gpd.read_file(IMPASSABLE_SHP).to_crs(ref.crs)
    if len(imp_gdf) > 0:
        shapes = zip(imp_gdf.geometry, [1]*len(imp_gdf))
        imp_raster = rasterize(
            shapes,
            out_shape=ref_arr.shape,
            transform=ref.transform,
            fill=0,
            dtype=np.uint8
        )
        imp_path = os.path.join(RASTER_OUTDIR, "impassible_areas.tif")
        with rasterio.open(imp_path, "w", **meta_uint8) as dst:
            dst.write(imp_raster, 1)
        print(f"Saved impassible areas raster: {imp_path}")
    else:
        print("No impassable areas found in shapefile.")
else:
    print(f"Impassable areas shapefile not found: {IMPASSABLE_SHP}")

# Close reference raster
ref.close()

print("\n=== FOCUSED VECTOR TO RASTER CONVERSION COMPLETE ===")
print(f"All Lantmäteriet features converted to rasters in: {RASTER_OUTDIR}")
