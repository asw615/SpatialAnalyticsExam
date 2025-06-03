#!/usr/bin/env python3
"""
Convert a GeoTIFF with GCPs to a properly georeferenced GeoTIFF with geotransform.
Uses GDAL's built-in transformation capabilities.
"""

import rasterio
from rasterio.crs import CRS
import subprocess
import tempfile
import os
import sys

def convert_gcps_to_geotransform(input_path, output_path, target_crs="EPSG:3006"):
    """
    Convert a file with GCPs to a properly georeferenced file with geotransform.
    Uses GDAL's gdalwarp with TPS transformation for accurate results.
    """
    with rasterio.open(input_path) as src:
        # Get GCPs and their CRS
        gcps, gcp_crs = src.gcps
        
        if not gcps:
            print(f"No GCPs found in {input_path}")
            return False
            
        print(f"Found {len(gcps)} GCPs")
        
        # Use GDAL's gdalwarp to properly transform GCPs to geotransform
        # This replicates what QGIS does internally
        temp_vrt = input_path.replace('.tif', '_temp.vrt')
        
        try:
            # First create a VRT with the GCPs
            cmd_vrt = [
                'gdal_translate', 
                '-of', 'VRT',
                input_path, 
                temp_vrt
            ]
            result = subprocess.run(cmd_vrt, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating VRT: {result.stderr}")
                return False
            
            # Then use gdalwarp with TPS to create properly georeferenced output
            cmd_warp = [
                'gdalwarp',
                '-r', 'near',
                '-tps',  # Thin Plate Spline transformation
                '-t_srs', target_crs,
                temp_vrt,
                output_path
            ]
            result = subprocess.run(cmd_warp, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error warping file: {result.stderr}")
                return False
                
            print(f"Successfully created georeferenced file: {output_path}")
            return True
            
        finally:
            # Clean up temp VRT
            if os.path.exists(temp_vrt):
                os.remove(temp_vrt)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_gcps_to_geotransform.py <input.tif> <output.tif>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_gcps_to_geotransform(input_file, output_file)
