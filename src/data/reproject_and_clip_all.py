"""
reproject_and_clip_all.py

Reproject and clip all spatial layers in data/raw/ to EPSG:3006 and the DEM/map bounding box.
Handles rasters (GeoTIFF), vectors (GPKG, SHP), and provides a template for GPS tracks (GPX).

Requirements:
    pip install geopandas rasterio fiona shapely gpxpy

Usage:
    python reproject_and_clip_all.py
"""
import os
import glob
import shutil
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, LineString
import fiona
import gpxpy
import gpxpy.gpx

# --- CONFIG ---
CRS_TARGET = "EPSG:3006"
BBOX_3006 = [585238.65, 6347920.63, 589520.92, 6351159.42]  # minx, miny, maxx, maxy
BBOX_POLY = box(*BBOX_3006)

# --- HELPERS ---
def reproject_raster(src_path, dst_path, crs_target, bbox_poly=None):
    with rasterio.open(src_path) as src:
        if src.crs.to_string() == crs_target:
            crs_dst = src.crs
        else:
            crs_dst = crs_target
        transform, width, height = calculate_default_transform(
            src.crs, crs_dst, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_dst,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_dst,
                    resampling=Resampling.nearest)
    # Clip if bbox_poly is provided and overlaps raster
    if bbox_poly is not None:
        with rasterio.open(dst_path) as clipped_src:
            # Check for overlap before masking
            raster_bounds = box(*clipped_src.bounds)
            if not raster_bounds.intersects(bbox_poly):
                print(f"[WARN] BBox does not overlap raster: {dst_path}. Skipping clipping.")
                return
            out_image, out_transform = mask(clipped_src, [bbox_poly], crop=True)
            out_meta = clipped_src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            clipped_path = dst_path.replace('.tif', '_clipped.tif')
            with rasterio.open(clipped_path, "w", **out_meta) as dest:
                dest.write(out_image)
            shutil.move(clipped_path, dst_path)


def reproject_vector(src_path, dst_path, crs_target, bbox_poly=None):
    gdf = gpd.read_file(src_path)
    if gdf.crs is None:
        raise ValueError(f"No CRS found in {src_path}")
    if gdf.crs.to_string() != crs_target:
        gdf = gdf.to_crs(crs_target)
    if bbox_poly is not None:
        gdf = gdf[gdf.geometry.intersects(bbox_poly)]
    gdf.to_file(dst_path)


def process_gpkg_layers(src_path, dst_path, crs_target, bbox_poly=None):
    layers = fiona.listlayers(src_path)
    for layer in layers:
        gdf = gpd.read_file(src_path, layer=layer)
        if gdf.crs is None:
            raise ValueError(f"No CRS in {src_path}:{layer}")
        if gdf.crs.to_string() != crs_target:
            gdf = gdf.to_crs(crs_target)
        if bbox_poly is not None:
            gdf = gdf[gdf.geometry.intersects(bbox_poly)]
        out_gpkg = dst_path.replace('.gpkg', f'_{layer}_3006.gpkg')
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")


def process_gps_gpx(src_path, dst_path, crs_target, bbox_poly=None):
    # Read GPX, convert to GeoDataFrame, reproject, clip, save as GPKG
    with open(src_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    lines = []
    for track in gpx.tracks:
        for segment in track.segments:
            coords = [(p.longitude, p.latitude) for p in segment.points]
            if len(coords) > 1:
                lines.append(coords)
    if not lines:
        print(f"No track lines in {src_path}")
        return
    gdf = gpd.GeoDataFrame(geometry=[LineString(line) for line in lines], crs="EPSG:4326")
    gdf = gdf.to_crs(crs_target)
    if bbox_poly is not None:
        gdf = gdf[gdf.geometry.intersects(bbox_poly)]
    gdf.to_file(dst_path, driver="GPKG")

# --- MAIN ---
if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("data/derived/dem", exist_ok=True)
    os.makedirs("data/derived/map", exist_ok=True)
    os.makedirs("data/derived/gps", exist_ok=True)
    os.makedirs("data/derived/ndvi", exist_ok=True)  # NDVI output

    # --- RASTER FILES ---
    raster_files = glob.glob("data/raw/dem/*.tif") + [
        "data/raw/map/oringen_e4_2024_h21elit_REFERENCED.tif"
    ]
    # Add all NDVI rasters in data/raw/ndvi
    ndvi_files = glob.glob("data/raw/ndvi/*NDVI*.tif")
    raster_files += ndvi_files
    for src in raster_files:
        # Output to derived, add _3006 suffix
        base = os.path.basename(src)
        # Place NDVI output in ndvi folder
        if "NDVI" in base.upper():
            out_dir = "data/derived/ndvi"
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = "data/derived/dem" if "dem" in src else "data/derived/map"
        dst = os.path.join(out_dir, base.replace('.tif', '_3006.tif'))
        print(f"Processing raster: {src} -> {dst}")
        reproject_raster(src, dst, CRS_TARGET, bbox_poly=BBOX_POLY)

    # --- VECTOR FILES ---

    # Control points (SHP)
    control_shp = "data/raw/map/control_points/control_points_race.shp"
    dst_shp = "data/derived/map/control_points_race_3006.shp"
    print(f"Processing vector: {control_shp} -> {dst_shp}")
    reproject_vector(control_shp, dst_shp, CRS_TARGET, bbox_poly=BBOX_POLY)

    # --- GPS TRACKS (GPX) ---
    gps_dir = "data/raw/gps/"
    gps_out_dir = "data/derived/gps/"
    if os.path.exists(gps_dir):
        for gpx_file in glob.glob(os.path.join(gps_dir, "*.gpx")):
            base = os.path.basename(gpx_file)
            out_gpkg = os.path.join(gps_out_dir, base.replace('.gpx', '_track_3006.gpkg'))
            print(f"Processing GPS track: {gpx_file} -> {out_gpkg}")
            process_gps_gpx(gpx_file, out_gpkg, CRS_TARGET, bbox_poly=BBOX_POLY)
    else:
        print("No GPS directory found. Add GPX files to data/raw/gps/ for processing.")

    print("\n✅ All layers processed and ready for analysis.")
