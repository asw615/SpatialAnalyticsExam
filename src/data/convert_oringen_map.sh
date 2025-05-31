#!/bin/zsh
# Script to convert O-Ringen KMZ map overlay to GeoTIFF in SWEREF 99 TM (EPSG:3006)
# Usage: ./convert_oringen_map.sh

set -e

# Paths (edit if your structure changes)
KMZ_PATH="data/raw/map/O-Ringen Smålandskusten, etapp 4, medel, H21 Elit.kmz"
UNZIP_DIR="data/raw/map/oringen_kmz"
PNG_PATH="$UNZIP_DIR/files/map.png"
TEMP_TIF="data/raw/map/temp.tif"
FINAL_TIF="data/raw/map/oringen_e4_2024_h21elit.tif"

# 1. Unzip KMZ (if not already unzipped)
if [ ! -f "$PNG_PATH" ]; then
  echo "Unzipping KMZ..."
  unzip -o "$KMZ_PATH" -d "$UNZIP_DIR"
fi

# 2. Georeference PNG using GCPs and rotation from KML
# KML bounds
WEST=16.414410364623624
EAST=16.48433932436722
SOUTH=57.266346980866828
NORTH=57.29463719510202
ROTATION=-6.2194907496088163
WIDTH=4978
HEIGHT=3719

# Calculate center
CENTER_X=$(echo "($WEST + $EAST) / 2" | bc -l)
CENTER_Y=$(echo "($SOUTH + $NORTH) / 2" | bc -l)

# Function to rotate a point (lon,lat) around center by ROTATION degrees
rotate_point() {
  local x0=$1
  local y0=$2
  local cx=$3
  local cy=$4
  local angle_deg=$5
  python3 -c "import math; angle=math.radians($angle_deg); x0=float($x0); y0=float($y0); cx=float($cx); cy=float($cy); x=cx+(x0-cx)*math.cos(angle)-(y0-cy)*math.sin(angle); y=cy+(x0-cx)*math.sin(angle)+(y0-cy)*math.cos(angle); print(f'{x} {y}')"
}

# Calculate rotated bounds for each corner
UL=$(rotate_point $WEST $NORTH $CENTER_X $CENTER_Y $ROTATION)
UR=$(rotate_point $EAST $NORTH $CENTER_X $CENTER_Y $ROTATION)
LR=$(rotate_point $EAST $SOUTH $CENTER_X $CENTER_Y $ROTATION)
LL=$(rotate_point $WEST $SOUTH $CENTER_X $CENTER_Y $ROTATION)

# Print GCPs for debugging
echo "GCPs used (rotated bounds):"
echo "  UL (0,0):         $UL"
echo "  UR (4977,0):      $UR"
echo "  LR (4977,3718):   $LR"
echo "  LL (0,3718):      $LL"

# 3. Apply manually determined GCPs from QGIS

gdal_translate -of GTiff \
  -gcp 3862.659 984.882 588642.703 6350816.912 \
  -gcp 3227.178 1066.077 588072.465 6350688.735 \
  -gcp 3366.612 1747.718 588170.21 6350107.751 \
  -gcp 2922.991 2452.432 587746.66 6349430.322 \
  -gcp 2054.163 2708.966 586952.384 6349134.658 \
  "$PNG_PATH" "$TEMP_TIF"

gdalwarp -r near -tps -co COMPRESS=NONE -t_srs EPSG:3006 "$TEMP_TIF" "$FINAL_TIF"

echo "Done! Output: $FINAL_TIF"
