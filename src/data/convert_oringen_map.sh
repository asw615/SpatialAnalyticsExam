#!/bin/zsh
# Script to convert O-Ringen KMZ map overlay to GeoTIFF in SWEREF 99 TM (EPSG:3006)
# Usage: ./convert_oringen_map.sh

set -e

# Paths (edit if your structure changes)
KMZ_PATH="data/raw/map/O-Ringen Smålandskusten, etapp 4, medel, H21 Elit.kmz"
UNZIP_DIR="data/raw/map/oringen_kmz"
PNG_PATH="$UNZIP_DIR/files/map.png"
TEMP_TIF="data/raw/map/temp.tif"
TEMP1_TIF="data/raw/map/oringen_e4_2024_h21elit.tif"
FINAL_TIF="data/raw/map/oringen_e4_2024_h21elit_REFERENCED.tif"

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

gdalwarp -r near -tps -co COMPRESS=NONE -t_srs EPSG:3006 "$TEMP_TIF" "$TEMP1_TIF"

gdal_translate -of GTiff \
  -gcp 3743.2 1906.16 588141.177 6349651.368 \
  -gcp 4716.913 923.964 588940.333 6350302.028 \
  -gcp 2514.648 2825.987 587156.175 6349102.879 \
  -gcp 1539.196 3885.551 586357.697 6348392.454 \
  -gcp 2185.592 2389.895 586910.048 6349486.591 \
  -gcp 3386.083 310.977 587920.885 6350997.285 \
  -gcp 4310.496 369.632 588641.589 6350819.798 \
  -gcp 3594.309 657.291 588072.56 6350687.888 \
  -gcp 4342.443 927.627 588651.098 6350370.619 \
  -gcp 1396.019 3272.627 586266.366 6348899.339 \
  -gcp 1624.919 2791.653 586461.099 6349251.496 \
  -gcp 2999.694 3639.21 587511.792 6348383.721 \
  -gcp 3243.024 2277.738 587746.707 6349428.371 \
  -gcp 3745.266 1359.296 588168.692 6350108.238 \
  -gcp 4598.308 1746.317 588827.651 6349670.738 \
  "$TEMP1_TIF" "$FINAL_TIF"
rm "$TEMP_TIF" "$TEMP1_TIF"

echo "Done! Output: $FINAL_TIF"