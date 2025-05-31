# Spatial Analytics for Orienteering Route Optimization: Technical Documentation

## Project Overview

This project implements a comprehensive spatial analytics framework for optimizing orienteering routes using machine learning techniques. The analysis combines GPS tracking data from elite orienteers with high-resolution environmental datasets to model movement costs and generate optimal path recommendations.

**Study Area**: O-Ringen Smålandskusten, Stage 4 (Medium distance), H21 Elite category  
**Date**: 2025 10Mila event  
**Coordinate System**: EPSG:3006 (SWEREF99 TM)  
**Spatial Resolution**: 1-10m depending on data source  

## Data Sources and Collection

### 1. GPS Tracking Data
**Source**: Tulospalvelu.fi via LiveLox viewer  
**URL**: https://www.livelox.com/Viewer/O-Ringen-Smalandskusten-etapp-4-medel/H21-Elit  
**Description**: Real-time GPS tracks from 71 elite orienteering athletes  
**Format**: GPX files converted to CSV with WKT geometry  
**Temporal Resolution**: ~1-5 second intervals  
**Attributes**:
- `runner_id`: Unique athlete identifier
- `timestamp`: GPS recording time
- `geometry`: Point geometry (EPSG:3006)
- `speed`: Instantaneous speed (km/h)
- `elevation`: GPS elevation (meters)
- `track_id`: Track segment identifier

**Data Processing**:
```python
# Speed filtering applied
track_df = track_df[
    (track_df["speed"] >= 0.5) & 
    (track_df["speed"] <= 25.0)
]
```

**Final Dataset**: 18,171 GPS points from multiple athletes

### 2. Digital Elevation Model (DEM)
**Source**: Lantmäteriet (Swedish National Land Survey)  
**Product**: Markhöjdmodell Grid 1+ (1m resolution)  
**API**: STAC (SpatioTemporal Asset Catalog)  
**Authentication**: Geotorget credentials required  
**Coverage**: 12 tiles covering study area  

**Technical Specifications**:
- Original resolution: 1m × 1m
- Processed resolution: 5m × 5m (aggregated using average)
- Vertical accuracy: ±0.5m (RMS)
- Format: GeoTIFF
- CRS: EPSG:3006

**Tile IDs Downloaded**:
```
634_58_5025, 634_58_5050, 634_58_5075,
634_58_7525, 634_58_7550, 634_58_7575,
634_59_5000, 634_59_7500,
635_58_0025, 635_58_0050, 635_58_0075, 635_59_0000
```

**Processing Pipeline**:
1. Download via STAC API with authentication
2. Mosaic tiles using GDAL VRT
3. Reproject and resample to 5m grid
4. Generate slope and aspect derivatives

```bash
gdalbuildvrt mosaic.vrt *.tif
gdalwarp -tr 5 5 -tap -r average -t_srs EPSG:3006 mosaic.vrt dem_5m.tif
gdaldem slope dem_5m.tif slope.tif
gdaldem aspect dem_5m.tif aspect.tif
```

### 3. Land Cover Classification
**Source**: Naturvårdsverket (Swedish EPA)  
**Product**: National Land Cover Database (NMD 2018/2023)  
**Resolution**: 10m × 10m  
**Format**: GeoTIFF  
**Classification System**: CORINE-compatible with Swedish extensions  

**Key Land Cover Classes**:
- Barr- och blandskog (Coniferous and mixed forest)
- Öppen mark (Open land)
- Våtmarker (Wetlands)
- Bebyggelse (Built-up areas)
- Vatten (Water bodies)

**Processing**:
- Rasterized to binary masks for each land cover type
- Resampled to match DEM grid (5m resolution)
- Created distance rasters using Euclidean distance transform

### 4. Vegetation Index (NDVI)
**Source**: Sentinel-2 Satellite Imagery  
**Date**: 2024-05-30 (closest cloud-free acquisition to race date)  
**Sensor**: Sentinel-2A MSI  
**Tile**: T33VWD  
**Resolution**: 10m native, resampled to 5m  

**Processing**:
- Downloaded from Copernicus Open Access Hub
- Atmospheric correction applied
- NDVI calculated: (NIR - Red) / (NIR + Red)
- Reprojected to EPSG:3006

### 5. OpenStreetMap (OSM) Features
**Source**: Geofabrik Sweden extract  
**Download Date**: Weekly updated PBF file  
**Processing Tool**: PyROSM with custom filters  

**Feature Categories Extracted**:

#### Transportation Network:
- `trails`: footway, path, track
- `roads`: all highway types
- Binary masks: on_trails, on_roads
- Distance rasters: dist_to_trails, dist_to_roads

#### Infrastructure:
- `buildings`: all building types
- `barriers`: fence, wall, hedge
- Binary masks: on_buildings
- Distance rasters: dist_to_buildings

#### Natural Features:
- `water`: natural=water, waterway
- Binary masks: on_water
- Distance rasters: dist_to_water

**Processing Workflow**:
1. Extract features using custom OSM queries
2. Clip to study area bounding box
3. Convert to EPSG:3006
4. Rasterize to 5m grid
5. Generate distance rasters using scipy.ndimage

### 6. Control Points
**Source**: Race organization data  
**Format**: Shapefile with race control locations  
**Attributes**:
- `cont_point`: Control number (1-N)
- `start`: Start point flag (1.0 if start)
- `stop`: Finish point flag (1.0 if finish)
- `geometry`: Point geometry (EPSG:3006)

**Total Controls**: 24 control points (start + 22 controls + finish)

## Feature Engineering

### Terrain-Based Features
1. **Elevation**: Direct elevation values (meters above sea level)
2. **Slope**: Gradient in degrees calculated from DEM
3. **Aspect**: Slope direction (0-360 degrees)
4. **NDVI**: Normalized Difference Vegetation Index (-1 to +1)

### Distance-Based Features
Euclidean distance calculations to nearest features:
- `dist_to_roads`: Distance to road network (meters)
- `dist_to_trails`: Distance to trail network (meters)
- `dist_to_water`: Distance to water bodies (meters)
- `dist_to_buildings`: Distance to built structures (meters)

### Binary Presence Features
Point-in-polygon/raster queries:
- `on_roads`: 1 if on road, 0 otherwise
- `on_trails`: 1 if on trail, 0 otherwise
- `on_water`: 1 if on water, 0 otherwise
- `on_buildings`: 1 if on building, 0 otherwise

### Land Cover Features
Binary indicators for major land cover types:
- `landcover_barr__och_blandskog`: Coniferous/mixed forest
- `landcover_öppen_mark`: Open land
- `rocky_terrain`: Rocky/exposed areas
- `firm_wetlands`: Firm wetland areas

## Machine Learning Pipeline

### Model Architecture: Random Forest Regression

**Framework**: scikit-learn RandomForestRegressor  
**Target Variable**: log(speed) in km/h  
**Prediction Task**: Movement speed estimation based on environmental conditions  

**Model Hyperparameters**:
```python
RandomForestRegressor(
    n_estimators=200,       # Number of trees
    max_depth=15,           # Maximum tree depth
    min_samples_split=5,    # Minimum samples to split node
    min_samples_leaf=2,     # Minimum samples in leaf
    max_features='sqrt',    # Features per split
    n_jobs=8,              # Parallel processing
    random_state=42,       # Reproducibility
    bootstrap=True,        # Bootstrap sampling
    oob_score=True         # Out-of-bag validation
)
```

**Training Strategy**:
- All available GPS points used (no subsampling)
- Log-transformation of speed for normal distribution
- Cross-validation: 5-fold with temporal independence
- Feature scaling: StandardScaler normalization

### Feature Selection Process
All 14 environmental features retained for maximum predictive power:
1. elevation
2. ndvi
3. on_roads
4. dist_to_roads
5. on_trails
6. dist_to_trails
7. on_water
8. dist_to_water
9. rocky_terrain
10. firm_wetlands
11. landcover_barr__och_blandskog
12. landcover_öppen_mark
13. on_buildings
14. dist_to_buildings

### Missing Value Handling
**Strategy**: Domain-specific imputation
- Distance features: 100m (conservative estimate)
- Binary features: 0 (not present)
- Elevation: Median value from valid data
- NDVI: 0.3 (typical forest value)

## Cost Surface Generation

### Methodology
Movement cost calculated as inverse of predicted speed:
```python
log_speeds = model.predict(X_scaled)
speeds = np.exp(log_speeds)
costs = 1.0 / np.maximum(speeds, 0.1)  # Prevent division by zero
```

**Technical Implementation**:
- Parallel processing across 8 CPU cores
- Chunk-based processing for memory efficiency
- Full spatial resolution: 3,909 × 5,168 pixels
- Cell size: 5m × 5m
- Total coverage: ~100 km²

**Impassable Areas**:
- Water bodies: Cost = 999.0 (effectively infinite)
- Built structures: Cost = 999.0
- Detected using spatial overlay with feature geometries

### Output Specifications
**Format**: GeoTIFF  
**File**: `random_forest_cost_surface.tif`  
**Data Type**: Float32  
**Units**: seconds per meter  
**NoData Value**: NaN  
**Compression**: LZW  

## Least-Cost Path Analysis

### Algorithm: Dijkstra's Shortest Path
**Implementation**: scikit-image `route_through_array`  
**Cost Function**: Accumulated time cost  
**Path Constraint**: 8-connected grid (allows diagonal movement)

### Control Point Routing
**Sequence**: Start → Control 1 → Control 2 → ... → Control N → Finish  
**Path Segments**: 23 individual route segments  
**Optimization**: Each segment independently optimized  

**Technical Process**:
1. Convert control point coordinates to raster indices
2. Apply Dijkstra's algorithm between consecutive controls
3. Convert pixel paths back to geographic coordinates
4. Create LineString geometries for each segment
5. Calculate path statistics (length, total cost, average cost)

**Output Format**: GeoJSON with path geometries and attributes
```json
{
  "segment_id": "start_to_control_01",
  "length_m": 1250.5,
  "total_cost": 312.8,
  "avg_cost": 0.25,
  "geometry": "LINESTRING(...)"
}
```

## Model Validation and Performance

### Cross-Validation Metrics
**Method**: 5-fold cross-validation with stratified splits  
**Metrics Computed**:
- R² (coefficient of determination)
- RMSE (root mean square error)
- MAE (mean absolute error)
- Out-of-bag score (Random Forest internal validation)

### Feature Importance Analysis
**Method**: Random Forest built-in feature importance  
**Calculation**: Mean decrease in impurity across all trees  
**Ranking**: Normalized importance scores (sum = 1.0)

**Saved Output**: `feature_importance.csv`
```csv
predictor,importance
elevation,0.234
dist_to_trails,0.187
ndvi,0.156
...
```

### Model Artifacts
**Saved Models**:
- `random_forest_model.pkl`: Trained RandomForest model
- `feature_scaler.pkl`: StandardScaler for feature normalization
- `model_metrics.json`: Cross-validation performance metrics

## Visualization Framework

### Technical Architecture
**Framework**: matplotlib + seaborn + geopandas  
**Configuration**: Centralized styling in `viz_config.py`  
**Output Formats**: PNG (300 DPI) + PDF for publication quality  

### Color Schemes
**Terrain Classification**:
- Fast terrain: Light green (#90EE90)
- Medium terrain: Gold (#FFD700)  
- Slow terrain: Tomato (#FF6347)
- Impassable: Dark red (#8B0000)

**Speed Visualization** (Strava-style):
- Low speed: Red (#FF0000)
- Medium speed: Yellow (#FFFF00)
- High speed: Green (#00FF00)

### Figure Types Generated

#### 1. Terrain Difficulty Classification
**Description**: Cost surface discretized into 3 difficulty categories  
**Method**: Percentile-based thresholds (33rd, 67th percentiles)  
**Overlays**: Optimal paths + control points with annotations  
**Enhancement**: White text with black stroke for visibility  

#### 2. Speed Distribution Analysis
**Description**: Histogram of GPS-recorded speeds  
**Statistics**: Mean, median, quartiles, sample size  
**Filtering**: 0.5-25 km/h range (removes GPS errors)  
**Sample Size**: 18,171 GPS points after filtering  

#### 3. Feature Importance Ranking
**Description**: Horizontal bar chart of predictor importance  
**Color Coding**: 
- High importance (>80%): Red
- Medium importance (50-80%): Orange  
- Lower importance (<50%): Blue
**Features**: All 14 environmental predictors displayed

#### 4. Route Speed Visualization
**Description**: GPS tracks colored by instantaneous speed  
**Style**: Strava-inspired color gradient  
**Overlays**: Optimal least-cost paths + control points  
**Background**: Subtle cost surface as grayscale  

## Data Quality and Limitations

### GPS Data Quality
**Accuracy**: ±3-5m typical GPS precision  
**Temporal Resolution**: 1-5 second intervals  
**Missing Data**: ~2% of points filtered for speed anomalies  
**Coverage**: Uneven spatial distribution (higher density on trails)

### Environmental Data Currency
**DEM**: Current to 2023 (annual updates)  
**Land Cover**: NMD 2018-2023 (may not reflect recent changes)  
**NDVI**: Single date (2024-05-30) - seasonal variation not captured  
**OSM**: Community-contributed data - variable completeness  

### Model Limitations
**Temporal Factors**: Time-of-day effects not modeled  
**Athlete Variability**: Individual fitness/skill differences not captured  
**Weather**: Conditions during race not incorporated  
**Route Choice**: Strategic decisions beyond optimal path not modeled  

## Computational Requirements

### Hardware Specifications
**CPU**: Multi-core processing (8 cores utilized)  
**Memory**: 16GB+ RAM recommended for full-resolution processing  
**Storage**: ~10GB for complete dataset  
**Processing Time**: ~45 minutes for full pipeline  

### Software Dependencies
**Core Libraries**:
- Python 3.9+
- scikit-learn 1.6.1
- geopandas 1.0.1
- rasterio 1.4.3
- matplotlib 3.10.3
- numpy 2.2.6
- pandas 2.2.3

**Geospatial Tools**:
- GDAL 3.8+
- PyROSM 0.6.2
- shapely 2.1.1
- pyproj 3.7.1

## File Structure and Organization

```
SpatialAnalyticsExam/
├── data/
│   ├── raw/                    # Original downloaded data
│   │   ├── dem/               # Lantmäteriet elevation tiles
│   │   ├── gps/               # Original GPX files
│   │   ├── landcover/         # NMD raster data
│   │   ├── ndvi/              # Sentinel-2 imagery
│   │   └── osm/               # OpenStreetMap PBF
│   └── derived/               # Processed datasets
│       ├── csv/               # GPS tracks with features
│       ├── dem/               # DEM mosaics and derivatives
│       ├── rasters/           # Environmental feature rasters
│       └── map/               # Control points and race data
├── src/                       # Analysis scripts
│   ├── analysis/              # Machine learning models
│   ├── dem/                   # DEM processing
│   ├── osm/                   # OSM feature extraction
│   └── visualizations/        # Plotting and figures
├── output/                    # Results and outputs
│   ├── cost_surfaces/         # Model predictions
│   ├── figures/               # Generated visualizations
│   └── model_trace/           # Model artifacts
└── cache/                     # API response cache
```

## Quality Assurance and Validation

### Data Validation Steps
1. **Coordinate System Consistency**: All datasets verified in EPSG:3006
2. **Spatial Alignment**: Raster grids aligned using GDAL -tap option
3. **GPS Track Validation**: Speed filtering removes GPS errors
4. **Feature Completeness**: Missing value patterns analyzed
5. **Model Overfitting Checks**: Cross-validation with temporal splits

### Reproducibility Measures
- **Random Seeds**: Fixed seed (42) for all stochastic processes
- **Version Control**: All dependencies pinned in requirements.txt
- **Parameter Documentation**: All hyperparameters explicitly stated
- **Processing Logs**: Detailed console output for debugging

## Future Enhancements

### Technical Improvements
1. **Temporal Modeling**: Incorporate time-of-day effects
2. **Weather Integration**: Add meteorological variables
3. **Deep Learning**: Explore CNN/RNN architectures
4. **Multi-objective Optimization**: Balance speed vs. route complexity

### Data Enhancements
1. **Higher Resolution DEM**: 0.5m lidar data where available
2. **Multi-temporal NDVI**: Seasonal vegetation changes
3. **Crowd-sourced Validation**: Additional GPS tracks for testing
4. **Terrain Micro-features**: Rock formations, vegetation density

This technical documentation provides comprehensive details for replicating the analysis and understanding the methodological choices made throughout the project.
