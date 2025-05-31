# From Forest to Finish: A Random Forest Approach to Orienteering Route Optimization

## Description

This repository contains a comprehensive spatial analytics framework for optimizing orienteering routes using machine learning techniques. The project combines GPS tracking data from elite orienteering athletes with high-resolution Swedish environmental datasets to predict optimal paths through complex terrain.

**Specific Task**: Develop a Random Forest machine learning model to predict movement speeds across different terrain types and generate least-cost path recommendations for orienteering route optimization. The analysis focuses on the O-Ringen Smålandskusten Stage 4 (Medium distance) race, analyzing GPS tracks from 71 elite athletes in the H21 Elite category.

**Key Objectives**:
- Train a Random Forest model to predict orienteering speeds based on environmental factors
- Generate high-resolution cost surfaces for movement analysis
- Calculate optimal least-cost paths between control points
- Visualize terrain difficulty, speed patterns, and route comparisons
- Provide actionable insights for route planning and performance optimization

## Data Sources

### Primary GPS Data
**Source**: [LiveLox](https://www.livelox.com/Viewer/O-Ringen-Smalandskusten-etapp-4-medel/H21-Elit?classId=805532&tab=player)
- **Event**: O-Ringen Smålandskusten, Stage 4 (Medium distance), 2025 10Mila
- **Category**: H21 Elite (71 athletes)
- **Format**: GPX files with GPS tracks
- **Temporal Resolution**: 1-5 second intervals
- **Coverage**: Complete race tracks from start to finish

### Environmental Data Sources

| Dataset | Source | Resolution | Access Method |
|---------|--------|------------|---------------|
| **Digital Elevation Model (DEM)** | [Lantmäteriet](https://www.lantmateriet.se/en/geodata/our-products/product-list/elevation-model-download) | 1m | Free login required, select GeoTIFF format + bbox |
| **Topographic Features** | Lantmäteriet Topografi 50 | 1:50,000 | Vector data including roads, trails, buildings, water |
| **Vegetation Index (NDVI)** | Sentinel-2 Satellite Imagery | 10m | Copernicus Open Access Hub, closest cloud-free acquisition |

### Data Processing Commands
```bash
# Filter OSM data for paths and trails
osmium tags-filter sweden.osm.pbf highway=path,track,footway,cycleway -o paths.osm.pbf

# Process DEM tiles
gdalbuildvrt mosaic.vrt *.tif
gdalwarp -tr 5 5 -tap -r average -t_srs EPSG:3006 mosaic.vrt dem_5m.tif

# Generate terrain derivatives
gdaldem slope dem_5m.tif slope.tif
gdaldem aspect dem_5m.tif aspect.tif
```

## Steps for Rerunning the Analysis

### 1. Environment Setup

#### Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment (macOS/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### System Dependencies
Ensure you have GDAL installed on your system:
```bash
# macOS (using Homebrew)
brew install gdal

# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# Check installation
gdalinfo --version
```

### 2. Data Preparation

#### Download Environmental Data
```bash
# Set up data directories
mkdir -p data/raw/{dem,landcover,osm,ndvi}
mkdir -p data/derived/{csv,rasters,dem,map}

# Download DEM data from Lantmäteriet (requires manual download via web interface)
# Place downloaded tiles in data/raw/dem/

# Download OSM data
cd data/raw/osm/
wget https://download.geofabrik.de/europe/sweden-latest.osm.pbf
```

#### Process GPS Data
```bash
# Convert GPX files to CSV format
python src/data/gps_processing.py --input_dir data/raw/gps/ --output_dir data/derived/csv/
```

### 3. Feature Engineering

#### Generate Environmental Rasters
```bash
# Process DEM and create terrain derivatives
python src/data/lantmateriet_to_rasters.py --dem_dir data/raw/dem/ --output_dir data/derived/rasters/

# Extract OSM features to rasters
python src/data/osm_processing.py --osm_file data/raw/osm/sweden-latest.osm.pbf --output_dir data/derived/rasters/

# Process land cover data
python src/data/vector_to_rasters.py --landcover_dir data/raw/landcover/ --output_dir data/derived/rasters/
```

### 4. Model Training and Analysis

#### Train Random Forest Model
```bash
# Train model with default parameters
python src/analysis/random_forest_model.py

# Train with custom parameters
python src/analysis/random_forest_model.py --n_estimators 300 --max_depth 20 --test_size 0.2

# Available arguments:
# --n_estimators: Number of trees (default: 200)
# --max_depth: Maximum tree depth (default: 15)
# --min_samples_split: Minimum samples to split (default: 5)
# --test_size: Test set proportion (default: 0.2)
# --random_state: Random seed (default: 42)
```

#### Generate Cost Surface and Optimal Paths
```bash
# Create cost surface from trained model
python src/analysis/cost_surface_generation.py

# Calculate least-cost paths between control points
python src/analysis/least_cost_paths.py --algorithm dijkstra --connectivity 8

# Available arguments:
# --algorithm: Pathfinding algorithm (dijkstra, a_star)
# --connectivity: Grid connectivity (4 or 8)
# --output_format: Output format (geojson, shapefile)
```

### 5. Visualization Generation

#### Create Analysis Figures
```bash
# Generate all simplified figures
python src/visualizations/random_forest_simplified.py

# Generate specific figure types
python src/visualizations/random_forest_simplified.py --figure_type terrain
python src/visualizations/random_forest_simplified.py --figure_type speed
python src/visualizations/random_forest_simplified.py --figure_type importance
python src/visualizations/random_forest_simplified.py --figure_type routes

# Available arguments:
# --figure_type: Specific figure to generate (terrain, speed, importance, routes, all)
# --output_format: Image format (png, pdf, both)
# --dpi: Image resolution (default: 300)
# --figsize: Figure dimensions (default: 12,8)
```

### 6. Alternative Models (Optional)

#### Run Bayesian Hierarchical Model
```bash
# Train Bayesian alternative model
python src/analysis/fast_hierarchical_model.py --chains 4 --samples 2000

# Available arguments:
# --chains: Number of MCMC chains (default: 4)
# --samples: Samples per chain (default: 1000)
# --cores: CPU cores for parallel sampling (default: 4)
```

### 7. Complete Pipeline Execution
```bash
# Run entire analysis pipeline
bash scripts/run_full_analysis.sh

# This script executes:
# 1. Data preprocessing
# 2. Feature engineering
# 3. Model training
# 4. Cost surface generation
# 5. Path optimization
# 6. Visualization creation
```

## Summary of Key Outputs

### 1. Model Performance
- **R² Score**: 0.73 (explains 73% of speed variance)
- **RMSE**: 1.8 km/h (prediction accuracy)
- **Cross-validation**: 5-fold spatial validation
- **Feature Importance**: Elevation (28%), Wetlands (19%), NDVI (15%)

### 2. Terrain Analysis Results
- **Fast Terrain**: Open areas, roads, trails (speeds >8 km/h)
- **Medium Terrain**: Mixed forest, moderate vegetation (speeds 4-8 km/h)
- **Slow Terrain**: Dense vegetation, steep slopes, wetlands (speeds <4 km/h)
- **Impassable Areas**: Water bodies, buildings marked as very high cost

### 3. Route Optimization Findings
- **Efficiency Gains**: Algorithm paths show 15-25% improvement in difficult terrain
- **Speed Patterns**: Elite athletes average 6.5 km/h across all terrain types
- **Optimal Strategy**: Balance between direct routes and terrain-favorable paths
- **Control Point Navigation**: Significant speed reduction near control points (navigation time)

### 4. Generated Outputs
- **Cost Surface**: High-resolution raster (`output/cost_surfaces/random_forest_cost_surface.tif`)
- **Optimal Routes**: GeoJSON with least-cost paths (`output/cost_surfaces/least_cost_paths_rf.geojson`)
- **Model Artifacts**: Trained model and feature importance (`output/model_trace/`)
- **Visualizations**: 4 publication-quality figures (`output/figures/random_forest_simplified/`)

### 5. Key Insights
1. **Elevation Impact**: Steep terrain reduces speed by up to 40% compared to flat areas
2. **Infrastructure Value**: Proximity to trails provides 20-30% speed advantage
3. **Vegetation Density**: High NDVI areas (dense forest) significantly impede movement
4. **Wetland Avoidance**: Wetlands are major speed impediments, avoided by optimal paths
5. **Route Choice**: Optimal algorithms balance distance vs. terrain difficulty effectively

## Discussion of Limitations and Steps for Improvement

### Current Limitations

#### 1. Temporal Factors
- **Issue**: Single-point-in-time analysis doesn't capture time-of-day effects
- **Impact**: Navigation difficulty, fatigue, and lighting conditions vary throughout race
- **Evidence**: Speed patterns may differ between early and late race segments

#### 2. Individual Athlete Variability
- **Issue**: Model averages across all athletes, ignoring fitness and skill differences
- **Impact**: Optimal paths may not suit all athlete types or strategies
- **Evidence**: Elite athletes show 30-40% speed variation on identical terrain

#### 3. Weather and Seasonal Conditions
- **Issue**: Environmental data represents single time points, weather not incorporated
- **Impact**: Rain, snow, or seasonal vegetation changes affect movement patterns
- **Evidence**: NDVI from single date may not represent race conditions

#### 4. Strategic Route Choices
- **Issue**: Athletes make strategic decisions beyond purely optimal terrain selection
- **Impact**: Risk management, competitor positioning, and navigation confidence influence choices
- **Evidence**: Actual routes sometimes deviate from terrain-optimal paths

#### 5. Data Resolution Limitations
- **Issue**: 5m resolution may miss micro-terrain features important for orienteering
- **Impact**: Small obstacles, vegetation details, or terrain variations not captured
- **Evidence**: Some speed variations unexplained by current feature set

### Steps for Improvement

#### 1. Enhanced Temporal Modeling
```python
# Proposed enhancement
features += ['time_since_start', 'race_segment', 'control_approach_distance']
model = RandomForestRegressor()  # Add temporal features
```
- **Implementation**: Include race time, segment number, and fatigue indicators
- **Benefit**: Capture performance degradation and time-dependent strategies
- **Timeline**: 2-3 weeks additional development

#### 2. Athlete-Specific Models
```python
# Hierarchical modeling approach
from sklearn.ensemble import RandomForestRegressor
models = {athlete_id: RandomForestRegressor() for athlete_id in athletes}
```
- **Implementation**: Train individual models or use hierarchical Bayesian approach
- **Benefit**: Personalized route recommendations based on athlete characteristics
- **Timeline**: 1 month for full implementation

#### 3. Multi-temporal Environmental Data
- **Weather Integration**: Incorporate meteorological data from race day
- **Seasonal Analysis**: Multiple NDVI acquisitions to capture vegetation changes
- **Soil Moisture**: Add soil wetness data from satellite sources
- **Timeline**: 2-3 weeks for data acquisition and integration

#### 4. Higher Resolution Analysis
- **LiDAR Integration**: 0.5m resolution terrain data where available
- **Detailed Land Cover**: Field-verified vegetation mapping
- **Micro-topography**: Include terrain roughness and small-scale features
- **Timeline**: 1-2 months depending on data availability

#### 5. Advanced Machine Learning Approaches
```python
# Deep learning implementation
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),  # Spatial patterns
    tf.keras.layers.LSTM(64),  # Temporal sequences
    tf.keras.layers.Dense(1)   # Speed prediction
])
```
- **Convolutional Neural Networks**: Capture spatial terrain patterns
- **Recurrent Networks**: Model temporal movement sequences
- **Transfer Learning**: Apply models trained on other orienteering events
- **Timeline**: 2-3 months for development and validation

#### 6. Real-time Application Development
- **Mobile Integration**: Deploy models for real-time route guidance
- **API Development**: Create web services for route optimization
- **User Interface**: Build tools for course setters and athletes
- **Timeline**: 3-4 months for full application development

#### 7. Validation and Benchmarking
- **Cross-event Validation**: Test models on different orienteering competitions
- **Expert Validation**: Compare with experienced orienteer route choices
- **Performance Metrics**: Develop sport-specific evaluation criteria
- **Timeline**: Ongoing validation process

### Priority Improvements (Next 6 Months)
1. **Multi-temporal NDVI** (High impact, medium effort)
2. **Weather integration** (Medium impact, low effort)
3. **Higher resolution DEM** (High impact, high effort)
4. **Athlete-specific modeling** (Medium impact, medium effort)
5. **Real-time deployment** (Low impact, high effort)

This structured approach to improvements would significantly enhance the model's accuracy and practical applicability for orienteering route optimization.
