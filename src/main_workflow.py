#!/usr/bin/env python3
"""
Main workflow script for spatial analytics orienteering route optimization.

This script orchestrates the complete data processing pipeline in the correct sequence:
1. convert_oringen_map.sh - Convert O-Ringen KMZ map to GeoTIFF format
2. dem_download.py - Download and process DEM data from Lantmäteriet
3. reproject_and_clip_all.py - Reproject and clip spatial data to EPSG:3006
4. extract_lantmateriet_features.py - Extract features from Swedish topographic data  
5. lantmateriet_to_rasters.py - Convert vector features to raster format
6. generate_track_features.py - Process GPS tracks and generate track features CSV
7. spatial_autocorrelation_analysis.py - Test for spatial autocorrelation in data
8. random_forest_model.py - Train random forest model for route optimization
9. least_cost_paths.py - Generate least cost paths using the trained model
10. random_forest_simplified.py - Create visualizations of results

Each step depends on the outputs of previous steps, so they must be run in order.
The script includes dependency checking and error handling between pipeline steps.

Usage:
    python src/main_workflow.py [--skip-map] [--skip-dem] [--skip-reproject] [--skip-features] [--skip-rasters] [--skip-tracks] [--skip-autocorr] [--skip-model] [--skip-paths] [--skip-viz]
    
    Note: This script must be run from the project root directory.
    
Options:
    --skip-map          Skip the O-Ringen map conversion step
    --skip-dem          Skip the DEM download and processing step
    --skip-reproject    Skip the reprojection step (assumes already done)
    --skip-features     Skip the feature extraction step
    --skip-rasters      Skip the raster conversion step  
    --skip-tracks       Skip the track processing step
    --skip-autocorr     Skip the spatial autocorrelation analysis step
    --skip-model        Skip the random forest model training step
    --skip-paths        Skip the least cost paths generation step
    --skip-viz          Skip the visualization generation step
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_file_exists(filepath, description=""):
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        print(f"✓ {description or filepath} exists")
        return True
    else:
        print(f"✗ {description or filepath} not found: {filepath}")
        return False

def check_directory_exists(dirpath, description=""):
    """Check if a directory exists and print status."""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"✓ {description or dirpath} directory exists")
        return True
    else:
        print(f"✗ {description or dirpath} directory not found: {dirpath}")
        return False

def run_shell_script(script_path, description, dependencies=None):
    """
    Run a shell script with dependency checking and error handling.
    
    Args:
        script_path: Path to the shell script to run
        description: Human-readable description of what the script does
        dependencies: List of files/directories that must exist before running
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    # Check if script exists
    if not check_file_exists(script_path, f"Script {script_path}"):
        return False
    
    # Check dependencies
    if dependencies:
        print(f"\nChecking dependencies for {description}:")
        missing_deps = []
        for dep in dependencies:
            if isinstance(dep, dict):
                dep_path = dep['path']
                dep_desc = dep.get('description', dep_path)
                dep_required = dep.get('required', True)
            else:
                dep_path = dep
                dep_desc = dep
                dep_required = True
                
            if Path(dep_path).is_dir():
                exists = check_directory_exists(dep_path, dep_desc)
            else:
                exists = check_file_exists(dep_path, dep_desc)
                
            if not exists and dep_required:
                missing_deps.append(dep_path)
        
        if missing_deps:
            print(f"\n❌ Cannot run {description} - missing required dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            return False
        
        print("✓ All dependencies satisfied")
    
    # Run the script
    print(f"\n🚀 Running {description}...")
    start_time = time.time()
    
    try:
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the shell script in the project root directory
        result = subprocess.run(
            ["zsh", script_path],
            cwd=Path(__file__).parent.parent,  # Go up to project root
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {description} completed successfully in {duration:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with return code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def run_script(script_path, description, dependencies=None):
    """
    Run a Python script with dependency checking and error handling.
    
    Args:
        script_path: Path to the script to run
        description: Human-readable description of what the script does
        dependencies: List of files/directories that must exist before running
    
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    # Check if script exists
    if not check_file_exists(script_path, f"Script {script_path}"):
        return False
    
    # Check dependencies
    if dependencies:
        print(f"\nChecking dependencies for {description}:")
        missing_deps = []
        for dep in dependencies:
            if isinstance(dep, dict):
                dep_path = dep['path']
                dep_desc = dep.get('description', dep_path)
                dep_required = dep.get('required', True)
            else:
                dep_path = dep
                dep_desc = dep
                dep_required = True
                
            if Path(dep_path).is_dir():
                exists = check_directory_exists(dep_path, dep_desc)
            else:
                exists = check_file_exists(dep_path, dep_desc)
                
            if not exists and dep_required:
                missing_deps.append(dep_path)
        
        if missing_deps:
            print(f"\n❌ Cannot run {description} - missing required dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            return False
        
        print("✓ All dependencies satisfied")
    
    # Run the script
    print(f"\n🚀 Running {description}...")
    start_time = time.time()
    
    try:
        # Run the script in the project root directory
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=Path(__file__).parent.parent,  # Go up to project root
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {description} completed successfully in {duration:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with return code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def create_output_directories():
    """Create necessary output directories."""
    directories = [
        "data/derived/dem",
        "data/derived/map", 
        "data/derived/gps",
        "data/derived/ndvi",
        "data/derived/lantmateriet",
        "data/derived/rasters",
        "data/derived/csv"
    ]
    
    print("Creating output directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}")

def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(
        description="Run the complete spatial analytics data processing pipeline"
    )
    parser.add_argument(
        "--skip-map", 
        action="store_true",
        help="Skip the O-Ringen map conversion step"
    )
    parser.add_argument(
        "--skip-dem", 
        action="store_true",
        help="Skip the DEM download and processing step"
    )
    parser.add_argument(
        "--skip-reproject", 
        action="store_true",
        help="Skip the reprojection step (assumes already done)"
    )
    parser.add_argument(
        "--skip-features", 
        action="store_true",
        help="Skip the feature extraction step"
    )
    parser.add_argument(
        "--skip-rasters", 
        action="store_true",
        help="Skip the raster conversion step"
    )
    parser.add_argument(
        "--skip-tracks", 
        action="store_true",
        help="Skip the track processing step"
    )
    parser.add_argument(
        "--skip-autocorr", 
        action="store_true",
        help="Skip the spatial autocorrelation analysis step"
    )
    parser.add_argument(
        "--skip-model", 
        action="store_true",
        help="Skip the random forest model training step"
    )
    parser.add_argument(
        "--skip-paths", 
        action="store_true",
        help="Skip the least cost paths generation step"
    )
    parser.add_argument(
        "--skip-viz", 
        action="store_true",
        help="Skip the visualization generation step"
    )
    
    args = parser.parse_args()
    
    print("🌲 SPATIAL ANALYTICS ORIENTEERING ROUTE OPTIMIZATION 🏃")
    print("="*70)
    print("Main data processing pipeline")
    print("This will run the complete data processing sequence:")
    print("1. Convert O-Ringen KMZ map to GeoTIFF format")
    print("2. Download and process DEM data from Lantmäteriet")
    print("3. Reproject and clip all spatial data to EPSG:3006")
    print("4. Extract features from Lantmäteriet topographic data")
    print("5. Convert vector features to raster format")  
    print("6. Generate track features CSV from GPS data")
    print("7. Perform spatial autocorrelation analysis")
    print("8. Train random forest model for route optimization")
    print("9. Generate least cost paths using the trained model") 
    print("10. Create visualizations of results")
    print("="*70)
    
    # Create output directories
    create_output_directories()
    
    # Track overall success
    pipeline_success = True
    failed_steps = []
    
    # Step 1: Convert O-Ringen map
    if not args.skip_map:
        dependencies = [
            {"path": "data/raw/map", "description": "Map data directory", "required": True},
            {"path": "data/raw/map/oringen_e4_2024_h21elit.kmz", 
             "description": "O-Ringen KMZ file", "required": False}
        ]
        
        success = run_shell_script(
            "src/data/convert_oringen_map.sh",
            "Convert O-Ringen KMZ map to GeoTIFF format",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Map conversion")
    else:
        print("\n⏭️  Skipping map conversion step (--skip-map)")
    
    # Step 2: Download and process DEM data
    if not args.skip_dem and (pipeline_success or args.skip_map):
        dependencies = [
            {"path": "data/raw", "description": "Raw data directory", "required": True}
        ]
        
        success = run_script(
            "src/data/dem_download.py",
            "Download and process DEM data from Lantmäteriet", 
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("DEM download")
    else:
        if args.skip_dem:
            print("\n⏭️  Skipping DEM download step (--skip-dem)")
        else:
            print("\n⏭️  Skipping DEM download due to previous failures")
    
    # Step 3: Reproject and clip all data
    if not args.skip_reproject and (pipeline_success or args.skip_map or args.skip_dem):
        dependencies = [
            {"path": "data/raw", "description": "Raw data directory", "required": True},
            {"path": "data/raw/dem", "description": "DEM data directory", "required": False},
            {"path": "data/raw/map", "description": "Map data directory", "required": False},
            {"path": "data/raw/gps", "description": "GPS data directory", "required": False},
            {"path": "data/raw/lantmateriet", "description": "Lantmäteriet data directory", "required": False}
        ]
        
        success = run_script(
            "src/data/reproject_and_clip_all.py",
            "Reproject and clip all spatial data to EPSG:3006",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Reprojection")
    else:
        print("\n⏭️  Skipping reprojection step (--skip-reproject)")
    
    # Step 4: Extract Lantmäteriet features  
    if not args.skip_features and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject):
        dependencies = [
            {"path": "data/raw/lantmateriet", "description": "Lantmäteriet raw data directory"},
            {"path": "data/derived/map/oringen_e4_2024_h21elit_REFERENCED_3006.tif", 
             "description": "Reference raster (from reprojection step)"}
        ]
        
        success = run_script(
            "src/data/extract_lantmateriet_features.py",
            "Extract features from Lantmäteriet topographic data",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Feature extraction")
    else:
        if args.skip_features:
            print("\n⏭️  Skipping feature extraction step (--skip-features)")
        else:
            print("\n⏭️  Skipping feature extraction due to previous failures")
    
    # Step 5: Convert vectors to rasters
    if not args.skip_rasters and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features):
        dependencies = [
            {"path": "data/derived/map/oringen_e4_2024_h21elit_REFERENCED_3006.tif", 
             "description": "Reference raster"},
            {"path": "data/derived/lantmateriet", "description": "Lantmäteriet features directory"}
        ]
        
        success = run_script(
            "src/data/lantmateriet_to_rasters.py", 
            "Convert vector features to raster format",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Raster conversion")
    else:
        if args.skip_rasters:
            print("\n⏭️  Skipping raster conversion step (--skip-rasters)")
        else:
            print("\n⏭️  Skipping raster conversion due to previous failures")
    
    # Step 6: Generate track features
    if not args.skip_tracks and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features or args.skip_rasters):
        dependencies = [
            {"path": "data/derived/gps", "description": "GPS tracks directory"},
            {"path": "data/derived/map/control_points_race_3006.shp", 
             "description": "Control points shapefile"}
        ]
        
        success = run_script(
            "src/data/generate_track_features.py",
            "Generate track features CSV from GPS data", 
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Track processing")
    else:
        if args.skip_tracks:
            print("\n⏭️  Skipping track processing step (--skip-tracks)")
        else:
            print("\n⏭️  Skipping track processing due to previous failures")
    
    # Step 7: Spatial autocorrelation analysis
    if not args.skip_autocorr and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features or args.skip_rasters or args.skip_tracks):
        dependencies = [
            {"path": "data/derived/csv/track_features.csv", "description": "Track features CSV file"},
            {"path": "data/derived/rasters", "description": "Raster features directory"},
            {"path": "data/derived/gps", "description": "GPS tracks directory"}
        ]
        
        success = run_script(
            "src/analysis/spatial_autocorrelation_analysis.py",
            "Perform spatial autocorrelation analysis on GPS tracks and environmental data",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Spatial autocorrelation analysis")
    else:
        if args.skip_autocorr:
            print("\n⏭️  Skipping spatial autocorrelation analysis step (--skip-autocorr)")
        else:
            print("\n⏭️  Skipping spatial autocorrelation analysis due to previous failures")
    
    # Step 8: Train random forest model
    if not args.skip_model and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features or args.skip_rasters or args.skip_tracks or args.skip_autocorr):
        dependencies = [
            {"path": "data/derived/csv/track_features.csv", "description": "Track features CSV file"},
            {"path": "data/derived/rasters", "description": "Raster features directory"}
        ]
        
        success = run_script(
            "src/analysis/random_forest_model.py",
            "Train random forest model for route optimization",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Model training")
    else:
        if args.skip_model:
            print("\n⏭️  Skipping model training step (--skip-model)")
        else:
            print("\n⏭️  Skipping model training due to previous failures")
    
    # Step 9: Generate least cost paths
    if not args.skip_paths and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features or args.skip_rasters or args.skip_tracks or args.skip_autocorr or args.skip_model):
        dependencies = [
            {"path": "data/derived/rasters", "description": "Raster features directory"},
            {"path": "data/derived/map/control_points_race_3006.shp", "description": "Control points shapefile"}
        ]
        
        success = run_script(
            "src/analysis/least_cost_paths.py",
            "Generate least cost paths using the trained model",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Path generation")
    else:
        if args.skip_paths:
            print("\n⏭️  Skipping path generation step (--skip-paths)")
        else:
            print("\n⏭️  Skipping path generation due to previous failures")
    
    # Step 10: Create visualizations
    if not args.skip_viz and (pipeline_success or args.skip_map or args.skip_dem or args.skip_reproject or args.skip_features or args.skip_rasters or args.skip_tracks or args.skip_autocorr or args.skip_model or args.skip_paths):
        dependencies = [
            {"path": "data/derived/rasters", "description": "Raster features directory"},
            {"path": "data/derived/csv/track_features.csv", "description": "Track features CSV file"}
        ]
        
        success = run_script(
            "src/visualizations/random_forest_simplified.py",
            "Create visualizations of results",
            dependencies
        )
        
        if not success:
            pipeline_success = False
            failed_steps.append("Visualization")
    else:
        if args.skip_viz:
            print("\n⏭️  Skipping visualization step (--skip-viz)")
        else:
            print("\n⏭️  Skipping visualization due to previous failures")
    
    # Final summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    
    if pipeline_success and not any([args.skip_map, args.skip_dem, args.skip_reproject, args.skip_features, args.skip_rasters, args.skip_tracks, args.skip_autocorr, args.skip_model, args.skip_paths, args.skip_viz]):
        print("🎉 Complete pipeline executed successfully!")
        print("\nNext steps:")
        print("- Review generated files in data/derived/")
        print("- Check analysis outputs and trained models")
        print("- Review spatial autocorrelation analysis results")
        print("- Review generated visualizations")
        print("- Open QGIS projects to view spatial results")
    elif failed_steps:
        print(f"❌ Pipeline completed with {len(failed_steps)} failed step(s):")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease review error messages above and fix issues before proceeding.")
    else:
        print("✅ Pipeline completed (some steps were skipped)")
        
    print(f"\nGenerated outputs should be in:")
    print("- data/derived/lantmateriet/ (vector features)")
    print("- data/derived/rasters/ (raster features)")  
    print("- data/derived/csv/track_features.csv (GPS track data)")
    print("- output/ (analysis results and visualizations)")
    print("- spatial_autocorrelation_report.txt (autocorrelation analysis)")
    print("- *.pkl (trained models)")
    print("- *.png (generated visualizations)")

if __name__ == "__main__":
    main()
