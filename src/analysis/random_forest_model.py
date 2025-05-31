#!/usr/bin/env python3
"""
Random Forest Model for Spatial Analytics
Full-resolution machine learning approach using scikit-learn's RandomForestRegressor.
Maintains the same structure as the Bayesian model with maximum accuracy.

Key features:
1. Random Forest regression for speed prediction
2. Feature importance analysis
3. Cross-validation with multiple metrics
4. Full-resolution cost surface generation
5. Model trace and coefficient saving
6. Uses all available data without subsampling
"""

import pandas as pd
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
import warnings
import os
import multiprocessing as mp
from multiprocessing import Pool
import time
import joblib
import pickle
from functools import partial
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Optimize for speed - use more cores for cost surface generation
N_CORES = min(mp.cpu_count(), 8)  # Allow up to 8 cores
print(f"Using {N_CORES} cores for parallel processing")

def prepare_track_data(track_df):
    """
    Prepare track data using all available tracks
    """
    if 'track_id' not in track_df.columns:
        track_df['track_id'] = track_df['runner_id']
    
    # Use all tracks - no limiting for maximum performance
    unique_tracks = track_df['track_id'].unique()
    print(f"  Using all {len(unique_tracks)} tracks")
    
    n_tracks = len(unique_tracks)
    track_mapping = {track_id: i for i, track_id in enumerate(unique_tracks)}
    track_df['track_num'] = track_df['track_id'].map(track_mapping)
    
    return track_df, n_tracks, track_mapping

def fit_random_forest_model(X, y, n_estimators=200, max_depth=15):
    """
    Fit Random Forest model for speed prediction
    """
    # Random Forest with optimized parameters for speed
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=N_CORES,
        random_state=42,
        bootstrap=True,
        oob_score=True  # Out-of-bag score for internal validation
    )
    
    print(f"  Training Random Forest with {n_estimators} trees, max_depth={max_depth}")
    start_time = time.time()
    
    model.fit(X, y)
    
    training_time = time.time() - start_time
    print(f"  Model trained in {training_time:.1f} seconds")
    print(f"  Out-of-bag R²: {model.oob_score_:.3f}")
    
    return model

def cross_validate_rf(X, y, n_folds=5, n_estimators=200):
    """
    Cross-validation for Random Forest model using full-sized models
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'r2': [], 'rmse': [], 'mae': [], 'oob_score': []
    }
    
    print(f"Cross-validation: {n_folds} folds, {n_estimators} trees each")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        start_time = time.time()
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        try:
            # Fit full-sized model for CV
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=N_CORES,
                random_state=42,
                bootstrap=True,
                oob_score=True
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_pred_exp = np.exp(y_pred)
            y_val_exp = np.exp(y_val)
            
            # Metrics
            r2_val = r2_score(y_val_exp, y_pred_exp)
            rmse_val = np.sqrt(mean_squared_error(y_val_exp, y_pred_exp))
            mae_val = mean_absolute_error(y_val_exp, y_pred_exp)
            oob_score = model.oob_score_
            
            cv_results['r2'].append(r2_val)
            cv_results['rmse'].append(rmse_val)
            cv_results['mae'].append(mae_val)
            cv_results['oob_score'].append(oob_score)
            
            print(f"    R²: {r2_val:.3f}, RMSE: {rmse_val:.3f}, OOB: {oob_score:.3f}")
            
        except Exception as e:
            print(f"    Fold failed: {e}")
            cv_results['r2'].append(0.0)
            cv_results['rmse'].append(999.0)
            cv_results['mae'].append(999.0)
            cv_results['oob_score'].append(0.0)
        
        fold_time = time.time() - start_time
        print(f"    Completed in {fold_time:.1f} seconds")
    
    return cv_results

def select_top_features(predictor_data, predictor_names, y):
    """
    Use all available features - no feature limiting for maximum performance
    """
    print(f"  Using all {len(predictor_names)} available features")
    for name in predictor_names:
        data = predictor_data[name]
        try:
            corr = abs(np.corrcoef(data, y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        except:
            corr = 0.0
        print(f"    {name}: correlation = {corr:.3f}")
    
    return predictor_names

def process_cost_chunk(chunk_info):
    """
    Process a single chunk of the cost surface in parallel
    """
    (row_start, row_end, col_start, col_end, new_transform, 
     predictor_paths, predictor_names, model_path, scaler_path, 
     impassable_gdf_path, ref_crs) = chunk_info
    
    # Load model and scaler in each process
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load impassable areas if available
    impassable_gdf = None
    if impassable_gdf_path and os.path.exists(impassable_gdf_path):
        try:
            impassable_gdf = gpd.read_file(impassable_gdf_path)
        except:
            pass
    
    # Get coordinates for this chunk
    rows, cols = np.meshgrid(
        np.arange(row_start, row_end),
        np.arange(col_start, col_end),
        indexing='ij'
    )
    
    xs, ys = rasterio.transform.xy(new_transform, rows.flatten(), cols.flatten())
    coords = list(zip(xs, ys))
    
    # Extract features
    chunk_features = []
    
    for name in predictor_names:
        if name in predictor_paths:
            try:
                with rasterio.open(predictor_paths[name]) as src:
                    values = list(src.sample(coords))
                    values = [v[0] if len(v) > 0 else 0.0 for v in values]
                    
                    # Simple missing value handling
                    values = [0.0 if np.isnan(v) else v for v in values]
                    chunk_features.append(values)
            except:
                chunk_features.append([0.0] * len(coords))
        else:
            chunk_features.append([0.0] * len(coords))
    
    if not chunk_features:
        return (row_start, row_end, col_start, col_end, 
                np.full((row_end - row_start, col_end - col_start), np.nan), 0)
    
    chunk_X = np.column_stack(chunk_features)
    chunk_X_scaled = scaler.transform(chunk_X)
    
    # Predict log speeds using Random Forest
    log_speeds = model.predict(chunk_X_scaled)
    speeds = np.exp(log_speeds)
    
    # Convert to costs
    costs = 1.0 / np.maximum(speeds, 0.1)
    
    # Track impassable pixels
    impassable_pixels = 0
    
    # Apply impassable areas - set to extremely high cost
    if impassable_gdf is not None:
        points = [Point(x, y) for x, y in coords]
        points_gdf = gpd.GeoDataFrame({'geometry': points}, crs=ref_crs)
        
        # Check which points are in impassable areas
        for idx, area in impassable_gdf.iterrows():
            within_mask = points_gdf.geometry.within(area.geometry)
            if within_mask.any():
                # Set impassable areas to very high cost (essentially infinite)
                costs[within_mask] = 999.0
                impassable_pixels += within_mask.sum()
    
    # Reshape costs
    costs_reshaped = costs.reshape(row_end - row_start, col_end - col_start)
    
    return (row_start, row_end, col_start, col_end, costs_reshaped, impassable_pixels)

def create_rf_cost_surface(model, predictor_paths, reference_raster, predictor_names, scaler):
    """
    Create cost surface using Random Forest predictions at full resolution with parallel processing
    """
    print("  Creating Random Forest cost surface at full resolution with parallel processing...")
    
    # Load impassable areas
    print("    Loading impassable areas...")
    impassable_gdf_path = None
    try:
        impassable_gdf = gpd.read_file("data/derived/map/impassible_areas.shp")
        print(f"    Found {len(impassable_gdf)} impassable areas")
        impassable_gdf_path = "data/derived/map/impassible_areas.shp"
        
        # Print summary of impassable areas
        if len(impassable_gdf) > 0:
            total_area = 0
            for idx, area in impassable_gdf.iterrows():
                area_size = area.geometry.area
                total_area += area_size
                print(f"      Area {idx+1}: {area.get('type', 'unknown')} ({area_size/1000000:.2f} km²)")
            print(f"    Total impassable area: {total_area/1000000:.2f} km²")
        
    except Exception as e:
        print(f"    Warning: Could not load impassable areas: {e}")
        impassable_gdf = None
    
    with rasterio.open(reference_raster) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = ref.shape
        
        # Use full resolution - no downsampling
        new_height = ref_shape[0]
        new_width = ref_shape[1]
        new_transform = ref_transform
    
    print(f"    Processing {new_height}x{new_width} pixels (full resolution)")
    print(f"    Using {N_CORES} cores for parallel processing")
    
    # Save model and scaler temporarily for multiprocessing
    temp_model_path = "temp_rf_model.pkl"
    temp_scaler_path = "temp_rf_scaler.pkl"
    joblib.dump(model, temp_model_path)
    joblib.dump(scaler, temp_scaler_path)
    
    # Create chunks for parallel processing
    chunk_size = 300  # Larger chunks for parallel processing
    chunk_list = []
    
    for row_start in range(0, new_height, chunk_size):
        row_end = min(row_start + chunk_size, new_height)
        for col_start in range(0, new_width, chunk_size):
            col_end = min(col_start + chunk_size, new_width)
            
            chunk_info = (row_start, row_end, col_start, col_end, new_transform,
                         predictor_paths, predictor_names, temp_model_path, temp_scaler_path,
                         impassable_gdf_path, ref_crs)
            chunk_list.append(chunk_info)
    
    print(f"    Processing {len(chunk_list)} chunks in parallel...")
    
    # Process chunks in parallel
    cost_data = np.full((new_height, new_width), np.nan, dtype=np.float32)
    total_impassable_pixels = 0
    
    start_time = time.time()
    
    try:
        with Pool(processes=N_CORES) as pool:
            # Process chunks in parallel with progress bar
            print("    Progress:")
            results = []
            with tqdm(total=len(chunk_list), desc="    Processing chunks", 
                     unit="chunk", ncols=80, bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                
                # Submit all jobs
                async_results = [pool.apply_async(process_cost_chunk, (chunk_info,)) for chunk_info in chunk_list]
                
                # Collect results with progress updates
                for async_result in async_results:
                    result = async_result.get()
                    results.append(result)
                    pbar.update(1)
            
            # Assemble results
            for row_start, row_end, col_start, col_end, costs_chunk, impassable_pixels in results:
                cost_data[row_start:row_end, col_start:col_end] = costs_chunk
                total_impassable_pixels += impassable_pixels
                
        processing_time = time.time() - start_time
        print(f"    Parallel processing completed in {processing_time:.1f} seconds")
        
    except Exception as e:
        print(f"    Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing with progress bar
        print("    Sequential processing progress:")
        for chunk_info in tqdm(chunk_list, desc="    Processing chunks", 
                              unit="chunk", ncols=80, bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            row_start, row_end, col_start, col_end, costs_chunk, impassable_pixels = process_cost_chunk(chunk_info)
            cost_data[row_start:row_end, col_start:col_end] = costs_chunk
            total_impassable_pixels += impassable_pixels
    
    # Clean up temporary files
    try:
        os.remove(temp_model_path)
        os.remove(temp_scaler_path)
    except:
        pass
    
    # Save cost surface
    output_path = "output/cost_surfaces/random_forest_cost_surface.tif"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=new_height,
        width=new_width,
        count=1,
        dtype=rasterio.float32,
        crs=ref_crs,
        transform=new_transform,
        compress='lzw'
    ) as dst:
        dst.write(cost_data, 1)
    
    # Print summary
    if total_impassable_pixels > 0:
        total_area_km2 = (total_impassable_pixels * 100) / 1000000
        total_raster_area_km2 = (new_height * new_width * 100) / 1000000
        coverage_percent = (total_impassable_pixels / (new_height * new_width)) * 100
        
        print(f"    Total impassable pixels: {total_impassable_pixels:,} ({total_area_km2:.3f} km²)")
        print(f"    Coverage: {coverage_percent:.2f}% of raster area ({total_raster_area_km2:.1f} km²)")
    
    print(f"  Random Forest cost surface saved to {output_path}")
    print(f"  Impassable areas included with cost value: 999.0")
    return output_path

def save_rf_model_and_results(model, feature_importance, predictor_names, scaler, output_dir="output/model_trace"):
    """
    Save Random Forest model, feature importance, and results.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        The trained Random Forest model
    feature_importance : array
        Feature importance scores
    predictor_names : list
        Names of the predictors
    scaler : StandardScaler
        The fitted scaler
    output_dir : str
        Directory to save outputs
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving Random Forest model and results to {output_dir}...")
    
    # Save the trained model
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Saved model to: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to: {scaler_path}")
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'predictor': predictor_names,
        'importance': feature_importance,
        'importance_rank': range(1, len(predictor_names) + 1)
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Update ranks after sorting
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    # Save feature importance
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"  Saved feature importance to: {importance_path}")
    
    # For compatibility with Bayesian model analysis, also save in coefficient format
    # Note: Random Forest doesn't have coefficients like linear models, but we can use importance as proxy
    coefficients_df = pd.DataFrame({
        'predictor': predictor_names,
        'coefficient_mean': feature_importance  # Using importance as coefficient proxy
    })
    
    coeff_path = os.path.join(output_dir, "coefficient_means.csv")
    coefficients_df.to_csv(coeff_path, index=False)
    print(f"  Saved coefficient proxy (importance) to: {coeff_path}")
    
    # Print feature importance summary
    print("\nFeature importance by predictor:")
    print("-" * 50)
    for _, row in importance_df.iterrows():
        print(f"  {row['predictor']:<25}: {row['importance']:>8.4f} (rank {row['importance_rank']})")
    
    print(f"\nRandom Forest model and results saved to: {output_dir}")
    
    return importance_df

def main():
    """
    Random Forest analysis pipeline
    """
    print("=== RANDOM FOREST MODEL FOR SPATIAL ANALYTICS ===")
    print(f"Using {N_CORES} cores with optimized Random Forest")
    
    # Create output directories
    os.makedirs("output/cost_surfaces", exist_ok=True)
    os.makedirs("output/stats_analysis", exist_ok=True)
    
    start_total = time.time()
    
    # Step 1: Load and subsample GPS data for speed
    print("\nStep 1: Loading GPS tracking data...")
    track_df = pd.read_csv("data/derived/csv/track_features.csv")
    
    # Convert geometry and add track_id
    track_df["geometry"] = track_df["geometry"].apply(wkt.loads)
    track_df["track_id"] = track_df["runner_id"]
    
    # Filter speeds
    track_df = track_df[
        (track_df["speed"] >= 0.5) & 
        (track_df["speed"] <= 25.0)
    ].copy()
    
    # Use all available data - no subsampling for maximum performance
    print(f"  Using all {len(track_df)} GPS points")
    print(f"  Speed range: {track_df['speed'].min():.1f} - {track_df['speed'].max():.1f} km/h")
    
    # Step 2: Load key environmental predictors
    print("\nStep 2: Loading key environmental predictors...")
    
    key_features = [
        'elevation', 'ndvi',
        'on_roads', 'dist_to_roads',
        'on_trails', 'dist_to_trails',
        'on_water', 'dist_to_water',
        'rocky_terrain', 'firm_wetlands',
        'landcover_barr__och_blandskog', 'landcover_öppen_mark',
        'on_buildings', 'dist_to_buildings'
    ]
    
    predictor_paths = {
        'elevation': 'data/derived/dem/dem_3006.vrt',
        'ndvi': 'data/derived/ndvi/VI_20240530T102021_S2A_T33VWD-010m_V101_NDVI_3006.tif'
    }
    
    # Add key Lantmäteriet features
    lantmateriet_features = [f for f in key_features if f not in ['elevation', 'ndvi']]
    for feature in lantmateriet_features:
        predictor_paths[feature] = f'data/derived/rasters/{feature}.tif'
    
    # Extract features
    predictor_data = {}
    predictor_names = []
    
    for name, path in predictor_paths.items():
        if name in key_features:
            try:
                with rasterio.open(path) as src:
                    coords = [(point.x, point.y) for point in track_df["geometry"]]
                    values = list(src.sample(coords))
                    values = [v[0] if len(v) > 0 else np.nan for v in values]
                    
                    # Simple missing value handling
                    if name.startswith('dist_to_'):
                        values = [100.0 if np.isnan(v) else v for v in values]
                    elif name.startswith('on_') or name.startswith('landcover_'):
                        values = [0.0 if np.isnan(v) else v for v in values]
                    elif name == 'elevation':
                        median_val = np.nanmedian(values) if np.any(~np.isnan(values)) else 100.0
                        values = [median_val if np.isnan(v) else v for v in values]
                    elif name == 'ndvi':
                        values = [0.3 if np.isnan(v) else v for v in values]
                    else:
                        values = [0.0 if np.isnan(v) else v for v in values]
                    
                    predictor_data[name] = values
                    predictor_names.append(name)
                    print(f"    {name}: loaded")
                    
            except Exception as e:
                print(f"    Warning: Could not load {name}: {e}")
    
    print(f"  Loaded {len(predictor_names)} features")
    
    # Step 3: Prepare track data
    print("\nStep 3: Preparing track structure...")
    track_df, n_tracks, track_mapping = prepare_track_data(track_df)
    track_nums = track_df['track_num'].values
    print(f"  Using {n_tracks} tracks")
    
    # Step 4: Feature selection
    y = np.log(track_df["speed"].values)
    
    # Re-extract features for the filtered dataset
    print("\nRe-extracting features for filtered dataset...")
    predictor_data = {}
    predictor_names = []
    
    for name, path in predictor_paths.items():
        if name in key_features:
            try:
                with rasterio.open(path) as src:
                    coords = [(point.x, point.y) for point in track_df["geometry"]]
                    values = list(src.sample(coords))
                    values = [v[0] if len(v) > 0 else np.nan for v in values]
                    
                    # Simple missing value handling
                    if name.startswith('dist_to_'):
                        values = [100.0 if np.isnan(v) else v for v in values]
                    elif name.startswith('on_') or name.startswith('landcover_'):
                        values = [0.0 if np.isnan(v) else v for v in values]
                    elif name == 'elevation':
                        median_val = np.nanmedian(values) if np.any(~np.isnan(values)) else 100.0
                        values = [median_val if np.isnan(v) else v for v in values]
                    elif name == 'ndvi':
                        values = [0.3 if np.isnan(v) else v for v in values]
                    else:
                        values = [0.0 if np.isnan(v) else v for v in values]
                    
                    predictor_data[name] = values
                    predictor_names.append(name)
                    
            except Exception as e:
                print(f"    Warning: Could not load {name}: {e}")
    
    top_features = select_top_features(predictor_data, predictor_names, y)
    
    # Prepare feature matrix with selected features only
    X = np.column_stack([predictor_data[name] for name in top_features])
    predictor_names = top_features  # Update to selected features
    
    print(f"  Final feature matrix: {X.shape}")
    print(f"  Target array: {y.shape}")
    
    # Step 5: Train/Test Split
    print("\nStep 4: Creating train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Step 6: Cross-validation
    print("\nStep 5: Cross-validation...")
    start_cv = time.time()
    
    cv_results = cross_validate_rf(X_train_scaled, y_train, n_folds=5, n_estimators=200)
    
    cv_time = time.time() - start_cv
    print(f"\nCross-validation completed in {cv_time:.1f} seconds")
    
    # Calculate statistics
    valid_results = [r for r in cv_results['r2'] if r != 0.0]
    if valid_results:
        r2_mean = np.mean(valid_results)
        r2_std = np.std(valid_results)
        rmse_mean = np.mean([r for r in cv_results['rmse'] if r != 999.0])
        oob_mean = np.mean([r for r in cv_results['oob_score'] if r != 0.0])
    else:
        r2_mean = 0.0
        r2_std = 0.0
        rmse_mean = 999.0
        oob_mean = 0.0
    
    print(f"  Cross-validation R²: {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"  Cross-validation RMSE: {rmse_mean:.3f} km/h")
    print(f"  Mean Out-of-Bag R²: {oob_mean:.3f}")
    
    # Step 7: Final model training
    print("\nStep 6: Training final Random Forest model...")
    
    final_model = fit_random_forest_model(
        X_train_scaled, y_train,
        n_estimators=300, max_depth=20  # Larger final model
    )
    
    print(f"  Final model Out-of-Bag R²: {final_model.oob_score_:.3f}")
    
    # Step 8: Test set evaluation
    print("\nStep 7: Test set evaluation...")
    
    y_pred_test = final_model.predict(X_test_scaled)
    y_pred_test_exp = np.exp(y_pred_test)
    y_test_exp = np.exp(y_test)
    
    test_r2 = r2_score(y_test_exp, y_pred_test_exp)
    test_rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_test_exp))
    test_mae = mean_absolute_error(y_test_exp, y_pred_test_exp)
    
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f} km/h")
    print(f"  Test MAE: {test_mae:.3f} km/h")
    
    # Step 9: Save model and feature importance
    print("\nStep 8: Saving model and feature importance...")
    feature_importance = final_model.feature_importances_
    importance_df = save_rf_model_and_results(final_model, feature_importance, predictor_names, scaler)
    
    # Step 10: Cost surface generation
    print("\nStep 9: Generating Random Forest cost surface...")
    reference_raster = 'data/derived/map/oringen_e4_2024_h21elit_REFERENCED_3006.tif'
    cost_surface = create_rf_cost_surface(
        final_model, predictor_paths, reference_raster, predictor_names, scaler
    )
    
    # Step 11: Save results
    print("\nStep 10: Saving results...")
    
    total_time = time.time() - start_total
    
    # Save summary
    summary = {
        'model_type': 'random_forest',
        'n_predictors': len(predictor_names),
        'n_observations': len(y),
        'n_tracks': n_tracks,
        'n_estimators': final_model.n_estimators,
        'max_depth': final_model.max_depth,
        'cv_r2_mean': r2_mean,
        'cv_r2_std': r2_std,
        'cv_rmse_mean': rmse_mean,
        'oob_score_final': final_model.oob_score_,
        'oob_score_mean_cv': oob_mean,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_time_seconds': cv_time,
        'total_time_seconds': total_time,
        'cores_used': N_CORES
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("output/stats_analysis/random_forest_summary.csv", index=False)
    
    print(f"\n=== RANDOM FOREST MODEL COMPLETE ===")
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Cross-validation R²: {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"Test set R²: {test_r2:.3f}")
    print(f"Final model Out-of-Bag R²: {final_model.oob_score_:.3f}")
    print(f"Used {len(predictor_names)} features on {len(y)} GPS points")
    print("Random Forest cost surface ready for analysis!")
    
    return cost_surface, final_model, importance_df

if __name__ == "__main__":
    main()
