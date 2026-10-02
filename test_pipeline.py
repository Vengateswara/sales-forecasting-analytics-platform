# test_pipeline.py
"""
Test script for the complete sales forecasting pipeline.
This will test all services without needing the web interface.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.data_service import DataService
from backend.services.preprocessing_service import PreprocessingService
from backend.services.feature_engineering_service import FeatureEngineeringService
from backend.services.eda_service import EDAService
from backend.services.model_service import ModelService
from backend.services.forecasting_service import ForecastingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample sales data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    # Create sales with trend and seasonality
    trend = np.linspace(100, 300, 200)
    seasonality = 50 * np.sin(2 * np.pi * np.arange(200) / 30)  # 30-day cycle
    noise = np.random.normal(0, 20, 200)
    
    sales = trend + seasonality + noise
    sales = np.maximum(sales, 10)  # Ensure positive values
    
    data = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'product': np.random.choice(['Product_A', 'Product_B', 'Product_C'], 200),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 200),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 200)
    })
    
    return data

def test_pipeline():
    """Test the complete pipeline."""
    logger.info("=" * 60)
    logger.info("Testing Sales Forecasting Pipeline")
    logger.info("=" * 60)
    
    # 1. Create sample data
    logger.info("\n1. Creating sample data...")
    df = create_sample_data()
    logger.info(f"   Created dataset with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"   Columns: {df.columns.tolist()}")
    
    # 2. Data Service - Generate Quality Report
    logger.info("\n2. Generating data quality report...")
    data_service = DataService()
    quality_report = data_service.generate_quality_report(df)
    logger.info(f"   Quality Score: {quality_report['overall_quality_score']:.1f}%")
    logger.info(f"   Missing Values: {len(quality_report['missing_values'])} columns")
    logger.info(f"   Duplicates: {quality_report['duplicate_rows']}")
    logger.info(f"   Outliers: {len(quality_report['outliers'])} columns")
    
    # 3. Preprocessing Service
    logger.info("\n3. Preprocessing data...")
    preprocess_service = PreprocessingService()
    df_processed, preprocess_log = preprocess_service.preprocess(df)
    logger.info(f"   Processed shape: {df_processed.shape}")
    logger.info(f"   Preprocessing steps: {len(preprocess_log)}")
    for step in preprocess_log[:3]:
        logger.info(f"     - {step}")
    if len(preprocess_log) > 3:
        logger.info(f"     ... and {len(preprocess_log) - 3} more steps")
    
    # 4. Feature Engineering
    logger.info("\n4. Engineering features...")
    feature_service = FeatureEngineeringService()
    
    # Identify date and target columns
    date_col = 'date'
    target_col = 'sales'
    
    df_engineered, feature_info = feature_service.engineer_features(
        df_processed, date_col, target_col
    )
    logger.info(f"   Engineered shape: {df_engineered.shape}")
    logger.info(f"   New features: {feature_info['total_features']}")
    logger.info(f"   Feature groups:")
    for group, features in feature_info.items():
        if group != 'total_features' and features:
            logger.info(f"     - {group}: {len(features)} features")
    
    # 5. EDA Service
    logger.info("\n5. Generating EDA...")
    eda_service = EDAService()
    eda_results = eda_service.generate_eda(df_engineered)
    logger.info(f"   EDA summary: {eda_results.get('summary', {})}")
    logger.info(f"   Missing data: {eda_results.get('missing_data', {})}")
    
    # 6. Model Service
    logger.info("\n6. Training models...")
    model_service = ModelService()
    
    # Prepare features and target
    X = df_engineered.drop(columns=[target_col, date_col])
    y = df_engineered[target_col]
    
    logger.info(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
    results, best_model = model_service.train_all_models(X, y)
    
    logger.info(f"   Best model: {best_model['name']}")
    logger.info(f"   R² Score: {best_model['metrics']['test_r2']:.4f}")
    logger.info(f"   MAE: {best_model['metrics']['test_mae']:.2f}")
    logger.info(f"   RMSE: {best_model['metrics']['test_rmse']:.2f}")
    logger.info(f"   Training time: {best_model['train_time']:.2f}s")
    
    # 7. Forecasting Service
    logger.info("\n7. Generating forecast...")
    forecast_service = ForecastingService()
    
    periods = 30
    forecast = forecast_service.generate_forecast(
        best_model['model'],
        df_engineered,
        periods
    )
    
    logger.info(f"   Forecast periods: {periods}")
    logger.info(f"   Trend: {forecast['trend']}")
    logger.info(f"   Growth rate: {forecast['growth_rate']:.2f}%")
    logger.info(f"   Average forecast: {forecast['summary']['mean']:.2f}")
    logger.info(f"   Min forecast: {forecast['summary']['min']:.2f}")
    logger.info(f"   Max forecast: {forecast['summary']['max']:.2f}")
    
    # 8. Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Pipeline test completed successfully!")
    logger.info("=" * 60)
    logger.info("\nSummary:")
    logger.info(f"  - Original data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"  - After preprocessing: {len(df_processed)} rows, {len(df_processed.columns)} columns")
    logger.info(f"  - After feature engineering: {len(df_engineered)} rows, {len(df_engineered.columns)} columns")
    logger.info(f"  - Best model: {best_model['name']} (R² = {best_model['metrics']['test_r2']:.4f})")
    logger.info(f"  - Forecast trend: {forecast['trend']} ({forecast['growth_rate']:.2f}%)")
    
    return {
        'success': True,
        'best_model': best_model['name'],
        'r2_score': best_model['metrics']['test_r2'],
        'forecast_trend': forecast['trend']
    }

if __name__ == "__main__":
    try:
        result = test_pipeline()
        print("\n✅ All tests passed!")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)