# debug_main.py
import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import services directly
from backend.services.data_service import DataService
from backend.services.preprocessing_service import PreprocessingService
from backend.services.feature_engineering_service import FeatureEngineeringService

def test_pipeline():
    """Test the pipeline step by step."""
    print("=" * 60)
    print("Testing Pipeline Step by Step")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sales = np.random.randn(100) * 50 + 200 + np.linspace(0, 50, 100)
    sales = np.maximum(sales, 10)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    print(f"   Created DataFrame with {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Sample data:\n{df.head()}")
    
    # 2. Test Preprocessing
    print("\n2. Testing Preprocessing Service...")
    try:
        preprocess_service = PreprocessingService()
        df_processed, log = preprocess_service.preprocess(df)
        print(f"   ✅ Preprocessing successful!")
        print(f"   Shape: {df_processed.shape}")
        print(f"   Steps: {len(log)}")
        for step in log:
            print(f"     - {step}")
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {str(e)}")
        traceback.print_exc()
        return
    
    # 3. Test Feature Engineering
    print("\n3. Testing Feature Engineering Service...")
    try:
        feature_service = FeatureEngineeringService()
        date_col = 'date'
        target_col = 'sales'
        
        print(f"   Using date_col={date_col}, target_col={target_col}")
        df_engineered, feature_info = feature_service.engineer_features(
            df_processed, date_col, target_col
        )
        
        print(f"   ✅ Feature engineering successful!")
        print(f"   Shape: {df_engineered.shape}")
        print(f"   New features: {feature_info['total_features']}")
        print(f"   Feature groups:")
        for group, features in feature_info.items():
            if group != 'total_features' and features:
                print(f"     - {group}: {len(features)} features")
                if len(features) <= 10:
                    print(f"       {features}")
        
        print(f"\n   Sample of engineered data:")
        print(df_engineered.head())
        
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {str(e)}")
        traceback.print_exc()
        return
    
    # 4. Test Model Training
    print("\n4. Testing Model Training...")
    try:
        from backend.services.model_service import ModelService
        
        model_service = ModelService()
        
        # Prepare features and target
        X = df_engineered.drop(columns=[target_col, date_col])
        y = df_engineered[target_col]
        
        print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        results, best_model = model_service.train_all_models(X, y)
        
        print(f"   ✅ Model training successful!")
        print(f"   Best model: {best_model['name']}")
        print(f"   R² Score: {best_model['metrics']['test_r2']:.4f}")
        print(f"   MAE: {best_model['metrics']['test_mae']:.2f}")
        print(f"   RMSE: {best_model['metrics']['test_rmse']:.2f}")
        
    except Exception as e:
        print(f"   ❌ Model training failed: {str(e)}")
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()