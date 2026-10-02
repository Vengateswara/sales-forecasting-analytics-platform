# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import traceback
from pathlib import Path

# Use relative imports instead of absolute
from .config import config
from .services.data_service import DataService
from .services.preprocessing_service import PreprocessingService
from .services.feature_engineering_service import FeatureEngineeringService
from .services.eda_service import EDAService
from .services.model_service import ModelService
from .services.forecasting_service import ForecastingService

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=config.ALLOWED_METHODS,
    allow_headers=config.ALLOWED_HEADERS,
)

# Initialize templates and static files
templates = Jinja2Templates(directory="backend/templates")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Initialize services
data_service = DataService()
preprocessing_service = PreprocessingService()
feature_engineering_service = FeatureEngineeringService()
eda_service = EDAService()
model_service = ModelService()
forecasting_service = ForecastingService()

# Global state
app.state.current_dataset = None
app.state.processed_data = None
app.state.engineered_data = None
app.state.best_model = None
app.state.model_results = None
app.state.forecast_results = None

@app.get("/")
async def home(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and validate dataset."""
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
        
        # Save file
        file_path = config.UPLOAD_DIR / file.filename
        content = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Load and validate data
        df = data_service.load_data(file_path, file_extension)
        
        # Validate dataset
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Store in app state
        app.state.current_dataset = df
        app.state.dataset_info = {
            "filename": file.filename,
            "file_size": len(content),
            "upload_time": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        # Generate data quality report
        quality_report = data_service.generate_quality_report(df)
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "data": app.state.dataset_info,
            "quality_report": quality_report
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
async def preprocess_data():
    """Preprocess the uploaded dataset."""
    try:
        if app.state.current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        # Run preprocessing
        df, preprocess_log = preprocessing_service.preprocess(
            app.state.current_dataset
        )
        
        app.state.processed_data = df
        
        return {
            "status": "success",
            "message": "Data preprocessing completed",
            "preprocess_log": preprocess_log,
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feature-engineering")
async def engineer_features():
    """Perform feature engineering on the processed data."""
    try:
        if app.state.processed_data is None:
            raise HTTPException(status_code=400, detail="No preprocessed data found")
        
        df = app.state.processed_data.copy()
        
        # ===== DROP PROBLEMATIC COLUMNS THAT CAUSE DATA LEAKAGE =====
        # These are unique identifiers that cause overfitting
        cols_to_drop = ['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name', 'Postal Code']
        
        dropped_cols = []
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped_cols.append(col)
                logger.info(f"Dropped column: {col}")
        
        if dropped_cols:
            logger.info(f"Dropped {len(dropped_cols)} columns: {dropped_cols}")
        
        # Also drop high-cardinality columns that could cause leakage
        # Keep only useful categorical columns
        useful_categorical = ['Segment', 'Category', 'Sub-Category', 'Ship Mode', 'Region']
        
        # Drop other categorical columns not in useful list
        for col in df.columns:
            if col not in useful_categorical and col not in ['Date', 'Time', 'Order Date', 'Ship Date']:
                if df[col].dtype == 'object' and df[col].nunique() > 20:
                    df = df.drop(columns=[col])
                    logger.info(f"Dropped high-cardinality column: {col}")
        
        # Update the processed data
        app.state.processed_data = df
        
        logger.info(f"Feature engineering - Data shape: {df.shape}")
        logger.info(f"Columns after dropping: {df.columns.tolist()}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Identify date column
        date_col = None
        
        # First check for datetime64 columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                logger.info(f"Found datetime column: {col}")
                break
        
        # If no datetime64, check for columns that look like dates
        if date_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'time' in col_lower or 'datetime' in col_lower:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        logger.info(f"Converted {col} to datetime")
                        break
                    except:
                        continue
        
        # If still no date, try Order Date or Ship Date
        if date_col is None:
            for col in ['Order Date', 'Ship Date', 'Date']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        logger.info(f"Converted {col} to datetime")
                        break
                    except:
                        continue
        
        # Identify target column (numeric column, likely 'Sales' or 'Profit')
        target_col = None
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_lower = col.lower()
                if 'sales' in col_lower or 'profit' in col_lower or 'revenue' in col_lower:
                    target_col = col
                    logger.info(f"Found target column: {col}")
                    break
        
        # If no obvious target, use the last numeric column
        if target_col is None:
            for col in reversed(df.columns):
                if pd.api.types.is_numeric_dtype(df[col]) and col != date_col:
                    target_col = col
                    logger.info(f"Using {col} as target column")
                    break
        
        if date_col is None:
            error_msg = "Could not identify date column. Please ensure your data has a date column."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        if target_col is None:
            error_msg = "Could not identify target column. Please ensure your data has a numeric target column."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Using date_col={date_col}, target_col={target_col}")
        
        # Engineer features
        engineered_df, feature_info = feature_engineering_service.engineer_features(
            df, date_col, target_col
        )
        
        app.state.engineered_data = engineered_df
        app.state.feature_info = feature_info
        
        return {
            "status": "success",
            "message": "Feature engineering completed",
            "feature_info": feature_info,
            "shape": engineered_df.shape,
            "columns": engineered_df.columns.tolist()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature engineering error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def train_models(background_tasks: BackgroundTasks):
    """Train all machine learning models."""
    try:
        if app.state.engineered_data is None:
            raise HTTPException(status_code=400, detail="No engineered data found")
        
        df = app.state.engineered_data
        target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Train models
        results, best_model_info = model_service.train_all_models(X, y)
        
        # Store the actual best_model object from the service
        app.state.best_model = model_service.best_model
        app.state.model_results = results
        
        logger.info(f"Best model stored: {app.state.best_model.get('name', 'Unknown')}")
        logger.info(f"Best model combined score: {app.state.best_model.get('combined_score', 0):.4f}")
        
        return {
            "status": "success",
            "message": "Model training completed",
            "results": results,
            "best_model": best_model_info['name'],
            "best_model_info": best_model_info,
            "metrics": best_model_info['metrics']
        }
        
    except Exception as e:
        logger.error(f"Model training error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast")
async def generate_forecast(period: int = 30):
    """Generate sales forecast for specified period."""
    try:
        if app.state.best_model is None:
            raise HTTPException(status_code=400, detail="No trained model found. Please train models first.")
        
        if app.state.engineered_data is None:
            raise HTTPException(status_code=400, detail="No data found. Please run feature engineering first.")
        
        # Get the model from app.state.best_model
        model = app.state.best_model.get('model')
        if model is None:
            raise HTTPException(status_code=400, detail="Model object not found. Please retrain models.")
        
        logger.info(f"Generating forecast for {period} periods using {app.state.best_model.get('name', 'Unknown')}")
        
        forecast = forecasting_service.generate_forecast(
            model,
            app.state.engineered_data,
            period
        )
        
        app.state.forecast_results = forecast
        
        return {
            "status": "success",
            "message": f"Forecast generated for {period} periods",
            "forecast": forecast
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/eda")
async def get_eda_visualizations():
    """Generate EDA visualizations."""
    try:
        if app.state.engineered_data is None:
            raise HTTPException(status_code=400, detail="No data found")
        
        eda_results = eda_service.generate_eda(app.state.engineered_data)
        
        return {
            "status": "success",
            "data": eda_results
        }
        
    except Exception as e:
        logger.error(f"EDA error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{file_type}")
async def download_file(file_type: str):
    """Download reports and data."""
    try:
        file_path = None
        filename = None
        
        if file_type == "cleaned_data":
            if app.state.processed_data is None:
                raise HTTPException(status_code=400, detail="No data available")
            file_path = config.REPORTS_DIR / "cleaned_data.csv"
            app.state.processed_data.to_csv(file_path, index=False)
            filename = "cleaned_dataset.csv"
            
        elif file_type == "forecast":
            if app.state.forecast_results is None:
                raise HTTPException(status_code=400, detail="No forecast available")
            file_path = config.REPORTS_DIR / "forecast_results.xlsx"
            forecasting_service.export_forecast(app.state.forecast_results, file_path)
            filename = "forecast_results.xlsx"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Dataset Preview Endpoints =====

@app.get("/api/get-readable-data")
async def get_readable_data():
    """Get the dataset with readable original values AND engineered features."""
    try:
        if app.state.processed_data is not None and app.state.engineered_data is not None:
            df_processed = app.state.processed_data.copy()
            df_engineered = app.state.engineered_data.copy()
            
            engineered_index = df_engineered.index
            df_processed_aligned = df_processed.loc[engineered_index].copy()
            
            df_combined = df_processed_aligned.copy()
            
            for col in df_engineered.columns:
                if col not in df_processed_aligned.columns:
                    df_combined[col] = df_engineered[col].values
            
            df = df_combined
            logger.info(f"Combined readable data: {df.shape}")
            
        elif app.state.engineered_data is not None:
            df = app.state.engineered_data.copy()
        elif app.state.processed_data is not None:
            df = app.state.processed_data.copy()
        elif app.state.current_dataset is not None:
            df = app.state.current_dataset.copy()
        else:
            raise HTTPException(status_code=400, detail="No data available")
        
        data = df.head(1000).to_dict(orient='records')
        
        for row in data:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, float):
                    row[key] = round(value, 2)
                elif isinstance(value, int):
                    row[key] = value
                elif isinstance(value, str):
                    pass
                elif isinstance(value, (np.integer, np.int64)):
                    row[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    row[key] = float(round(value, 2))
        
        return {
            "status": "success",
            "headers": df.columns.tolist(),
            "data": data,
            "total_rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error getting readable data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-engineered-data")
async def get_engineered_data():
    """Get the engineered dataset with all feature engineered columns."""
    try:
        if app.state.engineered_data is not None:
            df = app.state.engineered_data.copy()
            logger.info(f"Returning engineered data: {df.shape}")
        elif app.state.processed_data is not None:
            df = app.state.processed_data.copy()
            logger.info(f"Returning processed data: {df.shape}")
        else:
            raise HTTPException(status_code=400, detail="No data available")
        
        data = df.head(1000).to_dict(orient='records')
        
        for row in data:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, float):
                    row[key] = round(value, 2)
                elif isinstance(value, int):
                    row[key] = value
                elif isinstance(value, str):
                    pass
                elif isinstance(value, (np.integer, np.int64)):
                    row[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    row[key] = float(round(value, 2))
        
        return {
            "status": "success",
            "headers": df.columns.tolist(),
            "data": data,
            "total_rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error getting engineered data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/data")
async def debug_data():
    """Debug endpoint to check data state."""
    try:
        result = {
            "has_current_dataset": app.state.current_dataset is not None,
            "has_processed_data": app.state.processed_data is not None,
            "has_engineered_data": app.state.engineered_data is not None,
            "has_best_model": app.state.best_model is not None,
        }
        
        if app.state.processed_data is not None:
            df = app.state.processed_data
            result["processed_data"] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "sample": df.head(3).to_dict()
            }
        
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )