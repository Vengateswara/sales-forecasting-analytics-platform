from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UploadResponse(BaseModel):
    """Response model for file upload."""
    filename: str
    file_size: int
    upload_time: datetime
    rows: int
    columns: int
    memory_usage: str
    column_names: List[str]
    data_types: Dict[str, str]
    status: str

class DataQualityReport(BaseModel):
    """Data quality report model."""
    missing_values: Dict[str, int]
    duplicate_rows: int
    incorrect_data_types: Dict[str, str]
    constant_columns: List[str]
    high_cardinality: Dict[str, int]
    date_columns: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_column: Optional[str]
    outliers: Dict[str, Dict[str, Any]]
    invalid_records: int
    overall_quality_score: float

class PreprocessingResult(BaseModel):
    """Result of data preprocessing."""
    processed_data: List[Dict[str, Any]]
    cleaned_rows: int
    removed_rows: int
    preprocessing_steps: List[str]
    timestamp: datetime

class FeatureEngineeringResult(BaseModel):
    """Result of feature engineering."""
    new_features: List[str]
    feature_groups: Dict[str, List[str]]
    total_features: int
    timestamp: datetime

class ModelMetrics(BaseModel):
    """Model evaluation metrics."""
    r2_score: float
    adjusted_r2: float
    mae: float
    mse: float
    rmse: float
    mape: float
    median_absolute_error: float
    explained_variance: float
    max_error: float
    training_time: float
    prediction_time: float

class ModelComparison(BaseModel):
    """Model comparison results."""
    model_name: str
    training_score: float
    testing_score: float
    cv_score: float
    metrics: ModelMetrics
    is_best: bool = False

class ForecastResult(BaseModel):
    """Forecast results model."""
    forecast_dates: List[datetime]
    forecast_values: List[float]
    lower_bound: Optional[List[float]]
    upper_bound: Optional[List[float]]
    confidence_level: float = 0.95
    growth_rate: float
    trend: str
    period: int