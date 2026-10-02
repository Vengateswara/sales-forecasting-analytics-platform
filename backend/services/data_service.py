# backend/services/data_service.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataService:
    """Service for handling data loading and validation."""
    
    def load_data(self, file_path: Path, file_extension: str) -> pd.DataFrame:
        """Load data from various file formats."""
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(v) for v in obj)
        else:
            return obj
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        # Convert all numpy types in the report
        report = {
            "missing_values": {},
            "duplicate_rows": int(df.duplicated().sum()),  # Convert to int
            "incorrect_data_types": {},
            "constant_columns": [],
            "high_cardinality": {},
            "date_columns": [],
            "numerical_columns": [],
            "categorical_columns": [],
            "target_column": None,
            "outliers": {},
            "invalid_records": 0,
            "overall_quality_score": 0.0
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        # Convert numpy int64 to Python int
        report["missing_values"] = {k: int(v) for k, v in missing_dict.items()}
        
        # Duplicate rows - already converted to int above
        
        # Column types
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                report["numerical_columns"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                report["date_columns"].append(col)
            else:
                report["categorical_columns"].append(col)
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                report["constant_columns"].append(col)
        
        # High cardinality categorical columns
        for col in report["categorical_columns"]:
            cardinality = df[col].nunique()
            if cardinality > 20:
                report["high_cardinality"][col] = int(cardinality)  # Convert to int
        
        # Outliers (IQR method)
        for col in report["numerical_columns"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                report["outliers"][col] = {
                    "count": int(len(outliers)),  # Convert to int
                    "percentage": float(len(outliers) / len(df) * 100),  # Convert to float
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
        
        # Calculate overall quality score
        score = 100.0
        
        # Penalty for missing values
        total_missing = sum(report["missing_values"].values())
        if total_missing > 0:
            penalty = (total_missing / (df.shape[0] * df.shape[1])) * 100
            score -= min(30, penalty)
        
        # Penalty for duplicates
        if report["duplicate_rows"] > 0:
            penalty = (report["duplicate_rows"] / df.shape[0]) * 50
            score -= min(20, penalty)
        
        # Penalty for constant columns
        if len(report["constant_columns"]) > 0:
            penalty = len(report["constant_columns"]) * 5
            score -= min(20, penalty)
        
        # Penalty for high cardinality
        if len(report["high_cardinality"]) > 0:
            penalty = len(report["high_cardinality"]) * 5
            score -= min(20, penalty)
        
        # Penalty for outliers
        total_outliers = sum([out["count"] for out in report["outliers"].values()])
        if total_outliers > 0:
            penalty = (total_outliers / df.shape[0]) * 20
            score -= min(30, penalty)
        
        report["overall_quality_score"] = float(max(0, min(100, score)))  # Convert to float
        
        # Final conversion of all numpy types
        return self._convert_numpy_types(report)