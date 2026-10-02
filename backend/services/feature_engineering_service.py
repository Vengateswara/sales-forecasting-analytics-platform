# backend/services/feature_engineering_service.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class FeatureEngineeringService:
    """Service for automated feature engineering."""
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get only numeric columns from the dataframe."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Automatically engineer features."""
        try:
            logger.info(f"Starting feature engineering")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Date column: {date_col}, Target column: {target_col}")
            logger.info(f"Data types:\n{df.dtypes}")
            
            df_feat = df.copy()
            feature_info = {
                "time_features": [],
                "lag_features": [],
                "rolling_features": [],
                "trend_features": [],
                "seasonality_features": [],
                "total_features": 0
            }
            
            # Ensure we only work with numeric data for features
            numeric_cols = self._get_numeric_columns(df_feat)
            logger.info(f"Found {len(numeric_cols)} numeric columns")
            
            # Check if target column is numeric
            if target_col not in numeric_cols:
                logger.warning(f"Target column {target_col} is not numeric!")
                try:
                    df_feat[target_col] = pd.to_numeric(df_feat[target_col], errors='coerce')
                    if df_feat[target_col].isna().all():
                        raise ValueError(f"Target column {target_col} cannot be converted to numeric")
                    logger.info(f"Successfully converted {target_col} to numeric")
                except Exception as e:
                    raise ValueError(f"Target column {target_col} must be numeric: {str(e)}")
            
            # Ensure date column is datetime
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
                    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
                    logger.info(f"Successfully converted {date_col} to datetime")
            except Exception as e:
                logger.error(f"Error converting date column: {str(e)}")
                # Try with errors='coerce'
                try:
                    df_feat[date_col] = pd.to_datetime(df_feat[date_col], errors='coerce')
                    logger.info(f"Converted {date_col} to datetime with coerce")
                    # Drop rows with NaT
                    df_feat = df_feat.dropna(subset=[date_col])
                    logger.info(f"Dropped rows with invalid dates. New shape: {df_feat.shape}")
                except Exception as e2:
                    raise ValueError(f"Could not convert {date_col} to datetime: {str(e2)}")
            
            # Sort by date
            df_feat = df_feat.sort_values(by=date_col).reset_index(drop=True)
            
            # Get feature columns (exclude date and target)
            feature_cols = [col for col in df_feat.columns if col != date_col and col != target_col]
            logger.info(f"Using {len(feature_cols)} features for engineering")
            
            # Check if we have enough data
            n_rows = len(df_feat)
            logger.info(f"Data has {n_rows} rows after cleaning")
            
            # Only create features if we have enough data
            min_rows_for_features = 10
            
            # 1. Time Features (always possible)
            try:
                logger.info("Creating time features...")
                time_features = self._create_time_features(df_feat, date_col)
                df_feat = pd.concat([df_feat, time_features], axis=1)
                feature_info["time_features"] = list(time_features.columns)
                logger.info(f"Created {len(time_features.columns)} time features")
            except Exception as e:
                logger.warning(f"Error creating time features: {str(e)}")
            
            # 2. Lag Features - only if enough data
            if n_rows >= min_rows_for_features:
                try:
                    logger.info("Creating lag features...")
                    lag_features = self._create_lag_features(df_feat, target_col)
                    if not lag_features.empty:
                        df_feat = pd.concat([df_feat, lag_features], axis=1)
                        feature_info["lag_features"] = list(lag_features.columns)
                        logger.info(f"Created {len(lag_features.columns)} lag features")
                except Exception as e:
                    logger.warning(f"Error creating lag features: {str(e)}")
            else:
                logger.warning(f"Skipping lag features - only {n_rows} rows (need {min_rows_for_features})")
            
            # 3. Rolling Statistics - only if enough data
            if n_rows >= min_rows_for_features:
                try:
                    logger.info("Creating rolling features...")
                    rolling_features = self._create_rolling_features(df_feat, target_col)
                    if not rolling_features.empty:
                        df_feat = pd.concat([df_feat, rolling_features], axis=1)
                        feature_info["rolling_features"] = list(rolling_features.columns)
                        logger.info(f"Created {len(rolling_features.columns)} rolling features")
                except Exception as e:
                    logger.warning(f"Error creating rolling features: {str(e)}")
            else:
                logger.warning(f"Skipping rolling features - only {n_rows} rows")
            
            # 4. Trend Features
            if n_rows >= 5:
                try:
                    logger.info("Creating trend features...")
                    trend_features = self._create_trend_features(df_feat, target_col)
                    df_feat = pd.concat([df_feat, trend_features], axis=1)
                    feature_info["trend_features"] = list(trend_features.columns)
                    logger.info(f"Created {len(trend_features.columns)} trend features")
                except Exception as e:
                    logger.warning(f"Error creating trend features: {str(e)}")
            else:
                logger.warning(f"Skipping trend features - only {n_rows} rows")
            
            # 5. Seasonality Features
            try:
                logger.info("Creating seasonality features...")
                seasonality_features = self._create_seasonality_features(df_feat, date_col)
                df_feat = pd.concat([df_feat, seasonality_features], axis=1)
                feature_info["seasonality_features"] = list(seasonality_features.columns)
                logger.info(f"Created {len(seasonality_features.columns)} seasonality features")
            except Exception as e:
                logger.warning(f"Error creating seasonality features: {str(e)}")
            
            # Count total features
            feature_info["total_features"] = len(df_feat.columns) - len(df.columns)
            
            # Drop rows with NaN (from lag features)
            before_drop = len(df_feat)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            df_feat = df_feat.dropna()
            after_drop = len(df_feat)
            
            if after_drop < before_drop:
                logger.info(f"Dropped {before_drop - after_drop} rows with NaN values")
            
            # If all rows were dropped, return original data with time features only
            if len(df_feat) == 0:
                logger.warning("All rows dropped due to NaN. Returning data with only time features.")
                df_feat = df.copy()
                # Ensure date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
                    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
                time_features = self._create_time_features(df_feat, date_col)
                df_feat = pd.concat([df_feat, time_features], axis=1)
                feature_info["total_features"] = len(time_features.columns)
                return df_feat, feature_info
            
            logger.info(f"Feature engineering complete. Final shape: {df_feat.shape}")
            logger.info(f"New features created: {feature_info['total_features']}")
            
            return df_feat, feature_info
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Feature engineering failed: {str(e)}")
    
    def _create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame()
        dt = df[date_col]
        
        features['year'] = dt.dt.year
        features['quarter'] = dt.dt.quarter
        features['month'] = dt.dt.month
        features['day'] = dt.dt.day
        features['day_of_week'] = dt.dt.dayofweek
        features['week_of_year'] = dt.dt.isocalendar().week.astype(int)
        features['weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        
        return features
    
    def _create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create lag features."""
        features = pd.DataFrame()
        lags = [1, 7, 30]
        
        for lag in lags:
            if len(df) > lag:
                features[f'lag_{lag}'] = df[target_col].shift(lag)
        
        return features
    
    def _create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create rolling statistics features."""
        features = pd.DataFrame()
        windows = [7, 14, 30]
        
        for window in windows:
            if len(df) > window:
                try:
                    features[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
                    features[f'rolling_median_{window}'] = df[target_col].rolling(window=window, min_periods=1).median()
                    features[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
                    features[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
                    features[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
                except Exception as e:
                    logger.warning(f"Error creating rolling features for window {window}: {e}")
                    continue
        
        return features
    
    def _create_trend_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create trend-based features."""
        features = pd.DataFrame()
        
        try:
            features['growth_rate'] = df[target_col].pct_change().replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            logger.warning(f"Error creating growth_rate: {e}")
        
        try:
            features['difference'] = df[target_col].diff()
        except Exception as e:
            logger.warning(f"Error creating difference: {e}")
        
        try:
            features['ema'] = df[target_col].ewm(span=30, adjust=False, min_periods=1).mean()
        except Exception as e:
            logger.warning(f"Error creating ema: {e}")
        
        return features
    
    def _create_seasonality_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create seasonality features."""
        features = pd.DataFrame()
        dt = df[date_col]
        
        features['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
        features['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        
        return features