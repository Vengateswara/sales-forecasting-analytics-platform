# backend/services/preprocessing_service.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import logging
import traceback
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class PreprocessingService:
    """Service for data preprocessing."""
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Perform automated preprocessing."""
        try:
            logger.info(f"Starting preprocessing. Shape: {df.shape}")
            
            df_processed = df.copy()
            log_steps = []
            date_columns = []
            
            # Step 1: Handle date columns - KEEP THEM AS DATETIME
            logger.info("Identifying and converting date columns...")
            for col in df_processed.columns:
                try:
                    col_lower = col.lower()
                    if 'date' in col_lower or 'time' in col_lower or 'datetime' in col_lower:
                        df_processed[col] = pd.to_datetime(df_processed[col])
                        date_columns.append(col)
                        log_steps.append(f"Converted {col} to datetime")
                        logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
            
            # Step 2: Handle missing values
            logger.info("Handling missing values...")
            for col in df_processed.columns:
                missing_count = df_processed[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        median_val = df_processed[col].median()
                        df_processed[col].fillna(median_val, inplace=True)
                        log_steps.append(f"Filled {missing_count} missing values in {col} with median")
                    elif pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                        df_processed[col].fillna(method='ffill', inplace=True)
                        log_steps.append(f"Forward filled {missing_count} missing values in {col}")
                    else:
                        mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"
                        df_processed[col].fillna(mode_val, inplace=True)
                        log_steps.append(f"Filled {missing_count} missing values in {col} with mode")
            
            # Step 3: Remove duplicates
            initial_shape = df_processed.shape
            df_processed.drop_duplicates(inplace=True)
            removed = initial_shape[0] - df_processed.shape[0]
            if removed > 0:
                log_steps.append(f"Removed {removed} duplicate rows")
                logger.info(f"Removed {removed} duplicate rows")
            
            # Step 4: Handle outliers (cap using IQR) - only for numeric columns
            logger.info("Handling outliers...")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
                    if len(outliers) > 0:
                        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                        log_steps.append(f"Capped {len(outliers)} outliers in {col}")
                        logger.info(f"Capped {len(outliers)} outliers in {col}")
                except Exception as e:
                    logger.warning(f"Could not process outliers for {col}: {str(e)}")
            
            # Step 5: Encode categorical variables (EXCLUDING date columns)
            logger.info("Encoding categorical variables...")
            
            # Get all columns that are NOT date columns and NOT numeric
            categorical_cols = []
            for col in df_processed.columns:
                if col not in date_columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                    categorical_cols.append(col)
            
            logger.info(f"Found {len(categorical_cols)} categorical columns to encode: {categorical_cols}")
            
            # Encode each categorical column
            for col in categorical_cols:
                try:
                    # Skip if column is a date column
                    if col in date_columns:
                        continue
                    
                    # Convert to string first to handle mixed types
                    df_processed[col] = df_processed[col].astype(str)
                    
                    # Use LabelEncoder
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    log_steps.append(f"Encoded categorical column {col}")
                    logger.info(f"Encoded {col} with {len(le.classes_)} unique values")
                    
                except Exception as e:
                    logger.warning(f"Could not encode {col}: {str(e)}")
                    # If encoding fails, drop the column
                    df_processed = df_processed.drop(columns=[col])
                    log_steps.append(f"Dropped column {col} (encoding failed)")
            
            # Step 6: Scale numeric features (EXCLUDING date columns)
            logger.info("Scaling numeric features...")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            # Exclude date columns that might have been converted to numeric
            numeric_cols = [col for col in numeric_cols if col not in date_columns]
            
            if len(numeric_cols) > 0:
                logger.info(f"Scaling {len(numeric_cols)} numeric features...")
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                log_steps.append(f"Scaled {len(numeric_cols)} numeric features")
            
            # Step 7: Sort by date if date column exists
            if date_columns:
                df_processed.sort_values(by=date_columns[0], inplace=True)
                log_steps.append(f"Sorted data by {date_columns[0]}")
                logger.info(f"Sorted by {date_columns[0]}")
            
            # Step 8: Final check - ensure date columns remain as datetime
            for col in date_columns:
                if col in df_processed.columns and not pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                    try:
                        df_processed[col] = pd.to_datetime(df_processed[col])
                        logger.info(f"Re-converted {col} to datetime")
                    except:
                        logger.warning(f"Could not re-convert {col} to datetime")
            
            logger.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
            logger.info(f"Date columns: {date_columns}")
            logger.info(f"Final data types:\n{df_processed.dtypes}")
            
            return df_processed, log_steps
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Preprocessing failed: {str(e)}")