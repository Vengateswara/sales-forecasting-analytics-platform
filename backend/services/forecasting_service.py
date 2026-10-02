# backend/services/forecasting_service.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import joblib
from backend.config import config

logger = logging.getLogger(__name__)

class ForecastingService:
    """Service for generating sales forecasts."""
    
    def generate_forecast(
        self,
        model,
        data: pd.DataFrame,
        periods: int = 30
    ) -> Dict[str, Any]:
        """Generate forecasts for future periods."""
        try:
            logger.info(f"Generating forecast for {periods} periods")
            logger.info(f"Data shape: {data.shape}")
            
            # Get the target column (last column or 'Rating' or 'Sales')
            target_col = data.columns[-1]
            # Check if there's a more appropriate target column
            for col in ['Rating', 'Sales', 'sales', 'Total', 'Profit']:
                if col in data.columns:
                    target_col = col
                    break
            logger.info(f"Target column: {target_col}")
            
            # Get the date column
            date_col = None
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    date_col = col
                    break
            
            if date_col is None:
                # Try to find a date-like column
                for col in data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            data[col] = pd.to_datetime(data[col])
                            date_col = col
                            break
                        except:
                            continue
            
            if date_col is None:
                # Use index as date
                logger.warning("No date column found, using index as date")
                date_col = 'index'
                data = data.reset_index()
                data['index'] = pd.to_datetime(data['index'])
            
            # Get the last date
            last_date = data[date_col].iloc[-1]
            if isinstance(last_date, pd.Timestamp):
                last_date = last_date.to_pydatetime()
            elif isinstance(last_date, str):
                last_date = pd.to_datetime(last_date).to_pydatetime()
            
            logger.info(f"Last date: {last_date}")
            
            # Get feature columns (exclude target and date)
            feature_cols = [col for col in data.columns if col != target_col and col != date_col]
            # Only keep numeric columns
            feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
            logger.info(f"Using {len(feature_cols)} numeric feature columns")
            
            if len(feature_cols) == 0:
                logger.warning("No feature columns found, using simple forecasting")
                return self._simple_forecast(data, target_col, periods)
            
            # Prepare future dates
            future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            
            # Get the last row for forecasting
            last_row = data.iloc[-1:].copy()
            
            # Recursive forecasting
            future_predictions = []
            current_row = last_row.copy()
            
            for i in range(periods):
                try:
                    # Prepare features for prediction
                    X_pred = current_row[feature_cols]
                    
                    # Make prediction
                    pred = model.predict(X_pred)[0]
                    future_predictions.append(float(pred))
                    
                    # Update lag features for next iteration (if they exist)
                    for col in feature_cols:
                        if 'lag_' in col:
                            try:
                                lag_num = int(col.split('_')[-1])
                                if i >= lag_num - 1:
                                    # Use previous prediction for lag
                                    current_row[col] = future_predictions[i - lag_num + 1] if (i - lag_num + 1) >= 0 else pred
                            except:
                                pass
                        elif 'rolling_mean' in col or 'rolling_median' in col or 'rolling_max' in col or 'rolling_min' in col:
                            # For rolling features, keep the same value
                            pass
                        elif 'growth_rate' in col or 'difference' in col:
                            # For trend features, keep the same value
                            pass
                    
                    # Update time features if they exist
                    future_date = last_date + timedelta(days=i+1)
                    if 'year' in feature_cols:
                        current_row['year'] = future_date.year
                    if 'quarter' in feature_cols:
                        current_row['quarter'] = (future_date.month - 1) // 3 + 1
                    if 'month' in feature_cols:
                        current_row['month'] = future_date.month
                    if 'day' in feature_cols:
                        current_row['day'] = future_date.day
                    if 'day_of_week' in feature_cols:
                        current_row['day_of_week'] = future_date.weekday()
                    if 'weekend' in feature_cols:
                        current_row['weekend'] = 1 if future_date.weekday() >= 5 else 0
                    if 'week_of_year' in feature_cols:
                        current_row['week_of_year'] = future_date.isocalendar()[1]
                    
                    # Update target for next iteration
                    current_row[target_col] = pred
                    
                except Exception as e:
                    logger.warning(f"Error in iteration {i}: {str(e)}")
                    # Use last prediction or fallback
                    if future_predictions:
                        future_predictions.append(future_predictions[-1])
                    else:
                        future_predictions.append(float(data[target_col].mean()))
            
            # Calculate metrics
            forecast_values = np.array(future_predictions)
            
            # If all predictions are the same or zero, use simple forecasting
            if np.std(forecast_values) < 0.01 or np.all(forecast_values == 0):
                logger.warning("Model predictions are constant or zero, using simple forecasting")
                return self._simple_forecast(data, target_col, periods)
            
            # Calculate growth rate
            if forecast_values[0] != 0:
                growth_rate = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
            else:
                growth_rate = 0
            
            # Determine trend
            if growth_rate > 5:
                trend = "Strong Upward"
            elif growth_rate > 1:
                trend = "Upward"
            elif growth_rate > -1:
                trend = "Stable"
            elif growth_rate > -5:
                trend = "Downward"
            else:
                trend = "Strong Downward"
            
            # Calculate confidence intervals
            if len(forecast_values) > 1:
                std_dev = np.std(forecast_values)
                confidence_interval = 1.96 * std_dev if std_dev > 0 else 0.1
            else:
                confidence_interval = 0.1
            
            lower_bound = forecast_values - confidence_interval
            upper_bound = forecast_values + confidence_interval
            
            result = {
                'forecast_dates': [d.isoformat() for d in future_dates],
                'forecast_values': [float(v) for v in forecast_values.tolist()],
                'lower_bound': [float(v) for v in lower_bound.tolist()],
                'upper_bound': [float(v) for v in upper_bound.tolist()],
                'confidence_level': 0.95,
                'growth_rate': float(growth_rate),
                'trend': trend,
                'period': periods,
                'summary': {
                    'min': float(np.min(forecast_values)),
                    'max': float(np.max(forecast_values)),
                    'mean': float(np.mean(forecast_values)),
                    'median': float(np.median(forecast_values)),
                    'std': float(np.std(forecast_values))
                }
            }
            
            logger.info(f"Forecast generated successfully. Trend: {trend}, Growth: {growth_rate:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}", exc_info=True)
            # Fallback to simple forecasting
            logger.info("Falling back to simple forecasting")
            return self._simple_forecast(data, target_col, periods)
    
    def _simple_forecast(self, data: pd.DataFrame, target_col: str, periods: int) -> Dict[str, Any]:
        """Simple forecasting using last values and moving average."""
        try:
            logger.info(f"Using simple forecasting for {periods} periods")
            
            # Get the target values
            values = data[target_col].values
            
            # Simple forecast: use last value + trend
            last_value = values[-1] if len(values) > 0 else 0
            
            # Calculate trend from last N values
            n = min(30, len(values))
            if n > 1:
                recent_values = values[-n:]
                trend = np.mean(np.diff(recent_values))
            else:
                trend = 0
            
            # Generate forecast
            forecast_values = []
            for i in range(periods):
                pred = last_value + trend * (i + 1)
                forecast_values.append(max(0, pred))  # Ensure non-negative
            
            forecast_values = np.array(forecast_values)
            
            # Calculate metrics
            if forecast_values[0] != 0:
                growth_rate = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
            else:
                growth_rate = 0
            
            # Determine trend
            if growth_rate > 5:
                trend = "Strong Upward"
            elif growth_rate > 1:
                trend = "Upward"
            elif growth_rate > -1:
                trend = "Stable"
            elif growth_rate > -5:
                trend = "Downward"
            else:
                trend = "Strong Downward"
            
            # Confidence intervals
            std_dev = np.std(forecast_values) if len(forecast_values) > 1 else 1
            confidence_interval = 1.96 * std_dev
            
            # Get last date
            last_date = datetime.now()
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    last_date = data[col].iloc[-1]
                    if isinstance(last_date, pd.Timestamp):
                        last_date = last_date.to_pydatetime()
                    break
            
            future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            
            result = {
                'forecast_dates': [d.isoformat() for d in future_dates],
                'forecast_values': [float(v) for v in forecast_values.tolist()],
                'lower_bound': [float(v) for v in (forecast_values - confidence_interval).tolist()],
                'upper_bound': [float(v) for v in (forecast_values + confidence_interval).tolist()],
                'confidence_level': 0.95,
                'growth_rate': float(growth_rate),
                'trend': trend,
                'period': periods,
                'summary': {
                    'min': float(np.min(forecast_values)),
                    'max': float(np.max(forecast_values)),
                    'mean': float(np.mean(forecast_values)),
                    'median': float(np.median(forecast_values)),
                    'std': float(np.std(forecast_values))
                }
            }
            
            logger.info(f"Simple forecast generated. Trend: {trend}, Growth: {growth_rate:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Simple forecast error: {str(e)}", exc_info=True)
            # Return a default forecast
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, periods + 1)]
            return {
                'forecast_dates': [d.isoformat() for d in future_dates],
                'forecast_values': [0.0] * periods,
                'lower_bound': [0.0] * periods,
                'upper_bound': [0.0] * periods,
                'confidence_level': 0.95,
                'growth_rate': 0.0,
                'trend': 'Stable',
                'period': periods,
                'summary': {
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0
                }
            }
    
    def export_forecast(self, forecast_data: Dict[str, Any], file_path: str):
        """Export forecast results to Excel."""
        try:
            dates = forecast_data['forecast_dates']
            values = forecast_data['forecast_values']
            lower = forecast_data.get('lower_bound', [None] * len(values))
            upper = forecast_data.get('upper_bound', [None] * len(values))
            
            df = pd.DataFrame({
                'Date': dates,
                'Forecast': values,
                'Lower Bound': lower,
                'Upper Bound': upper
            })
            
            df.to_excel(file_path, index=False)
            logger.info(f"Forecast exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            raise