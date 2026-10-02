# backend/services/eda_service.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import logging
import traceback
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class EDAService:
    """Service for Exploratory Data Analysis."""
    
    def _clean_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN and drop rows with NaN."""
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types and handle infinite values for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def generate_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive EDA visualizations."""
        try:
            logger.info(f"Starting EDA for dataframe with shape {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Clean infinite values
            df = self._clean_infinite_values(df)
            
            # Drop rows with NaN in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df.dropna(subset=numeric_cols)
            logger.info(f"After cleaning infinite values: {df.shape}")
            
            # Ensure we only work with numeric data for some analyses
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Found {len(numeric_cols)} numeric columns")
            
            results = {
                "summary": self._generate_summary(df),
                "statistics": self._generate_statistics(df),
                "missing_data": self._analyze_missing_data(df),
                "distributions": [],
                "correlation": {},
                "trends": {},
                "boxplots": [],
                "scatter_plots": []
            }
            
            # Only create distributions if there are numeric columns
            if len(numeric_cols) > 0:
                try:
                    results["distributions"] = self._create_distribution_charts(df)
                    logger.info(f"Created {len(results['distributions'])} distribution charts")
                except Exception as e:
                    logger.warning(f"Error creating distribution charts: {str(e)}")
            
            # Only create correlation if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                try:
                    results["correlation"] = self._create_correlation_matrix(df)
                    logger.info("Created correlation matrix")
                except Exception as e:
                    logger.warning(f"Error creating correlation matrix: {str(e)}")
            
            # Create trend analysis if there's a date column
            try:
                results["trends"] = self._create_trend_analysis(df)
                logger.info("Created trend analysis")
            except Exception as e:
                logger.warning(f"Error creating trend analysis: {str(e)}")
            
            # Create boxplots
            if len(numeric_cols) > 0:
                try:
                    results["boxplots"] = self._create_boxplots(df)
                    logger.info(f"Created {len(results['boxplots'])} boxplots")
                except Exception as e:
                    logger.warning(f"Error creating boxplots: {str(e)}")
            
            # Create scatter plots
            if len(numeric_cols) >= 2:
                try:
                    results["scatter_plots"] = self._create_scatter_plots(df)
                    logger.info(f"Created {len(results['scatter_plots'])} scatter plots")
                except Exception as e:
                    logger.warning(f"Error creating scatter plots: {str(e)}")
            
            # Clean all results for JSON serialization
            results = self._convert_to_serializable(results)
            
            logger.info("EDA completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"EDA generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "trace": traceback.format_exc()}
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset summary."""
        try:
            return {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "numeric_cols": int(len(df.select_dtypes(include=[np.number]).columns)),
                "categorical_cols": int(len(df.select_dtypes(include=['object']).columns)),
                "datetime_cols": int(len(df.select_dtypes(include=['datetime64']).columns)),
                "memory_usage": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            }
        except Exception as e:
            logger.warning(f"Error generating summary: {str(e)}")
            return {"error": str(e)}
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Clean data for statistics
                data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 0:
                    stats[col] = {
                        "mean": float(data.mean()) if not np.isnan(data.mean()) else 0,
                        "median": float(data.median()) if not np.isnan(data.median()) else 0,
                        "std": float(data.std()) if not np.isnan(data.std()) else 0,
                        "min": float(data.min()) if not np.isnan(data.min()) else 0,
                        "max": float(data.max()) if not np.isnan(data.max()) else 0,
                        "q25": float(data.quantile(0.25)) if not np.isnan(data.quantile(0.25)) else 0,
                        "q75": float(data.quantile(0.75)) if not np.isnan(data.quantile(0.75)) else 0,
                        "skewness": float(data.skew()) if not np.isnan(data.skew()) else 0,
                        "kurtosis": float(data.kurtosis()) if not np.isnan(data.kurtosis()) else 0
                    }
            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {str(e)}")
                stats[col] = {"error": str(e)}
        
        return stats
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        return {
            "missing_values": missing[missing > 0].to_dict(),
            "missing_percentage": missing_pct[missing_pct > 0].to_dict(),
            "total_missing": int(missing.sum()),
            "total_cells": int(df.size)
        }
    
    def _create_distribution_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create distribution charts for numeric columns."""
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        
        for col in numeric_cols:
            try:
                # Clean data
                data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) == 0:
                    continue
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    nbinsx=30,
                    name='Distribution',
                    marker_color='#4a90d9'
                ))
                fig.update_layout(
                    title=f'Distribution of {col}',
                    xaxis_title=col,
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=400
                )
                
                charts.append({
                    "title": f"Distribution of {col}",
                    "chart": fig.to_json(),
                    "type": "histogram"
                })
            except Exception as e:
                logger.warning(f"Error creating distribution chart for {col}: {str(e)}")
                continue
        
        return charts
    
    def _create_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation matrix."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                # Clean data for correlation
                corr_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if len(corr_df) == 0:
                    return {"error": "No valid data for correlation"}
                
                corr_matrix = corr_df.corr()
                
                # Replace NaN with 0 for display
                corr_matrix = corr_matrix.fillna(0)
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(
                    title='Correlation Matrix',
                    template='plotly_white',
                    height=500
                )
                
                return {
                    "chart": fig.to_json(),
                    "matrix": corr_matrix.to_dict()
                }
            except Exception as e:
                logger.warning(f"Error creating correlation matrix: {str(e)}")
                return {"error": str(e)}
        return {"error": "Not enough numeric columns for correlation analysis"}
    
    def _create_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create trend analysis charts."""
        try:
            # Identify date column
            date_col = None
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
                elif 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
            
            if date_col and len(df.select_dtypes(include=[np.number]).columns) > 0:
                # Get the first numeric column for trend
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if numeric_col:
                    # Clean data
                    trend_df = df[[date_col, numeric_col]].copy()
                    trend_df = trend_df.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(trend_df) == 0:
                        return {"error": "No valid data for trend analysis"}
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trend_df[date_col],
                        y=trend_df[numeric_col],
                        mode='lines+markers',
                        name='Trend',
                        line=dict(color='#4a90d9', width=2),
                        marker=dict(size=4)
                    ))
                    fig.update_layout(
                        title=f'Trend Analysis: {numeric_col} over Time',
                        xaxis_title='Date',
                        yaxis_title=numeric_col,
                        template='plotly_white',
                        hovermode='x',
                        height=400
                    )
                    
                    return {
                        "chart": fig.to_json(),
                        "date_column": date_col,
                        "value_column": numeric_col
                    }
            return {"error": "No date column or numeric column found"}
        except Exception as e:
            logger.warning(f"Error creating trend analysis: {str(e)}")
            return {"error": str(e)}
    
    def _create_boxplots(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create box plots for numeric columns."""
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
        
        for col in numeric_cols:
            try:
                # Clean data
                data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) == 0:
                    continue
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=data,
                    name=col,
                    boxmean='sd',
                    marker_color='#4a90d9'
                ))
                fig.update_layout(
                    title=f'Box Plot: {col}',
                    yaxis_title=col,
                    template='plotly_white',
                    height=400
                )
                
                charts.append({
                    "title": f"Box Plot of {col}",
                    "chart": fig.to_json(),
                    "type": "boxplot"
                })
            except Exception as e:
                logger.warning(f"Error creating boxplot for {col}: {str(e)}")
                continue
        
        return charts
    
    def _create_scatter_plots(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create scatter plots between numeric columns."""
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Create scatter plots for first few combinations
            for i in range(min(3, len(numeric_cols) - 1)):
                try:
                    x_col = numeric_cols[i]
                    y_col = numeric_cols[i + 1]
                    
                    # Clean data
                    scatter_df = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(scatter_df) == 0:
                        continue
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=scatter_df[x_col],
                        y=scatter_df[y_col],
                        mode='markers',
                        name=f'{x_col} vs {y_col}',
                        marker=dict(
                            size=6,
                            color='#4a90d9',
                            opacity=0.6
                        )
                    ))
                    fig.update_layout(
                        title=f'{x_col} vs {y_col}',
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        template='plotly_white',
                        height=400
                    )
                    
                    charts.append({
                        "title": f"{x_col} vs {y_col}",
                        "chart": fig.to_json(),
                        "type": "scatter"
                    })
                except Exception as e:
                    logger.warning(f"Error creating scatter plot: {str(e)}")
                    continue
        
        return charts