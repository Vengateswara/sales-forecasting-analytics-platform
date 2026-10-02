# backend/services/model_service.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score, max_error
)
import time
import logging
from typing import Dict, List, Any, Tuple
import joblib
from backend.config import config
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)

class ModelService:
    """Service for training and evaluating machine learning models."""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            # Strong regularization for tree-based models to prevent overfitting
            'Random Forest': RandomForestRegressor(
                n_estimators=50,
                random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.3
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=50,
                random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.3
            ),
            'XGBoost': XGBRegressor(
                n_estimators=50,
                random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=2.0
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=50,
                random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS,
                verbose=-1,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=2.0
            ),
            'CatBoost': CatBoostRegressor(
                n_estimators=50,
                random_state=config.RANDOM_STATE,
                verbose=False,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=5,
                subsample=0.6
            )
        }
        
        self.results = {}
        self.best_model = None
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(v) for v in obj)
        else:
            return obj
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """Train all models and return results sorted by performance."""
        try:
            logger.info(f"Starting model training with {len(X)} samples and {len(X.columns)} features")
            
            # --- CHECK FOR DATA LEAKAGE ---
            logger.info("Checking for data leakage...")
            for col in X.columns:
                try:
                    corr = abs(X[col].corr(y))
                    if corr > 0.99:
                        logger.warning(f"⚠️ DATA LEAKAGE: '{col}' has {corr:.4f} correlation with target!")
                except:
                    pass
            
            # Ensure X has only numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) != len(X.columns):
                logger.warning(f"Found non-numeric columns: {set(X.columns) - set(numeric_cols)}")
                X = X[numeric_cols]
                logger.info(f"Using only {len(numeric_cols)} numeric features")
            
            # Ensure y is numeric
            if not pd.api.types.is_numeric_dtype(y):
                logger.error(f"Target column is not numeric: {y.dtype}")
                try:
                    y = pd.to_numeric(y, errors='coerce')
                    y = y.dropna()
                    X = X.loc[y.index]
                    logger.info(f"Converted target to numeric. Shape: {X.shape}")
                except Exception as e:
                    raise ValueError(f"Target column must be numeric: {str(e)}")
            
            # Remove constant columns
            constant_cols = [col for col in X.columns if X[col].std() == 0]
            if constant_cols:
                logger.info(f"Removing {len(constant_cols)} constant columns")
                X = X.drop(columns=constant_cols)
            
            # Check for NaN values
            if X.isna().any().any():
                logger.warning("Found NaN values in features. Dropping rows with NaN...")
                X = X.dropna()
                y = y.loc[X.index]
            
            if y.isna().any():
                logger.warning("Found NaN values in target. Dropping rows with NaN...")
                valid_idx = ~y.isna()
                X = X[valid_idx]
                y = y[valid_idx]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE
            )
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            logger.info(f"Final features: {len(X.columns)}")
            
            results = {}
            
            for name, model in self.models.items():
                try:
                    logger.info(f"Training {name}...")
                    
                    # Training time
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # Predictions
                    start_time = time.time()
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    pred_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        y_train, y_train_pred,
                        y_test, y_test_pred,
                        X_train.shape[1]
                    )
                    
                    # Convert metrics to serializable types
                    metrics = self._convert_to_serializable(metrics)
                    
                    # Cross-validation score
                    try:
                        cv_scores = cross_val_score(
                            model, X_train, y_train,
                            cv=min(5, len(X_train)),
                            scoring='r2'
                        )
                        cv_score = float(cv_scores.mean())
                        cv_std = float(cv_scores.std())
                    except Exception as e:
                        cv_score = 0.0
                        cv_std = 1.0
                        logger.warning(f"CV failed for {name}: {str(e)}")
                    
                    # Calculate overfitting score
                    train_r2 = metrics.get('train_r2', 0)
                    test_r2 = metrics.get('test_r2', 0)
                    overfitting_penalty = abs(train_r2 - test_r2)
                    
                    # HEAVY PENALTY for overfitting
                    extra_overfit_penalty = 0
                    if train_r2 > 0.95:
                        extra_overfit_penalty = 1.0
                    if train_r2 > 0.99:
                        extra_overfit_penalty = 2.0
                    
                    # Combined score with heavier penalty
                    combined_score = cv_score - overfitting_penalty * 5 - extra_overfit_penalty * 2
                    
                    # Bonus for good CV without overfitting
                    if cv_score > 0.5 and overfitting_penalty < 0.1:
                        combined_score += 0.3
                    
                    results[name] = {
                        'model': model,
                        'training_score': float(train_r2),
                        'testing_score': float(test_r2),
                        'cv_score': float(cv_score),
                        'cv_std': float(cv_std),
                        'combined_score': float(combined_score),
                        'overfitting_penalty': float(overfitting_penalty + extra_overfit_penalty),
                        'metrics': metrics,
                        'train_time': float(train_time),
                        'pred_time': float(pred_time)
                    }
                    
                    logger.info(f"{name}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, CV={cv_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No models were trained successfully")
            
            # Sort results by testing score (best to worst)
            sorted_results = dict(sorted(
                results.items(),
                key=lambda x: x[1]['combined_score'],
                reverse=True
            ))
            
            # Find best model (top of sorted list)
            best_name = list(sorted_results.keys())[0]
            
            # Store the best model with the actual model object
            self.best_model = {
                'name': best_name,
                'model': sorted_results[best_name]['model'],
                'training_score': float(sorted_results[best_name]['training_score']),
                'testing_score': float(sorted_results[best_name]['testing_score']),
                'cv_score': float(sorted_results[best_name]['cv_score']),
                'combined_score': float(sorted_results[best_name]['combined_score']),
                'metrics': self._convert_to_serializable(sorted_results[best_name]['metrics']),
                'train_time': float(sorted_results[best_name]['train_time']),
                'pred_time': float(sorted_results[best_name]['pred_time'])
            }
            
            self.results = sorted_results
            
            logger.info(f"✅ Best model: {best_name}")
            logger.info(f"  Train R² = {sorted_results[best_name]['training_score']:.4f}")
            logger.info(f"  Test R² = {sorted_results[best_name]['testing_score']:.4f}")
            logger.info(f"  CV Score = {sorted_results[best_name]['cv_score']:.4f}")
            
            # Save best model
            try:
                model_path = config.MODELS_DIR / 'best_model.pkl'
                joblib.dump(self.best_model['model'], model_path)
                logger.info(f"Best model saved to {model_path}")
            except Exception as e:
                logger.warning(f"Could not save model: {e}")
            
            # Create serializable results (already sorted)
            serializable_results = {}
            for name, data in sorted_results.items():
                serializable_results[name] = {
                    'training_score': float(data['training_score']),
                    'testing_score': float(data['testing_score']),
                    'cv_score': float(data['cv_score']),
                    'combined_score': float(data['combined_score']),
                    'metrics': self._convert_to_serializable(data['metrics']),
                    'train_time': float(data['train_time']),
                    'pred_time': float(data['pred_time'])
                }
            
            # Create serializable best model info
            best_model_info = {
                'name': best_name,
                'training_score': float(sorted_results[best_name]['training_score']),
                'testing_score': float(sorted_results[best_name]['testing_score']),
                'cv_score': float(sorted_results[best_name]['cv_score']),
                'combined_score': float(sorted_results[best_name]['combined_score']),
                'metrics': self._convert_to_serializable(sorted_results[best_name]['metrics']),
                'train_time': float(sorted_results[best_name]['train_time']),
                'pred_time': float(sorted_results[best_name]['pred_time'])
            }
            
            return serializable_results, best_model_info
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}", exc_info=True)
            raise
    
    def _calculate_metrics(
        self,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
        n_features: int
    ) -> Dict[str, float]:
        """Calculate all regression metrics."""
        def calc_metrics(y_true, y_pred, dataset_type):
            try:
                r2 = r2_score(y_true, y_pred)
                n = len(y_true)
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else 0
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                # MAPE - handle zeros
                y_true_adj = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
                mape = np.mean(np.abs((y_true_adj - y_pred) / y_true_adj)) * 100
                
                median_ae = median_absolute_error(y_true, y_pred)
                explained_var = explained_variance_score(y_true, y_pred)
                max_err = max_error(y_true, y_pred)
                
                return {
                    'r2': float(r2),
                    'adjusted_r2': float(adjusted_r2),
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'median_absolute_error': float(median_ae),
                    'explained_variance': float(explained_var),
                    'max_error': float(max_err)
                }
            except Exception as e:
                logger.warning(f"Error calculating metrics for {dataset_type}: {e}")
                return {
                    'r2': 0.0,
                    'adjusted_r2': 0.0,
                    'mae': 0.0,
                    'mse': 0.0,
                    'rmse': 0.0,
                    'mape': 0.0,
                    'median_absolute_error': 0.0,
                    'explained_variance': 0.0,
                    'max_error': 0.0
                }
        
        train_metrics = calc_metrics(y_train, y_train_pred, 'train')
        test_metrics = calc_metrics(y_test, y_test_pred, 'test')
        
        return {
            'train_r2': float(train_metrics['r2']),
            'test_r2': float(test_metrics['r2']),
            'train_adjusted_r2': float(train_metrics['adjusted_r2']),
            'test_adjusted_r2': float(test_metrics['adjusted_r2']),
            'train_mae': float(train_metrics['mae']),
            'test_mae': float(test_metrics['mae']),
            'train_mse': float(train_metrics['mse']),
            'test_mse': float(test_metrics['mse']),
            'train_rmse': float(train_metrics['rmse']),
            'test_rmse': float(test_metrics['rmse']),
            'train_mape': float(train_metrics['mape']),
            'test_mape': float(test_metrics['mape']),
            'train_median_absolute_error': float(train_metrics['median_absolute_error']),
            'test_median_absolute_error': float(test_metrics['median_absolute_error']),
            'train_explained_variance': float(train_metrics['explained_variance']),
            'test_explained_variance': float(test_metrics['explained_variance']),
            'train_max_error': float(train_metrics['max_error']),
            'test_max_error': float(test_metrics['max_error'])
        }
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform hyperparameter tuning on the best model."""
        try:
            if self.best_model is None:
                return {'message': 'No best model found. Train models first.'}
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) != len(X.columns):
                X = X[numeric_cols]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE
            )
            
            model_name = self.best_model['name']
            model_class = self.models[model_name]
            
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                },
                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [3, 5],
                    'reg_alpha': [0.5, 1.0],
                    'reg_lambda': [1.0, 2.0]
                },
                'LightGBM': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [3, 5],
                    'num_leaves': [31, 50],
                    'reg_alpha': [0.5, 1.0],
                    'reg_lambda': [1.0, 2.0]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
            
            if model_name not in param_grids:
                return {'message': f'No hyperparameter tuning available for {model_name}'}
            
            from sklearn.model_selection import GridSearchCV
            
            grid_search = GridSearchCV(
                model_class,
                param_grids[model_name],
                cv=min(5, len(X_train)),
                scoring='r2',
                n_jobs=config.N_JOBS
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_model['model'] = grid_search.best_estimator_
            self.best_model['best_params'] = grid_search.best_params_
            self.best_model['best_score'] = float(grid_search.best_score_)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': self._convert_to_serializable(grid_search.cv_results_)
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            return {'error': str(e)}