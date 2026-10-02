import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_DIR = BASE_DIR / 'uploads'
    REPORTS_DIR = BASE_DIR / 'reports'
    MODELS_DIR = BASE_DIR / 'saved_models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Create directories if they don't exist
    for dir_path in [UPLOAD_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # File upload settings
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}
    
    # Model settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Feature engineering settings
    LAG_FEATURES = [1, 7, 30]
    ROLLING_WINDOWS = [7, 14, 30]
    
    # Forecasting periods
    FORECAST_PERIODS = [7, 30, 90, 180, 365]
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = LOGS_DIR / 'app.log'
    
    # API settings
    API_TITLE = "Sales Forecasting Analytics Platform"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Enterprise-grade sales forecasting and analytics platform"
    
    # CORS settings
    ALLOWED_ORIGINS = ["*"]
    ALLOWED_METHODS = ["*"]
    ALLOWED_HEADERS = ["*"]

config = Config()