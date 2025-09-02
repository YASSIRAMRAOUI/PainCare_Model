"""
Production configuration for PainCare AI Model
Optimized settings for production deployment
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ProductionConfig:
    """Production configuration settings"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG_MODE = False  # Always False in production
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # Validate required environment variables
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required in production")
    
    # Security Configuration
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 60))
    
    # Firebase Configuration
    FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if not FIREBASE_SERVICE_ACCOUNT_PATH:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH is required in production")
    
    # Model Configuration
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
    MODEL_UPDATE_INTERVAL = int(os.getenv("MODEL_UPDATE_INTERVAL", 86400))  # 24 hours
    MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", 10))
    
    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", 30))
    KEEPALIVE_TIMEOUT = int(os.getenv("KEEPALIVE_TIMEOUT", 2))
    
    # Redis Configuration (for caching)
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_URL = f"redis://{':{}'.format(REDIS_PASSWORD) if REDIS_PASSWORD else ''}{REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # Logging Configuration
    LOG_LEVEL = "INFO"  # Fixed for production
    LOG_FILE = "paincare_ai.log"
    
    # Monitoring Configuration
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    PROMETHEUS_METRICS_ENABLED = os.getenv("PROMETHEUS_METRICS_ENABLED", "True").lower() == "true"
    
    # XAI Configuration
    XAI_ENABLED = True
    EXPLANATION_FEATURES_COUNT = 5
    SHAP_BACKGROUND_SIZE = 100
    
    # Model Features
    FEATURE_COLUMNS = [
        "pain_level", "sleep_hours", "energy_level", "mood",
        "stress_level", "physical_activity", "medication_taken",
        "weather_pressure", "menstrual_cycle", "location_type",
        "symptom_duration", "treatment_adherence", "medication_effectiveness"
    ]
    
    TARGET_COLUMNS = ["pain_improvement", "treatment_effectiveness"]
    
    # Database Configuration (if using additional database)
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    @classmethod
    def validate(cls):
        """Validate production configuration"""
        required_vars = [
            "SECRET_KEY",
            "FIREBASE_SERVICE_ACCOUNT_PATH"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True


# Model-specific configuration
class ModelConfig:
    """ML Model configuration for production"""
    
    # Random Forest Configuration
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Gradient Boosting Configuration
    GRADIENT_BOOSTING_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    # K-Means Configuration
    KMEANS_PARAMS = {
        "n_clusters": 5,
        "random_state": 42,
        "n_init": 10,
        "max_iter": 300
    }
    
    # Model Validation
    CROSS_VALIDATION_FOLDS = 5
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Performance Thresholds
    MIN_ACCURACY = 0.75
    MIN_R2_SCORE = 0.6
    MIN_SILHOUETTE_SCORE = 0.2


# API Configuration
class APIConfig:
    """API-specific configuration for production"""
    
    # Request/Response Configuration
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 30  # seconds
    
    # Pagination
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Caching
    CACHE_TTL = 300  # 5 minutes
    ENABLE_RESPONSE_CACHE = True
    
    # API Versioning
    API_VERSION = "v1"
    API_PREFIX = f"/api/{API_VERSION}"
    
    # Documentation
    DOCS_URL = "/docs" if ProductionConfig.DEBUG_MODE else None
    REDOC_URL = "/redoc" if ProductionConfig.DEBUG_MODE else None


# Validate configuration on import
if os.getenv("ENVIRONMENT") == "production":
    ProductionConfig.validate()


# Export configurations
config = ProductionConfig()
model_config = ModelConfig()
api_config = APIConfig()
