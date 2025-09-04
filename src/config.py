"""
PainCare AI Model Configuration
Central configuration for the AI model system
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for PainCare AI Model"""
    
    # Firebase Configuration
    FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "path/to/firebase-credentials.json")
    FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL", "https://your-project.firebaseio.com/")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "172.23.111.34")  # Use the specific IP
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # External API keys (load from env; do not hardcode)
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Model Configuration
    MODEL_VERSION = "1.0.0"
    MODEL_UPDATE_INTERVAL = 24 * 60 * 60  # 24 hours in seconds
    MIN_DATA_POINTS = 10  # Minimum data points required for personalized recommendations
    
    # XAI Configuration
    XAI_ENABLED = True
    EXPLANATION_FEATURES_COUNT = 5  # Top N features to explain
    SHAP_BACKGROUND_SIZE = 100
    
    # Data Processing
    FEATURE_COLUMNS = [
        "pain_level", "sleep_hours", "energy_level", "mood",
        "stress_level", "physical_activity", "medication_taken",
        "weather_pressure", "menstrual_cycle", "location_type"
    ]
    
    TARGET_COLUMNS = ["pain_improvement", "treatment_effectiveness"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "paincare_ai.log"
    
    # Security
    ENCRYPT_DATA = True
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    RATE_LIMIT_PER_MINUTE = 100


class ModelConfig:
    """ML Model specific configuration"""
    
    MODELS = {
        "pain_predictor": {
            "type": "classification",
            "algorithm": "random_forest",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        },
        "treatment_recommender": {
            "type": "recommendation",
            "algorithm": "gradient_boosting",
            "parameters": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6
            }
        },
        "symptom_analyzer": {
            "type": "clustering",
            "algorithm": "kmeans",
            "parameters": {
                "n_clusters": 5,
                "random_state": 42
            }
        }
    }
    
    FEATURE_ENGINEERING = {
        "temporal_features": True,
        "interaction_features": True,
        "polynomial_features": False,
        "scaling": "standard"
    }


class XAIConfig:
    """Explainable AI configuration"""
    
    EXPLANATION_METHODS = {
        "lime": {
            "enabled": True,
            "num_features": 5,
            "num_samples": 1000
        },
        "shap": {
            "enabled": True,
            "explainer_type": "tree",
            "background_size": 100
        },
        "counterfactual": {
            "enabled": True,
            "max_iterations": 100,
            "tolerance": 0.01
        }
    }
    
    VISUALIZATION = {
        "charts": True,
        "interactive_plots": True,
        "export_format": "png"
    }


# Export configurations
config = Config()
model_config = ModelConfig()
xai_config = XAIConfig()
