"""
PainCare AI Model Package
Main initialization module
"""

from .models.ai_model import PainCareAIModel
from .xai.explainer import XAIExplainer
from .services.firebase_service import FirebaseService
from .data.processor import DataProcessor
from .config import config, model_config, xai_config

__version__ = "1.0.0"
__author__ = "PainCare Development Team"

# Export main classes
__all__ = [
    "PainCareAIModel",
    "XAIExplainer", 
    "FirebaseService",
    "DataProcessor",
    "config",
    "model_config",
    "xai_config"
]
