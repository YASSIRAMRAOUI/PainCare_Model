"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add src to Python path for proper imports
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ai_model():
    """Mock AI model for testing"""
    model = Mock()
    model.is_trained = True
    model.last_update = None
    model.version = "test-1.0"
    model.get_model_status.return_value = "healthy"
    model.load_models = AsyncMock(return_value=True)
    model.predict = AsyncMock(return_value={
        "pain_level": 5,
        "recommendations": ["Rest", "Apply ice"],
        "confidence": 0.85
    })
    return model


@pytest.fixture
def mock_firebase_service():
    """Mock Firebase service for testing"""
    service = Mock()
    service.db = Mock()
    service.db.collection.return_value.limit.return_value.stream.return_value = []
    return service


@pytest.fixture
def mock_xai_explainer():
    """Mock XAI explainer for testing"""
    explainer = Mock()
    explainer.explain_prediction = AsyncMock(return_value={
        "feature_importance": {"symptom1": 0.8, "symptom2": 0.2},
        "explanation": "High pain level predicted due to severe symptoms"
    })
    return explainer
