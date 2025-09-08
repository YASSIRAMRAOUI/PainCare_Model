"""
FastAPI server for PainCare AI Model
Provides REST API endpoints for AI predictions and explanations
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
import asyncio
import pandas as pd

from ..models.ai_model import PainCareAIModel
from ..xai.explainer import XAIExplainer
from ..config import config
from ..services.firebase_service import FirebaseService
from .health import router as health_router, HealthChecker, set_health_checker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PainCare AI API",
    description="AI-powered pain management and recommendation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Global instances
ai_model = None
xai_explainer = None
firebase_service = None


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.method} {request.url}")
    logger.error(f"Request headers: {dict(request.headers)}")
    
    # Try to read the raw request body for debugging
    try:
        body = await request.body()
        logger.error(f"Raw request body: {body.decode()}")
        
        # Try to parse as JSON
        try:
            json_body = json.loads(body.decode())
            logger.error(f"Parsed JSON body: {json_body}")
            logger.error(f"JSON keys: {list(json_body.keys()) if isinstance(json_body, dict) else 'not a dict'}")
        except:
            logger.error("Could not parse request body as JSON")
            
    except Exception as e:
        logger.error(f"Could not read request body: {e}")
    
    logger.error(f"Validation errors: {exc.errors()}")
    
    # Return a more helpful error response
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Validation failed. Please check if you're sending 'symptoms' and 'diagnostic_tests' fields.",
            "expected_format": {
                "symptoms": [{"painLevel": 5, "location": "abdomen", "...": "other fields"}],
                "diagnostic_tests": [{"testType": "ovulation", "result": "positive", "...": "other fields"}]
            }
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize AI model and services on startup"""
    global ai_model, xai_explainer, firebase_service
    
    try:
        logger.info("Initializing PainCare AI services...")
        
        # Initialize Firebase service
        try:
            firebase_service = FirebaseService()
            logger.info("Firebase service initialized")
        except Exception as e:
            logger.error(f"Firebase service initialization failed: {e}")
            firebase_service = None
        
        # Initialize AI model
        try:
            ai_model = PainCareAIModel()
            logger.info("AI model initialized")
        except Exception as e:
            logger.error(f"AI model initialization failed: {e}")
            raise
        
        if ai_model is None:
            logger.error("AI model is None after initialization")
            raise RuntimeError("AI model initialization returned None")
        
        # Load pre-trained models if available
        try:
            logger.info(f"AI model object: {ai_model}")
            logger.info(f"load_models method: {ai_model.load_models}")
            result = await ai_model.load_models()
            logger.info(f"Model loading result: {result}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
        
        # Initialize XAI explainer
        try:
            xai_explainer = XAIExplainer(ai_model)
            logger.info("XAI explainer initialized")
        except Exception as e:
            logger.error(f"XAI explainer initialization failed: {e}")
            # Continue without XAI explainer for now
            xai_explainer = None
        
        # Initialize health checker
        try:
            health_checker_instance = HealthChecker(ai_model, firebase_service)
            set_health_checker(health_checker_instance)
            logger.info("Health checker initialized")
        except Exception as e:
            logger.error(f"Health checker initialization failed: {e}")
            # Continue without health checker for now
        
        logger.info("PainCare AI services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


# Pydantic models for request/response
class SymptomsRequest(BaseModel):
    pain_level: int = Field(..., ge=0, le=10, description="Pain level from 0-10")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep")
    stress_level: int = Field(..., ge=0, le=10, description="Stress level from 0-10")
    energy_level: int = Field(..., ge=0, le=10, description="Energy level from 0-10")
    mood: int = Field(..., ge=0, le=10, description="Mood level from 0-10")
    exercise: bool = Field(default=False, description="Did exercise today")
    medication_taken: bool = Field(default=False, description="Took medication")
    location: Optional[str] = Field(None, description="Pain location")
    triggers: Optional[List[str]] = Field(default=[], description="Identified triggers")
    notes: Optional[str] = Field(None, description="Additional notes")


class PredictionResponse(BaseModel):
    predicted_pain_level: int
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    cluster: int
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    timestamp: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_outcome: int = Field(..., ge=0, le=10)
    recommendation_helpful: bool
    recommendation_id: Optional[str] = None
    comments: Optional[str] = None


class ModelStatusResponse(BaseModel):
    is_trained: bool
    last_update: Optional[str]
    model_version: str
    models_available: List[str]
    update_required: bool


# New models for ML explanations and insights
class MLExplanationRequest(BaseModel):
    symptom_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Direct symptom data list")
    symptoms: Optional[List[Dict[str, Any]]] = Field(default=None, description="Symptoms from mobile app")
    diagnostic_tests: Optional[List[Dict[str, Any]]] = Field(default=None, description="Diagnostic tests from mobile app")
    research_sources: bool = Field(default=True, description="Enable internet research")
    include_citations: bool = Field(default=True, description="Include source citations")


class MLInsightRequest(BaseModel):
    symptom_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Direct symptom data list")
    symptoms: Optional[List[Dict[str, Any]]] = Field(default=None, description="Symptoms from mobile app")
    diagnostic_tests: Optional[List[Dict[str, Any]]] = Field(default=None, description="Diagnostic tests from mobile app")
    research_enabled: bool = Field(default=True, description="Enable internet research for evidence")
    evidence_threshold: float = Field(default=0.7, description="Confidence threshold for evidence")
    include_sources: bool = Field(default=True, description="Include research sources")


class MLSource(BaseModel):
    title: str
    url: str
    type: str
    relevance_score: float


class MLExplanation(BaseModel):
    topic: str
    explanation: str
    model_reasoning: str
    confidence: float
    sources: List[MLSource]
    evidence_strength: str
    last_updated: str


class MLSupportingEvidence(BaseModel):
    source_title: str
    source_url: str
    credibility_rating: float
    publication_date: str


class MLInsight(BaseModel):
    category: str
    insight: str
    confidence: float
    supporting_evidence: List[MLSupportingEvidence]
    recommended_actions: List[str]
    model_version: str


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase JWT token"""
    try:
        # In production, verify the JWT token with Firebase Admin SDK
        token = credentials.credentials
        
        # For now, return a mock user_id
        # In production: decoded_token = auth.verify_id_token(token)
        # return decoded_token['uid']
        
        return "mock_user_id"  # Replace with actual user verification
        
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": ai_model.get_model_status() if ai_model else "not_initialized"
    }


@app.post("/predict/pain")
async def predict_pain(
    request: Request,
    include_explanation: bool = True
):
    """
    Predict pain level based on current symptoms - supports both authenticated and diagnostic requests
    """
    try:
        # Parse request body manually to handle different formats
        body = await request.body()
        
        if len(body) == 0:
            # If no body, this might be a diagnostic test request - provide fallback
            return {
                "predicted_pain_level": 5,
                "confidence": 0.6,
                "probabilities": [0.2, 0.6, 0.2],
                "explanation": "AI model is training with more data to provide better predictions. This is a preliminary assessment based on general patterns.",
                "timestamp": datetime.now().isoformat(),
                "model_status": "training"
            }
        
        # Parse JSON body
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"Predict pain endpoint called with data: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in predict/pain: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract user ID if provided, otherwise use default for diagnostics
        user_id = json_data.get('userId', json_data.get('user_id', 'tZU34vbAyCZID8YjQSNIeZXorJA2'))
        
        # Check if AI model is available and trained
        if not ai_model or not getattr(ai_model, 'is_trained', False):
            # Provide intelligent fallback response
            symptoms = json_data.get('symptoms', json_data)
            
            # Extract basic symptom information for fallback prediction
            pain_level = symptoms.get('pain_level', symptoms.get('painLevel', 5))
            stress_level = symptoms.get('stress_level', symptoms.get('stressLevel', 5)) 
            energy_level = symptoms.get('energy_level', symptoms.get('energyLevel', 5))
            
            # Simple heuristic-based prediction
            predicted_pain = min(10, max(1, int((pain_level + stress_level - energy_level) / 2)))
            confidence = 0.6  # Lower confidence for fallback
            
            return {
                "predicted_pain_level": predicted_pain,
                "confidence": confidence,
                "probabilities": [0.3, 0.4, 0.3],  # Balanced probabilities
                "explanation": "Assessment based on reported symptoms. AI model is training with more data to provide better predictions." if include_explanation else None,
                "timestamp": datetime.now().isoformat(),
                "model_status": "training",
                "fallback_used": True
            }
        
        # If model is trained, use it
        # Convert JSON data to symptoms format
        symptoms_dict = json_data.get('symptoms', json_data)
        
        # Get prediction from trained model
        prediction = await ai_model.predict_pain_level(user_id, symptoms_dict)
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        # Generate explanation if requested
        explanation = None
        if include_explanation and xai_explainer:
            try:
                explanation = await xai_explainer.explain_pain_prediction(
                    user_id, symptoms_dict, prediction
                )
            except Exception as e:
                logger.warning(f"Could not generate explanation: {e}")
                explanation = "Explanation generation temporarily unavailable."
        
        return {
            "predicted_pain_level": prediction["predicted_pain_level"],
            "confidence": prediction["confidence"],
            "probabilities": prediction.get("probabilities", [0.3, 0.4, 0.3]),
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "model_status": "trained"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pain prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Public health check endpoint - no authentication required
    """
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": {
                "ai_model_loaded": ai_model is not None,
                "xai_explainer_loaded": xai_explainer is not None,
                "firebase_connected": firebase_service is not None
            }
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/recommend/treatment")
async def recommend_treatment(request: Request):
    """
    Get personalized treatment recommendations - mobile app compatible
    """
    try:
        # Parse request body manually to handle both mobile app format and Pydantic format
        body = await request.body()
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        try:
            json_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in recommend/treatment: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract user ID and symptoms - handle both formats
        user_id = json_data.get('userId', json_data.get('user_id', 'tZU34vbAyCZID8YjQSNIeZXorJA2'))
        symptoms_data = json_data.get('symptoms', json_data)
        include_explanation = True  # Always include explanation for mobile
        
        if not ai_model:
            # Provide fallback recommendations when model is not available
            return {
                "recommendations": [
                    {
                        "type": "therapy",
                        "title": "Heat/Cold Therapy", 
                        "description": "Apply heat pad for muscle pain or cold for inflammation",
                        "evidence_level": "high",
                        "references": ["Mayo Clinic Guidelines", "American Pain Society"],
                        "confidence": 0.85
                    },
                    {
                        "type": "mindfulness",
                        "title": "Breathing Exercises",
                        "description": "5-10 minutes of deep breathing or meditation", 
                        "evidence_level": "medium",
                        "references": ["Journal of Pain Management", "Mindfulness Research"],
                        "confidence": 0.75
                    }
                ],
                "cluster": 0,
                "confidence": 0.8,
                "explanation": {
                    "recommendations": {
                        "cluster": 0,
                        "confidence": 0.8,
                        "recommendations": [
                            {
                                "type": "therapy",
                                "title": "Heat/Cold Therapy", 
                                "description": "Apply heat pad for muscle pain or cold for inflammation",
                                "evidence_level": "high",
                                "references": ["Mayo Clinic Guidelines", "American Pain Society"],
                                "confidence": 0.85
                            },
                            {
                                "type": "mindfulness", 
                                "title": "Breathing Exercises",
                                "description": "5-10 minutes of deep breathing or meditation",
                                "evidence_level": "medium", 
                                "references": ["Journal of Pain Management", "Mindfulness Research"],
                                "confidence": 0.75
                            }
                        ],
                        "timestamp": datetime.now().isoformat()
                    },
                    "explanations": [
                        {
                            "topic": "Treatment Selection",
                            "explanation": "Based on general pain management principles, these treatments are commonly effective for managing endometriosis-related pain.",
                            "confidence": 0.8,
                            "evidence_strength": "strong"
                        },
                        {
                            "topic": "Personalization",
                            "explanation": "AI model is training to provide more personalized recommendations based on your specific symptoms and response patterns.",
                            "confidence": 0.6,
                            "evidence_strength": "moderate"
                        }
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Get recommendations from trained model
        recommendations = await ai_model.recommend_treatment(user_id, symptoms_data)
        
        if "error" in recommendations:
            raise HTTPException(status_code=500, detail=recommendations["error"])
        
        # Generate explanation if requested
        explanation = None
        if include_explanation and xai_explainer:
            try:
                explanation = await xai_explainer.explain_treatment_recommendation(
                    user_id, symptoms_data, recommendations
                )
            except Exception as e:
                logger.warning(f"Could not generate explanation: {e}")
                explanation = {
                    "recommendations": recommendations,
                    "explanations": [
                        {
                            "topic": "Treatment Recommendation",
                            "explanation": "Recommendations based on AI analysis of your symptoms and patterns.",
                            "confidence": recommendations["confidence"],
                            "evidence_strength": "moderate"
                        }
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
        
        # Extract cluster information safely
        cluster_info = recommendations.get("user_context", {}).get("symptom_cluster", {})
        cluster_number = cluster_info.get("cluster", 0)
        
        return {
            "recommendations": recommendations["recommendations"],
            "cluster": cluster_number, 
            "confidence": recommendations["confidence"],
            "explanation": explanation,
            "timestamp": recommendations["timestamp"],
            "user_context": recommendations.get("user_context", {}),
            "primary_recommendation": recommendations.get("primary_recommendation", "general_care")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in treatment recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    user_id: str = Depends(verify_token)
):
    """
    Submit user feedback for model improvement
    """
    try:
        feedback_data = {
            **feedback.dict(),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await firebase_service.save_user_feedback(user_id, feedback_data)
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pydantic model for the new endpoint
class UserDataSubmissionRequest(BaseModel):
    type: str = Field(..., description="Request type")
    symptoms_anonymized: Optional[Dict[str, Any]] = Field(default=None, description="Anonymized symptom data")
    recommendations_count: Optional[int] = Field(default=1, description="Number of recommendations requested")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Request metadata", alias="_metadata")


@app.post("/user/submit-data")
async def submit_user_data(request: Request):
    """
    Handle user data submission for analytics and model improvement
    """
    try:
        # Parse request body manually to handle different formats
        body = await request.body()
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"User data submission received: {json_data.get('type', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Handle different types of data submissions
        request_type = json_data.get('type', 'unknown')
        
        if request_type == 'recommendation_request':
            # Handle recommendation request type
            symptoms = json_data.get('symptoms_anonymized', {})
            metadata = json_data.get('_metadata', {})
            
            # Log the data for analytics (in production, save to database)
            logger.info(f"Recommendation request received - shared_for_improvement: {metadata.get('shared_for_improvement', False)}")
            
            # If this is meant to be processed, we could call appropriate endpoints
            # For now, just acknowledge receipt
            return {
                "message": "User data received successfully",
                "type": request_type,
                "processed": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Handle other types of submissions
            logger.info(f"Unknown data submission type: {request_type}")
            return {
                "message": "Data received but type not recognized",
                "type": request_type,
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user data submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model status and information - public endpoint
    """
    try:
        if not ai_model:
            raise HTTPException(
                status_code=503,
                detail="AI model not available"
            )
        
        status = ai_model.get_model_status()
        
        return ModelStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/train")
async def trigger_model_training():
    """
    Trigger model retraining - Public endpoint for legitimate model improvement requests
    """
    try:
        if not ai_model:
            raise HTTPException(
                status_code=503,
                detail="AI model not available"
            )
        
        # Add rate limiting in production to prevent abuse
        # For now, allow training requests to improve the model
        
        # Start training in background
        asyncio.create_task(ai_model.train_models())
        
        return {"message": "Model training started"}
        
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/xai/report")
async def get_interpretability_report(user_id: str = Depends(verify_token)):
    """
    Get model interpretability and transparency report
    """
    try:
        if not xai_explainer:
            raise HTTPException(
                status_code=503,
                detail="XAI service not available"
            )
        
        report = xai_explainer.get_model_interpretability_report()
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating interpretability report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/xai/explain")
async def get_xai_explanations(request: Request):
    """
    Generate XAI explanations for a prediction - mobile app compatible
    """
    try:
        # Parse request body manually to handle mobile app format
        body = await request.body()
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        try:
            json_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in xai/explain: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        if not xai_explainer:
            raise HTTPException(
                status_code=503,
                detail="XAI service not available"
            )
        
        # Extract symptoms - handle both formats
        symptoms = json_data.get('symptoms', json_data)
        user_id = json_data.get('userId', json_data.get('user_id', 'tZU34vbAyCZID8YjQSNIeZXorJA2'))
        
        # Make a simple prediction first 
        if not ai_model:
            # Simple fallback explanation when model not available
            return {
                "prediction": {
                    "predicted_pain_level": 5,
                    "confidence": 0.6,
                    "explanation": "AI model is training. Explanation will be more detailed once training is complete."
                },
                "explanations": {
                    "feature_importance": {
                        "pain_level": 0.3,
                        "stress_level": 0.25,
                        "sleep_hours": 0.2,
                        "energy_level": 0.15,
                        "mood": 0.1
                    },
                    "natural_explanation": "Based on general patterns, several factors may influence your pain levels. Tracking more symptoms will provide better insights."
                },
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "model_version": "training"
            }
        
        # Get prediction first
        prediction = await ai_model.predict_pain_level(user_id, symptoms)
        
        # Generate XAI explanation
        explanation = await xai_explainer.explain_pain_prediction(user_id, symptoms, prediction)
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating XAI explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/xai/feature-importance")
async def get_feature_importance():
    """
    Get feature importance for the current model
    """
    try:
        if not ai_model or not ai_model.is_trained:
            # Return generic feature importance when model is not trained
            return {
                "features": {
                    "pain_level": 0.30,
                    "stress_level": 0.25,
                    "sleep_hours": 0.20,
                    "energy_level": 0.15,
                    "mood": 0.10
                },
                "model_status": "training",
                "explanation": "Feature importance will be more accurate once the AI model completes training with your data."
            }
        
        if not xai_explainer:
            raise HTTPException(
                status_code=503,
                detail="XAI service not available"
            )
        
        # Get feature importance from the model
        feature_importance = xai_explainer.get_global_feature_importance()
        
        return {
            "features": feature_importance,
            "model_status": "trained"
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/{user_id}/symptoms/recent")
async def get_user_recent_symptoms(user_id: str):
    """
    Get user's recent symptom tracking data for ML analysis
    """
    try:
        if not firebase_service:
            # Return mock data for testing if Firebase is not available
            logger.warning("Firebase service not available, returning mock symptom data")
            return [
                {
                    "pain_level": 6,
                    "sleep_hours": 7.5,
                    "stress_level": 7,
                    "energy_level": 4,
                    "mood": 5,
                    "exercise": False,
                    "medication_taken": True,
                    "location": "lower abdomen",
                    "triggers": ["stress", "weather"],
                    "notes": "Moderate pain during work day",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "pain_level": 4,
                    "sleep_hours": 8.0,
                    "stress_level": 5,
                    "energy_level": 6,
                    "mood": 7,
                    "exercise": True,
                    "medication_taken": False,
                    "location": "pelvic area",
                    "triggers": ["hormonal"],
                    "notes": "Better after rest",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        # Get recent symptom data from Firestore (last 30 days)
        symptoms_data = await firebase_service.get_user_symptoms_recent(user_id, days=30)
        
        # Also get diagnostic test data
        diagnostic_data = await firebase_service.get_user_diagnostic_tests(user_id, days=30)
        
        logger.info(f"Retrieved {len(symptoms_data)} symptoms and {len(diagnostic_data)} diagnostic tests for user {user_id}")
        
        # Combine the data for ML analysis
        combined_data = {
            "symptoms": symptoms_data,
            "diagnostic_tests": diagnostic_data
        }
        
        if not symptoms_data and not diagnostic_data:
            logger.warning(f"No data found for user {user_id}")
            return []
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error retrieving user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def normalize_symptom_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize field names from mobile app format to AI model format
    """
    normalized_data = []
    
    for item in data:
        normalized_item = {}
        for key, value in item.items():
            # Convert camelCase to snake_case for AI model compatibility
            if key == 'painLevel':
                normalized_item['pain_level'] = value
            elif key == 'userId':
                normalized_item['user_id'] = value
            elif key == 'recordedAt':
                normalized_item['recorded_at'] = value
            elif key == 'createdAt':
                normalized_item['created_at'] = value
            else:
                # Convert other camelCase keys to snake_case
                snake_case_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                normalized_item[snake_case_key] = value
        
        normalized_data.append(normalized_item)
    
    return normalized_data


@app.post("/ml/{user_id}/explanations", response_model=List[MLExplanation])
async def get_ml_explanations(user_id: str, request: MLExplanationRequest):
    """
    Get ML-generated explanations with internet research and sources
    """
    try:
        if not ai_model:
            raise HTTPException(
                status_code=503,
                detail="AI model not available"
            )
        
        # Get user's actual data from Firestore
        user_symptoms = []
        user_diagnostic_tests = []
        
        if firebase_service:
            user_symptoms = await firebase_service.get_user_symptoms_recent(user_id, days=30)
            user_diagnostic_tests = await firebase_service.get_user_diagnostic_tests(user_id, days=30)
            
            logger.info(f"Using {len(user_symptoms)} symptoms from Firestore for ML explanations")
        
        # Determine which symptom data to use (prioritize mobile app data, then Firestore data)
        final_symptom_data = []
        
        if request.symptoms:
            # Mobile app sent symptoms in the new format - normalize field names
            final_symptom_data = normalize_symptom_data(request.symptoms)
            logger.info(f"Using {len(final_symptom_data)} symptoms from mobile app request (normalized)")
        elif request.symptom_data:
            # Direct symptom_data format - normalize field names
            final_symptom_data = normalize_symptom_data(request.symptom_data)
            logger.info(f"Using {len(final_symptom_data)} symptoms from direct symptom_data (normalized)")
        elif user_symptoms:
            # Use Firestore data as fallback - already in correct format
            final_symptom_data = user_symptoms
            logger.info(f"Using {len(final_symptom_data)} symptoms from Firestore")
        
        # Also include diagnostic test data if available (combine with symptom data)
        diagnostic_data = request.diagnostic_tests if request.diagnostic_tests else user_diagnostic_tests
        
        # Combine symptom and diagnostic data for analysis
        combined_data = final_symptom_data.copy() if final_symptom_data else []
        if diagnostic_data:
            # Add diagnostic test info to the analysis
            for test in diagnostic_data:
                # Extract relevant fields from diagnostic data for ML analysis
                diagnostic_summary = {
                    'diagnostic_test_id': test.get('id'),
                    'diagnostic_completed': test.get('isCompleted', False),
                    'diagnostic_progress': test.get('currentQuestion', 0) / test.get('totalQuestions', 1) if test.get('totalQuestions') else 0,
                    'diagnostic_date': test.get('date'),
                }
                
                # Add diagnostic answers as features if available
                if 'answers' in test and test['answers']:
                    answers = test['answers']
                    diagnostic_summary.update({
                        'has_bleeding_spotting': answers.get('bleeding_spotting') == 'yes',
                        'intercourse_pain_level': answers.get('intercourse_pain', '0'),
                        'cycle_pattern': answers.get('cycle_pattern', 'unknown'),
                        'period_nature': answers.get('period_nature', 'unknown'),
                        'family_history': answers.get('family_history') == 'yes',
                    })
                
                combined_data.append(diagnostic_summary)
        
        # Generate ML explanations with research
        explanations = await ai_model.generate_ml_explanations(
            user_id=user_id,
            symptom_data=combined_data,
            research_sources=request.research_sources,
            include_citations=request.include_citations
        )
        
        return explanations
        
    except Exception as e:
        logger.error(f"Error generating ML explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/{user_id}/insights", response_model=List[MLInsight])
async def get_ml_insights(user_id: str, request: MLInsightRequest):
    """
    Get ML-generated insights with evidence-based recommendations
    """
    try:
        if not ai_model:
            raise HTTPException(
                status_code=503,
                detail="AI model not available"
            )
        
        # Get user's actual data from Firestore
        user_symptoms = []
        user_diagnostic_tests = []
        
        if firebase_service:
            user_symptoms = await firebase_service.get_user_symptoms_recent(user_id, days=30)
            user_diagnostic_tests = await firebase_service.get_user_diagnostic_tests(user_id, days=30)
                
            logger.info(f"Using {len(user_symptoms)} symptoms from Firestore for ML insights")
        
        # Determine which symptom data to use (prioritize mobile app data, then Firestore data)
        final_symptom_data = []
        
        if request.symptoms:
            # Mobile app sent symptoms in the new format - normalize field names
            final_symptom_data = normalize_symptom_data(request.symptoms)
            logger.info(f"Using {len(final_symptom_data)} symptoms from mobile app request (normalized)")
        elif request.symptom_data:
            # Direct symptom_data format - normalize field names
            final_symptom_data = normalize_symptom_data(request.symptom_data)
            logger.info(f"Using {len(final_symptom_data)} symptoms from direct symptom_data (normalized)")
        elif user_symptoms:
            # Use Firestore data as fallback - already in correct format
            final_symptom_data = user_symptoms
            logger.info(f"Using {len(final_symptom_data)} symptoms from Firestore")
        
        # Also include diagnostic test data if available (combine with symptom data)
        diagnostic_data = request.diagnostic_tests if request.diagnostic_tests else user_diagnostic_tests
        
        # Combine symptom and diagnostic data for analysis
        combined_data = final_symptom_data.copy() if final_symptom_data else []
        if diagnostic_data:
            # Add diagnostic test info to the analysis
            for test in diagnostic_data:
                # Extract relevant fields from diagnostic data for ML analysis
                diagnostic_summary = {
                    'diagnostic_test_id': test.get('id'),
                    'diagnostic_completed': test.get('isCompleted', False),
                    'diagnostic_progress': test.get('currentQuestion', 0) / test.get('totalQuestions', 1) if test.get('totalQuestions') else 0,
                    'diagnostic_date': test.get('date'),
                }
                
                # Add diagnostic answers as features if available
                if 'answers' in test and test['answers']:
                    answers = test['answers']
                    diagnostic_summary.update({
                        'has_bleeding_spotting': answers.get('bleeding_spotting') == 'yes',
                        'intercourse_pain_level': answers.get('intercourse_pain', '0'),
                        'cycle_pattern': answers.get('cycle_pattern', 'unknown'),
                        'period_nature': answers.get('period_nature', 'unknown'),
                        'family_history': answers.get('family_history') == 'yes',
                    })
                
                combined_data.append(diagnostic_summary)
        
        # Generate ML insights with research backing
        insights = await ai_model.generate_ml_insights(
            user_id=user_id,
            symptom_data=combined_data,
            research_enabled=request.research_enabled,
            evidence_threshold=request.evidence_threshold,
            include_sources=request.include_sources
        )
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating ML insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/user/{user_id}/data")
async def debug_user_data(user_id: str):
    """
    Debug endpoint to check what data exists for a user
    """
    try:
        if not firebase_service:
            return {"error": "Firebase service not available"}
        
        debug_info = {}
        
        def fetch_debug_data():
            # Check Symptoms collection
            symptoms_ref = firebase_service.db.collection('Symptoms')
            symptoms_query = symptoms_ref.where('userId', '==', user_id).limit(5)
            symptoms_docs = list(symptoms_query.stream())
            
            debug_info['symptoms'] = {
                'count': len(symptoms_docs),
                'sample_data': []
            }
            
            for doc in symptoms_docs[:2]:  # Just show first 2
                data = doc.to_dict()
                debug_info['symptoms']['sample_data'].append({
                    'id': doc.id,
                    'keys': list(data.keys()),
                    'recordedAt_type': str(type(data.get('recordedAt', 'N/A'))),
                    'recordedAt_value': str(data.get('recordedAt', 'N/A'))
                })
            
            # Check DiagnosisTests collection
            tests_ref = firebase_service.db.collection('DiagnosisTests')
            tests_query = tests_ref.where('userId', '==', user_id).limit(5)
            tests_docs = list(tests_query.stream())
            
            debug_info['diagnosis_tests'] = {
                'count': len(tests_docs),
                'sample_data': []
            }
            
            for doc in tests_docs[:2]:  # Just show first 2
                data = doc.to_dict()
                debug_info['diagnosis_tests']['sample_data'].append({
                    'id': doc.id,
                    'keys': list(data.keys()),
                    'updatedAt_type': str(type(data.get('updatedAt', 'N/A'))),
                    'updatedAt_value': str(data.get('updatedAt', 'N/A'))
                })
            
            # Check Diagnostic collection
            diagnostic_ref = firebase_service.db.collection('Diagnostic')
            diagnostic_query = diagnostic_ref.where('userId', '==', user_id).limit(5)
            diagnostic_docs = list(diagnostic_query.stream())
            
            debug_info['diagnostic'] = {
                'count': len(diagnostic_docs),
                'sample_data': []
            }
            
            for doc in diagnostic_docs[:2]:  # Just show first 2
                data = doc.to_dict()
                debug_info['diagnostic']['sample_data'].append({
                    'id': doc.id,
                    'keys': list(data.keys()),
                    'updatedAt_type': str(type(data.get('updatedAt', 'N/A'))),
                    'updatedAt_value': str(data.get('updatedAt', 'N/A'))
                })
            
            return debug_info
        
        result = await asyncio.get_event_loop().run_in_executor(
            firebase_service.executor, fetch_debug_data
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return {"error": str(e)}


@app.post("/mobile/explanations")
async def get_ml_explanations_mobile(request: Request):
    """
    Mobile app specific endpoint - handles mobile app format with zero validation issues
    """
    try:
        # Get raw request body and parse manually
        body = await request.body()
        logger.info(f"Mobile explanations endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")
        logger.info(f"User-Agent: {request.headers.get('user-agent', 'Unknown')}")
        logger.info(f"Raw body length: {len(body)} bytes")
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Parse JSON manually
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"Successfully parsed JSON with keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract data with safe fallbacks
        symptoms_data = json_data.get('symptoms', [])
        diagnostic_data = json_data.get('diagnostic_tests', [])
        
        logger.info(f"Extracted {len(symptoms_data)} symptoms and {len(diagnostic_data)} diagnostic tests")
        
        # Validate that we have some data to work with
        if not symptoms_data and not diagnostic_data:
            return []
        
        # Create the request object manually to avoid any Pydantic validation issues
        request_data = {
            'symptom_data': None,
            'symptoms': symptoms_data,
            'diagnostic_tests': diagnostic_data,
            'research_sources': json_data.get('research_sources', True),
            'include_citations': json_data.get('include_citations', True)
        }
        
        # Manually call the logic without going through Pydantic validation
        user_id = "tZU34vbAyCZID8YjQSNIeZXorJA2"
        
        # Call get_ml_explanations with manually constructed request
        explanation_request = MLExplanationRequest(**request_data)
        result = await get_ml_explanations(user_id, explanation_request)
        
        logger.info(f"Successfully generated {len(result)} explanations for mobile app")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in mobile explanations endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/mobile/insights")
async def get_ml_insights_mobile(request: Request):
    """
    Mobile app specific endpoint for insights - handles mobile app format with zero validation issues
    """
    try:
        # Get raw request body and parse manually
        body = await request.body()
        logger.info(f"Mobile insights endpoint called")
        logger.info(f"Raw body length: {len(body)} bytes")
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Parse JSON manually
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"Successfully parsed JSON with keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract data with safe fallbacks
        symptoms_data = json_data.get('symptoms', [])
        diagnostic_data = json_data.get('diagnostic_tests', [])
        
        logger.info(f"Extracted {len(symptoms_data)} symptoms and {len(diagnostic_data)} diagnostic tests")
        
        # Validate that we have some data to work with
        if not symptoms_data and not diagnostic_data:
            return []
        
        # Create the request object manually to avoid any Pydantic validation issues
        request_data = {
            'symptom_data': None,
            'symptoms': symptoms_data,
            'diagnostic_tests': diagnostic_data,
            'research_enabled': json_data.get('research_enabled', True),
            'evidence_threshold': json_data.get('evidence_threshold', 0.7),
            'include_sources': json_data.get('include_sources', True)
        }
        
        # Manually call the logic without going through Pydantic validation
        user_id = "tZU34vbAyCZID8YjQSNIeZXorJA2"
        
        # Call get_ml_insights with manually constructed request
        insights_request = MLInsightRequest(**request_data)
        result = await get_ml_insights(user_id, insights_request)
        
        logger.info(f"Successfully generated {len(result)} insights for mobile app")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in mobile insights endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Backward compatibility routes - these will be deprecated
@app.get("/user/symptoms/recent")
async def get_user_recent_symptoms_legacy():
    """
    Legacy endpoint - redirects to new user-specific endpoint
    Uses default user ID from logs for backward compatibility
    """
    # Use the user ID from your logs
    user_id = "tZU34vbAyCZID8YjQSNIeZXorJA2"
    return await get_user_recent_symptoms(user_id)


@app.post("/ml/explanations")
async def get_ml_explanations_legacy(request: Request):
    """
    Legacy endpoint - handles mobile app format with zero validation issues
    """
    try:
        # Get raw request body and parse manually - same logic as mobile endpoints
        body = await request.body()
        logger.info(f"Legacy ML explanations endpoint called")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request URL: {request.url}")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")
        logger.info(f"User-Agent: {request.headers.get('user-agent', 'Unknown')}")
        logger.info(f"Raw body length: {len(body)} bytes")
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Parse JSON manually
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"Successfully parsed JSON with keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract data with safe fallbacks
        symptoms_data = json_data.get('symptoms', [])
        diagnostic_data = json_data.get('diagnostic_tests', [])
        
        logger.info(f"Extracted {len(symptoms_data)} symptoms and {len(diagnostic_data)} diagnostic tests")
        
        # Validate that we have some data to work with
        if not symptoms_data and not diagnostic_data:
            return []
        
        # Create the request object manually to avoid any Pydantic validation issues
        request_data = {
            'symptom_data': None,
            'symptoms': symptoms_data,
            'diagnostic_tests': diagnostic_data,
            'research_sources': json_data.get('research_sources', True),
            'include_citations': json_data.get('include_citations', True)
        }
        
        # Manually call the logic without going through Pydantic validation
        user_id = "tZU34vbAyCZID8YjQSNIeZXorJA2"
        
        # Call get_ml_explanations with manually constructed request
        explanation_request = MLExplanationRequest(**request_data)
        result = await get_ml_explanations(user_id, explanation_request)
        
        logger.info(f"Successfully generated {len(result)} explanations for legacy endpoint")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in legacy explanations endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/ml/insights")
async def get_ml_insights_legacy(request: Request):
    """
    Legacy endpoint - handles mobile app format with zero validation issues
    """
    try:
        # Get raw request body and parse manually - same logic as mobile endpoints
        body = await request.body()
        logger.info(f"Legacy ML insights endpoint called")
        logger.info(f"Raw body length: {len(body)} bytes")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")
        
        if len(body) == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Parse JSON manually
        try:
            json_data = json.loads(body.decode('utf-8'))
            logger.info(f"Successfully parsed JSON with keys: {list(json_data.keys()) if isinstance(json_data, dict) else type(json_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
        
        # Extract data with safe fallbacks
        symptoms_data = json_data.get('symptoms', [])
        diagnostic_data = json_data.get('diagnostic_tests', [])
        
        logger.info(f"Extracted {len(symptoms_data)} symptoms and {len(diagnostic_data)} diagnostic tests")
        
        # Validate that we have some data to work with
        if not symptoms_data and not diagnostic_data:
            return []
        
        # Create the request object manually to avoid any Pydantic validation issues
        request_data = {
            'symptom_data': None,
            'symptoms': symptoms_data,
            'diagnostic_tests': diagnostic_data,
            'research_enabled': json_data.get('research_enabled', True),
            'evidence_threshold': json_data.get('evidence_threshold', 0.7),
            'include_sources': json_data.get('include_sources', True)
        }
        
        # Manually call the logic without going through Pydantic validation
        user_id = "tZU34vbAyCZID8YjQSNIeZXorJA2"
        
        # Call get_ml_insights with manually constructed request
        insights_request = MLInsightRequest(**request_data)
        result = await get_ml_insights(user_id, insights_request)
        
        logger.info(f"Successfully generated {len(result)} insights for legacy endpoint")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in legacy insights endpoint: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# New user-specific prediction endpoints
@app.post("/predict/{user_id}/pain")
async def predict_user_pain(user_id: str, request: SymptomsRequest):
    """
    Predict pain level for specific user with personalized AI model
    """
    try:
        if not ai_model:
            raise HTTPException(status_code=503, detail="AI model not available")
        
        # Convert request to symptoms dictionary
        symptoms_dict = {
            'pain_level': request.pain_level,
            'sleep_hours': request.sleep_hours,
            'energy_level': request.energy_level,
            'mood': 'happy' if request.mood > 7 else 'sad' if request.mood < 4 else 'neutral',
            'pain_location': request.location or 'unknown'
        }
        
        # Get personalized prediction
        prediction = await ai_model.predict_pain_level(user_id, symptoms_dict)
        
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user pain prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/{user_id}/treatment")
async def recommend_user_treatment(user_id: str, request: SymptomsRequest):
    """
    Get personalized treatment recommendations for specific user
    """
    try:
        if not ai_model:
            raise HTTPException(status_code=503, detail="AI model not available")
        
        # Convert request to symptoms dictionary
        symptoms_dict = {
            'pain_level': request.pain_level,
            'sleep_hours': request.sleep_hours,
            'energy_level': request.energy_level,
            'mood': 'happy' if request.mood > 7 else 'sad' if request.mood < 4 else 'neutral',
            'pain_location': request.location or 'unknown'
        }
        
        # Get personalized recommendations
        recommendations = await ai_model.recommend_treatment(user_id, symptoms_dict)
        
        if "error" in recommendations:
            raise HTTPException(status_code=500, detail=recommendations["error"])
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user treatment recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/{user_id}/symptoms")
async def analyze_user_symptoms(user_id: str, request: SymptomsRequest):
    """
    Analyze symptom patterns for specific user using clustering
    """
    try:
        if not ai_model:
            raise HTTPException(status_code=503, detail="AI model not available")
        
        # Convert request to symptoms dictionary
        symptoms_dict = {
            'pain_level': request.pain_level,
            'sleep_hours': request.sleep_hours,
            'energy_level': request.energy_level,
            'mood': 'happy' if request.mood > 7 else 'sad' if request.mood < 4 else 'neutral',
            'pain_location': request.location or 'unknown'
        }
        
        # Get personalized symptom analysis
        analysis = await ai_model.analyze_symptoms(symptoms_dict, user_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in user symptom analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}/insights")
async def get_user_profile_insights(user_id: str):
    """
    Get comprehensive user profile insights combining all AI models
    """
    try:
        if not ai_model or not firebase_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get user's complete profile and history
        user_profile = await firebase_service.get_user_complete_profile(user_id)
        user_history = await ai_model.fetch_user_data(user_id, days=90)
        
        if user_history.empty:
            return {
                "user_id": user_id,
                "message": "No symptom data available for analysis",
                "recommendations": "Please track symptoms for better insights",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate user metrics
        avg_pain = user_history['painLevel'].mean()
        pain_trend = "improving" if user_history['painLevel'].tail(7).mean() < user_history['painLevel'].head(7).mean() else "stable"
        sleep_quality = "good" if user_history['sleep'].mean() > 7 else "poor"
        
        # Get AI-based insights
        recent_symptoms = {
            'pain_level': int(user_history['painLevel'].iloc[-1]),
            'sleep_hours': user_history['sleep'].iloc[-1],
            'energy_level': int(user_history['energy'].iloc[-1]),
            'mood': user_history['mood'].iloc[-1]
        }
        
        # Get symptom analysis
        symptom_analysis = await ai_model.analyze_symptoms(recent_symptoms, user_id)
        
        # Get treatment recommendations
        treatment_recommendations = await ai_model.recommend_treatment(user_id, recent_symptoms)
        
        return {
            "user_id": user_id,
            "profile_summary": {
                "age": user_profile.get('age', 'not specified'),
                "avg_pain_level": round(avg_pain, 1),
                "pain_trend": pain_trend,
                "sleep_quality": sleep_quality,
                "total_symptom_entries": len(user_history)
            },
            "symptom_analysis": symptom_analysis,
            "treatment_recommendations": treatment_recommendations,
            "personalization_note": "These insights are personalized based on your individual symptom patterns and history",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trigger-analysis/{user_id}")
async def get_comprehensive_trigger_analysis(user_id: str):
    """Get detailed trigger analysis with medical insights"""
    try:
        if not ai_model or not firebase_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get user's symptom history using the AI model's fetch method
        user_history = await ai_model.fetch_user_data(user_id, days=90)
        
        if user_history.empty:
            return {
                "analysis": "No symptom data available for trigger analysis",
                "confidence": 0.0,
                "medical_insights": ["Begin comprehensive symptom tracking to identify personal triggers"],
                "recommendations": ["Track potential triggers: stress, diet, weather, hormonal changes, physical activity"]
            }
        
        # Generate comprehensive trigger analysis
        analysis = await ai_model.generate_comprehensive_trigger_analysis(user_id, user_history)
        
        logger.info(f"Generated comprehensive trigger analysis for user {user_id}")
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trigger analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pain-pattern-analysis/{user_id}")
async def get_comprehensive_pain_pattern_analysis(user_id: str):
    """Get detailed pain pattern analysis with clinical interpretation"""
    try:
        logger.info(f"Starting pain pattern analysis for user {user_id}")
        
        if not ai_model or not firebase_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get user's symptom history using the AI model's fetch method
        logger.info(f"Fetching user data for {user_id}")
        user_history = await ai_model.fetch_user_data(user_id, days=90)
        logger.info(f"Fetched {len(user_history)} rows of user data")
        
        if user_history.empty:
            logger.info(f"No data found for user {user_id}")
            return {
                "analysis": "No pain data available for pattern analysis",
                "confidence": 0.0,
                "medical_insights": ["Begin consistent pain tracking for comprehensive analysis"],
                "clinical_recommendations": ["Track daily pain levels with associated symptoms and activities"]
            }
        
        # Generate comprehensive pain pattern analysis
        logger.info(f"Generating pain pattern analysis for user {user_id}")
        analysis = await ai_model.generate_comprehensive_pain_pattern_analysis(user_id, user_history)
        logger.info(f"Successfully generated pain pattern analysis for user {user_id}")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in pain pattern analysis: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in pain pattern analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/comprehensive-medical-insights/{user_id}")
async def get_comprehensive_medical_insights(user_id: str):
    """Get complete medical insights combining all analysis types"""
    try:
        if not ai_model or not firebase_service:
            raise HTTPException(status_code=503, detail="Services not available")
        
        # Get user's symptom history using the AI model's fetch method
        user_history = await ai_model.fetch_user_data(user_id, days=90)
        
        if user_history.empty:
            return {
                "error": "No data available",
                "message": "Begin comprehensive symptom tracking for detailed medical insights"
            }
        
        # Generate all types of analysis
        trigger_analysis = await ai_model.generate_comprehensive_trigger_analysis(user_id, user_history)
        pain_analysis = await ai_model.generate_comprehensive_pain_pattern_analysis(user_id, user_history)
        
        # Get standard predictions for context
        recent_symptoms = {
            'pain_level': int(user_history['painLevel'].iloc[-1]) if len(user_history) > 0 else 5,
            'sleep_hours': user_history['sleep'].iloc[-1] if len(user_history) > 0 else 7,
            'energy_level': int(user_history['energy'].iloc[-1]) if len(user_history) > 0 else 5,
            'mood': user_history['mood'].iloc[-1] if len(user_history) > 0 else 'neutral'
        }
        
        pain_prediction = await ai_model.predict_pain_level(user_id, recent_symptoms)
        treatment_recommendations = await ai_model.recommend_treatment(user_id, recent_symptoms)
        symptom_analysis = await ai_model.analyze_symptoms(recent_symptoms, user_id)
        
        # Combine into comprehensive report
        comprehensive_insights = {
            "user_id": user_id,
            "analysis_date": datetime.now().isoformat(),
            "data_quality": {
                "tracking_days": len(user_history),
                "confidence_score": min(trigger_analysis.get('confidence', 0), pain_analysis.get('confidence', 0))
            },
            "pain_prediction": pain_prediction,
            "trigger_analysis": trigger_analysis,
            "pain_pattern_analysis": pain_analysis,
            "treatment_recommendations": treatment_recommendations,
            "symptom_analysis": symptom_analysis,
            "summary_insights": [
                f"Analysis based on {len(user_history)} days of comprehensive tracking",
                "Personalized insights generated using advanced machine learning algorithms",
                "Medical interpretations provided for clinical context and understanding"
            ]
        }
        
        logger.info(f"Generated comprehensive medical insights for user {user_id}")
        return comprehensive_insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comprehensive insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG_MODE,
        log_level="info"
    )
