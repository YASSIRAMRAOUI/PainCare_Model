"""
Firebase Service for PainCare AI Model
Handles all Firebase Firestore interactions for user data and model updates
"""

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from ..config import config

logger = logging.getLogger(__name__)


class FirebaseService:
    """
    Service class for Firestore operations
    Handles user data retrieval, model updates, and real-time synchronization
    """
    
    def __init__(self):
        self.app = None
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        try:
            self._initialize_firebase()
        except Exception as e:
            logger.error(f"Firebase initialization failed during constructor: {e}")
            # Continue with None db for development
    
    def _initialize_firebase(self):
        """Initialize Firebase app and Firestore database reference"""
        try:
            if not firebase_admin._apps:
                # Check if service account path is provided
                if config.FIREBASE_SERVICE_ACCOUNT_PATH and os.path.exists(config.FIREBASE_SERVICE_ACCOUNT_PATH):
                    # Initialize Firebase Admin SDK with credentials
                    cred = credentials.Certificate(config.FIREBASE_SERVICE_ACCOUNT_PATH)
                    self.app = firebase_admin.initialize_app(cred)
                    logger.info("Firebase initialized with service account credentials")
                else:
                    # Initialize with default credentials for development
                    logger.warning("No Firebase service account found, initializing with default credentials")
                    self.app = firebase_admin.initialize_app()
            else:
                self.app = firebase_admin.get_app()
            
            self.db = firestore.client()
            logger.info("Firestore service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            # For development, continue without Firebase
            logger.warning("Continuing without Firebase integration")
            self.db = None
    
    async def get_user_symptoms(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Get user symptoms data for the specified date range from Firestore
        Uses the actual Firestore structure with top-level collections
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty symptoms data")
            return []
            
        try:
            def fetch_data():
                # Query the top-level Symptoms collection filtering by userId
                symptoms_ref = self.db.collection('Symptoms')
                
                # Filter by userId and date range using recordedAt field (from indexes)
                query = (symptoms_ref
                        .where('userId', '==', user_id)
                        .where('recordedAt', '>=', start_date)
                        .where('recordedAt', '<=', end_date)
                        .order_by('recordedAt', direction='DESCENDING'))
                
                docs = query.stream()
                
                symptoms_list = []
                for doc in docs:
                    symptom_data = doc.to_dict()
                    symptom_data['id'] = doc.id
                    
                    # Convert Firestore timestamp to datetime if needed
                    if 'recordedAt' in symptom_data:
                        if hasattr(symptom_data['recordedAt'], 'seconds'):
                            # Firestore timestamp
                            symptom_data['date'] = datetime.fromtimestamp(symptom_data['recordedAt'].seconds)
                        elif isinstance(symptom_data['recordedAt'], datetime):
                            symptom_data['date'] = symptom_data['recordedAt']
                    
                    symptoms_list.append(symptom_data)
                
                logger.info(f"Found {len(symptoms_list)} symptoms for user {user_id} in date range")
                return symptoms_list
            
            # Run in executor to avoid blocking
            symptoms = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_data
            )
            
            return symptoms
            
        except Exception as e:
            logger.error(f"Error fetching user symptoms from Firestore: {e}")
            # Fallback: try to get some recent symptoms without date filter for debugging
            try:
                def fallback_fetch():
                    symptoms_ref = self.db.collection('Symptoms')
                    query = (symptoms_ref
                            .where('userId', '==', user_id)
                            .limit(10))
                    
                    docs = query.stream()
                    
                    symptoms_list = []
                    for doc in docs:
                        symptom_data = doc.to_dict()
                        symptom_data['id'] = doc.id
                        symptoms_list.append(symptom_data)
                    
                    logger.info(f"Fallback: Found {len(symptoms_list)} symptoms for user {user_id}")
                    return symptoms_list
                
                fallback_symptoms = await asyncio.get_event_loop().run_in_executor(
                    self.executor, fallback_fetch
                )
                return fallback_symptoms
                
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                return []
    
    # Note: Aggregated data methods removed - service not used in current implementation
    # For model training, use direct Firestore queries in the model training pipeline
    
    async def get_aggregated_user_data_firestore(self, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Get aggregated data from all users for model training using Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty aggregated data")
            return {}
            
        try:
            def fetch_all_data():
                # Query all symptoms from Firestore
                symptoms_ref = self.db.collection('Symptoms')
                query = symptoms_ref.limit(limit).order_by('recordedAt', direction='DESCENDING')
                docs = query.stream()
                
                aggregated_data = {}
                
                for doc in docs:
                    symptom_data = doc.to_dict()
                    symptom_data['id'] = doc.id
                    user_id = symptom_data.get('userId')
                    
                    if user_id:
                        if user_id not in aggregated_data:
                            aggregated_data[user_id] = []
                        
                        # Convert Firestore timestamp to datetime if needed
                        if 'recordedAt' in symptom_data:
                            if hasattr(symptom_data['recordedAt'], 'seconds'):
                                symptom_data['date'] = datetime.fromtimestamp(symptom_data['recordedAt'].seconds)
                            elif isinstance(symptom_data['recordedAt'], datetime):
                                symptom_data['date'] = symptom_data['recordedAt']
                        
                        aggregated_data[user_id].append(symptom_data)
                
                # Convert lists to DataFrames
                for user_id in aggregated_data:
                    aggregated_data[user_id] = pd.DataFrame(aggregated_data[user_id])
                
                logger.info(f"Aggregated data for {len(aggregated_data)} users")
                return aggregated_data
            
            # Run in executor
            data = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_all_data
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching aggregated data from Firestore: {e}")
            return {}
    
    async def save_model_predictions_firestore(self, user_id: str, predictions: Dict[str, Any]):
        """
        Save AI model predictions to Firestore for the user
        """
        if self.db is None:
            logger.warning("Firestore not connected, cannot save predictions")
            return
            
        try:
            def save_data():
                predictions_ref = self.db.collection('ModelPredictions')
                
                prediction_data = {
                    'userId': user_id,
                    **predictions,
                    'timestamp': datetime.now(),
                    'model_version': config.MODEL_VERSION
                }
                
                predictions_ref.add(prediction_data)
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, save_data
            )
            
        except Exception as e:
            logger.error(f"Error saving predictions to Firestore: {e}")
    
    async def get_user_feedback_firestore(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Get user feedback on recommendations for model improvement from Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty feedback")
            return []
            
        try:
            def fetch_feedback():
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                feedback_ref = self.db.collection('UserFeedback')
                query = (feedback_ref
                        .where('userId', '==', user_id)
                        .where('timestamp', '>=', start_date)
                        .where('timestamp', '<=', end_date)
                        .order_by('timestamp', direction='DESCENDING'))
                
                docs = query.stream()
                
                feedback_list = []
                for doc in docs:
                    feedback_data = doc.to_dict()
                    feedback_data['id'] = doc.id
                    feedback_list.append(feedback_data)
                
                return feedback_list
            
            feedback = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_feedback
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error fetching feedback from Firestore: {e}")
            return []
    
    async def save_user_feedback_firestore(self, user_id: str, feedback_data: Dict):
        """
        Save user feedback for model improvement to Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, cannot save feedback")
            return
            
        try:
            def save_feedback():
                feedback_ref = self.db.collection('UserFeedback')
                
                feedback_entry = {
                    'userId': user_id,
                    **feedback_data,
                    'timestamp': datetime.now()
                }
                
                feedback_ref.add(feedback_entry)
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, save_feedback
            )
            
        except Exception as e:
            logger.error(f"Error saving feedback to Firestore: {e}")
    
    async def update_model_metadata_firestore(self, metadata: Dict):
        """
        Update model metadata in Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, cannot update metadata")
            return
            
        try:
            def update_data():
                model_ref = self.db.collection('ModelMetadata').document('current')
                model_ref.set({
                    **metadata,
                    'updated_at': datetime.now()
                })
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, update_data
            )
            
        except Exception as e:
            logger.error(f"Error updating model metadata in Firestore: {e}")
    
    async def get_model_metadata_firestore(self) -> Dict:
        """
        Get model metadata from Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty metadata")
            return {}
            
        try:
            def fetch_metadata():
                model_ref = self.db.collection('ModelMetadata').document('current')
                doc = model_ref.get()
                
                if doc.exists:
                    return doc.to_dict()
                return {}
            
            metadata = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_metadata
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching model metadata from Firestore: {e}")
            return {}
    
    # Note: Real-time listeners removed - not needed for current implementation
    # For real-time updates, implement Firestore listeners in the main application
    
    async def get_user_profile_firestore(self, user_id: str) -> Dict:
        """
        Get user profile data for personalization from Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty profile")
            return {}
            
        try:
            def fetch_profile():
                user_ref = self.db.collection('Users').document(user_id)
                doc = user_ref.get()
                
                if doc.exists:
                    return doc.to_dict()
                return {}
            
            profile = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_profile
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error fetching user profile from Firestore: {e}")
            return {}
    
    async def save_explanation_firestore(self, user_id: str, explanation_data: Dict):
        """
        Save XAI explanations for user transparency to Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, cannot save explanation")
            return
            
        try:
            def save_data():
                explanations_ref = self.db.collection('AIExplanations')
                
                explanation_entry = {
                    'userId': user_id,
                    **explanation_data,
                    'timestamp': datetime.now()
                }
                
                explanations_ref.add(explanation_entry)
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, save_data
            )
            
        except Exception as e:
            logger.error(f"Error saving explanation to Firestore: {e}")
    
    async def get_treatment_history_firestore(self, user_id: str, days: int = 90) -> List[Dict]:
        """
        Get user's treatment history for recommendation personalization from Firestore
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty treatment history")
            return []
            
        try:
            def fetch_treatments():
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                treatments_ref = self.db.collection('Treatments')
                query = (treatments_ref
                        .where('userId', '==', user_id)
                        .where('timestamp', '>=', start_date)
                        .where('timestamp', '<=', end_date)
                        .order_by('timestamp', direction='DESCENDING'))
                
                docs = query.stream()
                
                treatments_list = []
                for doc in docs:
                    treatment_data = doc.to_dict()
                    treatment_data['id'] = doc.id
                    treatments_list.append(treatment_data)
                
                return treatments_list
            
            treatments = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_treatments
            )
            
            return treatments
            
        except Exception as e:
            logger.error(f"Error fetching treatment history from Firestore: {e}")
            return []
    
    def close(self):
        """Close the Firebase service and cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
            if self.app:
                firebase_admin.delete_app(self.app)
        except Exception as e:
            logger.error(f"Error closing Firebase service: {e}")
    
    async def get_user_symptoms_recent(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Get user symptoms data for the specified number of days
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Call the existing method with date range
        return await self.get_user_symptoms(user_id, start_date, end_date)

    async def get_user_diagnostic_tests(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Get user diagnostic test data for the specified number of days
        Uses the actual Firestore structure with top-level collections
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty diagnostic test data")
            return []
            
        try:
            def fetch_data():
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                all_tests = []
                
                # Try DiagnosisTests collection (from indexes)
                try:
                    tests_ref = self.db.collection('DiagnosisTests')
                    query = (tests_ref
                            .where('userId', '==', user_id)
                            .where('updatedAt', '>=', start_date)
                            .where('updatedAt', '<=', end_date)
                            .order_by('updatedAt', direction='DESCENDING'))
                    
                    docs = query.stream()
                    
                    for doc in docs:
                        test_data = doc.to_dict()
                        test_data['id'] = doc.id
                        test_data['collection'] = 'DiagnosisTests'
                        
                        # Convert Firestore timestamp to datetime if needed
                        if 'updatedAt' in test_data:
                            if hasattr(test_data['updatedAt'], 'seconds'):
                                test_data['date'] = datetime.fromtimestamp(test_data['updatedAt'].seconds)
                            elif isinstance(test_data['updatedAt'], datetime):
                                test_data['date'] = test_data['updatedAt']
                        
                        all_tests.append(test_data)
                    
                    logger.info(f"Found {len(all_tests)} DiagnosisTests for user {user_id}")
                
                except Exception as e:
                    logger.warning(f"Error fetching from DiagnosisTests: {e}")
                
                # Try Diagnostic collection (from indexes) 
                try:
                    diagnostic_ref = self.db.collection('Diagnostic')
                    query = (diagnostic_ref
                            .where('userId', '==', user_id)
                            .where('updatedAt', '>=', start_date)
                            .where('updatedAt', '<=', end_date)
                            .order_by('updatedAt', direction='DESCENDING'))
                    
                    docs = query.stream()
                    
                    diagnostic_count = 0
                    for doc in docs:
                        test_data = doc.to_dict()
                        test_data['id'] = doc.id
                        test_data['collection'] = 'Diagnostic'
                        
                        # Convert Firestore timestamp to datetime if needed
                        if 'updatedAt' in test_data:
                            if hasattr(test_data['updatedAt'], 'seconds'):
                                test_data['date'] = datetime.fromtimestamp(test_data['updatedAt'].seconds)
                            elif isinstance(test_data['updatedAt'], datetime):
                                test_data['date'] = test_data['updatedAt']
                        
                        all_tests.append(test_data)
                        diagnostic_count += 1
                    
                    logger.info(f"Found {diagnostic_count} Diagnostic records for user {user_id}")
                
                except Exception as e:
                    logger.warning(f"Error fetching from Diagnostic: {e}")
                
                return all_tests
            
            # Run in executor to avoid blocking
            tests = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_data
            )
            
            logger.info(f"Found {len(tests)} total diagnostic tests for user {user_id}")
            return tests
            
        except Exception as e:
            logger.error(f"Error fetching user diagnostic tests from Firestore: {e}")
            return []

    async def get_treatment_outcomes(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Get real treatment outcome data from user feedback and treatment tracking
        """
        if self.db is None:
            logger.warning("Firestore not connected, returning empty treatment outcomes")
            return []
            
        try:
            def fetch_data():
                treatment_outcomes = []
                
                # Fetch from TreatmentFeedback collection (if exists)
                try:
                    feedback_ref = self.db.collection('TreatmentFeedback')
                    if user_id:
                        query = feedback_ref.where('userId', '==', user_id)
                    else:
                        query = feedback_ref.limit(1000)  # Get recent feedback from all users
                    
                    docs = query.stream()
                    
                    for doc in docs:
                        feedback_data = doc.to_dict()
                        feedback_data['id'] = doc.id
                        feedback_data['source'] = 'feedback'
                        treatment_outcomes.append(feedback_data)
                    
                    logger.info(f"Found {len(treatment_outcomes)} treatment feedback records")
                    
                except Exception as e:
                    logger.warning(f"Error fetching from TreatmentFeedback: {e}")
                
                # Also get treatment effectiveness from symptom improvement patterns
                try:
                    symptoms_ref = self.db.collection('Symptoms')
                    if user_id:
                        query = symptoms_ref.where('userId', '==', user_id).order_by('recordedAt', direction='DESCENDING').limit(100)
                    else:
                        query = symptoms_ref.order_by('recordedAt', direction='DESCENDING').limit(1000)
                    
                    docs = query.stream()
                    
                    # Analyze symptom patterns for treatment effectiveness
                    user_symptoms = {}
                    for doc in docs:
                        symptom_data = doc.to_dict()
                        uid = symptom_data.get('userId', 'unknown')
                        
                        if uid not in user_symptoms:
                            user_symptoms[uid] = []
                        user_symptoms[uid].append(symptom_data)
                    
                    # Calculate treatment effectiveness based on pain level improvements
                    for uid, symptoms in user_symptoms.items():
                        if len(symptoms) >= 5:  # Need at least 5 data points
                            sorted_symptoms = sorted(symptoms, key=lambda x: x.get('recordedAt', datetime.min))
                            
                            # Look for improvement patterns
                            recent_pain = [s.get('painLevel', 5) for s in sorted_symptoms[-5:]]
                            older_pain = [s.get('painLevel', 5) for s in sorted_symptoms[:5]]
                            
                            if recent_pain and older_pain:
                                improvement = sum(older_pain) / len(older_pain) - sum(recent_pain) / len(recent_pain)
                                
                                treatment_outcomes.append({
                                    'userId': uid,
                                    'treatment_type': 'symptom_tracking',
                                    'effectiveness_score': max(0, min(1, 0.5 + improvement / 10)),
                                    'improvement': improvement,
                                    'source': 'calculated',
                                    'sample_size': len(symptoms)
                                })
                    
                    logger.info(f"Calculated treatment patterns for {len(user_symptoms)} users")
                    
                except Exception as e:
                    logger.warning(f"Error calculating treatment effectiveness: {e}")
                
                return treatment_outcomes
            
            # Run in executor to avoid blocking
            outcomes = await asyncio.get_event_loop().run_in_executor(
                self.executor, fetch_data
            )
            
            logger.info(f"Found {len(outcomes)} total treatment outcome records")
            return outcomes
            
        except Exception as e:
            logger.error(f"Error fetching treatment outcomes from Firestore: {e}")
            return []

    async def save_model_performance(self, model_name: str, metrics: Dict[str, Any]) -> bool:
        """
        Save model performance metrics to Firestore for monitoring
        """
        if self.db is None:
            logger.warning("Firestore not connected, cannot save model performance")
            return False
            
        try:
            def save_data():
                performance_ref = self.db.collection('ModelPerformance')
                performance_ref.add({
                    'model_name': model_name,
                    'metrics': metrics,
                    'timestamp': datetime.now(),
                    'version': '1.0'
                })
                return True
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, save_data
            )
            
            logger.info(f"Saved performance metrics for model: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
            return False
