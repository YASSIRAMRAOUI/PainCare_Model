"""
Enhanced model training functionality with real Firebase data integration
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

from ..services.firebase_service import FirebaseService
from ..config import config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Enhanced model trainer with real data integration"""
    
    def __init__(self, firebase_service: Optional[FirebaseService] = None):
        self.firebase_service = firebase_service
        self.training_progress = 0
        self.training_status = "idle"
        self.training_logs = []
        self.current_model = None
        self.model_metrics = {}
        self.data_source = "unknown"  # real|sample|unknown
        
    def log_progress(self, message: str, progress: int = None):
        """Log training progress with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        
        if progress is not None:
            self.training_progress = progress
            
        logger.info(log_entry)
    
    async def prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data from Firebase"""
        try:
            self.log_progress("Fetching data from Firebase...", 10)
            
            if not self.firebase_service or not self.firebase_service.db:
                self.log_progress("Firebase not available, using sample data", 15)
                return self._generate_sample_data()
            
            # Fetch real data from Firebase
            aggregated_data = await self.firebase_service.get_aggregated_user_data_firestore(limit=2000)
            
            if not aggregated_data:
                self.log_progress("No data available, using sample data", 15)
                return self._generate_sample_data()
            
            # Convert to training DataFrame
            training_data = []
            for user_id, user_data in aggregated_data.items():
                if isinstance(user_data, list):
                    for record in user_data:
                        if self._is_valid_record(record):
                            training_data.append(self._normalize_record(record))
            
            if not training_data:
                self.log_progress("No valid training records found, using sample data", 15)
                return self._generate_sample_data()
            
            df = pd.DataFrame(training_data)
            self.log_progress(f"Prepared {len(df)} training records from Firebase", 20)
            self.data_source = "real"
            return df
            
        except Exception as e:
            self.log_progress(f"Error preparing training data: {str(e)}", 15)
            return self._generate_sample_data()
    
    def _is_valid_record(self, record: Dict) -> bool:
        """Validate if a record has required fields for training"""
        required_fields = ['pain_level', 'sleep_hours', 'stress_level', 'energy_level']
        return all(field in record and record[field] is not None for field in required_fields)
    
    def _normalize_record(self, record: Dict) -> Dict:
        """Normalize a record for training"""
        normalized = {
            'pain_level': float(record.get('pain_level', 0)),
            'sleep_hours': float(record.get('sleep_hours', 8)),
            'stress_level': float(record.get('stress_level', 5)),
            'energy_level': float(record.get('energy_level', 5)),
            'mood': float(record.get('mood', 5)),
            'physical_activity': 1.0 if record.get('exercise', False) else 0.0,
            'medication_taken': 1.0 if record.get('medication_taken', False) else 0.0,
            'weather_pressure': float(record.get('weather_pressure', 1013.25)),
            'menstrual_cycle': float(record.get('menstrual_cycle_day', 15)),
            'location_type': self._encode_location(record.get('location', 'unknown'))
        }
        
        # Target variables (what we want to predict)
        normalized['pain_improvement'] = max(0, 10 - normalized['pain_level']) / 10
        normalized['treatment_effectiveness'] = min(1.0, 
            (normalized['energy_level'] + normalized['mood']) / 10)
        
        return normalized
    
    def _encode_location(self, location: str) -> float:
        """Encode location string to numeric value"""
        location_map = {
            'lower abdomen': 1.0,
            'upper abdomen': 2.0,
            'pelvis': 3.0,
            'back': 4.0,
            'legs': 5.0,
            'unknown': 0.0
        }
        return location_map.get(location.lower(), 0.0)
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample training data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'pain_level': np.random.normal(5, 2, n_samples).clip(1, 10),
            'sleep_hours': np.random.normal(7.5, 1.5, n_samples).clip(4, 12),
            'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
            'energy_level': np.random.normal(6, 2, n_samples).clip(1, 10),
            'mood': np.random.normal(6, 2, n_samples).clip(1, 10),
            'physical_activity': np.random.binomial(1, 0.3, n_samples),
            'medication_taken': np.random.binomial(1, 0.4, n_samples),
            'weather_pressure': np.random.normal(1013, 20, n_samples),
            'menstrual_cycle': np.random.uniform(1, 28, n_samples),
            'location_type': np.random.uniform(0, 5, n_samples)
        }
        
        # Create target variables with some realistic correlations
        data['pain_improvement'] = np.clip(
            (10 - data['pain_level'] + data['energy_level'] * 0.1 + 
             data['mood'] * 0.1 - data['stress_level'] * 0.1) / 10,
            0, 1
        )
        
        data['treatment_effectiveness'] = np.clip(
            (data['energy_level'] + data['mood'] + 
             data['medication_taken'] * 2 - data['pain_level'] * 0.2) / 12,
            0, 1
        )
        
        # Mark data source and return
        self.data_source = "sample"
        return pd.DataFrame(data)
    
    async def train_model(self, progress_callback=None) -> Dict[str, Any]:
        """Train the AI model with progress tracking"""
        try:
            self.training_status = "running"
            self.training_progress = 0
            self.training_logs = []
            self.log_progress("Starting model training...", 0)
            
            # Prepare data
            df = await self.prepare_training_data()
            if df is None or df.empty:
                raise ValueError("No training data available")
            
            self.log_progress("Preprocessing data...", 30)
            
            # Prepare features and targets
            feature_columns = config.FEATURE_COLUMNS
            target_columns = config.TARGET_COLUMNS
            
            # Ensure we have the required columns
            available_features = [col for col in feature_columns if col in df.columns]
            available_targets = [col for col in target_columns if col in df.columns]
            
            if not available_features or not available_targets:
                raise ValueError("Required columns not found in training data")
            
            X = df[available_features]
            y = df[available_targets]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.log_progress("Training models...", 50)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {}
            metrics = {}
            
            # Train models for each target
            for i, target in enumerate(available_targets):
                self.log_progress(f"Training model for {target}...", 60 + i * 15)
                
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train[target])
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test[target], y_pred)
                r2 = r2_score(y_test[target], y_pred)
                
                models[target] = model
                metrics[target] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
                
                self.log_progress(f"{target} - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.3f}")
            
            self.log_progress("Saving models...", 90)
            
            # Save models and scaler
            model_dir = "models"
            import os
            os.makedirs(model_dir, exist_ok=True)
            
            # Save individual models
            for target, model in models.items():
                joblib.dump(model, f"{model_dir}/{target}_model.joblib")
            
            # Save scaler and metadata
            joblib.dump(scaler, f"{model_dir}/scaler.joblib")
            
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'model_version': config.MODEL_VERSION,
                'features': available_features,
                'targets': available_targets,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': metrics,
                'data_source': self.data_source
            }
            
            import json
            with open(f"{model_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.current_model = models
            self.model_metrics = metrics
            
            self.log_progress("Training completed successfully!", 100)
            self.training_status = "completed"
            
            # Save to Firebase if available
            if self.firebase_service and self.data_source == "real":
                try:
                    await self.firebase_service.save_model_performance(
                        "pain_predictor", metadata
                    )
                    self.log_progress("Model metadata saved to Firebase")
                except Exception as e:
                    self.log_progress(f"Warning: Could not save to Firebase: {e}")
            elif self.firebase_service and self.data_source != "real":
                self.log_progress("Skipping saving ModelPerformance because data source is sample")
            
            return {
                'status': 'success',
                'metrics': metrics,
                'metadata': metadata,
                'logs': self.training_logs
            }
            
        except Exception as e:
            self.log_progress(f"Training failed: {str(e)}", 0)
            self.training_status = "error"
            logger.error(f"Model training error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'logs': self.training_logs
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.training_status == "running",
            'progress': self.training_progress,
            'status': self.training_status,
            'logs': self.training_logs,
            'metrics': self.model_metrics
        }