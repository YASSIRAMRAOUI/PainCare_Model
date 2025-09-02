"""
Data Processing Module for PainCare AI Model
Handles data validation, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing service for PainCare AI model
    Handles data validation, cleaning, and feature engineering
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_specs = self._load_feature_specifications()
    
    def _load_feature_specifications(self) -> Dict:
        """Load feature specifications and validation rules"""
        return {
            "pain_level": {"type": "numeric", "range": (0, 10), "required": True},
            "sleep_hours": {"type": "numeric", "range": (0, 24), "required": False},
            "stress_level": {"type": "numeric", "range": (0, 10), "required": False},
            "energy_level": {"type": "numeric", "range": (0, 10), "required": False},
            "mood": {"type": "numeric", "range": (0, 10), "required": False},
            "exercise": {"type": "boolean", "required": False},
            "medication_taken": {"type": "boolean", "required": False},
            "location": {"type": "categorical", "categories": ["head", "back", "joints", "abdomen", "other"], "required": False},
            "weather": {"type": "categorical", "categories": ["sunny", "cloudy", "rainy", "humid"], "required": False},
            "menstrual_cycle_day": {"type": "numeric", "range": (1, 35), "required": False}
        }
    
    def validate_symptoms_data(self, symptoms: Dict) -> Tuple[bool, List[str]]:
        """
        Validate symptoms data against specifications
        """
        errors = []
        
        # Check required fields
        for field, spec in self.feature_specs.items():
            if spec.get("required", False) and field not in symptoms:
                errors.append(f"Required field '{field}' is missing")
        
        # Validate data types and ranges
        for field, value in symptoms.items():
            if field in self.feature_specs:
                spec = self.feature_specs[field]
                
                # Type validation
                if spec["type"] == "numeric":
                    try:
                        numeric_value = float(value)
                        # Range validation
                        if "range" in spec:
                            min_val, max_val = spec["range"]
                            if not (min_val <= numeric_value <= max_val):
                                errors.append(f"Field '{field}' value {numeric_value} is outside range [{min_val}, {max_val}]")
                    except (ValueError, TypeError):
                        errors.append(f"Field '{field}' must be numeric, got {type(value)}")
                
                elif spec["type"] == "boolean":
                    if not isinstance(value, bool):
                        errors.append(f"Field '{field}' must be boolean, got {type(value)}")
                
                elif spec["type"] == "categorical":
                    if "categories" in spec and value not in spec["categories"]:
                        errors.append(f"Field '{field}' value '{value}' not in allowed categories {spec['categories']}")
        
        return len(errors) == 0, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data for model training/prediction
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns
        for col in numeric_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_clean[[col]] = self.imputers[col].fit_transform(df_clean[[col]])
                else:
                    df_clean[[col]] = self.imputers[col].transform(df_clean[[col]])
        
        # Impute categorical columns
        for col in categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col].fillna('unknown', inplace=True)
        
        # Remove outliers (using IQR method)
        df_clean = self._remove_outliers(df_clean, numeric_columns)
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range (IQR) method
        """
        df_no_outliers = df.copy()
        
        for col in numeric_columns:
            if col in self.feature_specs and col in df_no_outliers.columns:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Only remove outliers if they're outside feature specifications
                spec_range = self.feature_specs[col].get("range")
                if spec_range:
                    lower_bound = max(lower_bound, spec_range[0])
                    upper_bound = min(upper_bound, spec_range[1])
                
                df_no_outliers = df_no_outliers[
                    (df_no_outliers[col] >= lower_bound) & 
                    (df_no_outliers[col] <= upper_bound)
                ]
        
        return df_no_outliers
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw symptom data
        """
        if df.empty:
            return df
        
        df_engineered = df.copy()
        
        # Temporal features
        if 'date' in df_engineered.columns:
            df_engineered['date'] = pd.to_datetime(df_engineered['date'])
            df_engineered['day_of_week'] = df_engineered['date'].dt.dayofweek
            df_engineered['hour_of_day'] = df_engineered['date'].dt.hour
            df_engineered['month'] = df_engineered['date'].dt.month
            df_engineered['quarter'] = df_engineered['date'].dt.quarter
            df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6]).astype(int)
            df_engineered['is_monday'] = (df_engineered['day_of_week'] == 0).astype(int)
        
        # Pain-related features
        if 'pain_level' in df_engineered.columns:
            df_engineered['pain_category'] = pd.cut(
                df_engineered['pain_level'],
                bins=[-0.1, 3, 6, 10],
                labels=['low', 'medium', 'high']
            )
            df_engineered['severe_pain'] = (df_engineered['pain_level'] >= 8).astype(int)
        
        # Sleep quality indicators
        if 'sleep_hours' in df_engineered.columns:
            df_engineered['sleep_quality'] = pd.cut(
                df_engineered['sleep_hours'],
                bins=[0, 5, 7, 9, 24],
                labels=['poor', 'fair', 'good', 'excessive']
            )
            df_engineered['sleep_deficit'] = np.maximum(0, 8 - df_engineered['sleep_hours'])
        
        # Stress and mood combinations
        if 'stress_level' in df_engineered.columns and 'mood' in df_engineered.columns:
            df_engineered['stress_mood_ratio'] = df_engineered['stress_level'] / (df_engineered['mood'] + 1)
            df_engineered['wellbeing_score'] = (10 - df_engineered['stress_level'] + df_engineered['mood']) / 2
        
        # Energy efficiency
        if 'energy_level' in df_engineered.columns and 'sleep_hours' in df_engineered.columns:
            df_engineered['energy_per_sleep_hour'] = df_engineered['energy_level'] / (df_engineered['sleep_hours'] + 1)
        
        # Treatment effectiveness indicators
        if 'medication_taken' in df_engineered.columns and 'pain_level' in df_engineered.columns:
            df_engineered['medication_pain_interaction'] = (
                df_engineered['medication_taken'].astype(int) * df_engineered['pain_level']
            )
        
        # Rolling averages and trends (if data is sorted by date and user)
        if 'date' in df_engineered.columns and 'user_id' in df_engineered.columns:
            df_engineered = df_engineered.sort_values(['user_id', 'date'])
            
            numeric_cols = ['pain_level', 'sleep_hours', 'stress_level', 'energy_level', 'mood']
            for col in numeric_cols:
                if col in df_engineered.columns:
                    # 3-day rolling average
                    df_engineered[f'{col}_3d_avg'] = df_engineered.groupby('user_id')[col].rolling(
                        window=3, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    
                    # 7-day rolling average
                    df_engineered[f'{col}_7d_avg'] = df_engineered.groupby('user_id')[col].rolling(
                        window=7, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    
                    # Trend (current vs 7-day average)
                    df_engineered[f'{col}_trend'] = (
                        df_engineered[col] - df_engineered[f'{col}_7d_avg']
                    )
        
        # Cyclical features for menstrual cycle
        if 'menstrual_cycle_day' in df_engineered.columns:
            df_engineered['cycle_sin'] = np.sin(2 * np.pi * df_engineered['menstrual_cycle_day'] / 28)
            df_engineered['cycle_cos'] = np.cos(2 * np.pi * df_engineered['menstrual_cycle_day'] / 28)
            
            # Menstrual phase indicators
            df_engineered['menstrual_phase'] = pd.cut(
                df_engineered['menstrual_cycle_day'],
                bins=[0, 7, 14, 21, 35],
                labels=['menstrual', 'follicular', 'ovulatory', 'luteal']
            )
        
        return df_engineered
    
    def scale_features(self, df: pd.DataFrame, fit: bool = False, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features
        """
        if df.empty:
            return df
        
        df_scaled = df.copy()
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
        
        # Exclude categorical encoded features and IDs
        exclude_cols = ['user_id', 'id', 'day_of_week', 'hour_of_day', 'month', 'quarter']
        scale_cols = [col for col in numeric_columns if col not in exclude_cols]
        
        if not scale_cols:
            return df_scaled
        
        scaler_key = f"{method}_scaler"
        
        if fit:
            if method == 'standard':
                self.scalers[scaler_key] = StandardScaler()
            elif method == 'minmax':
                self.scalers[scaler_key] = MinMaxScaler()
            
            df_scaled[scale_cols] = self.scalers[scaler_key].fit_transform(df_scaled[scale_cols])
        else:
            if scaler_key in self.scalers:
                df_scaled[scale_cols] = self.scalers[scaler_key].transform(df_scaled[scale_cols])
        
        return df_scaled
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical features using label encoding and one-hot encoding
        """
        if df.empty:
            return df
        
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in ['id', 'user_id', 'notes']:  # Skip ID columns and text
                continue
            
            unique_values = df_encoded[col].nunique()
            
            if unique_values <= 5:  # One-hot encode for low cardinality
                if fit:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    self.encoders[f"{col}_dummies"] = dummies.columns.tolist()
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                else:
                    # Create dummy columns based on training set
                    if f"{col}_dummies" in self.encoders:
                        for dummy_col in self.encoders[f"{col}_dummies"]:
                            col_value = dummy_col.replace(f"{col}_", "")
                            df_encoded[dummy_col] = (df_encoded[col] == col_value).astype(int)
                
                df_encoded.drop(col, axis=1, inplace=True)
                
            else:  # Label encode for high cardinality
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        known_categories = self.encoders[col].classes_
                        df_encoded[col] = df_encoded[col].astype(str)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_categories else known_categories[0]
                        )
                        df_encoded[col] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def create_user_features(self, user_data: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """
        Create user-specific features from historical data
        """
        if user_data.empty:
            return user_data
        
        df_user = user_data.copy()
        
        # Sort by date
        if 'date' in df_user.columns:
            df_user = df_user.sort_values('date')
        
        # Historical statistics
        if len(df_user) >= 7:  # Minimum data for meaningful statistics
            # Pain level statistics
            if 'pain_level' in df_user.columns:
                df_user['pain_level_mean'] = df_user['pain_level'].mean()
                df_user['pain_level_std'] = df_user['pain_level'].std()
                df_user['pain_level_min'] = df_user['pain_level'].min()
                df_user['pain_level_max'] = df_user['pain_level'].max()
                
                # Pain volatility (coefficient of variation)
                df_user['pain_volatility'] = df_user['pain_level_std'] / (df_user['pain_level_mean'] + 1)
            
            # Sleep pattern features
            if 'sleep_hours' in df_user.columns:
                df_user['sleep_consistency'] = 1 / (df_user['sleep_hours'].std() + 1)
                df_user['avg_sleep'] = df_user['sleep_hours'].mean()
            
            # Treatment response patterns
            if 'medication_taken' in df_user.columns:
                medication_days = df_user[df_user['medication_taken'] == True]
                if len(medication_days) > 0:
                    avg_pain_with_medication = medication_days['pain_level'].mean() if 'pain_level' in medication_days.columns else 0
                    avg_pain_without_medication = df_user[df_user['medication_taken'] == False]['pain_level'].mean() if 'pain_level' in df_user.columns else 0
                    
                    df_user['medication_effectiveness'] = avg_pain_without_medication - avg_pain_with_medication
        
        # Symptom frequency features
        df_user = self._add_frequency_features(df_user)
        
        # Pattern recognition features
        df_user = self._add_pattern_features(df_user)
        
        return df_user
    
    def _add_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on symptom frequency patterns"""
        if df.empty:
            return df
        
        df_freq = df.copy()
        
        # High pain days frequency
        if 'pain_level' in df_freq.columns:
            df_freq['high_pain_frequency'] = (df_freq['pain_level'] >= 7).astype(int).rolling(
                window=7, min_periods=1
            ).mean()
        
        # Sleep disruption frequency
        if 'sleep_hours' in df_freq.columns:
            df_freq['sleep_disruption_frequency'] = (df_freq['sleep_hours'] < 6).astype(int).rolling(
                window=7, min_periods=1
            ).mean()
        
        # Medication usage frequency
        if 'medication_taken' in df_freq.columns:
            df_freq['medication_frequency'] = df_freq['medication_taken'].astype(int).rolling(
                window=7, min_periods=1
            ).mean()
        
        return df_freq
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on temporal and cyclical patterns"""
        if df.empty or 'date' not in df.columns:
            return df
        
        df_pattern = df.copy()
        df_pattern['date'] = pd.to_datetime(df_pattern['date'])
        
        # Weekly patterns
        if 'pain_level' in df_pattern.columns:
            df_pattern['pain_weekly_pattern'] = df_pattern.groupby(
                df_pattern['date'].dt.dayofweek
            )['pain_level'].transform('mean')
        
        # Monthly patterns
        if 'menstrual_cycle_day' in df_pattern.columns:
            df_pattern['menstrual_pain_correlation'] = df_pattern.groupby(
                pd.cut(df_pattern['menstrual_cycle_day'], bins=4)
            )['pain_level'].transform('mean')
        
        # Consecutive high-pain days
        if 'pain_level' in df_pattern.columns:
            high_pain_mask = df_pattern['pain_level'] >= 7
            df_pattern['consecutive_high_pain'] = (
                high_pain_mask * (high_pain_mask.groupby((~high_pain_mask).cumsum()).cumcount() + 1)
            )
        
        # Recovery patterns (pain decreasing trends)
        if 'pain_level' in df_pattern.columns:
            df_pattern['pain_recovery_rate'] = df_pattern['pain_level'].diff().rolling(
                window=3, min_periods=1
            ).mean()
        
        return df_pattern
    
    def create_model_features(self, df: pd.DataFrame, target_variable: str = 'pain_level') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and target variable for model training
        """
        if df.empty:
            return df, pd.Series()
        
        # Feature engineering
        df_features = self.engineer_features(df)
        
        # Clean data
        df_clean = self.clean_data(df_features)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        
        # Scale numerical features
        df_scaled = self.scale_features(df_encoded, fit=True)
        
        # Separate features and target
        if target_variable in df_scaled.columns:
            y = df_scaled[target_variable]
            X = df_scaled.drop([target_variable], axis=1)
            
            # Remove non-feature columns
            non_feature_cols = ['id', 'user_id', 'date', 'timestamp', 'notes']
            X = X.drop([col for col in non_feature_cols if col in X.columns], axis=1)
        else:
            y = pd.Series()
            X = df_scaled.drop(['id', 'user_id', 'date', 'timestamp', 'notes'], axis=1, errors='ignore')
        
        return X, y
    
    def prepare_prediction_features(self, symptoms: Dict) -> pd.DataFrame:
        """
        Prepare features for single prediction
        """
        # Convert to DataFrame
        symptoms_df = pd.DataFrame([symptoms])
        
        # Add current timestamp if not present
        if 'date' not in symptoms_df.columns:
            symptoms_df['date'] = datetime.now()
        
        # Feature engineering (minimal for single prediction)
        df_features = self.engineer_features(symptoms_df)
        
        # Clean data
        df_clean = self.clean_data(df_features)
        
        # Encode categorical features (using pre-fitted encoders)
        df_encoded = self.encode_categorical_features(df_clean, fit=False)
        
        # Scale features (using pre-fitted scalers)
        df_scaled = self.scale_features(df_encoded, fit=False)
        
        # Remove non-feature columns
        non_feature_cols = ['id', 'user_id', 'date', 'timestamp', 'notes']
        df_final = df_scaled.drop([col for col in non_feature_cols if col in df_scaled.columns], axis=1)
        
        return df_final
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names"""
        base_features = list(self.feature_specs.keys())
        
        # Add engineered feature names
        engineered_features = [
            'day_of_week', 'hour_of_day', 'month', 'quarter', 'is_weekend', 'is_monday',
            'pain_category_low', 'pain_category_medium', 'pain_category_high',
            'severe_pain', 'sleep_quality_poor', 'sleep_quality_fair', 'sleep_quality_good',
            'sleep_deficit', 'stress_mood_ratio', 'wellbeing_score', 'energy_per_sleep_hour',
            'medication_pain_interaction', 'pain_level_3d_avg', 'pain_level_7d_avg',
            'pain_level_trend', 'sleep_hours_3d_avg', 'sleep_hours_7d_avg',
            'high_pain_frequency', 'sleep_disruption_frequency', 'medication_frequency',
            'consecutive_high_pain', 'pain_recovery_rate', 'cycle_sin', 'cycle_cos'
        ]
        
        return base_features + engineered_features
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality report
        """
        if df.empty:
            return {"error": "No data available"}
        
        report = {
            "total_records": len(df),
            "date_range": {
                "start": df['date'].min().isoformat() if 'date' in df.columns else None,
                "end": df['date'].max().isoformat() if 'date' in df.columns else None
            },
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "numerical_summary": df.describe().to_dict(),
            "categorical_summary": {}
        }
        
        # Categorical data summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            report["categorical_summary"][col] = df[col].value_counts().to_dict()
        
        return report
