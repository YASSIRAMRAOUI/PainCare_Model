"""
Generate realistic training dataset for PainCare AI model
Creates synthetic but medically plausible data based on real endometriosis patterns
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class EndometriosisDataGenerator:
    """
    Generate realistic endometriosis symptom data for AI training
    Based on real medical patterns and research
    """
    
    def __init__(self):
        # Define realistic patterns based on endometriosis research
        self.severity_profiles = {
            'mild': {
                'pain_range': (1, 4),
                'frequency': 0.3,  # 30% of days with symptoms
                'triggers': ['stress', 'exercise', 'fatigue'],
                'medications': ['ibuprofen', 'acetaminophen'],
                'energy_range': (6, 9),
                'mood_distribution': {'happy': 0.4, 'neutral': 0.4, 'sad': 0.2}
            },
            'moderate': {
                'pain_range': (4, 7),
                'frequency': 0.5,  # 50% of days with symptoms
                'triggers': ['stress', 'exercise', 'fatigue', 'cold', 'menstruation'],
                'medications': ['ibuprofen', 'naproxen', 'hormonal_therapy'],
                'energy_range': (4, 7),
                'mood_distribution': {'happy': 0.2, 'neutral': 0.5, 'sad': 0.3}
            },
            'severe': {
                'pain_range': (6, 10),
                'frequency': 0.7,  # 70% of days with symptoms
                'triggers': ['stress', 'exercise', 'fatigue', 'cold', 'menstruation', 'weather'],
                'medications': ['prescription_pain', 'hormonal_therapy', 'muscle_relaxants'],
                'energy_range': (2, 5),
                'mood_distribution': {'happy': 0.1, 'neutral': 0.3, 'sad': 0.6}
            }
        }
        
        # Pain locations based on real endometriosis data
        self.pain_locations = [
            'lower_abdomen', 'pelvis', 'lower_back', 'legs', 
            'rectum', 'vagina', 'bladder', 'upper_abdomen'
        ]
        
        # Common endometriosis symptoms
        self.symptom_types = [
            'pelvic_pain', 'menstrual_pain', 'pain_during_intercourse',
            'pain_with_bowel_movements', 'pain_with_urination',
            'excessive_bleeding', 'bleeding_between_periods',
            'infertility', 'fatigue', 'diarrhea', 'constipation',
            'bloating', 'nausea'
        ]
        
        # User demographics for realistic variation
        self.age_groups = {
            'young': (18, 25),
            'adult': (26, 35),
            'mature': (36, 45)
        }
        
    def generate_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate a realistic user profile"""
        age_group = random.choice(['young', 'adult', 'mature'])
        age = random.randint(*self.age_groups[age_group])
        severity = np.random.choice(['mild', 'moderate', 'severe'], p=[0.3, 0.5, 0.2])
        
        # Age influences severity patterns
        if age_group == 'young':
            # Younger patients often have less severe but more irregular symptoms
            severity_modifier = 0.8
        elif age_group == 'mature':
            # Older patients may have more consistent but potentially more severe symptoms
            severity_modifier = 1.2
        else:
            severity_modifier = 1.0
            
        return {
            'userId': user_id,
            'age': age,
            'age_group': age_group,
            'severity_profile': severity,
            'severity_modifier': severity_modifier,
            'diagnosed_date': datetime.now() - timedelta(days=random.randint(30, 1825)),  # 1-5 years ago
            'primary_symptoms': random.sample(self.symptom_types, random.randint(3, 6)),
            'pain_locations': random.sample(self.pain_locations, random.randint(1, 3))
        }
    
    def generate_symptoms_for_user(self, user_profile: Dict, num_days: int = 90) -> List[Dict]:
        """Generate symptom tracking data for a user over specified days"""
        symptoms = []
        severity_profile = self.severity_profiles[user_profile['severity_profile']]
        
        start_date = datetime.now() - timedelta(days=num_days)
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            # Determine if user tracks symptoms this day
            if random.random() < severity_profile['frequency']:
                # Generate realistic symptom data
                pain_level = random.randint(*severity_profile['pain_range'])
                pain_level = int(pain_level * user_profile['severity_modifier'])
                pain_level = min(10, max(1, pain_level))  # Clamp to 1-10
                
                # Sleep correlates with pain (higher pain = less sleep)
                base_sleep = 8 - (pain_level / 10) * 3
                sleep_hours = max(3, min(12, base_sleep + random.uniform(-1, 1)))
                
                # Energy inversely correlates with pain
                energy_base = severity_profile['energy_range'][1] - (pain_level / 10) * 4
                energy_level = max(1, min(10, energy_base + random.uniform(-1, 1)))
                
                # Mood distribution based on severity
                mood_probs = severity_profile['mood_distribution']
                mood = np.random.choice(list(mood_probs.keys()), p=list(mood_probs.values()))
                
                # Menstrual cycle effects (approximately 28-day cycle)
                cycle_day = day % 28
                if cycle_day <= 5:  # Menstruation period
                    pain_level = min(10, pain_level + random.randint(1, 3))
                    mood = 'sad' if random.random() < 0.6 else mood
                
                # Weekend effects (slightly better symptoms due to less stress)
                if current_date.weekday() >= 5:  # Weekend
                    pain_level = max(1, pain_level - random.randint(0, 1))
                    energy_level = min(10, energy_level + random.randint(0, 1))
                
                symptom_entry = {
                    'userId': user_profile['userId'],
                    'recordedAt': current_date.isoformat(),
                    'date': current_date.strftime('%Y-%m-%d'),
                    'painLevel': pain_level,
                    'location': random.choice(user_profile['pain_locations']),
                    'sleep': round(sleep_hours, 1),
                    'energy': round(energy_level),
                    'mood': mood,
                    'triggers': random.choice(severity_profile['triggers']) if random.random() < 0.4 else '',
                    'medications': random.choice(severity_profile['medications']) if random.random() < 0.3 else '',
                    'notes': self._generate_notes(pain_level, mood) if random.random() < 0.2 else '',
                    'symptoms': random.sample(user_profile['primary_symptoms'], random.randint(1, 3))
                }
                
                symptoms.append(symptom_entry)
        
        return symptoms
    
    def generate_diagnostic_data(self, user_profile: Dict) -> List[Dict]:
        """Generate realistic diagnostic test data"""
        diagnostics = []
        
        # Common endometriosis diagnostic tests
        diagnostic_tests = [
            {
                'test_type': 'ultrasound',
                'results': 'cysts_detected' if random.random() < 0.4 else 'normal',
                'confidence': random.uniform(0.7, 0.95)
            },
            {
                'test_type': 'mri',
                'results': 'endometrial_tissue_detected' if random.random() < 0.6 else 'inconclusive',
                'confidence': random.uniform(0.8, 0.98)
            },
            {
                'test_type': 'laparoscopy',
                'results': 'endometriosis_confirmed' if random.random() < 0.8 else 'no_visible_lesions',
                'confidence': random.uniform(0.9, 0.99)
            },
            {
                'test_type': 'blood_test',
                'results': f'ca125_level_{random.randint(10, 200)}',
                'confidence': random.uniform(0.6, 0.8)
            }
        ]
        
        # Generate 1-3 diagnostic tests per user
        num_tests = random.randint(1, 3)
        selected_tests = random.sample(diagnostic_tests, num_tests)
        
        for i, test in enumerate(selected_tests):
            diagnostic_entry = {
                'userId': user_profile['userId'],
                'id': str(uuid.uuid4()),
                'testType': test['test_type'],
                'results': test['results'],
                'confidence': test['confidence'],
                'testDate': (user_profile['diagnosed_date'] + timedelta(days=random.randint(-30, 30))).isoformat(),
                'doctorNotes': f"Test {i+1} for endometriosis diagnosis",
                'severity_correlation': user_profile['severity_profile'],
                'createdAt': datetime.now().isoformat()
            }
            diagnostics.append(diagnostic_entry)
        
        return diagnostics
    
    def generate_quiz_data(self, user_profile: Dict) -> Dict:
        """Generate realistic quiz/assessment data"""
        severity_profile = self.severity_profiles[user_profile['severity_profile']]
        
        # Endometriosis severity scoring based on symptoms
        base_score = {
            'mild': random.randint(15, 35),
            'moderate': random.randint(35, 65),
            'severe': random.randint(65, 85)
        }[user_profile['severity_profile']]
        
        quiz_data = {
            'userId': user_profile['userId'],
            'quizId': 'endometriosis_assessment',
            'score': base_score,
            'percentage': (base_score / 100) * 100,
            'severity_level': user_profile['severity_profile'],
            'primary_symptoms': user_profile['primary_symptoms'],
            'pain_locations': user_profile['pain_locations'],
            'impact_on_life': random.randint(1, 10),
            'quality_of_life_score': random.randint(30, 90),
            'createdAt': user_profile['diagnosed_date'].isoformat(),
            'version': 1
        }
        
        return quiz_data
    
    def _generate_notes(self, pain_level: int, mood: str) -> str:
        """Generate realistic user notes based on pain and mood"""
        pain_notes = {
            1: ["Mild discomfort", "Barely noticeable"],
            2: ["Light cramping", "Some tension"],
            3: ["Noticeable pain", "Uncomfortable"],
            4: ["Moderate cramping", "Hard to ignore"],
            5: ["Significant pain", "Affecting activities"],
            6: ["Strong pain", "Need to rest"],
            7: ["Severe cramping", "Very uncomfortable"],
            8: ["Intense pain", "Hard to concentrate"],
            9: ["Extreme pain", "Can barely function"],
            10: ["Unbearable pain", "Cannot do anything"]
        }
        
        mood_notes = {
            'happy': ["Feeling good today", "Positive mood"],
            'neutral': ["OK day", "Normal mood"],
            'sad': ["Feeling down", "Low mood", "Frustrated with pain"]
        }
        
        pain_note = random.choice(pain_notes.get(pain_level, ["Some pain"]))
        mood_note = random.choice(mood_notes.get(mood, ["Normal mood"]))
        
        return f"{pain_note}. {mood_note}."

def generate_complete_dataset(num_users: int = 100, days_per_user: int = 90) -> Dict[str, List[Dict]]:
    """Generate complete dataset with multiple users"""
    print(f"Generating dataset for {num_users} users over {days_per_user} days each...")
    
    generator = EndometriosisDataGenerator()
    
    all_users = []
    all_symptoms = []
    all_diagnostics = []
    all_quizzes = []
    
    for i in range(num_users):
        user_id = f"user_{i+1:03d}"
        print(f"Generating data for {user_id}...")
        
        # Generate user profile
        user_profile = generator.generate_user_profile(user_id)
        all_users.append(user_profile)
        
        # Generate symptoms
        symptoms = generator.generate_symptoms_for_user(user_profile, days_per_user)
        all_symptoms.extend(symptoms)
        
        # Generate diagnostics
        diagnostics = generator.generate_diagnostic_data(user_profile)
        all_diagnostics.extend(diagnostics)
        
        # Generate quiz data
        quiz_data = generator.generate_quiz_data(user_profile)
        all_quizzes.append(quiz_data)
    
    print(f"Generated:")
    print(f"  - {len(all_users)} user profiles")
    print(f"  - {len(all_symptoms)} symptom entries")
    print(f"  - {len(all_diagnostics)} diagnostic records")
    print(f"  - {len(all_quizzes)} quiz assessments")
    
    return {
        'users': all_users,
        'symptoms': all_symptoms,
        'diagnostics': all_diagnostics,
        'quizzes': all_quizzes
    }

def save_dataset_to_files(dataset: Dict[str, List[Dict]], output_dir: str = "generated_data"):
    """Save dataset to JSON and CSV files"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON files
    for data_type, data in dataset.items():
        json_path = f"{output_dir}/{data_type}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved {json_path}")
        
        # Also save as CSV for easy analysis
        if data:
            df = pd.DataFrame(data)
            csv_path = f"{output_dir}/{data_type}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

def create_firebase_import_format(dataset: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Convert dataset to Firebase import format"""
    firebase_data = {}
    
    # Convert to Firebase document structure
    for collection_name, documents in dataset.items():
        if collection_name == 'users':
            firebase_collection = 'Users'
        elif collection_name == 'symptoms':
            firebase_collection = 'Symptoms'
        elif collection_name == 'diagnostics':
            firebase_collection = 'Diagnostic'
        elif collection_name == 'quizzes':
            firebase_collection = 'Quiz'
        else:
            firebase_collection = collection_name
            
        firebase_data[firebase_collection] = {}
        
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            firebase_data[firebase_collection][doc_id] = doc
    
    return firebase_data

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_complete_dataset(num_users=150, days_per_user=120)
    
    # Save to files
    save_dataset_to_files(dataset)
    
    # Create Firebase import format
    firebase_data = create_firebase_import_format(dataset)
    
    # Save Firebase import file
    with open("generated_data/firebase_import.json", 'w') as f:
        json.dump(firebase_data, f, indent=2, default=str)
    
    print("\n✅ Dataset generation complete!")
    print("📁 Files created in 'generated_data/' directory")
    print("🔥 Firebase import file: firebase_import.json")
    print("📊 CSV files for analysis: users.csv, symptoms.csv, diagnostics.csv, quizzes.csv")
