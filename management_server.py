"""
PainCare AI Model Management Server
Production-ready web interface for model management, monitoring, and training
"""

import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import logging
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.services.firebase_service import FirebaseService
from src.models.ai_model import PainCareAIModel
from src.training.model_trainer import ModelTrainer
from flask import send_from_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='assets')
app.config['SECRET_KEY'] = config.SECRET_KEY

# Allow overriding async mode if eventlet/gevent issues cause crashes
_async_mode = os.getenv("SOCKETIO_ASYNC_MODE", "eventlet")  # eventlet | gevent | threading | asyncio
try:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_async_mode)
    logger.info(f"SocketIO initialized with async_mode={_async_mode}")
except Exception as e:
    # Fallback to threading to avoid container crash loops
    logger.error(f"Failed to initialize SocketIO with async_mode={_async_mode}: {e}. Falling back to 'threading'.")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global services
firebase_service: Optional[FirebaseService] = None
ai_model: Optional[PainCareAIModel] = None
model_trainer: Optional[ModelTrainer] = None
training_status = {
    'is_training': False,
    'progress': 0,
    'status': 'idle',
    'start_time': None,
    'logs': []
}

# Scheduler state
_scheduler_thread: Optional[threading.Thread] = None
_scheduler_stop_event = threading.Event()
_scheduler_started = False

class SystemMonitor:
    """System resource monitoring service"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'per_core': psutil.cpu_percent(percpu=True)
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
            # GPU stats if available
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    stats['gpu'] = []
                    for gpu in gpus:
                        stats['gpu'].append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_percent': gpu.memoryUtil * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'temperature': gpu.temperature,
                            'load': gpu.load * 100
                        })
                except Exception as e:
                    logger.warning(f"GPU monitoring failed: {e}")
                    stats['gpu'] = []
            else:
                stats['gpu'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_system_stats()
                if stats:
                    socketio.emit('system_stats', stats)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)

# Initialize monitor
system_monitor = SystemMonitor()

class FirebaseStatsCollector:
    """Collect real Firebase statistics"""
    
    def __init__(self, firebase_service: FirebaseService):
        self.firebase_service = firebase_service
    
    async def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics from Firebase using your actual Firestore structure"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    'total_users': 0,
                    'active_users_today': 0,
                    'active_users_week': 0,
                    'active_users_month': 0,
                    'error': 'Firebase not connected'
                }
            
            # Get user count from Users collection
            users_ref = self.firebase_service.db.collection('Users')
            users_docs = list(users_ref.stream())
            total_users = len(users_docs)
            
            # Get recent activity based on your Firestore structure
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Count active users based on Symptoms collection (most active collection)
            active_today = 0
            active_week = 0
            active_month = 0
            
            try:
                # Use your indexed query structure from firestore.indexes.json
                symptoms_ref = self.firebase_service.db.collection('Symptoms')
                
                # Get today's active users
                today_symptoms = symptoms_ref.where('recordedAt', '>=', today).stream()
                unique_users_today = set()
                for doc in today_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_today.add(data['userId'])
                active_today = len(unique_users_today)
                
                # Get week's active users
                week_symptoms = symptoms_ref.where('recordedAt', '>=', week_ago).stream()
                unique_users_week = set()
                for doc in week_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_week.add(data['userId'])
                active_week = len(unique_users_week)
                
                # Get month's active users
                month_symptoms = symptoms_ref.where('recordedAt', '>=', month_ago).stream()
                unique_users_month = set()
                for doc in month_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_month.add(data['userId'])
                active_month = len(unique_users_month)
                
            except Exception as e:
                logger.warning(f"Could not get active users: {e}")
                # Try alternative approach with other collections
                try:
                    # Check QuizResults for activity
                    quiz_ref = self.firebase_service.db.collection('QuizResults')
                    recent_quiz = quiz_ref.where('completedAt', '>=', today).stream()
                    unique_quiz_users = set()
                    for doc in recent_quiz:
                        data = doc.to_dict()
                        if 'userId' in data:
                            unique_quiz_users.add(data['userId'])
                    active_today = max(active_today, len(unique_quiz_users))
                except Exception as e2:
                    logger.warning(f"Could not get quiz activity: {e2}")
            
            return {
                'total_users': total_users,
                'active_users_today': active_today,
                'active_users_week': active_week,
                'active_users_month': active_month,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {
                'total_users': 0,
                'active_users_today': 0,
                'active_users_week': 0,
                'active_users_month': 0,
                'error': str(e)
            }
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get model performance statistics from your actual collections"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    'total_predictions': 0,
                    'predictions_today': 0,
                    'predictions_week': 0,
                    'total_diagnoses': 0,
                    'total_quizzes': 0,
                    'error': 'Firebase not connected'
                }
            
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            
            # Count DiagnosisResults (your main prediction/diagnosis data)
            total_diagnoses = 0
            diagnoses_today = 0
            diagnoses_week = 0
            
            try:
                diagnosis_ref = self.firebase_service.db.collection('DiagnosisResults')
                all_diagnoses = list(diagnosis_ref.stream())
                total_diagnoses = len(all_diagnoses)
                
                # Count recent diagnoses using your indexed structure
                today_diagnoses = diagnosis_ref.where('completedAt', '>=', today).stream()
                diagnoses_today = len(list(today_diagnoses))
                
                week_diagnoses = diagnosis_ref.where('completedAt', '>=', week_ago).stream()
                diagnoses_week = len(list(week_diagnoses))
                
            except Exception as e:
                logger.warning(f"Could not get diagnosis data: {e}")
            
            # Count QuizResults (another form of model interaction)
            total_quizzes = 0
            quizzes_today = 0
            quizzes_week = 0
            
            try:
                quiz_ref = self.firebase_service.db.collection('QuizResults')
                all_quizzes = list(quiz_ref.stream())
                total_quizzes = len(all_quizzes)
                
                today_quizzes = quiz_ref.where('completedAt', '>=', today).stream()
                quizzes_today = len(list(today_quizzes))
                
                week_quizzes = quiz_ref.where('completedAt', '>=', week_ago).stream()
                quizzes_week = len(list(week_quizzes))
                
            except Exception as e:
                logger.warning(f"Could not get quiz data: {e}")
            
            # Total predictions = diagnoses + quizzes (both use AI model)
            total_predictions = total_diagnoses + total_quizzes
            predictions_today = diagnoses_today + quizzes_today
            predictions_week = diagnoses_week + quizzes_week
            
            return {
                'total_predictions': total_predictions,
                'predictions_today': predictions_today,
                'predictions_week': predictions_week,
                'total_diagnoses': total_diagnoses,
                'total_quizzes': total_quizzes,
                'diagnoses_today': diagnoses_today,
                'quizzes_today': quizzes_today,
                'model_version': config.MODEL_VERSION,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {
                'total_predictions': 0,
                'predictions_today': 0,
                'predictions_week': 0,
                'error': str(e)
            }
    
    def get_user_statistics_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_user_statistics"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    'total_users': 0,
                    'active_users_today': 0,
                    'active_users_week': 0,
                    'active_users_month': 0,
                    'error': 'Firebase not connected'
                }
            
            # Get user count from Users collection
            users_ref = self.firebase_service.db.collection('Users')
            users_docs = list(users_ref.stream())
            total_users = len(users_docs)
            
            # Get recent activity based on your Firestore structure
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Count active users based on Symptoms collection (most active collection)
            active_today = 0
            active_week = 0
            active_month = 0
            
            try:
                # Use your indexed query structure from firestore.indexes.json
                symptoms_ref = self.firebase_service.db.collection('Symptoms')
                
                # Get today's active users
                today_symptoms = symptoms_ref.where('recordedAt', '>=', today).stream()
                unique_users_today = set()
                for doc in today_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_today.add(data['userId'])
                active_today = len(unique_users_today)
                
                # Get week's active users
                week_symptoms = symptoms_ref.where('recordedAt', '>=', week_ago).stream()
                unique_users_week = set()
                for doc in week_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_week.add(data['userId'])
                active_week = len(unique_users_week)
                
                # Get month's active users
                month_symptoms = symptoms_ref.where('recordedAt', '>=', month_ago).stream()
                unique_users_month = set()
                for doc in month_symptoms:
                    data = doc.to_dict()
                    if 'userId' in data:
                        unique_users_month.add(data['userId'])
                active_month = len(unique_users_month)
                
            except Exception as e:
                logger.warning(f"Could not get active users: {e}")
                # Try alternative approach with other collections
                try:
                    # Check QuizResults for activity
                    quiz_ref = self.firebase_service.db.collection('QuizResults')
                    recent_quiz = quiz_ref.where('completedAt', '>=', today).stream()
                    unique_quiz_users = set()
                    for doc in recent_quiz:
                        data = doc.to_dict()
                        if 'userId' in data:
                            unique_quiz_users.add(data['userId'])
                    active_today = max(active_today, len(unique_quiz_users))
                except Exception as e2:
                    logger.warning(f"Could not get quiz activity: {e2}")
            
            return {
                'total_users': total_users,
                'active_users_today': active_today,
                'active_users_week': active_week,
                'active_users_month': active_month,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {
                'total_users': 0,
                'active_users_today': 0,
                'active_users_week': 0,
                'active_users_month': 0,
                'error': str(e)
            }
    
    def get_model_statistics_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_model_statistics"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    'total_predictions': 0,
                    'predictions_today': 0,
                    'predictions_week': 0,
                    'total_diagnoses': 0,
                    'total_quizzes': 0,
                    'error': 'Firebase not connected'
                }
            
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            
            # Count DiagnosisResults (your main prediction/diagnosis data)
            total_diagnoses = 0
            diagnoses_today = 0
            diagnoses_week = 0
            
            try:
                diagnosis_ref = self.firebase_service.db.collection('DiagnosisResults')
                all_diagnoses = list(diagnosis_ref.stream())
                total_diagnoses = len(all_diagnoses)
                
                # Count recent diagnoses using your indexed structure
                today_diagnoses = diagnosis_ref.where('completedAt', '>=', today).stream()
                diagnoses_today = len(list(today_diagnoses))
                
                week_diagnoses = diagnosis_ref.where('completedAt', '>=', week_ago).stream()
                diagnoses_week = len(list(week_diagnoses))
                
            except Exception as e:
                logger.warning(f"Could not get diagnosis data: {e}")
            
            # Count QuizResults (another form of model interaction)
            total_quizzes = 0
            quizzes_today = 0
            quizzes_week = 0
            
            try:
                quiz_ref = self.firebase_service.db.collection('QuizResults')
                all_quizzes = list(quiz_ref.stream())
                total_quizzes = len(all_quizzes)
                
                today_quizzes = quiz_ref.where('completedAt', '>=', today).stream()
                quizzes_today = len(list(today_quizzes))
                
                week_quizzes = quiz_ref.where('completedAt', '>=', week_ago).stream()
                quizzes_week = len(list(week_quizzes))
                
            except Exception as e:
                logger.warning(f"Could not get quiz data: {e}")
            
            # Total predictions = diagnoses + quizzes (both use AI model)
            total_predictions = total_diagnoses + total_quizzes
            predictions_today = diagnoses_today + quizzes_today
            predictions_week = diagnoses_week + quizzes_week
            
            return {
                'total_predictions': total_predictions,
                'predictions_today': predictions_today,
                'predictions_week': predictions_week,
                'total_diagnoses': total_diagnoses,
                'total_quizzes': total_quizzes,
                'diagnoses_today': diagnoses_today,
                'quizzes_today': quizzes_today,
                'model_version': config.MODEL_VERSION,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {
                'total_predictions': 0,
                'predictions_today': 0,
                'predictions_week': 0,
                'error': str(e)
            }

    def get_counts_sync(self) -> Dict[str, Any]:
        """Get totals for requested collections/documents"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    'appointments': 0,
                    'diagnostic': 0,
                    'posts': 0,
                    'quiz': 0,
                    'ai_recommendations': 0
                }

            db = self.firebase_service.db

            def count_docs(ref):
                try:
                    return len(list(ref.stream()))
                except Exception:
                    return 0

            # Collections per user request
            appointments = count_docs(db.collection('Appointments'))
            diagnostic = count_docs(db.collection('Diagnostic'))
            posts = count_docs(db.collection('Posts'))
            quiz = count_docs(db.collection('Quiz'))
            ai_recommendations = count_docs(db.collection('ai_diagnoses'))

            return {
                'appointments': appointments,
                'diagnostic': diagnostic,
                'posts': posts,
                'quiz': quiz,
                'ai_recommendations': ai_recommendations,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting general counts: {e}")
            return {
                'appointments': 0,
                'diagnostic': 0,
                'posts': 0,
                'quiz': 0,
                'ai_recommendations': 0,
                'error': str(e)
            }

# Routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon for browsers requesting /favicon.ico"""
    return send_from_directory(app.static_folder, 'favicon.png')

@app.route('/api/system/stats')
def get_system_stats():
    """Get current system statistics"""
    stats = system_monitor.get_system_stats()
    return jsonify(stats)

@app.route('/api/firebase/stats')
def get_firebase_stats():
    """Get Firebase statistics"""
    if not firebase_service:
        return jsonify({'error': 'Firebase service not available'})
    
    try:
        collector = FirebaseStatsCollector(firebase_service)
        user_stats = collector.get_user_statistics_sync()
        model_stats = collector.get_model_statistics_sync()
        counts = collector.get_counts_sync()
        
        return jsonify({
            'users': user_stats,
            'models': model_stats,
            'counts': counts
        })
    except Exception as e:
        logger.error(f"Error fetching Firebase stats: {str(e)}")
        return jsonify({'error': f'Failed to fetch Firebase statistics: {str(e)}'}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training with real data"""
    global training_status, model_trainer
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    if not model_trainer:
        return jsonify({'error': 'Model trainer not available'}), 500
    
    # Start training in background
    def train_model():
        global training_status
        try:
            training_status.update({
                'is_training': True,
                'progress': 0,
                'status': 'initializing',
                'start_time': datetime.now().isoformat(),
                'logs': ['Training started...']
            })
            
            # Emit initial status
            socketio.emit('training_status', training_status)
            
            # Run actual training
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def training_task():
                result = await model_trainer.train_model()
                
                # Update training status with real progress
                while model_trainer.training_status == "running":
                    current_status = model_trainer.get_training_status()
                    training_status.update({
                        'progress': current_status['progress'],
                        'status': current_status['status'],
                        'logs': current_status['logs']
                    })
                    socketio.emit('training_status', training_status)
                    await asyncio.sleep(1)
                
                # Final update
                final_status = model_trainer.get_training_status()
                training_status.update({
                    'is_training': False,
                    'progress': final_status['progress'],
                    'status': final_status['status'],
                    'logs': final_status['logs']
                })
                socketio.emit('training_status', training_status)
                
                return result
            
            result = loop.run_until_complete(training_task())
            loop.close()
            
            if result['status'] == 'error':
                logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
            else:
                logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            training_status.update({
                'is_training': False,
                'status': 'error',
                'progress': 0,
                'logs': training_status['logs'] + [f'Error: {str(e)}']
            })
            socketio.emit('training_status', training_status)
    
    # Start training thread
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'message': 'Training started'})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop model training"""
    global training_status
    
    if not training_status['is_training']:
        return jsonify({'error': 'No training in progress'}), 400
    
    # In a real implementation, you'd need a way to stop the training process
    training_status.update({
        'is_training': False,
        'status': 'stopped',
        'progress': 0
    })
    training_status['logs'].append('Training stopped by user')
    
    socketio.emit('training_status', training_status)
    return jsonify({'message': 'Training stopped'})

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to management server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start system monitoring"""
    system_monitor.start_monitoring()
    emit('monitoring_started', {'status': 'System monitoring started'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop system monitoring"""
    system_monitor.stop_monitoring()
    emit('monitoring_stopped', {'status': 'System monitoring stopped'})

def initialize_services():
    """Initialize all services"""
    global firebase_service, ai_model, model_trainer
    
    try:
        # Initialize Firebase service
        logger.info("Initializing Firebase service...")
        firebase_service = FirebaseService()
        if firebase_service.db:
            logger.info("Firebase service initialized successfully")
        else:
            logger.warning("Firebase service initialized but database connection failed")
    except Exception as e:
        logger.error(f"Firebase service initialization failed: {e}")
        firebase_service = None
    
    try:
        # Initialize AI model
        logger.info("Initializing AI model...")
        ai_model = PainCareAIModel()
        logger.info("AI model initialized successfully")
    except Exception as e:
        logger.error(f"AI model initialization failed: {e}")
        ai_model = None
    
    try:
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        model_trainer = ModelTrainer(firebase_service)
        logger.info("Model trainer initialized successfully")
    except Exception as e:
        logger.error(f"Model trainer initialization failed: {e}")
        model_trainer = None

def _weekly_scheduler_loop():
    """Background loop that triggers weekly model evaluation/training using real data only."""
    global model_trainer, firebase_service
    logger.info("Weekly scheduler thread started")
    # Stagger first run to next week boundary to avoid immediate heavy work on startup
    next_run = datetime.now() + timedelta(days=7)
    while not _scheduler_stop_event.is_set():
        try:
            now = datetime.now()
            if now >= next_run:
                logger.info("Weekly scheduler: starting model training/evaluation run")
                if model_trainer:
                    # Run training in an isolated event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(model_trainer.train_model())
                        if result.get('status') == 'success':
                            if model_trainer.data_source != 'real':
                                logger.info("Weekly scheduler: training used sample data; skipping save to ModelPerformance")
                            else:
                                logger.info("Weekly scheduler: training completed with real data; ModelPerformance saved by trainer")
                        else:
                            logger.warning(f"Weekly scheduler: training failed: {result.get('error')}")
                    except Exception as e:
                        logger.error(f"Weekly scheduler run error: {e}")
                    finally:
                        loop.close()
                else:
                    logger.warning("Weekly scheduler: model_trainer not available")
                # Schedule next run
                next_run = now + timedelta(days=7)
            # Sleep with responsiveness to stop event
            _scheduler_stop_event.wait(timeout=60)  # check every 60s
        except Exception as e:
            logger.error(f"Weekly scheduler loop error: {e}")
            _scheduler_stop_event.wait(timeout=300)

def start_weekly_scheduler():
    global _scheduler_thread, _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    _scheduler_thread = threading.Thread(target=_weekly_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("Weekly scheduler started")

def stop_weekly_scheduler():
    _scheduler_stop_event.set()
    if _scheduler_thread:
        _scheduler_thread.join(timeout=2)
    logger.info("Weekly scheduler stopped")

@app.route('/api/models/performance/latest')
def get_latest_model_performance():
    """Return the latest saved real model performance from Firestore."""
    if not firebase_service:
        return jsonify({'error': 'Firebase service not available'}), 500
    try:
        # Use a short-lived event loop for the async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(firebase_service.get_latest_model_performance('pain_predictor'))
        loop.close()
        if not result:
            return jsonify({'message': 'No model performance found'}), 404
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching latest model performance: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize services
    initialize_services()
    # Start weekly scheduler
    start_weekly_scheduler()
    
    # Start the server
    logger.info("Starting PainCare AI Management Server...")
    # Allow configuring port via env var, default 7000
    port = int(os.getenv('MANAGEMENT_PORT', '7000'))
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=config.DEBUG_MODE,
        allow_unsafe_werkzeug=True
    )