"""
Health check endpoint for production monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import asyncio
import time
from datetime import datetime
import logging

from ..models.ai_model import PainCareAIModel
from ..services.firebase_service import FirebaseService

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthChecker:
    """Production health checking service"""
    
    def __init__(self, ai_model: PainCareAIModel, firebase_service: FirebaseService):
        self.ai_model = ai_model
        self.firebase_service = firebase_service
        
    async def check_model_health(self) -> Dict[str, Any]:
        """Check AI model health"""
        try:
            is_healthy = self.ai_model and self.ai_model.is_trained
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "model_loaded": is_healthy,
                "last_update": self.ai_model.last_update.isoformat() if self.ai_model.last_update else None,
                "model_version": getattr(self.ai_model, 'version', 'unknown')
            }
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_firebase_health(self) -> Dict[str, Any]:
        """Check Firebase connection health"""
        try:
            if not self.firebase_service or not self.firebase_service.db:
                return {
                    "status": "unhealthy",
                    "error": "Firebase service not initialized"
                }
            
            # Try a simple read operation
            test_doc = self.firebase_service.db.collection('_health_check').limit(1).stream()
            list(test_doc)  # Execute query
            
            return {
                "status": "healthy",
                "connected": True
            }
        except Exception as e:
            logger.error(f"Firebase health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "status": "healthy" if memory.percent < 85 else "warning",
                "usage_percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024),
                "total_mb": memory.total // (1024 * 1024)
            }
        except ImportError:
            return {
                "status": "unknown",
                "error": "psutil not available"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            usage_percent = (disk.used / disk.total) * 100
            
            return {
                "status": "healthy" if usage_percent < 85 else "warning",
                "usage_percent": round(usage_percent, 2),
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "total_gb": disk.total // (1024 * 1024 * 1024)
            }
        except ImportError:
            return {
                "status": "unknown",
                "error": "psutil not available"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Global health checker instance
health_checker = None


def get_health_checker() -> HealthChecker:
    """Get health checker instance"""
    global health_checker
    if not health_checker:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    return health_checker


@router.get("/health")
async def basic_health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PainCare AI Model"
    }


@router.get("/health/detailed")
async def detailed_health_check(checker: HealthChecker = Depends(get_health_checker)):
    """Detailed health check with all components"""
    start_time = time.time()
    
    # Run all health checks concurrently
    model_health, firebase_health, memory_health, disk_health = await asyncio.gather(
        checker.check_model_health(),
        checker.check_firebase_health(),
        checker.check_memory_usage(),
        checker.check_disk_usage(),
        return_exceptions=True
    )
    
    response_time = time.time() - start_time
    
    # Determine overall status
    components = [model_health, firebase_health, memory_health, disk_health]
    unhealthy_components = [c for c in components if isinstance(c, dict) and c.get("status") == "unhealthy"]
    warning_components = [c for c in components if isinstance(c, dict) and c.get("status") == "warning"]
    
    if unhealthy_components:
        overall_status = "unhealthy"
    elif warning_components:
        overall_status = "warning"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "response_time_ms": round(response_time * 1000, 2),
        "components": {
            "model": model_health if not isinstance(model_health, Exception) else {"status": "error", "error": str(model_health)},
            "firebase": firebase_health if not isinstance(firebase_health, Exception) else {"status": "error", "error": str(firebase_health)},
            "memory": memory_health if not isinstance(memory_health, Exception) else {"status": "error", "error": str(memory_health)},
            "disk": disk_health if not isinstance(disk_health, Exception) else {"status": "error", "error": str(disk_health)}
        }
    }


@router.get("/health/readiness")
async def readiness_check(checker: HealthChecker = Depends(get_health_checker)):
    """Kubernetes readiness probe endpoint"""
    model_health = await checker.check_model_health()
    firebase_health = await checker.check_firebase_health()
    
    if model_health.get("status") == "healthy" and firebase_health.get("status") == "healthy":
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}
