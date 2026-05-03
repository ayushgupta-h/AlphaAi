"""
FastAPI application entry point for Bitcoin Prediction API.

This module initializes the FastAPI application with all necessary middleware,
routes, and configuration for the production-ready Bitcoin prediction service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

from bitcoin_prediction_api.config import settings
from bitcoin_prediction_api.api import prediction_router, health_router
from bitcoin_prediction_api.middleware import setup_middleware
from bitcoin_prediction_api.logging_config import configure_logging
from bitcoin_prediction_api.services.prediction_service import PredictionService
from bitcoin_prediction_api.services.health_service import HealthService
from bitcoin_prediction_api.services.prediction_engine_service import PredictionEngineService
from bitcoin_prediction_api.dependencies import initialize_services, cleanup_services

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Bitcoin Prediction API",
    description="Production-ready real-time Bitcoin price prediction service",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Setup custom middleware
setup_middleware(app)

# Include routers
app.include_router(prediction_router, prefix="/api", tags=["predictions"])
app.include_router(health_router, prefix="/api", tags=["health"])

# Global service instances
prediction_engine_service: PredictionEngineService = None

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global prediction_engine_service
    
    logger.info("Starting Bitcoin Prediction API v1.0.0")
    logger.info(f"Configuration: debug={settings.debug}, log_level={settings.log_level}")
    logger.info(f"MongoDB URL: {settings.mongodb_url.split('@')[-1] if '@' in settings.mongodb_url else settings.mongodb_url}")
    
    try:
        # Initialize prediction engine service with MongoDB persistence
        logger.info("Initializing prediction engine service...")
        prediction_engine_service = PredictionEngineService()
        await prediction_engine_service.initialize()
        
        # Initialize other services
        logger.info("Initializing application services...")
        prediction_service = PredictionService(prediction_engine_service)
        health_service = HealthService(prediction_engine_service)
        
        # Set up dependency injection
        initialize_services(prediction_service, health_service)
        
        logger.info("Bitcoin Prediction API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start Bitcoin Prediction API: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    global prediction_engine_service
    
    logger.info("Shutting down Bitcoin Prediction API")
    
    try:
        # Cleanup services
        cleanup_services()
        
        # Cleanup prediction engine service
        if prediction_engine_service:
            await prediction_engine_service.cleanup()
        
        logger.info("Bitcoin Prediction API shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "bitcoin_prediction_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )