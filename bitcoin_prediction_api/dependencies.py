"""
Dependency injection for Bitcoin Prediction API.

This module provides FastAPI dependency injection functions for services
and other components used throughout the application.
"""

from fastapi import Depends
import logging

from bitcoin_prediction_api.services.prediction_service import PredictionService
from bitcoin_prediction_api.services.health_service import HealthService

logger = logging.getLogger(__name__)

# Global service instances (will be properly initialized during startup)
_prediction_service: PredictionService = None
_health_service: HealthService = None

async def get_prediction_service() -> PredictionService:
    """
    Dependency injection for PredictionService.
    
    Returns:
        PredictionService instance
    
    Raises:
        RuntimeError: If service is not initialized
    """
    if _prediction_service is None:
        logger.error("PredictionService not initialized")
        raise RuntimeError("PredictionService not initialized. Check application startup.")
    
    return _prediction_service

async def get_health_service() -> HealthService:
    """
    Dependency injection for HealthService.
    
    Returns:
        HealthService instance
    
    Raises:
        RuntimeError: If service is not initialized
    """
    if _health_service is None:
        logger.error("HealthService not initialized")
        raise RuntimeError("HealthService not initialized. Check application startup.")
    
    return _health_service

def initialize_services(
    prediction_service: PredictionService,
    health_service: HealthService
) -> None:
    """
    Initialize global service instances.
    
    This function should be called during application startup to set up
    the dependency injection system.
    
    Args:
        prediction_service: Initialized PredictionService instance
        health_service: Initialized HealthService instance
    """
    global _prediction_service, _health_service
    
    _prediction_service = prediction_service
    _health_service = health_service
    
    logger.info("Services initialized for dependency injection")

def cleanup_services() -> None:
    """
    Cleanup global service instances.
    
    This function should be called during application shutdown to clean up
    resources and reset the dependency injection system.
    """
    global _prediction_service, _health_service
    
    _prediction_service = None
    _health_service = None
    
    logger.info("Services cleaned up")