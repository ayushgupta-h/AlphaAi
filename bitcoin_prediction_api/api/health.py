"""
Health check API endpoints for Bitcoin Prediction API.

This module contains FastAPI route handlers for service health monitoring
and dependency status checking.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
import logging
import time

from bitcoin_prediction_api.models.health import HealthResponse, DependencyStatus
from bitcoin_prediction_api.services.health_service import HealthService
from bitcoin_prediction_api.dependencies import get_health_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
) -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    This endpoint checks the health of all service dependencies including
    MongoDB, Redis, and external APIs, returning detailed status information
    for monitoring and alerting systems.
    
    Args:
        health_service: Injected health service dependency
    
    Returns:
        HealthResponse containing overall status and dependency details
    """
    try:
        logger.debug("Performing health check")
        
        health_status = await health_service.check_health()
        
        logger.debug(
            "Health check completed",
            status=health_status.status,
            dependency_count=len(health_status.dependencies)
        )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        
        # Return degraded status if health check itself fails
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies={
                "health_service": DependencyStatus(
                    name="health_service",
                    status="down",
                    last_check=datetime.utcnow(),
                    error_message=str(e)
                )
            }
        )

@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    
    This endpoint returns application metrics in Prometheus format
    for monitoring and alerting systems.
    
    Returns:
        Plain text response with Prometheus-formatted metrics
    """
    try:
        # This will be implemented when we add metrics collection
        # For now, return basic placeholder metrics
        metrics_data = f"""# HELP api_info API information
# TYPE api_info gauge
api_info{{version="1.0.0"}} 1

# HELP api_uptime_seconds API uptime in seconds
# TYPE api_uptime_seconds counter
api_uptime_seconds {time.time()}

# HELP api_health_status API health status (1=healthy, 0=unhealthy)
# TYPE api_health_status gauge
api_health_status 1
"""
        
        from fastapi import Response
        return Response(content=metrics_data, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}", exc_info=True)
        from fastapi import Response
        return Response(
            content="# Metrics temporarily unavailable\n",
            media_type="text/plain",
            status_code=503
        )