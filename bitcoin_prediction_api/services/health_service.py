"""
Health service for Bitcoin Prediction API.

This module contains health monitoring logic for checking service
dependencies and overall system health.
"""

from datetime import datetime
import logging
import asyncio
import time

from bitcoin_prediction_api.models.health import HealthResponse, DependencyStatus

logger = logging.getLogger(__name__)

class HealthService:
    """
    Service for monitoring application and dependency health.
    
    This service checks the health of all critical dependencies including
    MongoDB, Redis, and external APIs, providing detailed status information
    for monitoring and alerting systems.
    """
    
    def __init__(self, persistence_layer, cache_manager, data_fetcher):
        """
        Initialize health service with dependencies.
        
        Args:
            persistence_layer: Database persistence service
            cache_manager: Caching service
            data_fetcher: External API data fetcher service
        """
        self.persistence = persistence_layer
        self.cache = cache_manager
        self.data_fetcher = data_fetcher
        self.start_time = time.time()
    
    async def check_health(self) -> HealthResponse:
        """
        Perform comprehensive health check of all dependencies.
        
        Returns:
            HealthResponse containing overall status and dependency details
        """
        logger.debug("Starting comprehensive health check")
        
        dependencies = {}
        overall_status = "healthy"
        
        # Check all dependencies concurrently
        dependency_checks = await asyncio.gather(
            self._check_database_health(),
            self._check_cache_health(),
            self._check_external_apis_health(),
            return_exceptions=True
        )
        
        # Process database health
        db_status = dependency_checks[0]
        if isinstance(db_status, Exception):
            dependencies["mongodb"] = DependencyStatus(
                name="mongodb",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(db_status)
            )
            overall_status = "degraded"
        else:
            dependencies["mongodb"] = db_status
            if db_status.status != "up":
                overall_status = "degraded"
        
        # Process cache health
        cache_status = dependency_checks[1]
        if isinstance(cache_status, Exception):
            dependencies["redis"] = DependencyStatus(
                name="redis",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(cache_status)
            )
            # Cache failure is not critical, service can continue
        else:
            dependencies["redis"] = cache_status
        
        # Process external API health
        api_status = dependency_checks[2]
        if isinstance(api_status, Exception):
            dependencies["external_apis"] = DependencyStatus(
                name="external_apis",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(api_status)
            )
            overall_status = "degraded"
        else:
            dependencies["external_apis"] = api_status
            if api_status.status != "up":
                overall_status = "degraded"
        
        # If multiple critical dependencies are down, mark as unhealthy
        critical_down = sum(1 for dep in ["mongodb", "external_apis"] 
                          if dependencies.get(dep, {}).status == "down")
        if critical_down >= 2:
            overall_status = "unhealthy"
        
        health_response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies=dependencies
        )
        
        logger.debug(
            "Health check completed",
            status=overall_status,
            dependency_count=len(dependencies)
        )
        
        return health_response
    
    async def _check_database_health(self) -> DependencyStatus:
        """
        Check MongoDB database health.
        
        Returns:
            DependencyStatus for MongoDB
        """
        start_time = time.time()
        
        try:
            # Attempt to ping the database
            await self.persistence.ping()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return DependencyStatus(
                name="mongodb",
                status="up",
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return DependencyStatus(
                name="mongodb",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_cache_health(self) -> DependencyStatus:
        """
        Check Redis cache health.
        
        Returns:
            DependencyStatus for Redis
        """
        start_time = time.time()
        
        try:
            # Attempt to ping the cache
            await self.cache.ping()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return DependencyStatus(
                name="redis",
                status="up",
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            return DependencyStatus(
                name="redis",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _check_external_apis_health(self) -> DependencyStatus:
        """
        Check external API health (Binance, CoinGecko).
        
        Returns:
            DependencyStatus for external APIs
        """
        start_time = time.time()
        
        try:
            # Attempt a lightweight health check on external APIs
            await self.data_fetcher.health_check()
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return DependencyStatus(
                name="external_apis",
                status="up",
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(f"External API health check failed: {e}")
            return DependencyStatus(
                name="external_apis",
                status="down",
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
    
    def get_uptime_seconds(self) -> float:
        """
        Get service uptime in seconds.
        
        Returns:
            Uptime in seconds since service start
        """
        return time.time() - self.start_time