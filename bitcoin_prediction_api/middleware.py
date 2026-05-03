"""
Middleware for Bitcoin Prediction API.

This module contains FastAPI middleware for request logging, metrics collection,
correlation ID tracking, and other cross-cutting concerns.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation IDs to requests for tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to request and response headers."""
        
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Add to request state for access in route handlers
        request.state.correlation_id = correlation_id
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details with timing."""
        
        start_time = time.time()
        correlation_id = getattr(request.state, 'correlation_id', None)
        
        # Create logger with correlation ID
        request_logger = logging.LoggerAdapter(
            logger, 
            {"correlation_id": correlation_id}
        )
        
        # Log incoming request
        request_logger.info(
            "Incoming request",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log response
            request_logger.info(
                "Request completed",
                status_code=response.status_code,
                response_time_ms=response_time
            )
            
            # Add timing header
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
            
            return response
            
        except Exception as e:
            # Log error
            response_time = (time.time() - start_time) * 1000
            request_logger.error(
                "Request failed",
                error=str(e),
                error_type=type(e).__name__,
                response_time_ms=response_time,
                exc_info=True
            )
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        # Initialize metrics collectors here when we implement them
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect request metrics."""
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Update metrics
            self.request_count += 1
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
            
        except Exception as e:
            # Update error metrics
            self.request_count += 1
            self.error_count += 1
            raise

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add API version header
        response.headers["X-API-Version"] = "1.0.0"
        
        return response

def setup_middleware(app: FastAPI) -> None:
    """
    Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is executed first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    
    logger.info("Middleware setup completed")