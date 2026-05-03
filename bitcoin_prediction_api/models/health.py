"""
Health check data models for Bitcoin Prediction API.

This module contains Pydantic models for health check responses
and dependency status information.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal
from datetime import datetime

class DependencyStatus(BaseModel):
    """Individual dependency health status."""
    
    name: str = Field(description="Dependency name")
    status: Literal["up", "down", "timeout"] = Field(description="Dependency status")
    response_time_ms: Optional[float] = Field(
        default=None,
        description="Response time in milliseconds"
    )
    last_check: datetime = Field(description="Last health check timestamp")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if dependency is down"
    )

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall service health status"
    )
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    dependencies: Dict[str, DependencyStatus] = Field(
        description="Status of all service dependencies"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }

class MetricsResponse(BaseModel):
    """Metrics response model for structured metrics endpoint."""
    
    timestamp: datetime = Field(description="Metrics collection timestamp")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    request_count: int = Field(description="Total request count")
    error_count: int = Field(description="Total error count")
    average_response_time_ms: float = Field(
        description="Average response time in milliseconds"
    )
    prediction_count: int = Field(description="Total predictions generated")
    cache_hit_rate: float = Field(description="Cache hit rate percentage")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }