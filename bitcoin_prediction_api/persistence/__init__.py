"""
Persistence layer for Bitcoin Prediction API.

This module provides database abstraction and MongoDB implementation
for storing and retrieving prediction data.
"""

from .mongodb_persistence import MongoDBPersistenceLayer

__all__ = ["MongoDBPersistenceLayer"]