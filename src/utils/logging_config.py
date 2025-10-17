"""Logging configuration and utilities"""

import logging
import time
import uuid
from functools import wraps
from contextvars import ContextVar
from typing import Optional

# Context variable for correlation ID
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.request_id = request_id.get() or 'N/A'
        return True


def setup_logging(level: str = "INFO", verbose: bool = False):
    """
    Configure application logging
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        verbose: Enable verbose output (sets DEBUG level)
    """
    if verbose:
        level = "DEBUG"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add correlation filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(CorrelationFilter())
    
    # Suppress noisy third-party loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context
    
    Args:
        correlation_id: Optional ID to use, generates UUID if None
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]  # Short UUID
    
    request_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return request_id.get()


def timed(func):
    """
    Decorator to measure and log function execution time
    
    Usage:
        @timed
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            
            logger.debug(
                f"{func.__name__} completed in {duration_ms:.2f}ms",
                extra={'function': func.__name__, 'duration_ms': duration_ms}
            )
            
            return result
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            logger.error(
                f"{func.__name__} failed after {duration_ms:.2f}ms: {e}",
                extra={'function': func.__name__, 'duration_ms': duration_ms},
                exc_info=True
            )
            raise
    
    return wrapper


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module
    
    Args:
        name: Usually __name__ of the module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
