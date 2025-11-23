"""Tracing and correlation ID utilities for request tracking."""

import uuid
import time
from typing import Optional, Dict, Any
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime


# Context variable for request tracing
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('trace_context', default=None)


@dataclass
class TraceContext:
    """Trace context for request tracking."""
    request_id: str
    correlation_id: str
    timestamp: float = field(default_factory=time.time)
    user_session: Optional[str] = None
    query_origin: Optional[str] = None
    provider_chain: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "user_session": self.user_session,
            "query_origin": self.query_origin,
            "provider_chain": self.provider_chain,
            "metadata": self.metadata
        }
    
    def add_provider(self, provider_name: str):
        """Add provider to chain."""
        if provider_name not in self.provider_chain:
            self.provider_chain.append(provider_name)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to trace context."""
        self.metadata[key] = value


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return f"corr_{uuid.uuid4().hex[:16]}"


def set_trace_context(
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_session: Optional[str] = None,
    query_origin: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TraceContext:
    """
    Set trace context for current request.
    
    Args:
        request_id: Request ID (generated if None)
        correlation_id: Correlation ID (generated if None)
        user_session: User session ID
        query_origin: Query origin (e.g., "web", "api", "voice")
        metadata: Additional metadata
    
    Returns:
        Trace context dictionary (mutable)
    """
    req_id = request_id or generate_request_id()
    corr_id = correlation_id or generate_correlation_id()
    
    context = TraceContext(
        request_id=req_id,
        correlation_id=corr_id,
        user_session=user_session,
        query_origin=query_origin,
        metadata=metadata or {}
    )
    
    context_dict = context.to_dict()
    
    _request_id.set(req_id)
    _correlation_id.set(corr_id)
    _trace_context.set(context_dict)
    
    return context_dict


def get_trace_context() -> Optional[Dict[str, Any]]:
    """Get current trace context."""
    return _trace_context.get()


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return _request_id.get()


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return _correlation_id.get()


def clear_trace_context():
    """Clear trace context."""
    _request_id.set(None)
    _correlation_id.set(None)
    _trace_context.set(None)


def with_trace_context(
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_session: Optional[str] = None,
    query_origin: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Decorator to set trace context for a function.
    
    Usage:
        @with_trace_context(query_origin="api")
        async def my_function():
            # Trace context is available here
            pass
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            context = set_trace_context(
                request_id=request_id,
                correlation_id=correlation_id,
                user_session=user_session,
                query_origin=query_origin,
                metadata=metadata
            )
            try:
                return await func(*args, **kwargs)
            finally:
                clear_trace_context()
        
        def sync_wrapper(*args, **kwargs):
            context = set_trace_context(
                request_id=request_id,
                correlation_id=correlation_id,
                user_session=user_session,
                query_origin=query_origin,
                metadata=metadata
            )
            try:
                return func(*args, **kwargs)
            finally:
                clear_trace_context()
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


__all__ = [
    "TraceContext",
    "generate_request_id",
    "generate_correlation_id",
    "set_trace_context",
    "get_trace_context",
    "get_request_id",
    "get_correlation_id",
    "clear_trace_context",
    "with_trace_context"
]

