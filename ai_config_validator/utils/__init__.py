"""Utility modules."""

from .query_classifier import QueryClassifier, QueryComplexity
from .errors import (
    ErrorType,
    ErrorSeverity,
    ProviderError,
    ProviderException,
    ProviderTimeoutException,
    ProviderRateLimitException,
    ProviderQuotaExceededException,
    ProviderAuthenticationException,
    ProviderNetworkException,
    ProviderInvalidRequestException,
    classify_error
)
from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerManager,
    circuit_breaker_manager
)
from .logger import StructuredLogger, logger
from .tracing import (
    TraceContext,
    generate_request_id,
    generate_correlation_id,
    set_trace_context,
    get_trace_context,
    get_request_id,
    get_correlation_id,
    clear_trace_context,
    with_trace_context
)
from .alerting import (
    AlertLevel,
    Alert,
    AlertManager,
    alert_manager,
    initialize_alerting
)

__all__ = [
    "QueryClassifier",
    "QueryComplexity",
    "ErrorType",
    "ErrorSeverity",
    "ProviderError",
    "ProviderException",
    "ProviderTimeoutException",
    "ProviderRateLimitException",
    "ProviderQuotaExceededException",
    "ProviderAuthenticationException",
    "ProviderNetworkException",
    "ProviderInvalidRequestException",
    "classify_error",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "circuit_breaker_manager",
    "StructuredLogger",
    "logger",
    "TraceContext",
    "generate_request_id",
    "generate_correlation_id",
    "set_trace_context",
    "get_trace_context",
    "get_request_id",
    "get_correlation_id",
    "clear_trace_context",
    "with_trace_context",
    "AlertLevel",
    "Alert",
    "AlertManager",
    "alert_manager",
    "initialize_alerting"
]

