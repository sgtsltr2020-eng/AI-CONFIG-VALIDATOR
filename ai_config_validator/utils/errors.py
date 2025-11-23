"""Custom exception classes and error handling for AI providers."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import time


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProviderError:
    """Structured error information for provider failures."""
    provider_name: str
    error_type: ErrorType
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    retry_count: int = 0
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "provider_name": self.provider_name,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "response_time_ms": self.response_time_ms,
            "status_code": self.status_code,
            "metadata": self.metadata or {}
        }


class ProviderException(Exception):
    """Base exception for provider errors."""
    
    def __init__(
        self,
        provider_name: str,
        error_type: ErrorType,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.provider_name = provider_name
        self.error_type = error_type
        self.severity = severity
        self.status_code = status_code
        self.metadata = metadata or {}
        super().__init__(message)
    
    def to_provider_error(self, retry_count: int = 0, response_time_ms: Optional[float] = None) -> ProviderError:
        """Convert exception to ProviderError."""
        return ProviderError(
            provider_name=self.provider_name,
            error_type=self.error_type,
            error_message=str(self),
            severity=self.severity,
            timestamp=time.time(),
            retry_count=retry_count,
            response_time_ms=response_time_ms,
            status_code=self.status_code,
            metadata=self.metadata
        )


class ProviderTimeoutException(ProviderException):
    """Exception raised when a provider request times out."""
    
    def __init__(self, provider_name: str, timeout_seconds: float, message: Optional[str] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.TIMEOUT,
            message=message or f"Provider {provider_name} request timed out after {timeout_seconds}s",
            severity=ErrorSeverity.MEDIUM,
            metadata={"timeout_seconds": timeout_seconds}
        )


class ProviderRateLimitException(ProviderException):
    """Exception raised when a provider rate limit is exceeded."""
    
    def __init__(self, provider_name: str, retry_after: Optional[int] = None, message: Optional[str] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.RATE_LIMIT,
            message=message or f"Provider {provider_name} rate limit exceeded",
            severity=ErrorSeverity.HIGH,
            status_code=429,
            metadata={"retry_after": retry_after} if retry_after else {}
        )


class ProviderQuotaExceededException(ProviderException):
    """Exception raised when a provider quota is exceeded."""
    
    def __init__(self, provider_name: str, message: Optional[str] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.QUOTA_EXCEEDED,
            message=message or f"Provider {provider_name} quota exceeded",
            severity=ErrorSeverity.HIGH,
            status_code=429
        )


class ProviderAuthenticationException(ProviderException):
    """Exception raised when authentication fails."""
    
    def __init__(self, provider_name: str, message: Optional[str] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.AUTHENTICATION,
            message=message or f"Provider {provider_name} authentication failed",
            severity=ErrorSeverity.CRITICAL,
            status_code=401
        )


class ProviderNetworkException(ProviderException):
    """Exception raised when a network error occurs."""
    
    def __init__(self, provider_name: str, message: Optional[str] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.NETWORK,
            message=message or f"Provider {provider_name} network error",
            severity=ErrorSeverity.HIGH
        )


class ProviderInvalidRequestException(ProviderException):
    """Exception raised when an invalid request is made."""
    
    def __init__(self, provider_name: str, message: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(
            provider_name=provider_name,
            error_type=ErrorType.INVALID_REQUEST,
            message=message or f"Provider {provider_name} invalid request",
            severity=ErrorSeverity.MEDIUM,
            status_code=status_code or 400
        )


def classify_error(provider_name: str, error: Exception) -> ProviderException:
    """
    Classify an exception into a ProviderException.
    
    Args:
        provider_name: Name of the provider
        error: The exception to classify
    
    Returns:
        ProviderException instance
    """
    error_str = str(error).lower()
    error_type_str = type(error).__name__.lower()
    
    # Check for timeout errors
    if "timeout" in error_str or "timed out" in error_str or "timeout" in error_type_str:
        timeout_seconds = 10.0  # Default timeout
        if hasattr(error, 'timeout'):
            timeout_seconds = error.timeout
        return ProviderTimeoutException(provider_name, timeout_seconds, str(error))
    
    # Check for rate limit errors
    if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
        retry_after = None
        if hasattr(error, 'retry_after'):
            retry_after = error.retry_after
        return ProviderRateLimitException(provider_name, retry_after, str(error))
    
    # Check for quota errors
    if "quota" in error_str or "quota exceeded" in error_str:
        return ProviderQuotaExceededException(provider_name, str(error))
    
    # Check for authentication errors
    if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str or "api key" in error_str:
        return ProviderAuthenticationException(provider_name, str(error))
    
    # Check for network errors
    if "network" in error_str or "connection" in error_str or "dns" in error_str:
        return ProviderNetworkException(provider_name, str(error))
    
    # Check for invalid request errors
    if "400" in error_str or "bad request" in error_str or "invalid" in error_str:
        status_code = 400
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        return ProviderInvalidRequestException(provider_name, str(error), status_code)
    
    # Default to provider error
    return ProviderException(
        provider_name=provider_name,
        error_type=ErrorType.PROVIDER_ERROR,
        message=str(error),
        severity=ErrorSeverity.MEDIUM
    )


__all__ = [
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
    "classify_error"
]

