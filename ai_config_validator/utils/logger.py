"""Structured logging for provider errors and operations."""

import json
import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

from .errors import ProviderError, ErrorSeverity
from .tracing import get_trace_context, get_request_id, get_correlation_id
from .alerting import alert_manager


class StructuredLogger:
    """Structured logger for provider operations and errors."""
    
    def __init__(self, name: str = "vesper", log_file: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (if provided)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def log_provider_error(self, error: ProviderError, enrich: bool = True):
        """
        Log a provider error in structured format.
        
        Args:
            error: ProviderError instance
            enrich: Whether to enrich with trace context and send alerts
        """
        error_dict = error.to_dict()
        error_dict["log_type"] = "provider_error"
        error_dict["severity"] = error.severity.value
        
        # Enrich with trace context
        if enrich:
            trace_context = get_trace_context()
            if trace_context:
                error_dict["trace_context"] = trace_context
                error_dict["request_id"] = get_request_id()
                error_dict["correlation_id"] = get_correlation_id()
            
            # Add contextual metadata
            error_dict["enriched"] = True
            error_dict["enrichment_timestamp"] = datetime.utcnow().isoformat()
        
        # Log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(json.dumps(error_dict))
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(json.dumps(error_dict))
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(json.dumps(error_dict))
        else:
            self.logger.info(json.dumps(error_dict))
        
        # Send alert for critical errors
        if enrich and error.severity == ErrorSeverity.CRITICAL:
            alert_manager.alert_critical_error(error, trace_context)
        
        # Record error for rate calculation
        if enrich:
            alert_manager.record_error(error.provider_name, error.error_type.value)
    
    def log_provider_request(
        self,
        provider_name: str,
        request_id: Optional[str] = None,
        message_length: Optional[int] = None,
        enrich: bool = True,
        **kwargs
    ):
        """
        Log a provider request.
        
        Args:
            provider_name: Name of the provider
            request_id: Optional request ID (overrides trace context if provided)
            message_length: Length of the message
            enrich: Whether to enrich with trace context
            **kwargs: Additional fields to log
        """
        log_data = {
            "log_type": "provider_request",
            "provider_name": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id or get_request_id(),
            "message_length": message_length,
            **kwargs
        }
        
        # Enrich with trace context
        if enrich:
            trace_context = get_trace_context()
            if trace_context:
                log_data["trace_context"] = trace_context
                log_data["correlation_id"] = get_correlation_id()
                if not request_id:
                    log_data["request_id"] = trace_context.get("request_id")
        
        self.logger.info(json.dumps(log_data))
    
    def log_provider_response(
        self,
        provider_name: str,
        response_time_ms: float,
        tokens_used: Optional[int] = None,
        request_id: Optional[str] = None,
        enrich: bool = True,
        **kwargs
    ):
        """
        Log a provider response.
        
        Args:
            provider_name: Name of the provider
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used
            request_id: Optional request ID (overrides trace context if provided)
            enrich: Whether to enrich with trace context
            **kwargs: Additional fields to log
        """
        log_data = {
            "log_type": "provider_response",
            "provider_name": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": response_time_ms,
            "tokens_used": tokens_used,
            "request_id": request_id or get_request_id(),
            **kwargs
        }
        
        # Enrich with trace context
        if enrich:
            trace_context = get_trace_context()
            if trace_context:
                log_data["trace_context"] = trace_context
                log_data["correlation_id"] = get_correlation_id()
                if not request_id:
                    log_data["request_id"] = trace_context.get("request_id")
        
        self.logger.info(json.dumps(log_data))
    
    def log_circuit_breaker_event(
        self,
        provider_name: str,
        event: str,
        state: str,
        failure_count: Optional[int] = None,
        enrich: bool = True,
        **kwargs
    ):
        """
        Log a circuit breaker event.
        
        Args:
            provider_name: Name of the provider
            event: Event type (e.g., "opened", "closed", "half_open")
            state: Current circuit state
            failure_count: Number of failures
            enrich: Whether to enrich with trace context and send alerts
            **kwargs: Additional fields to log
        """
        log_data = {
            "log_type": "circuit_breaker_event",
            "provider_name": provider_name,
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "state": state,
            "failure_count": failure_count,
            **kwargs
        }
        
        # Enrich with trace context
        if enrich:
            trace_context = get_trace_context()
            if trace_context:
                log_data["trace_context"] = trace_context
                log_data["request_id"] = get_request_id()
                log_data["correlation_id"] = get_correlation_id()
            
            # Send alert if circuit opened
            if event == "opened" and state == "open":
                alert_manager.alert_circuit_breaker_opened(
                    provider_name,
                    failure_count or 0,
                    trace_context
                )
        
        self.logger.warning(json.dumps(log_data))
    
    def log_cascade_fallback(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
        request_id: Optional[str] = None,
        enrich: bool = True,
        **kwargs
    ):
        """
        Log a cascade fallback event.
        
        Args:
            from_provider: Provider that failed
            to_provider: Provider being tried next
            reason: Reason for fallback
            request_id: Optional request ID (overrides trace context if provided)
            enrich: Whether to enrich with trace context
            **kwargs: Additional fields to log
        """
        log_data = {
            "log_type": "cascade_fallback",
            "from_provider": from_provider,
            "to_provider": to_provider,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "request_id": request_id or get_request_id(),
            **kwargs
        }
        
        # Enrich with trace context
        if enrich:
            trace_context = get_trace_context()
            if trace_context:
                log_data["trace_context"] = trace_context
                log_data["correlation_id"] = get_correlation_id()
                if not request_id:
                    log_data["request_id"] = trace_context.get("request_id")
        
        self.logger.warning(json.dumps(log_data))
    
    def info(self, message: str, *args, **kwargs):
        """Compatibility wrapper for info logging.
        """
        self.logger.info(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Compatibility wrapper for error logging.
        """
        self.logger.error(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Compatibility wrapper for warning logging.
        """
        self.logger.warning(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """Compatibility wrapper for debug logging.
        """
        self.logger.debug(message, *args, **kwargs)

    def log_info(self, message: str, **kwargs):
        """
        Log an info message.
        
        Args:
            message: Log message
            **kwargs: Additional fields to log
        """
        log_data = {
            "log_type": "info",
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))


from config import config

# Global logger instance
logger = StructuredLogger(log_file=config.LOG_FILE_PATH)
api_logger = logger


__all__ = [
    "StructuredLogger",
    "logger"
]

