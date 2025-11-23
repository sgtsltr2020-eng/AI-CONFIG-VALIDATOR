"""Alerting hooks for error thresholds and circuit breaker events."""

import json
import os
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time

from .errors import ErrorSeverity, ProviderError
from .circuit_breaker import CircuitState


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification."""
    level: AlertLevel
    title: str
    message: str
    metadata: Dict[str, Any]
    timestamp: float
    source: str


AlertHandler = Callable[[Alert], None]


class AlertManager:
    """Manages alerting hooks and notifications."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.handlers: List[AlertHandler] = []
        self.alert_history: List[Alert] = []
        self.max_history: int = 1000
        
        # Thresholds
        self.error_rate_threshold: float = 0.5  # 50% error rate
        self.circuit_open_threshold: int = 3  # 3 circuits open
        self.critical_error_threshold: int = 5  # 5 critical errors
        
        # Counters
        self.error_counts: Dict[str, int] = {}
        self.circuit_open_count: int = 0
        self.critical_error_count: int = 0
        
        # Time windows
        self.error_window_seconds: float = 60.0
        self.error_timestamps: List[float] = []
    
    def register_handler(self, handler: AlertHandler):
        """
        Register an alert handler.
        
        Args:
            handler: Function that takes an Alert and sends notification
        """
        self.handlers.append(handler)
    
    def register_webhook_handler(self, webhook_url: str):
        """
        Register a webhook alert handler.
        
        Args:
            webhook_url: Webhook URL to send alerts to
        """
        import httpx
        
        async def webhook_handler(alert: Alert):
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        webhook_url,
                        json={
                            "level": alert.level.value,
                            "title": alert.title,
                            "message": alert.message,
                            "metadata": alert.metadata,
                            "timestamp": alert.timestamp
                        },
                        timeout=5.0
                    )
            except Exception as e:
                print(f"⚠️ Failed to send webhook alert: {e}")
        
        # Wrap async handler for sync calls
        def sync_handler(alert: Alert):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(webhook_handler(alert))
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(webhook_handler(alert))
        
        self.register_handler(sync_handler)
    
    def register_email_handler(self, email_config: Dict[str, Any]):
        """
        Register an email alert handler.
        
        Args:
            email_config: Email configuration (smtp_host, smtp_port, from_email, to_emails, etc.)
        """
        # Placeholder for email handler
        # Would need email library (smtplib or sendgrid, etc.)
        def email_handler(alert: Alert):
            if alert.level in (AlertLevel.ERROR, AlertLevel.CRITICAL):
                # Send email for errors and critical alerts
                print(f"📧 Email alert: {alert.title} - {alert.message}")
                # TODO: Implement actual email sending
        
        self.register_handler(email_handler)
    
    def _send_alert(self, alert: Alert):
        """
        Send alert to all registered handlers.
        
        Args:
            alert: Alert to send
        """
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Send to all handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"⚠️ Alert handler failed: {e}")
    
    def alert_circuit_breaker_opened(self, provider_name: str, failure_count: int, trace_context: Optional[Dict[str, Any]] = None):
        """
        Alert when circuit breaker opens.
        
        Args:
            provider_name: Name of provider
            failure_count: Number of failures
            trace_context: Trace context for correlation
        """
        self.circuit_open_count += 1
        
        alert = Alert(
            level=AlertLevel.ERROR,
            title=f"Circuit Breaker Opened: {provider_name}",
            message=f"Circuit breaker opened for {provider_name} after {failure_count} failures",
            metadata={
                "provider_name": provider_name,
                "failure_count": failure_count,
                "circuit_open_count": self.circuit_open_count,
                "trace_context": trace_context or {}
            },
            timestamp=time.time(),
            source="circuit_breaker"
        )
        
        self._send_alert(alert)
        
        # Alert if multiple circuits are open
        if self.circuit_open_count >= self.circuit_open_threshold:
            self.alert_multiple_circuits_open()
    
    def alert_multiple_circuits_open(self):
        """Alert when multiple circuit breakers are open."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Multiple Circuit Breakers Open",
            message=f"{self.circuit_open_count} circuit breakers are currently open",
            metadata={
                "circuit_open_count": self.circuit_open_count
            },
            timestamp=time.time(),
            source="circuit_breaker"
        )
        
        self._send_alert(alert)
    
    def alert_high_error_rate(self, provider_name: str, error_rate: float, trace_context: Optional[Dict[str, Any]] = None):
        """
        Alert when error rate exceeds threshold.
        
        Args:
            provider_name: Name of provider
            error_rate: Current error rate (0.0 to 1.0)
            trace_context: Trace context for correlation
        """
        alert = Alert(
            level=AlertLevel.WARNING,
            title=f"High Error Rate: {provider_name}",
            message=f"Error rate for {provider_name} is {error_rate:.1%} (threshold: {self.error_rate_threshold:.1%})",
            metadata={
                "provider_name": provider_name,
                "error_rate": error_rate,
                "threshold": self.error_rate_threshold,
                "trace_context": trace_context or {}
            },
            timestamp=time.time(),
            source="error_rate"
        )
        
        self._send_alert(alert)
    
    def alert_critical_error(self, error: ProviderError, trace_context: Optional[Dict[str, Any]] = None):
        """
        Alert on critical errors.
        
        Args:
            error: Provider error
            trace_context: Trace context for correlation
        """
        if error.severity == ErrorSeverity.CRITICAL:
            self.critical_error_count += 1
            
            alert = Alert(
                level=AlertLevel.CRITICAL,
                title=f"Critical Error: {error.provider_name}",
                message=f"Critical error in {error.provider_name}: {error.error_message}",
                metadata={
                    "provider_name": error.provider_name,
                    "error_type": error.error_type.value,
                    "error_message": error.error_message,
                    "severity": error.severity.value,
                    "status_code": error.status_code,
                    "trace_context": trace_context or {}
                },
                timestamp=error.timestamp,
                source="provider_error"
            )
            
            self._send_alert(alert)
            
            # Alert if too many critical errors
            if self.critical_error_count >= self.critical_error_threshold:
                self.alert_critical_error_threshold_breach()
    
    def alert_critical_error_threshold_breach(self):
        """Alert when critical error threshold is breached."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            title="Critical Error Threshold Breached",
            message=f"{self.critical_error_count} critical errors detected",
            metadata={
                "critical_error_count": self.critical_error_count,
                "threshold": self.critical_error_threshold
            },
            timestamp=time.time(),
            source="error_threshold"
        )
        
        self._send_alert(alert)
    
    def record_error(self, provider_name: str, error_type: str):
        """
        Record an error for rate calculation.
        
        Args:
            provider_name: Name of provider
            error_type: Type of error
        """
        current_time = time.time()
        self.error_timestamps.append(current_time)
        
        # Clean old timestamps
        cutoff_time = current_time - self.error_window_seconds
        self.error_timestamps = [ts for ts in self.error_timestamps if ts >= cutoff_time]
        
        # Calculate error rate
        if len(self.error_timestamps) > 0:
            error_rate = len(self.error_timestamps) / (self.error_window_seconds / 60.0)  # Errors per minute
            # Normalize to 0-1 range (assuming max 100 errors per minute)
            normalized_rate = min(error_rate / 100.0, 1.0)
            
            if normalized_rate >= self.error_rate_threshold:
                self.alert_high_error_rate(provider_name, normalized_rate)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
        
        Returns:
            List of alerts
        """
        return self.alert_history[-limit:]
    
    def reset_counters(self):
        """Reset alert counters."""
        self.circuit_open_count = 0
        self.critical_error_count = 0
        self.error_counts.clear()
        self.error_timestamps.clear()


# Global alert manager instance
alert_manager = AlertManager()


# Initialize alerting from environment
def initialize_alerting():
    """Initialize alerting from environment variables."""
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if webhook_url:
        alert_manager.register_webhook_handler(webhook_url)
    
    email_config_str = os.getenv("ALERT_EMAIL_CONFIG")
    if email_config_str:
        try:
            email_config = json.loads(email_config_str)
            alert_manager.register_email_handler(email_config)
        except Exception as e:
            print(f"⚠️ Failed to parse email config: {e}")


# Auto-initialize on import
initialize_alerting()


__all__ = [
    "AlertLevel",
    "Alert",
    "AlertManager",
    "AlertHandler",
    "alert_manager",
    "initialize_alerting"
]

