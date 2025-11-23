"""Circuit breaker pattern implementation for AI providers."""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from collections import deque

from .errors import ErrorType


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    success_threshold: int = 2  # Number of successes to close circuit (half-open -> closed)
    timeout_seconds: float = 60.0  # Time to wait before attempting recovery
    timeout_threshold_ms: float = 5000.0  # Response time threshold (ms) to count as failure
    monitor_window_seconds: float = 60.0  # Time window for monitoring failures


@dataclass
class FailureRecord:
    """Record of a failure."""
    timestamp: float
    error_type: ErrorType
    response_time_ms: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker for provider failure handling."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker (usually provider name)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.failure_records: deque = deque(maxlen=100)  # Keep last 100 failures
        self.state_changed_at: float = time.time()
    
    def record_success(self, response_time_ms: Optional[float] = None):
        """
        Record a successful operation.
        
        Args:
            response_time_ms: Response time in milliseconds
        """
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            # If we have enough successes, close the circuit
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Clean old failures on success (success indicates recovery)
            # Keep recent failures for monitoring, but reduce failure count
            current_time = time.time()
            cutoff_time = current_time - self.config.monitor_window_seconds
            recent_failures = [fr for fr in self.failure_records if fr.timestamp >= cutoff_time]
            self.failure_records = deque(recent_failures, maxlen=100)
            self.failure_count = len(self.failure_records)
    
    def record_failure(self, error_type: ErrorType, response_time_ms: Optional[float] = None):
        """
        Record a failed operation.
        
        Args:
            error_type: Type of error that occurred
            response_time_ms: Response time in milliseconds (if available)
        """
        self.last_failure_time = time.time()
        current_time = time.time()
        
        # Record failure
        self.failure_records.append(FailureRecord(
            timestamp=current_time,
            error_type=error_type,
            response_time_ms=response_time_ms
        ))
        
        # Clean old failures outside monitoring window and count recent failures
        cutoff_time = current_time - self.config.monitor_window_seconds
        recent_failures = [fr for fr in self.failure_records if fr.timestamp >= cutoff_time]
        self.failure_records = deque(recent_failures, maxlen=100)
        
        # Update failure count based on recent failures in monitoring window
        self.failure_count = len(self.failure_records)
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._open_circuit()
    
    def record_slow_response(self, response_time_ms: float):
        """
        Record a slow response (counts as failure if over threshold).
        
        Args:
            response_time_ms: Response time in milliseconds
        """
        if response_time_ms > self.config.timeout_threshold_ms:
            self.record_failure(ErrorType.TIMEOUT, response_time_ms)
    
    def _open_circuit(self):
        """Open the circuit (stop allowing requests)."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.state_changed_at = time.time()
            self.success_count = 0
            print(f"🔴 Circuit breaker OPEN for {self.name} after {self.failure_count} failures")
            
            # Import here to avoid circular dependency
            from .alerting import alert_manager
            from .tracing import get_trace_context
            
            # Send alert
            trace_context = get_trace_context()
            alert_manager.alert_circuit_breaker_opened(
                self.name,
                self.failure_count,
                trace_context
            )
    
    def _close_circuit(self):
        """Close the circuit (allow requests)."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.state_changed_at = time.time()
            self.failure_count = 0
            self.success_count = 0
            print(f"🟢 Circuit breaker CLOSED for {self.name}")
    
    def _try_half_open(self):
        """Try to move to half-open state."""
        if self.state == CircuitState.OPEN:
            time_since_open = time.time() - self.state_changed_at
            if time_since_open >= self.config.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.state_changed_at = time.time()
                self.success_count = 0
                print(f"🟡 Circuit breaker HALF_OPEN for {self.name} (testing recovery)")
                return True
        return False
    
    def can_attempt(self) -> bool:
        """
        Check if a request can be attempted.
        
        Returns:
            True if request can be attempted, False otherwise
        """
        # Update state if needed
        if self.state == CircuitState.OPEN:
            self._try_half_open()
        
        # Allow requests in CLOSED or HALF_OPEN states
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        # Update state if needed
        if self.state == CircuitState.OPEN:
            self._try_half_open()
        
        return self.state
    
    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.
        
        Returns:
            Dictionary with circuit breaker statistics
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "state_changed_at": self.state_changed_at,
            "failure_records_count": len(self.failure_records),
            "can_attempt": self.can_attempt()
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.failure_records.clear()
        self.state_changed_at = time.time()
        print(f"🔄 Circuit breaker RESET for {self.name}")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker for a provider.
        
        Args:
            name: Name of the provider
            config: Circuit breaker configuration
        
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def get_all_stats(self) -> dict:
        """
        Get statistics for all circuit breakers.
        
        Returns:
            Dictionary with all circuit breaker statistics
        """
        return {
            name: cb.get_stats()
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb.reset()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "FailureRecord",
    "CircuitBreaker",
    "CircuitBreakerManager",
    "circuit_breaker_manager"
]

