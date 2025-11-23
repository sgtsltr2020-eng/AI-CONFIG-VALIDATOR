"""
Rate limiting utility for Vesper API.
Implements token bucket or sliding window algorithm for rate limiting.
"""

import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Optional, Dict, Tuple

from fastapi import Request, HTTPException
from config import config

class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    """
    def __init__(self):
        # Storage: ip -> {"timestamps": [t1, t2, ...]}
        self.requests: Dict[str, Dict[str, list]] = defaultdict(lambda: {"timestamps": []})
        self.enabled = config.RATE_LIMIT_ENABLED
        self.limit = config.RATE_LIMIT_PER_MINUTE
        self.window = 60  # 1 minute

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed for the given key (IP).
        Returns (allowed, retry_after_seconds).
        """
        if not self.enabled:
            return True, None

        now = time.time()
        history = self.requests[key]["timestamps"]
        
        # Clean up old timestamps
        history = [t for t in history if t > now - self.window]
        self.requests[key]["timestamps"] = history
        
        if len(history) >= self.limit:
            # Calculate wait time
            oldest = history[0]
            retry_after = int(self.window - (now - oldest)) + 1
            return False, retry_after
        
        # Add new request
        history.append(now)
        return True, None

# Global instance
rate_limiter = RateLimiter()

def rate_limit(limit: Optional[int] = None):
    """
    Decorator to apply rate limiting to an endpoint.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract Request object
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request and "request" in kwargs:
                request = kwargs["request"]
            
            if request:
                # Handle test clients where request.client might be None
                client_ip = request.client.host if request.client else "test-client"
                
                # Use global limit if not specified
                # Note: A real implementation might support per-endpoint limits in the RateLimiter class
                allowed, retry_after = rate_limiter.is_allowed(client_ip)
                
                if not allowed:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Try again in {retry_after} seconds."
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
