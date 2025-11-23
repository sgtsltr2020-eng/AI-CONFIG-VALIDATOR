"""
Quota tracking system for GitHub Models.
Manages rate limits (RPM, RPH, RPD) and token usage.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from memory import memory
from config import config

logger = logging.getLogger(__name__)

class QuotaTracker:
    """
    Tracks usage quotas for AI models.
    Persists state to SQLite to survive restarts.
    """
    
    # Default conservative limits (Safety First)
    DEFAULT_LIMITS = {
        "requests_minute": 10,
        "requests_hour": 100,
        "requests_day": 1000
    }
    
    def __init__(self):
        self.memory = memory
        self._ensure_table_exists()
        
    def _ensure_table_exists(self):
        """Ensure quota table exists (handled by memory.py, but good for safety)"""
        pass  # Assumed handled by memory.init_database()

    def check_availability(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if model is available based on current usage.
        Returns: (is_available, reason_if_unavailable)
        """
        if not config.ENABLE_GITHUB_MODELS_ROUTING:
            return False, "GitHub Models routing disabled"
            
        self._reset_counters_if_needed(model_name)
        
        usage = self._get_usage(model_name)
        if not usage:
            return True, "OK"  # No usage record yet
            
        # Check limits
        # Note: In future, load dynamic limits from config or DB
        limits = self.DEFAULT_LIMITS
        
        if usage['requests_minute'] >= limits['requests_minute']:
            return False, f"RPM limit reached ({usage['requests_minute']}/{limits['requests_minute']})"
            
        if usage['requests_hour'] >= limits['requests_hour']:
            return False, f"RPH limit reached ({usage['requests_hour']}/{limits['requests_hour']})"
            
        if usage['requests_day'] >= limits['requests_day']:
            return False, f"RPD limit reached ({usage['requests_day']}/{limits['requests_day']})"
            
        return True, "OK"

    def increment_usage(self, model_name: str, tokens: int = 0):
        """Increment usage counters for a model"""
        self._reset_counters_if_needed(model_name)
        
        now = datetime.now()
        cursor = self.memory.conn.cursor()
        
        # Upsert usage
        cursor.execute("""
            INSERT INTO model_quotas (
                model_name, 
                requests_minute, requests_hour, requests_day,
                minute_reset_at, hour_reset_at, day_reset_at,
                tokens_used_minute, tokens_used_day,
                last_updated
            ) VALUES (?, 1, 1, 1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                requests_minute = requests_minute + 1,
                requests_hour = requests_hour + 1,
                requests_day = requests_day + 1,
                tokens_used_minute = tokens_used_minute + excluded.tokens_used_minute,
                tokens_used_day = tokens_used_day + excluded.tokens_used_day,
                last_updated = excluded.last_updated
        """, (
            model_name,
            now + timedelta(minutes=1),
            now + timedelta(hours=1),
            now + timedelta(days=1),
            tokens, tokens,
            now
        ))
        
        self.memory.conn.commit()

    def _get_usage(self, model_name: str) -> Optional[Dict]:
        """Get current usage for a model"""
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM model_quotas WHERE model_name = ?", (model_name,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def _reset_counters_if_needed(self, model_name: str):
        """Reset counters if time window has passed"""
        usage = self._get_usage(model_name)
        if not usage:
            return

        now = datetime.now()
        updates = []
        params = []
        
        # Parse timestamps (SQLite stores as string usually)
        # Note: memory.py sets row_factory=sqlite3.Row, but types might be strings
        
        try:
            minute_reset = datetime.fromisoformat(str(usage['minute_reset_at'])) if usage['minute_reset_at'] else None
            hour_reset = datetime.fromisoformat(str(usage['hour_reset_at'])) if usage['hour_reset_at'] else None
            day_reset = datetime.fromisoformat(str(usage['day_reset_at'])) if usage['day_reset_at'] else None
        except ValueError:
            # Handle case where timestamp might be in different format or None
            minute_reset = None
            
        if minute_reset and now >= minute_reset:
            updates.append("requests_minute = 0, tokens_used_minute = 0, minute_reset_at = ?")
            params.append(now + timedelta(minutes=1))
            
        if hour_reset and now >= hour_reset:
            updates.append("requests_hour = 0, hour_reset_at = ?")
            params.append(now + timedelta(hours=1))
            
        if day_reset and now >= day_reset:
            updates.append("requests_day = 0, tokens_used_day = 0, day_reset_at = ?")
            params.append(now + timedelta(days=1))
            
        if updates:
            params.append(model_name)
            sql = f"UPDATE model_quotas SET {', '.join(updates)} WHERE model_name = ?"
            cursor = self.memory.conn.cursor()
            cursor.execute(sql, params)
            self.memory.conn.commit()

# Global instance
quota_tracker = QuotaTracker()
