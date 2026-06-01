"""
limiter.py — Simple rate limiter (per IP address)

Prevent abuse and control load on the local LLM.

"""

import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_calls: int = 10, period: int = 60):
        """
        max_calls: number of allowed requests
        period: time window in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self._records: dict = defaultdict(list)  # { ip: [timestamps] }

    def is_allowed(self, client_id: str) -> bool:
        """
        Returns True if the client is within the rate limit.
        Removes timestamps outside the current window.
        """
        now = time.time()
        window_start = now - self.period

        # Keep only timestamps within the current window
        self._records[client_id] = [
            t for t in self._records[client_id] if t > window_start
        ]

        if len(self._records[client_id]) < self.max_calls:
            self._records[client_id].append(now)
            return True

        return False  # limit exceeded
