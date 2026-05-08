"""
cache.py — Simple in-memory response cache

Why: Avoid calling the LLM for the same question twice.
     This reduces latency and saves compute.

Interview tip: "I used a TTL-based in-memory cache. 
For production, this could be replaced with Redis."
"""

import time


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        """
        ttl_seconds: how long to keep a cached answer (default 5 minutes)
        """
        self._store: dict = {}   # { question: (answer, timestamp) }
        self.ttl = ttl_seconds

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl

    def get(self, question: str):
        """Return cached answer if it exists and hasn't expired."""
        key = question.lower().strip()
        if key in self._store:
            answer, timestamp = self._store[key]
            if not self._is_expired(timestamp):
                return answer
            else:
                del self._store[key]  # clean up expired entry
        return None

    def set(self, question: str, answer: str):
        """Store answer with current timestamp."""
        key = question.lower().strip()
        self._store[key] = (answer, time.time())

    def size(self) -> int:
        """Return number of cached items."""
        return len(self._store)

    def clear(self):
        self._store.clear()
