"""Lightweight disk cache with TTL for market and fundamental data."""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.pkl"


def get_cached(key: str, ttl_seconds: int = 21600) -> Any | None:
    """Return cached object if it exists and has not expired."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if time.time() - payload["ts"] > ttl_seconds:
            return None
        return payload["data"]
    except Exception:
        return None


def set_cached(key: str, data: Any) -> None:
    """Persist object to disk cache with current timestamp."""
    path = _cache_path(key)
    try:
        with path.open("wb") as f:
            pickle.dump({"ts": time.time(), "data": data}, f)
    except Exception:
        pass


def clear_cache() -> int:
    """Delete all cached files and return count removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for p in CACHE_DIR.glob("*.pkl"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed
