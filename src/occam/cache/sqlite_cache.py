"""SQLite-based caching for API responses."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from occam.utils import stable_hash


class SQLiteCache:
    """SQLite cache for storing API responses.

    Keys are computed from provider, model, base_url, and request JSON.
    """

    def __init__(self, db_path: str | Path = "cache.db"):
        """Initialize the cache.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                base_url TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_raw TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup
            ON cache (provider, model, base_url, request_hash)
        """)

        self._conn.commit()

    def _compute_key(
        self,
        provider: str,
        model: str,
        base_url: str,
        request: dict[str, Any],
    ) -> tuple[str, str]:
        """Compute cache key from request parameters.

        Args:
            provider: Provider name.
            model: Model identifier.
            base_url: API base URL.
            request: Request payload.

        Returns:
            Tuple of (cache_key, request_hash).
        """
        request_hash = stable_hash(request)
        # Include all parameters in the key for uniqueness
        key_data = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "request_hash": request_hash,
        }
        cache_key = stable_hash(key_data)
        return cache_key, request_hash

    def get(
        self,
        provider: str,
        model: str,
        base_url: str,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retrieve a cached response.

        Args:
            provider: Provider name.
            model: Model identifier.
            base_url: API base URL.
            request: Request payload.

        Returns:
            Cached response dict with 'text' and 'raw' keys, or None if not found.
        """
        if self._conn is None:
            return None

        cache_key, _ = self._compute_key(provider, model, base_url, request)

        cursor = self._conn.execute(
            "SELECT response_text, response_raw FROM cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "text": row["response_text"],
            "raw": json.loads(row["response_raw"]),
        }

    def set(
        self,
        provider: str,
        model: str,
        base_url: str,
        request: dict[str, Any],
        response_text: str,
        response_raw: dict[str, Any],
    ) -> None:
        """Store a response in the cache.

        Args:
            provider: Provider name.
            model: Model identifier.
            base_url: API base URL.
            request: Request payload.
            response_text: Response text content.
            response_raw: Full response JSON.
        """
        if self._conn is None:
            return

        cache_key, request_hash = self._compute_key(provider, model, base_url, request)
        created_at = datetime.utcnow().isoformat()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache
            (cache_key, provider, model, base_url, request_hash, response_text, response_raw, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                provider,
                model,
                base_url,
                request_hash,
                response_text,
                json.dumps(response_raw),
                created_at,
            ),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._conn is None:
            return

        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with 'total_entries' count.
        """
        if self._conn is None:
            return {"total_entries": 0}

        cursor = self._conn.execute("SELECT COUNT(*) as count FROM cache")
        row = cursor.fetchone()
        return {"total_entries": row["count"] if row else 0}

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SQLiteCache":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class NoCache:
    """Dummy cache that doesn't cache anything.

    Used when --no-cache flag is passed.
    """

    def get(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:
        pass

    def clear(self) -> None:
        pass

    def stats(self) -> dict[str, int]:
        return {"total_entries": 0}

    def close(self) -> None:
        pass

    def __enter__(self) -> "NoCache":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
