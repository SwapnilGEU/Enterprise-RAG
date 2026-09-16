"""Postgres query history — notebook Section 18.

Every answered query is logged here: what was asked, what came back, which tool
ran, how long it took. sql_tool in agent.py queries this same table, which is
what lets the agent answer questions about its own usage.

Connection settings come from CONFIG, so this module and the SQL agent's
connection URI can never drift apart.

Thread safety
-------------
One process-wide connection, guarded by a lock. That was unnecessary while this
only ran from scripts, where there is exactly one caller. Under FastAPI it is
not: sync endpoints run in a threadpool, so several requests can reach
`save_query_history` at once, and a psycopg2 connection is **not** safe for
concurrent use across threads — interleaved statements on one connection
corrupt the protocol rather than merely racing.

A lock, not a pool, because the writes here are tiny and rare (one INSERT per
answered query, long after the slow part) so serialising them costs nothing
measurable. If query history ever becomes a read-heavy API of its own, swap in
`psycopg2.pool.ThreadedConnectionPool` — the lock is the cheap correct thing,
not the sophisticated one.
"""

import threading

from src.config import CONFIG, Config, logger


SCHEMA = """
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_query TEXT NOT NULL,
    generated_answer TEXT,
    latency_ms FLOAT,
    success BOOLEAN,
    tokens_used INTEGER,
    retrieval_count INTEGER,
    error_type VARCHAR(100),
    route VARCHAR(50),
    tool_used VARCHAR(100)
);
"""


_connection = None

# Guards creating the connection AND using it. See the module docstring.
_lock = threading.RLock()


def get_connection(config: Config = CONFIG):
    """Connect to Postgres. Lazy, one connection per process — importing this
    module must not open a database connection as a side effect. A plain
    singleton rather than @lru_cache, which cannot key on the Config dataclass.

    Double-checked under a lock: without it, two threads arriving together both
    see None and both connect, and one connection is silently orphaned.
    """
    global _connection
    if _connection is not None:
        return _connection

    with _lock:
        if _connection is not None:      # another thread won the race
            return _connection
        return _connect(config)


def _connect(config: Config):
    global _connection
    import psycopg2

    _connection = psycopg2.connect(
        dbname=config.pg_dbname,
        user=config.pg_user,
        password=config.pg_password,
        host=config.pg_host,
        port=config.pg_port,
        connect_timeout=config.pg_connect_timeout_seconds,
    )
    _connection.autocommit = True
    logger.info(f"Connected to Postgres {config.pg_host}:{config.pg_port}/{config.pg_dbname}")
    return _connection


def ensure_schema(config: Config = CONFIG) -> None:
    """Create query_history if it doesn't exist. Call once at startup."""
    with _lock, get_connection(config).cursor() as cursor:
        cursor.execute(SCHEMA)
    logger.info("query_history table ready.")


def save_query_history(session_id: str, user_query: str, generated_answer: str,
                       route: str, tool_used: str, latency_ms: float,
                       success: bool, config: Config = CONFIG) -> None:
    """Log one answered query. Never raises: a logging failure must not take
    down an answer the user already has."""
    insert = """
    INSERT INTO query_history (
        session_id, user_query, generated_answer, route, tool_used, latency_ms, success
    ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with _lock, get_connection(config).cursor() as cursor:
            cursor.execute(insert, (
                session_id, user_query, generated_answer, route, tool_used, latency_ms, success
            ))
    except Exception as exc:
        logger.error(f"save_query_history failed (continuing anyway): {exc!r}")


def recent_queries(limit: int = 10, config: Config = CONFIG) -> list[dict]:
    """Small convenience reader — handy in the notebook and in tests."""
    from psycopg2.extras import RealDictCursor

    with _lock, get_connection(config).cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT %s;", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
