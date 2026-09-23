"""query_history writes, tested with a fake connection — no Postgres."""

import pytest

from src import history


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self):
        self.log = []

    def cursor(self, **_):
        return FakeCursor(self.log)


@pytest.fixture
def conn(monkeypatch):
    fake = FakeConnection()
    monkeypatch.setattr(history, "get_connection", lambda config=None: fake)
    return fake


def test_insert_carries_token_metrics(conn):
    history.save_query_history(
        session_id="s", user_query="q", generated_answer="a", route="agent",
        tool_used="none", latency_ms=1500.0, success=True,
        tokens_used=420, tokens_per_sec=31.5, total_tokens_per_sec=280.0,
    )
    sql, params = conn.log[-1]
    assert "tokens_used, tokens_per_sec, total_tokens_per_sec" in sql
    assert "retrieval_count" not in sql
    assert params[-3:] == (420, 31.5, 280.0)


def test_token_metrics_are_optional(conn):
    history.save_query_history("s", "q", "a", "agent", "none", 10.0, True)
    assert conn.log[-1][1][-3:] == (None, None, None)


def test_schema_drops_retrieval_count_on_existing_tables(conn):
    history.ensure_schema()
    statements = [sql for sql, _ in conn.log]
    assert "retrieval_count" not in statements[0]          # fresh tables never get it
    assert any("DROP COLUMN IF EXISTS retrieval_count" in s for s in statements)
