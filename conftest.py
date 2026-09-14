"""Root conftest — pytest adds this file's directory to sys.path, which is what
makes `from src.payload import ...` work in tests without installing anything."""

import pytest  # type: ignore[import-not-found]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: needs live Qdrant / Ollama / Postgres — skipped by default"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless -m integration is passed explicitly."""
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="needs live services; run with -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
