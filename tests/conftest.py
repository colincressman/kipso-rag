"""
Session-level teardown for the test suite.

Ensures every transformer loaded during the session is released from VRAM
after all tests finish.  Without this, DeBERTa (~1.5 GB) and the CrossEncoder
(~400 MB) stay resident until the pytest process exits, which is too late on
shared / low-VRAM machines.
"""
from __future__ import annotations

import uuid
import pytest

_PG_ADMIN_DSN = "postgresql://postgres:postgres@localhost/postgres"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_postgres: mark test as requiring a live PostgreSQL server",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_postgres"):
        try:
            import psycopg
            conn = psycopg.connect(_PG_ADMIN_DSN, connect_timeout=2)
            conn.close()
        except Exception:
            pytest.skip("PostgreSQL not available (set up a local server to run these tests)")


@pytest.fixture()
def pg_dsn():
    """Create a fresh isolated PostgreSQL database for a single test, then drop it."""
    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not installed")

    db_name = f"rag_test_{uuid.uuid4().hex[:12]}"
    try:
        with psycopg.connect(_PG_ADMIN_DSN, autocommit=True, connect_timeout=2) as conn:
            conn.execute(f'CREATE DATABASE "{db_name}"')
    except Exception:
        pytest.skip("PostgreSQL not available")

    dsn = f"postgresql://postgres:postgres@localhost/{db_name}"
    try:
        from db.client import init_db
        init_db(dsn)
        yield dsn
    finally:
        try:
            with psycopg.connect(_PG_ADMIN_DSN, autocommit=True, connect_timeout=2) as conn:
                conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        except Exception:
            pass


@pytest.fixture(scope="session", autouse=True)
def _unload_transformer_models():
    """Unload all transformer singletons after the test session ends."""
    yield
    try:
        from retrieval import intent_classifier2
        intent_classifier2.unload()
    except Exception:  # noqa: BLE001
        pass
    try:
        from retrieval import cross_encoder
        cross_encoder.unload()
    except Exception:  # noqa: BLE001
        pass


# ── Section headers ────────────────────────────────────────────────────────────

_last_section: dict = {"module": None, "cls": None}


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Print a header line whenever the test file or class changes."""
    parts = nodeid.split("::")
    module = parts[0] if len(parts) >= 1 else ""
    cls    = parts[1] if len(parts) >= 3 else None   # class present only if 3 parts

    if module != _last_section["module"]:
        # New file
        label = module.replace("tests/", "").replace("tests\\", "")
        print(f"\n{'-' * 72}")
        print(f"  FILE  {label}")
        print(f"{'-' * 72}")
        _last_section["module"] = module
        _last_section["cls"] = None

    if cls and cls != _last_section["cls"]:
        print(f"\n  > {cls}")
        _last_section["cls"] = cls
