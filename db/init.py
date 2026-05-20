"""Database connection and schema initialisation.

Internal helper used by all db sub-modules.  Callers outside the db package
should import from ``db.client`` rather than here directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row


def _connect(db_dsn: str) -> psycopg.Connection:
	"""Open a psycopg3 connection with dict_row factory and pgvector support."""
	conn = psycopg.connect(db_dsn, row_factory=dict_row)
	try:
		from pgvector.psycopg import register_vector  # noqa: PLC0415
		register_vector(conn)
	except Exception:  # noqa: BLE001
		pass  # pgvector adapter optional; vector columns still work as text
	return conn


# Per-process guard: init_db() is idempotent but runs schema + migrations on
# every call.  Skip once schema is confirmed for this DSN in this process.
_initialized: set[str] = set()


# Idempotent column migrations — safe to run on existing databases.
# PostgreSQL supports ADD COLUMN IF NOT EXISTS so no try/except is needed.
_CHUNK_MIGRATIONS = [
	"ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'pdf_book'",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'pdf_book'",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS structural_role TEXT NOT NULL DEFAULT 'body'",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS collection_id TEXT",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_name TEXT",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_title TEXT",
	"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_path TEXT",
	"ALTER TABLE collections ADD COLUMN IF NOT EXISTS parent_id TEXT REFERENCES collections(collection_id) ON DELETE SET NULL",
	# Migration 8 — file content hash for change-detection (session 79)
	"ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash TEXT",
	# Migration 9 — background job queue table (session 79)
	"""CREATE TABLE IF NOT EXISTS jobs (
		job_id       TEXT PRIMARY KEY,
		job_type     TEXT NOT NULL,
		params_json  TEXT NOT NULL DEFAULT '{}',
		status       TEXT NOT NULL DEFAULT 'pending',
		attempts     INTEGER NOT NULL DEFAULT 0,
		max_attempts INTEGER NOT NULL DEFAULT 3,
		error        TEXT,
		created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		started_at   TIMESTAMPTZ,
		finished_at  TIMESTAMPTZ
	)""",
	"CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)",
]


def _split_sql_statements(sql: str) -> list[str]:
	"""Split a SQL script on semicolons, skipping empty statements."""
	return [s.strip() for s in sql.split(";") if s.strip()]


def _run_migrations(conn: psycopg.Connection) -> None:
	"""Apply schema migrations, recording each in schema_migrations."""
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS schema_migrations (
			id         INTEGER PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
		"""
	)
	conn.commit()
	applied = {row["id"] for row in conn.execute("SELECT id FROM schema_migrations").fetchall()}
	for idx, sql in enumerate(_CHUNK_MIGRATIONS):
		if idx not in applied:
			conn.execute(sql)
			conn.execute(
				"INSERT INTO schema_migrations (id) VALUES (%s) ON CONFLICT DO NOTHING", (idx,)
			)
			conn.commit()


def init_db(db_dsn: str, schema_path: Optional[str] = None) -> None:
	if db_dsn in _initialized:
		return
	if schema_path:
		schema_sql = Path(schema_path).read_text(encoding="utf-8")
	else:
		default_schema = Path(__file__).resolve().parent / "schema.sql"
		schema_sql = default_schema.read_text(encoding="utf-8")
	# DDL must run in autocommit mode in PostgreSQL (outside a transaction block).
	with psycopg.connect(db_dsn, autocommit=True) as ddl_conn:
		try:
			from pgvector.psycopg import register_vector  # noqa: PLC0415
			register_vector(ddl_conn)
		except Exception:  # noqa: BLE001
			pass
		for stmt in _split_sql_statements(schema_sql):
			try:
				ddl_conn.execute(stmt)
			except psycopg.errors.DuplicateObject:
				pass  # extension / index already exists
	conn = _connect(db_dsn)
	try:
		_run_migrations(conn)
		_initialized.add(db_dsn)
	finally:
		conn.close()
