"""Job queue CRUD — powers the background worker thread.

Jobs advance through:  pending → running → done | failed

On failure the worker increments attempts and re-queues (status back to
'pending') unless max_attempts is reached, in which case status is 'failed'.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from db.init import _connect, init_db


def enqueue_job(
	db_dsn: str,
	job_type: str,
	params: Optional[Dict[str, Any]] = None,
	*,
	max_attempts: int = 3,
) -> str:
	"""Insert a new pending job and return its job_id."""
	init_db(db_dsn)
	job_id = str(uuid.uuid4())
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"""
			INSERT INTO jobs (job_id, job_type, params_json, status, max_attempts)
			VALUES (%s, %s, %s, 'pending', %s)
			""",
			(job_id, job_type, json.dumps(params or {}, ensure_ascii=False), max_attempts),
		)
		conn.commit()
	finally:
		conn.close()
	return job_id


def claim_next_job(db_dsn: str) -> Optional[Dict[str, Any]]:
	"""Atomically claim the oldest pending job and transition it to 'running'.

	Returns the job dict (with parsed params) or None if no pending jobs exist.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"""
			UPDATE jobs
			SET status = 'running',
			    started_at = NOW(),
			    attempts = attempts + 1
			WHERE job_id = (
				SELECT job_id FROM jobs
				WHERE status = 'pending'
				  AND attempts < max_attempts
				ORDER BY created_at ASC
				LIMIT 1
				FOR UPDATE SKIP LOCKED
			)
			RETURNING *
			"""
		).fetchone()
		conn.commit()
		if row is None:
			return None
		d = dict(row)
		try:
			d["params"] = json.loads(d.get("params_json") or "{}")
		except Exception:
			d["params"] = {}
		return d
	finally:
		conn.close()


def complete_job(db_dsn: str, job_id: str) -> None:
	"""Mark a job as done."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"UPDATE jobs SET status='done', finished_at=NOW() WHERE job_id=%s",
			(job_id,),
		)
		conn.commit()
	finally:
		conn.close()


def fail_job(db_dsn: str, job_id: str, error: str, *, retry: bool = True) -> None:
	"""Record a failure.  Re-queues to 'pending' if retry=True and attempts < max_attempts.

	Otherwise sets status to 'failed' permanently.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if retry:
			conn.execute(
				"""
				UPDATE jobs
				SET error = %s,
				    finished_at = NOW(),
				    status = CASE
				        WHEN attempts < max_attempts THEN 'pending'
				        ELSE 'failed'
				    END
				WHERE job_id = %s
				""",
				(error[:2000], job_id),
			)
		else:
			conn.execute(
				"UPDATE jobs SET status='failed', error=%s, finished_at=NOW() WHERE job_id=%s",
				(error[:2000], job_id),
			)
		conn.commit()
	finally:
		conn.close()


def cancel_job(db_dsn: str, job_id: str) -> bool:
	"""Cancel a pending or running job.

	Sets status to 'cancelled' and finished_at to now.
	Returns True if a row was updated, False if the job was not found or
	already in a terminal state (done/failed/cancelled).
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"""
			UPDATE jobs
			SET status = 'cancelled', finished_at = NOW(), error = 'Cancelled by user'
			WHERE job_id = %s AND status IN ('pending', 'running')
			RETURNING job_id
			""",
			(job_id,),
		).fetchone()
		conn.commit()
		return row is not None
	finally:
		conn.close()


def get_job(db_dsn: str, job_id: str) -> Optional[Dict[str, Any]]:
	"""Return the job dict for *job_id*, or None if not found."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT * FROM jobs WHERE job_id = %s",
			(job_id,),
		).fetchone()
		if row is None:
			return None
		d = dict(row)
		try:
			d["params"] = json.loads(d.get("params_json") or "{}")
		except Exception:
			d["params"] = {}
		return d
	finally:
		conn.close()


def list_jobs(
	db_dsn: str,
	status: Optional[str] = None,
	limit: int = 50,
) -> List[Dict[str, Any]]:
	"""Return recent jobs, optionally filtered by status."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if status:
			rows = conn.execute(
				"SELECT * FROM jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
				(status, limit),
			).fetchall()
		else:
			rows = conn.execute(
				"SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s",
				(limit,),
			).fetchall()
		result = []
		for row in rows:
			d = dict(row)
			try:
				d["params"] = json.loads(d.get("params_json") or "{}")
			except Exception:
				d["params"] = {}
			result.append(d)
		return result
	finally:
		conn.close()
