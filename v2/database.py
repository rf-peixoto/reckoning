"""
database.py – SQLite persistence for workflows, executions, and update history.

All public functions open and close their own connection so they are safe
to call from any thread without holding a shared connection object.
WAL mode is enabled on every new connection for safe concurrent reads/writes.
"""
import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from models import (
    ExecutionResult,
    ToolConfig,
    UpdateResult,
    Workflow,
    WorkflowExecution,
)

DB_PATH = "reckoning.db"
_write_lock = threading.RLock()  # serialise writes; reads are lock-free (WAL)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip() == "":
            return None
        return int(v)
    except Exception:
        return None


# ──────────────────────────────────────────────
# Schema initialisation
# ──────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not already exist."""
    with _write_lock:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    id                       TEXT PRIMARY KEY,
                    name                     TEXT NOT NULL DEFAULT '',
                    description              TEXT NOT NULL DEFAULT '',
                    author                   TEXT NOT NULL DEFAULT 'anonymous',
                    run_mode                 TEXT NOT NULL DEFAULT 'once',
                    interval_minutes         INTEGER,
                    repeat_count             INTEGER,
                    repeat_interval_minutes  INTEGER,
                    scheduled_at             TEXT,
                    tools_json               TEXT NOT NULL DEFAULT '[]',
                    created_at               TEXT,
                    updated_at               TEXT
                );

                CREATE TABLE IF NOT EXISTS executions (
                    execution_id             TEXT PRIMARY KEY,
                    workflow_id              TEXT,
                    domain                   TEXT NOT NULL DEFAULT '',
                    notes                    TEXT NOT NULL DEFAULT '',
                    status                   TEXT NOT NULL DEFAULT 'queued',
                    run_mode                 TEXT NOT NULL DEFAULT 'once',
                    interval_minutes         INTEGER,
                    repeat_count             INTEGER,
                    repeat_interval_minutes  INTEGER,
                    scheduled_at             TEXT,
                    current_iteration        INTEGER NOT NULL DEFAULT 0,
                    planned_iterations       INTEGER,
                    created_at               TEXT,
                    started_at               TEXT,
                    completed_at             TEXT,
                    cancelled_at             TEXT,
                    cancel_requested         INTEGER NOT NULL DEFAULT 0,
                    version                  INTEGER NOT NULL DEFAULT 0,
                    last_updated_at          TEXT,
                    events_json              TEXT NOT NULL DEFAULT '[]',
                    results_json             TEXT NOT NULL DEFAULT '{}',
                    iterations_json          TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS update_history (
                    update_id   TEXT PRIMARY KEY,
                    tool_id     TEXT NOT NULL DEFAULT '',
                    tool_name   TEXT NOT NULL DEFAULT '',
                    status      TEXT NOT NULL DEFAULT 'pending',
                    start_time  TEXT,
                    end_time    TEXT,
                    output      TEXT NOT NULL DEFAULT '',
                    error       TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_exec_workflow
                    ON executions (workflow_id);
                CREATE INDEX IF NOT EXISTS idx_exec_status
                    ON executions (status);
                CREATE INDEX IF NOT EXISTS idx_exec_created
                    ON executions (created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_exec_scheduled
                    ON executions (scheduled_at)
                    WHERE status = 'scheduled';
                """
            )
            conn.commit()
        finally:
            conn.close()


# ──────────────────────────────────────────────
# Workflow CRUD
# ──────────────────────────────────────────────

def _row_to_workflow(row: sqlite3.Row) -> Workflow:
    tools_raw = row["tools_json"] or "[]"
    try:
        tools_list = json.loads(tools_raw)
    except Exception:
        tools_list = []
    tools = [ToolConfig.from_dict(t) for t in tools_list]
    return Workflow(
        workflow_id=row["id"],
        name=row["name"],
        description=row["description"],
        tools=tools,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        author=row["author"],
        run_mode=row["run_mode"],
        interval_minutes=row["interval_minutes"],
        repeat_count=row["repeat_count"],
        repeat_interval_minutes=row["repeat_interval_minutes"],
        scheduled_at=row["scheduled_at"],
    )


def list_workflows() -> List[Workflow]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM workflows ORDER BY updated_at DESC"
        ).fetchall()
        return [_row_to_workflow(r) for r in rows]
    finally:
        conn.close()


def get_workflow(workflow_id: str) -> Optional[Workflow]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
        ).fetchone()
        return _row_to_workflow(row) if row else None
    finally:
        conn.close()


def save_workflow(wf: Workflow) -> None:
    tools_json = json.dumps([t.to_dict() for t in wf.tools])
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO workflows
                    (id, name, description, author, run_mode, interval_minutes,
                     repeat_count, repeat_interval_minutes, scheduled_at,
                     tools_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name                    = excluded.name,
                    description             = excluded.description,
                    author                  = excluded.author,
                    run_mode                = excluded.run_mode,
                    interval_minutes        = excluded.interval_minutes,
                    repeat_count            = excluded.repeat_count,
                    repeat_interval_minutes = excluded.repeat_interval_minutes,
                    scheduled_at            = excluded.scheduled_at,
                    tools_json              = excluded.tools_json,
                    updated_at              = excluded.updated_at
                """,
                (
                    wf.id, wf.name, wf.description, wf.author, wf.run_mode,
                    wf.interval_minutes, wf.repeat_count, wf.repeat_interval_minutes,
                    wf.scheduled_at, tools_json, wf.created_at, wf.updated_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def delete_workflow(workflow_id: str) -> bool:
    with _write_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM workflows WHERE id = ?", (workflow_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ──────────────────────────────────────────────
# Execution CRUD
# ──────────────────────────────────────────────

def _row_to_execution(row: sqlite3.Row) -> WorkflowExecution:
    def _load_json(s, default):
        try:
            return json.loads(s or "") if s else default
        except Exception:
            return default

    results_raw = _load_json(row["results_json"], {})
    results: Dict[str, ExecutionResult] = {}
    for k, v in results_raw.items():
        try:
            results[k] = ExecutionResult.from_dict(v)
        except Exception:
            pass

    return WorkflowExecution(
        execution_id=row["execution_id"],
        workflow_id=row["workflow_id"],
        domain=row["domain"],
        notes=row["notes"],
        status=row["status"],
        results=results,
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        version=row["version"],
        last_updated_at=row["last_updated_at"],
        events=_load_json(row["events_json"], []),
        cancel_requested=bool(row["cancel_requested"]),
        cancelled_at=row["cancelled_at"],
        run_mode=row["run_mode"],
        interval_minutes=row["interval_minutes"],
        repeat_count=row["repeat_count"],
        repeat_interval_minutes=row["repeat_interval_minutes"],
        scheduled_at=row["scheduled_at"],
        current_iteration=row["current_iteration"],
        planned_iterations=row["planned_iterations"],
        iterations=_load_json(row["iterations_json"], []),
    )


def list_executions() -> List[WorkflowExecution]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM executions ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_execution(r) for r in rows]
    finally:
        conn.close()


def get_execution(execution_id: str) -> Optional[WorkflowExecution]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM executions WHERE execution_id = ?", (execution_id,)
        ).fetchone()
        return _row_to_execution(row) if row else None
    finally:
        conn.close()


def insert_execution(ex: WorkflowExecution) -> None:
    """Insert a brand-new execution row."""
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO executions
                    (execution_id, workflow_id, domain, notes, status, run_mode,
                     interval_minutes, repeat_count, repeat_interval_minutes,
                     scheduled_at, current_iteration, planned_iterations,
                     created_at, started_at, completed_at, cancelled_at,
                     cancel_requested, version, last_updated_at,
                     events_json, results_json, iterations_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ex.execution_id, ex.workflow_id, ex.domain, ex.notes,
                    ex.status, ex.run_mode, ex.interval_minutes, ex.repeat_count,
                    ex.repeat_interval_minutes, ex.scheduled_at,
                    ex.current_iteration, ex.planned_iterations,
                    ex.created_at, ex.started_at, ex.completed_at,
                    ex.cancelled_at, int(ex.cancel_requested), ex.version,
                    ex.last_updated_at,
                    json.dumps(ex.events),
                    json.dumps({k: v.to_dict() for k, v in ex.results.items()}),
                    json.dumps(ex.iterations),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def update_execution(ex: WorkflowExecution) -> None:
    """Persist updated state of an execution (called when it finishes)."""
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE executions SET
                    status                  = ?,
                    run_mode                = ?,
                    interval_minutes        = ?,
                    repeat_count            = ?,
                    repeat_interval_minutes = ?,
                    scheduled_at            = ?,
                    current_iteration       = ?,
                    planned_iterations      = ?,
                    started_at              = ?,
                    completed_at            = ?,
                    cancelled_at            = ?,
                    cancel_requested        = ?,
                    version                 = ?,
                    last_updated_at         = ?,
                    events_json             = ?,
                    results_json            = ?,
                    iterations_json         = ?
                WHERE execution_id = ?
                """,
                (
                    ex.status, ex.run_mode, ex.interval_minutes,
                    ex.repeat_count, ex.repeat_interval_minutes, ex.scheduled_at,
                    ex.current_iteration, ex.planned_iterations,
                    ex.started_at, ex.completed_at, ex.cancelled_at,
                    int(ex.cancel_requested), ex.version, ex.last_updated_at,
                    json.dumps(ex.events),
                    json.dumps({k: v.to_dict() for k, v in ex.results.items()}),
                    json.dumps(ex.iterations),
                    ex.execution_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def delete_execution(execution_id: str) -> bool:
    with _write_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM executions WHERE execution_id = ?", (execution_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def delete_all_executions() -> int:
    with _write_lock:
        conn = _connect()
        try:
            cur = conn.execute("DELETE FROM executions")
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def delete_executions_older_than(cutoff_iso: str) -> int:
    """Delete completed/failed/cancelled executions created before cutoff."""
    with _write_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                """
                DELETE FROM executions
                WHERE created_at < ?
                  AND status IN ('completed','failed','cancelled')
                """,
                (cutoff_iso,),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def get_executions_for_diff(
    workflow_id: str, domain: str
) -> List[WorkflowExecution]:
    """Return completed executions for the same workflow+domain, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM executions
            WHERE workflow_id = ? AND domain = ? AND status = 'completed'
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (workflow_id, domain),
        ).fetchall()
        return [_row_to_execution(r) for r in rows]
    finally:
        conn.close()


def get_due_scheduled_executions(now_iso: str) -> List[WorkflowExecution]:
    """Return scheduled executions whose time has come."""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM executions
            WHERE status = 'scheduled' AND scheduled_at <= ?
            ORDER BY scheduled_at
            """,
            (now_iso,),
        ).fetchall()
        return [_row_to_execution(r) for r in rows]
    finally:
        conn.close()


def mark_execution_queued(execution_id: str) -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE executions SET status = 'queued' WHERE execution_id = ?",
                (execution_id,),
            )
            conn.commit()
        finally:
            conn.close()


# ──────────────────────────────────────────────
# Update history
# ──────────────────────────────────────────────

def save_update_result(ur: UpdateResult) -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO update_history
                    (update_id, tool_id, tool_name, status, start_time, end_time, output, error)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(update_id) DO UPDATE SET
                    status     = excluded.status,
                    end_time   = excluded.end_time,
                    output     = excluded.output,
                    error      = excluded.error
                """,
                (
                    ur.update_id, ur.tool_id, ur.tool_name, ur.status,
                    ur.start_time, ur.end_time, ur.output, ur.error,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_update_result(update_id: str) -> Optional[UpdateResult]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM update_history WHERE update_id = ?", (update_id,)
        ).fetchone()
        if not row:
            return None
        return UpdateResult(
            update_id=row["update_id"], tool_id=row["tool_id"],
            tool_name=row["tool_name"], status=row["status"],
            start_time=row["start_time"], end_time=row["end_time"],
            output=row["output"], error=row["error"],
        )
    finally:
        conn.close()


def list_recent_updates(limit: int = 20) -> List[UpdateResult]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM update_history ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            UpdateResult(
                update_id=r["update_id"], tool_id=r["tool_id"],
                tool_name=r["tool_name"], status=r["status"],
                start_time=r["start_time"], end_time=r["end_time"],
                output=r["output"], error=r["error"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def prune_update_history(keep: int = 200) -> None:
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                DELETE FROM update_history WHERE update_id NOT IN (
                    SELECT update_id FROM update_history ORDER BY start_time DESC LIMIT ?
                )
                """,
                (keep,),
            )
            conn.commit()
        finally:
            conn.close()


# ──────────────────────────────────────────────
# Backup / restore
# ──────────────────────────────────────────────

def backup_db(dest_path: str) -> bool:
    """Hot backup using SQLite's built-in backup API."""
    try:
        with _write_lock:
            src = _connect()
            try:
                dst = sqlite3.connect(dest_path)
                try:
                    src.backup(dst)
                    return True
                finally:
                    dst.close()
            finally:
                src.close()
    except Exception:
        return False


def restore_db(src_path: str) -> bool:
    """Restore by copying a backup file over the live DB (requires restart or WAL flush)."""
    try:
        with _write_lock:
            shutil.copy2(src_path, DB_PATH)
            return True
    except Exception:
        return False
