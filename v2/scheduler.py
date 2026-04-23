"""
scheduler.py – Background daemon threads.

1. _cleanup_loop   – purge old execution rows and log files on a daily cadence.
2. _scheduler_loop – fire scheduled executions when their scheduled_at time arrives.
"""
import logging
import os
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _cleanup_loop() -> None:
    """Delete old executions and prune log files once per hour."""
    while True:
        try:
            _run_cleanup()
        except Exception as e:
            logger.warning(f"[cleanup] Error: {e}")
        time.sleep(3600)


def _run_cleanup() -> None:
    from settings import load_settings
    from database import delete_executions_older_than, prune_update_history

    settings = load_settings()

    # --- Execution retention ---
    retention_days = int(settings.get("execution_retention_days", 0) or 0)
    if retention_days > 0:
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        deleted = delete_executions_older_than(cutoff)
        if deleted:
            logger.info(f"[cleanup] Pruned {deleted} old execution(s)")

    # --- Log file retention ---
    log_days = int(settings.get("log_retention_days", 7) or 7)
    if log_days > 0:
        cutoff_ts = (datetime.now() - timedelta(days=log_days)).timestamp()
        log_dir = "logs"
        if os.path.isdir(log_dir):
            for fname in os.listdir(log_dir):
                fpath = os.path.join(log_dir, fname)
                try:
                    if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff_ts:
                        os.remove(fpath)
                        logger.info(f"[cleanup] Removed old log: {fpath}")
                except Exception as e:
                    logger.warning(f"[cleanup] Could not remove {fpath}: {e}")

    # Keep update_history table tidy
    prune_update_history(keep=500)


def _scheduler_loop() -> None:
    """Check every 30 seconds for scheduled executions that are due."""
    while True:
        try:
            _fire_due_scheduled()
        except Exception as e:
            logger.warning(f"[scheduler] Error: {e}")
        time.sleep(30)


def _fire_due_scheduled() -> None:
    from database import get_due_scheduled_executions, mark_execution_queued, get_workflow
    from executor import start_execution
    from models import WorkflowExecution

    now = datetime.now().isoformat()
    due = get_due_scheduled_executions(now)
    for ex in due:
        logger.info(f"[scheduler] Firing scheduled execution {ex.execution_id}")
        # Change status to queued so it isn't picked up again
        mark_execution_queued(ex.execution_id)
        ex.status = "queued"
        start_execution(ex)


def start_background_threads() -> None:
    for target, name in [(_cleanup_loop, "cleanup"), (_scheduler_loop, "scheduler")]:
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        logger.info(f"Started background thread: {name}")
