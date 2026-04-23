"""
executor.py – Tool and workflow execution engine.

Key design decisions:
- Active executions live in memory (_active_executions) while running.
- On completion the execution is persisted to SQLite via database.update_execution().
- Streaming output is pushed to per-execution queues (_stream_queues) so the SSE
  endpoint can forward lines to the browser in real-time.
- cancel_events and active_processes are in-memory only (they are OS-level handles).
"""
import hashlib
import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ExecutionResult,
    ToolConfig,
    UpdateResult,
    WorkflowExecution,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# In-memory state (transient)
# ──────────────────────────────────────────────
_active_executions: Dict[str, WorkflowExecution] = {}
_cancel_events: Dict[str, threading.Event] = {}
_active_processes: Dict[str, subprocess.Popen] = {}
_stream_queues: Dict[str, "queue.Queue[Optional[Dict]]"] = {}
_execution_lock = threading.RLock()


# ──────────────────────────────────────────────
# Public helpers for route handlers
# ──────────────────────────────────────────────

def get_active_execution(execution_id: str) -> Optional[WorkflowExecution]:
    with _execution_lock:
        return _active_executions.get(execution_id)


def get_or_load_execution(execution_id: str) -> Optional[WorkflowExecution]:
    """Return live execution if running, otherwise load from DB."""
    ex = get_active_execution(execution_id)
    if ex:
        return ex
    from database import get_execution
    return get_execution(execution_id)


def get_stream_queue(execution_id: str) -> Optional["queue.Queue[Optional[Dict]]"]:
    with _execution_lock:
        return _stream_queues.get(execution_id)


def request_cancel(execution_id: str) -> Optional[str]:
    """Signal cancellation; returns new status string or None if not found."""
    with _execution_lock:
        ex = _active_executions.get(execution_id)
        evt = _cancel_events.get(execution_id)
        if evt is None:
            evt = threading.Event()
            _cancel_events[execution_id] = evt
        evt.set()
        proc = _active_processes.get(execution_id)

    if proc is not None:
        _terminate_process_tree(proc)

    if ex:
        with _execution_lock:
            ex.cancel_requested = True
            if ex.status in ("queued", "running", "scheduled"):
                ex.status = "cancelling"
            ex.last_updated_at = _now_iso()
        _record_event(ex, "cancel_requested", "Cancel requested by user")
        return ex.status
    return None


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now().isoformat()


def _sha256_text(value: str) -> str:
    try:
        return hashlib.sha256((value or "").encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return ""


def _int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip() == "":
            return None
        return int(v)
    except Exception:
        return None


def _record_event(execution: WorkflowExecution, event_type: str, message: str,
                  level: str = "info", tool_id: Optional[str] = None) -> None:
    evt = {
        "ts": _now_iso(),
        "type": event_type,
        "level": level,
        "tool_id": tool_id,
        "message": message,
    }
    with _execution_lock:
        execution.events.append(evt)
        if len(execution.events) > 500:
            execution.events = execution.events[-500:]
        execution.version += 1
        execution.last_updated_at = _now_iso()


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
    except Exception:
        pass


def _push_stream_line(execution_id: str, tool_id: str, line: str, kind: str = "stdout") -> None:
    """Non-blocking push to the SSE queue; drops silently when full."""
    with _execution_lock:
        q = _stream_queues.get(execution_id)
    if q is None:
        return
    try:
        q.put_nowait({"tool_id": tool_id, "line": line, "kind": kind, "ts": _now_iso()})
    except queue.Full:
        pass


def _push_stream_sentinel(execution_id: str) -> None:
    with _execution_lock:
        q = _stream_queues.get(execution_id)
    if q:
        try:
            q.put_nowait(None)
        except queue.Full:
            pass


# ──────────────────────────────────────────────
# Command execution
# ──────────────────────────────────────────────

def _run_command(
    execution_id: str,
    tool_id: str,
    cmd: str,
    stdin_data: Optional[str],
    timeout_seconds: int,
    cancel_event: Optional[threading.Event],
) -> Tuple[int, str, str]:
    """
    Run *cmd* with real-time stdout streaming and cancellation support.
    Returns (returncode, full_stdout, full_stderr).
    Raises subprocess.TimeoutExpired on timeout.
    """
    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdin": subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        try:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
        except Exception:
            pass
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(cmd, **popen_kwargs)
    with _execution_lock:
        _active_processes[execution_id] = proc

    # Feed stdin without blocking the main reader
    if stdin_data is not None:
        def _feed_stdin():
            try:
                proc.stdin.write(stdin_data)
                proc.stdin.close()
            except Exception:
                pass
        threading.Thread(target=_feed_stdin, daemon=True).start()
    else:
        try:
            proc.stdin.close()
        except Exception:
            pass

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    start = time.monotonic()
    timed_out = False
    cancelled = False

    # Read stderr asynchronously so it doesn't block stdout reading
    def _read_stderr():
        try:
            for line in proc.stderr:
                stderr_chunks.append(line)
                _push_stream_line(execution_id, tool_id, line.rstrip("\n"), "stderr")
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    # Read stdout line-by-line (streaming)
    try:
        for line in proc.stdout:
            stdout_chunks.append(line)
            _push_stream_line(execution_id, tool_id, line.rstrip("\n"), "stdout")
            if cancel_event and cancel_event.is_set():
                cancelled = True
                _terminate_process_tree(proc)
                break
            if timeout_seconds and (time.monotonic() - start) > timeout_seconds:
                timed_out = True
                _terminate_process_tree(proc)
                break
    except Exception:
        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

    proc.wait(timeout=5)
    stderr_thread.join(timeout=5)

    with _execution_lock:
        if _active_processes.get(execution_id) is proc:
            _active_processes.pop(execution_id, None)

    if timed_out:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

    return proc.returncode, "".join(stdout_chunks), "".join(stderr_chunks)


# ──────────────────────────────────────────────
# Library / template helpers
# ──────────────────────────────────────────────

def resolve_tool_from_library(tool: ToolConfig, settings: dict) -> Dict[str, str]:
    """Merge library base with per-step overrides."""
    library = settings.get("tool_library", []) or []
    lib_entry = None
    if tool.library_tool_id:
        for t in library:
            if t.get("id") == tool.library_tool_id:
                lib_entry = t
                break

    def pick(override, fallback):
        if override is not None and str(override).strip():
            return str(override)
        return str(fallback) if fallback else ""

    if lib_entry:
        return {
            "command": pick(tool.command_override, lib_entry.get("path", lib_entry.get("command", ""))),
            "args_template": pick(tool.args_template_override, lib_entry.get("default_command", lib_entry.get("args_template", ""))),
            "update_command": pick(tool.update_command_override, lib_entry.get("update_command", "")),
            "description": pick(tool.description_override, lib_entry.get("description", "")),
        }
    return {
        "command": pick(tool.command_override, tool.command),
        "args_template": pick(tool.args_template_override, tool.args_template),
        "update_command": pick(tool.update_command_override, tool.update_command),
        "description": pick(tool.description_override, tool.description),
    }


def parse_args_template(template: str, context: Dict[str, Any]) -> str:
    for key, value in context.items():
        placeholder = f"{{{key}}}"
        if placeholder in template:
            val = str(value)
            if os.path.exists(val) and " " in val:
                val = f'"{val}"'
            template = template.replace(placeholder, val)
    return template


def find_tool_by_id(workflow, tool_id: str) -> Optional[ToolConfig]:
    tool_map: Dict[str, ToolConfig] = {t.id: t for t in workflow.tools}
    return tool_map.get(tool_id)


def find_tool_id_by_step(workflow, step_number: int) -> Optional[str]:
    if 1 <= step_number <= len(workflow.tools):
        return workflow.tools[step_number - 1].id
    return None


def _standardize_tool_output(output: str, temp_dir: str, tool_id: str) -> Dict[str, Any]:
    raw = output or ""
    trimmed = raw.rstrip("\r\n")
    lines = trimmed.splitlines()
    if len(lines) <= 1:
        return {"value": lines[0] if lines else "", "as_file": False,
                "file_path": None, "raw": raw, "line_count": len(lines)}
    fd, temp_path = tempfile.mkstemp(dir=temp_dir, prefix=f"out_{tool_id}_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(raw)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    return {"value": temp_path, "as_file": True, "file_path": temp_path,
            "raw": raw, "line_count": len(lines)}


# ──────────────────────────────────────────────
# Tool execution
# ──────────────────────────────────────────────

def execute_tool(
    tool: ToolConfig,
    domain: str,
    previous_outputs: Dict[str, Dict[str, Any]],
    temp_dir: str,
    execution_context: Dict[str, Any],
    cancel_event: Optional[threading.Event] = None,
    execution_id: Optional[str] = None,
) -> ExecutionResult:
    from settings import load_settings
    settings = load_settings()
    resolved = resolve_tool_from_library(tool, settings)

    result = ExecutionResult(
        tool_id=tool.id,
        tool_name=tool.name,
        status="running",
        start_time=_now_iso(),
    )

    try:
        context: Dict[str, Any] = {
            "domain": domain,
            "0": domain,
            "temp_dir": temp_dir,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "tool_id": tool.id,
        }

        # Inject previous outputs
        for idx, (tid, output_data) in enumerate(previous_outputs.items()):
            prev_tool = find_tool_by_id(execution_context["workflow"], tid)
            if prev_tool:
                placeholder = prev_tool.placeholder_name
                if placeholder in context and context.get(placeholder) != output_data["output"]:
                    context[f"{placeholder}_{idx}"] = context[placeholder]
                context[placeholder] = output_data["output"]
                context[str(idx + 1)] = output_data["output"]
                context[f"out_{tid}"] = output_data["output"]
                context[f"step_{idx + 1}"] = output_data["output"]

        # Resolve input content
        input_content: Optional[str] = None
        if tool.input_source != "none" and tool.input_method != "none":
            if tool.input_source == "specific":
                step = None
                try:
                    step = int(tool.specific_step) if tool.specific_step is not None else None
                except Exception:
                    pass
                if step:
                    source_id = find_tool_id_by_step(execution_context["workflow"], step)
                    if source_id and source_id in previous_outputs:
                        input_content = previous_outputs[source_id]["output"]
                if input_content is None:
                    if previous_outputs:
                        input_content = list(previous_outputs.values())[-1]["output"]
                    else:
                        input_content = domain
            elif tool.input_source == "initial":
                input_content = domain
            else:  # 'previous'
                if previous_outputs:
                    input_content = list(previous_outputs.values())[-1]["output"]
                else:
                    input_content = domain

        # Prepare input based on method
        stdin_data: Optional[str] = None
        temp_files_to_cleanup: List[str] = []

        if input_content and tool.input_method != "none":
            if tool.input_method == "file":
                fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".txt")
                temp_files_to_cleanup.append(temp_path)
                with os.fdopen(fd, "w") as f:
                    f.write(input_content)
                context[tool.placeholder_name] = temp_path
                context["input_file"] = temp_path
            elif tool.input_method == "stdin":
                stdin_data = input_content
                context[tool.placeholder_name] = "-"
            elif tool.input_method == "argument":
                if "\n" in input_content and len(input_content.strip().split("\n")) > 1:
                    args_t = resolved["args_template"]
                    if any(flag in args_t for flag in ["-l", "-i", "--input", "-f", "@"]):
                        fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix=".txt")
                        temp_files_to_cleanup.append(temp_path)
                        with os.fdopen(fd, "w") as f:
                            f.write(input_content)
                        context[tool.placeholder_name] = temp_path
                        for k in list(context.keys()):
                            if k.isdigit() and context[k] == input_content:
                                context[k] = temp_path
                    else:
                        input_content = " ".join(input_content.strip().split("\n"))
                        context[tool.placeholder_name] = input_content
                else:
                    context[tool.placeholder_name] = input_content

        # Output file
        if tool.output_handling == "file":
            output_file = tool.output_file_path or os.path.join(
                temp_dir, f"output_{tool.id}_{context['timestamp']}.txt"
            )
            context["output_file"] = output_file

        # Wordlists
        for wl_name, wl_path in (settings.get("wordlists", {}) or {}).items():
            context[wl_name] = wl_path

        args = parse_args_template(resolved["args_template"], context)
        cmd = f"{resolved['command']} {args}".strip()
        logger.info(f"[{tool.name}] CMD: {cmd}")

        # Determine timeout: per-step override > global setting
        global_timeout = int(settings.get("max_execution_time", 300))
        timeout = int(tool.timeout_override) if tool.timeout_override else global_timeout

        exec_id = execution_id or execution_context.get("execution_id") or ""
        rc, stdout, stderr = _run_command(
            execution_id=exec_id,
            tool_id=tool.id,
            cmd=cmd,
            stdin_data=stdin_data,
            timeout_seconds=timeout,
            cancel_event=cancel_event,
        )

        result.exit_code = rc

        # Read output
        if tool.output_handling == "file" and "output_file" in context and os.path.exists(context["output_file"]):
            with open(context["output_file"], "r") as f:
                result.output = f.read()
        else:
            result.output = stdout

        # Truncate to max_output_size
        max_out = int(settings.get("max_output_size", 999999))
        if len(result.output) > max_out:
            result.output = result.output[:max_out] + "\n[... output truncated ...]"

        result.error = stderr
        if cancel_event and cancel_event.is_set():
            result.status = "cancelled"
        else:
            result.status = "completed" if rc == 0 else "failed"

        if rc != 0 and result.status != "cancelled":
            logger.warning(f"[{tool.name}] exit={rc} stderr={stderr[:200]}")

    except subprocess.TimeoutExpired as e:
        result.status = "failed"
        result.error = f"Timed out after {e.timeout}s"
        logger.error(f"[{tool.name}] TIMEOUT: {e}")
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        logger.exception(f"[{tool.name}] EXCEPTION")
    finally:
        result.end_time = _now_iso()

    return result


# ──────────────────────────────────────────────
# Update execution
# ──────────────────────────────────────────────

def execute_update(tool: ToolConfig, workflow_id: str) -> UpdateResult:
    """Run a tool's update command and return the result."""
    from settings import load_settings
    settings = load_settings()
    resolved = resolve_tool_from_library(tool, settings)
    result = UpdateResult(tool_id=tool.id, tool_name=tool.name,
                          status="running", start_time=_now_iso())
    try:
        cmd = f"{resolved['command']} {resolved['update_command']}".strip()
        timeout = int(settings.get("max_execution_time", 300))
        process = subprocess.run(cmd, shell=True, capture_output=True,
                                 text=True, timeout=timeout)
        result.output = process.stdout
        result.error = process.stderr
        result.status = "completed" if process.returncode == 0 else "failed"
        if result.status == "completed":
            tool.last_updated = _now_iso()
            # Persist last_updated back
            if workflow_id == "__library__":
                settings = load_settings()
                for entry in (settings.get("tool_library", []) or []):
                    if entry.get("id") == tool.id:
                        entry["last_updated"] = tool.last_updated
                        break
                from settings import save_settings
                save_settings(settings)
            else:
                from database import get_workflow, save_workflow
                wf = get_workflow(workflow_id)
                if wf:
                    for wf_tool in wf.tools:
                        if wf_tool.id == tool.id:
                            wf_tool.last_updated = tool.last_updated
                    wf.updated_at = _now_iso()
                    save_workflow(wf)
    except subprocess.TimeoutExpired as e:
        result.status = "failed"
        result.error = f"Update timed out after {e.timeout}s"
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
    finally:
        result.end_time = _now_iso()
    return result


def execute_update_thread(tool: ToolConfig, workflow_id: str) -> None:
    from database import save_update_result
    result = execute_update(tool, workflow_id)
    save_update_result(result)


# ──────────────────────────────────────────────
# Workflow execution thread
# ──────────────────────────────────────────────

def run_workflow_thread(execution: WorkflowExecution) -> None:
    from database import get_workflow, update_execution
    from settings import load_settings

    app_settings = load_settings()
    base_temp = (app_settings.get("temp_directory") or "").strip()
    if base_temp:
        os.makedirs(base_temp, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="workflow_", dir=base_temp)
    else:
        temp_dir = tempfile.mkdtemp(prefix="workflow_")

    with _execution_lock:
        cancel_event = _cancel_events.get(execution.execution_id)

    try:
        workflow = get_workflow(execution.workflow_id)
        if not workflow:
            with _execution_lock:
                execution.status = "failed"
            return

        with _execution_lock:
            execution.status = "running"
            execution.started_at = _now_iso()
        _record_event(execution, "execution_started",
                      f"Started for target: {execution.domain}")

        execution_context = {
            "workflow": workflow,
            "execution_id": execution.execution_id,
            "temp_dir": temp_dir,
        }

        # Validate and normalise recurrence config
        mode = (getattr(workflow, "run_mode", "once") or "once").strip().lower()
        interval_minutes = _int_or_none(getattr(workflow, "interval_minutes", None))
        repeat_count = _int_or_none(getattr(workflow, "repeat_count", None))
        repeat_interval_minutes = _int_or_none(getattr(workflow, "repeat_interval_minutes", None))

        if mode not in ("once", "interval", "repeat"):
            mode = "once"
        if mode == "interval" and (not interval_minutes or interval_minutes <= 0):
            mode = "once"
        if mode == "repeat":
            if not repeat_count or repeat_count <= 0:
                mode = "once"
            if not repeat_interval_minutes or repeat_interval_minutes < 0:
                repeat_interval_minutes = 0

        planned_iterations = repeat_count if mode == "repeat" else None

        with _execution_lock:
            execution.run_mode = mode
            execution.interval_minutes = interval_minutes
            execution.repeat_count = repeat_count
            execution.repeat_interval_minutes = repeat_interval_minutes
            execution.planned_iterations = planned_iterations
            execution.current_iteration = 0

        def _sleep_cancellable(seconds: int) -> bool:
            if seconds <= 0:
                return True
            if cancel_event is None:
                time.sleep(seconds)
                return True
            return not cancel_event.wait(timeout=seconds)

        iteration = 0
        while True:
            if cancel_event and cancel_event.is_set():
                with _execution_lock:
                    execution.status = "cancelled"
                    execution.cancel_requested = True
                    execution.cancelled_at = _now_iso()
                _record_event(execution, "execution_cancelled",
                              "Cancelled by user before iteration")
                break

            iteration += 1
            with _execution_lock:
                execution.current_iteration = iteration
                execution.last_updated_at = _now_iso()

            iter_started = _now_iso()
            _record_event(execution, "iteration_started",
                          f"Iteration {iteration} started")

            previous_outputs: Dict[str, Dict[str, Any]] = {}
            iter_results_snapshot: Dict = {}

            for idx, tool in enumerate(workflow.tools):
                if cancel_event and cancel_event.is_set():
                    with _execution_lock:
                        execution.status = "cancelled"
                        execution.cancel_requested = True
                        execution.cancelled_at = _now_iso()
                    _record_event(execution, "execution_cancelled",
                                  "Cancelled between tools")
                    break

                if not tool.enabled:
                    _record_event(execution, "tool_skipped",
                                  f"Skipped (disabled): {tool.name}", tool_id=tool.id)
                    continue

                _record_event(execution, "tool_started",
                              f"Starting: {tool.name}", tool_id=tool.id)
                res = execute_tool(
                    tool, execution.domain, previous_outputs,
                    temp_dir, execution_context,
                    cancel_event=cancel_event,
                    execution_id=execution.execution_id,
                )

                with _execution_lock:
                    execution.results[tool.id] = res

                iter_results_snapshot[tool.id] = res.to_dict()
                _record_event(
                    execution, "tool_finished",
                    f"Finished: {tool.name} ({res.status})",
                    level="error" if res.status == "failed" else "info",
                    tool_id=tool.id,
                )

                if res.status == "completed" and tool.provides_output:
                    std = _standardize_tool_output(res.output, temp_dir, tool.id)
                    previous_outputs[tool.id] = {
                        "output": std["value"],
                        "raw_output": std["raw"],
                        "output_is_file": std["as_file"],
                        "output_file_path": std["file_path"],
                        "tool_name": tool.name,
                        "output_format": tool.output_format,
                    }

                if res.status == "cancelled":
                    with _execution_lock:
                        execution.status = "cancelled"
                        execution.cancel_requested = True
                        execution.cancelled_at = _now_iso()
                    _record_event(execution, "execution_cancelled",
                                  "Cancelled during tool execution")
                    break

            # Snapshot iteration
            iter_completed = _now_iso()
            iter_status = "cancelled" if (
                cancel_event and cancel_event.is_set()
            ) else "completed"
            with _execution_lock:
                execution.iterations.append({
                    "iteration": iteration,
                    "started_at": iter_started,
                    "completed_at": iter_completed,
                    "status": iter_status,
                    "results": iter_results_snapshot,
                })
                if len(execution.iterations) > 25:
                    execution.iterations = execution.iterations[-25:]
                execution.last_updated_at = _now_iso()

            _record_event(execution, "iteration_completed",
                          f"Iteration {iteration} completed")

            if mode == "once":
                break
            if mode == "repeat" and planned_iterations and iteration >= planned_iterations:
                break

            if mode == "interval":
                _record_event(execution, "iteration_sleep",
                              f"Sleeping {interval_minutes}m before next run")
                if not _sleep_cancellable(int(interval_minutes * 60)):
                    continue
            elif mode == "repeat":
                _record_event(execution, "iteration_sleep",
                              f"Sleeping {repeat_interval_minutes or 0}m before next run")
                if not _sleep_cancellable(int((repeat_interval_minutes or 0) * 60)):
                    continue

        with _execution_lock:
            if execution.status == "running":
                execution.status = "completed"
                _record_event(execution, "execution_completed",
                              "Completed successfully")

    except Exception as e:
        with _execution_lock:
            execution.status = "failed"
        _record_event(execution, "execution_failed",
                      f"Unhandled error: {e}", level="error")
        logger.exception("Workflow execution failed")
    finally:
        with _execution_lock:
            execution.completed_at = _now_iso()
            _active_processes.pop(execution.execution_id, None)

        # Signal SSE clients that the stream is done
        _push_stream_sentinel(execution.execution_id)

        # Persist final state to DB
        try:
            update_execution(execution)
        except Exception as e:
            logger.error(f"Failed to persist execution {execution.execution_id}: {e}")

        # Move out of active memory
        with _execution_lock:
            _active_executions.pop(execution.execution_id, None)
            _cancel_events.pop(execution.execution_id, None)
            _stream_queues.pop(execution.execution_id, None)

        # Cleanup temp files
        try:
            from settings import load_settings
            if load_settings().get("auto_cleanup", True):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def start_execution(execution: WorkflowExecution) -> None:
    """Register in memory and launch background thread."""
    with _execution_lock:
        _active_executions[execution.execution_id] = execution
        _cancel_events[execution.execution_id] = threading.Event()
        _stream_queues[execution.execution_id] = queue.Queue(maxsize=2000)

    t = threading.Thread(
        target=run_workflow_thread, args=(execution,), daemon=True,
        name=f"exec-{execution.execution_id[:8]}"
    )
    t.start()
