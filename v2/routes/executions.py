"""routes/executions.py – Execution lifecycle routes."""
import difflib
import hashlib
import json
import uuid
from datetime import datetime
from typing import Optional

from flask import (
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)

import database as db
from executor import (
    get_or_load_execution,
    get_stream_queue,
    request_cancel,
    start_execution,
    _execution_lock,
    _sha256_text,
)
from models import WorkflowExecution
from settings import load_settings


def _wants_json():
    if request.is_json:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    return "application/json" in (request.headers.get("Accept") or "").lower()


def _int_or_none(v):
    if v is None:
        return None
    try:
        return None if isinstance(v, str) and v.strip() == "" else int(v)
    except Exception:
        return None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.fromisoformat(ts.replace("Z", "").split("+")[0])
        except Exception:
            return None


def _human_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _enrich_execution(exec_dict: dict) -> dict:
    """Add computed fields (workflow name, duration) to an execution dict."""
    wf = db.get_workflow(exec_dict.get("workflow_id", ""))
    exec_dict["workflow_name"] = wf.name if wf else "Unknown Workflow"
    exec_dict["workflow_description"] = wf.description if wf else ""

    start_dt = _parse_iso(exec_dict.get("started_at"))
    end_dt = _parse_iso(exec_dict.get("completed_at"))
    if start_dt and end_dt:
        seconds = int((end_dt - start_dt).total_seconds())
    else:
        seconds = None
    exec_dict["duration_seconds"] = seconds
    exec_dict["duration_human"] = _human_duration(seconds)
    return exec_dict


def register(app):

    # ── List ──────────────────────────────────────────
    @app.route("/executions")
    def executions_list():
        all_execs = db.list_executions()
        result = []
        for ex in all_execs:
            d = ex.to_dict()
            d = _enrich_execution(d)
            wf = db.get_workflow(ex.workflow_id or "")
            total = len(wf.tools) if wf else 0
            completed = sum(1 for r in ex.results.values() if r.status == "completed")
            d["summary"] = {
                "total_tools": total,
                "completed_tools": completed,
                "success_rate": (completed / total * 100) if total else 0,
            }
            result.append(d)
        return render_template("executions.html", executions=result)

    # ── Execute ───────────────────────────────────────
    @app.route("/execute", methods=["POST"])
    def execute_workflow():
        data = request.get_json(silent=True) or {}
        workflow_id = data.get("workflow_id")
        notes = data.get("notes", "")

        if not workflow_id:
            return jsonify({"success": False, "error": "Workflow ID required"})

        wf = db.get_workflow(workflow_id)
        if not wf:
            return jsonify({"success": False, "error": "Workflow not found"})

        # Multi-target: split on newlines/commas
        raw_targets = data.get("domain", "")
        targets = [t.strip() for t in raw_targets.replace(",", "\n").splitlines() if t.strip()]
        if not targets:
            return jsonify({"success": False, "error": "At least one target required"})

        # Scheduled start (optional)
        scheduled_at = data.get("scheduled_at") or None

        created_ids = []
        for domain in targets:
            ex_id = str(uuid.uuid4())
            ex = WorkflowExecution(
                execution_id=ex_id,
                workflow_id=workflow_id,
                domain=domain,
                notes=notes,
                status="scheduled" if scheduled_at else "queued",
                run_mode=wf.run_mode,
                interval_minutes=getattr(wf, "interval_minutes", None),
                repeat_count=getattr(wf, "repeat_count", None),
                repeat_interval_minutes=getattr(wf, "repeat_interval_minutes", None),
                scheduled_at=scheduled_at,
                planned_iterations=(
                    wf.repeat_count if wf.run_mode == "repeat" else None
                ),
            )
            db.insert_execution(ex)
            if not scheduled_at:
                start_execution(ex)
            created_ids.append(ex_id)

        if len(created_ids) == 1:
            return jsonify({
                "success": True,
                "execution_id": created_ids[0],
                "redirect": url_for("execution_status", execution_id=created_ids[0]),
            })
        return jsonify({
            "success": True,
            "execution_ids": created_ids,
            "redirect": url_for("executions_list"),
        })

    # ── Detail view ───────────────────────────────────
    @app.route("/execution/<execution_id>")
    def execution_status(execution_id):
        ex = get_or_load_execution(execution_id)
        if not ex:
            return "Execution not found", 404

        wf = db.get_workflow(ex.workflow_id or "")
        ex_dict = ex.to_dict(include_results=False)
        ex_dict["workflow_name"] = wf.name if wf else "Unknown"
        ex_dict["workflow_description"] = wf.description if wf else ""

        tool_order = []
        if wf:
            for t in wf.tools:
                tool_order.append({
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "color": t.color,
                    "enabled": t.enabled,
                })

        return render_template(
            "execution_detail.html",
            execution=ex_dict,
            tool_order=tool_order,
            settings=load_settings(),
        )

    # ── Status API ────────────────────────────────────
    @app.route("/execution/<execution_id>/status")
    @app.route("/api/execution/<execution_id>/status")
    def get_execution_status(execution_id):
        ex = get_or_load_execution(execution_id)
        if not ex:
            return jsonify({"error": "Not found"}), 404

        wf = db.get_workflow(ex.workflow_id or "")
        base = ex.to_dict(include_results=False)
        base["workflow_name"] = wf.name if wf else "Unknown"
        base["workflow_description"] = wf.description if wf else ""

        with _execution_lock:
            results_snapshot = dict(ex.results)

        summary = {}
        for tid, res in results_snapshot.items():
            out = res.output or ""
            err = res.error or ""
            summary[tid] = {
                "tool_id": res.tool_id,
                "tool_name": res.tool_name,
                "status": res.status,
                "start_time": res.start_time,
                "end_time": res.end_time,
                "exit_code": res.exit_code,
                "output_len": len(out),
                "error_len": len(err),
                "output_sha256": _sha256_text(out) if out else "",
                "error_sha256": _sha256_text(err) if err else "",
            }

        base["results_summary"] = summary
        base["events_tail"] = (ex.events or [])[-20:]
        return jsonify(base)

    # ── Tool results API ──────────────────────────────
    @app.route("/api/execution/<execution_id>/results/<tool_id>")
    def get_execution_tool_results(execution_id, tool_id):
        ex = get_or_load_execution(execution_id)
        if not ex:
            return jsonify({"error": "Execution not found"}), 404
        with _execution_lock:
            res = (ex.results or {}).get(tool_id)
        if not res:
            return jsonify({"error": "Tool result not found"}), 404
        out = res.output or ""
        err = res.error or ""
        return jsonify({
            "execution_id": execution_id,
            "tool_id": tool_id,
            "tool_name": res.tool_name,
            "status": res.status,
            "start_time": res.start_time,
            "end_time": res.end_time,
            "exit_code": res.exit_code,
            "output": out,
            "error": err,
            "output_len": len(out),
            "error_len": len(err),
            "output_sha256": _sha256_text(out) if out else "",
            "error_sha256": _sha256_text(err) if err else "",
        })

    # ── Events API ────────────────────────────────────
    @app.route("/api/execution/<execution_id>/events")
    def get_execution_events(execution_id):
        ex = get_or_load_execution(execution_id)
        if not ex:
            return jsonify({"error": "Not found"}), 404
        with _execution_lock:
            return jsonify({
                "execution_id": execution_id,
                "version": ex.version,
                "events": ex.events or [],
            })

    # ── SSE streaming ─────────────────────────────────
    @app.route("/api/execution/<execution_id>/stream")
    def stream_execution_output(execution_id):
        import queue as _queue

        def _generate():
            q = get_stream_queue(execution_id)
            if q is None:
                # Execution may already be done; send a terminal event
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            while True:
                try:
                    item = q.get(timeout=20)
                    if item is None:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        return
                    yield f"data: {json.dumps(item)}\n\n"
                except _queue.Empty:
                    yield ": keepalive\n\n"

        return Response(
            stream_with_context(_generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Cancel ────────────────────────────────────────
    @app.route("/execution/cancel/<execution_id>", methods=["POST"])
    def cancel_execution(execution_id):
        new_status = request_cancel(execution_id)
        if new_status is None:
            if _wants_json():
                return jsonify({"success": False, "error": "Execution not found"}), 404
            flash("Execution not found.", "danger")
            return redirect(url_for("executions_list"))
        if _wants_json():
            return jsonify({"success": True, "status": new_status})
        flash("Cancel requested.", "info")
        return redirect(url_for("execution_status", execution_id=execution_id))

    # ── Delete ────────────────────────────────────────
    @app.route("/execution/delete/<execution_id>", methods=["POST"])
    def delete_execution(execution_id):
        # Cancel first if running
        request_cancel(execution_id)
        ok = db.delete_execution(execution_id)
        if _wants_json():
            return jsonify({"success": ok})
        if ok:
            flash("Execution deleted.", "success")
        else:
            flash("Execution not found.", "danger")
        return redirect(url_for("executions_list"))

    @app.route("/execution/clear_all", methods=["POST"])
    def clear_all_executions():
        count = db.delete_all_executions()
        return jsonify({"success": True, "message": f"Deleted {count} execution(s)."})

    # ── Export ────────────────────────────────────────
    @app.route("/execution/export/<execution_id>")
    def export_execution(execution_id):
        ex = get_or_load_execution(execution_id)
        if not ex:
            return "Not found", 404
        return jsonify(ex.to_dict())

    # ── Diff ─────────────────────────────────────────
    @app.route("/execution/diff")
    def execution_diff():
        id_a = request.args.get("a")
        id_b = request.args.get("b")
        if not id_a or not id_b:
            flash("Select two executions to compare.", "warning")
            return redirect(url_for("executions_list"))

        ex_a = get_or_load_execution(id_a)
        ex_b = get_or_load_execution(id_b)

        if not ex_a or not ex_b:
            flash("One or both executions not found.", "danger")
            return redirect(url_for("executions_list"))

        if ex_a.workflow_id != ex_b.workflow_id:
            flash("Diff only works for executions of the same workflow.", "warning")
            return redirect(url_for("executions_list"))

        if ex_a.domain != ex_b.domain:
            flash("Diff only works for executions against the same target.", "warning")
            return redirect(url_for("executions_list"))

        wf = db.get_workflow(ex_a.workflow_id or "")
        tool_order = wf.tools if wf else []

        diffs = {}
        for tool in tool_order:
            res_a = ex_a.results.get(tool.id)
            res_b = ex_b.results.get(tool.id)
            out_a = (res_a.output or "").splitlines(keepends=True) if res_a else []
            out_b = (res_b.output or "").splitlines(keepends=True) if res_b else []
            unified = list(difflib.unified_diff(
                out_a, out_b,
                fromfile=f"exec {id_a[:8]} – {tool.name}",
                tofile=f"exec {id_b[:8]} – {tool.name}",
                lineterm="",
            ))
            diffs[tool.id] = {
                "tool_name": tool.name,
                "tool_color": tool.color,
                "lines_a": len(out_a),
                "lines_b": len(out_b),
                "diff_lines": unified,
                "added": sum(1 for l in unified if l.startswith("+") and not l.startswith("+++")),
                "removed": sum(1 for l in unified if l.startswith("-") and not l.startswith("---")),
            }

        return render_template(
            "execution_diff.html",
            exec_a=ex_a.to_dict(include_results=False),
            exec_b=ex_b.to_dict(include_results=False),
            workflow_name=wf.name if wf else "Unknown",
            diffs=diffs,
            tool_order=[t.to_dict() for t in tool_order],
        )
