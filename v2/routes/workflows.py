"""routes/workflows.py – Workflow CRUD."""
import json
import uuid
from datetime import datetime

from flask import flash, jsonify, redirect, render_template, request, url_for

from database import delete_workflow, get_workflow, list_workflows, save_workflow
from models import ToolConfig, Workflow
from settings import load_settings


def _int_or_none(v):
    if v is None:
        return None
    try:
        return None if isinstance(v, str) and v.strip() == "" else int(v)
    except Exception:
        return None


def _wants_json():
    if request.is_json:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    return "application/json" in (request.headers.get("Accept") or "").lower()


def register(app):

    @app.route("/")
    def index():
        workflows = list_workflows()
        return render_template("index.html", workflows=[w.to_dict() for w in workflows])

    @app.route("/workflow/new")
    @app.route("/workflow/edit")
    def create_workflow():
        if request.method == "GET" and request.args.get("id"):
            return redirect(url_for("edit_workflow", workflow_id=request.args["id"]))
        app_settings = load_settings()
        return render_template(
            "workflow_editor.html",
            workflow=None,
            tool_library=app_settings.get("tool_library", []),
            wordlists=app_settings.get("wordlists", {}),
        )

    @app.route("/workflow/<workflow_id>/edit")
    def edit_workflow(workflow_id):
        wf = get_workflow(workflow_id)
        app_settings = load_settings()
        return render_template(
            "workflow_editor.html",
            workflow=wf.to_dict() if wf else None,
            tool_library=app_settings.get("tool_library", []),
            wordlists=app_settings.get("wordlists", {}),
        )

    @app.route("/workflow/save", methods=["POST"])
    def save_workflow_route():
        data = request.get_json(silent=True) or {}
        workflow_id = data.get("id") or str(uuid.uuid4())

        existing = get_workflow(workflow_id)
        created_at = existing.created_at if existing else datetime.now().isoformat()

        tools = [ToolConfig.from_dict(t) for t in data.get("tools", [])]
        workflow = Workflow(
            workflow_id=workflow_id,
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            tools=tools,
            created_at=created_at,
            updated_at=datetime.now().isoformat(),
            author=data.get("author", "anonymous"),
            run_mode=data.get("run_mode", "once"),
            interval_minutes=_int_or_none(data.get("interval_minutes")),
            repeat_count=_int_or_none(data.get("repeat_count")),
            repeat_interval_minutes=_int_or_none(data.get("repeat_interval_minutes")),
            scheduled_at=data.get("scheduled_at") or None,
        )
        save_workflow(workflow)
        return jsonify({"success": True, "workflow_id": workflow_id})

    @app.route("/workflow/delete/<workflow_id>", methods=["POST"])
    def delete_workflow_route(workflow_id):
        ok = delete_workflow(workflow_id)
        if _wants_json():
            return jsonify({"success": ok})
        if ok:
            flash("Workflow deleted.", "success")
        else:
            flash("Workflow not found.", "danger")
        return redirect(url_for("index"))

    @app.route("/workflow/<workflow_id>/export")
    def export_workflow(workflow_id):
        wf = get_workflow(workflow_id)
        if not wf:
            return "Workflow not found", 404
        return jsonify(wf.to_dict())

    @app.route("/workflow/import", methods=["POST"])
    def import_workflow():
        if "file" not in request.files:
            if _wants_json():
                return jsonify({"success": False, "error": "No file uploaded"})
            flash("No file uploaded.", "danger")
            return redirect(url_for("create_workflow"))

        file = request.files["file"]
        if not file.filename:
            if _wants_json():
                return jsonify({"success": False, "error": "No file selected"})
            flash("No file selected.", "danger")
            return redirect(url_for("create_workflow"))

        try:
            data = json.load(file)
            tools = [ToolConfig.from_dict(t) for t in data.get("tools", [])]
            wf = Workflow(
                workflow_id=str(uuid.uuid4()),
                name=data.get("name", "Imported Workflow"),
                description=data.get("description", ""),
                tools=tools,
                run_mode=data.get("run_mode", "once"),
                interval_minutes=_int_or_none(data.get("interval_minutes")),
                repeat_count=_int_or_none(data.get("repeat_count")),
                repeat_interval_minutes=_int_or_none(data.get("repeat_interval_minutes")),
                scheduled_at=data.get("scheduled_at"),
            )
            save_workflow(wf)
            if _wants_json():
                return jsonify({"success": True, "workflow": wf.to_dict()})
            flash("Workflow imported.", "success")
            return redirect(url_for("edit_workflow", workflow_id=wf.id))
        except Exception as e:
            if _wants_json():
                return jsonify({"success": False, "error": str(e)})
            flash(f"Import failed: {e}", "danger")
            return redirect(url_for("create_workflow"))
