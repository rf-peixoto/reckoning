"""routes/tools.py – Tool update management routes."""
import threading
import uuid
from datetime import datetime

from flask import jsonify, render_template, request

import database as db
from executor import execute_update_thread
from models import ToolConfig, UpdateResult
from settings import load_settings


def register(app):

    @app.route("/tools/update")
    def tools_update():
        settings = load_settings()
        all_tools = []

        for entry in (settings.get("tool_library", []) or []):
            all_tools.append({
                "id": entry.get("id"),
                "name": entry.get("name", "Unnamed"),
                "command": entry.get("path", entry.get("command", "")),
                "args_template": entry.get("default_command", entry.get("args_template", "")),
                "update_command": entry.get("update_command", ""),
                "last_updated": entry.get("last_updated"),
                "description": entry.get("description", ""),
                "color": entry.get("color", "#00ff41"),
                "workflow_id": "__library__",
                "workflow_name": "Tool Library",
            })

        for wf in db.list_workflows():
            for tool in wf.tools:
                td = tool.to_dict()
                td["workflow_id"] = wf.id
                td["workflow_name"] = wf.name
                all_tools.append(td)

        recent = db.list_recent_updates(10)
        return render_template(
            "tools_update.html",
            tools=all_tools,
            recent_updates=[u.to_dict() for u in recent],
        )

    @app.route("/api/tool/update", methods=["POST"])
    def update_tool():
        data = request.get_json(silent=True) or {}
        tool_id = data.get("tool_id")
        workflow_id = data.get("workflow_id")

        if not tool_id or not workflow_id:
            return jsonify({"success": False, "error": "Missing tool_id or workflow_id"})

        settings = load_settings()

        if workflow_id == "__library__":
            entry = next(
                (t for t in (settings.get("tool_library", []) or [])
                 if t.get("id") == tool_id),
                None,
            )
            if not entry:
                return jsonify({"success": False, "error": "Tool not found in library"})
            target_tool = ToolConfig(
                tool_id=entry.get("id", str(uuid.uuid4())),
                name=entry.get("name", "Unnamed"),
                command=entry.get("path", entry.get("command", "")),
                args_template=entry.get("default_command", entry.get("args_template", "")),
                update_command=entry.get("update_command", ""),
                description=entry.get("description", ""),
            )
        else:
            wf = db.get_workflow(workflow_id)
            if not wf:
                return jsonify({"success": False, "error": "Workflow not found"})
            target_tool = next((t for t in wf.tools if t.id == tool_id), None)
            if not target_tool:
                return jsonify({"success": False, "error": "Tool not found in workflow"})

        if not target_tool.update_command:
            return jsonify({"success": False, "error": "No update command configured"})

        # Pre-insert a 'queued' record so status polling works immediately
        update_id = str(uuid.uuid4())
        ur = UpdateResult(
            update_id=update_id,
            tool_id=target_tool.id,
            tool_name=target_tool.name,
            status="queued",
            start_time=datetime.now().isoformat(),
        )
        db.save_update_result(ur)

        t = threading.Thread(
            target=execute_update_thread,
            args=(target_tool, workflow_id),
            daemon=True,
        )
        t.start()

        return jsonify({
            "success": True,
            "message": f"Update started for {target_tool.name}",
            "tool_name": target_tool.name,
            "update_id": update_id,
        })

    @app.route("/api/tools/update-all", methods=["POST"])
    def update_all_tools():
        data = request.get_json(silent=True) or {}
        workflow_id = data.get("workflow_id")
        settings = load_settings()
        tools_to_update = []

        if workflow_id == "__library__":
            for entry in (settings.get("tool_library", []) or []):
                if entry.get("update_command"):
                    tools_to_update.append((
                        ToolConfig(
                            tool_id=entry.get("id", str(uuid.uuid4())),
                            name=entry.get("name", "Unnamed"),
                            command=entry.get("path", entry.get("command", "")),
                            args_template=entry.get("default_command", ""),
                            update_command=entry.get("update_command", ""),
                            description=entry.get("description", ""),
                        ),
                        "__library__",
                    ))
        elif workflow_id:
            wf = db.get_workflow(workflow_id)
            if wf:
                for t in wf.tools:
                    if t.update_command:
                        tools_to_update.append((t, workflow_id))
        else:
            for entry in (settings.get("tool_library", []) or []):
                if entry.get("update_command"):
                    tools_to_update.append((
                        ToolConfig(
                            tool_id=entry.get("id", str(uuid.uuid4())),
                            name=entry.get("name", "Unnamed"),
                            command=entry.get("path", entry.get("command", "")),
                            args_template=entry.get("default_command", ""),
                            update_command=entry.get("update_command", ""),
                        ),
                        "__library__",
                    ))
            for wf in db.list_workflows():
                for t in wf.tools:
                    if t.update_command:
                        tools_to_update.append((t, wf.id))

        if not tools_to_update:
            return jsonify({"success": False, "error": "No tools with update commands"})

        update_ids = []
        for tool, wf_id in tools_to_update:
            ur = UpdateResult(
                update_id=str(uuid.uuid4()),
                tool_id=tool.id,
                tool_name=tool.name,
                status="queued",
                start_time=datetime.now().isoformat(),
            )
            db.save_update_result(ur)
            update_ids.append(ur.update_id)
            threading.Thread(
                target=execute_update_thread, args=(tool, wf_id), daemon=True
            ).start()

        return jsonify({
            "success": True,
            "message": f"Started {len(tools_to_update)} update(s)",
            "update_count": len(tools_to_update),
            "update_ids": update_ids,
        })

    @app.route("/api/update/status/<update_id>")
    def get_update_status(update_id):
        ur = db.get_update_result(update_id)
        if not ur:
            return jsonify({"error": "Not found"}), 404
        return jsonify(ur.to_dict())

    @app.route("/api/updates/recent")
    def get_recent_updates():
        updates = db.list_recent_updates(20)
        return jsonify([u.to_dict() for u in updates])
