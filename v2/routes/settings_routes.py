"""routes/settings_routes.py – Application settings and backup/restore."""
import json
import os
from datetime import datetime

from flask import flash, jsonify, redirect, render_template, request, send_file, url_for

import database as db
from models import ToolConfig, Workflow, WorkflowExecution, ExecutionResult
from settings import DEFAULT_SETTINGS, load_settings, save_settings


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


def register(app):

    @app.route("/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "POST":
            return _save_settings_handler()
        current = load_settings()
        return render_template("settings.html", settings=current)

    def _save_settings_handler():
        data = request.get_json(silent=True) or {}
        if not data:
            # Form submission
            data = request.form.to_dict()

        current = load_settings()

        # Scalar settings
        for key in [
            "max_execution_time", "max_output_size", "log_retention_days",
            "execution_retention_days", "concurrent_executions", "output_directory",
            "temp_directory", "backup_frequency",
        ]:
            if key in data:
                current[key] = data[key]

        # Booleans
        for key in ["enable_logging", "auto_cleanup", "enable_debug_mode"]:
            if key in data:
                v = data[key]
                current[key] = v if isinstance(v, bool) else (str(v).lower() in ("true", "1", "on"))

        # tool_library and wordlists passed as JSON
        if "tool_library" in data:
            tl = data["tool_library"]
            current["tool_library"] = tl if isinstance(tl, list) else json.loads(tl)
        if "wordlists" in data:
            wl = data["wordlists"]
            current["wordlists"] = wl if isinstance(wl, dict) else json.loads(wl)

        ok = save_settings(current)
        if _wants_json():
            return jsonify({"success": ok})
        flash("Settings saved." if ok else "Failed to save settings.", "success" if ok else "danger")
        return redirect(url_for("settings"))

    @app.route("/settings/backup", methods=["GET", "POST"])
    def backup_settings():
        if request.method == "POST":
            return _create_backup()
        # Check for a pending download
        pending = request.args.get("download")
        return render_template("settings.html",
                               settings=load_settings(),
                               backup_file=pending)

    def _create_backup():
        os.makedirs("backups", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reckoning_backup_{ts}.db"
        dest = os.path.join("backups", filename)
        ok = db.backup_db(dest)
        if not ok:
            if _wants_json():
                return jsonify({"success": False, "error": "Backup failed"})
            flash("Backup failed.", "danger")
            return redirect(url_for("settings"))

        if _wants_json():
            return jsonify({"success": True, "filename": filename})
        return send_file(dest, as_attachment=True, download_name=filename)

    @app.route("/settings/restore", methods=["POST"])
    def restore_settings():
        if "file" not in request.files:
            if _wants_json():
                return jsonify({"success": False, "error": "No file uploaded"})
            flash("No file uploaded.", "danger")
            return redirect(url_for("settings"))

        file = request.files["file"]
        if not file.filename:
            if _wants_json():
                return jsonify({"success": False, "error": "No file selected"})
            flash("No file selected.", "danger")
            return redirect(url_for("settings"))

        # Save to a temp location then restore
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
        try:
            import os as _os
            _os.close(tmp_fd)
            file.save(tmp_path)
            ok = db.restore_db(tmp_path)
        finally:
            try:
                import os as _os
                _os.unlink(tmp_path)
            except Exception:
                pass

        if not ok:
            if _wants_json():
                return jsonify({"success": False, "error": "Restore failed"})
            flash("Restore failed – check that the uploaded file is a valid .db backup.", "danger")
            return redirect(url_for("settings"))

        if _wants_json():
            return jsonify({"success": True})
        flash("Backup restored. Restart the app for changes to take full effect.", "success")
        return redirect(url_for("index"))

    @app.route("/api/settings", methods=["GET"])
    def get_settings_api():
        return jsonify(load_settings())
