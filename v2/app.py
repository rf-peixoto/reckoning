"""
app.py – Flask application entry point.

Structure:
  models.py          – data classes
  database.py        – SQLite persistence
  settings.py        – settings load/save + stable secret key
  executor.py        – tool/workflow execution engine
  scheduler.py       – background cleanup + scheduled-execution threads
  routes/
    workflows.py     – workflow CRUD
    executions.py    – execution lifecycle + SSE + diff
    tools.py         – tool update management
    settings_routes.py – settings + backup/restore
"""
import logging
import os

from flask import send_from_directory
from flask import Flask

from database import init_db
from scheduler import start_background_threads
from settings import get_or_create_secret_key

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s – %(message)s"


def _configure_logging(debug: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(logging.Formatter(_LOG_FMT))
    root.addHandler(ch)

    os.makedirs("logs", exist_ok=True)
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            "logs/app.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_LOG_FMT))
        root.addHandler(fh)
    except Exception as e:
        logging.warning(f"File logging unavailable: {e}")


# ──────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, static_folder=".")
    app.secret_key = get_or_create_secret_key()
    app.config["UPLOAD_FOLDER"] = "uploads"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    # Ensure required directories exist
    for d in ("uploads", "logs", "exports", "backups", "img"):
        os.makedirs(d, exist_ok=True)

    # Initialise SQLite schema
    init_db()

    # Register all routes
    from routes.workflows import register as reg_workflows
    from routes.executions import register as reg_executions
    from routes.tools import register as reg_tools
    from routes.settings_routes import register as reg_settings

    reg_workflows(app)
    reg_executions(app)
    reg_tools(app)
    reg_settings(app)

    # Static image serving
    @app.route("/img/<path:filename>")
    def serve_image(filename):
        return send_from_directory("img", filename)

    # Start background threads (scheduler + cleanup)
    start_background_threads()

    return app


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from settings import load_settings
    _configure_logging(debug=load_settings().get("enable_debug_mode", False))
    application = create_app()
    application.run(debug=False, host="0.0.0.0", port=5000)
