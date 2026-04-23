"""
settings.py – Load / save application settings from app_settings.json.
Also provides a stable Flask secret key written once to secret_key.bin.
"""
import json
import logging
import os
import secrets

logger = logging.getLogger(__name__)

SETTINGS_FILE = "app_settings.json"
SECRET_KEY_FILE = "secret_key.bin"

DEFAULT_SETTINGS: dict = {
    "max_execution_time": 9999,
    "max_output_size": 999999,
    "auto_cleanup": True,
    "enable_logging": True,
    "concurrent_executions": 3,
    "log_retention_days": 7,
    "execution_retention_days": 0,   # 0 = never auto-delete
    "backup_frequency": "weekly",
    "output_directory": "exports",
    "temp_directory": "",
    "enable_debug_mode": False,
    "tool_library": [],
    "wordlists": {},
}


def load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            # Fill in any missing keys (forward-compat)
            for k, v in DEFAULT_SETTINGS.items():
                stored.setdefault(k, v)
            return stored
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> bool:
    try:
        valid: dict = {}
        for k in DEFAULT_SETTINGS:
            valid[k] = settings.get(k, DEFAULT_SETTINGS[k])
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(valid, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False


def get_or_create_secret_key() -> bytes:
    """Return a stable secret key, creating it once if it doesn't exist."""
    if os.path.exists(SECRET_KEY_FILE):
        try:
            with open(SECRET_KEY_FILE, "rb") as f:
                key = f.read()
            if len(key) >= 32:
                return key
        except Exception:
            pass
    key = secrets.token_bytes(48)
    try:
        with open(SECRET_KEY_FILE, "wb") as f:
            f.write(key)
    except Exception as e:
        logger.warning(f"Could not persist secret key: {e}")
    return key
