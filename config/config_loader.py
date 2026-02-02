from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT_DIR / "config" / "settings.yaml"
LOCAL_SETTINGS_PATH = ROOT_DIR / "config" / "settings.local.yaml"
ENV_PATH = ROOT_DIR / ".env"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load environment variables from a .env file if not already set."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        striped = line.strip()
        if not striped or striped.startswith("#") or "=" not in striped:
            continue
        key, value = striped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Dict[str, Any]:
    """
    Load settings.yaml merged with an optional settings.local.yaml.
    Local overrides replace or extend base values.
    """
    base = _load_yaml(SETTINGS_PATH)
    local = _load_yaml(LOCAL_SETTINGS_PATH)
    if not local:
        return base
    return _deep_merge(base, local.copy())

