from __future__ import annotations

import os
from functools import lru_cache
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


def _parse_env_lines(lines: list[str]) -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    for line in lines:
        striped = line.strip()
        if not striped or striped.startswith("#"):
            continue
        if striped.startswith("export "):
            striped = striped[len("export ") :].strip()
        if "=" not in striped:
            continue
        key, value = striped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            env_map[key] = value
    return env_map


@lru_cache(maxsize=4)
def _read_env_file_cached(path_str: str) -> Dict[str, str]:
    path = Path(path_str)
    if not path.exists():
        return {}
    return _parse_env_lines(path.read_text(encoding="utf-8").splitlines())


def read_env_file(path: Path = ENV_PATH) -> Dict[str, str]:
    """Read key-values from .env as a dictionary."""
    resolved = str(path.resolve())
    return dict(_read_env_file_cached(resolved))


def get_env_value(key: str, default: str = "", path: Path = ENV_PATH) -> str:
    """Read a value from .env only (does not fallback to process environment)."""
    value = read_env_file(path).get(key)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load .env into process environment (overwrites existing values for deterministic behavior)."""
    env_map = read_env_file(path)
    for key, value in env_map.items():
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
