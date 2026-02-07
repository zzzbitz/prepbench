from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Sequence


class PromptConfigError(ValueError):
    """Raised when a prompt config file is missing or malformed."""


def _validate_required_keys(name: str, data: Dict[str, Any], required_keys: Sequence[str]) -> None:
    missing: list[str] = []
    for key in required_keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    if missing:
        raise PromptConfigError(
            f"Prompt '{name}' is missing required key(s): {missing}"
        )


def load_prompt_yaml(name: str, *, required_keys: Sequence[str] | None = None) -> Dict[str, Any]:
    """Load YAML under agents/prompts/{name}.yaml and optionally validate required keys."""
    path = Path(__file__).resolve().parents[1] / "agents" / "prompts" / f"{name}.yaml"
    if not path.exists():
        raise PromptConfigError(f"Prompt config not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PromptConfigError(f"Failed to parse prompt YAML '{name}': {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise PromptConfigError(f"Prompt '{name}' must be a YAML mapping at top level.")

    if required_keys:
        _validate_required_keys(name, data, required_keys)
    return data


