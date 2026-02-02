from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any


def load_prompt_yaml(name: str) -> Dict[str, Any]:
    """Load YAML under agents/prompts/{name}.yaml; tolerant to missing PyYAML.
    Returns empty dict if file not found or parse fails.
    """
    path = Path(__file__).resolve().parents[1] / "agents" / "prompts" / f"{name}.yaml"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}



