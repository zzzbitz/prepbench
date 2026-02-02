from __future__ import annotations

import json
from pathlib import Path


def render_main_with_solve(solve_source: str, allowed_outputs: list[str], *, template_name: str = "solve_template.py.j2") -> str:
    tmpl_path = Path("core/templates") / template_name
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Template not found: {tmpl_path}")
    tmpl = tmpl_path.read_text(encoding="utf-8")
    content = tmpl.replace("{{ solve_source }}", solve_source)
    content = content.replace("{{ allowed_outputs_list }}", json.dumps(allowed_outputs, ensure_ascii=False))
    return content

