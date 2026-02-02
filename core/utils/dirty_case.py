from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _normalize_case_id(raw: Any) -> str | None:
    text = str(raw).strip()
    if not text or not text.isdigit():
        return None
    if len(text) < 3:
        text = text.zfill(3)
    return text


def normalize_case_name(case_name: str) -> str | None:
    name = (case_name or "").strip()
    if name.startswith("case_"):
        name = name[5:]
    case_id = _normalize_case_id(name)
    if case_id is None:
        return None
    return f"case_{case_id}"


def parse_case_arg_names(case_arg: str) -> list[str]:
    if not case_arg:
        return []

    case_arg = case_arg.strip()
    if not case_arg:
        return []

    def add_name(name: str | None, names: list[str], seen: set[str]) -> None:
        if not name:
            raise ValueError(f"Invalid --case value: {case_arg}")
        if name in seen:
            return
        names.append(name)
        seen.add(name)

    if case_arg.startswith("case_"):
        name = normalize_case_name(case_arg)
        if not name:
            raise ValueError(f"Invalid --case value: {case_arg}")
        return [name]

    if "/" in case_arg or case_arg.startswith("."):
        name = normalize_case_name(Path(case_arg).name)
        if not name:
            raise ValueError(f"Invalid --case path: {case_arg}")
        return [name]

    if "," in case_arg:
        names: list[str] = []
        seen: set[str] = set()
        tokens = [t.strip() for t in case_arg.split(",") if t.strip()]
        if not tokens:
            raise ValueError(f"Invalid --case value: {case_arg}")
        for token in tokens:
            if not token.isdigit():
                raise ValueError(f"Invalid --case token: {token}")
            add_name(normalize_case_name(token), names, seen)
        return names

    if "-" in case_arg:
        parts = case_arg.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = map(int, parts)
            if start > end:
                start, end = end, start
            names = []
            seen: set[str] = set()
            for i in range(start, end + 1):
                add_name(normalize_case_name(str(i)), names, seen)
            return names
        raise ValueError(f"Invalid --case range: {case_arg}")

    if case_arg.isdigit():
        name = normalize_case_name(case_arg)
        if not name:
            raise ValueError(f"Invalid --case value: {case_arg}")
        return [name]

    raise ValueError(f"Invalid --case value: {case_arg}")


def load_dirty_case_entries(repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "dirty_case" / "dirty_case.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dirty_case.json must be a JSON list.")
    return [e for e in data if isinstance(e, dict)]


def list_dirty_case_names(entries: Iterable[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        name = normalize_case_name(str(entry.get("id", "")))
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def find_dirty_case_entry(entries: Iterable[dict[str, Any]], case_name: str) -> dict[str, Any] | None:
    target = normalize_case_name(case_name)
    if not target:
        return None
    for entry in entries:
        name = normalize_case_name(str(entry.get("id", "")))
        if name == target:
            return entry
    return None


def build_cleanspec_appendix(entry: dict[str, Any]) -> str:
    issue_type = str(entry.get("issue_type") or "").strip()
    code_blocks = entry.get("code_blocks") or []

    lines: list[str] = [
        "## CleanSpec Cleaning Hints",
        "Use the following cleaning operations in addition to the full specification.",
        "",
    ]

    if issue_type:
        lines.append(f"Issue type: {issue_type}")
        lines.append("")

    item_index = 0
    for block in code_blocks:
        if not isinstance(block, dict):
            continue
        nl_explain = str(block.get("nl_explain") or "").strip()
        code = str(block.get("code") or "").strip()
        if not nl_explain and not code:
            continue
        item_index += 1
        if nl_explain:
            lines.append(f"{item_index}. {nl_explain}")
        else:
            lines.append(f"{item_index}.")
        if code:
            lines.append("```python")
            lines.append(code)
            lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_cleanspec_appendix_for_case(entries: Iterable[dict[str, Any]], case_name: str) -> str:
    entry = find_dirty_case_entry(entries, case_name)
    if entry is None:
        return ""
    return build_cleanspec_appendix(entry)
