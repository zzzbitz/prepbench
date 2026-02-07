from __future__ import annotations

from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def solution_assets_root(root: Optional[Path] = None) -> Path:
    base = root or repo_root()
    return base / "simulator" / "assets" / "solutions"


def external_solution_path(case_name: str, root: Optional[Path] = None) -> Path:
    return solution_assets_root(root) / f"{case_name}.py"


def legacy_solution_path(case_dir: Path) -> Path:
    return case_dir / "solution.py"


def resolve_reference_solution_path(case_dir: Path, root: Optional[Path] = None) -> Optional[Path]:
    """Resolve the reference solution path for a case.

    Search order:
    1. simulator/assets/solutions/<case_name>.py
    2. <case_dir>/solution.py (legacy layout)
    """
    case_name = case_dir.name
    external = external_solution_path(case_name, root)
    if external.exists():
        return external

    legacy = legacy_solution_path(case_dir)
    if legacy.exists():
        return legacy
    return None


def require_reference_solution_path(case_dir: Path, root: Optional[Path] = None) -> Path:
    path = resolve_reference_solution_path(case_dir, root)
    if path is not None:
        return path
    expected_external = external_solution_path(case_dir.name, root)
    expected_legacy = legacy_solution_path(case_dir)
    raise FileNotFoundError(
        f"Reference solution not found. Checked: {expected_external} and {expected_legacy}"
    )


def read_reference_solution_text(case_dir: Path, root: Optional[Path] = None) -> str:
    path = resolve_reference_solution_path(case_dir, root)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
