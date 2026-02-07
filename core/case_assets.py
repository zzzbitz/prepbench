from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def solution_assets_root(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "simulator" / "assets" / "solutions"


def external_solution_path(case_name: str, root: Path | None = None) -> Path:
    return solution_assets_root(root) / f"{case_name}.py"


def resolve_reference_solution_path(case_dir: Path, root: Path | None = None) -> Path | None:
    """Resolve the reference solution path for a case."""
    case_name = case_dir.name
    external = external_solution_path(case_name, root)
    if external.exists():
        return external
    return None


def require_reference_solution_path(case_dir: Path, root: Path | None = None) -> Path:
    path = resolve_reference_solution_path(case_dir, root)
    if path is not None:
        return path
    expected_external = external_solution_path(case_dir.name, root)
    raise FileNotFoundError(f"Reference solution not found: {expected_external}")


def read_reference_solution_text(case_dir: Path, root: Path | None = None) -> str:
    path = resolve_reference_solution_path(case_dir, root)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
