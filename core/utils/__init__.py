from __future__ import annotations

"""
Utility package for common helpers used across the project.
Exports selected helpers from submodules and implements light IO/schema utilities here.
"""

from pathlib import Path
from typing import List

from .code import extract_code_from_response  # noqa: F401
from core.data_head import SchemaHead, read_schema_and_head  # noqa: F401

def list_input_files(task_dir: str | Path) -> List[Path]:
    tdir = Path(task_dir)
    return sorted((tdir / "inputs").glob("*.csv"))


def _resolve_gt_dir(task_dir: Path) -> Path:
    gt_upper = task_dir / "GT"
    gt_lower = task_dir / "gt"
    return gt_upper if gt_upper.exists() else gt_lower


def list_gt_files(task_dir: str | Path) -> List[Path]:
    tdir = Path(task_dir)
    base = _resolve_gt_dir(tdir)
    return sorted(base.glob("*.csv"))


def list_output_files_from_gt(task_dir: str | Path) -> List[str]:
    return [p.name for p in list_gt_files(task_dir)]
