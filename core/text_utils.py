from __future__ import annotations

from typing import Optional

from core.utils.code import extract_single_code_block as _extract_single_code_block
from core.utils.code import extract_solve_from_raw as _extract_solve_from_raw


def extract_single_code_block(raw: Optional[str]) -> str:
    """Backward-compatible wrapper for code block extraction."""
    return _extract_single_code_block(raw)


def extract_single_solution_from_raw(raw: Optional[str]) -> str:
    """Backward-compatible wrapper for solve extraction."""
    return _extract_solve_from_raw(raw)
