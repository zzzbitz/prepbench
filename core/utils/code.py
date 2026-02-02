from __future__ import annotations

import re
from typing import Optional, Tuple

_FENCED_BLOCK_RE = re.compile(
    r"```\s*([A-Za-z0-9_+\-]*)\s*\n([\s\S]*?)\n```",
    re.MULTILINE,
)

_SOLVE_DEF_RE = re.compile(r"def\s+solve\s*\([^)]*\)\s*(?:->[^:]+)?\s*:\s*", re.MULTILINE)


def _strip_possible_lang_header(code: str) -> str:
    """If the first line looks like a language header, remove it."""
    lines = code.splitlines()
    if not lines:
        return code
    first = lines[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_+\-]+", first):
        return "\n".join(lines[1:])
    return code


def extract_single_code_block(raw: Optional[str]) -> str:
    """Extract the first fenced code block or return raw code-like text."""
    if not raw:
        return ""

    m = _FENCED_BLOCK_RE.search(raw)
    if not m:
        if raw.lstrip().startswith(("import ", "from ", "def ", "class ", "# ")):
            return raw.strip()
        return ""

    lang = (m.group(1) or "").lower()
    body = m.group(2)
    if lang not in {"python", "py"}:
        body = _strip_possible_lang_header(body)
    return body.strip("\n")


def extract_solve_from_raw(raw: Optional[str]) -> str:
    """Extract solve(...) implementation from raw LLM response."""
    if not raw:
        return ""

    code = extract_single_code_block(raw)
    candidate = code or raw

    m = _SOLVE_DEF_RE.search(candidate)
    if not m:
        return ""

    solve_src = candidate[m.start() :].lstrip("\n")

    has_return = False
    for line in solve_src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.search(r"\breturn\b", s):
            has_return = True
            break
    if not has_return:
        return ""

    return solve_src.rstrip()


def extract_code_from_response(response: str) -> str:
    if not isinstance(response, str):
        return ""

    text = response.strip()
    if not text:
        return ""

    pattern_lang = re.compile(r"```(?:python|py)\s+([\s\S]*?)```", re.IGNORECASE)
    m = pattern_lang.search(text)
    if m:
        return m.group(1).lstrip("\n")

    pattern_any = re.compile(r"```\s*([\s\S]*?)```", re.IGNORECASE)
    m = pattern_any.search(text)
    if m:
        return m.group(1).lstrip("\n")

    if text.startswith(("import ", "from ", "def ", "class ", "# ")):
        return text

    return ""


def extract_single_solution_from_raw(raw: Optional[str]) -> Tuple[str, bool]:
    """
    Extract code from raw response and validate syntax.
    Returns: (code, is_complete)
    """
    code = extract_code_from_response(raw)
    if not code:
        return "", False

    solve_src = extract_solve_from_raw(code) or extract_solve_from_raw(raw)
    final_code = solve_src or code

    # Verify syntax completeness
    try:
        compile(final_code, "<string>", "exec")
        return final_code, True
    except SyntaxError:
        return final_code, False
