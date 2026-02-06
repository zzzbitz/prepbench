from __future__ import annotations

import re

_LABELED_Q_RE = re.compile(r"^(?:q|Q)?(\d+)\s*[:\.\)]\s*(.+)$")


def parse_sub_questions(payload: str) -> tuple[list[str], str]:
    """Parse sub-questions from Ask payload.

    Returns:
        (questions, parse method: "labeled"|"semicolon"|"single")
    """
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]

    labeled: list[str] = []
    current_q: list[str] = []
    for ln in lines:
        m = _LABELED_Q_RE.match(ln)
        if m:
            if current_q:
                labeled.append(" ".join(current_q))
                current_q = []
            current_q.append(m.group(2).strip())
        elif current_q:
            current_q.append(ln)

    if current_q:
        labeled.append(" ".join(current_q))

    if labeled:
        return labeled, "labeled"

    subs = [q.strip() for q in payload.split(";") if q.strip()]
    if len(subs) > 1:
        return subs, "semicolon"

    return [payload.strip()], "single"


def _normalize_question_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_clarifier_alignment(
    *,
    expected_sub_questions: list[str],
    clarifier_answers: list[dict],
) -> tuple[bool, str]:
    if not expected_sub_questions:
        return True, ""
    if len(clarifier_answers) != len(expected_sub_questions):
        return (
            False,
            f"answer_count_mismatch: expected={len(expected_sub_questions)} got={len(clarifier_answers)}",
        )
    for i, exp in enumerate(expected_sub_questions):
        ans = clarifier_answers[i] or {}
        sq = ans.get("sub_question", "")
        if not isinstance(sq, str) or not sq.strip():
            return False, f"missing_sub_question_at_index={i}"
        if _normalize_question_for_match(sq) != _normalize_question_for_match(exp):
            return False, f"sub_question_mismatch_at_index={i}"
        answer = ans.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            return False, f"empty_answer_at_index={i}"
    return True, ""
