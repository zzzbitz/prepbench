from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agents.profile_agent import ProfileAgent
from config.experiment_config import ExperimentConfig
from core.executor import CodeExecutor


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except Exception:
        return False
    return True


def _read_cand_profile_summary(work_dir: Path) -> str:
    cand_path = work_dir / "cand" / "profile_summary.json"
    if cand_path.exists():
        try:
            return cand_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def _get_effective_stdout(stdout: str, work_dir: Path) -> str:
    text = stdout or ""
    if text.strip():
        if _looks_like_json(text):
            return text
        cand_text = _read_cand_profile_summary(work_dir)
        if cand_text:
            return cand_text
        return text
    cand_text = _read_cand_profile_summary(work_dir)
    return cand_text or text


def _save_profile_summary(
    profile_root: Path,
    summary: str,
    error: Optional[str],
    rounds_data: list[dict],
    metadata: dict,
) -> None:
    payload = {
        "summary": summary,
        "error": error,
        "rounds": len(rounds_data),
        **metadata,
    }
    (profile_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary:
        (profile_root / "summary.txt").write_text(summary, encoding="utf-8")
    if error:
        (profile_root / "error.txt").write_text(error, encoding="utf-8")


def run_profile_phase(
    *,
    tdir: Path,
    config: ExperimentConfig,
    output_root: Path,
    query_text: str,
    input_dir: Path,
    inputs: list[str],
    inputs_preview: Dict[str, Any],
) -> dict[str, Any]:
    """Two-round profile flow with LLM decision."""
    profile_root = output_root / "profile"
    profile_root.mkdir(parents=True, exist_ok=True)

    profile_agent = ProfileAgent(config.model_name)
    executor = CodeExecutor()
    input_files = {p.name: p for p in sorted(input_dir.glob("*.csv"))}

    session_state = {
        "task_dir": str(tdir),
        "query": query_text,
        "inputs": inputs,
        "inputs_preview": inputs_preview,
    }

    max_rounds = config.profile.max_rounds
    max_summary_chars = config.profile.max_summary_chars

    rounds_data: list[dict] = []
    final_summary = ""
    final_error: Optional[str] = None

    round1_dir = profile_root / "round-1"
    round1_dir.mkdir(parents=True, exist_ok=True)

    try:
        code1, raw1, msgs1 = profile_agent.generate_profile_code(session_state)
    except Exception as e:
        final_error = f"round1_codegen_failed: {e}"
        (profile_root / "error.txt").write_text(final_error, encoding="utf-8")
        _save_profile_summary(profile_root, "", final_error, [], {})
        return {"summary": "", "error": final_error, "rounds": []}

    if code1:
        (round1_dir / "code.py").write_text(code1, encoding="utf-8")
    (round1_dir / "messages.json").write_text(
        json.dumps(msgs1, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (round1_dir / "raw_response.txt").write_text(raw1 or "", encoding="utf-8")

    if not code1:
        final_error = "round1_no_code_generated"
        exec1 = {"ok": False, "stderr": "ProfileAgent generated no code", "stdout": "", "rc": 1}
        (round1_dir / "execution.json").write_text(
            json.dumps(exec1, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_profile_summary(profile_root, "", final_error, [], {})
        return {"summary": "", "error": final_error, "rounds": []}

    ok1, stderr1, stdout1, _ = executor.execute_code(
        code1, input_files, timeout=config.timeout, work_dir=round1_dir
    )
    effective_stdout1 = _get_effective_stdout(stdout1, round1_dir)
    exec1 = {"ok": ok1, "stderr": stderr1, "stdout": effective_stdout1, "rc": 0 if ok1 else 1}
    (round1_dir / "execution.json").write_text(
        json.dumps(exec1, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rounds_data.append({"round": 1, "execution": exec1})

    decision_overridden = False
    try:
        decision, content, raw_dec, msgs_dec = profile_agent.decide_or_summarize(
            session_state, exec1, max_summary_chars
        )
    except Exception as e:
        final_error = f"decision_failed: {e}"
        if ok1 and effective_stdout1:
            final_summary = effective_stdout1.strip()[:max_summary_chars]
        _save_profile_summary(profile_root, final_summary, final_error, rounds_data, {})
        return {"summary": final_summary, "error": final_error, "rounds": rounds_data}

    original_decision = decision
    if decision == "CODE" and max_rounds < 2:
        decision = "SUMMARY"
        decision_overridden = True
        if effective_stdout1:
            content = effective_stdout1.strip()
        else:
            content = f"Profiling completed. Output: {stderr1[:500] if stderr1 else 'No output'}"

    (round1_dir / "decision_raw.txt").write_text(raw_dec or "", encoding="utf-8")
    (round1_dir / "decision_messages.json").write_text(
        json.dumps(msgs_dec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    decision_info = {
        "decision": decision,
        "original_decision": original_decision,
        "decision_overridden": decision_overridden,
        "override_reason": "max_rounds < 2" if decision_overridden else None,
        "content_preview": content[:500] if content else "",
    }
    (round1_dir / "decision.json").write_text(
        json.dumps(decision_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if decision == "SUMMARY":
        final_summary = content[:max_summary_chars] if content else ""
        _save_profile_summary(
            profile_root, final_summary, None, rounds_data, {"decision_overridden": decision_overridden}
        )
        return {"summary": final_summary, "error": None, "rounds": rounds_data}

    round2_dir = profile_root / "round-2"
    round2_dir.mkdir(parents=True, exist_ok=True)

    code2 = content
    if not code2:
        final_error = "round2_no_code_from_decision"
        _save_profile_summary(profile_root, "", final_error, rounds_data, {})
        return {"summary": "", "error": final_error, "rounds": rounds_data}

    (round2_dir / "code.py").write_text(code2, encoding="utf-8")

    ok2, stderr2, stdout2, _ = executor.execute_code(
        code2, input_files, timeout=config.timeout, work_dir=round2_dir
    )
    effective_stdout2 = _get_effective_stdout(stdout2, round2_dir)
    exec2 = {"ok": ok2, "stderr": stderr2, "stdout": effective_stdout2, "rc": 0 if ok2 else 1}
    (round2_dir / "execution.json").write_text(
        json.dumps(exec2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rounds_data.append({"round": 2, "execution": exec2})

    try:
        summary, raw_sum, msgs_sum = profile_agent.summarize(
            session_state, exec1, exec2, max_summary_chars
        )
    except Exception as e:
        final_error = f"summarize_failed: {e}"
        if ok2 and effective_stdout2:
            final_summary = effective_stdout2.strip()[:max_summary_chars]
        elif ok1 and effective_stdout1:
            final_summary = effective_stdout1.strip()[:max_summary_chars]
        _save_profile_summary(profile_root, final_summary, final_error, rounds_data, {})
        return {"summary": final_summary, "error": final_error, "rounds": rounds_data}

    (round2_dir / "summary_raw.txt").write_text(raw_sum or "", encoding="utf-8")
    (round2_dir / "summary_messages.json").write_text(
        json.dumps(msgs_sum, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_summary = summary[:max_summary_chars] if summary else ""
    _save_profile_summary(profile_root, final_summary, None, rounds_data, {})
    return {"summary": final_summary, "error": None, "rounds": rounds_data}
