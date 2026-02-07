from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agents.code_agent import CodeAgent
from agents.clarify_agent import ClarifyAgent
from simulator.user_simulator import UserSimulator
from core.case_assets import read_reference_solution_text, require_reference_solution_path
from core.utils.paths import get_output_path
from .executor import CodeExecutor
from config.experiment_config import ExperimentConfig
from core.orchestration.common import copy_solution_artifacts, resolve_query_path
from core.orchestration.clarify_parse import parse_sub_questions, validate_user_simulator_alignment
from core.orchestration.flow_phase import run_flow_impl
from core.orchestration.mode_spec import allowed_run_modes
from core.orchestration.profile_phase import run_profile_phase
from core.utils.exec_utils import render_main_with_solve


class Orchestrator:
    def __init__(self, parallel_execution: bool = True, max_workers: Optional[int] = None) -> None:
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
    
    def _execute_solution(
        self,
        solve_source: str,
        round_dir: Path,
        input_dir: Path,
        executor: CodeExecutor,
        *,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        solution_dir = round_dir / "solution"
        solution_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            main_code = render_main_with_solve(solve_source)
        except Exception as e:
            exec_info = {"rc": 1, "stderr": str(e), "ok": False}
            (solution_dir / "execution.json").write_text(json.dumps(exec_info, ensure_ascii=False, indent=2), encoding="utf-8")
            solution_result = {
                "code": solve_source,
                "code_path": str((solution_dir / "code.py").resolve()),
                "cand_dir": str((solution_dir / "cand").resolve()),
                "execution": exec_info,
            }
            return solution_result
        
        (solution_dir / "code.py").write_text(solve_source, encoding="utf-8")
        code_path = solution_dir / "main.py"
        code_path.write_text(main_code, encoding="utf-8")
        

        input_files = {p.name: p for p in sorted(input_dir.glob("*.csv"))}
        ok, stderr, stdout, outputs = executor.execute_code(main_code, input_files, timeout=timeout, work_dir=solution_dir)
        exec_info = {"rc": 0 if ok else 1, "stderr": stderr, "stdout": stdout, "ok": ok}
        (solution_dir / "execution.json").write_text(json.dumps(exec_info, ensure_ascii=False, indent=2), encoding="utf-8")
        
        cand_dir = solution_dir / "cand"
          
        solution_result = {
            "code": solve_source,
            "code_path": str(code_path.resolve()),
            "cand_dir": str(cand_dir.resolve()),
            "execution": exec_info,
        }
        return solution_result

    def _prepare_interact_context(
        self,
        tdir: Path,
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        from core.utils.paths import get_output_path

        missing = []
        invalid_json = []

        if not (tdir / "inputs").is_dir():
            missing.append("inputs/")

        if not (tdir / "query.md").exists():
            missing.append("query.md")

        amb_kb_path = tdir / "amb_kb.json"
        amb_kb_json: Optional[Dict[str, Any]] = None
        if not amb_kb_path.exists():
            missing.append("amb_kb.json")
        else:
            if amb_kb_path.stat().st_size == 0:
                invalid_json.append("amb_kb.json (exists but empty)")
            else:
                try:
                    amb_kb_json = json.loads(amb_kb_path.read_text(encoding="utf-8"))
                    if not isinstance(amb_kb_json, dict):
                        invalid_json.append("amb_kb.json (not a JSON object)")
                        amb_kb_json = None
                except Exception:
                    invalid_json.append("amb_kb.json (invalid JSON)")

        errors = missing + invalid_json
        if errors:
            raise FileNotFoundError(
                f"[{config.run_mode} mode] Cannot start: {', '.join(errors)} (case: {tdir.name})"
            )

        from llm_connect.config import validate_user_simulator_settings
        validate_user_simulator_settings()
        # ================================================================

        output_root = get_output_path(tdir, config)
        input_dir = tdir / "inputs"

        solution_text = read_reference_solution_text(tdir)

        from core.data_head import DataHead
        data_head = DataHead()
        inputs_preview = data_head.get_preview(input_dir)

        query_full_path = tdir / "query_full.md"
        if not query_full_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_full_path}")
        query_full_text = query_full_path.read_text(encoding="utf-8")

        query_md_path = resolve_query_path(
            tdir,
            run_mode=config.run_mode,
        )

        if not query_md_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_md_path} (run_mode={config.run_mode})")

        query_text = query_md_path.read_text(encoding="utf-8")

        return {
            "output_root": output_root,
            "input_dir": input_dir,
            "query_text": query_text,
            "query_full_text": query_full_text,
            "inputs_preview": inputs_preview,
            "amb_kb_json": amb_kb_json or {},
            "solution_text": solution_text,
        }

    def _run_clarify_phase(
        self,
        *,
        tdir: Path,
        config: ExperimentConfig,
        output_root: Path,
        input_dir: Path,
        query_text: str,
        query_full_text: str,
        inputs_preview: Dict[str, Any],
        amb_kb_json: Dict[str, Any],
        solution_text: str,
    ) -> Dict[str, Any]:
        clarify_root = output_root / "clarify"
        clarify_root.mkdir(parents=True, exist_ok=True)

        amb_count = 0
        slot_id_set: set[str] = set()
        if isinstance(amb_kb_json, dict):
            ambiguities = amb_kb_json.get("ambiguities")
            if isinstance(ambiguities, list):
                amb_count = len(ambiguities)
                for entry in ambiguities:
                    if not isinstance(entry, dict):
                        continue
                    slot_id = entry.get("id")
                    if isinstance(slot_id, str) and slot_id:
                        slot_id_set.add(slot_id)

        max_questions = None
        if config.question_ratio is not None:
            max_questions = max(1, math.ceil(config.question_ratio * amb_count))
            if config.max_questions_cap is not None:
                # Cap is at least amb_count to ensure full coverage is possible
                effective_cap = max(int(config.max_questions_cap), amb_count)
                max_questions = min(max_questions, effective_cap)

        sut = ClarifyAgent(config.model_name)
        user_simulator = UserSimulator()

        from llm_connect.config import get_model_name
        try:
            user_simulator_model = get_model_name(agent="user_simulator") or None
        except Exception:
            user_simulator_model = None

        # ========== Phase 1: Clarify ==========
        clarify_session_state = {
            "task_dir": str(tdir),
            "input_dir": str(input_dir),
            "model_name": config.model_name,
            "run_mode": config.run_mode,
            "output_root": str(output_root),
            "query": query_text,
            "inputs_preview": inputs_preview,
        }

        qa_history_internal: list[dict] = []
        questions_used = 0
        clarify_hist: list[dict] = []
        clarify_stopped_reason = "done"
        runtime_feedback: Optional[Dict[str, Any]] = None

        for r in range(1, config.max_rounds_interact + 1):
            round_dir = clarify_root / f"round-{r}"
            round_dir.mkdir(parents=True, exist_ok=True)

            qa_history_for_sut = [
                {
                    "question": h.get("question", ""),
                    "answer": h.get("answer", ""),
                    "classification": h.get("classification"),
                    "source": h.get("source"),
                    "ref": h.get("ref"),
                }
                for h in qa_history_internal
            ]
            action, raw, messages = sut.generate_action(
                clarify_session_state,
                qa_history=qa_history_for_sut,
                runtime_feedback=runtime_feedback,
                code_started=False,  # Never started in clarify-only mode
                max_questions=max_questions,
                max_questions_per_ask=config.max_questions_per_ask,
                questions_used=questions_used,
            )
            (round_dir / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            (round_dir / "raw_response.txt").write_text(raw or "", encoding="utf-8")

            action_rec = {
                "round": r,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "run_mode": config.run_mode,
                "sut_model": config.model_name,
                "user_simulator_model": user_simulator_model,
                "questions_used": questions_used,
                "max_questions": max_questions,
                "action": action.action_type,
                "content": action.content,
                "parse_error": action.parse_error,
            }
            (round_dir / "action.json").write_text(json.dumps(action_rec, ensure_ascii=False, indent=2), encoding="utf-8")

            # Handle Done: action - clarify phase complete
            if action.action_type == "done":
                clarify_hist.append({"round": r, "action": "done"})
                clarify_stopped_reason = "done"
                runtime_feedback = None
                break

            # Handle Ask: action
            if action.action_type == "ask":
                runtime_feedback = None  # Clear previous feedback on valid action

                # Check quota exhausted
                if max_questions is not None and questions_used >= max_questions:
                    refusal_answer = "Question limit reached. Proceeding to code phase."
                    (round_dir / "user_simulator.json").write_text(json.dumps({
                        "classification": "refuse_illegal",
                        "source": "refuse",
                        "answer": refusal_answer,
                        "details": {"reason": "max_questions"},
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    (round_dir / "max_questions.json").write_text(json.dumps({
                        "used": questions_used,
                        "limit": max_questions,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    clarify_hist.append({"round": r, "action": "ask_refused", "reason": "max_questions"})
                    clarify_stopped_reason = "max_questions"
                    break

                # Parse sub-questions using new logic
                sub_questions, parse_method = parse_sub_questions(action.content)
                sub_count = len(sub_questions)

                # Write parse result
                (round_dir / "ask_parsed.json").write_text(json.dumps({
                    "parsed": sub_questions,
                    "method": parse_method,
                    "count": sub_count,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

                # Truncate if exceeds remaining quota
                truncated = False
                if max_questions is not None:
                    remaining = max_questions - questions_used
                    # Enforce max_questions_per_ask in code (prompt-only enforcement is not reliable).
                    per_ask = max(1, int(config.max_questions_per_ask or 1))
                    allowed = min(remaining, per_ask)
                    if sub_count > remaining:
                        original_count = sub_count
                        sub_questions = sub_questions[:remaining]
                        sub_count = remaining
                        truncated = True
                        # Write truncation audit
                        (round_dir / "ask_truncated.json").write_text(json.dumps({
                            "requested": original_count,
                            "used": remaining,
                            "method": parse_method,
                        }, ensure_ascii=False, indent=2), encoding="utf-8")
                    if sub_count > allowed:
                        original_count = sub_count
                        sub_questions = sub_questions[:allowed]
                        sub_count = allowed
                        truncated = True
                        (round_dir / "ask_truncated_per_ask.json").write_text(
                            json.dumps(
                                {"requested": original_count, "used": allowed, "limit": per_ask, "remaining": remaining},
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                else:
                    per_ask = max(1, int(config.max_questions_per_ask or 1))
                    if sub_count > per_ask:
                        original_count = sub_count
                        sub_questions = sub_questions[:per_ask]
                        sub_count = per_ask
                        truncated = True
                        (round_dir / "ask_truncated_per_ask.json").write_text(
                            json.dumps({"requested": original_count, "used": per_ask, "limit": per_ask}, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                # Build question payload for user simulator (stable, unambiguous formatting)
                # Note: we still keep asked_question for audit, but user simulator should use Expected Sub-Questions list.
                action_content_truncated = "\n".join([f"q{i+1}: {q}" for i, q in enumerate(sub_questions)])

                expected_sub_questions = list(sub_questions)

                # Call user simulator (with one retry on format/alignment mismatch)
                # Disable tracker for user simulator calls (only count ClarifyAgent, not oracle)
                from llm_connect.usage_tracker import get_tracker, set_tracker
                saved_tracker = get_tracker()
                set_tracker(None)
                try:
                    clar = user_simulator.answer(
                        query_full_text=query_full_text,
                        amb_kb_json=amb_kb_json,
                        solution_text=solution_text,
                        question=action_content_truncated,
                        expected_sub_questions=expected_sub_questions,
                        runtime_feedback="",
                    )
                finally:
                    set_tracker(saved_tracker)
                (round_dir / "user_simulator_messages.json").write_text(
                    json.dumps(clar.messages, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / "user_simulator_raw.txt").write_text(clar.raw_response or "", encoding="utf-8")
                if getattr(clar, "parse_error", None):
                    (round_dir / "user_simulator_parse_error.json").write_text(
                        json.dumps(
                            {
                                "error": clar.parse_error,
                                "expected_sub_questions": expected_sub_questions,
                                "raw_attempts": clar.raw_attempts or [],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                def _select_slot_id(ans: Any) -> Optional[str]:
                    if getattr(ans, "classification", "") != "hit":
                        return None
                    if not slot_id_set:
                        return None
                    candidates: list[str] = []
                    for val in (getattr(ans, "ref", None), getattr(ans, "canonical_value", None)):
                        if isinstance(val, str) and val:
                            candidates.append(val)
                    details = getattr(ans, "details", None)
                    if isinstance(details, dict):
                        for key in ("slot_id", "ref"):
                            val = details.get(key)
                            if isinstance(val, str) and val:
                                candidates.append(val)
                    for val in candidates:
                        if val in slot_id_set:
                            return val
                    return None

                def _to_answer_dict(ans: Any) -> dict:
                    slot_id = _select_slot_id(ans)
                    return {
                        "sub_question": getattr(ans, "sub_question", ""),
                        "classification": getattr(ans, "classification", "fallback"),
                        "source": getattr(ans, "source", "fallback"),
                        "answer": getattr(ans, "answer", ""),
                        "ref": getattr(ans, "ref", None),
                        "slot_id": slot_id,
                    }

                clar_answer_dicts = [_to_answer_dict(a) for a in (clar.answers or [])]
                ok_align, align_err = validate_user_simulator_alignment(
                    expected_sub_questions=expected_sub_questions,
                    user_simulator_answers=clar_answer_dicts,
                )

                if not ok_align:
                    (round_dir / "user_simulator_alignment_error.json").write_text(
                        json.dumps(
                            {
                                "error": align_err,
                                "expected_count": len(expected_sub_questions),
                                "got_count": len(clar_answer_dicts),
                                "expected_sub_questions": expected_sub_questions,
                                "got_sub_questions": [d.get("sub_question", "") for d in clar_answer_dicts],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    # Retry once with explicit feedback injected into the prompt.
                    # Disable tracker for user simulator retry (only count ClarifyAgent)
                    saved_tracker_retry = get_tracker()
                    set_tracker(None)
                    try:
                        clar_retry = user_simulator.answer(
                            query_full_text=query_full_text,
                            amb_kb_json=amb_kb_json,
                            solution_text=solution_text,
                            question=action_content_truncated,
                            expected_sub_questions=expected_sub_questions,
                            runtime_feedback=(
                                "Your previous response did not satisfy the Output Contract. "
                                f"Error={align_err}. Regenerate JSON with answers exactly matching Expected Sub-Questions."
                            ),
                        )
                    finally:
                        set_tracker(saved_tracker_retry)
                    (round_dir / "user_simulator_retry_raw.txt").write_text(clar_retry.raw_response or "", encoding="utf-8")
                    clar = clar_retry
                    clar_answer_dicts = [_to_answer_dict(a) for a in (clar.answers or [])]
                    ok_align, align_err = validate_user_simulator_alignment(
                        expected_sub_questions=expected_sub_questions,
                        user_simulator_answers=clar_answer_dicts,
                    )
                    (round_dir / "user_simulator_alignment_after_retry.json").write_text(
                        json.dumps({"ok": ok_align, "error": align_err}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                # If still misaligned, synthesize per-question refusal answers (safer than polluting qa_history).
                if not ok_align:
                    clar_answer_dicts = [
                        {
                            "sub_question": q,
                            "classification": "refuse_illegal",
                            "source": "refuse",
                            "answer": "I cannot answer this because the user simulator response was invalid.",
                            "ref": None,
                            "slot_id": None,
                        }
                        for q in expected_sub_questions
                    ]
                    # Preserve a combined-level classification/source for auditing.
                    combined_classification = "refuse_illegal"
                    combined_source = "refuse"
                    combined_answer = "User simulator output invalid; synthesized refusal answers."
                    combined_ref = None
                else:
                    combined_classification = clar.classification
                    combined_source = clar.source
                    combined_answer = clar.answer
                    combined_ref = clar.ref

                clar_rec = {
                    "model": user_simulator_model,
                    "answers": clar_answer_dicts,
                    "classification": combined_classification,
                    "source": combined_source,
                    "answer": combined_answer,
                    "ref": combined_ref,
                    "expected_sub_questions": expected_sub_questions,
                }
                (round_dir / "user_simulator.json").write_text(json.dumps(clar_rec, ensure_ascii=False, indent=2), encoding="utf-8")

                # Charge quota by asked sub-questions (not by user simulator reply length).
                questions_used += len(expected_sub_questions)

                for d in clar_answer_dicts:
                    qa_history_internal.append(
                        {
                            "question": d.get("sub_question", ""),
                            "answer": d.get("answer", ""),
                            "classification": d.get("classification"),
                            "source": d.get("source"),
                            "ref": d.get("ref"),
                            "slot_id": d.get("slot_id"),
                        }
                    )

                clarify_hist.append({
                    "round": r,
                    "action": "ask",
                    "question": action.content,
                    "asked_question": action_content_truncated,
                    "parsed_count": sub_count,
                    "actual_answer_count": len(clar_answer_dicts),
                    "parse_method": parse_method,
                    "truncated": truncated,
                    "user_simulator": clar_rec,
                })
                continue

            # Invalid action - provide feedback for next round
            (round_dir / "invalid_feedback.json").write_text(json.dumps({
                "parse_error": action.parse_error,
                "hint": "Response must start with Ask: or Done:; use q1:/q2: prefix for multiple sub-questions",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            clarify_hist.append({"round": r, "action": "invalid", "parse_error": action.parse_error})
            runtime_feedback = {
                "parse_error": action.parse_error,
                "hint": "Response must start with Ask: or Done:; use q1:/q2: prefix for multiple sub-questions",
            }
            # Continue to next round with feedback
        else:
            # Loop exhausted without break
            clarify_stopped_reason = "max_rounds"

        # Save clarify phase summary
        asked_count = len(qa_history_internal)
        hit_count = sum(1 for qa in qa_history_internal if qa.get("classification") == "hit")
        refuse_count = sum(
            1
            for qa in qa_history_internal
            if str(qa.get("classification") or "").startswith("refuse")
        )
        answered_count = asked_count - refuse_count
        hit_rate = (hit_count / asked_count) if asked_count else 0.0
        clarify_summary = {
            "stopped_reason": clarify_stopped_reason,
            "questions_used": questions_used,
            "rounds": len(clarify_hist),
            "asked_count": asked_count,
            "answered_count": answered_count,
            "refuse_count": refuse_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "qa_history": qa_history_internal,
        }
        (clarify_root / "summary.json").write_text(json.dumps(clarify_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "qa_history": qa_history_internal,
            "clarify_history": clarify_hist,
            "stopped_reason": clarify_stopped_reason,
            "questions_used": questions_used,
        }

    def _read_json_if_exists(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_profile_summary(self, cache_root: Path) -> Optional[str]:
        summary_path = cache_root / "profile" / "summary.json"
        if not summary_path.exists():
            return None
        data = self._read_json_if_exists(summary_path)
        if not isinstance(data, dict):
            return None
        summary = data.get("summary")
        error = data.get("error")
        if error or not isinstance(summary, str) or not summary.strip():
            return None
        return summary

    def _profile_cache_roots(self, tdir: Path, config: ExperimentConfig) -> list[Path]:
        # Profile artifacts are produced under interact/orig/disamb roots.
        # Keep current mode first when it is profile-capable.
        current_mode = str(config.run_mode or "").strip().lower()
        ordered_modes: list[str] = []
        if current_mode in {"orig", "disamb", "interact"}:
            ordered_modes.append(current_mode)
        for mode in ("interact", "disamb", "orig"):
            if mode not in ordered_modes:
                ordered_modes.append(mode)

        roots: list[Path] = []
        for mode in ordered_modes:
            mode_cfg = dataclasses.replace(config, run_mode=mode)
            roots.append(get_output_path(tdir, mode_cfg))
        return roots

    def _find_profile_cache(self, tdir: Path, config: ExperimentConfig) -> tuple[Optional[str], Optional[Path]]:
        for root in self._profile_cache_roots(tdir, config):
            summary = self._read_profile_summary(root)
            if summary is not None:
                return summary, root
        return None, None

    def _materialize_reused_profile_summary(
        self,
        target_root: Path,
        summary: str,
        source_root: Optional[Path],
    ) -> None:
        if not isinstance(summary, str) or not summary.strip():
            return
        profile_root = target_root / "profile"
        profile_root.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "summary": summary,
            "error": None,
            "rounds": 0,
            "reused": True,
        }
        if source_root is not None:
            payload["reused_from"] = str(source_root)
        (profile_root / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (profile_root / "summary.txt").write_text(summary, encoding="utf-8")

    def _run_interact(self, tdir: Path, config: ExperimentConfig) -> Dict[str, Any]:
        """Run interact mode as e2e pipeline without flow stage."""
        interact_cfg = dataclasses.replace(config, run_mode="interact")
        return self._run_e2e(tdir, interact_cfg, with_flow=False)

    def _run_e2e(self, tdir: Path, config: ExperimentConfig, *, with_flow: bool = True) -> Dict[str, Any]:
        """Run end-to-end pipeline.

        with_flow=False: clarify + profile + code (interact mode).
        with_flow=True: clarify + profile + code + flow (e2e mode).
        """
        ctx = self._prepare_interact_context(tdir, config)

        def _iter_round_dirs(clarify_root: Path) -> list[Path]:
            rounds: list[tuple[int, Path]] = []
            for p in clarify_root.iterdir():
                if not p.is_dir() or not p.name.startswith("round-"):
                    continue
                try:
                    idx = int(p.name.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                rounds.append((idx, p))
            return [p for _, p in sorted(rounds, key=lambda item: item[0])]

        def _load_clarify_cache(cache_root: Path) -> tuple[Optional[list[dict]], list[dict], int]:
            summary_path = cache_root / "clarify" / "summary.json"
            summary = self._read_json_if_exists(summary_path)
            qa_history: Optional[list[dict]] = None
            rounds = 0
            if isinstance(summary, dict):
                qa_history = summary.get("qa_history")
                if not isinstance(qa_history, list):
                    qa_history = None
                rounds = summary.get("rounds") if isinstance(summary.get("rounds"), int) else 0

            clarify_history: list[dict] = []
            clarify_root = cache_root / "clarify"
            if clarify_root.exists():
                for round_dir in _iter_round_dirs(clarify_root):
                    action = self._read_json_if_exists(round_dir / "action.json")
                    if not isinstance(action, dict):
                        continue
                    action_type = action.get("action") or "unknown"
                    round_num = action.get("round")
                    if not isinstance(round_num, int):
                        try:
                            round_num = int(round_dir.name.split("-", 1)[1])
                        except (IndexError, ValueError):
                            round_num = 0
                    rec = {"round": round_num, "action": action_type}

                    if action_type == "ask":
                        rec["question"] = action.get("content", "")
                        parsed = self._read_json_if_exists(round_dir / "ask_parsed.json")
                        if isinstance(parsed, dict):
                            sub_questions = parsed.get("parsed")
                            if isinstance(sub_questions, list):
                                rec["asked_question"] = "\n".join(
                                    [f"q{i + 1}: {q}" for i, q in enumerate(sub_questions)]
                                )
                            if isinstance(parsed.get("count"), int):
                                rec["parsed_count"] = parsed.get("count")
                            if isinstance(parsed.get("method"), str):
                                rec["parse_method"] = parsed.get("method")
                        rec["truncated"] = (
                            (round_dir / "ask_truncated.json").exists()
                            or (round_dir / "ask_truncated_per_ask.json").exists()
                        )
                        user_simulator_resp = self._read_json_if_exists(round_dir / "user_simulator.json")
                        if isinstance(user_simulator_resp, dict):
                            rec["user_simulator"] = user_simulator_resp
                            answers = user_simulator_resp.get("answers")
                            if isinstance(answers, list):
                                rec["actual_answer_count"] = len(answers)
                        if (round_dir / "max_questions.json").exists():
                            rec["action"] = "ask_refused"
                            rec["reason"] = "max_questions"
                    elif action_type == "invalid":
                        invalid = self._read_json_if_exists(round_dir / "invalid_feedback.json")
                        if isinstance(invalid, dict):
                            rec["parse_error"] = invalid.get("parse_error")
                        if not rec.get("parse_error"):
                            rec["parse_error"] = action.get("parse_error")
                    clarify_history.append(rec)

            if not rounds:
                rounds = len(clarify_history)
            return qa_history, clarify_history, rounds

        inputs = [p.name for p in sorted(Path(ctx["input_dir"]).glob("*.csv"))]

        # Load cached interact artifacts for this model; regenerate if missing.
        interact_cfg = dataclasses.replace(config, run_mode="interact")
        interact_output_root = get_output_path(tdir, interact_cfg)
        qa_history, clarify_history, clarify_rounds = _load_clarify_cache(interact_output_root)
        if qa_history is None:
            self._run_clarify_phase(
                tdir=tdir,
                config=interact_cfg,
                output_root=interact_output_root,
                input_dir=ctx["input_dir"],
                query_text=ctx["query_text"],
                query_full_text=ctx["query_full_text"],
                inputs_preview=ctx["inputs_preview"],
                amb_kb_json=ctx["amb_kb_json"],
                solution_text=ctx["solution_text"],
            )
            qa_history, clarify_history, clarify_rounds = _load_clarify_cache(interact_output_root)
            if qa_history is None:
                return {"skipped": True, "reason": "interact_regen_failed"}

        # Load cached profile summary for this model; regenerate if missing.
        profile_summary = ""
        profile_error: Optional[str] = None
        profile_cfg = dataclasses.replace(config, run_mode="interact")
        profile_output_root = interact_output_root
        profile_usage_root = interact_output_root
        cached_summary, cached_profile_root = self._find_profile_cache(tdir, profile_cfg)
        if cached_summary is None:
            profile_result = self._run_profile_phase(
                tdir=tdir,
                config=profile_cfg,
                output_root=profile_output_root,
                query_text=ctx["query_text"],
                input_dir=ctx["input_dir"],
                inputs=inputs,
                inputs_preview=ctx["inputs_preview"],
            )
            regen_error = profile_result.get("error") if isinstance(profile_result, dict) else None
            if isinstance(regen_error, str) and regen_error.strip():
                profile_error = regen_error
            cached_summary, cached_profile_root = self._find_profile_cache(tdir, profile_cfg)
            if cached_summary is None:
                regenerated_summary = profile_result.get("summary") if isinstance(profile_result, dict) else None
                if isinstance(regenerated_summary, str):
                    cached_summary = regenerated_summary
                else:
                    cached_summary = ""
                cached_profile_root = interact_output_root
            profile_usage_root = interact_output_root
        else:
            if cached_profile_root is not None:
                profile_usage_root = cached_profile_root
                if cached_profile_root != interact_output_root:
                    self._materialize_reused_profile_summary(
                        target_root=interact_output_root,
                        summary=cached_summary,
                        source_root=cached_profile_root,
                    )
        profile_summary = cached_summary

        def _zero_usage() -> dict:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "cost_usd": 0.0,
            }

        def _normalize_usage(section: Optional[dict]) -> dict:
            data = _zero_usage()
            if isinstance(section, dict):
                data["prompt_tokens"] = int(section.get("prompt_tokens") or 0)
                data["completion_tokens"] = int(section.get("completion_tokens") or 0)
                data["total_tokens"] = int(section.get("total_tokens") or 0)
                data["call_count"] = int(section.get("call_count") or 0)
                data["cost_usd"] = float(section.get("cost_usd") or 0.0)
            if data["total_tokens"] == 0:
                data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
            return data

        def _read_token_section(token_path: Path, section: str) -> dict:
            token_data = self._read_json_if_exists(token_path)
            if isinstance(token_data, dict):
                return _normalize_usage(token_data.get(section))
            return _normalize_usage(None)

        def _sum_usage(sections: list[dict]) -> dict:
            total = _zero_usage()
            for section in sections:
                total["prompt_tokens"] += int(section.get("prompt_tokens") or 0)
                total["completion_tokens"] += int(section.get("completion_tokens") or 0)
                total["total_tokens"] += int(section.get("total_tokens") or 0)
                total["call_count"] += int(section.get("call_count") or 0)
                total["cost_usd"] += float(section.get("cost_usd") or 0.0)
            if total["total_tokens"] == 0:
                total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
            return total

        def _copy_flow_artifacts(flow_solution_dir: Path, final_solution_dir: Path) -> None:
            import shutil
            if not flow_solution_dir.exists():
                return
            final_solution_dir.mkdir(parents=True, exist_ok=True)

            flow_json_src = flow_solution_dir / "flow.json"
            if flow_json_src.exists():
                shutil.copy2(flow_json_src, final_solution_dir / "flow.json")

            flow_cand_src = flow_solution_dir / "flow_cand"
            if flow_cand_src.exists():
                flow_cand_dst = final_solution_dir / "flow_cand"
                if flow_cand_dst.exists():
                    shutil.rmtree(flow_cand_dst)
                shutil.copytree(flow_cand_src, flow_cand_dst)

            flow_exec_src = flow_solution_dir / "execution.json"
            if flow_exec_src.exists():
                shutil.copy2(flow_exec_src, final_solution_dir / "flow_execution.json")

            flow_eval_src = flow_solution_dir / "evaluation.json"
            if flow_eval_src.exists():
                shutil.copy2(flow_eval_src, final_solution_dir / "flow_evaluation.json")

            flow_status_src = flow_solution_dir / "final_status.json"
            if flow_status_src.exists():
                shutil.copy2(flow_status_src, final_solution_dir / "flow_final_status.json")

        # Initialize usage tracker (code + flow for e2e)
        from llm_connect.usage_tracker import UsageTracker, set_tracker

        prompt_price, completion_price = 0.0, 0.0
        tracker = UsageTracker(
            model=config.model_name,
            prompt_price=prompt_price,
            completion_price=completion_price,
        )
        set_tracker(tracker)

        try:
            # ========== Code Phase ==========
            tracker.set_phase("code")
            code_output_root = interact_output_root
            code_session_state = {
                "task_dir": str(tdir),
                "input_dir": str(ctx["input_dir"]),
                "model_name": config.model_name,
                "run_mode": config.run_mode,
                "output_root": str(code_output_root),
                "query": ctx["query_text"],
                "qa_history": qa_history or [],
            }
            code_session_state["profile_summary"] = profile_summary

            # Check if code phase already completed (cand directory exists with outputs)
            cand_dir = code_output_root / "solution" / "cand"
            code_already_done = cand_dir.exists() and any(cand_dir.iterdir())
            skip_code_phase = False
            
            if code_already_done:
                # Code phase completed previously - skip to flow phase
                code_result = {"passed": False, "rounds": 0, "history": []}
                skip_code_phase = True
            else:
                code_result = self._run_code_phase(
                    tdir=tdir,
                    config=config,
                    session_state=code_session_state,
                    output_root=code_output_root,
                )
            
            # code_exec_ok: code ran without errors and produced candidate outputs
            code_exec_ok = code_already_done or (
                (code_output_root / "solution" / "cand").exists() and 
                any((code_output_root / "solution" / "cand").iterdir())
            )
            code_passed = bool(code_result.get("passed", False))
            code_reason = str(code_result.get("stopped_reason") or "")
            final_solution_dir = ctx["output_root"] / "solution"

            # Keep e2e output synchronized with interact code artifacts.
            code_solution_dir = code_output_root / "solution"
            if code_output_root != ctx["output_root"] and code_solution_dir.exists():
                copy_solution_artifacts(code_solution_dir, final_solution_dir)
            
            flow_result: Optional[Dict[str, Any]] = None
            flow_passed = not with_flow
            flow_ran = False
            flow_reason = "skipped_flow_disabled" if not with_flow else "skipped_code_failed"
            flow_output_root = ctx["output_root"] / "flow"
            flow_solution_dir = flow_output_root / "solution"

            # Run flow if code executed successfully (cand exists).
            if with_flow and code_exec_ok:
                solution_path = code_output_root / "solution" / "solution.py"
                solution_text = ""
                if solution_path.exists():
                    solution_text = solution_path.read_text(encoding="utf-8")
                else:
                    hist = code_result.get("history") or []
                    if hist and isinstance(hist[-1], dict):
                        last_solution = hist[-1].get("solution") or {}
                        if isinstance(last_solution, dict):
                            solution_text = last_solution.get("code") or ""

                if solution_text.strip():
                    tracker.set_phase("flow")
                    flow_result = self._run_flow_impl(
                        tdir=tdir,
                        config=config,
                        output_root=flow_output_root,
                        solution_text=solution_text,
                    )
                    flow_ran = True
                    flow_passed = bool(flow_result.get("passed", False))
                    flow_reason = flow_result.get("stopped_reason", "unknown")
                    _copy_flow_artifacts(flow_solution_dir, ctx["output_root"] / "solution")
                else:
                    flow_reason = "solution_missing"

            # Update final status with code/flow outcomes
            code_status_path = code_output_root / "solution" / "final_status.json"
            code_final_status = self._read_json_if_exists(code_status_path)
            if not isinstance(code_final_status, dict):
                code_final_status = {}

            flow_final_status = self._read_json_if_exists(flow_solution_dir / "final_status.json")
            if not isinstance(flow_final_status, dict):
                flow_final_status = {}

            code_ok = bool(code_final_status.get("ok", False))
            code_rc = int(code_final_status.get("rc", 1))
            flow_ok = bool(flow_final_status.get("ok", False)) if flow_ran else False
            flow_rc = int(flow_final_status.get("rc", 1))

            # Preserve code_passed/code_reason from existing status when skipping code phase
            if skip_code_phase:
                code_passed = bool(
                    code_final_status.get(
                        "code_passed",
                        code_final_status.get("passed", code_passed),
                    )
                )
                code_reason = str(code_final_status.get("code_reason") or code_final_status.get("reason") or code_reason)
            
            overall_passed = code_passed and (flow_passed if with_flow else True)
            if overall_passed:
                overall_reason = "passed"
            elif not code_passed:
                overall_reason = "code_failed"
            elif with_flow:
                overall_reason = "flow_failed"
            else:
                overall_reason = "passed"

            if not skip_code_phase:
                code_reason = str(code_final_status.get("reason") or code_reason)
            final_status = {
                "ok": bool(code_ok and (flow_ok if flow_ran else True)),
                "rc": flow_rc if flow_ran else code_rc,
                "passed": overall_passed,
                "reason": overall_reason,
                "rounds_used": code_final_status.get("rounds_used", code_result.get("rounds", 0)),
                "message": code_final_status.get("message", ""),
                "code_passed": code_passed,
                "code_reason": code_reason or code_result.get("stopped_reason"),
                "flow_passed": bool(flow_passed) if with_flow else False,
                "flow_reason": str(flow_final_status.get("reason", flow_reason)),
                "gui_passed": bool(flow_passed) if with_flow else False,
                "gui_reason": str(flow_final_status.get("reason", flow_reason)),
            }
            final_solution_dir.mkdir(parents=True, exist_ok=True)
            (final_solution_dir / "final_status.json").write_text(
                json.dumps(final_status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Merge token usage: clarify/profile from cache, code from existing or tracker, flow from tracker
            clarify_usage = _read_token_section(interact_output_root / "token_usage.json", "clarify")
            profile_usage = _read_token_section(profile_usage_root / "token_usage.json", "profile")
            tracker_data = tracker.to_dict()
            
            # If code phase was skipped, read code usage from existing token_usage.json
            if skip_code_phase:
                code_usage = _read_token_section(code_output_root / "token_usage.json", "code")
            else:
                code_usage = _normalize_usage(tracker_data.get("code"))
            
            flow_usage = _normalize_usage(tracker_data.get("flow"))
            total_usage = _sum_usage([clarify_usage, profile_usage, code_usage, flow_usage])

            token_payload = {
                "model": config.model_name,
                "clarify": clarify_usage,
                "profile": profile_usage,
                "code": code_usage,
                "flow": flow_usage,
                "total": total_usage,
            }
            if tracker.unknown_usage_calls > 0:
                token_payload["unknown_usage_calls"] = tracker.unknown_usage_calls
            (ctx["output_root"] / "token_usage.json").write_text(
                json.dumps(token_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            return {
                "passed": overall_passed,
                "code_passed": code_passed,
                "flow_passed": bool(flow_passed) if with_flow else False,
                "flow_ran": flow_ran,
                "flow_reason": final_status.get("flow_reason"),
                "clarify_rounds": clarify_rounds,
                "code_rounds": code_result.get("rounds", 0),
                "flow_rounds": flow_result.get("rounds", 0) if isinstance(flow_result, dict) else 0,
                "clarify_history": clarify_history,
                "code_history": code_result.get("history", []),
                "flow_history": flow_result.get("history", []) if isinstance(flow_result, dict) else [],
                "qa_history": qa_history or [],
                "profile_error": profile_error,
                "stopped_reason": final_status.get("reason", "unknown"),
            }
        finally:
            set_tracker(None)

    def _run_code_phase(
        self,
        tdir: Path,
        config: ExperimentConfig,
        session_state: Dict[str, Any],
        output_root: Path,
    ) -> Dict[str, Any]:
        """Run code generation phase (shared by raw mode and interact mode).
        
        This contains the core code generation loop with retry on execution errors.
        """
        rounds_root = output_root / "rounds"
        rounds_root.mkdir(parents=True, exist_ok=True)

        coder = CodeAgent(config.model_name)
        executor = CodeExecutor()

        prev_code: Optional[str] = None
        prev_exec_error: Optional[dict] = None
        hist: list[dict] = []
        passed = False
        stopped_reason = "max_rounds"
        
        max_rounds_debug = config.max_rounds_debug
        
        input_dir = Path(session_state["input_dir"])

        for r in range(1, max_rounds_debug + 1):
            round_dir = rounds_root / f"round-{r}"
            round_dir.mkdir(parents=True, exist_ok=True)

            feedback = None
            if r > 1 and prev_exec_error and not prev_exec_error.get("ok", False):
                feedback = {
                    "prev_code": prev_code,
                    "execution": prev_exec_error
                }
            
            try:
                solve_code, raw, messages = coder.generate_code(session_state, feedback=feedback)
            except Exception as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                     raise
                (round_dir / "error.txt").write_text(f"CodeGen failed: {e}", encoding="utf-8")
                stopped_reason = "codegen_failed"
                hist.append({
                    "round": r,
                    "passed": False,
                    "solution": None,
                    "error": f"codegen_failed: {e}"
                })
                break

            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "messages.json").write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            (round_dir / "raw_response.txt").write_text(raw or "", encoding="utf-8")
            
            if not solve_code:
                round_rec = {
                    "round": r,
                    "passed": False,
                    "solution": None,
                    "error": "No code generated",
                }
                hist.append(round_rec)
                stopped_reason = "no_code_generated"
                break
            
            solution_result = self._execute_solution(
                solve_code, round_dir, input_dir, executor, timeout=config.timeout
            )
            
            passed = solution_result["execution"].get("ok", False)
            
            round_rec = {
                "round": r,
                "passed": passed,
                "solution": solution_result,
                "error": None,
            }
            hist.append(round_rec)
            
            (round_dir / "solution_summary.json").write_text(
                json.dumps({
                    "passed": passed,
                    "solution": solution_result,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            exec_ok = solution_result.get("execution", {}).get("ok", False)

            if not exec_ok:
                round_rec["error"] = "execution_failed"
                prev_code = solution_result.get("code")
                prev_exec_error = solution_result.get("execution")
                continue

            if passed:
                stopped_reason = "passed"
                break
            
            stopped_reason = ""
            break

        # Finalize
        final_solution_dst = output_root / "solution"
        final_solution_dst.mkdir(parents=True, exist_ok=True)
        status = {"ok": False, "rc": 1, "passed": False, "reason": stopped_reason, "message": "no final solution available"}
        
        if hist:
            final_solution_src = rounds_root / f"round-{len(hist)}" / "solution"
            code_src = final_solution_src / "code.py"
            if code_src.exists():
                exec_res = {}
                try:
                    exec_res = json.loads((final_solution_src / "execution.json").read_text(encoding="utf-8"))
                except Exception:
                    pass
                
                status["ok"] = exec_res.get("ok", False)
                status["rc"] = exec_res.get("rc", 1)
                status["passed"] = passed
                status["message"] = "final snapshot from last round"
                status["rounds_used"] = len(hist)
                
                copy_solution_artifacts(final_solution_src, final_solution_dst)

        if "rounds_used" not in status:
            status["rounds_used"] = len(hist)

        (final_solution_dst / "final_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        
        return {
            "passed": passed,
            "rounds": len(hist),
            "history": hist,
            "stopped_reason": stopped_reason,
        }

    def _run_profile_phase(
        self,
        *,
        tdir: Path,
        config: ExperimentConfig,
        output_root: Path,
        query_text: str,
        input_dir: Path,
        inputs: list[str],
        inputs_preview: Dict[str, Any],
    ) -> dict[str, Any]:
        return run_profile_phase(
            tdir=tdir,
            config=config,
            output_root=output_root,
            query_text=query_text,
            input_dir=input_dir,
            inputs=inputs,
            inputs_preview=inputs_preview,
        )

    def _run_profile(self, tdir: Path, config: ExperimentConfig) -> Dict[str, Any]:
        from core.data_head import DataHead
        from core.utils.paths import get_output_path
        from core.utils import list_input_files

        output_root = get_output_path(tdir, config)
        input_dir = tdir / "inputs"
        inputs_preview = DataHead().get_preview(input_dir)
        inputs = [p.name for p in list_input_files(tdir)]

        query_md_path = resolve_query_path(
            tdir,
            run_mode=config.run_mode,
        )
        if not query_md_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_md_path} (run_mode={config.run_mode})")
        query_text = query_md_path.read_text(encoding="utf-8")

        # Initialize usage tracker
        from llm_connect.usage_tracker import UsageTracker, set_tracker
        
        prompt_price, completion_price = 0.0, 0.0
        tracker = UsageTracker(
            model=config.model_name,
            prompt_price=prompt_price,
            completion_price=completion_price,
        )
        set_tracker(tracker)
        
        try:
            # ========== Profile Phase ==========
            profile_summary = ""
            profile_error = None
            reuse_profile = False
            prev_profile_usage: Optional[dict] = None
            cached_summary, cached_profile_root = self._find_profile_cache(tdir, config)
            if cached_summary is not None:
                profile_summary = cached_summary
                reuse_profile = True
                usage_root = cached_profile_root or output_root
                prev_usage = self._read_json_if_exists(usage_root / "token_usage.json")
                if isinstance(prev_usage, dict):
                    prev_profile_usage = prev_usage.get("profile")
                if usage_root != output_root:
                    self._materialize_reused_profile_summary(
                        target_root=output_root,
                        summary=cached_summary,
                        source_root=usage_root,
                    )

            if not reuse_profile:
                tracker.set_phase("profile")
                profile_result = self._run_profile_phase(
                    tdir=tdir,
                    config=config,
                    output_root=output_root,
                    query_text=query_text,
                    input_dir=input_dir,
                    inputs=inputs,
                    inputs_preview=inputs_preview,
                )
                profile_summary = profile_result.get("summary", "")
                profile_error = profile_result.get("error")

            # ========== Code Phase ==========
            tracker.set_phase("code")
            session_state = {
                "task_dir": str(tdir),
                "input_dir": str(input_dir),
                "model_name": config.model_name,
                "run_mode": config.run_mode,
                "output_root": str(output_root),
                "query": query_text,
                "profile_summary": profile_summary,
            }

            result = self._run_code_phase(
                tdir=tdir,
                config=config,
                session_state=session_state,
                output_root=output_root,
            )
            result["profile_summary"] = profile_summary
            result["profile_error"] = profile_error
            return result
        finally:
            tracker.save(output_root / "token_usage.json")
            if 'reuse_profile' in locals() and reuse_profile and prev_profile_usage:
                usage_path = output_root / "token_usage.json"
                usage = self._read_json_if_exists(usage_path)
                if isinstance(usage, dict):
                    usage["profile"] = prev_profile_usage
                    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0, "cost_usd": 0.0}
                    for key, section in usage.items():
                        if key in {"model", "total", "timestamp", "unknown_usage_calls"}:
                            continue
                        if not isinstance(section, dict):
                            continue
                        total["prompt_tokens"] += int(section.get("prompt_tokens") or 0)
                        total["completion_tokens"] += int(section.get("completion_tokens") or 0)
                        total["total_tokens"] += int(section.get("total_tokens") or 0)
                        total["call_count"] += int(section.get("call_count") or 0)
                        total["cost_usd"] += float(section.get("cost_usd") or 0.0)
                    total["cost_usd"] = round(total["cost_usd"], 6)
                    usage["total"] = total
                    usage_path.write_text(
                        json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            set_tracker(None)

    def _run_flow_impl(
        self,
        *,
        tdir: Path,
        config: ExperimentConfig,
        output_root: Path,
        solution_text: str,
    ) -> Dict[str, Any]:
        return run_flow_impl(
            tdir=tdir,
            config=config,
            output_root=output_root,
            solution_text=solution_text,
        )

    def _run_flow(self, tdir: Path, config: ExperimentConfig) -> Dict[str, Any]:
        """Run flow mode: FlowAgent generates flow.json, DAGExecutor runs it.
        
        Multi-round execution: generate -> validate -> execute.
        Retries are allowed only for flow generation/validation/execution failures.
        Correctness evaluation is deferred to offline batch evaluation.
        """
        from core.utils.paths import get_output_path

        solution_path = require_reference_solution_path(tdir)
        solution_text = solution_path.read_text(encoding="utf-8")
        output_root = get_output_path(tdir, config)

        # Initialize usage tracker
        from llm_connect.usage_tracker import UsageTracker, set_tracker

        prompt_price, completion_price = 0.0, 0.0
        tracker = UsageTracker(
            model=config.model_name,
            prompt_price=prompt_price,
            completion_price=completion_price,
        )
        set_tracker(tracker)
        tracker.set_phase("flow")

        try:
            return self._run_flow_impl(
                tdir=tdir,
                config=config,
                output_root=output_root,
                solution_text=solution_text,
            )
        finally:
            tracker.save(output_root / "token_usage.json")
            set_tracker(None)

    def _run_code_only(self, tdir: Path, config: ExperimentConfig) -> Dict[str, Any]:
        """Run code generation only (no clarify/profile/flow)."""
        output_root = get_output_path(tdir, config)
        query_md_path = resolve_query_path(tdir, run_mode=config.run_mode)
        if not query_md_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_md_path} (run_mode={config.run_mode})")

        session_state = {
            "task_dir": str(tdir),
            "input_dir": str(tdir / "inputs"),
            "model_name": config.model_name,
            "run_mode": config.run_mode,
            "output_root": str(output_root),
            "query": query_md_path.read_text(encoding="utf-8"),
        }

        from llm_connect.usage_tracker import UsageTracker, set_tracker

        tracker = UsageTracker(model=config.model_name, prompt_price=0.0, completion_price=0.0)
        set_tracker(tracker)
        try:
            tracker.set_phase("code")
            result = self._run_code_phase(
                tdir=tdir,
                config=config,
                session_state=session_state,
                output_root=output_root,
            )
            last_eval = {"passed": False}
            hist = result.get("history", [])
            if hist:
                last_round = hist[-1]
                if last_round.get("solution"):
                    last_eval = last_round["solution"].get("eval_summary", {"passed": False})
            result["last_eval"] = last_eval
            return result
        finally:
            tracker.save(output_root / "token_usage.json")
            set_tracker(None)

    def run(
        self,
        task_dir: str,
        *,
        config: Optional[ExperimentConfig] = None,
    ) -> Dict[str, Any]:
        tdir = Path(task_dir).resolve()
        if config is None:
            config = ExperimentConfig.load_config()

        mode = str(config.run_mode or "").strip().lower()
        if mode not in set(allowed_run_modes()):
            raise ValueError(f"Unsupported run_mode '{config.run_mode}'. Allowed: {sorted(allowed_run_modes())}")

        if mode == "flow":
            return self._run_flow(tdir, config)
        if mode in {"orig", "disamb"}:
            return self._run_profile(tdir, config)
        if mode == "interact":
            return self._run_interact(tdir, config)
        if mode == "e2e":
            return self._run_e2e(tdir, config, with_flow=True)
        return self._run_code_only(tdir, config)
