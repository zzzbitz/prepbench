from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypeVar
from uuid import uuid4

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agents.clarify_agent import ClarifyAgent
from config.experiment_config import ExperimentConfig
from core.case_views import load_internal_case_view
from core.data_head import DataHead
from core.orchestration.clarify_parse import parse_sub_questions
from core.orchestration.code_phase import run_code_phase
from core.orchestration.flow_phase import run_flow_impl
from core.orchestration.profile_phase import run_profile_phase
from core.utils import list_input_files
from core.utils.logging_config import configure_logging, get_logger
from core.utils.paths import get_output_path
from simulator.local_api import LocalUserSimulatorAPI

T = TypeVar("T")


def dedupe_preserve_order(items: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    out: list[T] = []
    for item in items:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def resolve_cases(case_arg: str, data_dir: Path) -> List[Path]:
    if not case_arg:
        return sorted([p for p in data_dir.glob("case_*") if p.is_dir()])

    case_arg = case_arg.strip()
    if case_arg.startswith("case_"):
        p = data_dir / case_arg
        if not p.exists():
            raise FileNotFoundError(f"Case directory not found: {p}")
        return [p]

    maybe_path = Path(case_arg)
    if ("/" in case_arg or case_arg.startswith(".")) and maybe_path.exists():
        p = maybe_path.resolve()
        if not p.is_dir():
            raise NotADirectoryError(f"--case path is not a directory: {p}")
        return [p]

    if "," in case_arg:
        tokens = [t.strip() for t in case_arg.split(",") if t.strip()]
        if not tokens:
            raise ValueError("Comma list is empty")
        result: List[Path] = []
        for token in tokens:
            if not token.isdigit():
                raise ValueError(f"Invalid case token in list: {token}")
            p = data_dir / f"case_{int(token):03d}"
            if p.exists():
                result.append(p)
        result = dedupe_preserve_order(result)
        if not result:
            raise FileNotFoundError(f"No valid cases in list: {case_arg}")
        return result

    if "-" in case_arg:
        parts = case_arg.split("-")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Invalid range format: {case_arg}")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            start, end = end, start
        result = [p for i in range(start, end + 1) for p in [data_dir / f"case_{i:03d}"] if p.exists()]
        if not result:
            raise FileNotFoundError(f"No valid cases in range: {case_arg}")
        return result

    if case_arg.isdigit():
        p = data_dir / f"case_{int(case_arg):03d}"
        if not p.exists():
            raise FileNotFoundError(f"Case directory not found: {p}")
        return [p]

    raise ValueError(f"Invalid --case format: {case_arg}")


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_max_questions(config: ExperimentConfig, amb_kb_json: Dict[str, Any]) -> int:
    ambiguities = amb_kb_json.get("ambiguities")
    amb_count = len(ambiguities) if isinstance(ambiguities, list) else 0

    if config.question_ratio is None:
        default_budget = config.max_questions_cap if config.max_questions_cap is not None else 25
        return max(1, int(default_budget))

    budget = max(1, math.ceil(float(config.question_ratio) * amb_count))
    if config.max_questions_cap is not None:
        effective_cap = max(int(config.max_questions_cap), amb_count)
        budget = min(budget, effective_cap)
    return max(1, budget)


def copy_flow_artifacts(flow_solution_dir: Path, final_solution_dir: Path) -> None:
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


def collect_solution_text(output_root: Path, code_result: Dict[str, Any]) -> str:
    solution_path = output_root / "solution" / "solution.py"
    if solution_path.exists():
        return solution_path.read_text(encoding="utf-8")

    history = code_result.get("history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            solution = last.get("solution")
            if isinstance(solution, dict):
                code = solution.get("code")
                if isinstance(code, str):
                    return code
    return ""


class PrepAgentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.logger = get_logger("prep_agent")

    def _run_profile_stage(
        self,
        *,
        case_dir: Path,
        output_root: Path,
        query_text: str,
        input_dir: Path,
        inputs_preview: Dict[str, Any],
    ) -> Dict[str, Any]:
        inputs = [p.name for p in list_input_files(case_dir)]
        return run_profile_phase(
            tdir=case_dir,
            config=self.config,
            output_root=output_root,
            query_text=query_text,
            input_dir=input_dir,
            inputs=inputs,
            inputs_preview=inputs_preview,
        )

    def _run_clarify_stage(
        self,
        *,
        case_dir: Path,
        output_root: Path,
        query_text: str,
        inputs_preview: Dict[str, Any],
        amb_kb_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        clarify_root = output_root / "clarify"
        clarify_root.mkdir(parents=True, exist_ok=True)

        question_budget = compute_max_questions(self.config, amb_kb_json)
        run_id = f"prepagent_{uuid4().hex[:10]}"

        simulator_api = LocalUserSimulatorAPI(
            max_rounds=self.config.max_rounds_interact,
            max_questions=question_budget,
            max_questions_per_ask=self.config.max_questions_per_ask,
        )
        session = simulator_api.start_session(case_id=case_dir.name, run_id=run_id)

        clarifier = ClarifyAgent(self.config.model_name)
        session_state = {
            "task_dir": str(case_dir),
            "query": query_text,
            "inputs_preview": inputs_preview,
        }

        qa_history: list[Dict[str, Any]] = []
        questions_used = 0
        ask_round = 1
        runtime_feedback: Optional[Dict[str, Any]] = None
        stopped_reason = "max_rounds"
        rounds_history: list[Dict[str, Any]] = []

        for attempt in range(1, self.config.max_rounds_interact + 1):
            round_dir = clarify_root / f"round-{attempt}"
            round_dir.mkdir(parents=True, exist_ok=True)

            qa_for_prompt = [
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "classification": item.get("classification"),
                    "source": item.get("source"),
                    "ref": item.get("ref"),
                }
                for item in qa_history
            ]

            action, raw, messages = clarifier.generate_action(
                session_state,
                qa_history=qa_for_prompt,
                runtime_feedback=runtime_feedback,
                code_started=False,
                max_questions=question_budget,
                max_questions_per_ask=self.config.max_questions_per_ask,
                questions_used=questions_used,
            )

            write_json(round_dir / "messages.json", {"messages": messages})
            (round_dir / "raw_response.txt").write_text(raw or "", encoding="utf-8")
            write_json(
                round_dir / "action.json",
                {
                    "attempt": attempt,
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "action": action.action_type,
                    "content": action.content,
                    "parse_error": action.parse_error,
                    "questions_used": questions_used,
                    "question_budget": question_budget,
                },
            )

            if action.action_type == "done":
                rounds_history.append({"attempt": attempt, "action": "done"})
                stopped_reason = "done"
                break

            if action.action_type != "ask":
                runtime_feedback = {
                    "parse_error": action.parse_error or "unrecognized_action",
                    "hint": "Response must start with Ask: or Done:.",
                }
                rounds_history.append(
                    {
                        "attempt": attempt,
                        "action": "invalid",
                        "parse_error": action.parse_error,
                    }
                )
                continue

            sub_questions, parse_method = parse_sub_questions(action.content)
            sub_questions = [q for q in sub_questions if isinstance(q, str) and q.strip()]

            if not sub_questions:
                runtime_feedback = {
                    "parse_error": "empty_ask_questions",
                    "hint": "At least one concrete question is required.",
                }
                rounds_history.append(
                    {
                        "attempt": attempt,
                        "action": "invalid",
                        "parse_error": "empty_ask_questions",
                    }
                )
                continue

            if len(sub_questions) > self.config.max_questions_per_ask:
                sub_questions = sub_questions[: self.config.max_questions_per_ask]

            remaining = max(question_budget - questions_used, 0)
            if remaining <= 0:
                rounds_history.append(
                    {
                        "attempt": attempt,
                        "action": "ask_refused",
                        "reason": "max_questions",
                    }
                )
                stopped_reason = "max_questions"
                break
            if len(sub_questions) > remaining:
                sub_questions = sub_questions[:remaining]

            write_json(
                round_dir / "ask_parsed.json",
                {
                    "questions": sub_questions,
                    "method": parse_method,
                    "count": len(sub_questions),
                    "simulator_round": ask_round,
                },
            )

            try:
                simulator_resp = simulator_api.ask(
                    session_id=session["session_id"],
                    questions=sub_questions,
                    round=ask_round,
                )
            except Exception as exc:
                err_text = f"simulator_call_failed: {exc}"
                (round_dir / "simulator_error.txt").write_text(err_text, encoding="utf-8")
                runtime_feedback = {
                    "parse_error": "simulator_call_failed",
                    "hint": "Try fewer and more specific questions.",
                }
                rounds_history.append(
                    {
                        "attempt": attempt,
                        "action": "ask_failed",
                        "error": err_text,
                    }
                )
                stopped_reason = "simulator_error"
                break

            ask_round += 1
            write_json(round_dir / "user_simulator.json", simulator_resp)

            answers = simulator_resp.get("answers")
            normalized_answers: list[Dict[str, Any]] = []
            if isinstance(answers, list):
                for ans in answers:
                    if not isinstance(ans, dict):
                        continue
                    sub_q = str(ans.get("sub_question") or "").strip()
                    if not sub_q:
                        continue
                    rec = {
                        "sub_question": sub_q,
                        "classification": ans.get("classification"),
                        "source": ans.get("source"),
                        "answer": str(ans.get("answer") or ""),
                        "ref": ans.get("ref"),
                    }
                    normalized_answers.append(rec)
                    qa_history.append(
                        {
                            "question": rec["sub_question"],
                            "answer": rec["answer"],
                            "classification": rec["classification"],
                            "source": rec["source"],
                            "ref": rec["ref"],
                            "slot_id": rec["ref"],
                        }
                    )

            questions_used += len(normalized_answers)
            done = bool(simulator_resp.get("done", False))

            rounds_history.append(
                {
                    "attempt": attempt,
                    "action": "ask",
                    "simulator_round": ask_round - 1,
                    "questions": sub_questions,
                    "answers": normalized_answers,
                    "done": done,
                }
            )

            runtime_feedback = None
            if done:
                stopped_reason = "budget_exhausted"
                break

        hit_count = sum(1 for item in qa_history if item.get("classification") == "hit")
        refuse_count = sum(
            1 for item in qa_history if str(item.get("classification") or "").startswith("refuse")
        )
        asked_count = len(qa_history)
        answered_count = asked_count - refuse_count
        hit_rate = (hit_count / asked_count) if asked_count else 0.0

        summary = {
            "run_id": run_id,
            "session": session,
            "stopped_reason": stopped_reason,
            "rounds": len(rounds_history),
            "questions_used": questions_used,
            "question_budget": question_budget,
            "asked_count": asked_count,
            "answered_count": answered_count,
            "refuse_count": refuse_count,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "qa_history": qa_history,
        }
        write_json(clarify_root / "summary.json", summary)

        return {
            "qa_history": qa_history,
            "history": rounds_history,
            "stopped_reason": stopped_reason,
            "questions_used": questions_used,
            "question_budget": question_budget,
        }

    def run_case(self, case_dir: Path) -> Dict[str, Any]:
        case_view = load_internal_case_view(case_dir, require_reference_solution=True)
        output_root = get_output_path(case_dir, self.config)
        output_root.mkdir(parents=True, exist_ok=True)

        inputs_preview = DataHead().get_preview(case_view.input_dir)

        profile_result = self._run_profile_stage(
            case_dir=case_dir,
            output_root=output_root,
            query_text=case_view.query_text,
            input_dir=case_view.input_dir,
            inputs_preview=inputs_preview,
        )
        profile_summary = str(profile_result.get("summary") or "")

        clarify_result = self._run_clarify_stage(
            case_dir=case_dir,
            output_root=output_root,
            query_text=case_view.query_text,
            inputs_preview=inputs_preview,
            amb_kb_json=case_view.amb_kb_json,
        )

        code_session_state = {
            "task_dir": str(case_dir),
            "input_dir": str(case_view.input_dir),
            "model_name": self.config.model_name,
            "run_mode": self.config.run_mode,
            "output_root": str(output_root),
            "query": case_view.query_text,
            "qa_history": clarify_result.get("qa_history", []),
            "profile_summary": profile_summary,
        }
        code_result = run_code_phase(
            tdir=case_dir,
            config=self.config,
            session_state=code_session_state,
            output_root=output_root,
        )

        solution_dir = output_root / "solution"
        solution_dir.mkdir(parents=True, exist_ok=True)

        code_status = read_json_if_exists(solution_dir / "final_status.json") or {}
        code_ok = bool(code_status.get("ok", False))
        code_passed = bool(code_status.get("passed", False))
        code_reason = str(code_status.get("reason") or code_result.get("stopped_reason") or "unknown")
        code_rc = int(code_status.get("rc", 1))

        cand_dir = solution_dir / "cand"
        has_cand_outputs = cand_dir.exists() and any(cand_dir.glob("*.csv"))

        flow_ran = False
        flow_passed = False
        flow_ok = False
        flow_reason = "skipped_code_failed"
        flow_rc = 1
        flow_result: Dict[str, Any] = {}

        if code_ok and has_cand_outputs:
            solution_text = collect_solution_text(output_root, code_result)
            if solution_text.strip():
                flow_output_root = output_root / "flow"
                flow_result = run_flow_impl(
                    tdir=case_dir,
                    config=self.config,
                    output_root=flow_output_root,
                    solution_text=solution_text,
                )
                flow_ran = True
                flow_passed = bool(flow_result.get("passed", False))
                flow_reason = str(flow_result.get("stopped_reason") or "unknown")

                flow_status = read_json_if_exists(flow_output_root / "solution" / "final_status.json") or {}
                flow_ok = bool(flow_status.get("ok", False))
                flow_rc = int(flow_status.get("rc", 1))

                copy_flow_artifacts(flow_output_root / "solution", solution_dir)
            else:
                flow_reason = "solution_missing"
        elif code_ok and not has_cand_outputs:
            flow_reason = "no_code_outputs"

        overall_passed = code_passed and flow_passed
        if overall_passed:
            overall_reason = "passed"
        elif not code_passed:
            overall_reason = "code_failed"
        elif not flow_ran:
            overall_reason = "flow_skipped"
        else:
            overall_reason = "flow_failed"

        final_status = {
            "ok": bool(code_ok and flow_ok),
            "rc": flow_rc if flow_ran else code_rc,
            "passed": overall_passed,
            "reason": overall_reason,
            "message": str(code_status.get("message") or ""),
            "rounds_used": int(code_status.get("rounds_used", code_result.get("rounds", 0))),
            "code_passed": code_passed,
            "code_reason": code_reason,
            "flow_passed": flow_passed,
            "flow_reason": flow_reason,
            "gui_passed": flow_passed,
            "gui_reason": flow_reason,
            "pipeline": "prepagent_e2e",
            "profile_error": profile_result.get("error"),
        }
        write_json(solution_dir / "final_status.json", final_status)

        return {
            "case": case_dir.name,
            "output_root": str(output_root),
            "profile_error": profile_result.get("error"),
            "clarify_rounds": len(clarify_result.get("history", [])),
            "code_rounds": code_result.get("rounds", 0),
            "flow_rounds": flow_result.get("rounds", 0) if flow_ran else 0,
            "passed": overall_passed,
            "reason": overall_reason,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrepAgent reference pipeline (e2e only)")
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="Case ID (e.g., 1), range (e.g., 1-10), comma list (e.g., 1,3,5), case_xxx, or path. Empty means all cases.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name. If omitted, use .env/config defaults.",
    )
    parser.add_argument(
        "--list",
        type=int,
        default=0,
        help="List first N selected cases and exit.",
    )
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("prep_agent")

    data_dir = REPO_ROOT / "data"

    try:
        selected_cases = resolve_cases(args.case, data_dir)
    except Exception as exc:
        logger.error(f"Failed to resolve cases: {exc}")
        sys.exit(1)

    if args.list and args.list > 0:
        limit = min(args.list, len(selected_cases))
        print(f"Listing first {limit} selected cases:")
        for idx, case in enumerate(selected_cases[:limit], start=1):
            print(f"{idx:03d} {case.name}")
        return

    model_override = args.model.strip() or None
    try:
        base_config = ExperimentConfig.load_config(model_name_override=model_override)
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        sys.exit(1)

    config = dataclasses.replace(base_config, run_mode="e2e")

    logger.info(
        "PrepAgent configuration: model=%s run_mode=%s cases=%d timeout=%s",
        config.model_name,
        config.run_mode,
        len(selected_cases),
        config.timeout,
    )

    runner = PrepAgentRunner(config)

    results: list[Dict[str, Any]] = []
    failed = 0

    for case_dir in tqdm(selected_cases, desc="PrepAgent", unit="case", ascii=True, dynamic_ncols=True):
        try:
            result = runner.run_case(case_dir)
            results.append(result)
            if not result.get("passed", False):
                failed += 1
        except Exception as exc:
            failed += 1
            logger.error("[%s] failed: %s", case_dir.name, exc)
            output_root = get_output_path(case_dir, config)
            solution_dir = output_root / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                solution_dir / "final_status.json",
                {
                    "ok": False,
                    "rc": 1,
                    "passed": False,
                    "reason": "pipeline_exception",
                    "message": str(exc),
                    "code_passed": False,
                    "flow_passed": False,
                    "flow_reason": "skipped_exception",
                    "pipeline": "prepagent_e2e",
                },
            )
            results.append(
                {
                    "case": case_dir.name,
                    "output_root": str(output_root),
                    "passed": False,
                    "reason": f"exception: {exc}",
                }
            )

    print("\n== PrepAgent Summary ==")
    print(f"TotalCases: {len(results)}")
    print(f"Passed: {len(results) - failed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
