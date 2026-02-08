from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agents.code_agent import CodeAgent
from config.experiment_config import ExperimentConfig
from core.executor import CodeExecutor
from core.orchestration.common import copy_solution_artifacts
from core.utils.exec_utils import render_main_with_solve


def execute_solution(
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
    except Exception as exc:
        exec_info = {"rc": 1, "stderr": str(exc), "ok": False}
        (solution_dir / "execution.json").write_text(
            json.dumps(exec_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "code": solve_source,
            "code_path": str((solution_dir / "code.py").resolve()),
            "cand_dir": str((solution_dir / "cand").resolve()),
            "execution": exec_info,
        }

    (solution_dir / "code.py").write_text(solve_source, encoding="utf-8")
    code_path = solution_dir / "main.py"
    code_path.write_text(main_code, encoding="utf-8")

    input_files = {p.name: p for p in sorted(input_dir.glob("*.csv"))}
    ok, stderr, stdout, _ = executor.execute_code(
        main_code, input_files, timeout=timeout, work_dir=solution_dir
    )
    exec_info = {"rc": 0 if ok else 1, "stderr": stderr, "stdout": stdout, "ok": ok}
    (solution_dir / "execution.json").write_text(
        json.dumps(exec_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cand_dir = solution_dir / "cand"
    return {
        "code": solve_source,
        "code_path": str(code_path.resolve()),
        "cand_dir": str(cand_dir.resolve()),
        "execution": exec_info,
    }


def run_code_phase(
    *,
    tdir: Path,
    config: ExperimentConfig,
    session_state: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    """Run code generation phase with retry on execution errors."""
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

    for round_idx in range(1, max_rounds_debug + 1):
        round_dir = rounds_root / f"round-{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        feedback = None
        if round_idx > 1 and prev_exec_error and not prev_exec_error.get("ok", False):
            feedback = {"prev_code": prev_code, "execution": prev_exec_error}

        try:
            solve_code, raw, messages = coder.generate_code(session_state, feedback=feedback)
        except Exception as exc:
            if "timeout" in str(exc).lower() or "connection" in str(exc).lower():
                raise
            (round_dir / "error.txt").write_text(f"CodeGen failed: {exc}", encoding="utf-8")
            stopped_reason = "codegen_failed"
            hist.append(
                {
                    "round": round_idx,
                    "passed": False,
                    "solution": None,
                    "error": f"codegen_failed: {exc}",
                }
            )
            break

        (round_dir / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (round_dir / "raw_response.txt").write_text(raw or "", encoding="utf-8")

        if not solve_code:
            hist.append(
                {
                    "round": round_idx,
                    "passed": False,
                    "solution": None,
                    "error": "No code generated",
                }
            )
            stopped_reason = "no_code_generated"
            break

        solution_result = execute_solution(
            solve_code, round_dir, input_dir, executor, timeout=config.timeout
        )
        passed = bool(solution_result["execution"].get("ok", False))

        round_rec = {
            "round": round_idx,
            "passed": passed,
            "solution": solution_result,
            "error": None,
        }
        hist.append(round_rec)

        (round_dir / "solution_summary.json").write_text(
            json.dumps({"passed": passed, "solution": solution_result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        exec_ok = bool(solution_result.get("execution", {}).get("ok", False))
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

    final_solution_dst = output_root / "solution"
    final_solution_dst.mkdir(parents=True, exist_ok=True)
    status = {
        "ok": False,
        "rc": 1,
        "passed": False,
        "reason": stopped_reason,
        "message": "no final solution available",
    }

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

    (final_solution_dst / "final_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "passed": passed,
        "rounds": len(hist),
        "history": hist,
        "stopped_reason": stopped_reason,
    }
