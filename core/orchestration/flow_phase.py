from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agents.flow_agent import FlowAgent
from config.experiment_config import ExperimentConfig
from core.orchestration.common import copy_solution_artifacts, summarize_eval


def run_flow_impl(
    *,
    tdir: Path,
    config: ExperimentConfig,
    output_root: Path,
    solution_text: str,
) -> Dict[str, Any]:
    """Shared flow execution logic for flow mode and e2e."""
    from py2flow.errors import FlowExecutionError, FlowValidationError
    from py2flow.executor import DAGExecutor, DebugConfig
    from py2flow.flow_constraints import validate_script_constraints
    from py2flow.ir import DAG

    input_dir = tdir / "inputs"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"[flow mode] inputs/ directory not found: {tdir}")
    if not isinstance(solution_text, str) or not solution_text.strip():
        raise FileNotFoundError("[flow mode] solution.py content is empty")

    cfg = config
    output_root.mkdir(parents=True, exist_ok=True)
    rounds_root = output_root / "rounds"
    if rounds_root.exists():
        shutil.rmtree(rounds_root)

    session_state = {
        "task_dir": str(tdir),
        "input_dir": str(input_dir),
        "model_name": cfg.model_name,
        "run_mode": cfg.run_mode,
        "output_root": str(output_root),
        "case_name": tdir.name,
        "solution_text": solution_text,
    }

    max_rounds = getattr(cfg, "max_rounds_debug", 3)
    flow_agent = FlowAgent(model_name=cfg.model_name)

    feedback = None
    stopped_reason = "unknown"
    passed = False
    exec_info = {"ok": False, "rc": 1, "stderr": "", "stdout": "", "took_sec": 0}
    eval_report = {"passed": False, "errors": [], "diff_summary": {}}
    final_solution_dir: Optional[Path] = None
    hist: list[dict[str, Any]] = []

    def _write_solution_artifacts(solution_dir: Path, exec_data: dict, eval_data: dict) -> None:
        (solution_dir / "execution.json").write_text(
            json.dumps(exec_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (solution_dir / "evaluation.json").write_text(
            json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for round_num in range(1, max_rounds + 1):
        round_dir = rounds_root / f"round-{round_num}"
        if round_dir.exists():
            shutil.rmtree(round_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        solution_dir = round_dir / "solution"
        solution_dir.mkdir(parents=True, exist_ok=True)
        final_solution_dir = solution_dir

        protocol = {
            "benchmark": "code_to_flow",
            "agent_input": ["solution.py"],
            "agent_input_excludes": ["query.md", "query_full.md", "inputs_preview", "gt/allowed_outputs"],
            "feedback_enabled_for": ["json_parse", "validation", "execution", "no_outputs"],
            "evaluation_feedback_to_agent": False,
            "stop_on_execution_success": True,
            "case": tdir.name,
            "model_under_test": cfg.model_name,
            "max_rounds": max_rounds,
            "round": round_num,
        }
        (round_dir / "protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            flow_dict, raw_response, messages, parse_error = flow_agent.generate_flow(
                session_state, feedback=feedback
            )
        except Exception as e:
            stopped_reason = "execerror"
            exec_info = {
                "ok": False,
                "rc": 1,
                "stderr": f"Flow generation exception: {e}",
                "stdout": "",
                "took_sec": 0,
            }
            eval_report = {
                "passed": False,
                "errors": [{"error_type": "flowgen_exception", "message": str(e)}],
                "diff_summary": {},
            }
            (round_dir / "flow_generation_exception.json").write_text(
                json.dumps({"type": "generation_exception", "message": str(e)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            feedback = {"type": "generation_exception", "message": str(e), "details": {}}
            hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
            continue
        (round_dir / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (round_dir / "raw_response.txt").write_text(raw_response or "", encoding="utf-8")

        if parse_error:
            (round_dir / "flow_parse_error.json").write_text(
                json.dumps(parse_error, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            stopped_reason = "execerror"
            exec_info = {
                "ok": False,
                "rc": 1,
                "stderr": f"Flow generation failed: {parse_error.get('message', '')}",
                "stdout": "",
                "took_sec": 0,
            }
            eval_report = {
                "passed": False,
                "errors": [{"error_type": "flowgen_failed", "message": exec_info["stderr"], "details": parse_error}],
                "diff_summary": {},
            }
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            feedback = {
                "type": parse_error.get("type", "json_parse"),
                "message": parse_error.get("message", ""),
                "details": parse_error.get("details", {}),
            }
            hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
            continue

        (round_dir / "flow.json").write_text(
            json.dumps(flow_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (solution_dir / "flow.json").write_text(
            json.dumps(flow_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        try:
            dag = DAG.from_dict(flow_dict)

            is_valid, constraint_err, err_details = validate_script_constraints(flow_dict)
            if not is_valid:
                stopped_reason = "execerror"
                exec_info = {"ok": False, "rc": 1, "stderr": constraint_err, "stdout": "", "took_sec": 0}
                (round_dir / "flow_constraint_violation.json").write_text(
                    json.dumps(err_details, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                feedback = {"type": "validation", "message": constraint_err, "details": err_details}
                eval_report = {
                    "passed": False,
                    "errors": [
                        {
                            "error_type": "script_constraint_violation",
                            "message": constraint_err,
                            "details": err_details,
                        }
                    ],
                    "diff_summary": {},
                }
                _write_solution_artifacts(solution_dir, exec_info, eval_report)
                hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
                continue

            debug_cfg = DebugConfig(on_fail_dump=True)
            inputs_link = solution_dir / "inputs"
            if not inputs_link.exists():
                try:
                    inputs_link.symlink_to(tdir / "inputs")
                except OSError:
                    shutil.copytree(tdir / "inputs", inputs_link)

            executor = DAGExecutor(dag, base_path=solution_dir, debug=debug_cfg)
            start_time = time.time()
            executor.run(keep="none")
            took_sec = time.time() - start_time
            exec_info = {"ok": True, "rc": 0, "stderr": "", "stdout": "", "took_sec": took_sec}

            cand_dir = solution_dir / "flow_cand"
            output_files = sorted(cand_dir.glob("*.csv")) if cand_dir.exists() else []
            if not cand_dir.exists() or not output_files:
                stopped_reason = "execerror"
                exec_info["stderr"] = "No flow CSV outputs produced under flow_cand."
                exec_info["ok"] = False
                exec_info["rc"] = 1
                eval_report = {
                    "passed": False,
                    "errors": [{"error_type": "no_outputs", "message": exec_info["stderr"]}],
                    "diff_summary": {},
                }
                _write_solution_artifacts(solution_dir, exec_info, eval_report)
                feedback = {
                    "type": "execution",
                    "message": "No output files were created. Ensure all Output nodes write CSV files to flow_cand/.",
                    "details": {},
                }
                hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
                continue

            passed = True
            eval_report = {
                "passed": True,
                "errors": [],
                "diff_summary": {},
                "note": "evaluation_skipped_in_run_phase",
            }
            stopped_reason = "execok"
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            hist.append({"round": round_num, "passed": True, "stopped_reason": stopped_reason})
            break

        except FlowValidationError as e:
            stopped_reason = "execerror"
            exec_info = {"ok": False, "rc": 1, "stderr": str(e), "stdout": "", "took_sec": 0}
            (round_dir / "flow_parse_error.json").write_text(
                json.dumps(
                    {
                        "type": "validation",
                        "message": str(e),
                        "details": {
                            "error_code": getattr(e, "error_code", None),
                            "node_id": getattr(e, "node_id", None),
                            "step_kind": str(getattr(e, "step_kind", "")) if hasattr(e, "step_kind") else None,
                            "field": getattr(e, "field", None),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            feedback = {
                "type": "validation",
                "message": str(e),
                "details": {"error_code": getattr(e, "error_code", None), "node_id": getattr(e, "node_id", None)},
            }
            eval_report = {
                "passed": False,
                "errors": [{"error_type": "flowgen_failed", "message": str(e)}],
                "diff_summary": {},
            }
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
            continue

        except FlowExecutionError as e:
            stopped_reason = "execerror"
            exec_info = {
                "ok": False,
                "rc": 1,
                "stderr": str(e),
                "stdout": "",
                "took_sec": 0,
                "failed_node": getattr(e, "node_id", None),
            }
            feedback = {
                "type": "execution",
                "message": str(e),
                "details": {"node_id": getattr(e, "node_id", None), "error": str(e)},
            }
            eval_report = {
                "passed": False,
                "errors": [{"error_type": "execution_failed", "message": str(e)}],
                "diff_summary": {},
            }
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
            continue

        except Exception as e:
            stopped_reason = "execerror"
            exec_info = {"ok": False, "rc": 1, "stderr": str(e), "stdout": "", "took_sec": 0}
            feedback = {"type": "execution", "message": str(e), "details": {}}
            eval_report = {
                "passed": False,
                "errors": [{"error_type": "execution_failed", "message": str(e)}],
                "diff_summary": {},
            }
            _write_solution_artifacts(solution_dir, exec_info, eval_report)
            hist.append({"round": round_num, "passed": False, "stopped_reason": stopped_reason})
            continue

    if final_solution_dir:
        (final_solution_dir / "execution.json").write_text(
            json.dumps(exec_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (final_solution_dir / "evaluation.json").write_text(
            json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = summarize_eval(eval_report)
        (final_solution_dir / "solution_summary.json").write_text(
            json.dumps({"passed": passed, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    final_solution_dst = output_root / "solution"
    final_solution_dst.mkdir(parents=True, exist_ok=True)
    if final_solution_dir:
        copy_solution_artifacts(final_solution_dir, final_solution_dst)

    final_ok = exec_info.get("ok", False)
    final_status = {
        "ok": final_ok,
        "rc": exec_info.get("rc", 1),
        "passed": passed,
        "reason": stopped_reason,
        "rounds_used": len(hist),
        "message": "" if final_ok else exec_info.get("stderr", ""),
    }
    (final_solution_dst / "final_status.json").write_text(
        json.dumps(final_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "passed": passed,
        "rounds": len(hist),
        "history": hist,
        "stopped_reason": stopped_reason,
    }
