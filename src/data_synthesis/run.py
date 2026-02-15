#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from core.case_assets import require_reference_solution_path

from . import config as ds_config
from .disamb_builder import DisambBuildOptions, DisambBuilder
from .gt_codegen import GTCodeBuildOptions, GTCodeBuilder
from .pipeline_common import build_repo_paths, list_cases, normalize_model_dir, read_text
from .synthesizer import Synthesizer
from .workflow_builder import WorkflowBuildOptions, WorkflowBuilder


def _looks_like_no_ambiguity(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = (
        "no ambiguity",
        "no ambiguities",
        "no-ambiguity",
        "no change",
        "unchanged",
    )
    return any(k in t for k in keywords) and len(t) <= 120


def _read_optional_flow(case_dir: Path) -> str:
    for candidate in (case_dir / "flow.json", case_dir / "flow_compressed.json"):
        if candidate.exists():
            return read_text(candidate)
    return "{}"


def _mode_case_output_root(output_root: Path, mode: str, model_name: str, case_name: str) -> Path:
    return output_root / mode / normalize_model_dir(model_name) / case_name


def _run_parallel(
    fn: Callable[[str], dict[str, Any]],
    cases: list[str],
    jobs: int,
) -> list[dict[str, Any]]:
    if jobs <= 1:
        return [fn(case_name) for case_name in cases]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_case = {executor.submit(fn, case_name): case_name for case_name in cases}
        for future in as_completed(future_to_case):
            case_name = future_to_case[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"case": case_name, "passed": False, "reason": f"unhandled_exception: {exc}"})
    return results


def _print_summary(mode: str, model: str, results: list[dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if bool(r.get("passed")))
    failed = total - passed
    print(f"[{mode}] model={model} total={total} passed={passed} failed={failed}")
    if failed:
        for item in sorted(results, key=lambda x: str(x.get("case", ""))):
            if item.get("passed"):
                continue
            print(f"  - {item.get('case')}: {item.get('reason')}")


def _run_full_case(
    case_name: str,
    model_name: str,
    data_root: Path,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    case_dir = data_root / case_name
    query_path = case_dir / "query.md"
    if not query_path.exists():
        return {"case": case_name, "passed": False, "reason": "missing_query.md"}

    try:
        solution_path = require_reference_solution_path(case_dir)
    except Exception as exc:
        return {"case": case_name, "passed": False, "reason": f"missing_reference_solution: {exc}"}

    query_text = read_text(query_path)
    solution_text = read_text(solution_path)
    flow_text = _read_optional_flow(case_dir)
    case_out = _mode_case_output_root(output_root, "full", model_name, case_name)
    case_out.mkdir(parents=True, exist_ok=True)
    query_full_path = case_out / "query_full.md"
    if query_full_path.exists() and not force:
        return {"case": case_name, "passed": True, "reason": "skipped_existing_output_query_full"}

    syn = Synthesizer(model_full=model_name)
    generated = syn.generate_query_full(query_text, solution_text, flow_text)
    final_query_full = query_text if _looks_like_no_ambiguity(generated) else generated
    query_full_path.write_text(final_query_full, encoding="utf-8")
    return {"case": case_name, "passed": True, "reason": f"written_output_query_full: {query_full_path}"}


def _run_amb_case(
    case_name: str,
    model_name: str,
    data_root: Path,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    case_dir = data_root / case_name
    query_path = case_dir / "query.md"
    if not query_path.exists():
        return {"case": case_name, "passed": False, "reason": "missing_query.md"}

    case_out = _mode_case_output_root(output_root, "amb", model_name, case_name)
    case_out.mkdir(parents=True, exist_ok=True)
    query_full_path = case_out / "query_full.md"
    amb_path = case_out / "amb_kb.json"
    if amb_path.exists() and not force:
        return {"case": case_name, "passed": True, "reason": "skipped_existing_output_amb_kb.json"}

    try:
        solution_path = require_reference_solution_path(case_dir)
    except Exception as exc:
        return {"case": case_name, "passed": False, "reason": f"missing_reference_solution: {exc}"}

    query_text = read_text(query_path)
    solution_text = read_text(solution_path)
    flow_text = _read_optional_flow(case_dir)

    syn = Synthesizer(model_full=model_name)
    if not query_full_path.exists() or not query_full_path.read_text(encoding="utf-8").strip():
        generated = syn.generate_query_full(query_text, solution_text, flow_text)
        final_query_full = query_text if _looks_like_no_ambiguity(generated) else generated
        query_full_path.write_text(final_query_full, encoding="utf-8")

    query_full_text = query_full_path.read_text(encoding="utf-8")
    raw_amb = syn.generate_ambiguities_raw(query_text, solution_text, query_full_text, flow_text)
    amb_path.write_text(raw_amb, encoding="utf-8")
    return {"case": case_name, "passed": True, "reason": f"written_output_amb_kb: {amb_path}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Data synthesis and benchmark asset builders.")
    parser.add_argument(
        "mode",
        choices=["full", "amb", "build_gt_code", "build_disamb", "build_flow", "build_all"],
        help="Execution mode.",
    )
    parser.add_argument("--case", type=str, default="", help="Case selector: 1 | 1,2,3 | 1-5 | case_001")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs.")
    parser.add_argument("--model", type=str, default="", help="Model name. Defaults to config FULL default.")
    parser.add_argument("--force", action="store_true", help="Force overwrite previous outputs.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max rounds for GT code/workflow builders.")
    parser.add_argument("--max-rewrite-rounds", type=int, default=3, help="Max rewrite rounds for disamb builder.")
    parser.add_argument("--max-validation-rounds", type=int, default=3, help="Max code validation rounds per rewrite.")
    parser.add_argument("--timeout", type=int, default=120, help="Execution timeout (seconds) for code validation.")
    parser.add_argument("--publish-solution", action="store_true", help="Publish passed GT code to simulator assets.")
    parser.add_argument("--publish-flow", action="store_true", help="Publish passed flow to data/case_xxx/flow.json.")
    args = parser.parse_args()

    if args.jobs <= 0:
        print("[ERROR] --jobs must be a positive integer")
        return 1
    if args.max_rounds <= 0 or args.max_rewrite_rounds <= 0 or args.max_validation_rounds <= 0:
        print("[ERROR] round limits must be positive integers")
        return 1
    if args.timeout <= 0:
        print("[ERROR] --timeout must be positive")
        return 1

    repo_paths = build_repo_paths()
    if not repo_paths.data_root.is_dir():
        print(f"[ERROR] data root not found: {repo_paths.data_root}")
        return 1

    cases = list_cases(repo_paths.data_root, args.case)
    if not cases:
        print("[ERROR] no valid cases selected")
        return 1

    model_name = args.model.strip() if args.model.strip() else ds_config.get_full_default_model()
    print(
        f"[PLAN] mode={args.mode} model={model_name} cases={len(cases)} jobs={args.jobs} "
        f"force={args.force} output_model_dir={normalize_model_dir(model_name)}"
    )

    if args.mode == "full":
        fn = partial(
            _run_full_case,
            model_name=model_name,
            data_root=repo_paths.data_root,
            output_root=repo_paths.output_root,
            force=args.force,
        )
        results = _run_parallel(fn, cases, args.jobs)
        _print_summary(args.mode, model_name, results)
        return 0 if all(bool(r.get("passed")) for r in results) else 2

    if args.mode == "amb":
        fn = partial(
            _run_amb_case,
            model_name=model_name,
            data_root=repo_paths.data_root,
            output_root=repo_paths.output_root,
            force=args.force,
        )
        results = _run_parallel(fn, cases, args.jobs)
        _print_summary(args.mode, model_name, results)
        return 0 if all(bool(r.get("passed")) for r in results) else 2

    if args.mode == "build_gt_code":
        gt_builder = GTCodeBuilder(
            repo_paths,
            GTCodeBuildOptions(
                model_name=model_name,
                max_rounds=args.max_rounds,
                timeout_sec=args.timeout,
                publish_solution=args.publish_solution,
                force=args.force,
            ),
        )
        results = _run_parallel(gt_builder.build_case, cases, args.jobs)
        _print_summary(args.mode, model_name, results)
        return 0 if all(bool(r.get("passed")) for r in results) else 2

    if args.mode == "build_disamb":
        disamb_builder = DisambBuilder(
            repo_paths,
            DisambBuildOptions(
                model_name=model_name,
                max_rewrite_rounds=args.max_rewrite_rounds,
                max_validation_rounds=args.max_validation_rounds,
                timeout_sec=args.timeout,
                force=args.force,
            ),
        )
        results = _run_parallel(disamb_builder.build_case, cases, args.jobs)
        _print_summary(args.mode, model_name, results)
        return 0 if all(bool(r.get("passed")) for r in results) else 2

    if args.mode == "build_flow":
        flow_builder = WorkflowBuilder(
            repo_paths,
            WorkflowBuildOptions(
                model_name=model_name,
                max_rounds=args.max_rounds,
                publish_flow=args.publish_flow,
                force=args.force,
            ),
        )
        results = _run_parallel(flow_builder.build_case, cases, args.jobs)
        _print_summary(args.mode, model_name, results)
        return 0 if all(bool(r.get("passed")) for r in results) else 2

    # build_all
    gt_builder = GTCodeBuilder(
        repo_paths,
        GTCodeBuildOptions(
            model_name=model_name,
            max_rounds=args.max_rounds,
            timeout_sec=args.timeout,
            publish_solution=args.publish_solution,
            force=args.force,
        ),
    )
    flow_builder = WorkflowBuilder(
        repo_paths,
        WorkflowBuildOptions(
            model_name=model_name,
            max_rounds=args.max_rounds,
            publish_flow=args.publish_flow,
            force=args.force,
        ),
    )
    disamb_builder = DisambBuilder(
        repo_paths,
        DisambBuildOptions(
            model_name=model_name,
            max_rewrite_rounds=args.max_rewrite_rounds,
            max_validation_rounds=args.max_validation_rounds,
            timeout_sec=args.timeout,
            force=args.force,
        ),
    )

    def _build_one(case_name: str) -> dict[str, Any]:
        gt = gt_builder.build_case(case_name)
        flow = flow_builder.build_case(case_name) if bool(gt.get("passed")) else {
            "case": case_name,
            "passed": False,
            "reason": "skipped_due_to_gt_failure",
        }
        disamb = disamb_builder.build_case(case_name)
        passed = bool(gt.get("passed")) and bool(flow.get("passed")) and bool(disamb.get("passed"))
        return {
            "case": case_name,
            "passed": passed,
            "reason": (
                f"gt={gt.get('reason')} | flow={flow.get('reason')} | disamb={disamb.get('reason')}"
            ),
            "gt": gt,
            "flow": flow,
            "disamb": disamb,
        }

    results = _run_parallel(_build_one, cases, args.jobs)
    _print_summary(args.mode, model_name, results)
    return 0 if all(bool(r.get("passed")) for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
