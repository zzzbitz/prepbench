import argparse
import json
from pathlib import Path
import sys
import concurrent.futures
from typing import Iterable, List, Optional, TypeVar
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.orchestrator import Orchestrator
from config.experiment_config import ExperimentConfig
from core.utils.paths import get_output_path
from core.utils.logging_config import configure_logging, get_logger
from llm_connect.config import validate_clarifier_settings

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

    # Path or case_ prefix
    case_arg = case_arg.strip()
    if case_arg.startswith("case_"):
        p = data_dir / case_arg
        if not p.exists():
            print(f"Error: Case directory not found: {p}")
            sys.exit(1)
        return [p]

    maybe_path = Path(case_arg)
    if ("/" in case_arg or case_arg.startswith(".")) and maybe_path.exists():
        p = maybe_path.resolve()
        if not p.is_dir():
            print(f"Error: --case path is not a directory: {p}")
            sys.exit(1)
        return [p]

    if "," in case_arg:
        tokens = [t.strip() for t in case_arg.split(",") if t.strip()]
        for t in tokens:
            if not t.isdigit():
                print(f"Error: Invalid number '{t}' in comma list, only digits allowed. Example: 10,28,41")
                sys.exit(1)
        seen: set[Path] = set()
        result: List[Path] = []
        for t in tokens:
            i = int(t)
            p = data_dir / f"case_{i:03d}"
            if p.exists():
                if p not in seen:
                    result.append(p)
                    seen.add(p)
            else:
                    print(f"Warning: Case not found: case_{i:03d}")
        if not result:
            print(f"Error: No valid cases in comma list: {case_arg}")
            sys.exit(1)
        return result

    if "-" in case_arg:
        parts = case_arg.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = map(int, parts)
            if start > end:
                start, end = end, start
            result = []
            for i in range(start, end + 1):
                p = data_dir / f"case_{i:03d}"
                if p.exists():
                    result.append(p)
                else:
                    print(f"Warning: Case not found: case_{i:03d}")
            if not result:
                print(f"Error: No valid cases in range {case_arg}")
                sys.exit(1)
            return result
        else:
            print(f"Error: Invalid range format: {case_arg}, should be like 5-8")
            sys.exit(1)

    if case_arg.isdigit():
        p = data_dir / f"case_{int(case_arg):03d}"
        if not p.exists():
            print(f"Error: Case directory not found: {p}")
            sys.exit(1)
        return [p]

    print("Error: Invalid --case format, only supports: single ID (52) / range (5-8) / comma list (1,3,5)")
    sys.exit(1)

def read_final_status(final_status_path: Path) -> tuple[str, Optional[dict]]:
    if not final_status_path.exists():
        return "missing", None
    try:
        status = json.loads(final_status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            return "corrupt", None
        return "ok", status
    except Exception:
        return "corrupt", None

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None

def _valid_interact_cache(summary_path: Path) -> bool:
    data = _read_json(summary_path)
    if not data:
        return False
    qa_history = data.get("qa_history")
    return isinstance(qa_history, list)

def _valid_profile_cache(summary_path: Path) -> bool:
    data = _read_json(summary_path)
    if not data:
        return False
    if data.get("error"):
        return False
    summary = data.get("summary")
    return isinstance(summary, str) and bool(summary.strip())

def _classify_llm_error(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    if "llm" in lowered:
        return "llm_connection"
    if "openrouter" in lowered:
        return "llm_connection"
    if "connection reset" in lowered or "connection aborted" in lowered:
        return "llm_connection"
    if "response ended prematurely" in lowered:
        return "llm_connection"
    if "timeout" in lowered or "timed out" in lowered:
        return "llm_connection"
    if "connection" in lowered:
        return "llm_connection"
    return None

def _should_rerun_profile_due_to_llm(output_path: Path) -> bool:
    summary_path = output_path / "profile" / "summary.json"
    data = _read_json(summary_path) or {}
    err = str(data.get("error") or "")
    return _classify_llm_error(err) is not None

def _should_rerun_interact_due_to_llm(output_path: Path) -> bool:
    summary_path = output_path / "clarify" / "summary.json"
    data = _read_json(summary_path) or {}
    stopped_reason = str(data.get("stopped_reason") or "")
    return _classify_llm_error(stopped_reason) is not None

def _should_rerun_due_to_llm_status(status: Optional[dict]) -> bool:
    if not isinstance(status, dict):
        return False
    message = str(status.get("message") or "")
    return _classify_llm_error(message) is not None

def _should_rerun_e2e_due_to_llm(output_path: Path) -> tuple[bool, bool]:
    """Return (rerun_code, rerun_flow)."""
    status_path = output_path / "solution" / "final_status.json"
    status = _read_json(status_path) or {}
    code_reason = str(status.get("code_reason") or "")
    flow_reason = str(status.get("flow_reason") or "")
    code_passed = bool(status.get("code_passed", False))

    rerun_code = _classify_llm_error(code_reason) is not None
    rerun_flow = _classify_llm_error(flow_reason) is not None and code_passed
    return rerun_code, rerun_flow

def _should_run_flow_for_cand(output_path: Path) -> bool:
    """Check if flow should run for cases where code execution succeeded (cand exists)."""
    cand_dir = output_path / "solution" / "cand"
    if not cand_dir.exists() or not any(cand_dir.iterdir()):
        return False  # Code didn't produce output, no flow to run

    # Priority 1: Check flow_final_status.json (the actual flow result)
    flow_status_path = output_path / "solution" / "flow_final_status.json"
    flow_status = _read_json(flow_status_path) or {}
    flow_reason = str(flow_status.get("reason") or flow_status.get("flow_reason") or "")

    # If flow completed (passed or failed), don't re-run
    if flow_reason and not flow_reason.startswith("skipped_"):
        return False

    # Priority 2: Fallback to final_status.json
    status_path = output_path / "solution" / "final_status.json"
    status = _read_json(status_path) or {}
    final_flow_reason = str(status.get("flow_reason") or "")

    if final_flow_reason and not final_flow_reason.startswith("skipped_"):
        return False

    # Cand exists but flow was skipped or never ran - should run flow
    return True

def decide_run_from_status(status_state: str, status: Optional[dict]) -> tuple[str, Optional[str]]:
    """Decide whether to run or skip based on status.
    
    Skip only if status is ok and run completed normally.
    Retry on incomplete runs (codegen_failed, no_code_generated) or corrupt/missing status.
    """
    if status_state == "ok":
        reason = (status or {}).get("reason") or "unknown"
        # Treat codegen/no-code as incomplete runs that should be retried by default.
        if reason in {"codegen_failed", "no_code_generated"}:
            return "run_incomplete", reason
        return "skip_finished", reason
    if status_state == "corrupt":
        return "run_corrupt", None
    return "run_not_run", None

def _write_skip_status(output_path: Path, reason: str, message: str) -> None:
    """Write a skip marker to final_status.json so next run can skip early."""
    solution_dir = output_path / "solution"
    solution_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "ok": True,
        "passed": False,
        "reason": reason,
        "message": message,
        "skipped": True,
    }
    (solution_dir / "final_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _write_skipped_e2e_report(
    skipped_e2e: list[tuple[str, str, str]],
    path: Path = Path("run_e2e_fail.txt"),
) -> None:
    if not skipped_e2e:
        return
    filtered: list[tuple[str, str, str]] = []
    for item in skipped_e2e:
        _, _, reason = item
        if reason.startswith(("interact_cache_", "profile_cache_")) or "llm_connection" in reason:
            filtered.append(item)
    if not filtered:
        return
    grouped: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for model_name, case_name, reason in filtered:
        key = (model_name, case_name, reason)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(model_name, []).append((case_name, reason))
    lines: list[str] = []
    lines.append(f"total_skipped: {len(filtered)}")
    for model_name in sorted(grouped):
        entries = sorted(grouped[model_name], key=lambda item: (item[0], item[1]))
        lines.append(f"model: {model_name} (count={len(entries)})")
        for case_name, reason in entries:
            lines.append(f"- {case_name}: {reason}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def run_single_case(
    case_path: Path,
    config: ExperimentConfig,
):
    """Run experiment for a single case.
    
    Skip decision happens in main process prefilter.
    """
    
    # Skip decision happens in main process prefilter.
    output_path = get_output_path(case_path, config)

    logger = get_logger("run")

    try:
        import shutil
        if output_path.exists():
            cand_dir = output_path / "solution" / "cand"
            profile_summary = output_path / "profile" / "summary.json"
            # E2E: if cand exists but flow needs to run, clean flow outputs only
            if config.run_mode == "e2e" and _should_run_flow_for_cand(output_path):
                flow_dir = output_path / "flow"
                if flow_dir.exists():
                    shutil.rmtree(flow_dir)
                for name in ("flow_final_status.json", "flow_execution.json", "flow_evaluation.json", "flow_protocol.json", "flow.json"):
                    path = output_path / "solution" / name
                    if path.exists():
                        path.unlink()
                print(f"[{case_path.name}][{config.run_mode}][{config.model_name}] Cleaned flow outputs for continuation: {output_path}")
            elif config.run_mode in ("profile", "raw_profile") and _valid_profile_cache(profile_summary):
                rounds_dir = output_path / "rounds"
                solution_dir = output_path / "solution"
                if rounds_dir.exists():
                    shutil.rmtree(rounds_dir)
                if solution_dir.exists():
                    shutil.rmtree(solution_dir)
                print(f"[{case_path.name}][{config.run_mode}][{config.model_name}] Preserved profile summary; cleaned code outputs: {output_path}")
            else:
                shutil.rmtree(output_path)
                print(f"[{case_path.name}][{config.run_mode}][{config.model_name}] Cleaned previous outputs: {output_path}")
    except Exception as e:
        print(f"[{case_path.name}][{config.run_mode}][{config.model_name}] Warning: failed to clean outputs at {output_path}: {e}")

    orchestrator = Orchestrator(
        parallel_execution=config.parallel_execution,
        max_workers=config.jobs
    )
    try:
        result = orchestrator.run(str(case_path), config=config)
        logger.info(f"[{case_path.name}][{config.run_mode}][{config.model_name}] Completed. Passed: {result.get('passed')}")
        return result
    except Exception as e:
        logger.error(f"[{case_path.name}][{config.run_mode}] Failed: {e}")
        # import traceback
        # traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Run AutoETL Experiment")
    parser.add_argument("--case", type=str, help="Case ID (e.g., 52), Range (e.g., 5-8), case_XXX, or Path. If omitted, runs all cases.")
    parser.add_argument("--run_mode", "--mode", type=str, help="Run mode (comma-separated). Choices: raw, raw_profile, full, profile, interact, e2e, flow. Defaults to config.")
    parser.add_argument("--model", type=str, help="Model name (comma-separated). Defaults to config.")

    parser.add_argument("--subset", action="store_true", help="Only run cases that are multiples of 15 (e.g., 015, 030, 045...)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt and run immediately.")
    parser.add_argument("--list", type=int, default=0, help="List first N tasks after skip filter and exit.")
    # log-level and dry-run removed for simplicity
    
    args = parser.parse_args()
    
    # Configure logging early
    configure_logging()
    logger = get_logger("run")

    # Determine run modes from CLI (or config if not provided).
    run_modes: list[str] = []
    if args.run_mode:
        run_modes = [m.strip().lower() for m in args.run_mode.split(",") if m.strip()]

    # Determine models from CLI only (shared across modes).
    model_names_by_mode: dict[str, list[str]] = {}

    code_models: list[str] = []
    if args.model:
        cli_models = [m.strip() for m in args.model.split(",") if m.strip()]
        if cli_models:
            code_models = dedupe_preserve_order(cli_models)
            logger.info(f"Using CLI-specified models: {code_models}")
 
    # Load base config object (will hold common settings)
    base_config = ExperimentConfig.load_config(model_name_override=code_models[0] if code_models else None)

    if not run_modes:
        cfg_mode = str(base_config.run_mode or "").strip().lower()
        if cfg_mode:
            run_modes = [cfg_mode]

    run_modes = dedupe_preserve_order(run_modes)
    allowed_run_modes = {"raw", "raw_profile", "full", "profile", "interact", "e2e", "flow"}
    invalid_run_modes = [m for m in run_modes if m not in allowed_run_modes]
    if invalid_run_modes:
        print(f"Error: Unknown --run_mode values: {invalid_run_modes}. Allowed: {sorted(allowed_run_modes)}")
        sys.exit(1)
    if not run_modes:
        print("Error: run_mode is empty. Provide --run_mode or set experiment.run_mode in config/settings.yaml.")
        sys.exit(1)

    if not code_models:
        code_models = [base_config.model_name]
        logger.info(f"Using config model: {code_models}")

    for m in run_modes:
        model_names_by_mode[m] = code_models

    # Dirty-case-only modes removed; profile runs on all cases.

    # Preflight Clarifier settings only when all selected modes require it.
    needs_clarifier = any(m in ("interact", "e2e") for m in run_modes)
    only_clarifier_modes = all(m in ("interact", "e2e") for m in run_modes)
    if needs_clarifier and only_clarifier_modes:
        try:
            validate_clarifier_settings()
        except Exception as e:
            print(f"Error: Clarifier config invalid: {e}")
            sys.exit(1)
    elif needs_clarifier:
        logger.warning(
            "Clarifier config preflight skipped because non-clarifier modes are included; "
            "interact/e2e tasks may fail if misconfigured."
        )

    jobs = base_config.jobs or 1
    
    logger.info(f"Configuration: RunMode={run_modes}, Jobs={jobs}")
    
    data_dir = Path(__file__).parent / "data"
    repo_root = Path(__file__).parent

    # Load dirty case info if any dirty mode is selected
    # cleanspec removed; profile runs on all cases

    # Resolve cases based on user input (supports case id/range/list or explicit path).
    if args.case:
        user_cases = resolve_cases(args.case, data_dir)
    else:
        # No cases specified - resolve all cases per mode
        user_cases = None  # Will be resolved per mode

    # Run execution
    import dataclasses
    
    # Generate task matrix with per-mode case filtering
    tasks = []
    
    # Resolve cases per mode
    cases_for_mode: dict[str, list[Path]] = {}
    for mode in run_modes:
        if user_cases is not None:
            cases_for_mode[mode] = user_cases
        else:
            # No cases specified - use all cases
            cases_for_mode[mode] = resolve_cases(None, data_dir)
    
    # Validate case directories exist (handled earlier per mode)
    
    # Build task list
    for mode in run_modes:
        mode_cases = cases_for_mode.get(mode, [])
        # Apply --subset filter: only cases that are multiples of 15
        if args.subset:
            mode_cases = [c for c in mode_cases if int(c.name.split("_")[1]) % 15 == 0]
        for case in mode_cases:
            for model in model_names_by_mode.get(mode, []):
                # Create a specific config for this run
                run_config = dataclasses.replace(base_config, run_mode=mode, model_name=model)
                tasks.append((case, run_config))
    
    if not tasks:
        print("No tasks generated to run.")
        sys.exit(0)

    total_tasks = len(tasks)
    skipped_tasks = 0
    skipped_e2e: list[tuple[str, str, str]] = []  # (model_name, case_name, reason)
    
    # Status-based skip filter
    filtered_tasks = []
    for case, cfg in tasks:
        output_path = get_output_path(case, cfg)
        final_status_path = output_path / "solution" / "final_status.json"
        status_state, status = read_final_status(final_status_path)
        action, reason = decide_run_from_status(status_state, status)
        if action.startswith("skip_"):
            rerun_due_llm = False
            if cfg.run_mode in {"profile", "raw_profile", "interact", "flow"}:
                if _should_rerun_due_to_llm_status(status):
                    rerun_due_llm = True
                elif cfg.run_mode in {"profile", "raw_profile"} and _should_rerun_profile_due_to_llm(output_path):
                    rerun_due_llm = True
                elif cfg.run_mode == "interact" and _should_rerun_interact_due_to_llm(output_path):
                    rerun_due_llm = True
            elif cfg.run_mode == "e2e":
                rerun_code, rerun_flow = _should_rerun_e2e_due_to_llm(output_path)
                if rerun_code:
                    rerun_due_llm = True
                elif rerun_flow or _should_run_flow_for_cand(output_path):
                    # Flow needs to run (LLM error or cand exists but flow skipped)
                    rerun_due_llm = True
            if rerun_due_llm:
                filtered_tasks.append((case, cfg))
                continue
            skipped_tasks += 1
            if cfg.run_mode == "e2e":
                reason_tag = f"status_{action}"
                if reason:
                    reason_tag = f"{reason_tag}({reason})"
                message = ""
                if isinstance(status, dict):
                    message = str(status.get("message") or "")
                if _classify_llm_error(message):
                    reason_tag = f"{reason_tag}[llm_connection]"
                skipped_e2e.append((cfg.model_name, case.name, reason_tag))
            continue
        filtered_tasks.append((case, cfg))
    tasks = filtered_tasks
    if skipped_tasks:
        logger.info(
            f"Pre-skip filtered {skipped_tasks}/{total_tasks} tasks."
        )

    if not tasks:
        _write_skipped_e2e_report(skipped_e2e)
        print("No tasks to run after skip filter.")
        sys.exit(0)

    if args.list and args.list > 0:
        limit = min(args.list, len(tasks))
        print(f"Listing first {limit} tasks after skip filter:")
        for idx, (case, cfg) in enumerate(tasks[:limit], start=1):
            print(f"{idx:03d} {cfg.run_mode} {cfg.model_name} {case.name}")
        sys.exit(0)
    
    # Summary stats
    all_cases = set(case.name for case, _ in tasks)
    distinct_models = sorted({cfg.model_name for _, cfg in tasks})
    logger.info(
        f"Generated {total_tasks} tasks; to-run {len(tasks)} "
        f"(Cases={len(all_cases)}, RunModes={len(run_modes)}, Models={distinct_models})"
    )

    if not args.yes:
        print("\n== Confirm Run ==")
        print(f"RunModes: {run_modes}")
        print(f"Models: {distinct_models}")
        print(f"Cases: {len(all_cases)} (selector={args.case or 'ALL'})")
        print(f"Tasks: {len(tasks)} (cases x models x modes)")
        if skipped_tasks:
            print(f"SkippedAlready: {skipped_tasks}")
        print(f"Parallel: {bool(base_config.parallel_execution)} Jobs={jobs}")
        print(f"Timeout: {base_config.timeout}s")
        print(f"OutputTemplate: {base_config.output_root_template}")
        print(f"MaxRounds: debug={base_config.max_rounds_debug} interact={base_config.max_rounds_interact}")
        if any(m in ("interact", "e2e") for m in run_modes):
            print(
                "Clarify: question_ratio="
                f"{base_config.question_ratio} max_questions_cap={base_config.max_questions_cap} "
                f"max_questions_per_ask={base_config.max_questions_per_ask}"
            )
        if any(m in ("profile", "raw_profile") for m in run_modes):
            print(f"Profile: enabled={base_config.profile.enabled} max_rows_per_file={base_config.profile.max_rows_per_file} max_rounds={base_config.profile.max_rounds}")

        try:
            resp = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in {"y", "yes"}:
            print("Aborted by user.")
            sys.exit(0)

    use_thread_pool = any(cfg.run_mode == "e2e" for _, cfg in tasks)

    if jobs > 1 and len(tasks) > 1:
        if use_thread_pool:
            logger.info(f"Running in parallel with {jobs} workers (ThreadPool, e2e present)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(run_single_case, case, cfg): (case, cfg)
                    for case, cfg in tasks
                }
                with tqdm(total=len(tasks), desc="Running", unit="task", ascii=True, dynamic_ncols=True) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        case, cfg = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"[{case.name}][{cfg.run_mode}] Exception: {e}")
                        pbar.update(1)
        else:
            logger.info(f"Running in parallel with {jobs} workers (ProcessPool)...")
            # Use ProcessPoolExecutor for CPU-bound tasks (or mixed IO/CPU).
            # Some restricted environments disallow sysconf calls used by ProcessPool initialization.
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                    futures = {
                        executor.submit(run_single_case, case, cfg): (case, cfg)
                        for case, cfg in tasks
                    }
                    with tqdm(total=len(tasks), desc="Running", unit="task", ascii=True, dynamic_ncols=True) as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            case, cfg = futures[future]
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"[{case.name}][{cfg.run_mode}] Exception: {e}")
                            pbar.update(1)
            except PermissionError as e:
                logger.warning(f"ProcessPoolExecutor unavailable ({e}); falling back to sequential execution.")
                for case, cfg in tqdm(tasks, desc="Running", unit="task", ascii=True, dynamic_ncols=True):
                    run_single_case(case, cfg)
    else:
        logger.info("Running sequentially...")
        for case, cfg in tqdm(tasks, desc="Running", unit="task", ascii=True, dynamic_ncols=True):
            run_single_case(case, cfg)
    
    # Report skipped e2e cases
    _write_skipped_e2e_report(skipped_e2e)


if __name__ == "__main__":
    main()


# python run.py --case 15
# python run.py --case 1-5
# python run.py
# --force --onlyfalse
