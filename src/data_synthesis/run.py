#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Optional
from functools import partial

from core.case_assets import require_reference_solution_path, resolve_reference_solution_path
from .synthesizer import Synthesizer
from . import config as ds_config


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data"
OUTPUT_ROOT = CURRENT_DIR / "output"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_model_dir(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    return model.strip().replace("/", "__")


def get_output_dir(model: str) -> Path:
    return OUTPUT_ROOT / normalize_model_dir(model)


def _get_full_skip_reason(case_name: str, output_dir: Path, force: bool) -> Optional[str]:
    if force:
        return None
    query_full_path = DATA_DIR / case_name / "query_full.md"
    if query_full_path.exists() and query_full_path.stat().st_size > 0:
        return "existing query_full.md"
    return None


def filter_full_cases(cases: list[str], output_dir: Path, force: bool) -> tuple[list[str], dict[str, str]]:
    to_run: list[str] = []
    skipped: dict[str, str] = {}

    for c in cases:
        reason = _get_full_skip_reason(c, output_dir, force)
        if reason:
            skipped[c] = reason
        else:
            to_run.append(c)

    return to_run, skipped


def parse_model_selector(selector: str, all_models: list[str]) -> list[str]:
    if not selector:
        return list(all_models)
    parts = [p.strip() for p in selector.split(",") if p.strip()]
    if not parts:
        return list(all_models)
    unknown = [m for m in parts if m not in all_models]
    if unknown:
        print(f"[ERROR] --model contains unknown entries: {', '.join(unknown)}")
        sys.exit(1)
    result: list[str] = []
    for m in parts:
        if m not in result:
            result.append(m)
    return result


def parse_case_selector(selector: str) -> list[str]:
    if not selector:
        return []
    
    result = []
    for part in selector.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(f"case_{i:03d}" for i in range(start, end + 1))
        elif part.isdigit():
            result.append(f"case_{int(part):03d}")
        else:
            result.append(part)
    
    return list(dict.fromkeys(result))  


def get_all_cases() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("case_"))


def ensure_output_dir(case_name: str, output_dir: Path) -> Path:
    path = output_dir / case_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def preflight_validate(mode: str, cases: list[str], output_dir: Path) -> None:
    for case_name in cases:
        case_dir = DATA_DIR / case_name
        query_path = case_dir / "query.md"
        code_path = resolve_reference_solution_path(case_dir)
        flow_path = case_dir / "flow.json"
        if not query_path.exists() or code_path is None or not flow_path.exists():
            print(f"[ERROR] {case_name}: missing query.md or reference solution or flow.json")
            sys.exit(1)



def _looks_like_no_ambiguity(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keywords = [
        "no ambiguity", "no ambiguities", "no-ambiguity", "no-ambiguities",
        "zero change", "zero-change", "no change", "unchanged", "same as", "use original",
    ]
    short_hint = len(t) <= 120
    hit = any(k in t for k in keywords)
    return hit and short_hint



def process_full(case_name: str, model_full: str, output_dir: Path) -> str:
    case_dir = DATA_DIR / case_name
    query_path = case_dir / "query.md"
    code_path = require_reference_solution_path(case_dir)
    flow_path = case_dir / "flow.json"

    logger.info(f"[{case_name}][{model_full}] Generating query_full...")

    query = query_path.read_text(encoding="utf-8")
    code = code_path.read_text(encoding="utf-8")
    flow = flow_path.read_text(encoding="utf-8")

    syn = Synthesizer(model_full=model_full)  

    # Write query_full to data/{case}/query_full.md
    output_path = case_dir / "query_full.md"

    final_full = syn.generate_query_full(query, code, flow)

    # Zero-change handling: keep original query text
    zero_change = (final_full.strip() == query.strip()) or _looks_like_no_ambiguity(final_full)
    if zero_change:
        output_path.write_text(query, encoding="utf-8")
    else:
        output_path.write_text(final_full, encoding="utf-8")

    logger.info(f"[OK] {case_name} -> {output_path}")
    return case_name


def process_amb(case_name: str, model_full: str, output_dir: Path, force: bool) -> str:
    case_dir = DATA_DIR / case_name
    query_path = case_dir / "query.md"
    code_path = require_reference_solution_path(case_dir)
    flow_path = case_dir / "flow.json"

    query_full_path = case_dir / "query_full.md"
    output_path = case_dir / "amb_kb.json"

    if output_path.exists() and not force:
        logger.info(f"[SKIP] {case_name}: amb_kb.json already exists, skip")
        return case_name

    query = query_path.read_text(encoding="utf-8")
    code = code_path.read_text(encoding="utf-8")
    flow = flow_path.read_text(encoding="utf-8")

    logger.info(f"[{case_name}][{model_full}] Generating ambiguity base...")

    syn = Synthesizer(model_full=model_full)  

    if not query_full_path.exists() or query_full_path.stat().st_size == 0:
        logger.info(f"[{case_name}][{model_full}] Missing query_full.md, generating full first...")
        final_full = syn.generate_query_full(query, code, flow)
        zero_change = (final_full.strip() == query.strip()) or _looks_like_no_ambiguity(final_full)
        if zero_change:
            query_full_path.write_text(query, encoding="utf-8")
        else:
            query_full_path.write_text(final_full, encoding="utf-8")

    query_full = query_full_path.read_text(encoding="utf-8")
    result = syn.generate_ambiguities_raw(query, code, query_full, flow)
    output_path.write_text(result, encoding="utf-8")

    logger.info(f"[OK] {case_name} -> {output_path}")
    return case_name



def _run_in_parallel(func, cases: list[str], jobs: int, label: str = "") -> None:
    failed: list[str] = []

    if jobs <= 1:
        for c in cases:
            try:
                func(c)
            except Exception:
                failed.append(c)
        if failed:
            suffix = f" ({label})" if label else ""
            print(f"[SUMMARY] failed cases{suffix}: {', '.join(failed)}")
        return

    ex = ThreadPoolExecutor(max_workers=jobs)
    futures = []
    future_to_case = {}
    for c in cases:
        fut = ex.submit(func, c)
        futures.append(fut)
        future_to_case[fut] = c

    for fut in as_completed(futures):
        try:
            fut.result()
        except Exception:
            failed.append(future_to_case.get(fut, '?'))
            continue

    ex.shutdown(wait=True)

    if failed:
        suffix = f" ({label})" if label else ""
        print(f"[SUMMARY] failed cases{suffix}: {', '.join(sorted(failed))}")


def run_full(cases: list[str], jobs: int, model_full: str, output_dir: Path) -> None:
    func = partial(process_full, model_full=model_full, output_dir=output_dir)
    _run_in_parallel(func, cases, jobs, label=model_full)


def run_amb(cases: list[str], jobs: int, model_full: str, output_dir: Path, force: bool) -> None:
    func = partial(process_amb, model_full=model_full, output_dir=output_dir, force=force)
    _run_in_parallel(func, cases, jobs, label=model_full)


def main():
    parser = argparse.ArgumentParser(description="Data synthesis tool (supports parallel jobs)")
    parser.add_argument("mode", choices=["full", "amb"], help="Mode: full=generate query_full, amb=generate ambiguity base")
    parser.add_argument("--case", type=str, default="", help="Specify case: 1 | 1,2,3 | 1-5 | empty=all")
    parser.add_argument("--jobs", type=int, default=1, help="Parallelism, default 1; >1 for concurrent execution")
    parser.add_argument("--model", type=str, default="", help="Specify model: comma-separated; empty=use all from config.FULL['model']")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output")
    args = parser.parse_args()

    if args.jobs <= 0:
        print("[ERROR] --jobs must be a positive integer")
        sys.exit(1)
    
    if args.case:
        cases = parse_case_selector(args.case)
    else:
        cases = get_all_cases()
    
    if not cases:
        print("[ERROR] No valid cases found")
        sys.exit(1)

    full_models = ds_config.get_full_models()
    models = parse_model_selector(args.model, full_models)

    for model in models:
        output_dir = get_output_dir(model)
        model_cases = list(cases)
        skipped: dict[str, str] = {}

        if args.mode == "full":
            model_cases, skipped = filter_full_cases(model_cases, output_dir, args.force)
            if skipped:
                skipped_items = ", ".join(f"{k}({v})" for k, v in sorted(skipped.items()))
                print(f"[SKIP] full({model}): {len(skipped)} cases skipped: {skipped_items}")

        if not model_cases:
            print(f"[DONE] {model}: No cases to run")
            continue

        display_dir = output_dir if args.mode == "full" else DATA_DIR
        print(f"Model: {model} -> Output dir: {display_dir}")
        print(f"Target: {len(model_cases)} cases, parallelism: {args.jobs}")

        preflight_validate(args.mode, model_cases, output_dir)

        try:
            if args.mode == "full":
                run_full(model_cases, args.jobs, model, output_dir)
            else:
                run_amb(model_cases, args.jobs, model, output_dir, args.force)
        except Exception as e:
            print(f"[FAIL] {e}")
            sys.exit(1)

    print("[DONE]")


if __name__ == "__main__":
    main()
