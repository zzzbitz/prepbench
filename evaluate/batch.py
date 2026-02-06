from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

from evaluate.core import evaluate


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_case_dirs(results_root: Path) -> list[Path]:
    direct = sorted(
        [p for p in results_root.glob("case_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if direct:
        return direct

    nested = sorted(
        [p for p in results_root.rglob("case_*") if p.is_dir() and (p / "solution").is_dir()],
        key=lambda p: str(p),
    )
    return nested


def _pick_candidate_dir(case_dir: Path) -> tuple[str, Optional[Path]]:
    solution_dir = case_dir / "solution"
    flow_cand = solution_dir / "flow_cand"
    cand = solution_dir / "cand"

    if flow_cand.is_dir() and any(flow_cand.glob("*.csv")):
        return "flow_cand", flow_cand
    if cand.is_dir() and any(cand.glob("*.csv")):
        return "cand", cand
    return "none", None


def _to_csv_bool(v: bool) -> str:
    return "true" if v else "false"


def run_batch(results_root: Path) -> Path:
    repo_root = _resolve_repo_root()
    gt_root = repo_root / "evaluate" / "gt"
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    case_dirs = _discover_case_dirs(results_root)
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under: {results_root}")

    output_csv = results_root / "evaluation_summary.csv"
    rows: list[dict[str, str]] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        gt_case_dir = gt_root / case_name
        cand_kind, cand_dir = _pick_candidate_dir(case_dir)

        row = {
            "case_name": case_name,
            "evaluated": "false",
            "passed": "false",
            "candidate_kind": cand_kind,
            "candidate_dir": str(cand_dir.resolve()) if cand_dir else "",
            "gt_dir": str(gt_case_dir.resolve()),
            "error_type": "",
            "error_message": "",
        }

        if not gt_case_dir.is_dir():
            row["error_type"] = "GT_CASE_MISSING"
            row["error_message"] = f"GT case directory not found: {gt_case_dir}"
            rows.append(row)
            continue

        if cand_dir is None:
            row["error_type"] = "CANDIDATE_MISSING"
            row["error_message"] = "No candidate directory with CSV outputs found (checked flow_cand then cand)."
            rows.append(row)
            continue

        passed, first_error = evaluate(str(gt_case_dir), str(cand_dir))
        row["evaluated"] = "true"
        row["passed"] = _to_csv_bool(bool(passed))
        if not passed and isinstance(first_error, dict):
            row["error_type"] = str(first_error.get("error_type") or "UNKNOWN")
            row["error_message"] = str(first_error.get("message") or "")
        rows.append(row)

    fieldnames = [
        "case_name",
        "evaluated",
        "passed",
        "candidate_kind",
        "candidate_dir",
        "gt_dir",
        "error_type",
        "error_message",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    evaluated = sum(1 for r in rows if r["evaluated"] == "true")
    passed = sum(1 for r in rows if r["passed"] == "true")
    print(f"[evaluate.batch] total={total} evaluated={evaluated} passed={passed} output={output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate PrepBench results under a single results root."
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=str,
        help="Path like @output/<model>/<run_mode>, containing case_* directories.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root is not a directory: {results_root}")

    run_batch(results_root)


if __name__ == "__main__":
    main()
