#!/usr/bin/env python3
"""
Run all drawing scripts in sequence.

Usage:
  python3 draw/plot.py
  python3 draw/plot.py --subset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all RQ plots")
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Only consider cases with numbers that are multiples of 15 (RQ3/RQ4 only)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save PDF only",
    )
    args = parser.parse_args()

    pdf_arg = ["--pdf"] if args.pdf else []
    _run("rq2_plot.py", pdf_arg if pdf_arg else None)
    _run("rq3_plot.py", (["--subset"] if args.subset else []) + pdf_arg)
    _run("rq4_plot.py", (["--subset"] if args.subset else []) + pdf_arg)
    _run("rq5_plot.py", pdf_arg if pdf_arg else None)


if __name__ == "__main__":
    main()
