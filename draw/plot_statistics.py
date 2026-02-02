#!/usr/bin/env python3
"""
Plot case complexity and ambiguity distributions from statistics data.

Usage:
  python3 draw/plot_statistics.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from plot_style import (
    COLORS,
    apply_vldb_style,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
STATISTICS_DATA_ROOT = ROOT / "draw" / "data" / "statistics"
PROCESSING_STEPS_CSV = STATISTICS_DATA_ROOT / "processing_steps.csv"
AMBIGUITY_COUNT_CSV = STATISTICS_DATA_ROOT / "ambiguity_count.csv"
CODE_LENGTH_STATS = STATISTICS_DATA_ROOT / "code_length_stats.csv"
CODE_LINES_COLUMN = "code_lines"
INPUT_TABLE_ROW_COUNTS = STATISTICS_DATA_ROOT / "input_table_row_counts.csv"
FLOW_OPERATOR_COUNTS = STATISTICS_DATA_ROOT / "flow_operator_counts.csv"
DRAW_OUT = ROOT / "draw" / "output" / "statistics"

BAR_EDGE_COLOR = COLORS.gray
BAR_EDGE_WIDTH = 0.4
Y_GRID_ALPHA = 0.18
Y_GRID_WIDTH = 0.4
Y_MARGIN_RATIO = 0.08

COLOR_COMPLEXITY = COLORS.teal
COLOR_AMBIGUITY = COLORS.peach
COLOR_CODE = COLORS.navy
COLOR_INPUT = COLORS.green
COLOR_OPERATOR = COLORS.muted_orange

BASE_FIG_WIDTH = 9.4
AXIS_LABEL_SIZE = 27
TICK_LABEL_SIZE = 21
Y_TICK_PAD = 2
Y_LABEL_PAD = 2


def _load_case_metric_csv(path: Path, column: str) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    values: List[int] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Missing column {column} in {path}")
        for row in reader:
            if not row:
                continue
            raw_val = row.get(column)
            if raw_val is None:
                continue
            try:
                values.append(int(raw_val))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No valid {column} data in {path}")
    return values


def _load_code_lines() -> List[int]:
    if not CODE_LENGTH_STATS.exists():
        raise FileNotFoundError(f"Required file not found: {CODE_LENGTH_STATS}")
    values: List[int] = []
    with CODE_LENGTH_STATS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if CODE_LINES_COLUMN not in reader.fieldnames:
            raise ValueError(f"Missing column {CODE_LINES_COLUMN} in {CODE_LENGTH_STATS}")
        for row in reader:
            raw_val = row.get(CODE_LINES_COLUMN)
            if raw_val is None:
                continue
            try:
                values.append(int(raw_val))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No valid data in {CODE_LENGTH_STATS}")
    return values


def _load_input_table_row_counts() -> Tuple[Tuple[List[int], List[int]], List[int]]:
    """Load per-table row counts and derive table-count distribution + total rows per case."""
    if not INPUT_TABLE_ROW_COUNTS.exists():
        raise FileNotFoundError(f"Required file not found: {INPUT_TABLE_ROW_COUNTS}")
    table_counts: Dict[str, int] = {}
    row_totals: Dict[str, int] = {}
    with INPUT_TABLE_ROW_COUNTS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            case_id = row.get("case_id")
            row_count_raw = row.get("row_count")
            if case_id is None or row_count_raw is None:
                continue
            try:
                row_count = int(row_count_raw)
            except ValueError:
                continue
            table_counts[case_id] = table_counts.get(case_id, 0) + 1
            row_totals[case_id] = row_totals.get(case_id, 0) + row_count
    if not table_counts:
        raise ValueError(f"No valid data in {INPUT_TABLE_ROW_COUNTS}")
    table_dist = _build_counts(list(table_counts.values()))
    return table_dist, list(row_totals.values())


def _load_flow_operator_counts() -> List[int]:
    """Load operator_count values from flow_operator_counts.csv."""
    if not FLOW_OPERATOR_COUNTS.exists():
        raise FileNotFoundError(f"Required file not found: {FLOW_OPERATOR_COUNTS}")
    values: List[int] = []
    for line in FLOW_OPERATOR_COUNTS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("case_id"):
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            values.append(int(parts[1]))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No valid data in {FLOW_OPERATOR_COUNTS}")
    return values


def _build_counts(values: List[int]) -> Tuple[List[int], List[int]]:
    if not values:
        return [], []
    min_val, max_val = min(values), max(values)
    x_vals = list(range(min_val, max_val + 1))
    counts = {v: 0 for v in x_vals}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    y_vals = [counts[v] for v in x_vals]
    return x_vals, y_vals


def _build_cdf(values: List[int]) -> Tuple[List[int], List[float]]:
    if not values:
        return [], []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    y_vals = [(i + 1) / n for i in range(n)]
    return sorted_vals, y_vals


def _build_log_ticks(min_val: int, max_val: int) -> List[int]:
    if max_val < 1:
        return []
    min_val = max(1, min_val)
    ticks: List[int] = []
    power = 0
    while 10 ** power < min_val:
        power += 1
    while True:
        tick = 10 ** power
        if tick > max_val:
            break
        ticks.append(tick)
        power += 1
    return ticks


def _choose_tick_step(n: int) -> int:
    if n > 24:
        return 3
    if n > 16:
        return 2
    return 1


def _nice_step(raw_step: float) -> int:
    if raw_step <= 0:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(raw_step)))
    for mult in (1, 2, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            return int(step)
    return int(10 * magnitude)


def _build_even_ticks(min_val: int, max_val: int, target: int = 6) -> List[int]:
    if min_val == max_val:
        return [min_val]
    min_base = 0 if min_val > 0 else min_val
    raw_step = (max_val - min_base) / max(1, target - 1)
    step = _nice_step(raw_step)
    start = (min_base // step) * step
    end = int(math.ceil(max_val / step) * step)
    ticks = list(range(int(start), end + 1, step))

    while len(ticks) > 7:
        step = _nice_step(step * 1.6)
        end = int(math.ceil(max_val / step) * step)
        ticks = list(range(int(start), end + 1, step))
    while len(ticks) < 5 and step > 1:
        step = max(1, step // 2)
        end = int(math.ceil(max_val / step) * step)
        ticks = list(range(int(start), end + 1, step))
        if len(ticks) > 7:
            break

    return ticks


def _apply_extra_light_y_grid(ax: plt.Axes) -> None:
    ax.yaxis.grid(
        True,
        linestyle="--",
        color=COLORS.light_gray,
        alpha=Y_GRID_ALPHA,
        linewidth=Y_GRID_WIDTH,
        zorder=0,
    )
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _plot_axis_distribution(
    ax: plt.Axes,
    x_vals: List[int],
    y_vals: List[int],
    x_label: str,
    show_ylabel: bool,
    bar_color: str,
    tick_step: int | None,
    x_ticks: List[int] | None,
    use_log_y: bool,
    use_log_x: bool,
) -> None:
    if not x_vals:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS.medium_gray,
        )
        ax.set_axis_off()
        return

    plot_x = [float(x) for x in x_vals]
    if use_log_x:
        plot_x = [max(1.0, x) for x in plot_x]

    ax.bar(
        plot_x,
        y_vals,
        color=bar_color,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )

    min_val, max_val = min(x_vals), max(x_vals)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel("Case count", fontsize=AXIS_LABEL_SIZE, labelpad=Y_LABEL_PAD)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    if use_log_x:
        ax.set_xscale("log")
        ax.set_xlim(max(1, min_val), max_val)
    else:
        ax.set_xlim(min_val - 0.6, max_val + 0.6)
    max_y = max(y_vals) if y_vals else 0
    upper = max_y * (1.0 + Y_MARGIN_RATIO)
    if use_log_y:
        ax.set_yscale("log")
        # Start slightly below 1 so bars with count=1 are visible.
        ax.set_ylim(0.8, upper if upper > 1 else 2.0)
    else:
        ax.set_ylim(0, upper if upper > 0 else 1.0)

    if x_ticks:
        ax.set_xticks(x_ticks)
    elif not use_log_x:
        step = tick_step if tick_step is not None else _choose_tick_step(len(x_vals))
        ax.set_xticks(x_vals[::step])
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=Y_TICK_PAD)

    _apply_extra_light_y_grid(ax)


def _plot_axis_cdf(
    ax: plt.Axes,
    values: List[int],
    x_label: str,
    show_ylabel: bool,
    line_color: str,
    tick_step: int | None,
    x_ticks: List[int] | None,
    minor_x_ticks: List[int] | None,
    use_log_x: bool,
    log_x_ticks: List[int] | None,
    x_max: float | None = None,
) -> None:
    if not values:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS.medium_gray,
        )
        ax.set_axis_off()
        return

    x_vals, y_vals = _build_cdf(values)
    ax.step(x_vals, y_vals, where="post", color=line_color, linewidth=1.6)
    median_val = int(median(values))
    ax.axvline(
        median_val,
        color=line_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
        zorder=2,
    )

    min_val, max_val = min(x_vals), max(x_vals)
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel("CDF", fontsize=AXIS_LABEL_SIZE, labelpad=Y_LABEL_PAD)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

    if use_log_x:
        ax.set_xscale("log")
        ax.set_xlim(max(1, min_val), max_val if x_max is None else x_max)
    elif x_ticks:
        right = max(x_ticks) if x_max is None else x_max
        ax.set_xlim(min(x_ticks), right)
    else:
        right = max_val + 0.6 if x_max is None else x_max
        ax.set_xlim(min_val - 0.6, right)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])

    if x_ticks:
        ax.set_xticks(x_ticks)
        if minor_x_ticks:
            ax.set_xticks(minor_x_ticks, minor=True)
    elif use_log_x:
        ticks = log_x_ticks if log_x_ticks is not None else _build_log_ticks(min_val, max_val)
        if ticks:
            ax.set_xticks(ticks)
    elif tick_step is None:
        ax.set_xticks(_build_even_ticks(min_val, max_val, target=6))
    else:
        step = tick_step
        ax.set_xticks(sorted(set(x_vals))[::step])
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, pad=Y_TICK_PAD)
    ax.tick_params(axis="x", which="minor", length=2, width=0.6)


def _expand_distribution_values(x_vals: List[int], y_vals: List[int]) -> List[int]:
    values: List[int] = []
    for x, y in zip(x_vals, y_vals):
        if y <= 0:
            continue
        values.extend([x] * y)
    return values


def _calculate_median_from_distribution(x_vals: List[int], y_vals: List[int]) -> float | None:
    """Calculate median from distribution (x_vals, y_vals)."""
    values = _expand_distribution_values(x_vals, y_vals)
    if not values:
        return None
    return median(values)


def _plot_distributions(
    complexity_vals: List[int],
    ambiguity_vals: List[int],
    code_vals: List[int],
    input_table_dist: Tuple[List[int], List[int]],
    input_row_counts: List[int],
    operator_counts: List[int],
    out_prefix: Path,
    pdf_only: bool,
) -> None:
    apply_vldb_style()
    plt.rcParams["ytick.major.pad"] = Y_TICK_PAD
    plt.rcParams["axes.labelpad"] = Y_LABEL_PAD
    fig = plt.figure(figsize=(14, 4.5))

    # Layout: 3 plots on each row, 6 total.
    # Reduce wspace now that we share y-axis within each row.
    outer_gs = fig.add_gridspec(2, 1, hspace=0.55, top=0.95)
    top_gs = outer_gs[0].subgridspec(1, 3, wspace=0)
    bottom_gs = outer_gs[1].subgridspec(1, 3, wspace=0)

    # Top row: share Y-axis among the 3 bar charts.
    ax0 = fig.add_subplot(top_gs[0, 0])
    ax1 = fig.add_subplot(top_gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(top_gs[0, 2], sharey=ax0)
    # Bottom row: share Y-axis among the 3 CDFs.
    ax3 = fig.add_subplot(bottom_gs[0, 0])
    ax4 = fig.add_subplot(bottom_gs[0, 1], sharey=ax3)
    ax5 = fig.add_subplot(bottom_gs[0, 2], sharey=ax3)
    axes = [ax0, ax1, ax2, ax3, ax4, ax5]

    for ax in axes:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(0.5)

    # === Top row: 3 bar charts (Case count) ===

    # 1. Input table count (distribution/bar)
    x_vals, y_vals = input_table_dist
    _plot_axis_distribution(
        axes[0],
        x_vals,
        y_vals,
        "(a) Table count",
        True,  # show_ylabel
        COLOR_INPUT,
        None,
        [1, 3, 5, 7, 9, 11, 14, 17, 24],
        True,
        False,
    )
    axes[0].yaxis.grid(False)
    median_val = _calculate_median_from_distribution(x_vals, y_vals)
    if median_val is not None:
        axes[0].axvline(median_val, color="black", linestyle="--", linewidth=1.0, zorder=4)

    # 2. Ambiguity count (distribution/bar)
    amb_x, amb_y = _build_counts(ambiguity_vals)
    _plot_axis_distribution(
        axes[1],
        amb_x,
        amb_y,
        "(b) Ambiguity count",
        False,  # no y-axis label
        COLOR_AMBIGUITY,
        None,
        list(range(1,17,2)) + [18],
        False,
        False,
    )
    axes[1].yaxis.grid(False)
    median_val = _calculate_median_from_distribution(amb_x, amb_y)
    if median_val is not None:
        axes[1].axvline(median_val, color="black", linestyle="--", linewidth=1.0, zorder=4)

    # 3. Transformation steps (distribution/bar)
    step_x, step_y = _build_counts(complexity_vals)
    _plot_axis_distribution(
        axes[2],
        step_x,
        step_y,
        "(c) Preparation steps",
        False,  # no y-axis label
        COLOR_COMPLEXITY,
        None,
        list(range(3,21,3)),
        False,
        False,
    )
    axes[2].yaxis.grid(False)
    median_val = _calculate_median_from_distribution(step_x, step_y)
    if median_val is not None:
        axes[2].axvline(median_val, color="black", linestyle="--", linewidth=1.0, zorder=4)

    # === Bottom row: 3 CDFs ===

    # 4. Input rows (CDF, log X)
    _plot_axis_cdf(
        axes[3],
        input_row_counts,
        "(d) Row count",
        True,  # show_ylabel
        COLOR_INPUT,
        None,
        None,
        None,
        True,
        None,
    )

    # 5. Lines of Code (CDF)
    _plot_axis_cdf(
        axes[4],
        code_vals,
        "(e) Lines of Code",
        False,  # no y-axis label
        COLOR_CODE,
        None,
        [0, 100, 200, 300],
        [50, 150, 250],
        False,
        None,
    )

    # 6. Operator count (CDF)
    _plot_axis_cdf(
        axes[5],
        operator_counts,
        "(f) Operator count",
        False,  # no y-axis label
        COLOR_OPERATOR,
        None,
        [0, 20, 40, 60],
        None,
        False,
        None,
        70,
    )

    # fig.savefig(out_prefix.with_suffix(".pdf"))
    # fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    # NOTE: apply_vldb_style() uses savefig.bbox='tight', which trims outer whitespace.
    # Add an explicit "padding" around the whole exported figure.
    pad_inches = 0.32
    if pdf_only:
        fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=pad_inches)
    else:
        fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot case complexity and ambiguity distributions.")
    parser.add_argument("--output", type=Path, default=DRAW_OUT, help="Output directory.")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    complexity_values = _load_case_metric_csv(PROCESSING_STEPS_CSV, "processing_steps")
    ambiguity_values = _load_case_metric_csv(AMBIGUITY_COUNT_CSV, "ambiguity_count")
    code_values = _load_code_lines()
    input_table_dist, input_row_counts = _load_input_table_row_counts()
    operator_counts = _load_flow_operator_counts()
    _plot_distributions(
        complexity_values,
        ambiguity_values,
        code_values,
        input_table_dist,
        input_row_counts,
        operator_counts,
        out_dir / "workflow_ambiguity_code",
        args.pdf,
    )


if __name__ == "__main__":
    main()
