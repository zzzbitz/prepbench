#!/usr/bin/env python3
"""
RQ3: Interactive Clarification Effectiveness.

Figure 3.1: Pass rate by run mode (RAW / INTERACT / FULL).
Figure 3.2: Question budget usage.
Figure 3.3: Radar chart (Recall only).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    COLORS,
    apply_vldb_style,
    apply_light_y_grid,
    format_percentage_axis,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    HATCH_FORWARD,
    HATCH_BACKWARD,
    PALETTE_NATURE,
    PALETTE_MUTED,
    BASE_FIGSIZE,
    BASE_LEGEND_Y,
    apply_base_axes_layout,
    BASE_LABEL_SIZE,
    BASE_XTICK_SIZE,
    BASE_YTICK_SIZE,
    BASE_LEGEND_SIZE,
    FIXED_MODEL_ORDER,
    get_model_display_name,
)


ROOT = Path(__file__).resolve().parents[1]
DRAW_OUT = ROOT / "draw" / "output"
RQ3_OUT = DRAW_OUT / "rq3"
RQ3_DATA_CACHE = ROOT / "draw" / "data" / "rq3"
RQ3_STATUS_PATH = ROOT / "draw" / "data" / "rq3" / "rq3_task_status.csv"

# Slightly smaller typography for RQ3 plots
LABEL_SIZE = BASE_LABEL_SIZE - 1
XTICK_SIZE = BASE_XTICK_SIZE - 1
YTICK_SIZE = BASE_YTICK_SIZE - 1
LEGEND_SIZE = BASE_LEGEND_SIZE - 1
DELTA_BADGE_SIZE = 11
RADAR_LEGEND_SIZE = 9

# Ambiguity type order (same as RQ2)
TYPE_ORDER = [
    "Single-table reference",
    "Multi-table alignment",
    "Group-level concept",
    "Row-level concept",
    "Operation incomplete",
    "Operation inconsistent",
    "Operation boundary",
]

TYPE_TO_SUFFIX = {
    "Single-table reference": "single_table_grounding",
    "Multi-table alignment": "multi_table_alignment",
    "Group-level concept": "aggregate_rule",
    "Row-level concept": "non_aggregate_rule",
    "Operation incomplete": "operation_incomplete",
    "Operation inconsistent": "operation_inconsistent",
    "Operation boundary": "operation_boundary",
}


def _load_task_status() -> pd.DataFrame:
    """Load task status from CSV."""
    if not RQ3_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing task status: {RQ3_STATUS_PATH}")
    return pd.read_csv(RQ3_STATUS_PATH)


def _discover_models_from_status(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Discover models from task status CSV using fixed order."""
    available = set(df["model"].dropna().unique().tolist()) if "model" in df.columns else set()
    models: List[Tuple[str, str]] = []
    for model_id in FIXED_MODEL_ORDER:
        if model_id in available:
            models.append((model_id, get_model_display_name(model_id)))
            available.discard(model_id)
    for model_id in sorted(available):
        models.append((model_id, get_model_display_name(model_id)))
    return models


def _normalize_passed(series: pd.Series) -> pd.Series:
    """Normalize pass/fail values to bool or None."""
    def _coerce(val):
        if pd.isna(val):
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(int(val))
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            if s in ("false", "0", "no"):
                return False
        return None
    return series.map(_coerce)


def _simplify_model_label(label: str) -> str:
    """Simplify model labels for display."""
    if "GPT-5.1-Codex" in label:
        return "GPT-5.1-Codex"
    if "DeepSeek-V3.2" in label:
        return "DeepSeek-V3.2"
    replacements = {
        "Claude Sonnet 4.5": "Claude-Sonnet-4.5",
        "Gemini-3-Flash": "Gemini 3 Flash",
        "Grok-Code-1": "Grok Code Fast 1",
        "Kimi-k2-Thinking": "Kimi K2 Thinking",
        "Qwen3-235B": "Qwen3-235B-A22B",
        "DevStral": "DevStral 2",
    }
    for old, new in replacements.items():
        if old in label and new not in label:
            label = label.replace(old, new)
    label = label.replace("Qwen2.5", "Qwen")
    label = label.replace("Llama-3", "Llama")
    return label


def _compute_rq3_1_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 3.1 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)

    # Build DataFrame
    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        raw_vals = _normalize_passed(model_df["raw_profile_passed"]) if "raw_profile_passed" in model_df.columns else pd.Series([], dtype=object)
        interact_vals = _normalize_passed(model_df["e2e_code_passed"]) if "e2e_code_passed" in model_df.columns else pd.Series([], dtype=object)
        valid = raw_vals.notna() & interact_vals.notna()

        total = int(valid.sum())
        raw_pass = int(raw_vals[valid].sum()) if total else 0
        interact_pass = int(interact_vals[valid].sum()) if total else 0
        raw_rate = (raw_pass / total) if total else 0.0
        interact_rate = (interact_pass / total) if total else 0.0
        delta = interact_rate - raw_rate

        data.append({
            "model": model,
            "model_label": label,
            "raw_pass_rate": raw_rate,
            "interact_pass_rate": interact_rate,
            "delta": delta
        })

    return pd.DataFrame(data)


def _compute_rq3_2_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 3.2 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)
    if df.empty:
        return pd.DataFrame()

    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        if "questions_used" in model_df.columns:
            questions_used = pd.to_numeric(model_df["questions_used"], errors="coerce").fillna(0)
        else:
            questions_used = pd.Series([0] * len(model_df))
        if "max_questions" in model_df.columns:
            max_questions = pd.to_numeric(model_df["max_questions"], errors="coerce").fillna(0)
        else:
            max_questions = pd.Series([0] * len(model_df))

        total_used = float(questions_used.sum())
        total_max = float(max_questions.sum())
        usage_ratio = (total_used / total_max) if total_max else 0.0

        data.append({
            "model": model,
            "model_label": label,
            "usage_ratio": usage_ratio,
        })

    return pd.DataFrame(data)


def _compute_rq3_4a_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 3.4a from task status."""
    df = _load_task_status()
    if df.empty:
        return pd.DataFrame()
    models = _discover_models_from_status(df)

    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        for t in TYPE_ORDER:
            suffix = TYPE_TO_SUFFIX[t]
            q_col = f"qa_questions_{suffix}"
            hit_col = f"qa_unique_hits_{suffix}"
            total_col = f"qa_total_slots_{suffix}"

            q_series = model_df[q_col] if q_col in model_df.columns else pd.Series([0] * len(model_df))
            hit_series = model_df[hit_col] if hit_col in model_df.columns else pd.Series([0] * len(model_df))
            total_series = model_df[total_col] if total_col in model_df.columns else pd.Series([0] * len(model_df))

            q_cnt = float(pd.to_numeric(q_series, errors="coerce").fillna(0).sum())
            unique_hit_cnt = float(pd.to_numeric(hit_series, errors="coerce").fillna(0).sum())
            s_cnt = float(pd.to_numeric(total_series, errors="coerce").fillna(0).sum())

            precision = (unique_hit_cnt / q_cnt) if q_cnt > 0 else 0.0
            recall = (unique_hit_cnt / s_cnt) if s_cnt > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            data.append({
                "model": model,
                "model_label": label,
                "ambiguity_type": t,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    return pd.DataFrame(data)


def _save_plot_data(df: pd.DataFrame, filename: str) -> None:
    """Save plot data to CSV cache."""
    RQ3_DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = RQ3_DATA_CACHE / filename
    df.to_csv(cache_path, index=False)
    print(f"Cached: {cache_path}")


def _get_rq3_1_plot_data() -> pd.DataFrame:
    """Get RQ3.1 plot data."""
    df = _compute_rq3_1_plot_data()
    _save_plot_data(df, "rq3_1_plot_data.csv")
    return df


def _get_rq3_2_plot_data() -> pd.DataFrame:
    """Get RQ3.2 plot data."""
    df = _compute_rq3_2_plot_data()
    _save_plot_data(df, "rq3_2_plot_data.csv")
    return df


def _get_rq3_4a_plot_data() -> pd.DataFrame:
    """Get RQ3.4a plot data."""
    df = _compute_rq3_4a_plot_data()
    _save_plot_data(df, "rq3_4a_plot_data.csv")
    return df


def _with_ext(path: Path, ext: str) -> Path:
    return path.parent / f"{path.name}{ext}"


def _save_figure(fig: plt.Figure, output_path: Path, pdf_only: bool, **kwargs) -> None:
    if pdf_only:
        pdf_path = _with_ext(output_path, ".pdf")
        fig.savefig(pdf_path, **kwargs)
        print(f"Saved: {pdf_path}")
    else:
        png_path = _with_ext(output_path, ".png")
        fig.savefig(png_path, dpi=300, **kwargs)
        print(f"Saved: {png_path}")


def _plot_pass_rate_by_mode(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 3.1: Pass rate by run mode (RAW vs INTERACT)."""
    apply_vldb_style()

    # Match RQ2 paper template dimensions
    fig, ax = plt.subplots(figsize=BASE_FIGSIZE)

    # Match RQ2 paper spacing
    spacing = 0.8
    x = np.arange(len(df)) * spacing

    # Match RQ2 paper bar width
    width = 0.32

    raw_vals = df["raw_pass_rate"].values
    interact_vals = df["interact_pass_rate"].values

    # Use colors from plot_style for consistency
    from plot_style import STYLE_RAW, add_delta_badge
    RAW_COLOR = STYLE_RAW["color"]  # peach
    RAW_HATCH = STYLE_RAW["hatch"]  # forward slash
    INTERACT_COLOR = COLORS.teal
    INTERACT_HATCH = HATCH_BACKWARD  # Add hatch pattern
    DELTA_COLOR = COLORS.green

    ax.bar(
        x - width / 2,
        raw_vals,
        width,
        label="Orig",
        color=RAW_COLOR,
        hatch=RAW_HATCH,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        interact_vals,
        width,
        label="Interact",
        color=INTERACT_COLOR,
        hatch=INTERACT_HATCH,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )

    # No value labels on bars (per user request)

    # Delta annotations (match RQ2.1)
    delta_offset = 0.07
    for i, (rv, iv) in enumerate(zip(raw_vals, interact_vals)):
        delta = iv - rv
        x_pos = x[i]
        y_pos = max(rv, iv) + delta_offset
        add_delta_badge(ax, x_pos, y_pos, delta, color=DELTA_COLOR, fontsize=DELTA_BADGE_SIZE)

    ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
    ax.set_xlabel("", fontsize=LABEL_SIZE)
    ax.set_xticks(x)

    # Simplified model labels
    simplified_labels = [_simplify_model_label(label) for label in df["model_label"]]
    ax.set_xticklabels(
        simplified_labels,
        fontsize=XTICK_SIZE,
        rotation=30,
        ha="right"
    )

    # Set x-axis limits
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    ax.set_ylim(0, 0.85)  # Match RQ2 paper style
    ax.set_yticks([0, 0.25, 0.5, 0.75]) # upper limit
    format_percentage_axis(ax, "y", max_val=0.85)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE)

    # Legend above plot (2 columns)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, BASE_LEGEND_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    apply_light_y_grid(ax)

    RQ3_OUT.mkdir(parents=True, exist_ok=True)
    apply_base_axes_layout(fig)
    _save_figure(fig, RQ3_OUT / "rq3.1", pdf_only)
    plt.close(fig)


def _plot_budget_usage(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 3.2: Budget usage ratio (Questions Used / Budget)."""
    if df.empty:
        print("Warning: No data for rq3.2, skipping")
        return

    apply_vldb_style()

    # Match RQ3.1 template dimensions
    fig, ax = plt.subplots(figsize=BASE_FIGSIZE)

    # Match RQ3.1 spacing
    spacing = 0.8
    x = np.arange(len(df)) * spacing

    usage_ratios = df["usage_ratio"].values

    # Single bar per model - wider for better visual weight
    width = 0.50

    ax.bar(
        x,
        usage_ratios,
        width,
        label="Budget Usage",
        color=COLORS.muted_orange,
        hatch=HATCH_FORWARD,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        zorder=3,
    )

    ax.set_ylabel("Questions Used / Budget", fontsize=LABEL_SIZE - 2)
    ax.set_xlabel("", fontsize=LABEL_SIZE)
    ax.set_xticks(x)

    # Simplified model labels
    simplified_labels = [_simplify_model_label(label) for label in df["model_label"]]
    ax.set_xticklabels(
        simplified_labels,
        fontsize=XTICK_SIZE,
        rotation=30,
        ha="right"
    )

    # Set x-axis limits
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    ax.set_ylim(0, 1.18)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE)
    format_percentage_axis(ax, "y")
    apply_light_y_grid(ax)

    RQ3_OUT.mkdir(parents=True, exist_ok=True)
    apply_base_axes_layout(fig)
    _save_figure(fig, RQ3_OUT / "rq3.2", pdf_only)
    plt.close(fig)


def _plot_radar_by_type(df: pd.DataFrame, pdf_only: bool) -> None:
    """
    Figure 3.3: Recall-only radar chart by ambiguity type for all models.

    Style matches reference: bold type labels, max values shown, no radial percentage labels.
    """
    if df.empty:
        print("Warning: No data for rq3.3, skipping")
        return

    apply_vldb_style()

    # Number of types (axes)
    num_types = len(TYPE_ORDER)
    angles = np.linspace(0, 2 * np.pi, num_types, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Color palette for models
    all_colors = PALETTE_NATURE + PALETTE_MUTED


    # Extract unique models
    unique_models = df[["model", "model_label"]].drop_duplicates().values.tolist()

    def _draw_radar(ax, metric_key: str, metric_name: str, show_legend: bool = False):
        """Draw a single radar chart with labels outside."""
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise

        # Collect max values per type
        max_vals = {t: 0.0 for t in TYPE_ORDER}

        # Draw radar for each model
        for model_idx, (model, label) in enumerate(unique_models):
            model_df = df[(df["model"] == model)]
            values = []
            for t in TYPE_ORDER:
                type_row = model_df[model_df["ambiguity_type"] == t]
                if not type_row.empty:
                    val = type_row.iloc[0][metric_key]
                    values.append(val)
                    if val > max_vals[t]:
                        max_vals[t] = val
                else:
                    values.append(0.0)

            values_closed = values + values[:1]  # Close the polygon

            color = all_colors[model_idx % len(all_colors)]
            ax.plot(
                angles,
                values_closed,
                linewidth=1.2,
                linestyle='-',
                color=color,
                label=label if show_legend else None,
                alpha=0.85,
            )
            ax.fill(angles, values_closed, color=color, alpha=0.06)

        # Keep a small margin beyond the r=0.90 circle
        ax.set_ylim(0, 0.95)
        ax.set_yticks([0.25, 0.5, 0.75, 0.9])
        ax.set_yticklabels([])  # No percentage labels

        # Remove default x-axis labels (we'll add custom ones outside)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])  # Clear default labels

        # Grid styling - only show up to r=0.85
        ax.spines['polar'].set_visible(False)  # Hide outer spine
        ax.grid(color=COLORS.light_gray, linestyle='-', linewidth=0.5, alpha=0.4)

        # Draw a circle at r=0.90 to represent the boundary
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_circle, [0.90]*100, color=COLORS.light_gray, linewidth=0.8, alpha=0.6)

        # No axis labels or max value annotations (per user request)

    # Build a single-panel recall figure (radar only)
    fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 5.5), subplot_kw=dict(projection='polar'))

    _draw_radar(ax1, "recall", "Recall", show_legend=True)

    # Extract legend handles/labels, then remove legend from the radar figure
    handles, labels = ax1.get_legend_handles_labels()
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    RQ3_OUT.mkdir(parents=True, exist_ok=True)
    fig1.tight_layout()
    _save_figure(fig1, RQ3_OUT / "rq3.3.radar", pdf_only, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig1)

    # Build a separate legend-only figure
    fig2 = plt.figure(figsize=(3.2, 6.0))
    ax2 = fig2.add_subplot(111)
    ax2.axis("off")

    n_cols = 1
    n_items = len(handles)
    n_rows = max(1, (n_items + n_cols - 1) // n_cols)
    if n_cols > 1 and n_items > 0:
        order = []
        for r in range(n_rows):
            for c in range(n_cols):
                idx = c * n_rows + r
                if idx < n_items:
                    order.append(idx)
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

    legend = fig2.legend(
        handles, labels,
        loc="center",
        ncol=n_cols,
        frameon=True,
        fancybox=False,
        edgecolor=COLORS.gray,
        fontsize=RADAR_LEGEND_SIZE,
        handlelength=1.8,
        columnspacing=1.0,
        labelspacing=0.6,
    )
    legend.get_frame().set_linewidth(0.8)

    fig2.tight_layout()
    _save_figure(fig2, RQ3_OUT / "rq3.3.legend", pdf_only, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ3 figures")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    # Load plot data and refresh cache
    df_3_1 = _get_rq3_1_plot_data()
    df_3_2 = _get_rq3_2_plot_data()
    df_3_4a = _get_rq3_4a_plot_data()

    # Generate figures
    _plot_pass_rate_by_mode(df_3_1, args.pdf)
    _plot_budget_usage(df_3_2, args.pdf)
    _plot_radar_by_type(df_3_4a, args.pdf)


if __name__ == "__main__":
    main()
