#!/usr/bin/env python3
"""
RQ5: Can LLMs reconstruct auditable workflows?

Figure 5.1: 100% stacked bar chart showing flow outcome distribution.
Figure 5.2: Small multiples showing outcome rates by case complexity per model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import (
    COLORS,
    apply_vldb_style,
    apply_light_y_grid,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    STYLE_EXEC_ERROR,
    STYLE_EXEC_WRONG,
    STYLE_EXEC_CORRECT,
    BASE_FIGSIZE,
    BASE_LABEL_SIZE,
    BASE_XTICK_SIZE,
    BASE_YTICK_SIZE,
    BASE_LEGEND_SIZE,
    BASE_VALUE_LABEL_SIZE,
    BASE_SCALE,
    BASE_LEGEND_Y,
    apply_base_axes_layout,
    FIXED_MODEL_ORDER,
    get_model_display_name,
)


ROOT = Path(__file__).resolve().parents[1]
DRAW_OUT = ROOT / "draw" / "output" / "rq5"
RQ5_DATA_CACHE = ROOT / "draw" / "data" / "rq5"
RQ5_STATUS_PATH = ROOT / "draw" / "data" / "rq5" / "rq5_task_status.csv"

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


def _load_task_status() -> pd.DataFrame:
    """Load task status from CSV."""
    if not RQ5_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing task status: {RQ5_STATUS_PATH}")
    return pd.read_csv(RQ5_STATUS_PATH)


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
    return [m for m in models if not m[0].startswith("_")]


def _simplify_model_label(label: str) -> str:
    """Simplify model label for cleaner display."""
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


def _compute_rq5_1_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 5.1 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)

    # Build DataFrame
    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        outcomes = model_df["flow_outcome"].astype(str).tolist() if "flow_outcome" in model_df.columns else []
        counts = {"execerror": 0, "execok-fail": 0, "execok-pass": 0}
        for outcome in outcomes:
            if outcome in counts:
                counts[outcome] += 1

        total = counts["execerror"] + counts["execok-fail"] + counts["execok-pass"]
        if total > 0:
            execerror_pct = counts["execerror"] / total * 100
            execok_fail_pct = counts["execok-fail"] / total * 100
            execok_pass_pct = counts["execok-pass"] / total * 100
        else:
            execerror_pct = 0.0
            execok_fail_pct = 0.0
            execok_pass_pct = 0.0

        data.append({
            "model": model,
            "model_label": label,
            "execerror_pct": execerror_pct,
            "execok_fail_pct": execok_fail_pct,
            "execok_pass_pct": execok_pass_pct,
            "total": total
        })

    return pd.DataFrame(data)


def _compute_rq5_3_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 5.3 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)
    if "processing_steps" not in df.columns:
        return pd.DataFrame()

    # Use fixed bins for processing steps
    # Bins: 3-5, 6-8, 9-11, 12-18
    bin_edges = [3, 6, 9, 12, 19]  # upper bound is exclusive
    bin_labels = ["3-5", "6-8", "9-11", "12-18"]
    n_actual_bins = len(bin_labels)

    def get_bin(steps: int) -> int:
        for i in range(n_actual_bins):
            if bin_edges[i] <= steps < bin_edges[i + 1]:
                return i
        return -1

    # Build DataFrame
    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        # Group by complexity bin
        bin_counts = {i: {"execerror": 0, "execok-fail": 0, "execok-pass": 0, "total": 0} for i in range(n_actual_bins)}
        for _, row in model_df.iterrows():
            outcome = str(row.get("flow_outcome", ""))
            steps_val = row.get("processing_steps")
            if pd.isna(steps_val):
                continue
            steps = int(steps_val)
            bin_idx = get_bin(steps)
            if bin_idx < 0:
                continue
            if outcome not in bin_counts[bin_idx]:
                continue
            bin_counts[bin_idx][outcome] += 1
            bin_counts[bin_idx]["total"] += 1

        # Add rows for each bin
        for bin_idx in range(n_actual_bins):
            total = bin_counts[bin_idx]["total"]
            if total > 0:
                execerror_pct = bin_counts[bin_idx]["execerror"] / total * 100
                execok_fail_pct = bin_counts[bin_idx]["execok-fail"] / total * 100
                execok_pass_pct = bin_counts[bin_idx]["execok-pass"] / total * 100
            else:
                execerror_pct = float("nan")
                execok_fail_pct = float("nan")
                execok_pass_pct = float("nan")

            data.append({
                "model": model,
                "model_label": label,
                "complexity_bin": bin_idx,
                "bin_label": bin_labels[bin_idx],
                "execerror_pct": execerror_pct,
                "execok_fail_pct": execok_fail_pct,
                "execok_pass_pct": execok_pass_pct,
                "total": total,
                "n_actual_bins": n_actual_bins,
            })

    return pd.DataFrame(data)


def _save_plot_data(df: pd.DataFrame, filename: str) -> None:
    """Save plot data to CSV cache."""
    RQ5_DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = RQ5_DATA_CACHE / filename
    df.to_csv(cache_path, index=False)
    print(f"Cached: {cache_path}")


def _get_rq5_1_plot_data() -> pd.DataFrame:
    """Get RQ5.1 plot data."""
    df = _compute_rq5_1_plot_data()
    _save_plot_data(df, "rq5_1_plot_data.csv")
    return df


def _get_rq5_3_plot_data() -> pd.DataFrame:
    """Get RQ5.3 plot data."""
    df = _compute_rq5_3_plot_data()
    if df.empty:
        raise ValueError("No processing_steps data found. Check draw/data/statistics/processing_steps.csv.")
    _save_plot_data(df, "rq5_3_plot_data.csv")
    return df


def _plot_flow_outcomes(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 5.1: 100% stacked bar chart for flow outcomes."""
    apply_vldb_style()

    fig, ax = plt.subplots(figsize=BASE_FIGSIZE)

    # Comfortable spacing between models
    spacing = 0.8
    x = np.arange(len(df)) * spacing
    width = 0.50

    exec_error_pct = df["execerror_pct"].values
    exec_ok_fail_pct = df["execok_fail_pct"].values
    exec_ok_pass_pct = df["execok_pass_pct"].values
    totals = df["total"].values

    # Stacked bars
    ax.bar(x, exec_error_pct, width, label="Fail",
           color=STYLE_EXEC_ERROR["color"], hatch=STYLE_EXEC_ERROR["hatch"],
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    ax.bar(x, exec_ok_fail_pct, width, bottom=exec_error_pct, label="Wrong",
           color=STYLE_EXEC_WRONG["color"], hatch=STYLE_EXEC_WRONG["hatch"],
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    bottom_correct = exec_error_pct + exec_ok_fail_pct
    ax.bar(x, exec_ok_pass_pct, width, bottom=bottom_correct, label="Correct",
           color=STYLE_EXEC_CORRECT["color"], hatch=STYLE_EXEC_CORRECT["hatch"],
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)

    # Segment labels with scaled font
    model_labels = df["model_label"].tolist()
    for i, (e, w, c, t, model_label) in enumerate(zip(exec_error_pct, exec_ok_fail_pct, exec_ok_pass_pct, totals, model_labels)):
        label_fontsize = BASE_VALUE_LABEL_SIZE
        if e > 12:
            ax.text(x[i], e / 2, f"{e:.0f}%", ha="center", va="center", fontsize=label_fontsize, color="white", fontweight="bold")
        if w > 12:
            fail_color = "white" if "Qwen3-235B" in model_label else COLORS.gray
            ax.text(x[i], e + w / 2, f"{w:.0f}%", ha="center", va="center", fontsize=label_fontsize, color=fail_color, fontweight="bold")
        if c > 12 or "DevStral" in model_label:
            ax.text(x[i], e + w + c / 2, f"{c:.0f}%", ha="center", va="center", fontsize=label_fontsize, color="white", fontweight="bold")

    # Simplified model labels
    simplified_labels = [_simplify_model_label(label) for label in df["model_label"]]
    ax.set_xticks(x)
    ax.set_xticklabels(simplified_labels, fontsize=BASE_XTICK_SIZE, rotation=30, ha="right")

    ax.set_ylabel("Percentage (%)", fontsize=BASE_LABEL_SIZE)
    ax.set_xlabel("Model", fontsize=BASE_LABEL_SIZE)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=BASE_YTICK_SIZE)

    # Legend at top center
    legend_y = BASE_LEGEND_Y + 0.04
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=BASE_LEGEND_SIZE,
    )
    apply_light_y_grid(ax)

    DRAW_OUT.mkdir(parents=True, exist_ok=True)
    apply_base_axes_layout(fig)
    _save_figure(fig, DRAW_OUT / "rq5.1", pdf_only)
    plt.close(fig)


def _plot_complexity_trends(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 5.2: Small multiples showing outcome rates by case complexity per model."""
    if df.empty:
        print("Warning: No data for rq5.2, skipping")
        return

    apply_vldb_style()

    # Extract number of bins
    n_actual_bins = int(df["n_actual_bins"].iloc[0])
    
    # Filter for specific models requested by user
    target_models = [
        "gpt-5.1-codex-max",
        "kimi-k2-thinking",
    ]
    
    # Get unique models from dataframe but filter to keep only targets in correct order
    all_models_in_data = df[["model", "model_label"]].drop_duplicates().values.tolist()
    model_map = {m[0]: m[1] for m in all_models_in_data}
    
    unique_models = []
    for tm in target_models:
        if tm in model_map:
            unique_models.append((tm, model_map[tm]))
            
    n_models = len(unique_models)

    # Create small multiples - 1 row, 2 columns for 2 models
    n_cols = 2
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=BASE_FIGSIZE, sharey=True)
    axes = axes.flatten() if n_models > 1 else [axes]

    # Scaling factor for fonts
    SCALE = BASE_SCALE

    x = np.arange(n_actual_bins)

    for idx, (model, label) in enumerate(unique_models):
        ax = axes[idx]

        # Filter data for this model
        model_df = df[df["model"] == model].sort_values("complexity_bin")

        error_pct = model_df["execerror_pct"].values
        fail_pct = model_df["execok_fail_pct"].values
        pass_pct = model_df["execok_pass_pct"].values
        bin_labels = model_df["bin_label"].values

        # Plot lines
        ax.plot(x, error_pct, marker='o', markersize=6, linewidth=2,
                color=STYLE_EXEC_ERROR["color"], label="Fail")
        ax.plot(x, fail_pct, marker='s', markersize=6, linewidth=2,
                color=STYLE_EXEC_WRONG["color"], label="Wrong")
        ax.plot(x, pass_pct, marker='^', markersize=6, linewidth=2,
                color=STYLE_EXEC_CORRECT["color"], label="Correct")

        # Title as simplified model name
        ax.set_title(_simplify_model_label(label), fontsize=BASE_LABEL_SIZE, fontweight="bold")

        # X-axis
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=BASE_XTICK_SIZE, rotation=30, ha="right")
        ax.set_xlim(-0.3, n_actual_bins - 0.7)

        # Y-axis
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=BASE_YTICK_SIZE)
        if idx % n_cols != 0:
            ax.tick_params(axis="y", labelleft=False)

        # Grid
        apply_light_y_grid(ax)

        # Only leftmost plots get y-label
        if idx % n_cols == 0:
            ax.set_ylabel("Percentage (%)", fontsize=BASE_LABEL_SIZE)

        # All plots get x-label since it's a single row
        ax.set_xlabel("Preparation Steps", fontsize=BASE_LABEL_SIZE)

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Adjust layout to make room for legend at top
    apply_base_axes_layout(fig, wspace=0.15)

    fig.canvas.draw()

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color=STYLE_EXEC_ERROR["color"],
               markersize=8, linewidth=2, label="Fail"),
        Line2D([0], [0], marker='s', color=STYLE_EXEC_WRONG["color"],
               markersize=8, linewidth=2, label="Wrong"),
        Line2D([0], [0], marker='^', color=STYLE_EXEC_CORRECT["color"],
               markersize=8, linewidth=2, label="Correct"),
    ]

    # Shared legend at top - positioned higher to avoid overlap
    legend_y = BASE_LEGEND_Y + 0.18
    leg = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=True,
        fontsize=BASE_LABEL_SIZE,
        handlelength=1.5,
        columnspacing=1.2,
        edgecolor="gray",
        fancybox=False,
    )
    leg.get_frame().set_linestyle("--")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("white")

    _save_figure(fig, DRAW_OUT / "rq5.2", pdf_only)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ5 figures")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    # Load plot data and refresh cache
    df_5_1 = _get_rq5_1_plot_data()
    df_5_3 = _get_rq5_3_plot_data()

    # Generate figures
    _plot_flow_outcomes(df_5_1, args.pdf)
    _plot_complexity_trends(df_5_3, args.pdf)


if __name__ == "__main__":
    main()
