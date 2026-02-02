#!/usr/bin/env python3
"""
RQ4: ProfileAgent impact on full cases + dirty-type breakdown.

Figure 4.1: Pass rates by setting for all cases (Full vs Full+Profile).
Figure 4.2: Small multiples showing delta (Full+Profile - Full) by category.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    COLORS,
    apply_vldb_style,
    apply_light_y_grid,
    format_percentage_axis,
    add_delta_badge,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    HATCH_FORWARD,
    HATCH_BACKWARD,
    BASE_FIGSIZE,
    BASE_LEGEND_Y,
    apply_base_axes_layout,
    BASE_LABEL_SIZE,
    BASE_XTICK_SIZE,
    BASE_YTICK_SIZE,
    BASE_LEGEND_SIZE,
    FIXED_MODEL_ORDER,
    get_model_display_name,
    PALETTE_NATURE,
    PALETTE_MUTED,
)


ROOT = Path(__file__).resolve().parents[1]
DRAW_OUT = ROOT / "draw" / "output"
RQ4_OUT = DRAW_OUT / "rq4"
RQ4_DATA_CACHE = ROOT / "draw" / "data" / "rq4"
RQ4_STATUS_PATH = ROOT / "draw" / "data" / "rq4" / "rq4_task_status.csv"

# New category order: HI, RI, FV, SV, MV
# HI = Header Irregularity
# RI = Row Irregularity
# FV = Format Variation
# SV = Spelling Variation
# MV = Missing Value
IRREGULAR_CATEGORY_ORDER = ["HI", "RI", "FV", "SV", "MV"]


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
    if not RQ4_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing task status: {RQ4_STATUS_PATH}")
    return pd.read_csv(RQ4_STATUS_PATH)


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


def _parse_categories(value: object) -> List[str]:
    if not isinstance(value, str):
        return []
    return [c.strip().upper() for c in value.split("|") if c.strip()]


def _load_irregular_cases_from_status(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load irregular cases from task status CSV."""
    cases_set: set = set()
    by_type: Dict[str, List[str]] = defaultdict(list)
    if "case" not in df.columns:
        return [], {}
    if "irregular_categories" not in df.columns:
        return [], {}

    for case_name, group in df.groupby("case", dropna=True):
        raw_val = None
        for v in group["irregular_categories"].tolist():
            if isinstance(v, str) and v.strip():
                raw_val = v
                break
        categories = _parse_categories(raw_val)
        if not categories:
            continue
        cases_set.add(str(case_name))
        for category in categories:
            if category not in IRREGULAR_CATEGORY_ORDER:
                continue
            if str(case_name) not in by_type[category]:
                by_type[category].append(str(case_name))

    return sorted(cases_set), dict(by_type)


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


def _compute_rq4_1_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 4.1 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)

    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        full_vals = _normalize_passed(model_df["full_passed"]) if "full_passed" in model_df.columns else pd.Series([], dtype=object)
        profile_vals = _normalize_passed(model_df["profile_passed"]) if "profile_passed" in model_df.columns else pd.Series([], dtype=object)

        full_valid = full_vals.notna()
        profile_valid = profile_vals.notna()

        full_total = int(full_valid.sum())
        profile_total = int(profile_valid.sum())
        full_pass = int(full_vals[full_valid].sum()) if full_total else 0
        profile_pass = int(profile_vals[profile_valid].sum()) if profile_total else 0

        full_rate = (full_pass / full_total) if full_total else 0.0
        profile_rate = (profile_pass / profile_total) if profile_total else 0.0

        data.append({
            "model": model,
            "model_label": label,
            "full_pass_rate": full_rate,
            "profile_pass_rate": profile_rate,
            "delta": profile_rate - full_rate
        })

    return pd.DataFrame(data)


def _compute_rq4_4_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 4.4 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)

    # Load irregular cases
    _, irregular_by_type = _load_irregular_cases_from_status(df)

    # Build lookup: model -> case -> passed
    model_cases: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for model, _ in models:
        model_df = df[df["model"] == model]
        full_vals = _normalize_passed(model_df["full_passed"]) if "full_passed" in model_df.columns else pd.Series([], dtype=object)
        profile_vals = _normalize_passed(model_df["profile_passed"]) if "profile_passed" in model_df.columns else pd.Series([], dtype=object)

        case_map: Dict[str, Dict[str, bool]] = {}
        for case_name, full_val, profile_val in zip(
            model_df["case"].astype(str).tolist(),
            full_vals.tolist(),
            profile_vals.tolist(),
        ):
            case_map[case_name] = {
                "full": full_val if full_val is not None else None,
                "profile": profile_val if profile_val is not None else None,
            }
        model_cases[model] = case_map

    # Compute delta per model per category
    data = []
    for cat in IRREGULAR_CATEGORY_ORDER:
        cases = irregular_by_type.get(cat, [])
        for model, label in models:
            full_passed = []
            profile_passed = []
            for case_name in cases:
                case_entry = model_cases.get(model, {}).get(case_name, {})
                full_p = case_entry.get("full")
                profile_p = case_entry.get("profile")
                if full_p is not None:
                    full_passed.append(bool(full_p))
                if profile_p is not None:
                    profile_passed.append(bool(profile_p))

            full_rate = sum(1 for p in full_passed if p) / len(full_passed) if full_passed else 0
            profile_rate = sum(1 for p in profile_passed if p) / len(profile_passed) if profile_passed else 0
            delta = profile_rate - full_rate

            data.append({
                "model": model,
                "model_label": label,
                "category": cat,
                "full_rate": full_rate,
                "profile_rate": profile_rate,
                "delta": delta
            })

    return pd.DataFrame(data)


def _save_plot_data(df: pd.DataFrame, filename: str) -> None:
    """Save plot data to CSV cache."""
    RQ4_DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = RQ4_DATA_CACHE / filename
    df.to_csv(cache_path, index=False)
    print(f"Cached: {cache_path}")


def _get_rq4_1_plot_data() -> pd.DataFrame:
    """Get RQ4.1 plot data."""
    df = _compute_rq4_1_plot_data()
    _save_plot_data(df, "rq4_1_plot_data.csv")
    return df


def _get_rq4_4_plot_data() -> pd.DataFrame:
    """Get RQ4.4 plot data."""
    df = _compute_rq4_4_plot_data()
    _save_plot_data(df, "rq4_4_plot_data.csv")
    return df


def _plot_full_vs_profile(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 4.1: Grouped bar chart for full vs profile (all cases).

    Style matches RQ2.1 exactly:
    - Same figure dimensions (7.0, 3.7)
    - Scaled fonts (SCALE = 7.0/4.2)
    - Delta badges above bars (no value labels on bars)
    - Y-axis to 100% (not 75% like RQ2.1)
    """
    apply_vldb_style()

    # Ensure proper font rendering
    plt.rcParams["axes.unicode_minus"] = False

    # Use the earlier RQ3.1 blue-gray mapping
    RQ41_FULL_COLOR = COLORS.slate
    RQ41_PROFILE_COLOR = COLORS.muted_blue
    RQ41_FULL_HATCH = HATCH_FORWARD
    RQ41_PROFILE_HATCH = HATCH_BACKWARD

    # Baseline dimensions aligned to RQ4.1
    fig, ax = plt.subplots(figsize=BASE_FIGSIZE)

    # Comfortable spacing between models
    spacing = 0.8
    x = np.arange(len(df)) * spacing
    width = 0.32

    full_vals = df["full_pass_rate"].values
    profile_vals = df["profile_pass_rate"].values
    simplified_labels = [_simplify_model_label(label) for label in df["model_label"]]

    # Bars with same styling as RQ2.1
    ax.bar(
        x - width / 2, full_vals, width,
        label="Disamb", color=RQ41_FULL_COLOR, hatch=RQ41_FULL_HATCH,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3,
    )
    ax.bar(
        x + width / 2, profile_vals, width,
        label="Profile", color=RQ41_PROFILE_COLOR, hatch=RQ41_PROFILE_HATCH,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3,
    )

    # No value labels on bars for cleaner look - only delta badges

    # Delta badges with % sign - larger font (match RQ2.1)
    delta_offset = 0.07
    for i, (fv, pv) in enumerate(zip(full_vals, profile_vals)):
        delta = pv - fv
        y_pos = max(fv, pv) + delta_offset
        add_delta_badge(ax, x[i], y_pos, delta, color=COLORS.green, fontsize=12)

    ax.set_ylabel("Accuracy (%)", fontsize=BASE_LABEL_SIZE)
    ax.set_xticks(x)
    # Model names centered under bars - larger for paper
    ax.set_xticklabels(simplified_labels, fontsize=BASE_XTICK_SIZE, rotation=30, ha="right")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    # Y-axis to 100% (RQ4.1 difference from RQ2.1 which uses 75%)
    format_percentage_axis(ax, "y", max_val=1.10)
    ax.set_ylim(0, 1.10)
    ax.tick_params(axis="y", labelsize=BASE_YTICK_SIZE)

    # Legend at top center - scaled to match RQ2.1, raised higher
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, BASE_LEGEND_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=BASE_LEGEND_SIZE,
    )
    apply_light_y_grid(ax)

    RQ4_OUT.mkdir(parents=True, exist_ok=True)
    apply_base_axes_layout(fig)
    _save_figure(fig, RQ4_OUT / "rq4.1", pdf_only)
    plt.close(fig)


def _plot_delta_small_multiples(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 4.4: Small multiples showing delta (Full+Profile - Full) by category.

    Layout like reference: 5 horizontal panels with vertical bars.
    Each panel: one category, vertical bars for each model.
    Shared legend at top showing all models with colors/hatches.
    """
    apply_vldb_style()

    model_ids = df["model"].unique().tolist()
    model_labels = [_simplify_model_label(df[df["model"] == m]["model_label"].iloc[0]) for m in model_ids]

    # Color and hatch palette for models (like reference image)
    all_colors = PALETTE_NATURE + PALETTE_MUTED
    hatches = ['', '//', '\\\\', 'xx', '..', '++', 'OO', '**', '||', '--']

    # Full category names for display
    cat_full_names = {
        "HI": "Header\nIrregularities",
        "RI": "Row\nIrregularities",
        "FV": "Format\nVariants",
        "SV": "Spelling\nVariants",
        "MV": "Missing\nVariants",
    }

    # Create 5 subplots (1 row x 5 columns) - like reference image
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.2), sharey=True)

    for idx, cat in enumerate(IRREGULAR_CATEGORY_ORDER):
        ax = axes[idx]

        # Get deltas for this category
        cat_df = df[df["category"] == cat]
        cat_deltas = [cat_df[cat_df["model"] == m]["delta"].iloc[0] if m in cat_df["model"].values else 0 for m in model_ids]

        x = np.arange(len(model_ids))
        width = 0.75

        # Draw vertical bars with model-specific colors and hatches
        for i, (model, delta) in enumerate(zip(model_ids, cat_deltas)):
            color = all_colors[i % len(all_colors)]
            hatch = hatches[i % len(hatches)]
            ax.bar(
                i, delta, width,
                color=color, hatch=hatch,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
                label=model_labels[i] if idx == 0 else None,  # Only add label once
            )

        # X-axis: no labels (models are in legend)
        ax.set_xticks([])
        ax.axhline(y=0, color=COLORS.gray, linewidth=0.8, linestyle="-", zorder=2)

        # Full category name as xlabel
        ax.set_xlabel(f"({chr(97 + idx)}) {cat_full_names[cat]}", fontsize=10, fontweight="bold", labelpad=8)

        # Y-axis
        if idx == 0:
            ax.set_ylabel("Δ Accuracy (%)", fontsize=11, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)

        # Set y-axis limits: -15% to +40%
        ax.set_ylim(-0.15, 0.40)

        # Format y-axis as percentage with larger font
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:+.0%}' if y != 0 else '0%'))
        ax.tick_params(axis='y', labelsize=10)

        apply_light_y_grid(ax)

    # Shared legend at top (like reference image) - larger, use more space
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,  # 2 rows of 5
        frameon=True,
        fancybox=False,
        edgecolor=COLORS.gray,
        fontsize=9,
        handlelength=2.0,
        handleheight=1.2,
        columnspacing=0.8,
        labelspacing=0.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for legend
    RQ4_OUT.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, RQ4_OUT / "rq4.2", pdf_only, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ4 figures")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    # Load plot data and refresh cache
    df_4_1 = _get_rq4_1_plot_data()
    df_4_4 = _get_rq4_4_plot_data()

    # Generate figures
    _plot_full_vs_profile(df_4_1, args.pdf)
    _plot_delta_small_multiples(df_4_4, args.pdf)


if __name__ == "__main__":
    main()
