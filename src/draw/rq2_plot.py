#!/usr/bin/env python3
"""
RQ2 Paper Figures: Generates the two main paper figures for RQ2.

- rq2.1: RAW_PROFILE vs PROFILE pass rate comparison across all models (grouped bar chart)
- rq2.2: Per-ambiguity-type comparison for GPT-5.1 + DeepSeek (2-panel)

Both figures share consistent styling for publication consistency.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plot_style import (
    COLORS,
    apply_vldb_style,
    apply_light_y_grid,
    format_percentage_axis,
    add_delta_badge,
    BAR_EDGE_COLOR,
    BAR_EDGE_WIDTH,
    HATCH_NONE,
    HATCH_FORWARD,
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
RQ2_OUT = DRAW_OUT / "rq2"
RQ2_DATA_CACHE = ROOT / "draw" / "data" / "rq2"
RQ2_STATUS_PATH = ROOT / "draw" / "data" / "rq2" / "rq2_task_status.csv"

MODES = ("raw_profile", "profile")

# Models for 2.2 combined plot
SELECTED_MODELS = [
    ("gpt-5.1-codex-max", "GPT-5.1-Codex"),
    ("deepseek-v3.2", "DeepSeek-V3.2"),
]

TYPE_ORDER = [
    "Single-table reference",
    "Multi-table alignment",
    "Group-level concept",
    "Row-level concept",
    "Operation incomplete",
    "Operation inconsistent",
    "Operation boundary",
]

TYPE_TO_CIRCLE = {
    "Single-table reference": "(1)",
    "Multi-table alignment": "(2)",
    "Group-level concept": "(3)",
    "Row-level concept": "(4)",
    "Operation incomplete": "(5)",
    "Operation inconsistent": "(6)",
    "Operation boundary": "(7)",
}

# Consistent styling for both plots
RAW_PROFILE_STYLE = {"color": COLORS.slate, "hatch": HATCH_FORWARD}
PROFILE_STYLE = {"color": COLORS.green, "hatch": HATCH_NONE}


def _with_ext(path: Path, ext: str) -> Path:
    return path.parent / f"{path.name}{ext}"


def _save_figure(fig: plt.Figure, output_path: Path, pdf_only: bool) -> None:
    if pdf_only:
        pdf_path = _with_ext(output_path, ".pdf")
        fig.savefig(pdf_path)
        print(f"Saved: {pdf_path}")
    else:
        png_path = _with_ext(output_path, ".png")
        fig.savefig(png_path, dpi=300)
        print(f"Saved: {png_path}")


def _load_task_status() -> pd.DataFrame:
    """Load task status from CSV."""
    if not RQ2_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing task status: {RQ2_STATUS_PATH}")
    return pd.read_csv(RQ2_STATUS_PATH)


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


def _parse_kinds(value: object) -> Set[str]:
    if not isinstance(value, str):
        return set()
    return {v.strip() for v in value.split("|") if v.strip()}


def _load_amb_meta_from_task_status(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """Load ambiguity metadata from task status CSV."""
    meta: Dict[str, Dict[str, object]] = {}
    if "case" not in df.columns:
        return meta
    if "ambiguity_kinds" not in df.columns:
        return {case: {"kinds": set(), "count": 0} for case in df["case"].dropna().unique()}
    for case_name, group in df.groupby("case", dropna=True):
        kind_val = None
        for v in group["ambiguity_kinds"].tolist():
            if isinstance(v, str) and v.strip():
                kind_val = v
                break
        kinds = _parse_kinds(kind_val)
        meta[str(case_name)] = {"kinds": kinds, "count": len(kinds)}
    return meta


def _compute_rq2_1_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 2.1 from task status."""
    df = _load_task_status()
    models = _discover_models_from_status(df)

    data = []
    for model, label in models:
        model_df = df[df["model"] == model]
        raw_profile_vals = _normalize_passed(model_df["raw_profile_passed"])
        profile_vals = _normalize_passed(model_df["profile_passed"])
        valid = raw_profile_vals.notna() & profile_vals.notna()
        total = int(valid.sum())
        raw_profile_pass = int(raw_profile_vals[valid].sum())
        profile_pass = int(profile_vals[valid].sum())
        raw_profile_rate = (raw_profile_pass / total) if total else 0.0
        profile_rate = (profile_pass / total) if total else 0.0
        delta = profile_rate - raw_profile_rate

        data.append({
            "model": model,
            "model_label": label,
            "raw_profile_pass_rate": raw_profile_rate,
            "profile_pass_rate": profile_rate,
            "delta": delta
        })

    return pd.DataFrame(data)


def _compute_rq2_2_plot_data() -> pd.DataFrame:
    """Compute plot data for Figure 2.2 from task status."""
    df = _load_task_status()

    model_cases: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for model, _ in SELECTED_MODELS:
        model_df = df[df["model"] == model]
        raw_profile_vals = _normalize_passed(model_df["raw_profile_passed"])
        profile_vals = _normalize_passed(model_df["profile_passed"])
        valid = raw_profile_vals.notna() & profile_vals.notna()

        case_map: Dict[str, Dict[str, bool]] = {}
        for case_name, raw_val, profile_val in zip(
            model_df.loc[valid, "case"].astype(str).tolist(),
            raw_profile_vals[valid].tolist(),
            profile_vals[valid].tolist(),
        ):
            case_map[case_name] = {
                "raw_profile": bool(raw_val),
                "profile": bool(profile_val),
            }
        model_cases[model] = case_map

    amb_meta = _load_amb_meta_from_task_status(df)

    # Find common cases across selected models
    all_case_sets = [set(model_cases[m].keys()) for m, _ in SELECTED_MODELS if m in model_cases]
    common_cases = set.intersection(*all_case_sets) if all_case_sets else set()

    # Collect stats
    stats: Dict[str, Dict[str, Dict[str, List[bool]]]] = {
        t: {m: {mode: [] for mode in MODES} for m, _ in SELECTED_MODELS}
        for t in TYPE_ORDER
    }

    for case in common_cases:
        kinds = amb_meta.get(case, {}).get("kinds", set())
        if not isinstance(kinds, set):
            kinds = set()
        for t in TYPE_ORDER:
            if t in kinds:
                for model, _ in SELECTED_MODELS:
                    for mode in MODES:
                        stats[t][model][mode].append(model_cases[model].get(case, {}).get(mode, False))

    # Build DataFrame
    data = []
    for model, label in SELECTED_MODELS:
        for amb_type in TYPE_ORDER:
            raw_profile_vals = stats[amb_type][model]["raw_profile"]
            profile_vals = stats[amb_type][model]["profile"]
            raw_profile_rate = (
                sum(1 for v in raw_profile_vals if v) / len(raw_profile_vals)
            ) if raw_profile_vals else float("nan")
            profile_rate = (
                sum(1 for v in profile_vals if v) / len(profile_vals)
            ) if profile_vals else float("nan")

            data.append({
                "model": model,
                "model_label": label,
                "ambiguity_type": amb_type,
                "raw_profile_pass_rate": raw_profile_rate,
                "profile_pass_rate": profile_rate
            })

    return pd.DataFrame(data)


def _save_plot_data(df: pd.DataFrame, filename: str) -> None:
    """Save plot data to CSV cache."""
    RQ2_DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = RQ2_DATA_CACHE / filename
    df.to_csv(cache_path, index=False)
    print(f"Cached: {cache_path}")


def _get_rq2_1_plot_data() -> pd.DataFrame:
    """Get RQ2.1 plot data."""
    df = _compute_rq2_1_plot_data()
    _save_plot_data(df, "rq2_1_plot_data_raw_profile.csv")
    return df


def _get_rq2_2_plot_data() -> pd.DataFrame:
    """Get RQ2.2 plot data."""
    df = _compute_rq2_2_plot_data()
    _save_plot_data(df, "rq2_2_plot_data_raw_profile.csv")
    return df


def _simplify_model_label(label: str) -> str:
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


def plot_rq2_1(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 2.1: RAW_PROFILE vs PROFILE pass rate per model."""
    apply_vldb_style()

    # Ensure proper font rendering
    plt.rcParams["axes.unicode_minus"] = False

    # 2.1 uses different colors from 2.2 for visual distinction
    RQ21_RAW_COLOR = "#7FCDCD"  # Light teal
    RQ21_FULL_COLOR = "#2E8B8B"  # Dark teal
    RQ21_RAW_HATCH = HATCH_FORWARD
    RQ21_FULL_HATCH = HATCH_NONE

    fig, ax = plt.subplots(figsize=BASE_FIGSIZE)

    # Comfortable spacing between models
    spacing = 0.8
    x = np.arange(len(df)) * spacing
    width = 0.32

    raw_profile_vals = df["raw_profile_pass_rate"].values
    profile_vals = df["profile_pass_rate"].values
    deltas = df["delta"].values
    simplified_labels = [_simplify_model_label(label) for label in df["model_label"]]

    # Bars with distinct color
    ax.bar(
        x - width / 2, raw_profile_vals, width,
        label="Orig", color=RQ21_RAW_COLOR, hatch=RQ21_RAW_HATCH,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3,
    )
    ax.bar(
        x + width / 2, profile_vals, width,
        label="Disamb", color=RQ21_FULL_COLOR, hatch=RQ21_FULL_HATCH,
        edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3,
    )

    # Delta badges with % sign - larger font
    delta_offset = 0.07
    for i, (rv, pv, delta) in enumerate(zip(raw_profile_vals, profile_vals, deltas)):
        y_pos = max(rv, pv) + delta_offset
        add_delta_badge(ax, x[i], y_pos, delta, color=COLORS.green, fontsize=12)

    ax.set_ylabel("Accuracy", fontsize=BASE_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(simplified_labels, fontsize=BASE_XTICK_SIZE, rotation=30, ha="right")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    format_percentage_axis(ax, "y", max_val=1.0)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="y", labelsize=BASE_YTICK_SIZE)

    # Legend at top center - scaled to match 2.2, raised higher
    legend_y = BASE_LEGEND_Y + 0.05
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=BASE_LEGEND_SIZE,
    )
    apply_light_y_grid(ax)

    RQ2_OUT.mkdir(parents=True, exist_ok=True)
    apply_base_axes_layout(fig)
    _save_figure(fig, RQ2_OUT / "rq2.1", pdf_only)
    plt.close(fig)


def plot_rq2_2_combined(df: pd.DataFrame, pdf_only: bool) -> None:
    """Figure 2.2: GPT-5.1 vs DeepSeek by ambiguity type."""
    apply_vldb_style()

    # Use parenthesized digits for type labels
    plt.rcParams["axes.unicode_minus"] = False
    current_serif = plt.rcParams.get("font.serif", [])
    if isinstance(current_serif, str):
        current_serif = [current_serif]
    plt.rcParams["font.serif"] = current_serif + [
        "Arial Unicode MS", "Heiti TC", "Hiragino Sans GB", "DejaVu Sans"
    ]

    # Figure: Each subplot has its own y-axis
    fig, axes = plt.subplots(1, 2, figsize=BASE_FIGSIZE, sharey=True)

    n_types = len(TYPE_ORDER)
    x = np.arange(n_types) * 0.5
    width = 0.18  # Thinner bars
    
    for idx, (model_key, model_label) in enumerate(SELECTED_MODELS):
        ax = axes[idx]

        # Filter data for this model
        model_df = df[df["model"] == model_key]

        raw_profile_rates = []
        profile_rates = []
        for amb_type in TYPE_ORDER:
            type_row = model_df[model_df["ambiguity_type"] == amb_type]
            if not type_row.empty:
                raw_profile_rates.append(type_row.iloc[0]["raw_profile_pass_rate"])
                profile_rates.append(type_row.iloc[0]["profile_pass_rate"])
            else:
                raw_profile_rates.append(float("nan"))
                profile_rates.append(float("nan"))

        ax.bar(
            x - width / 2, raw_profile_rates, width,
            color=RAW_PROFILE_STYLE["color"], hatch=RAW_PROFILE_STYLE["hatch"],
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3
        )
        ax.bar(
            x + width / 2, profile_rates, width,
            color=PROFILE_STYLE["color"], hatch=PROFILE_STYLE["hatch"],
            edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3
        )

        # Delta badges (match RQ2.1 styling)
        delta_offset = 0.07
        y_cap = 0.97  # keep inside ylim(0, 1.0)
        for i, (rv, pv) in enumerate(zip(raw_profile_rates, profile_rates)):
            if np.isnan(rv) or np.isnan(pv):
                continue
            y_pos = min(max(rv, pv) + delta_offset, y_cap)
            add_delta_badge(
                ax,
                x[i],
                y_pos,
                pv - rv,
                color=COLORS.green,
                fontsize=9,
                box_pad=0.10,
                rounding_size=0.12,
            )

        circle_labels = [TYPE_TO_CIRCLE[t] for t in TYPE_ORDER]
        ax.set_xticks(x)
        ax.set_xticklabels(circle_labels, fontsize=BASE_XTICK_SIZE, fontname="Arial Unicode MS")
        ax.tick_params(axis="x", length=0, pad=2)

        ax.set_xlabel(model_label, fontsize=BASE_LABEL_SIZE, fontweight="bold", labelpad=6)

        # Both subplots have their own y-axis
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=BASE_LABEL_SIZE)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        format_percentage_axis(ax, "y", max_val=1.0)
        ax.tick_params(axis="y", labelsize=BASE_YTICK_SIZE)

        ax.set_ylim(0, 1.0)
        apply_light_y_grid(ax)
        ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)

    # Reduce inter-panel gap since only left plot shows y labels
    apply_base_axes_layout(fig, wspace=0.12)

    # Legend centered on full figure with dashed border

    legend_elements = [
        Patch(facecolor=RAW_PROFILE_STYLE["color"], hatch=RAW_PROFILE_STYLE["hatch"], edgecolor=BAR_EDGE_COLOR, label="Orig"),
        Patch(facecolor=PROFILE_STYLE["color"], hatch=PROFILE_STYLE["hatch"], edgecolor=BAR_EDGE_COLOR, label="Disamb"),
    ]
    legend_y = BASE_LEGEND_Y + 0.12
    leg = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        fontsize=BASE_LABEL_SIZE,
        handlelength=1.0,
        columnspacing=0.8,
        edgecolor="gray",
        fancybox=False,
        borderpad=0.20,
    )
    leg.get_frame().set_linestyle("--")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_facecolor("white")

    RQ2_OUT.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, RQ2_OUT / "rq2.2", pdf_only)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate RQ2 figures")
    parser.add_argument("--pdf", action="store_true", help="Save PDF only")
    args = parser.parse_args()

    # Load plot data and refresh cache
    df_2_1 = _get_rq2_1_plot_data()
    df_2_2 = _get_rq2_2_plot_data()

    # Generate figures
    plot_rq2_1(df_2_1, args.pdf)
    plot_rq2_2_combined(df_2_2, args.pdf)


if __name__ == "__main__":
    main()
