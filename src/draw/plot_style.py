#!/usr/bin/env python3
"""
Shared Matplotlib styling for publication-quality plots.

Design based on AGENTS.md style guide for top-tier academic venues.
Uses PALETTE_NATURE for a "premium" look with consistent hatch patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


@dataclass(frozen=True)
class PlotColors:
    """
    Academic publication color palette based on PALETTE_NATURE from AGENTS.md.
    
    Design principles:
    - 6 main colors for readability
    - Muted but distinctive tones
    - Good contrast for grayscale printing
    """
    # PALETTE_NATURE (Nature/Science journal style) - premium academic look
    # Primary 6 colors ordered by visual weight
    coral: str = "#E64B35"       # Strong red-coral (accent, "Ours", emphasis)
    teal: str = "#4DBBD5"        # Cyan-teal (secondary, good contrast)
    green: str = "#00A087"       # Emerald green (positive, pass, success)
    navy: str = "#3C5488"        # Deep navy blue (stable, baseline)
    peach: str = "#F39B7F"       # Soft peach (warning, intermediate)
    slate: str = "#8491B4"       # Muted slate-blue (neutral)
    
    # Accent for "Ours" emphasis
    accent: str = "#DC0000"      # Deep red accent
    
    # Extended muted palette for multi-series (PALETTE_MUTED)
    muted_orange: str = "#E69F00"
    muted_blue: str = "#56B4E9"
    muted_green: str = "#009E73"
    muted_pink: str = "#CC79A7"
    muted_navy: str = "#0072B2"
    
    # Neutral tones
    gray: str = "#333333"        # Dark gray for text/edges
    medium_gray: str = "#666666" # Medium gray
    light_gray: str = "#C0C0C0"  # Light gray for grid
    very_light_gray: str = "#F0F0F0"
    
    # Special
    white: str = "#FFFFFF"
    black: str = "#000000"


COLORS = PlotColors()

# ============================================================================
# ACADEMIC PALETTES from AGENTS.md
# ============================================================================

PALETTE_NATURE = [COLORS.coral, COLORS.teal, COLORS.green, COLORS.navy, COLORS.peach, COLORS.slate]
PALETTE_MUTED = [COLORS.muted_orange, COLORS.muted_blue, COLORS.muted_green, COLORS.muted_pink, COLORS.muted_navy]
PALETTE_SEQUENTIAL = ["#DEEBF7", "#9ECAE1", "#4292C6", "#2171B5", "#084594"]  # Blue gradient

# ============================================================================
# HATCH PATTERNS for grayscale distinction (per AGENTS.md)
# ============================================================================
# Priority: None (best) > // > \\\\ > xx > .. > --
# Coarse patterns for print readability

HATCH_NONE = ''           # Reserved for "Ours" / Best / Primary
HATCH_FORWARD = '//'      # Secondary (most common baseline)
HATCH_BACKWARD = '\\\\'   # Third variant
HATCH_CROSS = 'xx'        # Fourth (grid pattern)
HATCH_DOT = '..'          # Fifth (sparse dots)
HATCH_HORIZONTAL = '--'   # Sixth (horizontal lines)

# Standard bar styling for academic papers
BAR_EDGE_COLOR = COLORS.black
BAR_EDGE_WIDTH = 0.7

# ============================================================================
# SEMANTIC STYLE MAPPINGS
# ============================================================================
# Consistent mapping for specific semantic meanings across all figures

# RAW vs FULL comparison (RQ2)
STYLE_RAW = {"color": COLORS.peach, "hatch": HATCH_FORWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_FULL = {"color": COLORS.teal, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}

# Pass/Fail outcomes
STYLE_PASS = {"color": COLORS.green, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_FAIL = {"color": COLORS.coral, "hatch": HATCH_CROSS, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}

# P/R/F1 metrics (RQ3)
STYLE_PRECISION = {"color": COLORS.teal, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_RECALL = {"color": COLORS.peach, "hatch": HATCH_FORWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_F1 = {"color": COLORS.slate, "hatch": HATCH_BACKWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}

# QA Classification (RQ3)
STYLE_HIT = {"color": COLORS.green, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_FALLBACK = {"color": COLORS.peach, "hatch": HATCH_FORWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_REFUSE = {"color": COLORS.coral, "hatch": HATCH_CROSS, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}

# Settings comparison (RQ4): Baseline < +Profile < +CleanSpec
STYLE_BASELINE = {"color": COLORS.peach, "hatch": HATCH_CROSS, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_PROFILE = {"color": COLORS.teal, "hatch": HATCH_FORWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_CLEANSPEC = {"color": COLORS.green, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}

# Flow outcomes (RQ5): Error < Wrong < Correct
STYLE_EXEC_ERROR = {"color": COLORS.coral, "hatch": HATCH_CROSS, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_EXEC_WRONG = {"color": COLORS.peach, "hatch": HATCH_FORWARD, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}
STYLE_EXEC_CORRECT = {"color": COLORS.green, "hatch": HATCH_NONE, "edgecolor": BAR_EDGE_COLOR, "linewidth": BAR_EDGE_WIDTH}


def apply_vldb_style() -> None:
    """
    Apply consistent academic paper style to matplotlib.
    Based on AGENTS.md recommendations for top-tier venues.
    """
    plt.rcParams.update({
        # Font settings - serif for academic look (per AGENTS.md Section 5)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        
        # Axes
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS.gray,
        "axes.labelcolor": COLORS.gray,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": COLORS.gray,
        "ytick.color": COLORS.gray,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Legend (per AGENTS.md Section 5)
        "legend.fontsize": 9,
        "legend.frameon": False,      # Cleaner without frame
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        
        # Text
        "text.color": COLORS.gray,
        
        # Grid (per AGENTS.md - very subtle)
        "grid.color": COLORS.light_gray,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        
        # Hatch (per AGENTS.md Section 3)
        "hatch.linewidth": 0.8,
        
        # PDF quality (per AGENTS.md Section 6)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def apply_light_y_grid(ax: plt.Axes) -> None:
    """Add subtle horizontal grid lines (per AGENTS.md)."""
    ax.yaxis.grid(
        True,
        linestyle="--",
        color=COLORS.light_gray,
        alpha=0.3,
        linewidth=0.5,
        zorder=0,
    )
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def apply_light_x_grid(ax: plt.Axes) -> None:
    """Add subtle vertical grid lines."""
    ax.xaxis.grid(
        True,
        linestyle="--",
        color=COLORS.light_gray,
        alpha=0.3,
        linewidth=0.5,
        zorder=0,
    )
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def format_percentage_axis(ax: plt.Axes, axis: str = "y", max_val: float = 1.0) -> None:
    """Format axis as percentage with clean labels."""
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ticks = [t for t in ticks if t <= max_val + 0.01]
    labels = [f"{int(t * 100)}%" for t in ticks]
    
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def add_value_labels(
    ax: plt.Axes,
    bars,
    fmt: str = ".0%",
    offset: int = 3,
    fontsize: int = 8,
    color: str = None,
) -> None:
    """Add value labels on top of bars."""
    if color is None:
        color = COLORS.gray
    
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        if fmt.endswith("%"):
            label = f"{height * 100:.0f}%"
        else:
            label = f"{height:{fmt}}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def add_delta_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    delta: float,
    color: str = None,
    fontsize: int = 7,
    box_alpha: float = 0.15,
    box_pad: float = 0.25,
    rounding_size: float = 0.3,
    transform=None,
) -> None:
    """
    Add a delta annotation with rounded box badge.
    
    Args:
        ax: Matplotlib axes
        x: X position
        y: Y position  
        delta: Delta value (will be formatted as percentage)
        color: Text and box color (default: COLORS.green)
        fontsize: Font size
        box_alpha: Alpha for box background
        transform: Optional transform for annotation coordinates
    """
    if color is None:
        color = COLORS.green
    
    # Convert to percentage
    pp_val = delta * 100
    
    # Format text as arrowed percentage, no space before "%"
    if abs(round(pp_val)) == 0:
        text = "0%"
    else:
        symbol = "↑" if pp_val >= 0 else "↓"
        text = f"{symbol}{abs(pp_val):.0f}%"
    
    ax.annotate(
        text,
        xy=(x, y),
        xycoords=transform if transform is not None else "data",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle=f"round,pad={box_pad},rounding_size={rounding_size}",
            facecolor=color,
            edgecolor=color,
            alpha=box_alpha,
            linewidth=0.8,
        ),
    )


# Standard figure sizes for different layouts (per AGENTS.md Section 5)
FIG_SINGLE = (3.5, 2.5)       # Single column figure
FIG_WIDE = (5.0, 2.8)         # Wide single figure  
FIG_DOUBLE = (7.0, 3.0)       # Two-panel figure
FIG_TRIPLE = (10.0, 3.0)      # Three-panel figure

# Baseline layout for main bar charts (aligned with RQ4.1)
BASE_FIGSIZE = (7.0, 3.7)
BASE_SCALE = 7.0 / 4.2
BASE_LABEL_SIZE = 10.5 * BASE_SCALE
BASE_XTICK_SIZE = 8.0 * BASE_SCALE
BASE_YTICK_SIZE = 9.0 * BASE_SCALE
BASE_LEGEND_SIZE = 9.5 * BASE_SCALE
BASE_VALUE_LABEL_SIZE = 8.0 * BASE_SCALE

# Baseline axes box and legend placement (aligned with RQ4.1)
BASE_AXES_LEFT = 0.10
BASE_AXES_RIGHT = 0.98
BASE_AXES_BOTTOM = 0.24
BASE_AXES_TOP = 0.78
BASE_LEGEND_GAP = 0.06
BASE_LEGEND_Y = BASE_AXES_TOP + BASE_LEGEND_GAP


def apply_base_axes_layout(fig: plt.Figure, wspace: float | None = None) -> None:
    """Apply the shared axes rectangle to keep consistent axes height."""
    kwargs = dict(
        left=BASE_AXES_LEFT,
        right=BASE_AXES_RIGHT,
        bottom=BASE_AXES_BOTTOM,
        top=BASE_AXES_TOP,
    )
    if wspace is not None:
        kwargs["wspace"] = wspace
    fig.subplots_adjust(**kwargs)


# ============================================================================
# MODEL DISCOVERY AND MAPPING
# ============================================================================

# Fixed model order for consistent plotting across all figures
# Order: GPT-5.1-Codex, Kimi K2 Thinking, Claude-Sonnet-4.5, Gemini 3 Flash,
#        GLM-4.7, Qwen3-235B-A22B, DeepSeek-V3.2, Grok Code Fast 1, DevStral 2, GPT-4o
FIXED_MODEL_ORDER = [
    "gpt-5.1-codex-max",
    "kimi-k2-thinking",
    "claude-sonnet-4.5",
    "gemini-3-flash-preview",
    "glm-4.7",
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3.2",
    "grok-code-fast-1",
    "devstral-2512",
    "gpt-4o-2024-11-20",
]


def get_model_display_name(model_id: str) -> str:
    """
    Convert model directory name to display label.
    
    Args:
        model_id: Model directory name (e.g., "deepseek-v3.2")
        
    Returns:
        Display label (e.g., "DeepSeek-V3.2")
    """
    # Model name mapping (optimized for chart readability)
    mapping = {
        "deepseek-v3.2": "DeepSeek-V3.2",
        "grok-code-fast-1": "Grok Code Fast 1",
        "gpt-5.1-codex-max": "GPT-5.1-Codex",
        "gpt-4o-2024-11-20": "GPT-4o",
        "claude-sonnet-4.5": "Claude-Sonnet-4.5",
        "gemini-3-flash-preview": "Gemini 3 Flash",
        "glm-4.7": "GLM-4.7",
        "kimi-k2-thinking": "Kimi K2 Thinking",
        "qwen3-235b-a22b-thinking-2507": "Qwen3-235B-A22B",
        "devstral-2512": "DevStral 2",
    }
    
    # If mapping exists, use it; otherwise format the name nicely
    if model_id in mapping:
        return mapping[model_id]
    
    # Fallback: capitalize and replace hyphens/underscores
    display = model_id.replace("_", " ").replace("-", " ")
    parts = display.split()
    display = " ".join(p.capitalize() for p in parts)
    return display


def discover_models(output_root: str | Path) -> List[Tuple[str, str]]:
    """
    Automatically discover all models from output directory.
    
    Args:
        output_root: Path to @output directory (str or Path)
        
    Returns:
        List of (model_id, display_name) tuples, sorted by FIXED_MODEL_ORDER
    """
    output_path = Path(output_root)
    if not output_path.exists():
        return []
    
    exclude_dirs = {"_cache", "summary"}
    available_models = set()
    for model_dir in output_path.iterdir():
        if (
            model_dir.is_dir()
            and not model_dir.name.startswith(".")
            and model_dir.name not in exclude_dirs
        ):
            available_models.add(model_dir.name)
    
    # Use FIXED_MODEL_ORDER for models that exist, preserving order
    models = []
    for model_id in FIXED_MODEL_ORDER:
        if model_id in available_models:
            display_name = get_model_display_name(model_id)
            models.append((model_id, display_name))
            available_models.discard(model_id)
    
    # Append any remaining models not in FIXED_MODEL_ORDER (sorted alphabetically)
    for model_id in sorted(available_models):
        display_name = get_model_display_name(model_id)
        models.append((model_id, display_name))
    
    return models
