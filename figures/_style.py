"""Shared design system for every figure: one font, one palette, one layout grammar.

Goal: figures that look *designed*, not default-matplotlib — editorial title blocks (a coloured
kicker bar + bold title + muted subtitle), a calm off-white canvas, a restrained colourblind-aware
palette, and generous spacing. Import `setup()`, the palette, `header()` and `save()`.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- font: prefer a designed geometric sans if the system has it ------------------------------
_AVAIL = {f.name for f in _fm.fontManager.ttflist}
FONT = next((f for f in ("Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
             if f in _AVAIL), "DejaVu Sans")

# --- palette -----------------------------------------------------------------------------------
INK = "#1d1d29"        # primary text
SUBINK = "#5b6270"     # secondary text
MUTED = "#9aa1ad"      # captions, grid, inactive
HAIR = "#e7e9f0"       # hairline grid
CANVAS = "#ffffff"     # axes background
PAPER = "#f6f7fb"      # figure background (soft)

BLUE = "#2f6db3"       # class 0 / cool accent
ORANGE = "#e8622c"     # class 1 / warm accent
GREEN = "#2a9d68"      # "good" / wins
RED = "#d1495b"        # "bad" / errors / hurts
GOLD = "#e0a72e"

OK = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "red": RED, "vermillion": ORANGE,
      "sky": BLUE, "grey": MUTED, "gold": GOLD, "ink": INK, "black": INK, "yellow": GOLD}

# distinct, clean 10-class palette (refined tab10)
DIGITS = ["#3b6fb0", "#e8843c", "#4f9d5d", "#cf4b52", "#8a6bbf",
          "#8c6d5b", "#d98cc0", "#7d7f88", "#b7b03a", "#3fb0c4"]


def setup():
    plt.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"],   # per-glyph fallback (arrows etc. → DejaVu)
        "figure.facecolor": PAPER, "axes.facecolor": CANVAS, "savefig.facecolor": PAPER,
        "savefig.dpi": 200, "figure.dpi": 120,
        "text.color": INK, "axes.labelcolor": SUBINK, "axes.edgecolor": "#cfd3dc",
        "xtick.color": SUBINK, "ytick.color": SUBINK,
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1, "lines.linewidth": 2.8, "lines.solid_capstyle": "round",
        "grid.color": HAIR, "grid.linewidth": 1.1, "axes.grid": False,
        "figure.constrained_layout.use": False,
    })


def header(fig, title, subtitle=None, kicker="ATLAS · FEW-LABEL LEARNING", x=0.055, y=0.99,
           accent=BLUE):
    """Editorial title block: a coloured kicker bar + kicker text, bold title, muted subtitle."""
    fig.text(x, y - 0.028, kicker, fontsize=10.5, va="top", ha="left",
             color=accent, fontweight="bold")
    fig.text(x, y - 0.072, title, fontsize=21, va="top", ha="left", color=INK,
             fontweight="bold")
    if subtitle:
        fig.text(x, y - 0.132, subtitle, fontsize=13.5, va="top", ha="left", color=SUBINK)


def footer(fig, text, x=0.055, y=0.015):
    fig.text(x, y, text, fontsize=10.5, va="bottom", ha="left", color=MUTED)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=1.0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return path
