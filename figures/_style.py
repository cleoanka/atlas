"""Shared design system — dark, high-contrast, glowing. One font, one palette, one grammar.

Aim: figures that look designed and modern, not default-matplotlib. Deep near-black canvas,
vivid accents that pop, a soft glow under lines/bars/points, editorial title blocks, generous
space. Import `setup()`, the palette, `header()`, `glow_line`, `glow_bars`, `glow_scatter`, `save()`.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

_AVAIL = {f.name for f in _fm.fontManager.ttflist}
FONT = next((f for f in ("Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")
             if f in _AVAIL), "DejaVu Sans")

# --- dark palette ------------------------------------------------------------------------------
PAPER = "#0d1017"      # figure background (near-black, faint blue)
CANVAS = "#12151f"     # axes panel (a touch lighter → subtle card)
INK = "#f2f5fb"        # primary text (near-white)
SUBINK = "#aab4c8"     # secondary text
MUTED = "#6a7488"      # captions, inactive, grid-ish
HAIR = "#242a38"       # gridlines / spines

BLUE = "#4d9fff"       # class 0 / cool accent
ORANGE = "#ff7a45"     # class 1 / warm accent
GREEN = "#2ee6a6"      # wins / good
RED = "#ff5d6c"        # hurts / errors
GOLD = "#ffcf5c"

OK = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "red": RED, "vermillion": ORANGE,
      "sky": BLUE, "grey": MUTED, "gold": GOLD, "ink": INK, "black": INK, "yellow": GOLD}

# vivid 10-class palette on dark
DIGITS = ["#4d9fff", "#ff7a45", "#2ee6a6", "#ff5d6c", "#b980ff",
          "#ffcf5c", "#ff8ad4", "#7f8aa3", "#00c2d1", "#a3e635"]


def setup():
    plt.rcParams.update({
        "font.family": [FONT, "DejaVu Sans"],
        "figure.facecolor": PAPER, "axes.facecolor": CANVAS, "savefig.facecolor": PAPER,
        "savefig.dpi": 200, "figure.dpi": 120,
        "text.color": INK, "axes.labelcolor": SUBINK, "axes.edgecolor": HAIR,
        "xtick.color": SUBINK, "ytick.color": SUBINK,
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1, "lines.linewidth": 2.8, "lines.solid_capstyle": "round",
        "grid.color": HAIR, "grid.linewidth": 1.0, "axes.grid": False,
    })


def header(fig, title, subtitle=None, kicker="ATLAS · FEW-LABEL LEARNING", x=0.055, y=0.99,
           accent=BLUE):
    """Editorial title block: coloured kicker, bold near-white title, muted subtitle."""
    fig.text(x, y - 0.028, kicker, fontsize=10.5, va="top", ha="left", color=accent,
             fontweight="bold")
    fig.text(x, y - 0.072, title, fontsize=21, va="top", ha="left", color=INK, fontweight="bold")
    if subtitle:
        fig.text(x, y - 0.132, subtitle, fontsize=13.5, va="top", ha="left", color=SUBINK)


def footer(fig, text, x=0.055, y=0.015):
    fig.text(x, y, text, fontsize=10.5, va="bottom", ha="left", color=MUTED)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=1.0)
    ax.set_axisbelow(True)


# --- glow helpers (soft blurred underlay, then the crisp element on top) ------------------------
def glow_line(ax, x, y, color, lw=2.8, glow=9.0, zorder=3, **kw):
    for extra, a in ((glow, 0.05), (glow * 0.6, 0.09), (glow * 0.3, 0.13)):
        ax.plot(x, y, color=color, lw=lw + extra, alpha=a, solid_capstyle="round", zorder=zorder)
    return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=zorder + 1, **kw)


def glow_scatter(ax, x, y, color, s=40, glow=2.6, zorder=3, **kw):
    ax.scatter(x, y, s=s * glow, c=color, alpha=0.12, linewidths=0, zorder=zorder)
    return ax.scatter(x, y, s=s, c=color, zorder=zorder + 1, **kw)


def glow_bars(ax, xs, heights, colors, width=0.6, zorder=3):
    for xi, h, c in zip(xs, heights, colors):
        ax.bar(xi, h, width=width * 1.18, color=c, alpha=0.16, zorder=zorder - 1)  # halo
    return ax.bar(xs, heights, width=width, color=colors, zorder=zorder)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.22, facecolor=PAPER)
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return path
