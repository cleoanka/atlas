"""Shared plotting style: white background, large fonts, colorblind-safe (Okabe-Ito) palette.

Every figure imports `setup()` and the OK palette so the repo has one consistent look that
reads at social-media thumbnail size and survives colorblind vision.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette
OK = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}
# 10-class categorical (digits 0-9), colorblind-tuned
DIGITS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2", "#D55E00", "#CC79A7", "#999999", "#8B4513"]

HERE = os.path.dirname(os.path.abspath(__file__))


def setup():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 150,
        "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold",
        "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.3, "lines.linewidth": 2.6, "font.family": "DejaVu Sans",
    })


def save(fig, name):
    setup_dir = HERE
    path = os.path.join(setup_dir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote figures/{name}")
    return path
