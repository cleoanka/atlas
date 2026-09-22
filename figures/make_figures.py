"""Regenerate every figure. `python figures/make_figures.py` (needs results/*.json first).

Order: static explainers, then the data-driven plots, then the animated cover last (slowest).
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "make_banner",
    "make_fig1_why_euclid_fails",
    "make_fig3_bridge_edge",
    "make_fig4_phase_transition",
    "make_fig5_label_budget",
    "make_fig6_embedding",
    "make_fig7_headline",
    "make_fig8_representation",
    "make_fig2_diffusion",   # cover GIF last (slowest)
]


def main():
    for name in MODULES:
        t0 = time.time()
        print(f"[{name}]")
        importlib.import_module(name).main()
        print(f"  ...{time.time()-t0:.0f}s")
    print("\nall figures written to figures/")


if __name__ == "__main__":
    main()
