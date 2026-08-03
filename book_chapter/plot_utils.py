"""Grayscale-safe matplotlib helpers for the book chapter.

The book may print in black and white, so figures must not encode information by
color alone. ``gray_style`` sets a grayscale palette combined with distinct
linestyles; callers add markers/hatches for further separation.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo"]


def gray_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "image.cmap": "gray",
        "axes.prop_cycle":
            plt.cycler(color=["0.1", "0.4", "0.6", "0.75"]) +
            plt.cycler(linestyle=["-", "--", "-.", ":"]),
    })


def savefig(fig, name):
    """Write outputs/<name>.png (300 dpi) and outputs/<name>.pdf."""
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
