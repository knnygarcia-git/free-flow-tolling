"""
One visual style for every chart in the project.

White background, Poppins font (in assets/fonts, OFL license), a quiet grid, and a left-aligned
title with one line of explanation under it. The palette was checked with a color-blindness and
contrast validator:

  GREEN  highlight, the result that matters (the model, Paraná, "real data")
  BLUE   second series
  AMBER  third series, and anything that is an assumption or a simulation
  GRAY   context and baseline (never used alone to identify a series)
  ALERT  only for alarms and failures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "assets" / "fonts"

GREEN, BLUE, AMBER = "#0b8457", "#2563eb", "#d97706"
GRAY, LIGHT_GRAY = "#a3aca8", "#e3e8e5"
ALERT = "#c2410c"
INK, INK_2, INK_3 = "#101814", "#47524c", "#7c8781"
GRID = "#eef1ef"
BACKGROUND = "#ffffff"


def apply() -> str:
    family = "DejaVu Sans"
    for ttf in sorted(FONTS.glob("Poppins-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
        family = "Poppins"
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "font.weight": "regular",
        "text.color": INK,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": LIGHT_GRAY,
        "axes.linewidth": 1,
        "axes.labelcolor": INK_3,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.facecolor": BACKGROUND,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "axes.formatter.use_locale": False,
        "text.parse_math": False,          # keeps "R$" from turning into a formula
    })
    return family


def title(fig, text: str, sub: str | None = None):
    """Left-aligned title above the plot area, with one line of explanation under it.
    It sits outside the figure and gets into the image through savefig's 'tight' bbox."""
    from matplotlib.transforms import offset_copy
    fig.canvas.draw()
    x0 = min(ax.get_tightbbox(fig.canvas.get_renderer()).x0 for ax in fig.axes) / fig.bbox.width
    y_sub = offset_copy(fig.transFigure, fig=fig, y=8, units="points")
    y_title = offset_copy(fig.transFigure, fig=fig, y=26 if sub else 10, units="points")
    fig.text(x0, 1.0, text, ha="left", va="bottom", fontsize=14, fontweight="semibold",
             color=INK, transform=y_title)
    if sub:
        fig.text(x0, 1.0, sub, ha="left", va="bottom", fontsize=9.5, color=INK_3,
                 transform=y_sub)


def grid_x(ax):
    """For horizontal bars: vertical grid instead of horizontal."""
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(LIGHT_GRAY)


def num(n: float, decimals: int = 0) -> str:
    """Number with thousands separator: 12,345.6"""
    return f"{n:,.{decimals}f}"


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


apply()
