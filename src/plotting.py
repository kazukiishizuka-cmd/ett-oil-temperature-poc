"""図表の共通スタイル。

配色は data-viz の検証済みパレット（light）を使う。
淡色をベースに置き、注目させたい系列だけに彩度を与える方針。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 配色 -------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
BAND_TRAIN = "#eef1f5"
BAND_VAL = "#fdf1e9"
BAND_TEST = "#e9f5f0"


def setup_style() -> None:
    """日本語が出るフォントと、線の細い抑制的なスタイルを適用する。"""
    rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = SURFACE
    rcParams["axes.facecolor"] = SURFACE
    rcParams["savefig.facecolor"] = SURFACE
    rcParams["axes.edgecolor"] = GRID
    rcParams["axes.labelcolor"] = INK_SECONDARY
    rcParams["axes.titlecolor"] = INK
    rcParams["text.color"] = INK
    rcParams["xtick.color"] = INK_SECONDARY
    rcParams["ytick.color"] = INK_SECONDARY
    rcParams["grid.color"] = GRID
    rcParams["grid.linewidth"] = 0.8
    rcParams["axes.grid"] = True
    rcParams["axes.grid.axis"] = "y"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["lines.linewidth"] = 1.6
    rcParams["font.size"] = 11
    rcParams["axes.titlesize"] = 13
    rcParams["legend.frameon"] = False
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 140
    rcParams["savefig.bbox"] = "tight"


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")
