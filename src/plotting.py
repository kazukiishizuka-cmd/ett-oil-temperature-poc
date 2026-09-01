"""図の共通スタイル。

方向は Industrial / spec-sheet。
配色はグレースケール＋機能アクセント1色に絞り、
「注目させたい系列だけに色を与える」ことで序列を作る。
多系列はグレーの濃淡で段階を作り、結論に効く系列だけを琥珀で塗る。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- 地とインク ---------------------------------------------------------------
SURFACE = "#ffffff"        # 貼り込み先の紙面に合わせる
INK = "#15171a"            # 主たるインク
INK_SECONDARY = "#494d52"
INK_MUTED = "#85898e"
RULE = "#d6d7d3"
RULE_STRONG = "#b7b8b4"
GRID = RULE

# --- 機能色（数値と結論にだけ与える） -------------------------------------------
ACCENT = "#a94d13"         # 琥珀。結論に効く系列・強調
ACCENT_PALE = "#eadfd5"
OK = "#2c6349"
BAD = "#94352f"

# --- グレーの段階（多系列の序列づけに使う） ---------------------------------------
G1 = "#c9cac6"             # 最も背景寄り
G2 = "#a4a6a2"
G3 = "#7d8085"
G4 = "#4e5358"
G5 = "#22262a"             # 最も前景寄り
GRAY_STEPS = [G1, G2, G3, G4, G5]

# 期間の帯。塗りではなく地の濃淡で区切る
BAND_TRAIN = "#ececea"
BAND_VAL = "#e3e3e0"
BAND_TEST = "#f0e7de"

# 後方互換のための別名（既存コードが参照している名前を意味に対応づける）
BLUE = G5
ORANGE = ACCENT
AQUA = G3
YELLOW = G2
MAGENTA = G2
GREEN = OK
VIOLET = G4
RED = BAD
CATEGORICAL = [G5, ACCENT, G3, G2, G4, OK, G1, BAD]


def setup_style() -> None:
    """日本語が出るフォントと、罫線の細い抑制的なスタイルを適用する。"""
    rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = SURFACE
    rcParams["axes.facecolor"] = SURFACE
    rcParams["savefig.facecolor"] = SURFACE
    rcParams["axes.edgecolor"] = RULE_STRONG
    rcParams["axes.labelcolor"] = INK_SECONDARY
    rcParams["axes.titlecolor"] = INK
    rcParams["text.color"] = INK
    rcParams["xtick.color"] = INK_SECONDARY
    rcParams["ytick.color"] = INK_SECONDARY
    rcParams["xtick.labelsize"] = 9.5
    rcParams["ytick.labelsize"] = 9.5
    rcParams["grid.color"] = RULE
    rcParams["grid.linewidth"] = 0.7
    rcParams["axes.grid"] = True
    rcParams["axes.grid.axis"] = "y"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.left"] = False
    rcParams["lines.linewidth"] = 1.5
    rcParams["font.size"] = 10.5
    rcParams["axes.titlesize"] = 12
    rcParams["legend.frameon"] = False
    rcParams["legend.fontsize"] = 9.5
    rcParams["figure.dpi"] = 140
    rcParams["savefig.dpi"] = 140
    rcParams["savefig.bbox"] = "tight"


def save(fig, path) -> None:
    """図を保存する。

    背景は貼り込み先の紙面と同じ白で塗る。透明にすると PDF 化のときに
    アルファ付きのまま埋め込まれて圧縮が効かず、ファイルが倍近くに膨らむ。
    """
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")
