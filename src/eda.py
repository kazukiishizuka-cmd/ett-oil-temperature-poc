"""EDA: 図表を outputs/figures に生成する。

このスクリプトが出す図は、そのまま報告スライドのボディになる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import EXOG, FIGURE_DIR, RESULT_DIR, SPLIT_TRAIN_END, SPLIT_VAL_END
from data import load_dataset, describe_integrity
from plotting import (
    ACCENT, BAD, BAND_TEST, BAND_TRAIN, BAND_VAL, G1, G2, G3, G4, GRID, INK,
    INK_MUTED, INK_SECONDARY, setup_style, save,
)

TRAIN_END = pd.Timestamp(SPLIT_TRAIN_END)
VAL_END = pd.Timestamp(SPLIT_VAL_END)


def _shade_splits(ax, index) -> None:
    """train / val / test の期間を淡い帯で塗り分ける。"""
    lo, hi = index.min(), index.max()
    ax.axvspan(lo, TRAIN_END, color=BAND_TRAIN, zorder=0)
    ax.axvspan(TRAIN_END, VAL_END, color=BAND_VAL, zorder=0)
    ax.axvspan(VAL_END, hi, color=BAND_TEST, zorder=0)


def fig_ot_timeline() -> None:
    """OTの全期間推移。ETTh1に強い下降トレンドがあることを示す。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, name in zip(axes, ["ETTh1", "ETTh2"]):
        ot = load_dataset(name)["OT"]
        _shade_splits(ax, ot.index)
        ax.plot(ot.index, ot.values, color=INK_MUTED, lw=0.5, alpha=0.7)
        # 30日移動平均で水準の推移を強調
        ax.plot(ot.index, ot.rolling(24 * 30, center=True, min_periods=100).mean(),
                color=ACCENT if name == "ETTh1" else G4, lw=2.2,
                label="30日移動平均")
        ax.set_ylabel("油温 OT")
        ax.set_title(f"{name}", loc="left", fontsize=12)
        ax.legend(loc="lower left")
    # 分割の帯ラベルは軸の外側（上端）に置き、凡例と重ならないようにする
    for x_pos, label in [(TRAIN_END - pd.Timedelta(days=210), "train"),
                         (TRAIN_END + pd.Timedelta(days=60), "val"),
                         (VAL_END + pd.Timedelta(days=58), "test")]:
        axes[0].annotate(label, xy=(x_pos, 1.02), xycoords=("data", "axes fraction"),
                         color=INK_SECONDARY, ha="center", va="bottom", fontsize=10)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("ETTh1は夏のピークが1年で36℃→21℃に低下、ETTh2は同水準を維持", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig01_ot_timeline.png")


def fig_yearly_shift() -> None:
    """同月の年次比較。季節性では説明できないレベルシフトを可視化する。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, name in zip(axes, ["ETTh1", "ETTh2"]):
        ot = load_dataset(name)["OT"]
        m = ot.groupby([ot.index.year, ot.index.month]).mean()
        for year, color in zip([2016, 2017, 2018], [G2, G4, ACCENT]):
            vals = [m.get((year, mo), np.nan) for mo in range(1, 13)]
            ax.plot(range(1, 13), vals, marker="o", ms=5, color=color, label=str(year))
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("月")
        ax.set_ylabel("月平均 油温 OT")
        ax.set_title(name, loc="left")
        ax.legend(title="年", loc="upper left")
    fig.suptitle("ETTh1は同月比で毎年 5〜15℃低下、ETTh2は年による差がほぼない", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig02_yearly_shift.png")


def fig_missing_block() -> None:
    """負荷6変数が厳密にゼロになる58時間の区間を示す。"""
    df = load_dataset("ETTh1")
    win = df.loc["2016-12-03":"2016-12-10"]
    mask = (df[EXOG] == 0).all(axis=1)
    zero_idx = df.index[mask]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for col, color in zip(["HUFL", "MUFL", "LUFL"], [G2, G3, G4]):
        axes[0].plot(win.index, win[col], color=color, label=col)
    axes[0].axvspan(zero_idx.min(), zero_idx.max(), color="#fbe4e4", zorder=0)
    axes[0].set_ylabel("負荷")
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].plot(win.index, win["OT"], color=INK_SECONDARY)
    axes[1].axvspan(zero_idx.min(), zero_idx.max(), color="#fbe4e4", zorder=0)
    axes[1].set_ylabel("油温 OT")
    axes[1].text(zero_idx.min() + pd.Timedelta(hours=29), win["OT"].max(),
                 "負荷6変数が厳密に0\n（58時間）", color=BAD, ha="center", va="top", fontsize=10)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle("負荷は0なのに油温は変動し続ける＝欠測のゼロ埋め", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig03_missing_block.png")


def fig_quantization() -> None:
    """OTの記録分解能。ETTh1は0.07℃刻みに量子化されている。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, name, color in zip(axes, ["ETTh1", "ETTh2"], [ACCENT, G3]):
        ot = load_dataset(name)["OT"]
        d = np.diff(np.unique(ot.values))
        ax.hist(d, bins=60, color=color, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("隣接するユニーク値の差分（℃）")
        ax.set_ylabel("件数（対数）")
        step = np.median(d)
        ax.set_title(f"{name}  ユニーク値 {len(np.unique(ot.values))} 個 / 中央刻み {step:.4f}℃", loc="left", fontsize=11)
    fig.suptitle("ETTh1のOTは0.07℃格子に量子化＝MAE 0.035℃が原理的な下限", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig04_quantization.png")


def fig_seasonality() -> None:
    """トレンド除去後の日内・曜日パターン。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, color in zip(["ETTh1", "ETTh2"], [G4, ACCENT]):
        ot = load_dataset(name)["OT"]
        dev = ot - ot.rolling(24 * 30, center=True, min_periods=100).mean()
        axes[0].plot(range(24), dev.groupby(dev.index.hour).mean(), marker="o", ms=4, color=color, label=name)
        axes[1].plot(range(7), dev.groupby(dev.index.dayofweek).mean(), marker="o", ms=5, color=color, label=name)
    axes[0].set_xlabel("時刻")
    axes[0].set_ylabel("30日移動平均からの偏差（℃）")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_title("日内パターン", loc="left")
    axes[0].axhline(0, color=GRID, lw=1)
    axes[0].legend()
    axes[1].set_xlabel("曜日")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["月", "火", "水", "木", "金", "土", "日"])
    axes[1].set_title("曜日パターン", loc="left")
    axes[1].axhline(0, color=GRID, lw=1)
    axes[1].legend()
    fig.suptitle("日内変動は数℃規模で明確、曜日効果は1℃未満", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig05_seasonality.png")


def fig_acf() -> None:
    """OTの自己相関。直近値の情報量が支配的であることを示す。"""
    lags = list(range(1, 181))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, color in zip(["ETTh1", "ETTh2"], [G4, ACCENT]):
        ot = load_dataset(name)["OT"]
        acf = [ot.autocorr(k) for k in lags]
        ax.plot(lags, acf, color=color, label=name)
    for k, lbl in [(24, "24h"), (168, "168h")]:
        ax.axvline(k, color=GRID, lw=1, ls="--")
        ax.text(k, 0.62, lbl, color=INK_SECONDARY, ha="center", fontsize=9)
    ax.set_xlabel("ラグ（時間）")
    ax.set_ylabel("自己相関")
    ax.set_ylim(0.6, 1.005)
    ax.legend()
    fig.suptitle("1時間前との相関0.994、1週間前でも0.83〜0.87＝自己履歴が主要な情報源", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig06_acf.png")


def fig_load_vs_ot() -> None:
    """負荷とOTの相関。水準では相関があるが変化量では消える。"""
    df = load_dataset("ETTh1")
    raw = df.corr()["OT"].drop("OT")
    dif = df.diff().corr()["OT"].drop("OT")
    x = np.arange(len(EXOG))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(x - 0.2, raw[EXOG], width=0.38, color=G3, label="水準（生の値）")
    axes[0].bar(x + 0.2, dif[EXOG], width=0.38, color=ACCENT, label="変化量（1階差分）")
    axes[0].set_xticks(x); axes[0].set_xticklabels(EXOG)
    axes[0].set_ylabel("OTとの相関係数")
    axes[0].axhline(0, color=GRID, lw=1)
    axes[0].legend()
    axes[0].set_title("同時刻の負荷とOTの相関", loc="left")
    sub = df.loc[df.index <= TRAIN_END]
    axes[1].scatter(sub["HULL"], sub["OT"], s=3, color=G4, alpha=0.15, linewidths=0)
    axes[1].set_xlabel("HULL（最も相関が高い負荷変数）")
    axes[1].set_ylabel("油温 OT")
    axes[1].set_title(f"散布図（train期間, r={sub['HULL'].corr(sub['OT']):.2f}）", loc="left")
    axes[1].grid(axis="both")
    fig.suptitle("同時刻の負荷はOTの水準を最大r=0.22しか説明せず、変化量では相関が消える", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig07_load_vs_ot.png")


def export_integrity_table() -> None:
    """データ整合性チェックの結果をCSVで残す。"""
    from config import DATASETS
    rows = []
    for name in DATASETS:
        df = load_dataset(name)
        r = describe_integrity(name, df)
        mask = (df[EXOG] == 0).all(axis=1)
        r["all_load_zero_points"] = int(mask.sum())
        r["ot_unique_values"] = int(df["OT"].nunique())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "data_integrity.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    setup_style()
    print("EDA図表を生成:")
    fig_ot_timeline()
    fig_yearly_shift()
    fig_missing_block()
    fig_quantization()
    fig_seasonality()
    fig_acf()
    fig_load_vs_ot()
    print("\nデータ整合性:")
    export_integrity_table()


if __name__ == "__main__":
    main()
