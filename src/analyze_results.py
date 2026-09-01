"""実験結果の集計と図の生成。

表形式モデルと深層モデルの結果を1つの表に統合し、
精度指標と、探索的な相対高温の分類指標を出す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import FIGURE_DIR, RESULT_DIR, TARGET
from data import clean_dataset, load_dataset
from evaluate import event_level_metrics, rolling_threshold, threshold_event_metrics
from plotting import (
    ACCENT, ACCENT_PALE, BAD, G1, G2, G3, G4, G5, GRID, INK, INK_MUTED,
    INK_SECONDARY, RULE_STRONG, setup_style, save,
)

# モデルの並び順と色を固定する（系列の同一性が図をまたいで保たれるように）
MODEL_ORDER = ["Persistence", "SeasonalNaive", "Ridge", "PatchTST", "DLinear",
               "LightGBM", "LightGBM+外気温"]
MODEL_COLOR = {
    # 学習の重さの順にグレーを濃くし、結論に効く系列だけを琥珀で塗る
    "SeasonalNaive": G1, "TrainMean": G1, "TrainMean+外気温": G1,
    "Ridge": G2, "Ridge+外気温": G2,
    "PatchTST": G3,
    "DLinear": G4,
    "Persistence": G3, "Persistence+外気温": G3,
    "LightGBM": G5,
    "LightGBM+外気温": ACCENT,
}
HORIZON_LABEL = {1: "1時間先", 24: "24時間先", 168: "1週間先"}


METRIC_FILES = ["metrics_tabular.csv", "metrics_deep_ETTh1.csv", "metrics_external.csv",
                "metrics_ETTh2_base.csv", "metrics_ETTh2_external.csv"]


def load_all_metrics() -> pd.DataFrame:
    frames = []
    for name in METRIC_FILES:
        p = RESULT_DIR / name
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    # 外気温ありの基準線は中身が同一なので、重複を落として1本にまとめる
    dup = df.model.str.endswith("+外気温") & ~df.model.str.startswith("LightGBM")
    return df[~dup].reset_index(drop=True)


def load_all_predictions() -> pd.DataFrame:
    frames = []
    for p in sorted(RESULT_DIR.glob("predictions_*.csv")):
        frames.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = pd.concat(frames, ignore_index=True)
    if "dataset" not in df.columns:
        df["dataset"] = "ETTh1"
    df["dataset"] = df["dataset"].fillna("ETTh1")
    dup = df.model.str.endswith("+外気温") & ~df.model.str.startswith("LightGBM")
    return df[~dup].reset_index(drop=True)


def event_table(preds: pd.DataFrame, dataset: str = "ETTh1") -> pd.DataFrame:
    """相対高温区間の分類性能を、起点時点で計算できる閾値で評価する。

    時刻単位（各タイムスタンプを1件と数える）と
    イベント単位（連続する高温時間帯を1件と数える）の両方を出す。
    実設備の危険イベントや運用警報を直接評価するものではない。
    """
    from config import DATASETS
    spd = DATASETS[dataset]["steps_per_day"]
    step_hours = 24.0 / spd
    df, _ = clean_dataset(load_dataset(dataset))
    thr = rolling_threshold(df[TARGET], steps_per_day=spd)
    rows = []
    fc = preds[(preds.task == "forecast") & (preds.dataset == dataset)]
    for (h, model), g in fc.groupby(["horizon", "model"]):
        g = g.sort_values("timestamp")
        # 判定は「基準時刻tで計算できる閾値」で行う
        t = thr.reindex(g["timestamp"]).to_numpy()
        keep = ~np.isnan(t)
        ts = pd.DatetimeIndex(g["timestamp"].to_numpy()[keep])
        y_true = pd.Series(g["y_true"].to_numpy()[keep], index=ts)
        y_pred = pd.Series(g["y_pred"].to_numpy()[keep], index=ts)
        thr_s = pd.Series(t[keep], index=ts)
        m = threshold_event_metrics(y_true.reset_index(drop=True),
                                    y_pred.reset_index(drop=True),
                                    thr_s.reset_index(drop=True))
        ev = event_level_metrics(y_true, y_pred, thr_s, horizon_hours=float(h),
                                 step_hours=step_hours)
        rows.append({"horizon": h, "model": model, **m, **ev})
    return pd.DataFrame(rows)


# --- 図 ---------------------------------------------------------------------

def fig_model_comparison(metrics: pd.DataFrame) -> None:
    """ホライズン別のMAE比較。Persistenceを基準線として引く。"""
    fc = metrics[(metrics.task == "forecast") & (metrics.dataset == "ETTh1")]
    piv = fc.pivot_table(index="horizon", columns="model", values="MAE", aggfunc="mean")
    models = [m for m in MODEL_ORDER if m in piv.columns]
    horizons = sorted(piv.index)
    x = np.arange(len(horizons))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, m in enumerate(models):
        vals = piv.loc[horizons, m]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.9,
                      color=MODEL_COLOR[m], label=m)
        for b, v in zip(bars, vals):
            # 基準線の破線と数値が重ならないよう、軸の高さに比例した余白を空ける
            ax.text(b.get_x() + b.get_width() / 2, v + piv.to_numpy().max() * 0.018,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    for i, h in enumerate(horizons):
        base = piv.loc[h, "Persistence"]
        ax.plot([x[i] - 0.45, x[i] + 0.45], [base, base], color=INK, lw=1.4, ls="--",
                zorder=5, label="Persistence水準" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_LABEL[h] for h in horizons])
    ax.set_ylabel("MAE（℃, 4分割検証の平均）")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5, columnspacing=1.2, handlelength=1.4)
    ax.margins(y=0.16)
    fig.suptitle("将来の外部気象8変数が完全既知なら / 24時間先 +32% / 1週間先 +36%", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig08_model_comparison.png")


def fig_skill_score(metrics: pd.DataFrame) -> None:
    """Persistence比の改善率。プラスが改善、マイナスが悪化。"""
    fc = metrics[(metrics.task == "forecast") & (metrics.dataset == "ETTh1")]
    piv = fc.pivot_table(index="horizon", columns="model", values="MAE", aggfunc="mean")
    base = piv["Persistence"]
    skill = (1 - piv.div(base, axis=0)) * 100
    models = [m for m in MODEL_ORDER if m in skill.columns and m != "Persistence"]
    horizons = sorted(skill.index)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(horizons))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = skill.loc[horizons, m]
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.9,
                      color=MODEL_COLOR[m], label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -0.6),
                    f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=INK_SECONDARY)
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([HORIZON_LABEL[h] for h in horizons])
    ax.set_ylabel("Persistence比の改善率（%）")
    ax.legend(ncol=3, fontsize=9)
    fig.suptitle("外部気象の完全情報条件では24時間先 +32% / なしでは +0.5%", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig09_skill_score.png")


def fig_forecast_example(preds: pd.DataFrame) -> None:
    """test期間の予測系列。h=24とh=168を並べる。"""
    fc = preds[(preds.task == "forecast") & (preds.fold == "fold4") & (preds.dataset == "ETTh1")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, h in zip(axes, [24, 168]):
        g = fc[fc.horizon == h]
        if g.empty:
            continue
        win_lo = pd.Timestamp("2018-04-01")
        win_hi = pd.Timestamp("2018-04-21")
        truth = g[g.model == "LightGBM"].sort_values("timestamp")
        truth = truth[(truth.timestamp >= win_lo) & (truth.timestamp <= win_hi)]
        ax.plot(truth.timestamp, truth.y_true, color=INK_SECONDARY, lw=2.0, label="実測")
        for m in ["Persistence", "LightGBM"]:
            gm = g[g.model == m].sort_values("timestamp")
            gm = gm[(gm.timestamp >= win_lo) & (gm.timestamp <= win_hi)]
            if not gm.empty:
                ax.plot(gm.timestamp, gm.y_pred, color=MODEL_COLOR[m], lw=1.5,
                        alpha=0.95 if m == "LightGBM" else 0.8,
                        ls="-" if m == "LightGBM" else "--", label=m)
        ax.set_ylabel("油温 OT")
        ax.set_title(HORIZON_LABEL[h], loc="left", fontsize=12)
        ax.legend(ncol=3, loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle("予測は日内の山谷に追随するが水準の急変には遅れる / 2018年4月", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig10_forecast_example.png")


def fig_fold_stability(metrics: pd.DataFrame) -> None:
    """fold別のMAE。特定期間だけの結論でないことを確認する。"""
    fc = metrics[(metrics.task == "forecast") & (metrics.dataset == "ETTh1")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    fold_label = {"fold1": "17年3-6月", "fold2": "17年7-10月", "fold3": "17年11-18年2月", "fold4": "18年3-6月"}
    for ax, h in zip(axes, [1, 24, 168]):
        g = fc[fc.horizon == h]
        folds = sorted(g.fold.unique())
        for m in [m for m in MODEL_ORDER if m in g.model.unique()]:
            vals = [g[(g.fold == f) & (g.model == m)]["MAE"].mean() for f in folds]
            ax.plot(range(len(folds)), vals, marker="o", ms=5, color=MODEL_COLOR[m], label=m)
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([fold_label.get(f, f) for f in folds], rotation=30, ha="right", fontsize=8)
        ax.set_title(HORIZON_LABEL[h], loc="left")
        ax.set_ylabel("MAE（℃）")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("順位は4分割すべてで安定 / 水準は季節で2倍以上動く", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig11_fold_stability.png")


def fig_nowcast(preds: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """負荷だけからOTを推定した結果と、外部気象を足した差。"""
    nc = metrics[metrics.task == "nowcast"]
    piv = nc.pivot_table(index=["dataset", "model"], values="MAE", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})

    # 左: データセット×入力条件のMAE比較
    conditions = [("TrainMean", "平均を答えるだけ"), ("LightGBM", "負荷のみ"),
                  ("LightGBM+外気温", "負荷＋外部気象")]
    colors = [G2, G4, ACCENT]  # 2016 / 2017 / 2018。結論に効く2018年だけ塗る
    x = np.arange(2)
    width = 0.26
    for i, ((key, label), c) in enumerate(zip(conditions, colors)):
        vals = [piv.loc[(ds, key), "MAE"] if (ds, key) in piv.index else np.nan
                for ds in ["ETTh1", "ETTh2"]]
        bars = axes[0].bar(x + (i - 1) * width, vals, width * 0.9, color=c, label=label)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                             va="bottom", fontsize=8.5, color=INK_SECONDARY)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["ETTh1", "ETTh2"])
    axes[0].set_ylabel("MAE（℃, 4分割の平均）")
    axes[0].legend(fontsize=9)
    axes[0].set_title("油温を一切見ずに推定した誤差", loc="left", fontsize=11)

    # 右: ETTh2 の推定系列（外部気象あり）
    g = preds[(preds.task == "nowcast") & (preds.dataset == "ETTh2") & (preds.fold == "fold4")]
    truth = g[g.model == "LightGBM+外気温"].sort_values("timestamp")
    if truth.empty:
        truth = g[g.model == "LightGBM"].sort_values("timestamp")
    axes[1].plot(truth.timestamp, truth.y_true, color=INK_SECONDARY, lw=1.3, label="実測")
    for model, color, lbl in [("LightGBM", G3, "負荷のみ"), ("LightGBM+外気温", ACCENT, "負荷＋外部気象")]:
        gm = g[g.model == model].sort_values("timestamp")
        if not gm.empty:
            axes[1].plot(gm.timestamp, gm.y_pred, color=color, lw=1.2, alpha=0.9, label=lbl)
    axes[1].set_ylabel("油温 OT")
    axes[1].legend(ncol=3, fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_title("ETTh2 test期間の推定値", loc="left", fontsize=11)

    fig.suptitle("外部気象を足したETTh2の正常値推定候補はMAE 2.1℃ / 平均予測の5分の1",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig12_nowcast.png")


def fig_event_detection(events: pd.DataFrame) -> None:
    """探索的な相対高温の分類性能。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, h in zip(axes, [1, 24, 168]):
        g = events[events.horizon == h]
        models = [m for m in MODEL_ORDER if m in g.model.values]
        x = np.arange(len(models))
        rec = [g[g.model == m]["recall"].iloc[0] for m in models]
        pre = [g[g.model == m]["precision"].iloc[0] for m in models]
        ax.bar(x - 0.2, rec, 0.38, color=ACCENT, label="再現率")
        ax.bar(x + 0.2, pre, 0.38, color=G3, label="適合率")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(HORIZON_LABEL[h], loc="left")
    axes[0].set_ylabel("スコア")
    axes[0].legend(fontsize=9)
    fig.suptitle("直近30日の上位5%を超える高温を1時間先なら約8割捉えられる", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig13_event_detection.png")


def fig_importance() -> None:
    """LightGBMが実際に何を見ているかを確認する。"""
    from features import build_forecast_features
    from models_tabular import LightGBMModel
    from splits import expanding_folds

    df, mask = clean_dataset(load_dataset("ETTh1"))
    X = build_forecast_features(df)
    ot = df[TARGET]
    fold = expanding_folds(df.index)[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, h in zip(axes, [24, 168]):
        y = ot.shift(-h) - ot
        valid = (~mask) & y.notna()
        Xv, yv = X[valid], y[valid]
        tr = Xv.index <= fold.train_end - pd.Timedelta(hours=h)
        m = LightGBMModel()
        cut = Xv[tr].index.max() - pd.DateOffset(months=2)
        inner = Xv[tr].index <= cut
        m.fit(Xv[tr][inner], yv[tr][inner], Xv[tr][~inner], yv[tr][~inner])
        imp = m.importance().head(15)[::-1]
        ax.barh(range(len(imp)), imp.values, color=G4, height=0.7)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=8)
        ax.set_xlabel("分割回数")
        ax.set_title(f"{HORIZON_LABEL[h]}の予測", loc="left")
        ax.grid(axis="x")
    fig.suptitle("上位を占めるのはOT自身の履歴と移動統計 / 負荷変数は下位に沈む", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig14_importance.png")


def fig_weather_relation() -> None:
    """候補地点の気温と油温の関係。共通季節性を含む関連を示す。"""
    import warnings
    warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
    from external_features import DEFAULT_CITY, WEATHER_COLS, load_weather

    est = pd.read_csv(RESULT_DIR / "weather_location_estimate.csv")
    w = load_weather(DEFAULT_CITY)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.3)

    # 左上: 候補地点のスコア
    ax = fig.add_subplot(gs[0, 0])
    e2 = est[est.dataset == "ETTh2"].sort_values("score", ascending=False).head(8)[::-1]
    colors = [ACCENT if c == DEFAULT_CITY else G2 for c in e2.city]
    ax.barh(range(len(e2)), e2.score, color=colors, height=0.7)
    ax.set_yticks(range(len(e2))); ax.set_yticklabels(e2.city, fontsize=8.5)
    ax.set_xlabel("整合度スコア")
    ax.set_title("候補16都市との整合度（上位8）", loc="left", fontsize=11)
    ax.grid(axis="x")

    # 中上: 変数ごとの相関（負荷 vs 気象）
    ax = fig.add_subplot(gs[0, 1:])
    df2, _ = clean_dataset(load_dataset("ETTh2"))
    ww = w.reindex(df2.index).interpolate(method="time", limit_direction="both")
    ot = df2[TARGET]
    from config import EXOG
    names, vals, cols = [], [], []
    for c in EXOG:
        names.append(c); vals.append(abs(ot.corr(df2[c]))); cols.append(G1)
    for c in ["temperature_2m", "dew_point_2m", "surface_pressure", "shortwave_radiation"]:
        names.append({"temperature_2m": "外気温", "dew_point_2m": "露点",
                      "surface_pressure": "気圧", "shortwave_radiation": "日射"}[c])
        vals.append(abs(ot.corr(ww[c])))
        cols.append(ACCENT if c == "temperature_2m" else G3)
    order = np.argsort(vals)
    ax.barh(range(len(vals)), [vals[i] for i in order],
            color=[cols[i] for i in order], height=0.7)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("OTとの相関の絶対値（ETTh2）")
    # 基準線はこの図が使っているデータセットの実測から引く（他のデータセットの値を流用しない）
    load_max = max(abs(ot.corr(df2[c])) for c in EXOG)
    ax.axvline(load_max, color=BAD, ls="--", lw=1.2)
    ax.text(load_max + 0.01, 0.5, "データ内の負荷変数の上限", color=BAD, fontsize=8.5, va="bottom")
    ax.set_title("データに入っていた変数と入っていなかった変数", loc="left", fontsize=11)
    ax.grid(axis="x")

    # 下段: 時系列の重ね描き。単位はどちらも℃なので1つの軸に載せる
    # （2軸グラフは目盛りの取り方次第で相関を演出できてしまうため使わない）
    ax = fig.add_subplot(gs[1, :])
    daily_ot = ot.resample("D").mean()
    daily_t = ww["temperature_2m"].resample("D").mean()
    ax.fill_between(daily_ot.index, daily_t, daily_ot, color=ACCENT_PALE, alpha=.85, lw=0,
                    label="差＝負荷による温度上昇ぶん")
    ax.plot(daily_t.index, daily_t, color=ACCENT, lw=1.5, label="外気温（日平均）")
    ax.plot(daily_ot.index, daily_ot, color=INK_SECONDARY, lw=1.5, label="油温 OT（日平均）")
    ax.set_ylabel("温度（℃）")
    gap = (daily_ot - daily_t).mean()
    ax.set_title(f"ETTh2の油温と外気温（{DEFAULT_CITY}）  r = {daily_ot.corr(daily_t):.3f}"
                 f"／平均の差 {gap:+.1f}℃", loc="left", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(ncol=3, loc="upper right", fontsize=9)

    fig.suptitle("候補地点の気温と油温は強く共変 r=0.97 / 共通季節性を含む",
                 x=0.01, ha="left", fontsize=14)
    save(fig, FIGURE_DIR / "fig15_weather_relation.png")


def fig_quantile_tradeoff() -> None:
    """分位点を上げると見逃しは減り、誤報は増える。その交換レートを示す。"""
    path = RESULT_DIR / "metrics_quantile_ETTh1.csv"
    if not path.exists():
        print("  skip: 分位点回帰の結果がない")
        return
    q = pd.read_csv(path)
    agg = q.groupby(["horizon", "model"])[["MAE", "precision", "recall", "f1",
                                           "false_alarm_rate"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    palette = {"LGBM q=0.5": G1, "LGBM q=0.7": G3, "LGBM q=0.8": G4,
               "LGBM q=0.9": ACCENT, "Persistence": G2}

    # 左: 再現率と誤報率のトレードオフ（h=168）
    ax = axes[0]
    g = agg[agg.horizon == 168]
    for _, r in g.iterrows():
        ax.scatter(r.false_alarm_rate, r.recall, s=110, color=palette.get(r.model, G3),
                   zorder=3, edgecolors="white", linewidths=1.5)
        ax.annotate(r.model.replace("LGBM ", ""), (r.false_alarm_rate, r.recall),
                    textcoords="offset points", xytext=(7, -3), fontsize=8.5,
                    color=INK_SECONDARY)
    gs_ = g[g.model.str.startswith("LGBM")].sort_values("false_alarm_rate")
    ax.plot(gs_.false_alarm_rate, gs_.recall, color=G3, lw=1.2, alpha=0.6, zorder=2)
    ax.set_xlabel("誤報率（正常時に警報を出す割合）")
    ax.set_ylabel("再現率（高温を捉えた割合）")
    ax.set_title("1週間先：見逃しと空振りの交換レート", loc="left", fontsize=11)
    ax.set_xlim(left=-0.02)
    ax.margins(y=0.12)
    ax.grid(axis="both")

    # 中: MAEとF1が逆方向に動く
    ax = axes[1]
    for h, marker in [(24, "o"), (168, "s")]:
        g = agg[agg.horizon == h]
        gl = g[g.model.str.startswith("LGBM")].sort_values("model")
        ax.plot(gl.MAE, gl.f1, marker=marker, ms=7, lw=1.4,
                color=G3 if h == 24 else ACCENT, label=HORIZON_LABEL[h])
        base = g[g.model == "Persistence"]
        ax.scatter(base.MAE, base.f1, s=110, color=G2, marker=marker,
                   zorder=3, edgecolors="white", linewidths=1.5)
    ax.set_xlabel("MAE（℃）← 小さいほど高精度")
    ax.set_ylabel("F1（イベント検知）")
    ax.set_title("MAEとF1は同じ方向に動かない", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="both")

    # 右: 分位点別の再現率（棒）
    ax = axes[2]
    models = ["Persistence", "LGBM q=0.5", "LGBM q=0.7", "LGBM q=0.8", "LGBM q=0.9"]
    x = np.arange(len(models))
    for i, h in enumerate([24, 168]):
        g = agg[agg.horizon == h].set_index("model")
        vals = [g.loc[m, "recall"] if m in g.index else np.nan for m in models]
        bars = ax.bar(x + (i - 0.5) * 0.4, vals, 0.36,
                      color=G3 if h == 24 else ACCENT, label=HORIZON_LABEL[h])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("LGBM ", "") for m in models], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("再現率")
    ax.legend(fontsize=9)
    ax.set_title("分位点を上げると見逃しが減る", loc="left", fontsize=11)

    fig.suptitle("同じ評価期間で分位点を変えると相対高温の再現率は16%→81%",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig16_quantile_tradeoff.png")


def fig_dataset_comparison(metrics: pd.DataFrame) -> None:
    """2台の変圧器で結論が変わることを示す。"""
    fc = metrics[metrics.task == "forecast"]
    piv = fc.pivot_table(index=["dataset", "horizon"], columns="model", values="MAE", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9.5, 4))
    horizons = [1, 24, 168]
    x = np.arange(len(horizons))
    for i, ds in enumerate(["ETTh1", "ETTh2"]):
        gains = []
        for h in horizons:
            base = piv.loc[(ds, h), "Persistence"]
            best = min(piv.loc[(ds, h), "LightGBM"], piv.loc[(ds, h), "LightGBM+外気温"])
            gains.append((1 - best / base) * 100)
        bars = ax.bar(x + (i - 0.5) * 0.38, gains, 0.34,
                      color=[G3, ACCENT][i], label=ds)
        for b, v in zip(bars, gains):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.0f}%", ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([HORIZON_LABEL[h] for h in horizons])
    ax.set_ylabel("Persistence比の改善率（%）")
    ax.legend(title="変圧器")
    fig.suptitle("同じ手法でも1時間先の改善幅は+8%と+58% / 1台の検証では判断できない",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig17_dataset_comparison.png")


def fig_forecast_noise() -> None:
    """気象予報の誤差に対する感度。改善幅がどこまで残るかを示す。"""
    path = RESULT_DIR / "metrics_sensitivity_ETTh1.csv"
    if not path.exists():
        print("  skip: 感度分析の結果がない")
        return
    sens = pd.read_csv(path).groupby(["horizon", "noise_std"])["MAE"].mean().reset_index()
    base = pd.read_csv(RESULT_DIR / "metrics_all.csv")
    base = base[(base.task == "forecast") & (base.dataset == "ETTh1") & (base.model == "Persistence")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for h, color, marker in [(24, G3, "o"), (168, ACCENT, "s")]:
        g = sens[sens.horizon == h].sort_values("noise_std")
        p = base[base.horizon == h]["MAE"].mean()
        axes[0].plot(g.noise_std, g.MAE, marker=marker, ms=6, color=color, label=HORIZON_LABEL[h])
        axes[0].axhline(p, color=color, ls="--", lw=1, alpha=.6)
        axes[0].text(3.05, p, f"Persistence {p:.2f}", color=color, fontsize=8, va="center")
        gain = (1 - g.MAE / p) * 100
        axes[1].plot(g.noise_std, gain, marker=marker, ms=6, color=color, label=HORIZON_LABEL[h])
        for x, y in zip(g.noise_std, gain):
            axes[1].text(x, y + 0.8, f"{y:.0f}%", ha="center", fontsize=8, color=INK_SECONDARY)
    # 実際の気温予報の誤差水準
    for ax in axes:
        ax.axvspan(1.0, 3.0, color=ACCENT, alpha=.06, zorder=0)
        ax.set_xlabel("気温予報に与えた誤差の標準偏差（℃）")
        ax.set_xticks([0, 1, 2, 3])
    axes[0].set_ylabel("MAE（℃）")
    axes[0].set_title("誤差を与えたときの予測精度", loc="left", fontsize=11)
    axes[0].legend()
    axes[1].set_ylabel("Persistence比の改善率（%）")
    axes[1].set_title("改善幅がどこまで残るか", loc="left", fontsize=11)
    axes[1].set_ylim(0, 42)
    axes[1].legend()
    axes[1].text(2.0, 4, "実際の気温予報の\n誤差水準", ha="center", fontsize=8.5, color=ACCENT)
    fig.suptitle("実際の予報誤差（1〜3℃）でも改善幅は25〜32%残る", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, FIGURE_DIR / "fig18_forecast_noise.png")


def main() -> None:
    setup_style()
    metrics = load_all_metrics()
    preds = load_all_predictions()

    for ds in ["ETTh1", "ETTh2"]:
        sub = metrics[(metrics.task == "forecast") & (metrics.dataset == ds)]
        if sub.empty:
            continue
        print(f"=== {ds} Forecast（4分割の平均） ===")
        print(sub.pivot_table(index="horizon", columns="model",
                              values=["MAE", "R2"], aggfunc="mean").round(4).to_string())
        print()

    print("=== Nowcast ===")
    nc = metrics[metrics.task == "nowcast"].pivot_table(
        index=["dataset", "model"], values=["MAE", "R2"], aggfunc="mean")
    print(nc.round(4).to_string())

    events = event_table(preds)
    events.to_csv(RESULT_DIR / "event_metrics.csv", index=False)
    print("\n=== 相対高温の探索的分類: 時刻単位（ETTh1） ===")
    print(events[["horizon", "model", "n_hot_steps", "precision", "recall", "f1"]]
          .round(3).to_string(index=False))
    print("\n=== 同: イベント単位（連続する高温時間帯を1件と数える） ===")
    print(events[["horizon", "model", "n_events", "events_detected", "event_recall",
                  "median_lead_hours", "alarms_per_week"]].round(2).to_string(index=False))

    metrics.to_csv(RESULT_DIR / "metrics_all.csv", index=False)
    print("\n図を生成:")
    fig_model_comparison(metrics)
    fig_skill_score(metrics)
    fig_forecast_example(preds)
    fig_fold_stability(metrics)
    fig_nowcast(preds, metrics)
    fig_event_detection(events)
    fig_importance()
    fig_weather_relation()
    fig_quantile_tradeoff()
    fig_dataset_comparison(metrics)
    fig_forecast_noise()


if __name__ == "__main__":
    main()
