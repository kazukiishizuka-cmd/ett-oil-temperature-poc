"""気象予報の誤差に対する感度分析。

将来予測の改善幅は「予測対象時刻の気象を完全に知る」oracle条件で測っている。
実運用で入るのは予報なので、気温予報に誤差を与えたとき改善幅がどこまで残るかを見る。

気温にのみ正規ノイズを与える。主要な説明変数であり、℃単位で誤差水準を解釈できるため。
実際の気温予報のMAEは24時間先で1〜2℃、1週間先で2〜3℃程度とされる。
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

from config import DATASETS, RESULT_DIR, TARGET
from data import clean_dataset, load_dataset
from evaluate import regression_metrics
from external_features import DEFAULT_CITY, holiday_flags, weather_features
from features import build_forecast_features
from models_tabular import LightGBMModel, ZeroDelta
from splits import expanding_folds

VAL_MONTHS = 2
NOISE_LEVELS = [0.0, 1.0, 2.0, 3.0]


def run(dataset: str = "ETTh1", horizons=(24, 168)) -> pd.DataFrame:
    df, miss = clean_dataset(load_dataset(dataset))
    ot = df[TARGET]
    spd = DATASETS[dataset]["steps_per_day"]
    sph = spd / 24.0
    base_X = build_forecast_features(df, steps_per_day=spd)
    holidays = holiday_flags(df.index)
    folds = expanding_folds(df.index)
    rows = []

    for h in horizons:
        h_steps = int(round(h * sph))
        y_level = ot.shift(-h_steps)
        y_delta = y_level - ot
        valid = (~miss) & y_delta.notna() & (~miss.shift(-h_steps).fillna(False).astype(bool))

        for noise in NOISE_LEVELS:
            wx = weather_features(df.index, city=DEFAULT_CITY, horizon=h_steps,
                                  steps_per_day=spd, forecast_noise_std=noise)
            X_all = pd.concat([base_X, wx, holidays], axis=1)
            X, yt, yl, base = X_all[valid], y_delta[valid], y_level[valid], ot[valid]

            for fold in folds:
                tr = X.index <= fold.train_end - pd.Timedelta(hours=h)
                te = (X.index >= fold.test_start) & (X.index <= fold.test_end)
                if tr.sum() < 1000 or te.sum() == 0:
                    continue
                cut = X[tr].index.max() - pd.DateOffset(months=VAL_MONTHS)
                inner = X[tr].index <= cut
                m = LightGBMModel()
                m.fit(X[tr][inner], yt[tr][inner], X[tr][~inner], yt[tr][~inner])
                pred = pd.Series(base[te].to_numpy() + m.predict(X[te]), index=X[te].index)
                rows.append({"dataset": dataset, "horizon": h, "noise_std": noise,
                             "fold": fold.name, **regression_metrics(yl[te], pred)})
            mae = np.mean([r["MAE"] for r in rows if r["horizon"] == h and r["noise_std"] == noise])
            print(f"  h={h:3d} 予報誤差±{noise:.0f}℃: MAE {mae:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / f"metrics_sensitivity_{dataset}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    args = ap.parse_args()
    res = run(args.dataset)

    # Persistence を基準に改善率を出す
    m = pd.read_csv(RESULT_DIR / "metrics_all.csv")
    base = m[(m.task == "forecast") & (m.dataset == args.dataset) & (m.model == "Persistence")]
    print("\n=== 予報誤差に対する改善率（Persistence比） ===")
    agg = res.groupby(["horizon", "noise_std"])["MAE"].mean().reset_index()
    for h in sorted(agg.horizon.unique()):
        p = base[base.horizon == h]["MAE"].mean()
        print(f"  {h}時間先（Persistence {p:.3f}）")
        for _, r in agg[agg.horizon == h].iterrows():
            print(f"    予報誤差 ±{r.noise_std:.0f}℃ : MAE {r.MAE:.3f} / 改善 {(1 - r.MAE / p) * 100:+.1f}%")


if __name__ == "__main__":
    main()
