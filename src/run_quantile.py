"""分位点回帰による高温イベント検知の改善検証。

MAEを最小化する学習では、モデルは自信のない場面で平均寄りの無難な値を返す。
その結果、閾値を超える「尖った予測」が出なくなり、警報を出す仕事では見逃しが増える
（実際に h=24 で LightGBM の F1 が Persistence を下回った）。

そこで損失関数を分位点損失に替え、条件付き分布の上側を予測させる。
℃単位の平均誤差は悪化するが、運用上の指標（再現率・F1）が改善するかを確かめる。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import DATASETS, RESULT_DIR, TARGET
from data import clean_dataset, load_dataset
from evaluate import regression_metrics, rolling_threshold, threshold_event_metrics
from external_features import DEFAULT_CITY, holiday_flags, weather_features
from features import build_forecast_features
from models_tabular import LightGBMModel, ZeroDelta
from splits import expanding_folds

QUANTILES = [0.5, 0.7, 0.8, 0.9]
VAL_MONTHS = 2


def run(dataset: str = "ETTh1", horizons=(24, 168), use_external: bool = True) -> pd.DataFrame:
    df, miss = clean_dataset(load_dataset(dataset))
    ot = df[TARGET]
    X_all = build_forecast_features(df, steps_per_day=DATASETS[dataset]["steps_per_day"])
    if use_external:
        X_all = pd.concat([X_all,
                           weather_features(df.index, city=DEFAULT_CITY, future_known=True),
                           holiday_flags(df.index)], axis=1)
    thr_series = rolling_threshold(ot)
    folds = expanding_folds(df.index)
    rows = []

    for h in horizons:
        y_level = ot.shift(-h)
        y_delta = y_level - ot
        valid = (~miss) & y_delta.notna() & (~miss.shift(-h).fillna(False).astype(bool))
        X, yt, yl, base = X_all[valid], y_delta[valid], y_level[valid], ot[valid]

        for fold in folds:
            tr = X.index <= fold.train_end - pd.Timedelta(hours=h)
            te = (X.index >= fold.test_start) & (X.index <= fold.test_end)
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            cut = X[tr].index.max() - pd.DateOffset(months=VAL_MONTHS)
            inner = X[tr].index <= cut
            X_fit, y_fit = X[tr][inner], yt[tr][inner]
            X_val, y_val = X[tr][~inner], yt[tr][~inner]

            truth = yl[te]
            thr = thr_series.reindex(truth.index)

            # 比較対象としてPersistenceも同じ枠で測る
            preds = {"Persistence": pd.Series(base[te].to_numpy(), index=truth.index)}
            for q in QUANTILES:
                m = LightGBMModel(objective="quantile", alpha=q, metric="quantile")
                m.fit(X_fit, y_fit, X_val, y_val)
                preds[f"LGBM q={q}"] = pd.Series(base[te].to_numpy() + m.predict(X[te]), index=truth.index)

            for name, pr in preds.items():
                rows.append({
                    "dataset": dataset, "horizon": h, "fold": fold.name, "model": name,
                    **regression_metrics(truth, pr),
                    **threshold_event_metrics(truth, pr, thr),
                })
            print(f"  [h={h}/{fold.name}] " + " ".join(
                f"{n.replace('LGBM ','')}:F1={rows[-len(preds) + i]['f1']:.3f}" for i, n in enumerate(preds)))

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / f"metrics_quantile_{dataset}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--horizons", nargs="+", type=int, default=[24, 168])
    args = ap.parse_args()
    res = run(args.dataset, horizons=tuple(args.horizons))
    print("\n=== 分位点別の性能（4分割の平均） ===")
    agg = res.groupby(["horizon", "model"])[["MAE", "precision", "recall", "f1", "false_alarm_rate"]].mean()
    print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
