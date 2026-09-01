"""実験ランナー。

Nowcast（同時刻推定）と Forecast（将来予測）の2タスクを、
共通の分割・共通の指標で回して outputs/results に書き出す。
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from config import DATASETS, HORIZONS_HOURLY, RESULT_DIR, TARGET
from data import clean_dataset, load_dataset
from evaluate import regression_metrics
from external_features import DEFAULT_CITY, holiday_flags, weather_features
from features import build_forecast_features, build_nowcast_features
from models_tabular import (
    LightGBMModel, RidgeModel, SeasonalNaiveDelta, TrainMean, ZeroDelta,
)
from splits import expanding_folds

# NumPy 2.0 + macOS Accelerate の matmul が偽のFP例外フラグを立てるため抑制する。
# 係数・予測値が有限であることは確認済み（値そのものは正常）。
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

VAL_MONTHS = 2  # fold内でearly stoppingに使う末尾の期間


def _inner_val_split(X_tr: pd.DataFrame, y_tr: pd.Series):
    """学習区間の末尾をearly stopping用の検証に回す。"""
    if len(X_tr) == 0:
        return X_tr, y_tr, X_tr, y_tr
    cut = X_tr.index.max() - pd.DateOffset(months=VAL_MONTHS)
    tr = X_tr.index <= cut
    if tr.sum() < 500 or (~tr).sum() < 100:
        return X_tr, y_tr, None, None
    return X_tr[tr], y_tr[tr], X_tr[~tr], y_tr[~tr]


def _build_models(task: str, horizon: int, steps_per_hour: float = 1.0) -> list:
    if task == "forecast":
        return [
            ZeroDelta(),
            SeasonalNaiveDelta(horizon=horizon, steps_per_hour=steps_per_hour),
            RidgeModel(alpha=10.0),
            LightGBMModel(),
        ]
    return [
        TrainMean(),
        RidgeModel(alpha=10.0, predicts_delta=False),
        LightGBMModel(predicts_delta=False),
    ]


def run_task(dataset: str, task: str, horizons=None, save_predictions: bool = True,
             use_external: bool = False, external_city: str = DEFAULT_CITY) -> pd.DataFrame:
    raw = load_dataset(dataset)
    df, miss_mask = clean_dataset(raw)
    ot = df[TARGET]

    if task == "forecast":
        X_all = build_forecast_features(df, steps_per_day=DATASETS[dataset]["steps_per_day"])
        horizons = horizons or HORIZONS_HOURLY
    else:
        X_all = build_nowcast_features(df, steps_per_day=DATASETS[dataset]["steps_per_day"])
        horizons = [0]

    base_X = X_all
    holidays = holiday_flags(df.index) if use_external else None
    # ホライズンは時間で受け取る。行数へは粒度で換算する
    # （ETTmは1行15分なので、24時間先は96行先になる）
    steps_per_hour = DATASETS[dataset]["steps_per_day"] / 24.0

    folds = expanding_folds(df.index)
    records, preds_store = [], []

    for h in horizons:
        h_steps = int(round(h * steps_per_hour))
        if task == "forecast":
            y_level = ot.shift(-h_steps)
            y_target = y_level - ot          # Δを学習ターゲットにする
        else:
            y_level = ot
            y_target = ot

        if use_external:
            # 気象特徴量はホライズンごとに作り直す。
            # 予報として使う列は予測対象時刻 t+h の値を起点 t の行へ割り当てる必要がある。
            X_all = pd.concat([
                base_X,
                weather_features(df.index, city=external_city, horizon=h_steps,
                                 steps_per_day=DATASETS[dataset]["steps_per_day"]),
                holidays,
            ], axis=1)
        else:
            X_all = base_X

        # 欠測ゼロ埋め区間と、ターゲットが欠ける行を除外
        valid = (~miss_mask) & y_target.notna() & y_level.notna()
        if task == "forecast":
            # 予測先が欠測区間に入る行も外す
            valid &= ~miss_mask.shift(-h_steps).fillna(False).astype(bool)
        X = X_all[valid]
        yt = y_target[valid]
        yl = y_level[valid]
        base = ot[valid]

        for fold in folds:
            tr = X.index <= fold.train_end - pd.Timedelta(hours=h)  # gapは実時間で確保
            te = (X.index >= fold.test_start) & (X.index <= fold.test_end)
            X_tr, y_tr = X[tr], yt[tr]
            X_te = X[te]
            if len(X_tr) < 1000 or te.sum() == 0:
                continue

            X_fit, y_fit, X_val, y_val = _inner_val_split(X_tr, y_tr)

            for model in _build_models(task, h, steps_per_hour):
                t0 = time.time()
                # early stopping を使わないモデルには学習区間を削らず全量を渡す
                if model.needs_validation:
                    model.fit(X_fit, y_fit, X_val, y_val)
                else:
                    model.fit(X_tr, y_tr)
                raw_pred = model.predict(X_te)
                elapsed = time.time() - t0

                pred_level = base[te].to_numpy() + raw_pred if model.predicts_delta else raw_pred
                pred_level = pd.Series(pred_level, index=X_te.index)
                truth = yl[te]

                records.append({
                    "dataset": dataset, "task": task, "horizon": h,
                    "fold": fold.name,
                    "model": model.name + ("+外気温" if use_external else ""),
                    **regression_metrics(truth, pred_level),
                    "fit_seconds": round(elapsed, 2),
                })
                if save_predictions:
                    preds_store.append(pd.DataFrame({
                        "timestamp": X_te.index, "dataset": dataset, "task": task,
                        "model": model.name + ("+外気温" if use_external else ""),
                        "horizon": h, "fold": fold.name,
                        "y_true": truth.to_numpy(), "y_pred": pred_level.to_numpy(),
                    }))
            print(f"  [{dataset}/{task}/h={h}/{fold.name}] "
                  + " ".join(f"{r['model']}:{r['MAE']:.3f}" for r in records[-len(_build_models(task, h, steps_per_hour)):]))

    out = pd.DataFrame(records)
    if preds_store:
        suffix = "_external" if use_external else ""
        pd.concat(preds_store).to_csv(
            RESULT_DIR / f"predictions_{dataset}_{task}{suffix}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ETTh1"])
    ap.add_argument("--tasks", nargs="+", default=["forecast", "nowcast"])
    ap.add_argument("--external", action="store_true", help="外気温・祝日を特徴量に加える")
    ap.add_argument("--out", default=None, help="出力するCSV名")
    args = ap.parse_args()

    all_rows = []
    for ds in args.datasets:
        for task in args.tasks:
            print(f"\n=== {ds} / {task}{' / 外気温あり' if args.external else ''} ===")
            all_rows.append(run_task(ds, task, use_external=args.external))
    res = pd.concat(all_rows, ignore_index=True)
    path = RESULT_DIR / (args.out or ("metrics_external.csv" if args.external else "metrics_tabular.csv"))
    res.to_csv(path, index=False)
    print(f"\n保存: {path}  ({len(res)} 行)")

    print("\n=== fold平均 MAE ===")
    piv = res.pivot_table(index=["dataset", "task", "horizon"], columns="model", values="MAE", aggfunc="mean")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
