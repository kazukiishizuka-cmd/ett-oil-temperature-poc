"""深層学習モデルの実験ランナー。

表形式モデルと同じ fold・同じ指標で評価し、直接比較できるようにする。
Nowcast（負荷のみからOTを推定）は入力にOTが無くRevINが成立しないため、
深層モデルはForecastタスクに限定している。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import ALL_COLS, DATASETS, HORIZONS_HOURLY, RESULT_DIR, TARGET
from data import clean_dataset, load_dataset
from evaluate import regression_metrics
from models_deep import DeepForecaster, make_windows
from splits import expanding_folds

SEQ_LEN = 336  # 14日ぶんの窓
VAL_MONTHS = 2


def run(dataset: str, horizons=None, kinds=("DLinear", "PatchTST"), seq_len: int = SEQ_LEN) -> pd.DataFrame:
    raw = load_dataset(dataset)
    df, miss_mask = clean_dataset(raw)
    values = df[ALL_COLS].to_numpy(dtype=np.float32)
    target = df[TARGET].to_numpy(dtype=np.float32)
    valid_flags = (~miss_mask).to_numpy()
    index = df.index
    folds = expanding_folds(index)
    horizons = horizons or HORIZONS_HOURLY

    records, preds_store = [], []
    steps_per_hour = DATASETS[dataset]["steps_per_day"] / 24.0
    for h in horizons:
        h_steps = int(round(h * steps_per_hour))
        X, y, idx = make_windows(values, target, seq_len, h_steps, valid_flags)
        ts = index[idx]                    # 予測の基準時刻 t
        base = target[idx]                 # OT(t)（Persistence比較用）
        print(f"  窓データ: h={h}  {X.shape}")

        for fold in folds:
            tr = ts <= fold.train_end - pd.Timedelta(hours=h)  # gapは実時間で確保
            te = (ts >= fold.test_start) & (ts <= fold.test_end)
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            cut = ts[tr].max() - pd.DateOffset(months=VAL_MONTHS)
            inner_tr = tr & (ts <= cut)
            inner_val = tr & (ts > cut)

            for kind in kinds:
                t0 = time.time()
                m = DeepForecaster(kind=kind, seq_len=seq_len, horizon=h_steps)
                m.fit(X[inner_tr], y[inner_tr], X[inner_val], y[inner_val])
                pred = m.predict(X[te])
                elapsed = time.time() - t0

                truth = pd.Series(y[te], index=ts[te])
                pr = pd.Series(pred, index=ts[te])
                records.append({
                    "dataset": dataset, "task": "forecast", "horizon": h,
                    "fold": fold.name, "model": kind,
                    **regression_metrics(truth, pr),
                    "fit_seconds": round(elapsed, 1),
                })
                preds_store.append(pd.DataFrame({
                    "timestamp": ts[te], "dataset": dataset, "task": "forecast",
                    "model": kind, "horizon": h, "fold": fold.name,
                    "y_true": truth.to_numpy(), "y_pred": pr.to_numpy(),
                }))
                print(f"  [{dataset}/h={h}/{fold.name}/{kind}] MAE={records[-1]['MAE']:.4f} "
                      f"({elapsed:.0f}s, val={m.best_val_:.4f})")

    out = pd.DataFrame(records)
    if preds_store:
        pd.concat(preds_store).to_csv(RESULT_DIR / f"predictions_{dataset}_deep.csv", index=False)
    out.to_csv(RESULT_DIR / f"metrics_deep_{dataset}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()
    res = run(args.dataset, horizons=args.horizons, seq_len=args.seq_len)
    print("\n=== fold平均 MAE ===")
    print(res.pivot_table(index=["horizon"], columns="model", values="MAE", aggfunc="mean").round(4).to_string())


if __name__ == "__main__":
    main()
