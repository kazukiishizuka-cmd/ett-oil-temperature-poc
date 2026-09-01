"""粒度換算の回帰テスト。

ラグ・移動窓・周期・系列長・閾値の窓は、すべて時間または日数で指定して
データ粒度に応じた行数へ換算している。ETTm（1行15分）で「24時間前」の
つもりが「6時間前」になる事故を防ぐため、換算結果をここで固定する。

実行: PYTHONPATH=src python src/test_granularity.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATASETS
from data import clean_dataset, load_dataset
from evaluate import rolling_threshold
from external_features import DEFAULT_CITY, load_weather, weather_features
from features import _to_steps, build_forecast_features, build_nowcast_features
from models_tabular import SeasonalNaiveDelta

FAILED = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: {got}" + ("" if ok else f"  (期待 {want})"))
    if not ok:
        FAILED.append(label)


def main() -> None:
    print("=== 時間→行数の換算 ===")
    check("ETTh 24時間", _to_steps((24,), 24)[0], 24)
    check("ETTm 24時間", _to_steps((24,), 96)[0], 96)
    check("ETTh 168時間", _to_steps((168,), 24)[0], 168)
    check("ETTm 168時間", _to_steps((168,), 96)[0], 672)

    print("\n=== 油温・負荷の特徴量 ===")
    for ds in ["ETTh1", "ETTm1"]:
        spd = DATASETS[ds]["steps_per_day"]
        df, _ = clean_dataset(load_dataset(ds))
        f = build_forecast_features(df, steps_per_day=spd)
        n = build_nowcast_features(df, steps_per_day=spd)
        lag24 = int(round(24 * spd / 24))
        col = f"ot_lag{lag24}"
        ot = df["OT"]
        same = np.allclose(f[col].dropna(), ot.shift(lag24).dropna())
        check(f"{ds} {col} が24時間前", bool(same), True)
        check(f"{ds} 列数（forecast/nowcast）", (f.shape[1], n.shape[1]), (162, 191))

    print("\n=== 気象特徴量 ===")
    for ds in ["ETTh1", "ETTm1"]:
        spd = DATASETS[ds]["steps_per_day"]
        df, _ = clean_dataset(load_dataset(ds))
        h_steps = int(round(24 * spd / 24))
        wx = weather_features(df.index, horizon=h_steps, steps_per_day=spd)
        raw = load_weather(DEFAULT_CITY)["temperature_2m"].reindex(df.index).interpolate(
            method="time", limit_direction="both")
        col = f"wx_obs_temperature_2m_lag{h_steps}"
        m = raw.shift(h_steps).notna()
        check(f"{ds} {col} が24時間前", bool(np.allclose(wx[col][m], raw.shift(h_steps)[m])), True)
        m2 = raw.shift(-h_steps).notna()
        check(f"{ds} wx_fc_temperature_2m が24時間先",
              bool(np.allclose(wx["wx_fc_temperature_2m"][m2], raw.shift(-h_steps)[m2])), True)

    print("\n=== SeasonalNaive の参照ラグ ===")
    for ds, want in [("ETTh1", 24), ("ETTm1", 96)]:
        spd = DATASETS[ds]["steps_per_day"]
        m = SeasonalNaiveDelta(horizon=24, steps_per_hour=spd / 24)
        check(f"{ds} h=24時間で参照する行", m.back_now, want)

    print("\n=== 高温閾値の窓 ===")
    for ds in ["ETTh1", "ETTm1"]:
        spd = DATASETS[ds]["steps_per_day"]
        df, _ = clean_dataset(load_dataset(ds))
        thr = rolling_threshold(df["OT"], steps_per_day=spd)
        # 30日ぶん貯まるまで値が出ない
        first = thr.first_valid_index()
        elapsed_days = (first - df.index[0]).total_seconds() / 86400
        check(f"{ds} 最小期間が7日", round(elapsed_days), 7)

    print("\n=== ETTh1 が従来と一致すること ===")
    df, _ = clean_dataset(load_dataset("ETTh1"))
    legacy = df["OT"].rolling(720, min_periods=168).quantile(0.95)
    now = rolling_threshold(df["OT"], steps_per_day=24)
    check("閾値が旧実装（720行固定）と一致",
          bool(np.allclose(legacy.dropna(), now.dropna())), True)

    print()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} 件が期待と異なる: {FAILED}")
    print("すべて期待どおり")


if __name__ == "__main__":
    main()
