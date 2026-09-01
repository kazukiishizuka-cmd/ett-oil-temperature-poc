"""特徴量エンジニアリング。

Nowcast（同時刻推定）と Forecast（将来予測）で使える情報が異なるため、
生成関数を分けている。列名の接頭辞で由来が分かるようにした。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXOG, TARGET

# 欠測のゼロ埋め区間を判定する閾値（負荷6変数がすべて厳密に0）
def missing_mask(df: pd.DataFrame) -> pd.Series:
    """負荷6変数が厳密にゼロの行＝欠測のゼロ埋めとみなすマスクを返す。"""
    return (df[EXOG] == 0).all(axis=1)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """時刻・曜日・年内位置を周期エンコードする。

    木の分割でも扱えるよう生の整数も残し、線形モデル向けに sin/cos も持たせる。
    """
    out = pd.DataFrame(index=index)
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek
    doy = index.dayofyear
    out["cal_hour"] = hour
    out["cal_dow"] = dow
    out["cal_month"] = index.month
    out["cal_is_weekend"] = (dow >= 5).astype(int)
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def _finalize(X: pd.DataFrame) -> pd.DataFrame:
    """特徴量テーブルの後処理。

    ±inf をNaNに寄せる。木モデルは inf を分割点として扱えてしまうが、
    線形モデルは標準化の時点で分散が壊れて発散するため、入口で潰しておく。
    """
    return X.replace([np.inf, -np.inf], np.nan)


def _lags(s: pd.Series, lags, prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_lag{k}": s.shift(k) for k in lags}, index=s.index)


def _rollings(s: pd.Series, windows, prefix: str) -> pd.DataFrame:
    out = {}
    for w in windows:
        r = s.rolling(w, min_periods=max(2, w // 4))
        out[f"{prefix}_rmean{w}"] = r.mean()
        out[f"{prefix}_rstd{w}"] = r.std()
        out[f"{prefix}_rmin{w}"] = r.min()
        out[f"{prefix}_rmax{w}"] = r.max()
    return pd.DataFrame(out, index=s.index)


def build_forecast_features(
    df: pd.DataFrame,
    steps_per_day: int = 24,
    ot_lags=(0, 1, 2, 3, 6, 12, 24, 23, 48, 167, 168),
    exog_lags=(0, 1, 2, 3, 6, 12, 24),
    windows=(3, 6, 24, 168),
) -> pd.DataFrame:
    """将来予測用の特徴量。t=T までに観測済みの情報だけを使う。

    OT の自己履歴（ラグ・移動統計・変化率）が主役で、負荷は補助情報として入る。
    """
    parts = [calendar_features(df.index)]

    ot = df[TARGET]
    parts.append(_lags(ot, ot_lags, "ot"))
    parts.append(_rollings(ot, windows, "ot"))

    # 変化量・変化率（水準ではなく動きを捉えるため）
    diffs = {}
    for k in (1, 2, 3, 6, 12, 24, 168):
        diffs[f"ot_diff{k}"] = ot.diff(k)
    diffs["ot_diff1_diff1"] = ot.diff(1).diff(1)  # 加速度
    parts.append(pd.DataFrame(diffs, index=df.index))

    # 直近の水準からの乖離（トレンド非定常に対する正規化）
    dev = {}
    for w in (24, 168, 720):
        dev[f"ot_dev{w}"] = ot - ot.rolling(w, min_periods=w // 4).mean()
    parts.append(pd.DataFrame(dev, index=df.index))

    for col in EXOG:
        s = df[col]
        parts.append(_lags(s, exog_lags, col.lower()))
        parts.append(_rollings(s, (24, 168), col.lower()))
        parts.append(pd.DataFrame({f"{col.lower()}_diff1": s.diff(1),
                                   f"{col.lower()}_diff24": s.diff(steps_per_day)}, index=df.index))

    # 負荷の合成量（総負荷・有効/無効比）
    total = df[["HUFL", "MUFL", "LUFL"]].sum(axis=1)
    useless = df[["HULL", "MULL", "LULL"]].sum(axis=1)
    agg = pd.DataFrame({
        "load_useful_total": total,
        "load_useless_total": useless,
        "load_total": total + useless,
        "load_ratio": (useless / total.abs().clip(lower=1.0)).clip(-20, 20),
    }, index=df.index)
    parts.append(agg)
    parts.append(_rollings(agg["load_total"], (24, 168), "load_total"))

    return _finalize(pd.concat(parts, axis=1))


def build_nowcast_features(
    df: pd.DataFrame,
    steps_per_day: int = 24,
    exog_lags=(0, 1, 2, 3, 6, 12, 24, 48),
    windows=(3, 6, 24, 168),
) -> pd.DataFrame:
    """同時刻推定用の特徴量。OT の自己履歴を一切使わない。

    t=T の負荷を使ってよいという課題の許可を、ここで本質的に活用する。
    油温は熱容量で遅れて追随するため、負荷の移動平均・積算が効くはず、という仮説。
    """
    parts = [calendar_features(df.index)]

    for col in EXOG:
        s = df[col]
        parts.append(_lags(s, exog_lags, col.lower()))
        parts.append(_rollings(s, windows, col.lower()))
        parts.append(pd.DataFrame({f"{col.lower()}_diff1": s.diff(1),
                                   f"{col.lower()}_diff24": s.diff(steps_per_day)}, index=df.index))

    total = df[["HUFL", "MUFL", "LUFL"]].sum(axis=1)
    useless = df[["HULL", "MULL", "LULL"]].sum(axis=1)
    agg = pd.DataFrame({
        "load_useful_total": total,
        "load_useless_total": useless,
        "load_total": total + useless,
        "load_ratio": (useless / total.abs().clip(lower=1.0)).clip(-20, 20),
    }, index=df.index)
    parts.append(agg)
    # 熱の蓄積を表す積算負荷（指数移動平均を時定数違いで複数）
    ewm = {}
    for span in (6, 24, 72, 168, 720):
        ewm[f"load_total_ewm{span}"] = agg["load_total"].ewm(span=span, min_periods=span // 4).mean()
    parts.append(pd.DataFrame(ewm, index=df.index))
    parts.append(_rollings(agg["load_total"], windows, "load_total"))

    return _finalize(pd.concat(parts, axis=1))
