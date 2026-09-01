"""ETTデータの読み込みと基礎的な整形。"""
from __future__ import annotations

import pandas as pd

from config import ALL_COLS, DATA_DIR, DATASETS, EXOG


def load_dataset(name: str) -> pd.DataFrame:
    """指定したETTデータセットを DatetimeIndex 付きの DataFrame として返す。"""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name}. choose from {list(DATASETS)}")
    path = DATA_DIR / DATASETS[name]["file"]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df[ALL_COLS]


def describe_integrity(name: str, df: pd.DataFrame) -> dict:
    """欠損・重複・タイムスタンプの連続性を点検した結果を辞書で返す。"""
    freq = DATASETS[name]["freq"]
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return {
        "dataset": name,
        "n_rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "expected_rows": len(full_index),
        "missing_timestamps": len(full_index.difference(df.index)),
        "duplicated_timestamps": int(df.index.duplicated().sum()),
        "nan_cells": int(df.isna().sum().sum()),
        "zero_rows_all_cols": int((df == 0).all(axis=1).sum()),
    }


def clean_dataset(df: pd.DataFrame) -> tuple:
    """欠測・計測異常・停止の疑いがあるゼロ区間を補正し、マスクを返す。

    負荷6変数が厳密に0の区間は疑わしい区間として扱う。
    設備ログがないため、欠測・計測異常・実停止のいずれかは確定できない。
    ゼロのまま移動平均に流し込むと前後数日ぶんの特徴量まで汚染されるので、
    いったんNaNにしてから時間方向に線形補間する。評価対象からはこの区間を外す。
    """
    mask = (df[EXOG] == 0).all(axis=1)
    if not mask.any():
        return df.copy(), mask
    cleaned = df.copy()
    cleaned.loc[mask, EXOG] = float("nan")
    cleaned[EXOG] = cleaned[EXOG].interpolate(method="time", limit_direction="both")
    return cleaned, mask
