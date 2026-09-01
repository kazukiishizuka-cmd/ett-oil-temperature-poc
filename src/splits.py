"""時系列の分割ロジック。

hold-out（最後の4ヶ月）と、季節の偏りを打ち消す expanding-window 検証の2本立て。
学習期間と評価期間の間には予測ホライズン分の gap を空け、境界でのリークを防ぐ。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import SPLIT_TRAIN_END, SPLIT_VAL_END


@dataclass(frozen=True)
class Split:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def holdout_split(index: pd.DatetimeIndex) -> dict:
    """train / val / test の3分割（時刻の境界）を返す。"""
    return {
        "train_end": pd.Timestamp(SPLIT_TRAIN_END),
        "val_end": pd.Timestamp(SPLIT_VAL_END),
        "data_end": index.max(),
    }


def expanding_folds(index: pd.DatetimeIndex, n_folds: int = 4, test_months: int = 4) -> list:
    """学習期間を伸ばしながら4ヶ月ずつ評価する fold を作る。

    最終 fold の評価区間は hold-out test と一致するので、
    「たまたま最後の4ヶ月が特殊だっただけ」ではないことを他の fold で確認できる。
    """
    end = index.max().normalize() + pd.Timedelta(days=1)
    folds = []
    for i in range(n_folds):
        # 後ろから n_folds 個ぶんの評価区間を切り出す
        test_end = end - pd.DateOffset(months=test_months * (n_folds - 1 - i))
        test_start = test_end - pd.DateOffset(months=test_months)
        folds.append(Split(
            name=f"fold{i + 1}",
            train_end=test_start - pd.Timedelta(seconds=1),
            test_start=test_start,
            test_end=min(test_end - pd.Timedelta(seconds=1), index.max()),
        ))
    return folds


def apply_split(X: pd.DataFrame, y: pd.Series, split: Split, horizon: int, freq_steps_per_hour: float = 1.0):
    """1つの fold について (X_train, y_train, X_test, y_test) を切り出す。

    train の末尾から horizon 分を落とすことで、
    「学習に使った行のターゲットが評価区間に食い込む」リークを防ぐ。
    """
    gap = pd.Timedelta(hours=horizon / freq_steps_per_hour)
    tr = (X.index <= split.train_end - gap)
    te = (X.index >= split.test_start) & (X.index <= split.test_end)
    return X[tr], y[tr], X[te], y[te]
