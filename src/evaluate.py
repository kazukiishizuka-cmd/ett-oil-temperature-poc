"""評価指標。

精度そのもの（MAE/RMSE）に加えて、
「高温イベントを事前に捉えられるか」という運用側の指標も出す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """MAE / RMSE / R2 と、persistence 比の改善率を計算する。"""
    e = y_true - y_pred
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "R2": float(1 - (e ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
        "n": int(len(y_true)),
    }


def rolling_threshold(ot: pd.Series, window: int = 720, q: float = 0.95,
                      min_periods: int = 168) -> pd.Series:
    """「直近の平常水準から見て高温」と言える境界を、時点ごとに引く。

    固定閾値は使えない。ETTh1は2年で水準が20℃近く下がっており、
    学習期間の絶対値で線を引くと評価期間で一度も超えないためである
    （実際に最初の設計では高温イベントが0件になった）。
    窓は過去30日ぶんで、時点tまでの観測しか使わないので運用でもそのまま計算できる。
    """
    return ot.rolling(window, min_periods=min_periods).quantile(q)


def threshold_event_metrics(y_true: pd.Series, y_pred: pd.Series, threshold) -> dict:
    """閾値超過を「警報を出すべき事象」とみなした時の検知性能。

    予防保全では、℃単位の誤差そのものより
    「危険水準に入る前に気づけるか」が意思決定に効く。
    threshold はスカラーでも、時点ごとに動く Series でもよい。
    """
    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(y_true.index)
        keep = threshold.notna()
        y_true, y_pred, threshold = y_true[keep], y_pred[keep], threshold[keep]
    actual = y_true >= threshold
    predicted = y_pred >= threshold
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0 else float("nan")
    return {
        "threshold": float(np.mean(threshold)),
        "n_events": int(actual.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
    }


def summarize(records: list) -> pd.DataFrame:
    """実験結果のリストを見やすい表にまとめる。"""
    df = pd.DataFrame(records)
    return df.sort_values(["task", "horizon", "MAE"]).reset_index(drop=True)
