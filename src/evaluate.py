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
        "n_hot_steps": int(actual.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
    }


def event_level_metrics(y_true: pd.Series, y_pred: pd.Series, threshold,
                        horizon_hours: float, step_hours: float = 1.0) -> dict:
    """連続する高温時間帯を1つの設備イベントとして数え直す。

    時刻単位の再現率は「高温だった時間の何割に警報を出せたか」を測る。
    保全部門が知りたいのはそこではなく、
      何件の高温事象に気づけたか / 何時間前に気づけたか / 警報は週に何回鳴るか
    なので、イベント単位でも出す。

    渡される系列のインデックスは**予測起点 t** で、値は t+h の実測と予測。
    したがってここで数えるのは「起点の軸で見た高温区間」であり、
    イベント開始の起点で警報が出ていれば horizon_hours 前に気づけたことになる。
    d ステップ遅れて最初の警報が出た場合のリードタイムは horizon_hours − d。

    検知の定義は「イベント区間内で一度でも警報が出たか」。
    対象時刻の軸でイベント開始より前に出た警報は数えていないため、
    実運用の「開始前に鳴ったか」とは厳密には異なる。資料にもこの定義を明記すること。
    """
    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(y_true.index)
        keep = threshold.notna()
        y_true, y_pred, threshold = y_true[keep], y_pred[keep], threshold[keep]

    actual = (y_true >= threshold).astype(bool)
    predicted = (y_pred >= threshold).astype(bool)

    def _runs(flag: pd.Series):
        grp = (flag != flag.shift()).cumsum()
        return [g for _, g in flag.groupby(grp) if bool(g.iloc[0])]

    events = _runs(actual)
    alarms = _runs(predicted)

    detected, lead_times = 0, []
    for ev in events:
        hit = predicted.reindex(ev.index).fillna(False)
        if hit.any():
            detected += 1
            # イベント開始から数えて何ステップ目で最初の警報が出たか
            delay_steps = int(hit.to_numpy().argmax())
            lead_times.append(horizon_hours - delay_steps * step_hours)

    span_hours = len(y_true) * step_hours
    weeks = span_hours / (24 * 7) if span_hours else float("nan")
    return {
        "n_events": len(events),
        "events_detected": detected,
        "event_recall": detected / len(events) if events else float("nan"),
        "median_lead_hours": float(np.median(lead_times)) if lead_times else float("nan"),
        "n_alarms": len(alarms),
        "alarms_per_week": len(alarms) / weeks if weeks else float("nan"),
        "alarm_hours_ratio": float(predicted.mean()),
    }


def summarize(records: list) -> pd.DataFrame:
    """実験結果のリストを見やすい表にまとめる。"""
    df = pd.DataFrame(records)
    return df.sort_values(["task", "horizon", "MAE"]).reset_index(drop=True)
