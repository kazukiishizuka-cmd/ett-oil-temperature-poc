"""表形式モデル（ベースライン・線形・勾配ブースティング）。

すべて fit(X, y) / predict(X) の共通インターフェースに揃え、
run_experiment.py から同じループで回せるようにしている。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from config import RANDOM_SEED


class BaseModel:
    name = "base"
    #: 予測がターゲットの差分（Δ）を対象にするか、水準そのものかを表す
    predicts_delta = True
    #: early stopping のために学習区間の末尾を検証に回す必要があるか。
    #: Falseのモデルには学習データを削らずに全量を渡す（比較の公平性のため）。
    needs_validation = False

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ZeroDelta(BaseModel):
    """Persistence。「h時間後も今と同じ温度」＝Δを常に0と予測する。

    自己相関が極めて高い系列では、これが侮れない強さを持つ。
    """
    name = "Persistence"

    def fit(self, X, y, X_val=None, y_val=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class SeasonalNaiveDelta(BaseModel):
    """季節ナイーブ。「前日の同時刻に起きた変化が今日も起きる」と仮定する。

    Δ(t→t+h) ≈ OT(t+h-24k) - OT(t-24k) を、観測済みのラグ列だけで構成する。
    """
    name = "SeasonalNaive"

    def __init__(self, horizon: int, period_hours: int = 24, steps_per_hour: float = 1.0):
        """horizon と period は時間で受け取り、参照するラグ列は行数へ換算する。

        ETTm のように1行が15分のデータでは、周期24を行数として使うと
        「前日の同時刻」ではなく「6時間前」を見てしまう。
        """
        self.horizon = horizon
        self.period_hours = period_hours
        k = int(np.ceil(horizon / period_hours))
        # t 時点で観測済みにするため period*k だけ遡る
        self.back_now = int(round(period_hours * k * steps_per_hour))
        self.back_future = int(round((period_hours * k - horizon) * steps_per_hour))

    def fit(self, X, y, X_val=None, y_val=None):
        self.cols_ = (f"ot_lag{self.back_now}", f"ot_lag{self.back_future}")
        missing = [c for c in self.cols_ if c not in X.columns]
        if missing:
            raise KeyError(f"SeasonalNaive に必要な列がない: {missing}")
        return self

    def predict(self, X):
        c_now, c_fut = self.cols_
        return np.nan_to_num(X[c_fut].to_numpy() - X[c_now].to_numpy())


class TrainMean(BaseModel):
    """Nowcast用のベースライン。学習期間の平均油温を常に返す。"""
    name = "TrainMean"
    predicts_delta = False

    def fit(self, X, y, X_val=None, y_val=None):
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


class RidgeModel(BaseModel):
    """線形回帰（L2正則化）。標準化と欠損補完を含むパイプライン。"""
    name = "Ridge"

    def __init__(self, alpha: float = 10.0, predicts_delta: bool = True):
        self.alpha = alpha
        self.predicts_delta = predicts_delta

    def fit(self, X, y, X_val=None, y_val=None):
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=self.alpha, random_state=RANDOM_SEED)),
        ])
        self.pipe_.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe_.predict(X)


class LightGBMModel(BaseModel):
    """勾配ブースティング木。early stopping に fold 内の検証区間を使う。"""
    name = "LightGBM"
    needs_validation = True

    def __init__(self, predicts_delta: bool = True, **params):
        self.predicts_delta = predicts_delta
        self.params = {
            "objective": "l1",          # MAEを直接最適化（外れ値に引っ張られにくい）
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "n_estimators": 3000,
            "verbose": -1,
            "seed": RANDOM_SEED,
            "n_jobs": -1,
        }
        self.params.update(params)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model_ = lgb.LGBMRegressor(**self.params)
        if X_val is not None and len(X_val) > 0:
            self.model_.fit(
                X, y, eval_set=[(X_val, y_val)], eval_metric="l1",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            self.best_iteration_ = self.model_.best_iteration_
        else:
            self.params["n_estimators"] = 500
            self.model_ = lgb.LGBMRegressor(**self.params)
            self.model_.fit(X, y)
            self.best_iteration_ = 500
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def importance(self) -> pd.Series:
        return pd.Series(self.model_.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
