"""プロジェクト共通の設定値。

パス・データ分割・評価ホライズンなど、実験全体で共有する定数をここに集約する。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

for _d in (FIGURE_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- データセット定義 -------------------------------------------------------
DATASETS = {
    "ETTh1": {"file": "ETTh1.csv", "freq": "h", "steps_per_day": 24},
    "ETTh2": {"file": "ETTh2.csv", "freq": "h", "steps_per_day": 24},
    "ETTm1": {"file": "ETTm1.csv", "freq": "15min", "steps_per_day": 96},
    "ETTm2": {"file": "ETTm2.csv", "freq": "15min", "steps_per_day": 96},
}

TARGET = "OT"
EXOG = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
ALL_COLS = EXOG + [TARGET]

# --- 予測設定 ---------------------------------------------------------------
# 運用起点で2つのタスクを併記する（README / スライドの「仮定」節と対応）
#   nowcast : t=T の負荷実測を既知として OT(T) を推定する（正常値推定の候補）
#   forecast: t=T までの情報のみで OT(T+h) を予測する（予防保全の意思決定）
HORIZONS_HOURLY = [1, 24, 168]  # 1時間先 / 1日先 / 1週間先

# --- データ分割 -------------------------------------------------------------
# 時系列順に train / val / test を切る。最後の4ヶ月をhold-out testとする。
SPLIT_TRAIN_END = "2017-10-31 23:59:59"
SPLIT_VAL_END = "2018-02-28 23:59:59"

# 季節の偏りを避けるため、expanding-window のローリング検証も併走させる
N_CV_FOLDS = 4

RANDOM_SEED = 42
