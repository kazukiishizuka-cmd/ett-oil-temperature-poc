"""外部データ（気象・祝日）の読み込みと特徴量化。

データ出典:
  - 気象: Open-Meteo Historical Weather API (model unspecified / Best Match)
          https://open-meteo.com/en/docs/historical-weather-api
          CC BY 4.0 / 再解析・数値気象モデルを組み合わせた既定系列
  - 祝日: 中国国務院公布の法定休日（2016-2018）を手動で定義
          https://www.gov.cn/zhengce/content/  （年度ごとの放假安排通知）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR
from features import _to_steps

EXTERNAL_DIR = DATA_DIR / "external"

#: 観測地点の推定結果。候補16都市とOTの整合度から選んだ（fetch_weather.py 参照）。
#: 選定に使うのは最初の分割の学習期間までで、評価期間の油温は見ない。
DEFAULT_CITY = "Nanjing"

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "shortwave_radiation", "surface_pressure", "precipitation",
                "cloud_cover", "dew_point_2m"]

# 中国の法定連休（国務院公布の放假安排より）。ETTのデータ期間に該当する分だけ定義する。
CHINA_HOLIDAYS = {
    "spring_festival": [("2017-01-27", "2017-02-02"), ("2018-02-15", "2018-02-21")],
    "national_day":    [("2016-10-01", "2016-10-07"), ("2017-10-01", "2017-10-08")],
    "labour_day":      [("2017-04-29", "2017-05-01"), ("2018-04-29", "2018-05-01")],
    "new_year":        [("2016-12-31", "2017-01-02"), ("2017-12-30", "2018-01-01")],
    "qingming":        [("2017-04-02", "2017-04-04"), ("2018-04-05", "2018-04-07")],
    "dragon_boat":     [("2017-05-28", "2017-05-30"), ("2018-06-16", "2018-06-18")],
    "mid_autumn":      [("2016-09-15", "2016-09-17"), ("2018-09-22", "2018-09-24")],
}


def load_weather(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """取得済みの気象データを読む。"""
    path = EXTERNAL_DIR / f"weather_{city}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} がない。先に src/fetch_weather.py を実行すること")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def holiday_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """祝日フラグと、連休からの経過日数を作る。

    公開されている全国休日日程を、設備稼働カレンダーの代理変数として使う。
    実際の操業日・振替出勤・保守停止を表すものではない。
    """
    out = pd.DataFrame(index=index)
    dates = index.normalize()
    is_any = pd.Series(False, index=index)
    for name, spans in CHINA_HOLIDAYS.items():
        flag = pd.Series(False, index=index)
        for lo, hi in spans:
            flag |= (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        out[f"hol_{name}"] = flag.astype(int)
        is_any |= flag
    out["hol_any"] = is_any.astype(int)

    # 春節からの符号つき経過日数（±21日でクリップ）
    sf_days = pd.Series(np.nan, index=index)
    for lo, hi in CHINA_HOLIDAYS["spring_festival"]:
        center = pd.Timestamp(lo)
        d = (dates - center).days.astype(float)
        near = np.abs(d) <= 21
        sf_days[near] = d[near]
    out["hol_days_from_spring_festival"] = sf_days.fillna(99)
    return out


def weather_features(index: pd.DatetimeIndex, city: str = DEFAULT_CITY,
                     horizon: int = 0, steps_per_day: int = 24,
                     obs_lag_hours=(0, 1, 3, 6, 12, 24),
                     window_hours=(6, 24, 168), use_forecast: bool = True,
                     forecast_noise_std: float = 0.0, noise_seed: int = 42) -> pd.DataFrame:
    """気象データを特徴量にする。

    列を2種類に分ける。混ぜると「どの時刻の気象を使っているか」が追えなくなる。

    wx_obs_* : 予測起点 t までに観測済みの実測値。ラグと移動統計はここから作る
    wx_fc_*  : 予測対象時刻 t+horizon の再解析値。実予報ではなく、
               気象8変数を完全に知るoracle条件として上限性能を測る

    horizon=0（ナウキャスト）では対象時刻＝起点なので wx_obs_*_lag0 がそれに当たる。
    課題文の「t=Tの油温を予測する際には t=Tでの特徴量を使用して良い」に対応する。

    ラグ・窓・差分・EWMはすべて時間で指定し、データ粒度に応じて行数へ換算する
    （horizon は呼び出し側で行数に換算済みの値を受け取る）。
    """
    w = load_weather(city).reindex(index).interpolate(method="time", limit_direction="both")
    obs_lags = _to_steps(obs_lag_hours, steps_per_day)
    windows = _to_steps(window_hours, steps_per_day)
    step_1h, step_24h = _to_steps((1,), steps_per_day)[0], _to_steps((24,), steps_per_day)[0]
    ewm_spans = _to_steps((6, 24, 72, 168), steps_per_day)
    parts = []

    # --- 起点までの実測 ---
    for col in WEATHER_COLS:
        s = w[col]
        d = {f"wx_obs_{col}_lag{k}": s.shift(k) for k in obs_lags}
        for win in windows:
            r = s.rolling(win, min_periods=max(2, win // 4))
            d[f"wx_obs_{col}_rmean{win}"] = r.mean()
            if col == "temperature_2m":
                d[f"wx_obs_{col}_rmin{win}"] = r.min()
                d[f"wx_obs_{col}_rmax{win}"] = r.max()
        d[f"wx_obs_{col}_diff1"] = s.diff(step_1h)
        d[f"wx_obs_{col}_diff24"] = s.diff(step_24h)
        parts.append(pd.DataFrame(d, index=index))

    t = w["temperature_2m"]
    parts.append(pd.DataFrame({
        # 冷却の効きやすさ（風速×気温差の代理）
        "wx_obs_cooling_proxy": w["wind_speed_10m"] * (30.0 - t),
        # 熱の蓄積（気温の指数移動平均を時定数違いで）
        **{f"wx_obs_temp_ewm{span}": t.ewm(span=span, min_periods=max(1, span // 4)).mean()
           for span in ewm_spans},
    }, index=index))

    # --- 予測対象時刻の気象（予報の代理） ---
    if use_forecast and horizon > 0:
        fc = {col: w[col].shift(-horizon) for col in WEATHER_COLS}
        if forecast_noise_std > 0:
            # 気象予報の誤差を模擬する。実運用では再解析の実測ではなく予報が入るため、
            # 完全既知を前提にした改善幅がどこまで残るかを測るためのノイズ。
            # 気温にだけ与える（主要な説明変数であり、℃単位で誤差水準を解釈できるため）。
            rng = np.random.default_rng(noise_seed)
            noise = pd.Series(rng.normal(0.0, forecast_noise_std, len(index)), index=index)
            fc["temperature_2m"] = fc["temperature_2m"] + noise
        d = {f"wx_fc_{col}": v for col, v in fc.items()}
        # 起点から対象時刻までに気温がどれだけ動くか。予報が持つ本質的な追加情報はここ
        d["wx_fc_temp_delta"] = fc["temperature_2m"] - t
        d["wx_fc_temp_delta_abs"] = (fc["temperature_2m"] - t).abs()
        parts.append(pd.DataFrame(d, index=index))

    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
