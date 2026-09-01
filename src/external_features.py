"""外部データ（気象・祝日）の読み込みと特徴量化。

データ出典:
  - 気象: Open-Meteo Historical Weather API (ERA5 reanalysis)
          https://open-meteo.com/en/docs/historical-weather-api
          CC BY 4.0 / 原データ Copernicus Climate Change Service (C3S)
  - 祝日: 中国国務院公布の法定休日（2016-2018）を手動で定義
          https://www.gov.cn/zhengce/content/  （年度ごとの放假安排通知）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR

EXTERNAL_DIR = DATA_DIR / "external"

#: 観測地点の推定結果。候補16都市とOTの整合度から選んだ（fetch_weather.py 参照）
DEFAULT_CITY = "Wuhan"

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

    中国の製造業は春節に2週間近く止まるため、負荷の構造が年に一度大きく変わる。
    フラグだけでなく前後の日数も持たせて、立ち上がり・立ち下がりを表せるようにする。
    """
    out = pd.DataFrame(index=index)
    dates = index.normalize()
    is_any = pd.Series(False, index=index)
    for name, spans in CHINA_HOLIDAYS.items():
        flag = pd.Series(False, index=index)
        for lo, hi in spans:
            flag |= (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi) + pd.Timedelta(days=1))
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
                     lags=(0, 1, 3, 6, 12, 24), windows=(6, 24, 168),
                     future_known: bool = True) -> pd.DataFrame:
    """気象データを特徴量にする。

    future_known=True は「予測時点で対象時刻の気象が分かっている」前提。
    実運用では気象予報がこれに相当する。False の場合は t 時点までの実測しか使わない。
    """
    w = load_weather(city).reindex(index).interpolate(method="time", limit_direction="both")
    parts = []
    for col in WEATHER_COLS:
        s = w[col]
        d = {}
        for k in lags:
            if k == 0 and not future_known:
                continue
            d[f"wx_{col}_lag{k}"] = s.shift(k)
        for win in windows:
            r = s.rolling(win, min_periods=max(2, win // 4))
            d[f"wx_{col}_rmean{win}"] = r.mean()
            if col == "temperature_2m":
                d[f"wx_{col}_rmin{win}"] = r.min()
                d[f"wx_{col}_rmax{win}"] = r.max()
        d[f"wx_{col}_diff1"] = s.diff(1)
        d[f"wx_{col}_diff24"] = s.diff(24)
        parts.append(pd.DataFrame(d, index=index))

    # 気温と油温の物理的な関係を直接表す量
    t = w["temperature_2m"]
    extra = pd.DataFrame({
        # 冷却の効きやすさ（風速×気温差の代理）
        "wx_cooling_proxy": w["wind_speed_10m"] * (30.0 - t),
        # 熱の蓄積（気温の指数移動平均を時定数違いで）
        **{f"wx_temp_ewm{span}": t.ewm(span=span, min_periods=span // 4).mean()
           for span in (6, 24, 72, 168)},
    }, index=index)
    parts.append(extra)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
