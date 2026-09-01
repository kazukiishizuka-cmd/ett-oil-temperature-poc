"""外部データ（外気温）の取得と、観測地点の推定。

EDAで「同時刻の負荷はOTの水準を最大 r=0.22 しか説明しない」ことが分かった。
一方でOTの日内パターン（早朝に最低・午後に最高）と季節変動は外気温の形そのもので、
油温 = 外気温 + 負荷による温度上昇 という物理を踏まえると、
説明変数として最も効くはずの外気温がデータセットに含まれていない。

ETDatasetは観測地点を「中国のある省の2地域」としか公開していないため、
気候帯を広くとった候補都市の気温とOTの相関を測り、最も整合する地点を推定して使う。

データ出典:
  Open-Meteo Historical Weather API (ERA5 reanalysis)
  https://open-meteo.com/en/docs/historical-weather-api
  ライセンス: CC BY 4.0 / 原データ Copernicus Climate Change Service (C3S) ERA5
"""
from __future__ import annotations

import json
import time
import warnings
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULT_DIR

EXTERNAL_DIR = DATA_DIR / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# NumPy 2.0 + macOS Accelerate の matmul が偽のFP例外フラグを立てるため抑制する
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

API = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2016-07-01", "2018-06-30"

# 中国の気候帯を広くカバーする候補地点
CANDIDATES = {
    "Beijing":   (39.90, 116.41),
    "Shanghai":  (31.23, 121.47),
    "Guangzhou": (23.13, 113.26),
    "Chengdu":   (30.57, 104.07),
    "Xian":      (34.34, 108.94),
    "Shenyang":  (41.80, 123.43),
    "Wuhan":     (30.59, 114.31),
    "Jinan":     (36.65, 117.12),
    "Zhengzhou": (34.75, 113.62),
    "Nanjing":   (32.06, 118.80),
    "Harbin":    (45.80, 126.53),
    "Lanzhou":   (36.06, 103.83),
    "Kunming":   (25.04, 102.72),
    "Urumqi":    (43.83, 87.62),
    "Hangzhou":  (30.27, 120.15),
    "Changsha":  (28.23, 112.94),
}

HOURLY_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
               "shortwave_radiation", "surface_pressure", "precipitation",
               "cloud_cover", "dew_point_2m"]


def fetch_city(name: str, lat: float, lon: float, force: bool = False) -> pd.DataFrame:
    """1地点ぶんの毎時気象データを取得する（取得済みならキャッシュを読む）。"""
    path = EXTERNAL_DIR / f"weather_{name}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")

    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Shanghai",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=90) as r:
        payload = json.load(r)
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["date"] = pd.to_datetime(df.pop("time"))
    df = df.set_index("date").sort_index()
    df.to_csv(path)
    print(f"  取得: {name:10s} {len(df):5d} 点 -> {path.name}")
    return df


def fetch_all(force: bool = False) -> dict:
    out = {}
    for name, (lat, lon) in CANDIDATES.items():
        try:
            out[name] = fetch_city(name, lat, lon, force=force)
            time.sleep(0.4)  # 公開APIへの配慮
        except Exception as e:
            print(f"  失敗: {name}: {e}")
    return out


def estimate_location(weather: dict) -> pd.DataFrame:
    """候補地点の気温とOTの整合度を測り、観測地点を推定する。

    絶対水準のトレンドに引きずられないよう、日次平均に集約したうえで
      1) 生の相関
      2) 年周期を除いた残差どうしの相関
      3) 日内パターン（時刻別の平均偏差）の相関
    の3つを見る。
    """
    from data import clean_dataset, load_dataset

    rows = []
    ot_daily, ot_hourly_shape = {}, {}
    for ds in ["ETTh1", "ETTh2"]:
        df, _ = clean_dataset(load_dataset(ds))
        ot = df["OT"]
        ot_daily[ds] = ot.resample("D").mean()
        dev = ot - ot.rolling(24 * 30, center=True, min_periods=100).mean()
        ot_hourly_shape[ds] = dev.groupby(dev.index.hour).mean()

    for name, w in weather.items():
        t = w["temperature_2m"]
        t_daily = t.resample("D").mean()
        t_dev = t - t.rolling(24 * 30, center=True, min_periods=100).mean()
        t_shape = t_dev.groupby(t_dev.index.hour).mean()
        for ds in ["ETTh1", "ETTh2"]:
            a, b = ot_daily[ds].align(t_daily, join="inner")
            r_daily = a.corr(b)
            # 年周期（日付の sin/cos）で説明できる分を落としてから相関を見る
            doy = a.index.dayofyear.to_numpy()
            X = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                                 np.ones_like(doy, dtype=float)])
            ra = a.to_numpy() - X @ np.linalg.lstsq(X, a.to_numpy(), rcond=None)[0]
            rb = b.to_numpy() - X @ np.linalg.lstsq(X, b.to_numpy(), rcond=None)[0]
            r_resid = float(np.corrcoef(ra, rb)[0, 1])
            r_shape = float(ot_hourly_shape[ds].corr(t_shape))
            rows.append({"city": name, "dataset": ds, "r_daily": r_daily,
                         "r_seasonal_residual": r_resid, "r_daily_shape": r_shape})
    out = pd.DataFrame(rows)
    out["score"] = out[["r_daily", "r_seasonal_residual", "r_daily_shape"]].mean(axis=1)
    return out.sort_values(["dataset", "score"], ascending=[True, False])


def main() -> None:
    print("外気温データを取得:")
    weather = fetch_all()
    print(f"\n{len(weather)} 地点を取得")
    est = estimate_location(weather)
    est.to_csv(RESULT_DIR / "weather_location_estimate.csv", index=False)
    pd.set_option("display.width", 200)
    for ds in ["ETTh1", "ETTh2"]:
        print(f"\n=== {ds} との整合度（上位6地点） ===")
        print(est[est.dataset == ds].head(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
