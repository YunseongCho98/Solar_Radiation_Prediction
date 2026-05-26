"""Step 3: 시간 피처, 주기 인코딩, 태양 파생 피처, lag 피처, country 인코딩."""
import numpy as np
import pandas as pd

import config


def add_features(df):
    df = df.copy()
    ts = df["timestamp"]
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["doy"] = ts.dt.dayofyear
    df["month_id"] = df["year"] * 100 + df["month"]

    # 주기 인코딩: 23시-1시가 가깝다는 것을 모델이 인식하도록
    tod = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * tod / 24)
    df["hour_cos"] = np.cos(2 * np.pi * tod / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 태양 파생 피처
    z = df["solar_zenith"].clip(0, 90)
    df["cos_zenith"] = np.cos(np.radians(z))
    df["air_mass"] = 1.0 / np.cos(np.radians(z.clip(0, 89)))

    # 이슬점 온도 (대기 수분량 proxy)
    df["dewpoint"] = df["temp"] - (100 - df["humidity"]) / 5.0

    # lag 피처: (station, month_id) 그룹 내에서만 (월 경계 NaN → 현재값 대체)
    df = df.sort_values(["station", "timestamp"])
    grp = df.groupby(["station", "month_id"], sort=False)
    for col in ["humidity", "temp"]:
        for lag in [1, 2, 3, 4, 8, 24]:
            df[f"{col}_lag{lag}"] = grp[col].shift(lag).fillna(df[col])
    for lag in [1, 4]:
        df[f"precip_lag{lag}"] = grp["precip"].shift(lag).fillna(df["precip"])

    # 기온 변화량 (날씨 변화 속도 신호)
    df["temp_diff1"] = df["temp"] - df["temp_lag1"]
    df["temp_diff4"] = df["temp"] - df["temp_lag4"]

    # rolling 통계: station 단위 전체 시계열 기준 (월 경계 허용)
    stn_grp = df.groupby("station", sort=False)
    for col in ["humidity", "temp"]:
        for win in [3, 6, 24]:
            df[f"{col}_roll{win}_mean"] = (
                stn_grp[col]
                .transform(lambda x: x.rolling(win, min_periods=1).mean())
            )
    df["precip_roll6_sum"] = (
        stn_grp["precip"]
        .transform(lambda x: x.rolling(6, min_periods=1).sum())
    )
    return df


def main():
    train = pd.read_parquet(config.DATA_DIR / "train_solar.parquet")
    test = pd.read_parquet(config.DATA_DIR / "test_solar.parquet")

    # country 인코딩 (train+test 통합 기준으로 일관성 보장)
    countries = sorted(pd.concat([train["country"], test["country"]]).unique())
    country_map = {c: i for i, c in enumerate(countries)}

    for split, df in [("train", train), ("test", test)]:
        df = add_features(df)
        df["country_enc"] = df["country"].map(country_map)
        df = df.sort_index()
        df.to_parquet(config.DATA_DIR / f"{split}_feat.parquet")
        print(f"[03] {split}_feat {df.shape}")


if __name__ == "__main__":
    main()
