"""Step 2: pvlib으로 태양 위치 + clearsky 복사량 계산.

timestamp는 UTC로 검증됨 (radiation 피크 vs clearsky 피크 offset ~0).
pvlib 태양고도각 컬럼은 'solar_*'로 명명하여 관측소 'elevation_m'과 구분.
"""
import pandas as pd
import pvlib

import config


def add_solar(df):
    parts = []
    for station, g in df.groupby("station", sort=False):
        g = g.copy()
        lat = g["latitude"].iloc[0]
        lon = g["longitude"].iloc[0]
        alt = g["elevation_m"].iloc[0]
        times = pd.DatetimeIndex(g["timestamp"]).tz_localize("UTC")

        loc = pvlib.location.Location(lat, lon, altitude=alt)
        solpos = loc.get_solarposition(times)
        cs = loc.get_clearsky(times, model=config.CLEARSKY_MODEL)

        g["solar_zenith"] = solpos["apparent_zenith"].values
        g["solar_elevation"] = solpos["apparent_elevation"].values
        g["solar_azimuth"] = solpos["azimuth"].values
        g["clearsky_ghi"] = cs["ghi"].values
        g["clearsky_dni"] = cs["dni"].values
        g["clearsky_dhi"] = cs["dhi"].values
        parts.append(g)
    return pd.concat(parts).sort_index()


def main():
    for split in ["train", "test"]:
        df = pd.read_parquet(config.DATA_DIR / f"{split}_clean.parquet")
        df = add_solar(df)
        df.to_parquet(config.DATA_DIR / f"{split}_solar.parquet")
        print(f"[02] {split}_solar {df.shape} | "
              f"clearsky_ghi max={df['clearsky_ghi'].max():.0f}")


if __name__ == "__main__":
    main()
