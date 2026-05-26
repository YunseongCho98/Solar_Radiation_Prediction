"""Step 8: 후처리. station별 CV bias 보정 + 물리 범위 클리핑.

station_bias가 있으면 station별로 개별 보정,
없는 station은 전역 day_bias로 fallback.
"""
import pickle

import pandas as pd

import config


def main():
    pred = pd.read_parquet(config.DATA_DIR / "test_pred.parquet")
    with open(config.MODEL_DIR / "cv_results.pkl", "rb") as f:
        cv = pickle.load(f)

    global_bias = cv["day_bias"]
    station_bias = cv.get("station_bias", {})
    day_mask = ~pred["is_night"]

    pred["pred"] = pred["pred_raw"]

    if station_bias:
        # station별 bias 차감 (없으면 전역 bias로 fallback)
        biases = pred["station"].map(station_bias).fillna(global_bias)
        pred.loc[day_mask, "pred"] = (
            pred.loc[day_mask, "pred_raw"] - biases[day_mask]
        )
        covered = pred.loc[day_mask, "station"].isin(station_bias).sum()
        total = day_mask.sum()
        print(f"[08] station별 bias 보정: {len(station_bias)}개 station")
        print(f"     주간 행 중 station 매칭: {covered:,}/{total:,}")
        print(f"     전역 fallback bias: {global_bias:+.3f} W/m2")
    else:
        # CV 미실행 시 전역 bias (0.0)
        pred.loc[day_mask, "pred"] = pred.loc[day_mask, "pred_raw"] - global_bias
        print(f"[08] 전역 bias 보정: {global_bias:+.3f} W/m2")

    pred["pred"] = pred["pred"].clip(0, config.RAD_CLIP_MAX)
    pred.loc[pred["is_night"], "pred"] = 0.0

    pred.to_parquet(config.DATA_DIR / "test_pred_final.parquet")

    print(f"[08] 보정 전 mean={pred['pred_raw'].mean():.2f} -> "
          f"보정 후 mean={pred['pred'].mean():.2f}")
    print(f"[08] 최종 예측: min={pred['pred'].min():.2f} "
          f"max={pred['pred'].max():.2f}")


if __name__ == "__main__":
    main()
