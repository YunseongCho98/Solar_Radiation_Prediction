"""CV 전용 스크립트: station별 day_bias 추정 후 cv_results.pkl 갱신.

최종 모델(models.pkl) 재학습 없이 station별 bias 보정값만 도출한다.
실행: python3 src3/06_cv.py
완료 후 07_predict.py → 08_postprocess.py → 09_submit.py 순으로 재실행하면
station별 bias가 반영된 제출 파일이 생성된다.
"""
import pickle

import numpy as np
import pandas as pd

import config
from model_utils import train_station_models, predict_kt, kt_to_radiation


def run_cv(train):
    day = train[~train["is_night"]]
    fit = day[day["clearsky_ghi"] > config.MIN_TRAIN_GHI]
    months = sorted(train["month"].unique())
    n = len(months)
    oof_kt = pd.Series(index=train.index, dtype=float)

    for i, m in enumerate(months, 1):
        tr = fit[fit["month"] != m]
        va = day[day["month"] == m]
        print(f"\n[CV {i}/{n}] month={m:2d} | 학습 {len(tr):,}행, 평가 {len(va):,}행")
        gm, sm = train_station_models(tr, desc=f"  fold {i}/{n} (m={m:02d})")
        oof_kt.loc[va.index] = predict_kt(va, gm, sm)

    day_idx = day.index
    oof_rad = pd.Series(0.0, index=train.index)
    oof_rad.loc[day_idx] = kt_to_radiation(
        oof_kt.loc[day_idx].values,
        train.loc[day_idx, "clearsky_ghi"].values,
    )
    oof_rad = oof_rad.clip(0, config.RAD_CLIP_MAX)

    err = oof_rad - train["radiation"]
    rmse = float(np.sqrt((err ** 2).mean()))
    mbe = float(err.mean())
    day_bias = float(err.loc[day_idx].mean())
    score = 0.5 * abs(mbe) + 0.5 * rmse

    # station별 주간 bias 계산
    day_err = err.loc[day_idx]
    station_bias = (
        train.loc[day_idx, "station"]
        .to_frame()
        .assign(err=day_err.values)
        .groupby("station")["err"]
        .mean()
        .to_dict()
    )

    return {
        "rmse": rmse,
        "mbe": mbe,
        "score": score,
        "day_bias": day_bias,           # 전역 bias (fallback용)
        "station_bias": station_bias,   # station별 bias
    }


def main():
    train = pd.read_parquet(config.DATA_DIR / "train_final.parquet")

    print("[CV] 홀수월 leave-one-month-out CV 시작 (6 fold)...")
    cv = run_cv(train)

    print(f"\n[CV] 전체 결과:")
    print(f"     RMSE         = {cv['rmse']:.3f}")
    print(f"     |MBE|        = {abs(cv['mbe']):.3f}")
    print(f"     Score        = {cv['score']:.3f}  (= 0.5*|MBE| + 0.5*RMSE)")
    print(f"     전역 day_bias = {cv['day_bias']:+.3f} W/m2")
    print(f"     bias 보정 시 예상 Score ≈ {0.5 * cv['rmse']:.3f}")

    print(f"\n[CV] station별 bias (상위 10개):")
    sorted_bias = sorted(cv["station_bias"].items(), key=lambda x: abs(x[1]), reverse=True)
    for st, b in sorted_bias[:10]:
        print(f"     {st}: {b:+.3f} W/m2")

    with open(config.MODEL_DIR / "cv_results.pkl", "wb") as f:
        pickle.dump(cv, f)
    print(f"\n[CV] cv_results.pkl 갱신 완료 (station {len(cv['station_bias'])}개 bias 저장)")
    print(f"     이제 07 → 08 → 09 를 재실행하면 station별 bias 보정이 반영됩니다.")


if __name__ == "__main__":
    main()
