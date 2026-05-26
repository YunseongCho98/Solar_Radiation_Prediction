"""Step 6: station별/global LightGBM 최종 학습.

CV는 기본적으로 비활성화되어 있다 (day_bias=0, 보정 없음).
성능 개선이 필요할 때 06_cv.py를 별도 실행하면 cv_results.pkl의 day_bias가
갱신되고, 08_postprocess.py가 이를 자동으로 반영한다.
최종 모델 재학습 불필요.
"""
import pickle

import pandas as pd

import config
from model_utils import train_station_models


def main():
    train = pd.read_parquet(config.DATA_DIR / "train_final.parquet")

    day = train[~train["is_night"]]
    train_fit = day[day["clearsky_ghi"] > config.MIN_TRAIN_GHI]
    print(f"[06] 최종 모델 학습 시작")
    print(f"     학습: {len(train_fit):,}행 / 전체 주간: {len(day):,}행")
    print(f"     global 1개 + station 최대 {train_fit['station'].nunique()}개")

    global_model, station_models = train_station_models(train_fit, desc="station 모델")
    print(f"[06] 학습 완료: global 1개 + station {len(station_models)}개")

    with open(config.MODEL_DIR / "models.pkl", "wb") as f:
        pickle.dump({"global": global_model, "stations": station_models}, f)

    # CV를 실행하지 않았으므로 day_bias=0 (보정 없음)
    # bias 보정이 필요하면 06_cv.py를 실행해서 cv_results.pkl을 갱신할 것
    cv = {"rmse": None, "mbe": None, "score": None, "day_bias": 0.0}
    with open(config.MODEL_DIR / "cv_results.pkl", "wb") as f:
        pickle.dump(cv, f)

    print("[06] models.pkl 저장 완료 (bias 보정 없음, day_bias=0)")


if __name__ == "__main__":
    main()
