"""Step 7: test 예측. 주간은 kt 앙상블 예측 후 복사량 역변환, 야간은 0."""
import pickle

import numpy as np
import pandas as pd

import config
from model_utils import predict_kt, kt_to_radiation


def main():
    test = pd.read_parquet(config.DATA_DIR / "test_final.parquet")
    with open(config.MODEL_DIR / "models.pkl", "rb") as f:
        models = pickle.load(f)

    pred_rad = np.zeros(len(test))
    day_mask = (~test["is_night"]).values
    day = test[~test["is_night"]]

    kt_pred = predict_kt(day, models["global"], models["stations"])
    pred_rad[day_mask] = kt_to_radiation(kt_pred, day["clearsky_ghi"].values)

    out = test[["ID", "station", "is_night"]].copy()
    out["pred_raw"] = pred_rad
    out.to_parquet(config.DATA_DIR / "test_pred.parquet")

    print(f"[07] 예측 완료: 주간 {int(day_mask.sum())} 행, "
          f"야간 {int((~day_mask).sum())} 행")
    print(f"[07] pred_raw: mean={pred_rad.mean():.2f} max={pred_rad.max():.2f}")


if __name__ == "__main__":
    main()
