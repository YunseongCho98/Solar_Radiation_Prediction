"""Step 4: 야간 마스킹.

태양 고도각 <= 0 -> 복사량 = 0 (물리적 확실). 전체의 약 50%를 ML 없이 처리.
train의 야간 radiation을 0으로 정리(라벨 클리닝), test는 is_night 플래그만 추가.
"""
import pandas as pd

import config


def main():
    for split in ["train", "test"]:
        df = pd.read_parquet(config.DATA_DIR / f"{split}_feat.parquet")
        df["is_night"] = df["solar_elevation"] <= 0

        if split == "train":
            df.loc[df["is_night"], "radiation"] = 0.0

        n_night = int(df["is_night"].sum())
        df.to_parquet(config.DATA_DIR / f"{split}_masked.parquet")
        print(f"[04] {split}_masked {df.shape} | "
              f"night {n_night} ({n_night / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
