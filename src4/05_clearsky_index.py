"""Step 5: Clear Sky Index(kt) 타깃 생성.

kt = radiation / (clearsky_ghi + eps), [0, KT_CLIP]로 클리핑.
원래 복사량(0~1427) 대신 0~1.5 범위의 kt를 예측하면 학습이 안정적이고
태양 기하 정보가 분모에 흡수되어 모델은 '구름 정도'만 학습하면 된다.
"""
import pandas as pd

import config


def main():
    for split in ["train", "test"]:
        df = pd.read_parquet(config.DATA_DIR / f"{split}_masked.parquet")

        if split == "train":
            kt = df["radiation"] / (df["clearsky_ghi"] + config.CLEARSKY_EPS)
            df["kt"] = kt.clip(0, config.KT_CLIP)
            day = df[~df["is_night"]]
            print(f"[05] train kt (주간 {len(day)} 행): "
                  f"mean={day['kt'].mean():.3f} std={day['kt'].std():.3f}")

        df.to_parquet(config.DATA_DIR / f"{split}_final.parquet")
        print(f"[05] {split}_final {df.shape}")


if __name__ == "__main__":
    main()
