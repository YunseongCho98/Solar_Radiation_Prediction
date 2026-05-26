"""Step 1: 원본 CSV 로드, 컬럼명 정리, timestamp 파싱, parquet 저장."""
import pandas as pd

import config


def preprocess(path):
    df = pd.read_csv(path)
    df = df.rename(columns=config.RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    return df


def main():
    train = preprocess(config.TRAIN_CSV)
    test = preprocess(config.TEST_CSV)
    train.to_parquet(config.DATA_DIR / "train_clean.parquet")
    test.to_parquet(config.DATA_DIR / "test_clean.parquet")
    print(f"[01] train_clean {train.shape}, test_clean {test.shape}")
    print(f"[01] columns: {list(train.columns)}")


if __name__ == "__main__":
    main()
