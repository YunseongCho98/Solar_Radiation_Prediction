"""Step 9: 제출 파일 생성. TargetMBE = TargetRMSE = 예측 복사량."""
import pandas as pd

import config


def main():
    pred = pd.read_parquet(config.DATA_DIR / "test_pred_final.parquet")
    sample = pd.read_csv(config.SAMPLE_SUB_CSV)

    sub = sample[["ID"]].merge(pred[["ID", "pred"]], on="ID", how="left")

    n_missing = int(sub["pred"].isna().sum())
    if n_missing:
        print(f"[09] 경고: 예측 누락 {n_missing} 행 -> 0으로 채움")
        sub["pred"] = sub["pred"].fillna(0.0)

    sub["TargetMBE"] = sub["pred"]
    sub["TargetRMSE"] = sub["pred"]
    sub = sub[["ID", "TargetMBE", "TargetRMSE"]]
    sub.to_csv(config.SUBMISSION_CSV, index=False)

    print(f"[09] submission.csv 저장: {sub.shape}")
    print(f"[09] 행 수 일치: {len(sub) == len(sample)}")
    print(sub.head())


if __name__ == "__main__":
    main()
