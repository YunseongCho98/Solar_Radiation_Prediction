"""전체 파이프라인 순차 실행."""
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "01_preprocess.py",
    "02_solar_features.py",
    "03_feature_engineering.py",
    "04_night_mask.py",
    "05_clearsky_index.py",
    "06_train.py",
    "07_predict.py",
    "08_postprocess.py",
    "09_submit.py",
]


def main():
    src = Path(__file__).resolve().parent
    t0 = time.time()
    for step in STEPS:
        print(f"\n{'=' * 60}\n실행: {step}\n{'=' * 60}", flush=True)
        r = subprocess.run([sys.executable, str(src / step)])
        if r.returncode != 0:
            print(f"\n[run_all] {step} 실패 (exit {r.returncode}) — 중단")
            sys.exit(1)
    print(f"\n{'=' * 60}\n[run_all] 전체 완료 ({time.time() - t0:.1f}s)\n{'=' * 60}")


if __name__ == "__main__":
    main()
