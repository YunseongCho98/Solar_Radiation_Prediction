# TAHMO Solar Radiation Prediction

## 대회 개요

**주최**: Trans-African Hydro-Meteorological Observatory (TAHMO)  
**플랫폼**: Zindi  
**목표**: 아프리카 기상 관측소 40곳에서 센서 드리프트로 누락된 짝수월 태양 복사량을 ML로 재구성

TAHMO는 아프리카 23개국에 600개 이상의 기상 관측소를 운영한다. 센서는 약 2년의 환경 노출 후 비선형 드리프트가 발생하여 데이터 품질이 저하되며, 이 대회는 ML로 실제 복사량을 복원하는 것을 목표로 한다.

### 데이터 구조
- **Train**: 홀수월 (1, 3, 5, 7, 9, 11) — 15분 간격 기상 데이터
- **Test**: 짝수월 (2, 4, 6, 8, 10, 12) — 같은 연도, 같은 관측소
- **핵심 인사이트**: Test는 예측/외삽이 아닌 **홀수월 사이의 보간** 문제

### 평가 지표
```
Score = 0.5 × |MBE| + 0.5 × RMSE
```
- MBE(Mean Bias Error)와 RMSE를 동시에 최소화해야 함
- 제출 형식: `ID, TargetMBE, TargetRMSE` — 두 컬럼에 동일한 예측값 기입

---

## 모델 구조 (src4 — 최고 성능)

### 핵심 설계 원칙

#### 1. Clear Sky Index (kt) 타겟
원시 복사량 대신 Clear Sky Index를 예측 타겟으로 사용한다.

```
kt = radiation / (clearsky_ghi + 1)   # 범위: [0, 1.5]
radiation = kt × (clearsky_ghi + 1)   # 역변환
```

태양 기하학(위치, 계절, 시간)이 복사량의 대부분을 설명하며, clearsky 모델(Ineichen)이 이를 흡수한다. kt는 구름/대기 조건만 반영하여 모델이 학습해야 할 패턴이 단순해진다.

#### 2. 야간 마스킹
`solar_elevation ≤ 0` 조건으로 야간을 판별하여 예측값을 0으로 강제한다 (전체 데이터의 약 49.6%). ML 모델은 주간 데이터만 학습한다.

#### 3. 저고도 구간 학습 제외 (MIN_TRAIN_GHI = 50)
태양 고도가 낮은 새벽/저녁 구간(clearsky_ghi < 50)은 kt가 불안정하게 폭발하므로 학습에서 제외한다. 예측은 전체 주간에 적용한다.

#### 4. 2단계 앙상블
```
예측값 = 0.7 × station_model_pred + 0.3 × global_model_pred
```
- **global model**: 전체 40개 관측소 데이터로 학습한 일반 모델
- **station model**: 각 관측소 데이터만으로 학습한 특화 모델 (최소 100행 이상인 관측소만)

### 3종 모델 앙상블 (XGB + LGB + CAT)

각 global/station 모델은 3개 알고리즘의 가중 평균으로 구성된다.

```
앙상블 예측 = 0.4 × XGBoost + 0.3 × LightGBM + 0.3 × CatBoost
```

| 모델 | GPU 가속 | n_estimators | learning_rate |
|------|---------|-------------|---------------|
| XGBoost | CUDA | 500 | 0.05 |
| LightGBM | OpenCL | 500 | 0.05 |
| CatBoost | CUDA | 500 | 0.05 |

### 피처 (48개)

| 그룹 | 피처 |
|------|------|
| 태양 물리 (7) | cos_zenith, solar_zenith, solar_azimuth, air_mass, clearsky_ghi, clearsky_dni, clearsky_dhi |
| 시간 인코딩 (9) | hour/doy/month sin·cos, hour, doy, month |
| 기상 현재값 (4) | temp, humidity, precip, dewpoint |
| Lag 피처 (14) | temp lag 1,2,3,4,8,24 / humidity lag 1,2,3,4,8,24 / precip lag 1,4 |
| 변화량 (2) | temp_diff1, temp_diff4 |
| Rolling 통계 (7) | temp/humidity roll 3h·6h·24h mean, precip roll 6h sum |
| 관측소 (5) | latitude, longitude, elevation_m, installation_height, country_enc |

### 파이프라인 흐름

```
Train.csv / Test.csv
        │
        ▼
01_preprocess.py     컬럼 정리, timestamp 파싱
        │
        ▼
02_solar_features.py  pvlib로 태양 위치·clearsky GHI 계산 (UTC 기준)
        │
        ▼
03_feature_engineering.py  시간 인코딩, lag, rolling 통계, 이슬점
        │
        ▼
04_night_mask.py     solar_elevation ≤ 0 → is_night 플래그
        │
        ▼
05_clearsky_index.py  kt = radiation / (clearsky_ghi + 1) 계산
        │
        ▼
06_train.py          XGB+LGB+CAT global 모델 + 40개 station 모델 학습
        │
        ▼
07_predict.py        Test 주간 구간 kt 예측 → radiation 역변환
        │
        ▼
08_postprocess.py    물리 범위 클리핑 [0, 1500], 야간 0 강제
        │
        ▼
09_submit.py         SampleSubmission 형식으로 submission4.csv 생성
```

> **06_cv.py**: CV로 day_bias를 추정하는 선택적 스크립트.
> 단, 홀수월 CV bias와 짝수월 test bias 간 분포 차이로 인해
> 적용 시 리더보드 점수가 하락하는 경향이 있어 기본 파이프라인에서 제외되어 있다.

---

## 시작하기

### 1. 저장소 클론 (Git LFS 필요)

이 저장소는 대용량 데이터 파일(`*.csv`, `*.parquet`, `*.pkl`)을 **Git LFS**로 관리한다.
클론 전 반드시 Git LFS를 설치해야 한다.

```bash
# Git LFS 설치 (최초 1회)
# Ubuntu / Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# 설치 후 초기화
git lfs install

# 저장소 클론 (LFS 파일 포함)
git clone <repository-url>
cd Solar_Radiation_Prediction

# 이미 클론한 경우 LFS 파일만 별도 다운로드
git lfs pull
```

> LFS 없이 클론하면 `Train.csv`, `Test.csv`, `SampleSubmission.csv`,
> `data/*.parquet`, `models4/*.pkl` 등이 포인터 파일로만 받아진다.
> 이 경우 `git lfs pull`을 실행하면 실제 파일이 다운로드된다.

### 2. 환경 설정

```bash
python3 -m venv env
source env/bin/activate      # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. 전체 파이프라인 실행

```bash
python3 -u src4/run_all.py
```

### 4. 06단계부터만 실행 (전처리 완료 시)

```bash
python3 -u src4/06_train.py && \
python3 -u src4/07_predict.py && \
python3 -u src4/08_postprocess.py && \
python3 -u src4/09_submit.py
```

---

## 폴더 구조

```
Solar_Radiation_Prediction/
├── src4/                        # 최고 성능 파이프라인 (XGB+LGB+CAT 앙상블)
│   ├── config.py                # 경로, 피처 목록, 모델 파라미터
│   ├── model_utils.py           # 3종 앙상블 학습/예측 함수
│   ├── 01_preprocess.py
│   ├── 02_solar_features.py
│   ├── 03_feature_engineering.py
│   ├── 04_night_mask.py
│   ├── 05_clearsky_index.py
│   ├── 06_train.py              # 최종 모델 학습
│   ├── 06_cv.py                 # CV bias 추정 (선택 실행)
│   ├── 07_predict.py
│   ├── 08_postprocess.py
│   ├── 09_submit.py
│   └── run_all.py               # 01~09 순차 실행
├── models4/                     # 학습된 모델 (Git LFS)
│   ├── models.pkl               # global + station 앙상블 모델
│   └── cv_results.pkl
├── data/                        # 전처리 중간 파일 (Git LFS)
│   ├── train_final.parquet
│   ├── test_final.parquet
│   └── ...
├── Train.csv                    # 원본 학습 데이터 (Git LFS)
├── Test.csv                     # 원본 테스트 데이터 (Git LFS)
├── SampleSubmission.csv         # 제출 형식 예시 (Git LFS)
├── requirements.txt
├── CLAUDE.md                    # AI 세션 컨텍스트
└── COMPETITION_INFO.md
```

---

## 의존성

```
numpy, pandas, scipy, pyarrow
lightgbm==4.6.0   # GPU: OpenCL
xgboost==3.2.0    # GPU: CUDA
catboost==1.2.10  # GPU: CUDA
pvlib==0.15.1     # 태양 위치 / clearsky 계산
scikit-learn      # lightgbm sklearn API
tqdm              # 학습 진행 상태바
```

**GPU 요구사항**: NVIDIA GPU (CUDA 및 OpenCL 지원)
