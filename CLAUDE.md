# Solar Radiation Prediction — 프로젝트 컨텍스트

## 대회 개요
- **플랫폼**: Zindi — TAHMO Solar Radiation Prediction
- **목표**: 아프리카 기상 관측소 40곳의 짝수월(2,4,6,8,10,12) 태양 복사량 예측
- **평가 지표**: `0.5 × |MBE| + 0.5 × RMSE` (낮을수록 좋음)
- **제출 형식**: `ID, TargetMBE, TargetRMSE` — 두 컬럼에 동일한 예측값을 씀
- **제출 제한**: 하루 10회, 총 100회

## 핵심 데이터 구조
- **Train**: 홀수월 (1,3,5,7,9,11), **Test**: 짝수월 (2,4,6,8,10,12)
- 같은 연도, 같은 station → 짝수월은 홀수월 사이의 **보간** 문제 (예측/외삽이 아님)
- Station 40개, 국가: 아프리카 여러 나라
- 타임존: 실증 검증으로 **UTC** 확인 (국가별 오프셋 적용 시 태양 피크가 어긋남)
- 야간 비율: 약 49.6%

## 모델 구조

### 핵심 아이디어
1. **Clear Sky Index (kt)** 를 타겟으로 사용: `kt = radiation / (clearsky_ghi + 1)`
   - 물리적으로 안정된 범위 [0, 1.5], 태양 기하학을 clearsky 모델이 흡수
2. **야간 마스킹**: `solar_elevation ≤ 0` → 예측값 0 (학습/예측 모두)
3. **MIN_TRAIN_GHI = 50**: clearsky_ghi < 50인 저고도 구간은 학습에서 제외 (kt 폭발 방지)
   - 예측은 전체 주간에 적용
4. **앙상블**: station 모델(0.7) + global 모델(0.3) 블렌딩
5. **CV bias 보정은 효과 없음**: 홀수월 CV bias ≠ 짝수월 test bias → 적용 시 성능 저하

## 현재 최고 성능 파이프라인: `src4`

### 모델
- **XGBoost CUDA** + **LightGBM GPU(OpenCL)** + **CatBoost GPU** 3종 앙상블
- 가중치: XGB 0.4 / LGB 0.3 / CAT 0.3
- n_estimators: 500 (전 모델 공통)

### 피처 (48개)
- 태양 물리: cos_zenith, solar_zenith, solar_azimuth, air_mass, clearsky_ghi/dni/dhi
- 시간: hour/doy/month sin/cos 인코딩, hour, doy, month
- 기상 현재값: temp, humidity, precip, dewpoint
- Lag 피처: temp/humidity lag 1,2,3,4,8,24 / precip lag 1,4
- 기온 변화량: temp_diff1, temp_diff4
- Rolling 통계: temp/humidity roll 3h/6h/24h mean, precip roll 6h sum
- 관측소: latitude, longitude, elevation_m, installation_height, country_enc

### 실행 방법
```bash
cd /home/ai/train_model/Training/Solar_Radiation_Prediction
source env/bin/activate

# 전체 파이프라인 (처음부터)
python3 -u src4/run_all.py

# 06단계부터만 (데이터 전처리 완료 시)
python3 -u src4/06_train.py && \
python3 -u src4/07_predict.py && \
python3 -u src4/08_postprocess.py && \
python3 -u src4/09_submit.py
```

### 결과물
- 모델: `models4/models.pkl`
- 제출 파일: `submission4.csv` (타임스탬프 버전으로도 저장됨)

## 폴더 구조

| 폴더 | 설명 | 피처 수 | 비고 |
|------|------|---------|------|
| `src` | LightGBM OpenCL, 초기 버전 | 28개 | 기준점 |
| `src2` | XGBoost CUDA | 49개 | src 대비 빠름 |
| `src3` | src2 + station별 bias 보정 | 49개 | bias 보정 비효과적 |
| `src4` | XGB+LGB+CAT 앙상블 | 48개 | **현재 최고 성능** |
| `src5` | src3 + 시계열 피처 추가 | 62개 | hours_since_rain 등 |

## 환경 설정
```bash
cd /home/ai/train_model/Training/Solar_Radiation_Prediction
source env/bin/activate
# requirements.txt: numpy, pandas, scipy, pyarrow, lightgbm, pvlib, tqdm, xgboost, catboost, scikit-learn
```

GPU: NVIDIA RTX PRO 6000 Black (97GB VRAM), CUDA 13.0
- XGBoost: `device='cuda'`
- LightGBM: `device='gpu'` (OpenCL)
- CatBoost: `task_type='GPU'`, `bootstrap_type='Bernoulli'` (GPU에서 subsample 사용 시 필수)

## 다음 개선 방향 (우선순위 순)

1. **앙상블 가중치 최적화**: CV OOF로 XGB/LGB/CAT 개별 성능 측정 후 가중치 재설정
2. **Optuna 하이퍼파라미터 튜닝**: learning_rate, depth, subsample 자동 탐색
3. **피처 중요도 기반 가지치기**: 하위 피처 제거로 노이즈 감소
4. **스태킹 앙상블**: OOF 예측을 메타 피처로 사용하는 2단계 모델

## 주의사항
- **CV bias 보정 금지**: `06_cv.py` 실행 후 bias 적용 시 리더보드 점수 일관되게 하락
- CatBoost GPU 파라미터 제약: `colsample_bylevel` 미지원, `subsample` 사용 시 `bootstrap_type='Bernoulli'` 필수
- 데이터 전처리(01~05)는 src 전체 공용 (`data/` 폴더 공유)
- 모델과 제출 파일은 src별로 분리: `models4/`, `submission4.csv`
