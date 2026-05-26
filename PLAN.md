# 구현 계획: Solar Radiation Prediction

## 목표
- 평가 지표: `0.5 × |MBE| + 0.5 × RMSE` 최소화
- 40개 관측소의 짝수월(2,4,6,8,10,12) 15분 간격 태양 복사량 예측

---

## 진행 현황 (2026-05-22 기준)

| 단계 | 상태 | 비고 |
|------|------|------|
| 타임존 검증 | ✅ 완료 | timestamp는 **UTC**로 확인 (radiation 피크 vs clearsky 피크 offset ≈ 0) |
| 01 전처리 | ✅ 완료 | `data/*_clean.parquet` 저장 |
| 02 태양 피처 | ✅ 완료 | `data/*_solar.parquet` 저장 |
| 03 피처 엔지니어링 | ✅ 완료 | `data/*_feat.parquet` 저장 |
| 04 야간 마스킹 | ✅ 완료 | 야간 49.6%, `data/*_masked.parquet` 저장 |
| 05 Clear Sky Index | ✅ 완료 | 주간 kt 평균 0.734, `data/*_final.parquet` 저장 |
| 06 학습 | ⏳ 미완료 | GPU 전환 후 사용자가 직접 실행 예정 |
| 07~09 | ⏳ 대기 | 06 완료 후 진행 |

### 진행 중 확정된 주요 변경점
1. **타임존**: 경험적으로 UTC 확정 — naive timestamp를 그대로 UTC로 localize.
2. **저고도 구간 처리**: 태양 고도 0~5°에서 clearsky_ghi가 0에 가까워 kt가 폭발
   (주간의 6.5%가 clip 상한). 학습은 `clearsky_ghi > 50`(MIN_TRAIN_GHI) 안정 구간만,
   예측은 전체 주간에 적용 — 비대칭 처리.
3. **모델**: LightGBM `device='gpu'`(OpenCL) 사용. `device='cuda'`는 PyPI 휠 미포함.
4. **CV**: 최종 모델은 홀수월 전체로 학습하되, 별도로 월 기반 leave-one-out CV를
   진단용으로 수행 (점수·bias 추정). CV 모델은 최종 모델에 영향 없음.
5. **공통 모듈**: `config.py`(설정), `model_utils.py`(학습/예측 함수), `run_all.py`
   (오케스트레이터), `requirements.txt` 추가.

---

## 전체 파이프라인 개요

```
Raw Data (Train/Test)
  → 1. 데이터 전처리
  → 2. 물리 기반 피처 생성 (pvlib)
  → 3. 피처 엔지니어링
  → 4. 야간 마스킹
  → 5. Clear Sky Index 변환 (주간 타깃)
  → 6. Station별 LightGBM 학습
  → 7. 예측 및 역변환
  → 8. 후처리 (bias correction, clip)
  → 9. 제출 파일 생성
```

---

## Step 1. 데이터 전처리

**파일**: `src/01_preprocess.py`

- Train.csv, Test.csv 로드
- `timestamp` → `datetime` 변환 (UTC 기준)
- 컬럼명 단순화 (공백/특수문자 제거)
  ```
  'precipitation (mm)'        → 'precip'
  'radiation (W/m2)'          → 'radiation'
  'relativehumidity (-)'      → 'humidity'
  'temperature (degrees Celsius)' → 'temp'
  ```
- station별 위도/경도/고도 메타데이터 딕셔너리 생성
- 처리된 데이터 저장: `data/train_clean.parquet`, `data/test_clean.parquet`

---

## Step 2. 물리 기반 피처 생성 (pvlib)

**파일**: `src/02_solar_features.py`

### 2-1. 태양 위치 계산
각 관측소의 위도/경도 + timestamp로 매 15분마다 계산:

```python
import pvlib

location = pvlib.location.Location(
    latitude=lat,
    longitude=lon,
    altitude=elevation
)
solar_pos = location.get_solarposition(timestamps)
```

생성 피처 (관측소 해발고도 `elevation_m`과 충돌 방지를 위해 `solar_` 접두사 사용):
| 피처명 | 설명 | 범위 |
|--------|------|------|
| `solar_zenith` | 태양 천정각 (apparent) | 0°(정오) ~ 180°(자정) |
| `solar_elevation` | 태양 고도각 (apparent) | -90° ~ 90° |
| `solar_azimuth` | 태양 방위각 | 0° ~ 360° |

### 2-2. 맑은 하늘 복사량 계산 (Clear Sky GHI)
구름이 없을 때의 이론적 복사량:

```python
clearsky = location.get_clearsky(timestamps, model='ineichen')
# clearsky['ghi']: Global Horizontal Irradiance
```

생성 피처:
| 피처명 | 설명 |
|--------|------|
| `clearsky_ghi` | 이론적 맑은 하늘 복사량 (W/m²) |
| `clearsky_dni` | Direct Normal Irradiance |
| `clearsky_dhi` | Diffuse Horizontal Irradiance |

### 2-3. 타임존 — UTC로 검증 완료
국가별 대표 station 7곳에서 `radiation` 가중 평균 시각과 `clearsky_ghi` 가중 평균
시각을 비교한 결과 offset 평균 0.09h (국가별 타임존 1~3h와 무관) → **timestamp는 UTC**.
naive timestamp를 `tz_localize('UTC')`로 변환 후 pvlib에 전달.

저장: `data/train_solar.parquet`, `data/test_solar.parquet`

---

## Step 3. 피처 엔지니어링

**파일**: `src/03_feature_engineering.py`

### 3-1. 시간 피처

```python
df['hour']       = ts.dt.hour
df['minute']     = ts.dt.minute
df['doy']        = ts.dt.dayofyear   # 1~366
df['month']      = ts.dt.month
df['day']        = ts.dt.day

# 주기성 인코딩 (sin/cos): 모델이 "23시와 1시가 가깝다"는 것을 인식하도록
df['hour_sin']   = np.sin(2 * np.pi * (df['hour'] + df['minute']/60) / 24)
df['hour_cos']   = np.cos(2 * np.pi * (df['hour'] + df['minute']/60) / 24)
df['doy_sin']    = np.sin(2 * np.pi * df['doy'] / 365)
df['doy_cos']    = np.cos(2 * np.pi * df['doy'] / 365)
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
```

### 3-2. 태양 피처 파생

```python
z = df['solar_zenith'].clip(0, 90)
# 대기 통과 경로 (zenith가 클수록 대기를 더 많이 통과)
df['air_mass'] = 1 / np.cos(np.radians(z.clip(0, 89)))
# zenith의 cos값 (복사량과 선형에 가까운 관계)
df['cos_zenith'] = np.cos(np.radians(z))
```

### 3-3. 기상 피처
- `temp`, `humidity`, `precip` (원값 그대로 사용)
- 구름과 가장 상관이 높은 것이 humidity → 중요 피처

### 3-4. 관측소 메타 피처
- `latitude`, `longitude`, `elevation_m` (고도, 'elevation' 컬럼과 이름 충돌 주의)
- `installation_height`
- `country` → label encoding

### 3-5. Lag 피처 (동일 (station, 월) 그룹 내에서만)
이전 타임스텝의 기상값만 사용 (radiation 제외 — 타깃 누출 방지). humidity·temp의
lag1(15분 전), lag4(1시간 전):

```python
grp = df.groupby(['station', 'month_id'], sort=False)
for col in ['humidity', 'temp']:
    for lag in [1, 4]:
        df[f'{col}_lag{lag}'] = grp[col].shift(lag).fillna(df[col])
```

월 경계의 NaN은 **현재값으로 대체** (월 시작 시점의 "직전 기상값" 최선 추정).
별도 imputer 불필요.

---

## Step 4. 야간 마스킹

**파일**: `src/04_night_mask.py`

태양 고도각 ≤ 0° → 복사량 = 0 (물리적 확실성)

```python
df['is_night'] = df['solar_elevation'] <= 0
df.loc[df['is_night'], 'radiation'] = 0.0   # train만 (라벨 클리닝)
```

- 전체 데이터의 약 50%를 ML 없이 완벽하게 처리
- MBE, RMSE 모두 개선
- 예측 시에도 동일하게 적용 (야간 예측값 강제로 0)

---

## Step 5. Clear Sky Index (kt) 변환

**파일**: `src/05_clearsky_index.py`

주간 데이터에 대해서만 타깃 변환:

```python
epsilon = 1.0  # 0 나누기 방지

df['kt'] = df['radiation'] / (df['clearsky_ghi'] + epsilon)
df['kt'] = df['kt'].clip(0, 1.5)  # 물리적 범위 초과 클리핑
```

- kt ≈ 1.0: 맑은 하늘
- kt ≈ 0.0: 완전 흐린 날
- kt > 1.0: 반사광 등으로 이론값 초과 (드물지만 발생, 1.5로 clip)

모델이 0~1427 대신 0~1.5 범위를 예측하므로 학습이 안정적.

역변환 (예측 시):
```python
predicted_radiation = predicted_kt * (clearsky_ghi + epsilon)
predicted_radiation = predicted_radiation.clip(0, None)  # 음수 방지
```

### 5-1. 저고도 구간 불안정성 (구현 중 발견)
태양 고도 0~5°에서는 clearsky_ghi가 0에 가까워 `kt = radiation/clearsky`가
폭발한다. 실제로 주간 데이터의 **6.5%가 kt clip 상한(1.5)에 몰려** 있었고,
이는 거의 전부 `clearsky_ghi < 50` 구간에서 발생 (해당 구간 clip 비율 71%).

**대응 (비대칭 처리)** — `MIN_TRAIN_GHI = 50`:
- **학습**: `clearsky_ghi > 50` 안정 구간만 사용 (주간의 89.7%, kt p99 = 1.39로 깨끗)
- **예측**: 전체 주간에 적용. 저고도 test 행은 `kt × 작은 clearsky_ghi`라
  결과 복사량이 자연히 작아 영향 미미.

---

## Step 6. Station별 LightGBM 학습

**파일**: `src/06_train.py`

### 6-1. 피처 목록 (최종)

```python
features = [
    # 태양 물리
    'cos_zenith', 'solar_zenith', 'solar_azimuth', 'air_mass',
    'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi',
    # 시간
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month_sin', 'month_cos', 'hour', 'doy', 'month',
    # 기상
    'temp', 'humidity', 'precip',
    'temp_lag1', 'humidity_lag1',
    'temp_lag4', 'humidity_lag4',
    # 관측소
    'latitude', 'longitude', 'elevation_m',
    'installation_height', 'country_enc',
]                      # 총 28개
target = 'kt'   # 주간만, 야간은 별도 처리
```

### 6-2. LightGBM 하이퍼파라미터

```python
lgb_params = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 30,
    'subsample': 0.8,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
    'n_estimators': 500,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu',     # OpenCL GPU 학습
    'max_bin': 255,      # GPU 권장값
}
```

station별 모델이 ~6천 행으로 작아 과적합 방지를 위해 plan 초안 대비
`num_leaves` 127→63, `n_estimators` 1000→500로 축소.

### 6-3. 학습 전략

Train(홀수월)과 Test(짝수월)는 완전히 분리된 파일이므로 누출 없음.
**최종 모델은 validation 분할 없이 홀수월 전체로 학습**.

학습 데이터는 주간(`is_night == False`) 중 안정 구간(`clearsky_ghi > 50`)만 사용:

```python
day = train[~train['is_night']]
train_fit = day[day['clearsky_ghi'] > MIN_TRAIN_GHI]   # 안정 구간

global_model.fit(train_fit)                  # 전체 station
for each station:
    g = train_fit[station]
    if len(g) >= 100:  station_model[station].fit(g)
    else:              global_model로 대체
```

### 6-3b. CV 진단 (점수·bias 추정)

최종 모델과 별개로, 홀수월 leave-one-month-out CV(6 fold)를 진단용으로 수행.
짝수월 보간 상황을 모사하여 리더보드 점수와 예측 bias를 추정한다.
- 각 fold: 한 홀수월을 빼고 학습 → 그 달 예측 (OOF)
- OOF 복사량으로 RMSE·MBE·Score 계산
- 주간 bias는 08단계 보정에 사용
- CV 모델은 진단용으로 버려지며 최종 모델에 영향 없음
- 결과는 `models/cv_results.pkl`에 저장

### 6-4. Global Fallback 모델
모든 station 데이터를 합쳐서 하나의 global 모델 학습. Station 데이터가 부족하거나 성능이 낮을 때 사용.

### 6-5. 앙상블
```python
final_prediction = 0.7 * station_model_pred + 0.3 * global_model_pred
```

---

## Step 7. 예측 및 역변환

**파일**: `src/07_predict.py`

```python
for each test station:
    # 주간 예측
    kt_pred = model.predict(daytime_features)
    radiation_pred = kt_pred * (clearsky_ghi + epsilon)
    radiation_pred = radiation_pred.clip(0, None)
    
    # 야간 강제 0
    radiation_pred[night_mask] = 0.0
```

---

## Step 8. 후처리

**파일**: `src/08_postprocess.py`

### 8-1. Bias Correction
CV(06-3b)에서 추정한 **주간 전역 bias**를 test 주간 예측에서 차감:

```python
day_bias = cv_results['day_bias']        # CV OOF 주간 평균 (pred - obs)
test_pred[daytime] -= day_bias
```

평가의 MBE는 test 전체 행에 대한 `mean(pred-obs)`이고 야간 행은 오차 0이므로,
주간 예측에서 주간 평균 bias를 빼면 전역 MBE가 0에 가까워진다. (관측소별이 아닌
**전역** 보정 — 메트릭 MBE가 전체 평균이기 때문.) MBE 가중치 0.5라 점수에 직접 기여.

### 8-2. 물리적 범위 클리핑
```python
radiation_pred = radiation_pred.clip(0, 1500)  # 물리적 최대치
```

---

## Step 9. 제출 파일 생성

**파일**: `src/09_submit.py`

```python
submission = pd.DataFrame({
    'ID': test['ID'],
    'TargetMBE': radiation_pred,
    'TargetRMSE': radiation_pred,  # 두 컬럼 동일값
})
submission.to_csv('submission.csv', index=False)
```

---

## 파일 구조

```
Solar_Radiation_Prediction/
├── COMPETITION_INFO.md
├── PLAN.md
├── requirements.txt
├── data/                       # 단계별 중간 산출물 (parquet)
│   ├── {train,test}_clean.parquet
│   ├── {train,test}_solar.parquet
│   ├── {train,test}_feat.parquet
│   ├── {train,test}_masked.parquet
│   ├── {train,test}_final.parquet
│   ├── test_pred.parquet
│   └── test_pred_final.parquet
├── models/
│   ├── models.pkl              # {global, stations:{...}} 통합 저장
│   └── cv_results.pkl          # CV 점수·bias
├── src/
│   ├── config.py               # 경로·피처·파라미터 설정
│   ├── model_utils.py          # 학습/예측 공통 함수
│   ├── 01_preprocess.py
│   ├── 02_solar_features.py
│   ├── 03_feature_engineering.py
│   ├── 04_night_mask.py
│   ├── 05_clearsky_index.py
│   ├── 06_train.py
│   ├── 07_predict.py
│   ├── 08_postprocess.py
│   ├── 09_submit.py
│   └── run_all.py              # 01~09 순차 실행
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
└── submission.csv
```

---

## 구현 우선순위

| 우선순위 | 항목 | 예상 성능 기여 |
|----------|------|----------------|
| 1 | pvlib 태양 위치 + clear sky GHI 피처 | 매우 큼 |
| 2 | 야간 마스킹 (zenith > 90° → 0) | 큼 |
| 3 | Clear Sky Index 타깃 변환 | 중간 |
| 4 | LightGBM (RandomForest → 교체) | 중간 |
| 5 | Station별 모델 | 중간 |
| 6 | Bias Correction 후처리 | 작음 |
| 7 | Lag 피처, 앙상블 | 작음 |

---

## 주의사항

- **데이터 누출 방지**: radiation lag 피처 사용 금지 (타깃 누출). humidity/temp lag만 사용
- **시드 고정**: 모든 랜덤 요소에 `random_state=42`
- **타임존**: timestamp는 UTC로 검증됨 (진행 현황 참고). naive timestamp를 UTC로 localize
- **elevation 컬럼명 충돌**: pvlib 태양 고도각과 관측소 해발고도 충돌 → pvlib 결과는
  `solar_*` 접두사, 관측소 해발고도는 `elevation_m`으로 명명
- **GPU**: LightGBM `device='gpu'`(OpenCL). CPU 전환 시 config의 `device` 줄 제거
- **실행**: `python3 src/run_all.py`로 01~09 일괄 실행, 또는 단계별 개별 실행 가능
  (각 스크립트는 `data/`의 parquet을 읽고 쓰므로 중간 단계부터 재개 가능)
