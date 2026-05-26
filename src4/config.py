"""공통 설정: 경로, 컬럼 매핑, 피처 목록, 모델 파라미터."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models4"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# 원본 CSV
TRAIN_CSV = BASE_DIR / "Train.csv"
TEST_CSV = BASE_DIR / "Test.csv"
SAMPLE_SUB_CSV = BASE_DIR / "SampleSubmission.csv"
SUBMISSION_CSV = BASE_DIR / "submission4.csv"

RENAME_MAP = {
    "precipitation (mm)": "precip",
    "radiation (W/m2)": "radiation",
    "relativehumidity (-)": "humidity",
    "temperature (degrees Celsius)": "temp",
    "elevation": "elevation_m",
}

CLEARSKY_MODEL = "ineichen"
CLEARSKY_EPS = 1.0
KT_CLIP = 1.5
RAD_CLIP_MAX = 1500.0
MIN_TRAIN_GHI = 50.0

TARGET = "kt"

FEATURES = [
    # 태양 물리
    "cos_zenith", "solar_zenith", "solar_azimuth", "air_mass",
    "clearsky_ghi", "clearsky_dni", "clearsky_dhi",
    # 시간
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "hour", "doy", "month",
    # 기상 현재값
    "temp", "humidity", "precip", "dewpoint",
    # lag 피처
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_lag4", "temp_lag8", "temp_lag24",
    "humidity_lag1", "humidity_lag2", "humidity_lag3", "humidity_lag4",
    "humidity_lag8", "humidity_lag24",
    "precip_lag1", "precip_lag4",
    # 변화량
    "temp_diff1", "temp_diff4",
    # rolling 통계
    "temp_roll3_mean", "temp_roll6_mean", "temp_roll24_mean",
    "humidity_roll3_mean", "humidity_roll6_mean", "humidity_roll24_mean",
    "precip_roll6_sum",
    # 관측소
    "latitude", "longitude", "elevation_m", "installation_height", "country_enc",
]

# XGBoost CUDA 파라미터
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "device": "cuda",
    "tree_method": "hist",
    "verbosity": 0,
}

# LightGBM GPU(OpenCL) 파라미터
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 5,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
    "device": "gpu",
    "max_bin": 255,
}

# CatBoost GPU 파라미터
CAT_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.05,
    "depth": 7,
    "min_data_in_leaf": 30,
    "bootstrap_type": "Bernoulli",  # GPU에서 subsample 사용 시 필요
    "subsample": 0.8,
    "reg_lambda": 0.2,
    "iterations": 500,
    "random_seed": RANDOM_STATE,
    "task_type": "GPU",
    "verbose": 0,
}

# 앙상블 모델 가중치 (XGB : LGB : CAT)
ENSEMBLE_MODEL_W = [0.4, 0.3, 0.3]

ENSEMBLE_W_STATION = 0.7
ENSEMBLE_W_GLOBAL = 0.3
MIN_STATION_ROWS = 100
