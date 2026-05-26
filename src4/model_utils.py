"""모델 학습/예측 공통 함수 — XGBoost + LightGBM + CatBoost 앙상블."""
import sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from xgboost.callback import TrainingCallback
from tqdm import tqdm

import config


class _XGBTqdmCallback(TrainingCallback):
    def __init__(self, total, desc):
        self.bar = tqdm(total=total, desc=desc, ncols=80, leave=False, file=sys.stdout)

    def after_iteration(self, model, epoch, evals_log):
        self.bar.update(1)
        return False

    def after_training(self, model):
        self.bar.close()
        return model


def make_xgb(callbacks=None):
    if callbacks:
        return xgb.XGBRegressor(**config.XGB_PARAMS, callbacks=callbacks)
    return xgb.XGBRegressor(**config.XGB_PARAMS)


def make_lgb():
    return lgb.LGBMRegressor(**config.LGB_PARAMS)


def make_cat():
    return cat.CatBoostRegressor(**config.CAT_PARAMS)


def _fit_trio(X, y, desc_prefix=""):
    """XGBoost + LightGBM + CatBoost 3개 모델 학습."""
    n = config.XGB_PARAMS["n_estimators"]

    xgb_m = make_xgb(callbacks=[_XGBTqdmCallback(n, f"    {desc_prefix}XGB")])
    xgb_m.fit(X, y)
    xgb_m.callbacks = None

    lgb_m = make_lgb()
    lgb_m.fit(X, y)

    cat_m = make_cat()
    cat_m.fit(X, y)

    return (xgb_m, lgb_m, cat_m)


def _predict_trio(models, X):
    """3개 모델 가중 평균 예측."""
    preds = [m.predict(X) for m in models]
    w = config.ENSEMBLE_MODEL_W
    return sum(p * wi for p, wi in zip(preds, w))


def train_station_models(train_day, desc="station 모델"):
    """주간 학습 데이터로 global 앙상블 1세트 + station별 앙상블 학습."""
    X_all = train_day[config.FEATURES]
    y_all = train_day[config.TARGET]

    tqdm.write(f"  [global] 학습 중 ({len(train_day):,}행) — XGB+LGB+CAT")
    global_models = _fit_trio(X_all, y_all, desc_prefix="global ")
    tqdm.write(f"  [global] 완료")

    groups = [(s, g) for s, g in train_day.groupby("station", sort=False)
              if len(g) >= config.MIN_STATION_ROWS]
    station_models = {}
    for station, g in tqdm(groups, desc=f"  {desc}", ncols=80,
                           leave=True, file=sys.stdout):
        station_models[station] = _fit_trio(g[config.FEATURES], g[config.TARGET])
    return global_models, station_models


def predict_kt(df, global_models, station_models):
    """앙상블 kt 예측. station 모델이 있으면 가중 블렌딩, 없으면 global."""
    X = df[config.FEATURES]
    g_pred = _predict_trio(global_models, X)
    pred = g_pred.copy()
    stations = df["station"].values
    for station, models in station_models.items():
        mask = stations == station
        if mask.any():
            s_pred = _predict_trio(models, X[mask])
            pred[mask] = (config.ENSEMBLE_W_STATION * s_pred
                          + config.ENSEMBLE_W_GLOBAL * g_pred[mask])
    return pred


def kt_to_radiation(kt, clearsky_ghi):
    kt = np.clip(kt, 0, config.KT_CLIP)
    return kt * (clearsky_ghi + config.CLEARSKY_EPS)
