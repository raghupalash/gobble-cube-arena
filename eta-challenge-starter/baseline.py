#!/usr/bin/env python
"""Baseline: gradient-boosted trees on six simple features.

Trains in ~5 minutes on a laptop CPU. Produces `model.pkl` which `predict.py`
loads at inference.

Prerequisites:
    python data/download_data.py   # one-time, ~500 MB download

Run:
    python baseline.py             # trains and saves model.pkl

Your job is to replace this file with something better. The grader only cares
about `predict.py` — this file just needs to produce a `model.pkl` that
`predict.py` can load.
"""

#!/usr/bin/env python
from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
METADATA_PATH = Path(__file__).parent / "metadata.pkl"
ZONE_CENTROIDS_PATH = DATA_DIR / "zone_centroids.csv"

_AIRPORT_ZONES = {1, 132, 138}
_R = 6371.0

def load_zone_centroids() -> dict:
    df = pd.read_csv(ZONE_CENTROIDS_PATH)
    return {row.zone_id: (row.latitude, row.longitude) for row in df.itertuples()}

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * _R * np.arcsin(np.sqrt(a))

def build_metadata(train: pd.DataFrame) -> dict:
    print("Computing trimmed means and congestion stats...")
    def get_trimmed_mean(group):
        q1, q3 = group.quantile(0.1), group.quantile(0.9)
        return group[(group >= q1) & (group <= q3)].mean()

    # Zone Pair Means
    zp_means = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].apply(get_trimmed_mean).to_dict()
    # Marginal Means (Congestion proxies)
    pu_means = train.groupby("pickup_zone")["duration_seconds"].apply(get_trimmed_mean).to_dict()
    do_means = train.groupby("dropoff_zone")["duration_seconds"].apply(get_trimmed_mean).to_dict()
    
    return {
        "zp_means": zp_means,
        "pu_means": pu_means,
        "do_means": do_means,
        "global_mean": float(train["duration_seconds"].mean())
    }

def engineer_features(df: pd.DataFrame, meta: dict, centroids: dict) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    pu = df["pickup_zone"].astype(int)
    do = df["dropoff_zone"].astype(int)
    
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    # Feature Engineering
    df_feat = pd.DataFrame({
        "pickup_zone": pu.astype("category"),
        "dropoff_zone": do.astype("category"),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": dow.astype("int8"),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "haversine_km": [
            haversine(*centroids[p], *centroids[d]) if p in centroids and d in centroids else 0.0 
            for p, d in zip(pu, do)
        ],
        "zp_mean": [meta["zp_means"].get((p, d), meta["global_mean"]) for p, d in zip(pu, do)],
        "pu_congestion": pu.map(meta["pu_means"]).fillna(meta["global_mean"]),
        "do_congestion": do.map(meta["do_means"]).fillna(meta["global_mean"]),
        "is_rush": ((hour.between(7, 9) | hour.between(16, 19)) & (dow < 5)).astype("int8"),
        "is_airport": (pu.isin(_AIRPORT_ZONES) | do.isin(_AIRPORT_ZONES)).astype("int8"),
    })
    return df_feat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    train_path = DATA_DIR / ("train.parquet" if args.full else "sample_1M.parquet")
    train = pd.read_parquet(train_path)
    train = train[train["duration_seconds"].between(60, 10800)] # Filter <1m and >3h
    
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    centroids = load_zone_centroids()
    meta = build_metadata(train)

    X_train = engineer_features(train, meta, centroids)
    y_train = np.log1p(train["duration_seconds"].to_numpy())
    
    X_dev = engineer_features(dev, meta, centroids)
    y_dev = dev["duration_seconds"].to_numpy()

    print("Training XGBoost (optimized for MAE tail)...")
    # Using larger learning rate with more trees for 37M
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True, 
        n_jobs=-1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.0f}s")

    preds = np.expm1(model.predict(X_dev))
    print(f"Dev MAE: {np.mean(np.abs(preds - y_dev)):.2f}s")

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(METADATA_PATH, "wb") as f: pickle.dump(meta, f)

if __name__ == "__main__":
    main()