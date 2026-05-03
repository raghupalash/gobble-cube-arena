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

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"
ZONE_CENTROIDS_PATH = DATA_DIR / "zone_centroids.csv"

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "zone_pair_mean", "haversine_km"]

_R = 6371.0  # Earth radius in km


def load_zone_centroids() -> dict:
    df = pd.read_csv(ZONE_CENTROIDS_PATH)
    return {row.zone_id: (row.latitude, row.longitude) for row in df.itertuples()}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _R * np.arcsin(np.sqrt(a))


def build_zone_pair_means(train: pd.DataFrame) -> dict:
    means = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].mean()
    return means.to_dict()


def engineer_features(
    df: pd.DataFrame,
    zone_pair_means: dict,
    global_mean: float,
    centroids: dict,
) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    pair_mean = [
        zone_pair_means.get((int(pu), int(do)), global_mean)
        for pu, do in zip(df["pickup_zone"], df["dropoff_zone"])
    ]
    # fallback to 0.0 for any zone not in the shapefile (2 zones missing)
    hav = [
        haversine(*centroids[pu], *centroids[do])
        if pu in centroids and do in centroids else 0.0
        for pu, do in zip(df["pickup_zone"].astype(int), df["dropoff_zone"].astype(int))
    ]
    return pd.DataFrame({
        "pickup_zone":    df["pickup_zone"].astype("int32"),
        "dropoff_zone":   df["dropoff_zone"].astype("int32"),
        "hour":           ts.dt.hour.astype("int8"),
        "dow":            ts.dt.dayofweek.astype("int8"),
        "month":          ts.dt.month.astype("int8"),
        "zone_pair_mean": pair_mean,
        "haversine_km":   hav,
    })[FEATURES]


def main() -> None:
    train_path = DATA_DIR / "sample_1M.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nLoading zone centroids...")
    centroids = load_zone_centroids()
    print(f"  {len(centroids)} zones loaded")

    print("\nBuilding zone-pair means...")
    zone_pair_means = build_zone_pair_means(train)
    global_mean = float(train["duration_seconds"].mean())
    print(f"  {len(zone_pair_means):,} unique zone pairs | global mean: {global_mean:.1f}s")

    X_train = engineer_features(train, zone_pair_means, global_mean, centroids)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev, zone_pair_means, global_mean, centroids)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")

    with open(ZONE_PAIR_MEANS_PATH, "wb") as f:
        pickle.dump({"means": zone_pair_means, "global_mean": global_mean}, f)
    print(f"Saved zone-pair means to {ZONE_PAIR_MEANS_PATH}")


if __name__ == "__main__":
    main()
