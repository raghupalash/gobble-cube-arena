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

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"
ZONE_PAIR_HOUR_MEANS_PATH = Path(__file__).parent / "zone_pair_hour_means.pkl"
ZONE_CENTROIDS_PATH = DATA_DIR / "zone_centroids.csv"

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "zone_pair_mean", "haversine_km", "is_rush_hour", "is_weekend", "is_airport", "zone_pair_hour_mean"]

MIN_BUCKET_COUNT = 30

_AIRPORT_ZONES = {1, 132, 138}  # EWR, JFK, LGA

_R = 6371.0  # Earth radius in km


def load_zone_centroids() -> dict:
    df = pd.read_csv(ZONE_CENTROIDS_PATH)
    return {row.zone_id: (row.latitude, row.longitude) for row in df.itertuples()}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _R * np.arcsin(np.sqrt(a))


def build_zone_pair_hour_means(train: pd.DataFrame) -> dict:
    def trimmed_mean(x: pd.Series) -> float:
        if len(x) < MIN_BUCKET_COUNT:
            return None
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        trimmed = x[(x >= lo) & (x <= hi)]
        return trimmed.mean() if len(trimmed) else x.mean()

    ts = pd.to_datetime(train["requested_at"])
    train = train.copy()
    train["hour"] = ts.dt.hour
    means = train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].apply(trimmed_mean)
    return means.dropna().to_dict()


def build_zone_pair_means(train: pd.DataFrame) -> dict:
    # trimmed mean: drop bottom and top 10% per pair, average the middle 80%
    def trimmed_mean(x: pd.Series) -> float:
        lo, hi = x.quantile(0.10), x.quantile(0.90)
        trimmed = x[(x >= lo) & (x <= hi)]
        return trimmed.mean() if len(trimmed) else x.mean()

    means = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].apply(trimmed_mean)
    return means.to_dict()


def engineer_features(
    df: pd.DataFrame,
    zone_pair_means: dict,
    zone_pair_hour_means: dict,
    global_mean: float,
    centroids: dict,
) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])
    hours = ts.dt.hour.astype(int)
    pair_mean = [
        zone_pair_means.get((int(pu), int(do)), global_mean)
        for pu, do in zip(df["pickup_zone"], df["dropoff_zone"])
    ]
    pair_hour_mean = [
        zone_pair_hour_means.get(
            (int(pu), int(do), int(h)),
            zone_pair_means.get((int(pu), int(do)), global_mean)
        )
        for pu, do, h in zip(df["pickup_zone"], df["dropoff_zone"], hours)
    ]
    # fallback to 0.0 for any zone not in the shapefile (2 zones missing)
    hav = [
        haversine(*centroids[pu], *centroids[do])
        if pu in centroids and do in centroids else 0.0
        for pu, do in zip(df["pickup_zone"].astype(int), df["dropoff_zone"].astype(int))
    ]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    is_rush_hour = (
        ((hour >= 7) & (hour < 9) & (dow < 5)) |
        ((hour >= 16) & (hour < 19) & (dow < 5))
    ).astype("int8")
    is_weekend = (dow >= 5).astype("int8")
    is_airport = (
        df["pickup_zone"].astype(int).isin(_AIRPORT_ZONES) |
        df["dropoff_zone"].astype(int).isin(_AIRPORT_ZONES)
    ).astype("int8")
    return pd.DataFrame({
        "pickup_zone":    df["pickup_zone"].astype("int32"),
        "dropoff_zone":   df["dropoff_zone"].astype("int32"),
        "hour":           hour.astype("int8"),
        "dow":            dow.astype("int8"),
        "month":          ts.dt.month.astype("int8"),
        "zone_pair_mean": pair_mean,
        "haversine_km":   hav,
        "is_rush_hour":          is_rush_hour,
        "is_weekend":            is_weekend,
        "is_airport":            is_airport,
        "zone_pair_hour_mean":   pair_hour_mean,
    })[FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Train on full 37M dataset instead of 1M sample")
    args = parser.parse_args()

    train_path = DATA_DIR / ("train.parquet" if args.full else "sample_1M.parquet")
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )
    if args.full:
        print("** Full training mode (37M rows) **")

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

    print(f"\nBuilding zone-pair-hour means (min count={MIN_BUCKET_COUNT})...")
    zone_pair_hour_means = build_zone_pair_hour_means(train)
    print(f"  {len(zone_pair_hour_means):,} buckets with >= {MIN_BUCKET_COUNT} trips")

    X_train = engineer_features(train, zone_pair_means, zone_pair_hour_means, global_mean, centroids)
    y_train = np.log1p(train["duration_seconds"].to_numpy())
    X_dev = engineer_features(dev, zone_pair_means, zone_pair_hour_means, global_mean, centroids)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
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

    preds = np.expm1(model.predict(X_dev))
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")

    with open(ZONE_PAIR_MEANS_PATH, "wb") as f:
        pickle.dump({"means": zone_pair_means, "global_mean": global_mean}, f)
    print(f"Saved zone-pair means to {ZONE_PAIR_MEANS_PATH}")

    with open(ZONE_PAIR_HOUR_MEANS_PATH, "wb") as f:
        pickle.dump(zone_pair_hour_means, f)
    print(f"Saved zone-pair-hour means to {ZONE_PAIR_HOUR_MEANS_PATH}")


if __name__ == "__main__":
    main()
