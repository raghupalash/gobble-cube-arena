#!/usr/bin/env python3
"""Error analysis on dev set. Run after training to find failure patterns."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"
ZONE_PAIR_HOUR_MEANS_PATH = Path(__file__).parent / "zone_pair_hour_means.pkl"

TOP_N = 5000

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ZONE_PAIR_MEANS_PATH, "rb") as f:
        zp = pickle.load(f)
    hour_means = {}
    if ZONE_PAIR_HOUR_MEANS_PATH.exists():
        with open(ZONE_PAIR_HOUR_MEANS_PATH, "rb") as f:
            hour_means = pickle.load(f)
    return model, zp["means"], zp["global_mean"], hour_means

def main():
    print("Loading artifacts and dev data...")
    model, zp_means, global_mean, hour_means = load_artifacts()
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")

    # Replicate feature engineering inline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from baseline import engineer_features
    X_dev = engineer_features(dev, zp_means, hour_means, global_mean, {})

    # Need centroids for haversine — skip for analysis (haversine will be 0, still valid for residuals)
    # Actually re-do with centroids
    from baseline import load_zone_centroids
    centroids = load_zone_centroids()
    X_dev = engineer_features(dev, zp_means, hour_means, global_mean, centroids)

    y_true = dev["duration_seconds"].to_numpy()
    y_pred = np.expm1(model.predict(X_dev))
    residuals = y_pred - y_true  # positive = over-predict, negative = under-predict
    abs_errors = np.abs(residuals)

    ts = pd.to_datetime(dev["requested_at"])
    dev = dev.copy()
    dev["hour"] = ts.dt.hour
    dev["dow"] = ts.dt.dayofweek
    dev["abs_error"] = abs_errors
    dev["residual"] = residuals
    dev["y_pred"] = y_pred

    overall_mae = abs_errors.mean()
    print(f"\nOverall MAE: {overall_mae:.1f}s  (n={len(dev):,})")

    # --- Top N worst errors ---
    worst = dev.nlargest(TOP_N, "abs_error")
    print(f"\n=== Top {TOP_N} worst errors (MAE={worst['abs_error'].mean():.1f}s) ===")
    print(f"  Under-predictions (pred < true): {(worst['residual'] < 0).sum():,} ({(worst['residual'] < 0).mean()*100:.1f}%)")
    print(f"  Over-predictions  (pred > true): {(worst['residual'] > 0).sum():,} ({(worst['residual'] > 0).mean()*100:.1f}%)")
    print(f"  True duration — median: {worst['duration_seconds'].median():.0f}s, mean: {worst['duration_seconds'].mean():.0f}s")
    print(f"  True duration — >1h: {(worst['duration_seconds'] > 3600).mean()*100:.1f}%,  >30min: {(worst['duration_seconds'] > 1800).mean()*100:.1f}%")

    # --- Errors by hour ---
    print("\n=== MAE by hour (all dev) ===")
    by_hour = dev.groupby("hour")["abs_error"].mean().round(1)
    for h, mae in by_hour.items():
        bar = "█" * int(mae / 10)
        print(f"  {h:02d}h  {mae:6.1f}s  {bar}")

    # --- Errors by day of week ---
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\n=== MAE by day of week (all dev) ===")
    by_dow = dev.groupby("dow")["abs_error"].mean().round(1)
    for d, mae in by_dow.items():
        print(f"  {dow_names[d]}  {mae:6.1f}s")

    # --- Zone 264/265 analysis ---
    print("\n=== Zone 264/265 (unknown/OOB) ===")
    for z in [264, 265]:
        mask = (dev["pickup_zone"] == z) | (dev["dropoff_zone"] == z)
        if mask.sum() == 0:
            print(f"  Zone {z}: not present in dev")
            continue
        z_mae = dev[mask]["abs_error"].mean()
        z_pct = mask.mean() * 100
        worst_pct = ((worst["pickup_zone"] == z) | (worst["dropoff_zone"] == z)).mean() * 100
        print(f"  Zone {z}: {mask.sum():,} trips ({z_pct:.2f}% of dev), MAE={z_mae:.1f}s, {worst_pct:.1f}% of top-{TOP_N} errors")

    # --- Worst zone pairs ---
    print(f"\n=== Top 15 zone pairs by MAE (min 20 trips in dev) ===")
    pair_stats = dev.groupby(["pickup_zone", "dropoff_zone"]).agg(
        n=("abs_error", "count"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
    ).query("n >= 20").nlargest(15, "mae").round(1)
    print(pair_stats.to_string())

    # --- Worst hours in top errors ---
    print(f"\n=== Hour distribution in top {TOP_N} vs overall ===")
    overall_hour_pct = dev.groupby("hour").size() / len(dev) * 100
    worst_hour_pct = worst.groupby("hour").size() / len(worst) * 100
    combined = pd.DataFrame({"overall%": overall_hour_pct, "worst%": worst_hour_pct}).fillna(0).round(1)
    combined["lift"] = (combined["worst%"] / combined["overall%"]).round(2)
    print(combined[combined["lift"] > 1.3].to_string())

    # --- Residual distribution ---
    print(f"\n=== Residual percentiles (all dev) ===")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  P{p:2d}: {np.percentile(residuals, p):+.0f}s")

    # --- Short trip analysis ---
    short = dev[dev["duration_seconds"] < 120]
    if len(short):
        print(f"\n=== Short trips (<2min): {len(short):,} trips, MAE={short['abs_error'].mean():.1f}s ===")
        print(f"  Bias (pred-true): {short['residual'].mean():+.1f}s")

if __name__ == "__main__":
    main()
