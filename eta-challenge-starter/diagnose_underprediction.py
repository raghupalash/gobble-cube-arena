#!/usr/bin/env python3
"""Diagnostic: root causes of long-trip under-prediction."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"
ZONE_PAIR_HOUR_MEANS_PATH = Path(__file__).parent / "zone_pair_hour_means.pkl"

def main():
    print("Loading artifacts...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ZONE_PAIR_MEANS_PATH, "rb") as f:
        zp = pickle.load(f)
    zp_means, global_mean = zp["means"], zp["global_mean"]
    hour_means = {}
    if ZONE_PAIR_HOUR_MEANS_PATH.exists():
        with open(ZONE_PAIR_HOUR_MEANS_PATH, "rb") as f:
            hour_means = pickle.load(f)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from baseline import engineer_features, load_zone_centroids
    centroids = load_zone_centroids()

    print("Loading dev data...")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    X_dev = engineer_features(dev, zp_means, hour_means, global_mean, centroids)

    y_true = dev["duration_seconds"].to_numpy()
    y_pred = np.expm1(model.predict(X_dev))
    residuals = y_pred - y_true

    ts = pd.to_datetime(dev["requested_at"])
    dev = dev.copy()
    dev["hour"] = ts.dt.hour
    dev["residual"] = residuals
    dev["y_pred"] = y_pred
    dev["zone_pair_mean"] = X_dev["zone_pair_mean"].values

    # =========================================================
    # HYPOTHESIS 2: Is zone_pair_mean over-anchoring?
    # For worst under-predictions, how close is y_pred to zone_pair_mean?
    # =========================================================
    print("\n\n" + "="*60)
    print("HYPOTHESIS 2: zone_pair_mean anchoring")
    print("="*60)

    under = dev[dev["residual"] < 0].copy()
    long_under = under[under["duration_seconds"] > 1800].copy()  # >30min, under-predicted

    long_under["pred_vs_zpm"] = long_under["y_pred"] - long_under["zone_pair_mean"]
    long_under["true_vs_zpm"] = long_under["duration_seconds"] - long_under["zone_pair_mean"]

    print(f"\nLong trips (>30min) that are under-predicted: {len(long_under):,}")
    print(f"\n  zone_pair_mean stats for these trips:")
    print(f"    mean zone_pair_mean:  {long_under['zone_pair_mean'].mean():.1f}s")
    print(f"    mean y_pred:          {long_under['y_pred'].mean():.1f}s")
    print(f"    mean y_true:          {long_under['duration_seconds'].mean():.1f}s")
    print(f"\n  How far does the model deviate from zone_pair_mean?")
    print(f"    mean(y_pred - zone_pair_mean): {long_under['pred_vs_zpm'].mean():+.1f}s")
    print(f"    median(y_pred - zone_pair_mean): {long_under['pred_vs_zpm'].median():+.1f}s")
    print(f"\n  How far is y_true from zone_pair_mean?")
    print(f"    mean(y_true - zone_pair_mean):  {long_under['true_vs_zpm'].mean():+.1f}s")
    print(f"    median(y_true - zone_pair_mean): {long_under['true_vs_zpm'].median():+.1f}s")

    print(f"\n  Anchoring ratio (how much of the gap does model close?):")
    gap = long_under["true_vs_zpm"].mean()
    model_move = long_under["pred_vs_zpm"].mean()
    print(f"    True is {gap:.1f}s above zone_pair_mean")
    print(f"    Model moves {model_move:.1f}s above zone_pair_mean")
    print(f"    Model closes {model_move/gap*100:.1f}% of the gap")

    # By duration bucket
    bins   = [1800, 3600, 7200, 999999]
    labels = ["30-60min", "1-2hr", ">2hr"]
    long_under["dur_bucket"] = pd.cut(long_under["duration_seconds"], bins=bins, labels=labels)
    print(f"\n  Anchoring by duration bucket:")
    for bucket, grp in long_under.groupby("dur_bucket", observed=True):
        gap_b = grp["true_vs_zpm"].mean()
        move_b = grp["pred_vs_zpm"].mean()
        pct = move_b / gap_b * 100 if gap_b != 0 else 0
        print(f"    {bucket}: true is {gap_b:+.0f}s above zpm, model moves {move_b:+.0f}s ({pct:.1f}% of gap)")

    # =========================================================
    # HYPOTHESIS 3: High-variance routes
    # =========================================================
    print("\n\n" + "="*60)
    print("HYPOTHESIS 3: Structurally high-variance routes")
    print("="*60)

    print("\nLoading training data (1M sample)...")
    train = pd.read_parquet(DATA_DIR / "sample_1M.parquet")

    pair_stats = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].agg(
        n="count", mean="mean", std="std", median="median"
    ).dropna()
    pair_stats["cv"] = pair_stats["std"] / pair_stats["mean"]  # coefficient of variation

    print(f"\n  Overall CV distribution across zone pairs (min 20 trips):")
    high_vol = pair_stats[pair_stats["n"] >= 20]
    for p in [25, 50, 75, 90, 95]:
        print(f"    P{p}: CV={high_vol['cv'].quantile(p/100):.2f}")

    # For the worst under-predicted pairs — what's their CV?
    worst_under_pairs = (
        dev[dev["residual"] < -600]
        .groupby(["pickup_zone", "dropoff_zone"])
        .size()
        .reset_index(name="n_bad")
        .nlargest(20, "n_bad")
    )
    worst_under_pairs = worst_under_pairs.merge(
        pair_stats[["cv", "mean", "std", "n"]].reset_index(),
        on=["pickup_zone", "dropoff_zone"],
        how="left"
    )
    print(f"\n  Top 20 most frequently under-predicted pairs (>600s miss) — their CV in training:")
    print(worst_under_pairs.to_string(index=False))

    print(f"\n  Mean CV of worst under-predicted pairs: {worst_under_pairs['cv'].mean():.2f}")
    print(f"  Mean CV of all pairs (min 20 trips):    {high_vol['cv'].mean():.2f}")

    # =========================================================
    # HYPOTHESIS 4: Long trip representation in train vs dev
    # =========================================================
    print("\n\n" + "="*60)
    print("HYPOTHESIS 4: Long trip representation — train vs dev")
    print("="*60)

    bins   = [0, 300, 600, 1200, 1800, 3600, 7200, 999999]
    labels = ["<5min", "5-10min", "10-20min", "20-30min", "30-60min", "1-2hr", ">2hr"]

    train_dist = pd.cut(train["duration_seconds"], bins=bins, labels=labels).value_counts(normalize=True).sort_index() * 100
    dev_dist   = pd.cut(dev["duration_seconds"],   bins=bins, labels=labels).value_counts(normalize=True).sort_index() * 100

    comparison = pd.DataFrame({"train_1M%": train_dist, "dev%": dev_dist}).round(2)
    comparison["dev_vs_train_ratio"] = (comparison["dev%"] / comparison["train_1M%"]).round(2)
    print(f"\n  Duration distribution — train (1M) vs dev:")
    print(comparison.to_string())

    print(f"\n  Absolute counts:")
    train_counts = pd.cut(train["duration_seconds"], bins=bins, labels=labels).value_counts().sort_index()
    dev_counts   = pd.cut(dev["duration_seconds"],   bins=bins, labels=labels).value_counts().sort_index()
    counts = pd.DataFrame({"train_1M": train_counts, "dev": dev_counts})
    print(counts.to_string())


if __name__ == "__main__":
    main()
