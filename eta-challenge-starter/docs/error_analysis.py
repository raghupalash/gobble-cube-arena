#!/usr/bin/env python
"""Error analysis on dev set predictions.

Loads model.pkl, runs inference on dev.parquet, pulls worst 1000 predictions
by absolute error, finds clusters, checks bias, writes docs/error-analysis.md.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

AIRPORT_ZONES = {1, 132, 138}  # EWR, JFK, LGA

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "zone_pair_mean", "haversine_km"]

_R = 6371.0


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _R * np.arcsin(np.sqrt(a))


def load_artifacts():
    with open(ROOT / "model.pkl", "rb") as f:
        model = pickle.load(f)
    if hasattr(model, "get_booster"):
        model.get_booster().feature_names = None
    with open(ROOT / "zone_pair_means.pkl", "rb") as f:
        zp = pickle.load(f)
    centroids_df = pd.read_csv(DATA_DIR / "zone_centroids.csv")
    centroids = {row.zone_id: (row.latitude, row.longitude) for row in centroids_df.itertuples()}
    return model, zp["means"], zp["global_mean"], centroids


def engineer_features(df, zone_pair_means, global_mean, centroids):
    ts = pd.to_datetime(df["requested_at"])
    pu = df["pickup_zone"].astype(int)
    do = df["dropoff_zone"].astype(int)
    pair_mean = [zone_pair_means.get((p, d), global_mean) for p, d in zip(pu, do)]
    hav = [
        haversine_vec(*centroids[p], *centroids[d])
        if p in centroids and d in centroids else 0.0
        for p, d in zip(pu, do)
    ]
    return pd.DataFrame({
        "pickup_zone":    pu.astype("int32"),
        "dropoff_zone":   do.astype("int32"),
        "hour":           ts.dt.hour.astype("int8"),
        "dow":            ts.dt.dayofweek.astype("int8"),
        "month":          ts.dt.month.astype("int8"),
        "zone_pair_mean": pair_mean,
        "haversine_km":   hav,
    })[FEATURES]


def hour_bucket(h):
    if 7 <= h <= 9:   return "morning_rush"
    if 16 <= h <= 19: return "evening_rush"
    if h >= 22 or h <= 5: return "night"
    return "off_peak"


def duration_bucket(s):
    if s < 300:   return "short(<5m)"
    if s < 1200:  return "medium(5-20m)"
    if s < 3600:  return "long(20-60m)"
    return "very_long(>60m)"


def main():
    print("Loading artifacts...")
    model, zone_pair_means, global_mean, centroids = load_artifacts()

    print("Loading dev.parquet...")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")
    print(f"  {len(dev):,} rows")

    print("Engineering features...")
    X_dev = engineer_features(dev, zone_pair_means, global_mean, centroids)

    print("Running inference...")
    preds = model.predict(X_dev)
    actual = dev["duration_seconds"].to_numpy()

    signed_error = preds - actual
    abs_error = np.abs(signed_error)
    overall_mae = abs_error.mean()
    print(f"  Overall dev MAE: {overall_mae:.1f}s")
    print(f"  Overall signed error (bias): {signed_error.mean():+.1f}s")

    # --- Worst 1000 ---
    ts = pd.to_datetime(dev["requested_at"])
    analysis = pd.DataFrame({
        "pickup_zone":   dev["pickup_zone"].astype(int).values,
        "dropoff_zone":  dev["dropoff_zone"].astype(int).values,
        "actual":        actual,
        "predicted":     preds,
        "signed_error":  signed_error,
        "abs_error":     abs_error,
        "hour":          ts.dt.hour.values,
        "zone_pair_mean": X_dev["zone_pair_mean"].values,
    })
    analysis["is_airport"] = (
        analysis["pickup_zone"].isin(AIRPORT_ZONES) |
        analysis["dropoff_zone"].isin(AIRPORT_ZONES)
    )
    analysis["hour_bucket"]     = analysis["hour"].map(hour_bucket)
    analysis["duration_bucket"] = analysis["actual"].map(duration_bucket)

    worst = analysis.nlargest(1000, "abs_error").copy()

    print(f"\n--- Worst 1000 ---")
    print(f"  Mean abs error:    {worst['abs_error'].mean():.1f}s")
    print(f"  Mean signed error: {worst['signed_error'].mean():+.1f}s")
    print(f"  Median actual:     {worst['actual'].median():.0f}s")
    print(f"  Airport trips:     {worst['is_airport'].sum()} / 1000")

    # --- Cluster analysis ---
    n_total = len(analysis)
    n_worst = len(worst)

    def cluster_report(col, top_n=5):
        overall_dist = analysis[col].value_counts(normalize=True)
        worst_dist   = worst[col].value_counts(normalize=True)
        rows = []
        for val in worst_dist.index[:top_n]:
            w_pct  = worst_dist.get(val, 0) * 100
            o_pct  = overall_dist.get(val, 0) * 100
            ratio  = w_pct / o_pct if o_pct > 0 else float("inf")
            mae_contribution = worst[worst[col] == val]["abs_error"].sum() / (worst["abs_error"].sum()) * 100
            bias   = worst[worst[col] == val]["signed_error"].mean()
            rows.append({
                "value": val,
                "worst_%": round(w_pct, 1),
                "overall_%": round(o_pct, 1),
                "ratio": round(ratio, 1),
                "mae_share_%": round(mae_contribution, 1),
                "bias_s": round(bias, 0),
            })
        return pd.DataFrame(rows)

    print("\n--- Clusters by hour_bucket ---")
    print(cluster_report("hour_bucket").to_string(index=False))
    print("\n--- Clusters by duration_bucket ---")
    print(cluster_report("duration_bucket").to_string(index=False))
    print("\n--- Clusters by is_airport ---")
    print(cluster_report("is_airport").to_string(index=False))

    # Top pickup zones in worst 1000
    print("\n--- Top pickup zones in worst 1000 ---")
    print(cluster_report("pickup_zone", top_n=8).to_string(index=False))

    print("\n--- Top dropoff zones in worst 1000 ---")
    print(cluster_report("dropoff_zone", top_n=8).to_string(index=False))

    # --- Bad labels ---
    bad_labels = analysis[(analysis["actual"] > 14400) | (analysis["actual"] < 60)]
    outlier_pair = analysis[analysis["actual"] > 3 * analysis["zone_pair_mean"]]
    print(f"\n--- Bad labels ---")
    print(f"  Trips > 4 hours or < 1 min: {len(bad_labels):,} ({len(bad_labels)/n_total*100:.2f}% of dev)")
    print(f"  Trips > 3x zone_pair_mean:  {len(outlier_pair):,} ({len(outlier_pair)/n_total*100:.2f}% of dev)")
    if len(bad_labels):
        print(f"  Bad label actual range: {bad_labels['actual'].min():.0f}s – {bad_labels['actual'].max():.0f}s")
    print(f"  Bad labels in worst 1000: {worst[worst['actual'] > 14400]['abs_error'].count()}")

    # --- Save results for markdown ---
    return {
        "overall_mae": overall_mae,
        "overall_bias": signed_error.mean(),
        "worst_mae": worst["abs_error"].mean(),
        "worst_bias": worst["signed_error"].mean(),
        "bad_label_count": len(bad_labels),
        "bad_label_pct": len(bad_labels) / n_total * 100,
        "outlier_pair_pct": len(outlier_pair) / n_total * 100,
        "hour_clusters":     cluster_report("hour_bucket"),
        "duration_clusters": cluster_report("duration_bucket"),
        "airport_clusters":  cluster_report("is_airport"),
        "pickup_clusters":   cluster_report("pickup_zone", top_n=8),
        "dropoff_clusters":  cluster_report("dropoff_zone", top_n=8),
        "worst": worst,
        "analysis": analysis,
    }


if __name__ == "__main__":
    results = main()
    print("\nDone. Review output above then run with --write-md to generate docs/error-analysis.md")
