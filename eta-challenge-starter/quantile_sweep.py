#!/usr/bin/env python3
"""Sweep quantile_alpha values and compare grade.py MAE."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"

def main():
    from baseline import build_zone_pair_means, engineer_features, load_zone_centroids

    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "sample_1M.parquet")
    dev   = pd.read_parquet(DATA_DIR / "dev.parquet")

    print("Building zone-pair means...")
    zone_pair_means = build_zone_pair_means(train)
    global_mean = float(train["duration_seconds"].mean())
    centroids = load_zone_centroids()

    X_train = engineer_features(train, zone_pair_means, global_mean, centroids)
    X_dev   = engineer_features(dev,   zone_pair_means, global_mean, centroids)
    y_train_raw = train["duration_seconds"].to_numpy()
    y_dev       = dev["duration_seconds"].to_numpy()

    alphas = [0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"\n{'Alpha':>6}  {'Dev MAE':>10}  {'Bias (P50 resid)':>18}  {'Time':>6}")
    print("-" * 50)

    results = []
    for alpha in alphas:
        if alpha == 0.50:
            # log1p baseline (current approach)
            y_train = np.log1p(y_train_raw)
            objective = "reg:squarederror"
        else:
            y_train = np.log1p(y_train_raw)
            objective = "reg:quantileerror"

        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            objective=objective,
            **({"quantile_alpha": alpha} if objective == "reg:quantileerror" else {}),
        )
        t0 = time.time()
        model.fit(X_train, y_train, verbose=False)
        elapsed = time.time() - t0

        preds = np.expm1(model.predict(X_dev))
        residuals = preds - y_dev
        mae = float(np.abs(residuals).mean())
        bias = float(np.median(residuals))
        results.append((alpha, mae, bias))

        label = f"log1p+squarederror (baseline)" if alpha == 0.50 else f"quantile α={alpha}"
        print(f"  α={alpha}  MAE={mae:8.1f}s  median_resid={bias:+.1f}s  [{elapsed:.0f}s]  {label}")

    print("\n--- Summary ---")
    best_alpha, best_mae, _ = min(results[1:], key=lambda x: x[1])
    baseline_mae = results[0][1]
    print(f"Baseline (squarederror): {baseline_mae:.1f}s")
    print(f"Best quantile:  α={best_alpha}  MAE={best_mae:.1f}s  (Δ={best_mae - baseline_mae:+.1f}s)")

if __name__ == "__main__":
    main()
