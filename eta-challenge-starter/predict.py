"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import math

import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"
_ZONE_CENTROIDS_PATH = Path(__file__).parent / "data" / "zone_centroids.csv"

_AIRPORT_ZONES = {1, 132, 138}  # EWR, JFK, LGA

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

with open(_ZONE_PAIR_MEANS_PATH, "rb") as _f:
    _zp = pickle.load(_f)
    _ZONE_PAIR_MEANS: dict = _zp["means"]
    _GLOBAL_MEAN: float = _zp["global_mean"]

_df = pd.read_csv(_ZONE_CENTROIDS_PATH)
_CENTROIDS: dict = {row.zone_id: (row.latitude, row.longitude) for row in _df.itertuples()}

_R = 6371.0

# Feature order must match baseline.py:
#   pickup_zone, dropoff_zone, hour, dow, month, zone_pair_mean, haversine_km, is_rush_hour, is_weekend, is_airport


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _R * math.asin(math.sqrt(a))


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    pu, do = int(request["pickup_zone"]), int(request["dropoff_zone"])
    h, dow = ts.hour, ts.weekday()
    zone_pair_mean = _ZONE_PAIR_MEANS.get((pu, do), _GLOBAL_MEAN)
    pu_cent, do_cent = _CENTROIDS.get(pu), _CENTROIDS.get(do)
    haversine_km = _haversine(*pu_cent, *do_cent) if pu_cent and do_cent else 0.0
    is_rush_hour = int((7 <= h < 9 and dow < 5) or (16 <= h < 19 and dow < 5))
    is_weekend = int(dow >= 5)
    is_airport = int(pu in _AIRPORT_ZONES or do in _AIRPORT_ZONES)
    x = np.array(
        [[pu, do, h, dow, ts.month, zone_pair_mean, haversine_km, is_rush_hour, is_weekend, is_airport]],
        dtype=np.float32,
    )
    return float(_MODEL.predict(x)[0])
