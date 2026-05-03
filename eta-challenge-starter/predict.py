"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"
_ZONE_PAIR_MEANS_PATH = Path(__file__).parent / "zone_pair_means.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
if hasattr(_MODEL, "get_booster"):
    _MODEL.get_booster().feature_names = None

with open(_ZONE_PAIR_MEANS_PATH, "rb") as _f:
    _zp = pickle.load(_f)
    _ZONE_PAIR_MEANS: dict = _zp["means"]
    _GLOBAL_MEAN: float = _zp["global_mean"]

# Feature order must match baseline.py:
#   pickup_zone, dropoff_zone, hour, dow, month, passenger_count, zone_pair_mean


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
    zone_pair_mean = _ZONE_PAIR_MEANS.get((pu, do), _GLOBAL_MEAN)
    x = np.array(
        [[
            pu,
            do,
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
            zone_pair_mean,
        ]],
        dtype=np.float32,
    )
    return float(_MODEL.predict(x)[0])
