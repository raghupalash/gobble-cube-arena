# Under-Prediction Research — ETA Challenge

Model: 1M sample, XGBoost 800/6, log1p target, zone_pair_hour_mean (exp/zone-pair-hour-guarded)
Dev MAE: 280.8s | Date: 2026-05-05

---

## Summary

37.4% of dev trips are under-predicted. The problem is almost entirely concentrated in long trips and is driven by two compounding root causes: zone_pair_mean anchoring and structurally high-variance routes.

---

## Key Numbers

- Under-predicted trips: 460,913 (37.4% of dev)
- Mean under-prediction magnitude: -308.7s
- Median under-prediction magnitude: -170.3s

### By trip duration

| Bucket | % of bucket under-predicted | Mean miss |
|---|---|---|
| <5min | 7% | -35s |
| 5-10min | 29% | -88s |
| 10-20min | 41% | -191s |
| 20-30min | 50% | -351s |
| 30-60min | 55% | -624s |
| 1-2hr | 70% | -1341s |
| >2hr | 100% | -5849s |

Under-prediction is monotonically worse as trips get longer. >2hr trips are 100% under-predicted.

### By hour

Under-prediction **rate** is flat (~34-48% across all hours). The problem is **magnitude**:
- Worst magnitude: 14h–16h (afternoon peak, -344 to -367s mean miss)
- Best magnitude: 1-3am (-193 to -199s)
- 7am has 2.7x lift in worst errors but similar under-prediction rate to other hours

### By day of week

Weekdays worse than weekends. Wednesday and Thursday are worst (-332 to -331s mean miss). Sunday is best (-255s).

---

## Root Cause Diagnosis

### H2: zone_pair_mean anchoring (confirmed — primary cause)

For long trips (>30min) that are under-predicted:
- True duration is on average **+1081s above zone_pair_mean**
- Model only moves **+310s above zone_pair_mean** (28.7% of the gap)

By bucket:
| Bucket | True above zpm | Model moves above zpm | Gap closed |
|---|---|---|---|
| 30-60min | +843s | +219s | 26% |
| 1-2hr | +2074s | +733s | 35% |
| >2hr | +6437s | +588s | 9% |

**Interpretation:** zone_pair_mean acts as a ceiling. The model can deviate upward a bit (via hour, dow, flags), but it cannot reach the true duration of a genuinely long trip. For >2hr trips, it closes only 9% of the gap.

### H3: Structurally high-variance routes (confirmed — amplifies H2)

Worst under-predicted zone pairs have mean CV of **0.62** vs 0.38 for typical pairs.

Worst offenders:
| Pair | CV | Mean duration | Note |
|---|---|---|---|
| 132→132 (JFK→JFK) | 1.87 | 576s | Airport loops, huge wait time variance |
| 230→230 | 0.91 | 459s | Same-zone, unpredictable |
| 264→264 | 0.81 | 985s | Unknown zone, no signal |
| 161→161 | 0.79 | 411s | Same-zone loops |

High CV means zone_pair_mean (the trimmed mean) is a poor anchor for this route — the distribution has a very long tail that the average doesn't represent.

### H1: Leaf averaging (mechanism behind H2 and H3)

XGBoost leaf values are means of training samples. When a leaf contains mostly normal trips and some rare very-long trips, it predicts near the average — systematically under-shooting the tail. This is not fixable without either a distinguishing feature (real-time traffic) or a different loss objective (quantile regression).

### H4: Long trip representation in train vs dev (negligible)

>2hr trips are 1.5x more common in dev than train (0.03% vs 0.02%), but absolute numbers are tiny (379 dev, 235 train). Distribution shift is not the driver.

---

## What Can Actually Help

| Approach | Expected gain | Feasibility |
|---|---|---|
| Real-time traffic features | Large (could close most of the gap) | Requires external data source |
| Quantile regression (predict 55-60th pct) | Medium — trades over-prediction on short trips for less under-prediction on long | Easy to implement, needs tuning |
| Per-pair variance feature (CV as a feature) | Small — tells model "this route is high variance, adjust up" | Easy |
| Separate high-variance route model | Medium — specialized model for CV > 0.5 routes | Moderate complexity |

---

## Raw Output Files

- `under_prediction_research.txt` — full under-prediction breakdown by duration, hour, DOW, zone pairs
- `underprediction_diagnosis.txt` — H2/H3/H4 diagnostic output
