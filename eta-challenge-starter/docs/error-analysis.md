# Error Analysis — Dev Set Residuals

**Model:** master branch (1M sample, XGBoost, 7 features)
**Dev set:** 1,230,911 rows
**Overall dev MAE:** 291.2s | **Overall bias:** +118.1s (model over-predicts on average)

---

## TL;DR

The model has two distinct failure modes that point in opposite directions:
- On average it **over-predicts** (bias +118s) — likely pulling short trips up toward the mean
- On very long trips it **massively under-predicts** — XGBoost can't reach the tail of the distribution

96% of the worst errors are very long trips (>60min). Fix that first.

---

## Top Error Clusters

### 1. Very Long Trips (>60 min) — 96.8% of worst MAE
| Metric | Value |
|---|---|
| Share of worst 1000 | 96.0% |
| Share of all dev | 1.8% |
| Overrepresentation ratio | **53.7x** |
| Mean bias (signed error) | **-4553s** (under-predicts by ~75 min) |

The model can't reach the tail. XGBoost predictions are bounded by training leaf values, and very long trips are rare enough that the model regresses toward the pair mean instead.

**Hypothesis:** Log-transform the target (`log1p(duration_seconds)`) during training, exponentiate at inference. This compresses the tail, lets the model learn the full distribution, and should cut this cluster's error significantly.

---

### 2. Unknown/Out-of-Borough Zones (264, 265) — ~18x overrepresented as dropoff
| Zone | Role | worst_% | overall_% | Ratio |
|---|---|---|---|---|
| 265 | dropoff | 8.4% | 0.4% | **18.8x** |
| 264 | pickup | 3.3% | 0.3% | **9.5x** |
| 264 | dropoff | 4.1% | 0.5% | **8.5x** |

Zones 264/265 are special TLC codes (likely NaN/out-of-borough trips). These have no meaningful zone_pair_mean and the model falls back to global_mean — a terrible predictor for what are likely very long out-of-borough trips.

**Hypothesis:** Add an `is_unknown_zone` flag for zones 264/265. These trips may warrant their own fallback (e.g. median of all long trips) rather than global_mean.

---

### 3. JFK Airport (Zone 132) — 2.8x overrepresented as pickup, 4x as dropoff
| Role | worst_% | overall_% | Ratio | Bias |
|---|---|---|---|---|
| Pickup | 14.8% | 5.2% | 2.8x | -3873s |
| Dropoff | 5.2% | 1.3% | 4.0x | -4559s |

Airport trips are structurally longer and more variable (traffic, terminal routing, waiting). The zone_pair_mean partially captures this but not the time-of-day variation — a JFK pickup at 5pm is very different from 2am.

**Hypothesis:** Add airport zone flag (pickup or dropoff in {1, 132, 138}) + interaction with hour_bucket. Or try zone_pair_hour_mean with min-count guard specifically for airport pairs, which have enough volume.

---

### 4. Morning Rush Hour — 1.5x overrepresented
| Metric | Value |
|---|---|
| Share of worst 1000 | 13.9% |
| Share of all dev | 9.5% |
| Overrepresentation ratio | 1.5x |
| Bias | -4153s |

Weaker signal than the others (mostly driven by long trips during rush hour), but morning rush is the one time bucket that's consistently overrepresented. Current `hour` feature is treated as a categorical integer — the model may not be capturing rush-hour nonlinearity well.

**Hypothesis:** Binary `is_rush_hour` flag (7–9am and 4–7pm on weekdays). Low cost to add, may help at the margin.

---

### 5. Bad Labels (Trips <60s) — filter from training
| Metric | Value |
|---|---|
| Count in dev | 3,442 (0.28%) |
| In worst 1000 | 0 |
| Actual range | 30s – 59s |

These aren't driving the worst errors (model isn't predicting 30s trips) but they're corrupting the zone_pair_mean lookup and training signal. A 45-second trip from JFK to Midtown is almost certainly a bad record.

**Hypothesis:** Filter trips <60s from training data before computing zone_pair_means and fitting the model.

---

## Methodological Trade-offs

| Trade-off | Decision | Rationale |
|---|---|---|
| Worst-N cutoff | 1000 | Arbitrary — clusters would look similar at 500 or 2000. Sufficient for identifying systematic failure modes; not suitable for measuring their exact magnitude. |

---

## Recommended Experiment Order

1. **Log-transform target** — fixes cluster 1 (96% of worst MAE). Highest expected impact.
2. **Filter bad labels <60s** — clean training signal, minimal risk.
3. **Unknown zone flag (264/265)** — targeted fix, easy to add.
4. **Airport flag + zone_pair_hour_mean** — already in backlog, now confirmed by data.
5. **Rush hour flag** — last, weakest signal.
