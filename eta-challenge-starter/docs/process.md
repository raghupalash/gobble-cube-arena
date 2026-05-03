I'm thinking of doing this workflow for this problem:

1. Research on how to train ML model for this use case
2. Train the model and measure output (MAE, inference latency, container size) against the baseline
3. Research and Iterate the model untill baseline satisfied.


XGBoost it is, because it's a default choice. Lets use it to create an MVP.

---

## Scoreboard

`grade.py` = fixed 50k sample (mirrors eval). `full dev` = all 1.2M dev rows.
All runs below use 1M sample unless noted in the Full Training table.

| Experiment | grade.py MAE | Δ grade.py | full dev MAE | Δ full dev |
|---|---|---|---|---|
| Baseline (6 features) | 357.2s | — | — | — |
| Exp 1: + zone_pair_mean | 305.9s | -51.3s | — | — |
| Exp 2: - passenger_count | 304.4s | -1.5s | — | — |
| Exp 3: + haversine_km | 303.4s | -1.0s | — | — |
| Exp 4: zone_pair_mean → median | 302.7s | -0.7s | 300.9s | — |
| Exp 5: zone_pair_mean → trimmed mean (10%) | 302.3s | -0.4s | 300.4s | -0.5s |
| Exp 6a: + zone_pair_hour_mean, trimmed mean 10% | 306.3s | +4.0s | 305.2s | +4.8s |
| Exp 6b: zone_pair_hour_mean → trimmed mean 5% | 306.6s | +0.3s | 305.3s | +0.1s |
| Exp 6c: zone_pair_hour_mean → mean | 308.2s | +1.6s | 306.9s | +1.6s |
| Exp 6d: zone_pair_hour_mean → median | 305.8s | -2.4s | 304.6s | -2.3s |

## Full Training Runs (37M rows)

*See docs/archive.md for previous unreliable runs (parallel execution caused model.pkl conflicts).*

| Features at time of run | grade.py MAE | full dev MAE | Training time |
|---|---|---|---|
| — | — | — | — |

---

## Experiment Log

### Experiment 1 — Zone-Pair Mean as a Feature

**Hypothesis:** The baseline GBT (Dev MAE ~357s) treats pickup_zone and dropoff_zone as independent integers, so it never cleanly learns the (pickup, dropoff) pair interaction. A 10-line zone-pair average lookup beats it at ~300s. Adding the precomputed pair mean as an explicit feature hands that signal directly to the model, letting it learn corrections on top (e.g. rush-hour deltas) rather than rediscovering the interaction through splits.

**Approach:**
- At training time: compute mean `duration_seconds` per `(pickup_zone, dropoff_zone)` pair from training data. Save as `zone_pair_means.pkl` alongside `model.pkl`.
- New feature: `zone_pair_mean` (fallback = global mean for unseen pairs).
- At inference time: load the lookup, compute `zone_pair_mean` for the incoming request, pass as 7th feature to XGBoost.

**Result:** Dev MAE 305.9s (grade.py 50k sample). Down from 357.2s baseline — 51s improvement. Now on par with the 10-line zone-pair lookup (~300s), with XGBoost on top adding time-of-day corrections.

### Experiment 2 — Drop passenger_count

**Hypothesis:** Passenger count doesn't affect road speed or route. It's likely noise or at best a weak proxy for trip type (e.g. large groups → airport). Removing it should have neutral-to-positive effect.

**Approach:** Remove `passenger_count` from the feature list entirely.

**Result:** Dev MAE 305.9s → 304.4s. Marginal improvement — confirmed as noise.

**Note:** Could revisit `passenger_count` as a proxy for trip type (airport/hotel runs tend to have more passengers and longer durations). Worth trying again if we add explicit airport zone flags.

### Experiment 3 — Haversine distance between zone centroids (planned)

**Hypothesis:** The model currently has no concept of physical distance between zones. Straight-line distance between zone centroids is the strongest signal we're missing — closer zones = shorter trips.

**Approach:** Derive zone centroid lat/lon from the official NYC TLC shapefile (authoritative source, not a third-party CSV). Add haversine distance as a feature. Fold centroid extraction into `download_data.py`.

**Result:** Dev MAE 304.4s → 303.4s. Marginal gain — `zone_pair_mean` already captures most of the distance signal implicitly. Haversine adds value for unseen zone pairs (fallback path) and gives the model an explicit continuous distance signal.

### Experiment 4 — zone_pair_mean → median

**Hypothesis:** Mean is pulled by outliers (e.g. stuck-in-traffic trips). Trip durations are right-skewed so median should be more robust and representative.

**Approach:** Replace `.mean()` with `.median()` in `build_zone_pair_means()`. One-line change.

**Result:** Dev MAE 303.4s → 302.7s. Small improvement confirming right-skew effect. Training time increased 7s → 26s (median is slower to compute). Worth keeping.

### Experiment 5 — zone_pair_mean → trimmed mean (10%)

**Hypothesis:** Median discards too much information — it only looks at the middle value. Trimmed mean (drop bottom/top 10% per pair, average the rest) keeps more of the distribution while still ignoring extreme outliers.

**Approach:** Replace median with a custom `trimmed_mean` function using 10th/90th percentile cutoffs per group.

**Result:** grade.py MAE 302.7s → 302.3s, full dev 300.9s → 300.4s. Consistent improvement over median. Trimmed mean beats both mean and median for this distribution.

### Experiment 6 — Zone-Pair × Hour mean (branch: exp/zone-pair-hour-mean)

**Hypothesis:** zone_pair_mean is static across all hours. A (pickup, dropoff, hour) bucket captures time-of-day variation for the same route.

**Approach:** Add `zone_pair_hour_mean` as an 8th feature. Trimmed mean per (pair, hour) bucket with fallback to zone_pair_mean.

**Result:** grade.py 302.3s → 306.3s (+4.0s), full dev 300.4s → 305.2s (+4.8s). **Worse.** Root cause: 108k buckets on 1M rows = ~9 trips/bucket on average. Too sparse for reliable trimmed mean estimates. No minimum count guard — noisy buckets hurt more than they help. Branch not merged.

**Next step if revisiting:** add minimum count threshold (e.g. skip bucket if < 30 trips, fall back to zone_pair_mean). Will be more effective on full 37M data (~34 trips/bucket average).

