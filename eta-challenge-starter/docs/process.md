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
| Exp 6: + zone_pair_hour_mean (branch: exp/zone-pair-hour-mean) | 306.3s | +4.0s | 305.2s | +4.8s |
| Exp 7: + zone_pair_dow_mean (branch: exp/zone-pair-dow-mean) | 304.7s | +2.4s | 303.0s | +2.6s |
| Exp 8: + is_rush_hour, is_weekend, is_airport flags (branch: exp/flags) | 302.8s | +0.5s | 300.9s | +0.5s |
| Exp 8 (full 37M): flags + 800/6 on full dataset | 293.4s | -8.9s | 292.2s | -8.2s |
| Exp 9a: n_estimators=800, max_depth=6 (branch: exp/hyperparams) | 303.1s | +0.8s | 301.2s | +0.8s |
| Exp 9b: n_estimators=800, max_depth=8 (branch: exp/hyperparams) | 303.3s | +1.0s | 301.4s | +1.0s |

## Full Training Runs (37M rows)

*See docs/archive.md for previous unreliable runs (parallel execution caused model.pkl conflicts).*

| Features at time of run | grade.py MAE | full dev MAE | Training time |
|---|---|---|---|
| flags + 800/6 (exp/flags branch) | 293.4s | 292.2s | ~34 min |

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

### Experiment 7 — Zone-Pair × DOW mean (branch: exp/zone-pair-dow-mean)

**Hypothesis:** Reducing from 24 hour-buckets to 7 day-of-week buckets reduces sparsity (55k buckets vs 108k on 1M rows). DOW captures weekend vs weekday variation per route without the extreme sparsity that killed exp6.

**Approach:** Replace `zone_pair_hour_mean` with `zone_pair_dow_mean`: trimmed mean (5%) per (pickup, dropoff, dow) with fallback to `zone_pair_mean`. No minimum count guard.

**Result:** grade.py 302.3s → 304.7s (+2.4s), full dev 300.4s → 303.0s (+2.6s). **Worse.** Even 7 buckets is too sparse on 1M sample. DOW signal already captured by raw `dow` feature. Branch not merged.

### Experiment 8 — Binary flags: rush hour, weekend, airport (branch: exp/flags)

**Hypothesis:** Explicit binary flags for structurally different trip types (rush hour, weekend, airport) give XGBoost cleaner splits than inferring these from raw hour/dow integers.

**Approach:** Add three features: `is_rush_hour` (7–9am or 4–7pm on weekdays), `is_weekend` (dow ≥ 5), `is_airport` (pickup or dropoff in EWR=1, JFK=132, LGA=138).

**Result:** grade.py 302.3s → 302.8s (+0.5s), full dev 300.4s → 300.9s (+0.5s). **Worse.** XGBoost already learns these patterns via splits on `hour` and `dow`. Explicit flags add no new information. Branch not merged.

### Experiment 9 — Hyperparameter tuning: n_estimators, max_depth (branch: exp/hyperparams)

**Hypothesis:** Current 400/8 may be leaving gains on the table. More trees (800) and/or shallower depth (6) could improve generalization.

**Approach:** Tested two variants on clean master features (no flags): 800/6 and 800/8.

**Result:** Both worse than master. 800/6: +0.8s grade.py. 800/8: +1.0s grade.py. 400 trees at depth 8 is near-optimal for the 1M sample — extra trees memorize noise. Hyperparameter gains likely only realizable on full 37M. Branch not merged.

**Hypothesis:** Reducing from 24 hour-buckets to 7 day-of-week buckets reduces sparsity (55k buckets vs 108k on 1M rows). DOW captures weekend vs weekday variation per route without the extreme sparsity that killed exp6.

**Approach:** Replace `zone_pair_hour_mean` with `zone_pair_dow_mean`: trimmed mean (5%) per (pickup, dropoff, dow) with fallback to `zone_pair_mean`. No minimum count guard.

**Result:** grade.py 302.3s → 304.7s (+2.4s), full dev 300.4s → 303.0s (+2.6s). **Worse.** Even 7 buckets is too sparse on 1M sample (~18 trips/bucket avg for active pairs). DOW signal is likely already captured by the raw `dow` feature. Branch not merged.

