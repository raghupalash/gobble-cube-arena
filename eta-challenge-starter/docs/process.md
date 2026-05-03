I'm thinking of doing this workflow for this problem:

1. Research on how to train ML model for this use case
2. Train the model and measure output (MAE, inference latency, container size) against the baseline
3. Research and Iterate the model untill baseline satisfied.


XGBoost it is, because it's a default choice. Lets use it to create an MVP.

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

