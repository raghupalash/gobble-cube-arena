# Ideas to Try

## Feature Engineering

- **zone_pair_hour_mean with minimum count guard** — skip bucket if < 30 trips, fall back to zone_pair_mean. Sparsity was the root cause of Exp 6 failure. Likely works on full 37M (~34 trips/bucket avg).
- **Rush hour flag** — binary: is it rush hour (7-9am, 4-7pm on weekdays)? Captures the key time-of-day variation more directly than raw hour.
- **Weekend flag** — binary: is it Saturday or Sunday? dow gives day of week but no explicit weekend signal.
- **Airport zone flag** — binary: is pickup or dropoff an airport zone? JFK=132, LGA=138, EWR=1. Airport trips have structurally different duration patterns.
- **zone_pair_dow_mean** — trimmed mean per (pickup, dropoff, day-of-week). Less sparse than hour (7 buckets vs 24). Captures weekday vs weekend variation per route.
- **passenger_count revisit** — may be a weak proxy for airport/hotel trips (larger groups). Worth retrying if airport zone flags are added.

## Data

- **NOAA weather join** — hourly weather (rain, snow, temperature) for JFK/LGA/Central Park. Rain/snow significantly increases trip duration. Join on timestamp.
- **Full 37M clean prod run** — run after next experiment merges to master with properly matched code and model.

## Model

- **zone_pair_mean with dow interaction** — (pickup, dropoff, dow) trimmed mean. Only 7 buckets per pair vs 24 for hour — much less sparse, may work on 1M sample.
- **LightGBM** — often faster and slightly better than XGBoost on tabular data. Drop-in replacement worth benchmarking.
- **Increase n_estimators** — current is 400. More trees may help now that features are richer.
