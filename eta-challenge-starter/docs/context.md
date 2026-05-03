# Session Context — ETA Challenge

Use this file to resume work across chats. Update after each session.

---

## Problem

Predict NYC taxi trip duration in seconds. Input: pickup_zone, dropoff_zone, requested_at, passenger_count. Scored on MAE (lower = better). Baseline: ~367s eval MAE.

---

## Current State (as of last session)

**Master branch:** Exp 1–5 code, 7 features, full run model.pkl
**Exp worktree:** `../gobblecube-arena-exp-zone-pair-hour/` on branch `exp/zone-pair-hour-mean`

**Current features (master):**
- pickup_zone, dropoff_zone, hour, dow, month
- zone_pair_mean — trimmed mean (10%) per (pickup, dropoff) pair
- haversine_km — straight-line distance between zone centroids

**Scoreboard (1M sample):**

| Experiment | grade.py | Δ grade.py | full dev | Δ full dev |
|---|---|---|---|---|
| Baseline (6 features) | 357.2s | — | — | — |
| Exp 1: + zone_pair_mean | 305.9s | -51.3s | — | — |
| Exp 2: - passenger_count | 304.4s | -1.5s | — | — |
| Exp 3: + haversine_km | 303.4s | -1.0s | — | — |
| Exp 4: zone_pair_mean → median | 302.7s | -0.7s | 300.9s | — |
| Exp 5: zone_pair_mean → trimmed mean 10% | 302.3s | -0.4s | 300.4s | -0.5s |
| Exp 6 (branch): zone_pair_hour_mean variants | 305.8s–308.2s | worse | — | — |

**Full training runs:** Unreliable so far due to parallel run conflicts (see docs/archive.md). Clean prod run pending.

---

## Key Learnings

- zone_pair_mean is the biggest lever (-51s) — gives XGBoost the pair interaction directly
- Trimmed mean > median > mean for zone_pair_mean (right-skewed distribution)
- haversine mostly redundant — zone_pair_mean already encodes distance
- zone_pair_hour_mean fails on 1M sample (sparsity: ~9 trips/bucket) — needs full 37M + min count guard
- Full training gives ~9s improvement over 1M sample

---

## Open Experiments (try next)

1. zone_pair_hour_mean with minimum count guard (e.g. skip bucket if < 30 trips, fall back to zone_pair_mean) — promising on full 37M
2. Rush hour / weekend binary flags
3. Airport zone flags (JFK=132, LGA=138, EWR=1)
4. Clean full training run on master after next merge

---

## Infra Notes

- `python baseline.py` — 1M sample, ~30s
- `python baseline.py --full` — 37M rows, ~18 min, run in background
- Full run logs: `/tmp/full_train_N.log` (increment N each time)
- Worktree: `../gobblecube-arena-exp-zone-pair-hour/`
- Data symlinked in worktree from master data dir
