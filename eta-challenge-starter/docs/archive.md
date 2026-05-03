# Archive — Unreliable Production Scores

These full training results are archived because parallel runs caused model.pkl
to be overwritten mid-experiment, making scores untrustworthy.

| Run | Features | grade.py MAE | full dev MAE | Why unreliable |
|---|---|---|---|---|
| Full run 1 (37M) | Exp 1-3 (zone_pair_mean mean, no passenger_count, haversine_km) | 292.9s | 291.4s | model.pkl later overwritten by 1M zone_pair_median run |
| Full run 2 (37M) | Exp 1-5 (trimmed mean, haversine_km) | not scored | 291.2s | grade.py ran against 8-feature predict.py but 7-feature model — mismatch |

**Root cause:** full training runs and dev experiments shared the same model.pkl.
Fixed going forward with experiment branching — full runs only happen on master
after an experiment is validated on a branch.
