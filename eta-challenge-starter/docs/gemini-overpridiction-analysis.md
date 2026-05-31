The code confirms exactly what the Bias Summary suggested: Your model is
suffering from a "Feature Scale Mismatch."

While you are log-transforming your target (duration_seconds), you are providing
your most influential features (zone_pair_mean and zone_pair_hour_mean) as raw
seconds.

Why this causes Over-prediction on Short Trips:

In a regression model, if the target is \log(y) and a feature is x, the model
has to learn that x has an exponential relationship to the target.

1.  Your zone_pair_mean values likely range from 300s to 3000s.
2.  Your target log(duration) only ranges from about 4 to 8.
3.  Because the "Mean" features are so large and numerically dominant, the
    XGBoost trees split on them early. For a short trip (e.g., 120s) where the
    zone_pair_mean is 600s, the model finds it "mathematically expensive" to
    pull the prediction all the way down from the 600s-anchor to
    the 120s-target.

Recommended Changes

1. Log-Transform your "Mean" Features

You must put your mean features on the same scale as your target. This is the
single biggest reason the model is "anchoring" too high.

In baseline.py and predict.py:

# Change this:
"zone_pair_mean": pair_mean,
"zone_pair_hour_mean": pair_hour_mean,

# To this:
"zone_pair_mean": np.log1p(pair_mean),
"zone_pair_hour_mean": np.log1p(pair_hour_mean),

2. Add an is_intra_zone Feature

Short trips are often trips that stay within the same zone. Currently, your
model only knows the zones by their IDs (which XGBoost treats as numbers, not
categories—see below).

In engineer_features:

is_intra_zone = (df["pickup_zone"].astype(int) == df["dropoff_zone"].astype(int)).astype("int8")
# Add this to your return DataFrame

3. Enable Categorical Support in XGBoost

Currently, you are passing pickup_zone (e.g., 138) as an integer. XGBoost treats
this as a continuous value, meaning it thinks Zone 139 is "closer" to Zone 138
than Zone 1 is. This is wrong for NYC geography.

In baseline.py:

# 1. Update FEATURES list to include new ones
# 2. In engineer_features, cast zones to 'category'
df["pickup_zone"] = df["pickup_zone"].astype("category")
df["dropoff_zone"] = df["dropoff_zone"].astype("category")

# 3. Update XGBRegressor
model = xgb.XGBRegressor(
    enable_categorical=True, # Critical!
    # ... other params
)

4. Change the Objective Function (Poisson)

For duration data, reg:squarederror on log-data is okay, but count:poisson or
reg:tweedie is often better at handling the "long tail" of travel times without
over-predicting the short ones.

In baseline.py:

model = xgb.XGBRegressor(
    objective="count:poisson", # Great for durations
    # ...
)

5. Distance non-linearity

A 1km trip in midtown is not the same as a 1km trip in Staten Island. Add
log(haversine_km):

"log_haversine_km": np.log1p(hav),

Why the Airport Over-predictions (138→1) are happening

Your summary says 138→1 (LGA to EWR) is 100% over-predicted.

1.  These trips are long-distance but often fast (highways).
2.  Because you use zone_pair_mean, and that mean is likely influenced by heavy
    traffic samples, the model assumes all LGA→EWR trips are slow.
3.  The Fix: Ensure the model has an Average Speed feature for the zone pair.
      - pair_speed = haversine_km / (zone_pair_mean / 3600)
      - This tells the model: "This zone pair is typically a high-speed route,"
        allowing it to adjust the duration downward even if the distance is
        high.

Summary Checklist for your next run:

1.  Log-transform zone_pair_mean and zone_pair_hour_mean.
2.  Log-transform haversine_km.
3.  Add is_intra_zone (Boolean).
4.  Set enable_categorical=True in XGBoost and pass Zone IDs as category dtypes.
5.  Check your expm1 back-transformation. If you are still over-predicting, you
    can try a small Bias Multiplier (e.g., return 0.95 * float(math.expm1(...)))
    to specifically counter the net +49s bias.
