# ETA Challenge Submission

## Problem Statement & Objectives

This project tackles the Gobblecube ETA challenge, broken down into three core components:

1. **Model Building:** Train an ML model on historical NYC Yellow Taxi trip records. Given a ride request (`pickup_zone`, `dropoff_zone`, `requested_at`, `passenger_count`), predict the actual trip duration (ETA) in seconds.
2. **Packaging & Constraints:** Containerize the solution using Docker, strictly adhering to constraints:
    - Inference latency ≤ 200 ms per request on CPU.
    - Total Docker image size ≤ 2.5 GB.
    - No external API calls at inference time.
3. **Documentation & Iteration:** Pair effectively with AI tooling (recorded via Git history) and document the approach, iterations, and reasoning.
