# Safety Gym Supplementary Results

This supplement reports experiments on three Safety Gym continuous-control environments. The table is generated directly from `results/safety_gym_results.csv`.

| Environment | Raw Return | Raw Cost | REP-PD-NN Return | REP-PD-NN Cost |
| --- | ---: | ---: | ---: | ---: |
| PointGoal1 | -1.38 +/- 0.58 | 27.85 +/- 18.99 | **-0.14 +/- 0.03** | **16.45 +/- 12.69** |
| PointPush1 | -1.40 +/- 0.60 | 32.17 +/- 14.66 | **-0.17 +/- 0.12** | **12.13 +/- 6.95** |
| PointPush2 | -1.40 +/- 0.71 | 13.00 +/- 8.17 | **-0.03 +/- 0.01** | **10.90 +/- 3.85** |

All rows use 1M interaction steps, 3 random seeds, and the final deterministic evaluation checkpoint.

REP-PD-NN is the practical neural-network variant of REP-PD-TEG. It replaces the theoretical per-round full-batch MLE representation step with a mini-batch SPEDER-style stochastic representation objective [1], uses the learned state-action representation in neural reward and cost critics, and trains the policy with a Lagrangian-style constrained policy update.

[1] T. Ren et al., Spectral Decomposition Representation for Reinforcement Learning.
