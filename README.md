# REP-PD Experimental Reproduction Package

This repository provides the code and results for the Safety Gym experiments in [*Primal-Dual Representation Learning for Low-Rank Constrained MDPs*](https://arxiv.org/abs/2610.30756). It includes the practical neural implementation REP-PD-NN and its raw-state baseline; the paper's stochastic-oracle algorithm is analyzed separately.

## Contents

| Path | Purpose |
| --- | --- |
| `cmdp/safe_sac_lag_speder.py` | Constrained actor-critic training and evaluation for Safety Gym |
| `cmdp/representations.py` | SPEDER-style stochastic representation learner |
| `scripts/run_safety_gym_experiments.py` | Prepare 18 training commands; dry run by default |
| `scripts/aggregate_safety_gym_runs.py` | Recalculate means and standard deviations from new run logs |
| `scripts/reproduce_safety_gym_results.py` | Render a result CSV as Markdown/LaTeX |
| `results/safety_gym_results.csv` | Reported aggregate results, preserved as a snapshot |

## Safety Gym Setup

The reported tasks are `Safexp-PointGoal1-v0`, `Safexp-PointPush1-v0`, and `Safexp-PointPush2-v0`. Each compares the raw-state baseline with `REP-PD-NN` at seeds 0, 1, and 2, for one million environment interactions per run. Both variants use the same entropy-regularized actor-critic and Lagrangian cost update. `REP-PD-NN` adds a mini-batch SPEDER-style representation learner, inspired by Ren et al.'s *Spectral Decomposition Representation for Reinforcement Learning*, and feeds its state-action features to the reward and cost critics; the raw baseline feeds the critics raw state-action pairs. Task-specific representation settings are in `scripts/run_safety_gym_experiments.py`, and shared defaults are in `TrainConfig` in the training script.

The reported metrics are return and cost at the final deterministic evaluation checkpoint, averaged across three seeds. Lower cost is better; a lower mean cost does not imply that every seed satisfies the cost budget.

Use a dedicated Python environment with PyTorch and the **legacy** Safety Gym `Safexp-*` tasks. Install the requirements and a MuJoCo runtime compatible with `mujoco-py`:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists package names, not a tested lockfile; reproduction may require resolving legacy Gym/MuJoCo compatibility for the local OS. Safety-Gymnasium tasks are not interchangeable with the reported `Safexp-*` environments. Each full run was performed on one RTX 4090 GPU (24 GB). The tests and result-rendering commands below need neither Safety Gym nor a GPU.

## Reproduction

From the repository root, run the fast checks and inspect the training plan:

```bash
python -m unittest discover -s tests
python scripts/run_safety_gym_experiments.py --device cuda
```

The second command is a dry run: it prints 18 commands and writes `results/safety_gym_experiment_manifest.json`; it does not train. To run all jobs sequentially:

```bash
python scripts/run_safety_gym_experiments.py --run --device cuda
```

Jobs can instead be distributed across GPUs using the manifest. Do not launch duplicate jobs into the same `--log-dir`. Training logs are written under `outputs/safety_gym_runs/` and ignored by Git. After all 18 runs finish, aggregate their final evaluations and render the newly computed table:

```bash
python scripts/aggregate_safety_gym_runs.py
python scripts/reproduce_safety_gym_results.py --input results/safety_gym_results_reproduced.csv --tex docs/reproduced_results.tex --markdown docs/reproduced_results.md
```

Aggregation requires all 18 logs and fails rather than silently reporting incomplete results. It uses the last finite deterministic evaluation in each log and the **sample** standard deviation across seeds. New runs record an evaluation on the final interaction step; older logs may contain their last evaluation within one evaluation interval of the final step. The reproduced CSV is written separately so the reported snapshot is not overwritten.

To regenerate the manuscript's archived table *from the supplied aggregate CSV*, without running training:

```bash
python scripts/reproduce_safety_gym_results.py
```

This formats the supplied aggregate CSV without rerunning training. The included `docs/anonymous_safe_rl_results.pdf` is a formatted table artifact; a PDF compiler is needed only to regenerate it with `--compile-pdf`.

## Reported Results

| Environment | Raw return | Raw cost | REP-PD-NN return | REP-PD-NN cost |
| --- | ---: | ---: | ---: | ---: |
| PointGoal1 | -1.38 +/- 0.58 | 27.85 +/- 18.99 | -0.14 +/- 0.03 | 16.45 +/- 12.69 |
| PointPush1 | -1.40 +/- 0.60 | 32.17 +/- 14.66 | -0.17 +/- 0.12 | 12.13 +/- 6.95 |
| PointPush2 | -1.40 +/- 0.71 | 13.00 +/- 8.17 | -0.03 +/- 0.01 | 10.90 +/- 3.85 |

The values above are recorded in `results/safety_gym_results.csv`. Use the aggregation script to compute the same metrics from new runs.

## License

This code is released under the [MIT License](LICENSE).

## Citation

If you use this code or its results, please cite the paper:

```bibtex
@misc{zhou2026primaldual,
  title         = {Primal-Dual Representation Learning for Low-Rank Constrained MDPs},
  author        = {Chenhao Zhou and Chao Zhang and Wensong Bai and Hanbin Zhao and Hui Qian},
  year          = {2026},
  eprint        = {2610.30756},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.30756}
}
```
