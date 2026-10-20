"""Aggregate final deterministic evaluations from the 18 training logs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

if __package__:
    from .run_safety_gym_experiments import ROOT, SAFETY_GYM_CONFIGS, SEEDS, short_env
else:
    from run_safety_gym_experiments import ROOT, SAFETY_GYM_CONFIGS, SEEDS, short_env


HEADERS = [
    "Environment",
    "Raw Return Mean",
    "Raw Return Std",
    "Raw Cost Mean",
    "Raw Cost Std",
    "REP-PD-NN Return Mean",
    "REP-PD-NN Return Std",
    "REP-PD-NN Cost Mean",
    "REP-PD-NN Cost Std",
]


def final_evaluation(path: Path) -> tuple[int, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run log: {path}")
    last: tuple[int, float, float] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("eval_return") or not row.get("eval_cost"):
                continue
            ret, cost = float(row["eval_return"]), float(row["eval_cost"])
            if math.isfinite(ret) and math.isfinite(cost):
                last = (int(row["step"]), ret, cost)
    if last is None:
        raise ValueError(f"No finite evaluation in {path}")
    return last


def aggregate(log_dir: Path, expected_steps: int, eval_interval: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for env_id, config in SAFETY_GYM_CONFIGS.items():
        values: dict[str, list[tuple[float, float]]] = {"raw": [], "REP-PD-NN": []}
        for seed in SEEDS:
            for method, exp_name in (
                ("raw", f"raw_{short_env(env_id)}"),
                ("REP-PD-NN", f"rep_pd_nn_{short_env(env_id)}_{config.variant_name}"),
            ):
                path = log_dir / f"{exp_name}_{env_id}_seed{seed}" / "metrics.csv"
                step, ret, cost = final_evaluation(path)
                if step < expected_steps - 1 - eval_interval:
                    raise ValueError(f"Evaluation is too early for {expected_steps} steps: {path} (step {step})")
                values[method].append((ret, cost))

        row: dict[str, object] = {"Environment": env_id}
        for method, prefix in (("raw", "Raw"), ("REP-PD-NN", "REP-PD-NN")):
            for index, metric in ((0, "Return"), (1, "Cost")):
                samples = [pair[index] for pair in values[method]]
                row[f"{prefix} {metric} Mean"] = statistics.mean(samples)
                row[f"{prefix} {metric} Std"] = statistics.stdev(samples)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "outputs" / "safety_gym_runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "safety_gym_results_reproduced.csv")
    parser.add_argument("--expected-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    args = parser.parse_args()
    rows = aggregate(args.log_dir, args.expected_steps, args.eval_interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated {len(SAFETY_GYM_CONFIGS) * len(SEEDS) * 2} runs into {args.output}")


if __name__ == "__main__":
    main()
