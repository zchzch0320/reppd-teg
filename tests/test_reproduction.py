from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aggregate_safety_gym_runs import aggregate, final_evaluation
from scripts.run_safety_gym_experiments import SAFETY_GYM_CONFIGS, SEEDS, build_commands, short_env


class ReproductionTests(unittest.TestCase):
    def test_launcher_prepares_exactly_eighteen_distinct_jobs(self) -> None:
        args = SimpleNamespace(python="python", device="cpu", total_steps=100, log_dir=Path("outputs/test"))
        jobs = build_commands(args)
        self.assertEqual(len(jobs), 18)
        self.assertEqual(len({(job["environment"], job["method"], job["seed"]) for job in jobs}), 18)

    def test_aggregate_final_checkpoint_and_sample_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for env_id, config in SAFETY_GYM_CONFIGS.items():
                for seed in SEEDS:
                    for method, exp_name in (
                        ("raw", f"raw_{short_env(env_id)}"),
                        ("REP-PD-NN", f"rep_pd_nn_{short_env(env_id)}_{config.variant_name}"),
                    ):
                        path = root / f"{exp_name}_{env_id}_seed{seed}" / "metrics.csv"
                        path.parent.mkdir(parents=True)
                        with path.open("w", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(handle, fieldnames=["step", "eval_return", "eval_cost"])
                            writer.writeheader()
                            writer.writerow({"step": 0, "eval_return": "nan", "eval_cost": "nan"})
                            writer.writerow({"step": 99, "eval_return": seed + 1, "eval_cost": seed + 2})
            self.assertEqual(final_evaluation(path), (99, 3.0, 4.0))
            rows = aggregate(root, expected_steps=100, eval_interval=10)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["Raw Return Mean"], 2.0)
            self.assertEqual(rows[0]["Raw Return Std"], 1.0)
            self.assertEqual(rows[0]["REP-PD-NN Cost Mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
