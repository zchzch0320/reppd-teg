from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = "cmdp/safe_sac_lag_speder.py"


@dataclass(frozen=True)
class RepConfig:
    variant_name: str
    feature_lr: float
    speder_lam: float
    speder_coef: float
    cost_coef: float


SAFETY_GYM_CONFIGS: dict[str, RepConfig] = {
    "Safexp-PointGoal1-v0": RepConfig(
        variant_name="lam512_sc05_cc0_lr1e5",
        feature_lr=1e-5,
        speder_lam=512.0,
        speder_coef=0.5,
        cost_coef=0.0,
    ),
    "Safexp-PointPush1-v0": RepConfig(
        variant_name="fix_default",
        feature_lr=5e-6,
        speder_lam=1024.0,
        speder_coef=1.0,
        cost_coef=0.1,
    ),
    "Safexp-PointPush2-v0": RepConfig(
        variant_name="lam1024_sc025_cc0_lr5e6",
        feature_lr=5e-6,
        speder_lam=1024.0,
        speder_coef=0.25,
        cost_coef=0.0,
    ),
}

SEEDS = (0, 1, 2)


def short_env(env_id: str) -> str:
    return env_id.removeprefix("Safexp-").removesuffix("-v0")


def build_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for env_id, rep_cfg in SAFETY_GYM_CONFIGS.items():
        for seed in SEEDS:
            raw_exp_name = f"raw_{short_env(env_id)}"
            raw_cmd = [
                args.python,
                str(TRAIN_SCRIPT),
                "--env-id",
                env_id,
                "--feature-type",
                "raw",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--total-steps",
                str(args.total_steps),
                "--log-dir",
                args.log_dir.as_posix(),
                "--exp-name",
                raw_exp_name,
            ]
            jobs.append(
                {
                    "environment": env_id,
                    "method": "raw",
                    "seed": seed,
                    "variant": "default",
                    "command": raw_cmd,
                }
            )

            rep_exp_name = f"rep_pd_nn_{short_env(env_id)}_{rep_cfg.variant_name}"
            rep_cmd = [
                args.python,
                str(TRAIN_SCRIPT),
                "--env-id",
                env_id,
                "--feature-type",
                "speder",
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--total-steps",
                str(args.total_steps),
                "--log-dir",
                args.log_dir.as_posix(),
                "--exp-name",
                rep_exp_name,
                "--feature-lr",
                str(rep_cfg.feature_lr),
                "--speder-lam",
                str(rep_cfg.speder_lam),
                "--speder-coef",
                str(rep_cfg.speder_coef),
                "--cost-coef",
                str(rep_cfg.cost_coef),
            ]
            jobs.append(
                {
                    "environment": env_id,
                    "method": "REP-PD-NN",
                    "seed": seed,
                    "variant": rep_cfg.variant_name,
                    "rep_config": asdict(rep_cfg),
                    "command": rep_cmd,
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or list the three Safety Gym REP-PD-NN experiments."
    )
    parser.add_argument("--run", action="store_true", help="Execute jobs sequentially. By default, commands are only printed.")
    parser.add_argument("--python", default="python", help="Python executable in the target environment.")
    parser.add_argument("--device", default="cuda", help="Torch device passed to the training script.")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs") / "safety_gym_runs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "safety_gym_experiment_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_commands(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    print(f"Prepared {len(jobs)} jobs: 3 environments x 2 methods x 3 seeds.")
    print(f"Manifest: {args.manifest}")
    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx:02d}/{len(jobs):02d}] {job['environment']} {job['method']} seed={job['seed']}")
        print(" ".join(str(part) for part in job["command"]))

    if not args.run:
        print("Dry run only. Add --run to execute sequentially.")
        return

    for idx, job in enumerate(jobs, start=1):
        print(f"Running [{idx:02d}/{len(jobs):02d}] {job['environment']} {job['method']} seed={job['seed']}")
        subprocess.run([str(part) for part in job["command"]], cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
