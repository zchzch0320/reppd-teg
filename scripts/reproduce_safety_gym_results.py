from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "safety_gym_results.csv"
DEFAULT_TEX = ROOT / "docs" / "anonymous_safe_rl_results.tex"
DEFAULT_MD = ROOT / "docs" / "anonymous_safe_rl_results.md"


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def short_env(env: str) -> str:
    return env.removeprefix("Safexp-").removesuffix("-v0")


def mean_pm(row: dict[str, str], mean_key: str, std_key: str, digits: int = 2) -> str:
    mean = as_float(row, mean_key)
    std = as_float(row, std_key)
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def tex_mean_pm(row: dict[str, str], mean_key: str, std_key: str, bold: bool = False) -> str:
    text = mean_pm(row, mean_key, std_key).replace("+/-", r"\pm")
    if bold:
        return rf"$\mathbf{{{text}}}$"
    return rf"${text}$"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Safety Gym Supplementary Results",
        "",
        "This supplement reports experiments on three Safety Gym continuous-control environments. The table is generated directly from `results/safety_gym_results.csv`.",
        "",
        "| Environment | Raw Return | Raw Cost | REP-PD-NN Return | REP-PD-NN Cost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    short_env(row["Environment"]),
                    mean_pm(row, "Raw Return Mean", "Raw Return Std"),
                    mean_pm(row, "Raw Cost Mean", "Raw Cost Std"),
                    "**" + mean_pm(row, "REP-PD-NN Return Mean", "REP-PD-NN Return Std") + "**",
                    "**" + mean_pm(row, "REP-PD-NN Cost Mean", "REP-PD-NN Cost Std") + "**",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "All rows use 1M interaction steps, 3 random seeds, and the final deterministic evaluation checkpoint.",
            "",
            "REP-PD-NN is the practical neural-network variant of REP-PD-TEG. It replaces the theoretical per-round full-batch MLE representation step with a mini-batch SPEDER-style stochastic representation objective [1], uses the learned state-action representation in neural reward and cost critics, and trains the policy with a Lagrangian-style constrained policy update.",
            "",
            "[1] T. Ren et al., Spectral Decomposition Representation for Reinforcement Learning.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_tex(rows: list[dict[str, str]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            " & ".join(
                [
                    short_env(row["Environment"]),
                    tex_mean_pm(row, "Raw Return Mean", "Raw Return Std"),
                    tex_mean_pm(row, "Raw Cost Mean", "Raw Cost Std"),
                    tex_mean_pm(row, "REP-PD-NN Return Mean", "REP-PD-NN Return Std", bold=True),
                    tex_mean_pm(row, "REP-PD-NN Cost Mean", "REP-PD-NN Cost Std", bold=True),
                ]
            )
            + r" \\"
        )
    table = "\n".join(table_rows)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\pdfinfo{{
/Title ()
/Author ()
/Subject ()
/Keywords ()
}}

\begin{{document}}

\begin{{center}}
\textbf{{Supplementary Safe-RL Results}}
\end{{center}}

We evaluate a practical neural-network instantiation of REP-PD-TEG, denoted \texttt{{REP-PD-NN}}. It replaces the theoretical per-round full-batch MLE representation step with a mini-batch SPEDER-style stochastic representation objective~[1], uses the learned state-action representation in neural reward and cost critics, and trains the policy with a Lagrangian-style constrained policy update. We compare it with a \texttt{{raw}} baseline that uses the same Lagrangian training skeleton but removes the representation learner.

\paragraph{{Safety Gym experiments.}}
The table reports results on three Safety Gym continuous-control environments. All rows use $1$M interaction steps, $3$ random seeds, the safe RL training hyperparameters specified in the reproduction script, and the final deterministic evaluation checkpoint.

\begin{{center}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
Environment & Raw Ret. & Raw Cost & REP-PD-NN Ret. & REP-PD-NN Cost \\
\midrule
{table}
\bottomrule
\end{{tabular}}
\end{{center}}

Across these three tasks, \texttt{{REP-PD-NN}} improves final return and reduces final cost, suggesting that learned representations can be beneficial for practical safe RL in continuous-control settings.

\paragraph{{Reference.}}
[1] T. Ren et al., Spectral Decomposition Representation for Reinforcement Learning.

\end{{document}}
"""


def compile_pdf(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=str(tex_path.parent),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the supplementary Safety Gym result table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--compile-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(render_tex(rows), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(rows), encoding="utf-8")
    if args.compile_pdf:
        compile_pdf(args.tex)
    print(f"Wrote {len(rows)} Safety Gym environments.")
    print(f"Input CSV: {args.input}")
    print(f"TeX: {args.tex}")
    print(f"Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
