# SINDy-RL for Inverted Double Pendulum

**Course:** ME 595 · Spring 2026  
**Authors:** Patrick Smith & Andrew Falcone

This project applies the SINDy-RL framework from Zolman et al. (2024) to the MuJoCo `InvertedDoublePendulum-v5` benchmark. The goal is to reduce real-environment interaction cost while producing a compact, inspectable controller.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab
```

Package workflow:

```bash
uv add <package>
uv add --dev <package>
uv sync
```

`.venv/` is gitignored. Commit both `pyproject.toml` and `uv.lock` when dependencies change.

## Recommended Notebook Run Order

These are the primary notebooks to run or review for the report results.

| Order | Notebook | Purpose |
|---:|---|---|
| 0 | [`notebooks/inverted_double_pendulum_intro.ipynb`](notebooks/inverted_double_pendulum_intro.ipynb) | Optional orientation notebook for the MuJoCo environment, state/observation layout, reward, and termination geometry. |
| 1 | [`notebooks/full-order-simulation.ipynb`](notebooks/full-order-simulation.ipynb) | Baseline PPO trained directly in MuJoCo. Produces the 400,000-step, 9,731-parameter PPO reference policy and baseline trajectory data. |
| 2 | [`notebooks/sindy-rl.ipynb`](notebooks/sindy-rl.ipynb) | Main result notebook. Runs the SINDy-RL Dyna loop: Schroeder bootstrap, E-SINDy surrogate fitting, PPO training in the surrogate, real MuJoCo data collection, and polynomial policy distillation. |
| 3 | [`prelim-report/prelim-report-figures.ipynb`](prelim-report/prelim-report-figures.ipynb) | Rebuilds report figures from saved results. Use this before regenerating the preliminary report PDF if figures need refreshing. |

To regenerate the preliminary report PDF:

```bash
cd prelim-report
pandoc prelim-report-v2.md -H pandoc-header.tex -o prelim-report-v2.pdf --pdf-engine=xelatex
```

## Current Results Summary

| Approach | Real-env steps | Mean ep len | Success >=500 | Policy |
|---|---:|---:|---:|---|
| Baseline PPO | 400,000 | 1,000 | 100% | 9,731-param MLP |
| SINDy-RL NN, best Dyna checkpoint | 27,512 | 763 | 75% | 9,731-param MLP trained via SINDy surrogate |
| Distilled polynomial controller | 77,512 | 672 | 65% | 165-term degree-3 polynomial |

The distilled controller uses 27,512 Dyna-loop real steps plus 50,000 additional MuJoCo rollout states for post-training behavioral-cloning data. Gaussian perturbation augmentation re-queries the trained PPO policy and does not require additional MuJoCo stepping.

## Exploratory Notebooks

Exploratory notebooks are collected under [`notebooks/Exploratory/`](notebooks/Exploratory/). They are useful for background, diagnostics, and future work, but are not the main report reproduction path.

| Notebook | Purpose |
|---|---|
| [`notebooks/Exploratory/simulation.ipynb`](notebooks/Exploratory/simulation.ipynb) | Earlier baseline PPO training notebook, now archived because `full-order-simulation.ipynb` is the primary baseline path. |
| [`notebooks/Exploratory/sparse-policy.ipynb`](notebooks/Exploratory/sparse-policy.ipynb) | Early sparse policy distillation from the baseline PPO expert. |
| [`notebooks/Exploratory/sindy-feature-engineering.ipynb`](notebooks/Exploratory/sindy-feature-engineering.ipynb) | Feature-library experiments for improving polynomial policy fit. |
| [`notebooks/Exploratory/trackA_sindy_lqr_transfer.ipynb`](notebooks/Exploratory/trackA_sindy_lqr_transfer.ipynb) | Track-A SINDy/LQR transfer diagnostic. |
| [`notebooks/Exploratory/trackA_sindy_ppo_checkpoint_sweep.ipynb`](notebooks/Exploratory/trackA_sindy_ppo_checkpoint_sweep.ipynb) | Fixed-surrogate PPO checkpoint and action-penalty sweep. |
| [`notebooks/Exploratory/trackA_sindy_trig_ppo_checkpoint_sweep.ipynb`](notebooks/Exploratory/trackA_sindy_trig_ppo_checkpoint_sweep.ipynb) | Trig-feature branch of the fixed-surrogate PPO sweep. |

Paths inside these notebooks are resolved from the project root so they can be launched from the repo root, `notebooks/`, or `notebooks/Exploratory/`.

## Repository Structure

```text
ME_595/
├── notebooks/
│   ├── full-order-simulation.ipynb
│   ├── sindy-rl.ipynb
│   ├── inverted_double_pendulum_intro.ipynb
│   └── Exploratory/
│       └── archived exploratory notebooks
├── data/                Saved trajectory datasets
├── results/             Saved trained policies, sweeps, and evaluation outputs
├── prelim-report/       Preliminary report source, figures, and PDF
├── presentation/        Slide deck source and generated HTML/PDF
├── src/                 Shared plotting and rendering utilities
├── references/          Papers
└── eng-docs/            Project notes, planning, and diagrams
```

## Environment

| Property | Value |
|---|---|
| Gym ID | `InvertedDoublePendulum-v5` |
| Observation | 9-dim: `[x, sin(theta1), sin(theta2), cos(theta1), cos(theta2), xdot, theta1dot, theta2dot, constraint_force]` |
| Physical state for SINDy | 6-dim: `[x, theta1, theta2, xdot, theta1dot, theta2dot]` |
| Action | 1-dim cart force |
| Timestep | `dt = 0.05 s` |
| Max episode | 1,000 steps |
| Termination | Tip height <= 1.0 m |

## References

- Zolman et al. 2024, *SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning*
- Brunton, Proctor, and Kutz 2016, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*
- Kaiser, Kutz, and Brunton 2018, *Sparse identification of nonlinear dynamics for model predictive control*
- Fasel et al. 2022, *Ensemble-SINDy*
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*
