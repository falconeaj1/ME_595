# SINDy-RL for Inverted Double Pendulum

**Course:** ME 595 · Spring 2026
**Authors:** Patrick Smith & Andrew Falcone

---

## Key Notebooks

These two notebooks contain all results reported in the preliminary report:

| Notebook | Description |
|----------|-------------|
| [`notebooks/full-order-simulation.ipynb`](notebooks/full-order-simulation.ipynb) | **Baseline PPO** — full-order MuJoCo training (400k steps, 100% success, 9,731-param network). Performance ceiling. |
| [`notebooks/sindy-rl.ipynb`](notebooks/sindy-rl.ipynb) | **SINDy-RL Dyna loop** — E-SINDy surrogate + iterative PPO training + sparse polynomial distillation. Converges in 4 iterations (27,512 real steps); distilled 165-term degree-3 policy achieves 65% success. |

---

## Overview

This project applies the SINDy-RL framework (Zolman et al. 2024) to an inverted double pendulum on a cart. The goal is to train an interpretable sparse polynomial controller using far fewer real-simulator interactions than a standard PPO baseline.

**Two deliverables:**
1. A data-efficient NN policy trained via a SINDy-RL Dyna loop (E-SINDy surrogate + PPO, 14.5× fewer real steps than baseline)
2. A sparse degree-3 polynomial distilled from that policy (165 terms, 65% task success, fully auditable)

**Environment:** `InvertedDoublePendulum-v5` — MuJoCo cart-pole with two linked pendulum segments, 9-dim observation, 1-dim cart force action.

---

## Results Summary

| Approach | Real-env steps | Mean ep len | Success ≥500 | Policy |
|----------|---------------|-------------|--------------|--------|
| Baseline PPO | 400,000 | 1,000 | 100% | 9,731-param MLP |
| SINDy-RL NN (best Dyna checkpoint) | 27,512 | 763 | 75% | 9,731-param MLP |
| SINDy-RL Sparse (degree-3 poly) | 77,512† | 672 | 65% | 165 terms |

†27,512 Dyna steps + 50,000 MuJoCo rollouts for distillation data collection.

---

## Repository Structure

```
ME_595/
├── notebooks/
│   ├── full-order-simulation.ipynb     ★ Baseline PPO oracle
│   ├── sindy-rl.ipynb                  ★ SINDy-RL Dyna loop + distillation
│   ├── inverted_double_pendulum_intro.ipynb  Environment setup / exploration
│   ├── trackA_sindy_dynamics.ipynb     Track A: SINDy dynamics ID (Andrew)
│   ├── trackA_sindy_lqr_transfer.ipynb Track A: LQR transfer (Andrew)
│   └── [other exploratory notebooks]
├── src/                                Shared utilities (plotting, envs)
├── data/
│   ├── baseline/                       Baseline PPO checkpoints + trajectories
│   └── sindy_rl/                       Dyna loop data, SINDy models, PPO checkpoints
├── prelim-report/                      Preliminary report draft + figures
├── results/                            Saved evaluation outputs
├── references/                         Papers
└── eng-docs/
    └── project_tracker.md              Full project plan & task breakdowns
```

---

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab
```

### Managing packages

```bash
uv add <package>          # add a dependency
uv add --dev <package>    # add a dev-only dependency
uv sync                   # install after editing pyproject.toml
```

`.venv/` is gitignored — commit both `pyproject.toml` and `uv.lock` so collaborators get the same versions.

---

## Environment

| Property | Value |
|----------|-------|
| Gym ID | `InvertedDoublePendulum-v5` |
| Observation | 9-dim — `[x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, ẋ, θ̇₁, θ̇₂, f_constraint]` |
| State (for SINDy) | 6-dim — `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` |
| Action | 1-dim — cart force |
| Timestep | dt = 0.05 s (20 Hz) |
| Max episode | 50 s / 1000 steps |
| Termination | Tip height ≤ 1.0 m |
| Reward | `10·𝟙 − (h_tip − 2)² − ε‖θ̇‖²` |

---

## References

- Zolman et al. 2024 — *SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning* (arXiv:2403.09110)
- Brunton, Proctor & Kutz 2016 — *Discovering governing equations from data* (PNAS)
- Kaiser, Kutz & Brunton 2018 — *Sparse identification of nonlinear dynamics for model predictive control* (Proc. R. Soc. A)
- Fasel et al. 2022 — *Ensemble-SINDy* (Proc. R. Soc. A)
- Schulman et al. 2017 — *Proximal Policy Optimization* (arXiv:1707.06347)
- PySINDy: <https://pysindy.readthedocs.io/>
- Stable-Baselines3: <https://stable-baselines3.readthedocs.io/>
