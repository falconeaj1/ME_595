# SINDy-RL for Inverted Double Pendulum

**Course:** ME595  
**Authors:** Andrew Falcone & Patrick Smith

## Goal

Evaluate whether a SINDy-learned surrogate model of the inverted double pendulum can support RL policy training with better sample efficiency than training directly on the full MuJoCo simulator.

## Approach

1. **Phase 1** — Confirm MuJoCo `InvertedDoublePendulum-v5` environment ✅
2. **Phase 2a** (Andrew) — Collect trajectory data; fit and validate a SINDyc surrogate
3. **Phase 2b** (Patrick) — Train PPO directly on the full-order MuJoCo environment as the baseline
4. **Phase 3** (Patrick) — Train PPO inside the SINDy surrogate; evaluate transfer to MuJoCo
5. **Phase 4** (Shared) — Compare policies on reward, sample efficiency, and wall-clock time

Stretch goals: RL algorithm comparison on the surrogate (Andrew), DMDc linear surrogate baseline (Patrick).

See [`eng-docs/project_plan.md`](eng-docs/project_plan.md) for the full plan.

## Setup

Requires Python 3.14. Uses [uv](https://docs.astral.sh/uv/) to manage the environment.

```bash
uv sync
```

Run Python or tools inside the environment:

```bash
uv run python
uv run jupyter lab
```

### Managing packages

```bash
uv add <package>          # add a dependency
uv add --dev <package>    # add a dev-only dependency
uv remove <package>       # remove a dependency
uv sync                   # install after editing pyproject.toml
uv lock                   # update lockfile without installing
```

`.venv/` is gitignored — commit both `pyproject.toml` and `uv.lock` so collaborators get the same versions.

## Environment

- **Simulator:** MuJoCo `InvertedDoublePendulum-v5` via Gymnasium
- **Observation:** 9-dim `[x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, ẋ, θ̇₁, θ̇₂, f_constraint]`
- **State (for SINDy):** 6-dim `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]`
- **Action:** 1-dim continuous force on cart, `Box(-1, 1)`

## Repository Structure

```
me595/
├── notebooks/
│   ├── inverted_double_pendulum_intro.ipynb  # Phase 1 — env setup ✅
│   ├── data_collection.ipynb                 # Phase 2a — trajectory data
│   ├── sindy_surrogate.ipynb                 # Phase 2a — SINDy fit & validation
│   ├── ppo_fullorder.ipynb                   # Phase 2b — PPO baseline
│   ├── ppo_sindy_surrogate.ipynb             # Phase 3 — PPO on surrogate
│   ├── evaluation.ipynb                      # Phase 4 — comparison
│   ├── rl_comparison.ipynb                   # Stretch A — RL algorithm comparison
│   └── dmdc_surrogate.ipynb                  # Stretch B — DMDc surrogate
├── sindy_rl/
│   ├── envs/
│   │   ├── sindy_env.py                      # SINDySurrogateEnv wrapper
│   │   └── dmdc_env.py                       # DMDcSurrogateEnv wrapper
│   └── models/                               # Saved surrogate models
├── data/
│   └── trajectories.npz                      # Collected (X, U, X_next) dataset
├── results/
│   ├── ppo_fullorder/                        # PPO baseline checkpoints & curves
│   └── ppo_sindy/                            # PPO surrogate checkpoints & curves
└── eng-docs/
    └── project_plan.md                       # Detailed project plan & task tracker
```
