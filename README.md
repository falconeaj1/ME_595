# SINDy-RL for Inverted Double Pendulum

**Course:** ME595  
**Authors:** Patrick Smith & Andrew Falcone

---

## Overview

This project applies the SINDy-RL framework (Zolman et al.) to an inverted double pendulum on a cart. Two parallel tracks are compared against a full-order PPO baseline: Track A identifies a sparse SINDy dynamics surrogate via an iterative active-learning loop and trains an RL controller inside it; Track B trains a reduced-order NN policy directly on the full-order MuJoCo simulator and distills it into a sparse SINDy polynomial control law `π(x)→u`. Phase 3 joins both tracks — loading Track A's validated SINDy dynamics model and applying Track B's distillation methodology inside it to produce a fully interpretable closed-loop system. The four-way comparison (Baseline, Track A, Track B, Phase 3 Join) evaluates the trade-offs between interpretability, real-sim cost, and task performance.

**Environment:** `InvertedDoublePendulum-v5` — MuJoCo cart-pole with a double-linked pendulum, 9-dim observation, 1-dim cart force action.

---

## Repository Structure

```
me595/
├── notebooks/
│   ├── inverted_double_pendulum_intro.ipynb  Phase 1  — environment setup ✅
│   ├── simulation.ipynb                      Baseline — full-order PPO reference policy ✅
│   ├── trackA_sindy_dynamics.ipynb           Phase 2a — Active SINDy + RL controller (Andrew) ⬜
│   ├── trackB_sindy_rl_policy.ipynb          Phase 2b — NN policy → sparse SINDy distillation (Patrick) ⬜
│   ├── join_sindy_rl.ipynb                   Phase 3  — Track A surrogate + Track B distillation ⬜
│   └── evaluation.ipynb                      Phase 4  — four-way comparison ⬜
├── sindy_rl/
│   ├── envs/
│   │   └── sindy_env.py                      SINDySurrogateEnv wrapper (Track A + Phase 3)
│   └── models/
│       ├── trackA_sindy_iter{N}.pkl           SINDy dynamics model per iteration (Track A)
│       ├── trackA_controller/                PPO controller (Track A)
│       └── trackB_sindy_policy.pkl           Sparse SINDy policy π(x)→u (Track B)
├── data/
│   └── trajectories_trackA_iter{N}.npz       Controller-generated near-equilibrium data (Track A)
├── results/
│   ├── trackA/                               Track A rollout errors, learning curves, eval
│   ├── trackB/                              Track B learning curves, eval + sparse policy
│   └── join/                                Phase 3 join learning curve + eval
└── eng-docs/
    └── project_tracker.md                    Full project plan & task breakdowns
```

---

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

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
```

`.venv/` is gitignored — commit both `pyproject.toml` and `uv.lock` so collaborators get the same versions.

---

## Environment

| Property | Value |
|----------|-------|
| Gym ID | `InvertedDoublePendulum-v5` |
| Observation | 9-dim `Box(-∞, ∞)` — `[x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, ẋ, θ̇₁, θ̇₂, f_constraint]` |
| State (for SINDy) | 6-dim — `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` from `qpos`/`qvel` |
| Action | 1-dim `Box(-1, 1)` — cart force |
| Timestep | dt = 0.05 s (20 Hz) |
| Max episode | 50 s / 1000 steps |
| Termination | Tip height ≤ 1 m or \|cart x\| > 0.2 m |
| Reward | `10 − dist_penalty − vel_penalty` (alive bonus implicit) |

---

## Project Phases

| Phase | Description | Owner | Status |
|-------|-------------|-------|--------|
| 1 | Simulation environment setup | Patrick | ✅ Complete |
| B | Full-order PPO baseline (`simulation.ipynb`) | Patrick | ✅ Complete |
| 2a | Track A: iterative active SINDy + RL controller | Andrew | ⬜ Next |
| 2b | Track B: Train 6-dim NN on full-order MuJoCo → sparse SINDy policy distillation | Patrick | ⬜ Next |
| 3 | Join: Train 6-dim NN in Track A's surrogate → distill to sparse SINDy π(x)→u | Shared | ⬜ |
| 4 | Evaluation & comparison (four-way) | Shared | ⬜ |

See [eng-docs/project_tracker.md](eng-docs/project_tracker.md) for full task breakdowns.

---

## Full-Order PPO Baseline (`simulation.ipynb`)

Trains PPO directly on the MuJoCo simulator — no surrogate, no model reduction. This is the performance ceiling for Phase 4: the best achievable result when given unlimited access to the real simulator. Track A and Phase 3 Join aim to approach this with far fewer real-simulator interactions; Track B trades real-sim cost for an interpretable sparse policy.

- **Policy:** MLP `[64, 64]`, maps 9-dim obs → 1-dim cart force
- **Training:** 1M steps, 8 parallel envs, no BC initialisation needed (dense reward from step 1)
- **Result:** pending — performance ceiling for Phase 4

---

## References

- Zolman et al., *SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning* (2024)
- PySINDy: <https://pysindy.readthedocs.io/>
- Stable-Baselines3: <https://stable-baselines3.readthedocs.io/>
- Gymnasium MuJoCo environments: <https://gymnasium.farama.org/environments/mujoco/>
