# SINDy-RL for Inverted Double Pendulum — Project Tracker

**Authors:** Andrew Falcone & Patrick Smith  
**Course:** ME595  
**Last updated:** 2026-04-28

---

## Goal

Evaluate whether a SINDy-learned surrogate model of the inverted double pendulum can support RL policy training with better sample efficiency than training directly on the full MuJoCo simulator. Compare the two approaches on episode reward, sample efficiency (simulator interactions required), and policy stability.

**Stretch goals:** Compare PPO policy against other RL policies, and/or extend the surrogate comparison to DMDc as a linear baseline.

---

## Workstream Ownership

| Phase | Area | Owner |
|-------|------|-------|
| 1 | Simulation environment setup | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| 2a | Data collection | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> |
| 2a | SINDy surrogate fitting & validation | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> |
| 2b | PPO full-order baseline | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| 3 | PPO on SINDy surrogate | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| 4 | Evaluation & comparison | Shared |
| Stretch A | RL algorithm comparison on SINDy surrogate | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> |
| Stretch B | DMDc surrogate comparison | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |

---

## Project Overview

```
                ┌── 2a: Data Collection → SINDy Surrogate   [Andrew]  ──┐
1: Setup ───────┤                                                         ├──► 3: PPO on Surrogate ──► 4: Compare
                └── 2b: PPO Full-Order Baseline             [Patrick] ───┘
```

**Phase 2a — Dynamics modeling track** (<span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>):
- Collect `(X, U, X_next)` trajectory data from MuJoCo (offline dataset for SINDy)
- Fit a SINDyc surrogate `(xₖ, uₖ) → xₖ₊₁` using PySINDy and validate rollout accuracy
- Output: saved surrogate model + `SINDySurrogateEnv` Gymnasium wrapper

**Phase 2b — RL baseline track** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>):
- Train PPO directly on the full MuJoCo environment (no surrogate)
- PPO collects its own experience online — no separate data collection needed
- Output: trained policy + learning curve (reward vs. simulator interactions + wall-clock time)

**Phase 3 — Join** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>):
- PPO is trained inside the SINDy surrogate using the Gymnasium wrapper from Andrew
- The resulting policy is transferred and evaluated in the real MuJoCo environment
- Simulator interaction count is limited to the Phase 2 data collection only

**Phase 4 — Compare** (Shared):
- Head-to-head comparison of both policies on reward, sample efficiency, and compute cost

**Stretch:** DMDc follows the same dynamics modeling path as Andrew's Phase 2 track, using the same dataset, and adds a third column to the Phase 4 comparison. Alternative RL policies add other comparisons as well.

---

## Phase Summary

| Phase | Description | Owner | Status |
|-------|-------------|-------|--------|
| 1 | Simulation environment setup | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ✅ Complete |
| 2a | Data collection + SINDy fit & validation | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> | ⬜ Next |
| 2b | PPO full-order baseline | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ⬜ Next |
| 3 | PPO on SINDy surrogate | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ⬜ |
| 4 | Compare results, performance & compute | Shared | ⬜ |
| Stretch A | RL algorithm comparison on SINDy surrogate (PPO/SAC/TD3/Dreamer/TD-MPC2) | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> | ⬜ |
| Stretch B | DMD / DMDc surrogate comparison | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ⬜ |

> Phases 2a and 2b are fully independent and run in parallel.

---

## Phases

### Phase 1 — Simulation Environment ✅ COMPLETE

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>  

**Deliverable:** `notebooks/inverted_double_pendulum_intro.ipynb`

The MuJoCo `InvertedDoublePendulum-v5` environment is confirmed working. Key facts established:

- **Observation space:** 9-dim — `[x, sin θ₁, sin θ₂, cos θ₁, cos θ₂, ẋ, θ̇₁, θ̇₂, f_constraint]`
- **Underlying state:** 6-dim — `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` (from `qpos`, `qvel`)
- **Action space:** 1-dim continuous force on cart, `Box(-1, 1)`
- **Reward:** `alive_bonus − dist_penalty − vel_penalty`
- **Termination:** tip height < 1 m or |cart x| > 0.2 m

---

### Phase 2a — Data Collection + SINDy Surrogate (Dynamics Modeling Track)

**Owner:** <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>  
**Deliverable:** `notebooks/data_collection.ipynb` + `notebooks/sindy_surrogate.ipynb` + `data/trajectories.npz` + saved model

Collect trajectory data from the full-order simulator to train the SINDy surrogate. The SINDy model learns the discrete-time map `(xₖ, uₖ) → xₖ₊₁`.

**State representation decision:** Use the raw 6-dim state `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` (from `qpos`/`qvel`) rather than the 9-dim observation. This avoids redundancy in the sin/cos encoding and gives SINDy cleaner targets. Angles can be recovered exactly via `arctan2(sin, cos)` from the observation.

**Collection strategy:**
- Mix of random policy, slightly-perturbed policy, and near-equilibrium rollouts to ensure state space coverage near the upright equilibrium (where the controller must operate)
- Target: ~50–100 episodes, ~500–1000 total steps (episodes are short under random policy)
- Optionally augment with Minari dataset `mujoco/inverteddoublependulum/expert-v0` for near-equilibrium coverage

**Dataset format:** Save `(X, U, X_next)` arrays where:
- `X[i]` = 6-dim state at step i
- `U[i]` = 1-dim action at step i  
- `X_next[i]` = 6-dim state at step i+1

**Key tasks:**
- [ ] Write collection loop with configurable policy (random / mixed / expert)
- [ ] Convert 9-dim observation → 6-dim state using `env.unwrapped.data.qpos/qvel`
- [ ] Save dataset to `data/trajectories.npz`
- [ ] Visualize state space coverage (scatter plots of θ₁ vs θ₂, x vs ẋ)

---

**SINDy fit & validation:**

Fit a SINDyc (SINDy with control) surrogate using PySINDy. The model predicts `xₖ₊₁ = f(xₖ, uₖ)` using a sparse selection from a nonlinear feature library.

**Approach:**
- Start with a polynomial feature library (degree 2 or 3) over `[x, u]`
- Fit one model per state dimension (6 models total): each predicts `Δxᵢ = xᵢ,ₖ₊₁ − xᵢ,ₖ`
- Use STLSQ (Sequential Thresholded Least Squares) as the sparse regression solver
- Consider E-SINDy (ensemble) for uncertainty quantification as in Zolman et al.

**Evaluation metrics:**
- One-step prediction RMSE on held-out trajectories
- Multi-step rollout error (roll out surrogate from initial condition, compare to MuJoCo)
- Qualitative check: does the surrogate predict instability correctly near the falling threshold?

**Key tasks:**
- [ ] Set up PySINDy feature library and fit baseline polynomial model
- [ ] Tune sparsity threshold (λ in STLSQ)
- [ ] Evaluate one-step and multi-step errors on test split
- [ ] Plot predicted vs actual rollouts for visual validation
- [ ] Save fitted model for use in Phase 3

---

### Phase 2b — PPO Full-Order Baseline (RL Baseline Track)

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>  
**Deliverable:** `notebooks/ppo_fullorder.ipynb` + saved policy

Train a PPO policy directly on `InvertedDoublePendulum-v5`. This is the performance and sample-efficiency baseline against which the surrogate-trained policy is compared.

**Algorithm:** PPO (Proximal Policy Optimization) via Stable-Baselines3.  
Standard hyperparameters from the SB3 MuJoCo defaults as a starting point; tune if needed.

**Logging:** Record at each checkpoint:
- Episode reward (mean over last N episodes)
- Total environment steps consumed
- Wall-clock time per N steps

**Key tasks:**
- [ ] Install Stable-Baselines3 (`uv add stable-baselines3`)
- [ ] Train PPO on full env; save checkpoints every 50k steps
- [ ] Plot learning curve: reward vs. simulator interactions
- [ ] Evaluate final policy: mean episode reward over 20 test episodes
- [ ] Save learning curve data for comparison in Phase 4

---

### Phase 3 — PPO on SINDy Surrogate

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> (uses <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>'s surrogate from Phase 2a)  
**Deliverable:** `notebooks/ppo_sindy_surrogate.ipynb` + saved policy

Wrap the SINDy surrogate as a Gymnasium-compatible environment and train PPO inside it. Periodically return to the real MuJoCo environment to evaluate and (optionally) refresh the surrogate — following the SINDy-RL loop from Zolman et al.

**Surrogate Gymnasium wrapper:**
```
SINDySurrogateEnv(gymnasium.Env):
    - observation_space: Box(-inf, inf, (9,))  # match MuJoCo obs format
    - action_space: Box(-1, 1, (1,))
    - step(): call SINDy model to predict next state; compute reward identically to MuJoCo
    - reset(): sample from dataset initial conditions or small perturbation of upright
    - terminated: replicate MuJoCo termination conditions on predicted state
```

**Training loop options (start simple, add complexity as needed):**
1. **Static surrogate:** train PPO entirely in surrogate, evaluate in MuJoCo (simplest)
2. **Dagger-style refresh:** every N surrogate episodes, collect K real episodes, refit surrogate
3. **Zolman-style interleaving:** switch between surrogate and real env based on uncertainty

Start with option 1. If transfer gap is large, move to option 2.

**Key tasks:**
- [ ] Implement `SINDySurrogateEnv` wrapper
- [ ] Verify wrapper passes `gymnasium.utils.env_checker`
- [ ] Train PPO in surrogate; log surrogate interactions separately from real interactions
- [ ] Evaluate final surrogate-trained policy in real MuJoCo env
- [ ] Record wall-clock training time for comparison

---

### Phase 4 — Evaluation & Comparison

**Owner:** Shared  
**Deliverable:** `notebooks/evaluation.ipynb` + figures for report

Compare the two conditions side-by-side:

| Metric | PPO (full-order) | PPO (SINDy surrogate) |
|--------|-----------------|----------------------|
| Final mean episode reward | | |
| Real simulator interactions to convergence | | |
| Wall-clock training time | | |
| Policy stability (% episodes surviving > 500 steps) | | |

**Key tasks:**
- [ ] Load saved policies and learning curves from Phases 2b and 3
- [ ] Plot reward vs. real simulator interactions on same axes
- [ ] Plot reward vs. wall-clock time on same axes
- [ ] Test both policies in MuJoCo for 50 episodes; report statistics
- [ ] Analyze SINDy model: print discovered equations, discuss sparsity and interpretability

---

### Stretch A — RL Algorithm Comparison on SINDy Surrogate

**Owner:** <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>  
**Deliverable:** `notebooks/rl_comparison.ipynb`

With the SINDy surrogate environment already in place, evaluate a range of RL algorithms to determine which learns the most efficiently inside the surrogate. All algorithms are evaluated in the real MuJoCo environment after training and compared against the SINDy-RL PPO result from Phase 3.

| Algorithm | Type | Sample Efficiency | Key Advantage |
|-----------|------|-------------------|---------------|
| PPO | On-policy | Moderate | Stable, easy to tune — primary baseline |
| SAC | Off-policy | High | Best asymptotic performance, robust exploration |
| TD3 | Off-policy | High | Reduces value overestimation vs. SAC |
| Dreamer | Model-based | Very High | Excellent planning; learns its own latent model |
| TD-MPC2 | Model-based | Very High | Fast, efficient planning with a learned world model |

**Note on model-based algorithms (Dreamer, TD-MPC2):** These learn their own internal dynamics model on top of the SINDy surrogate. Running them inside the surrogate tests whether a learned world model compounds with or conflicts with the SINDy structure.

**Key tasks:**
- [ ] Train SAC and TD3 inside `SINDySurrogateEnv`; evaluate in MuJoCo
- [ ] Train Dreamer and TD-MPC2 inside `SINDySurrogateEnv`; evaluate in MuJoCo
- [ ] Plot learning curves for all algorithms on same axes (reward vs. surrogate interactions)
- [ ] Report final MuJoCo performance and wall-clock training time for each

---

### Stretch B — DMD / DMDc Surrogate Comparison

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>  
**Deliverable:** `notebooks/dmdc_surrogate.ipynb`

Fit DMD and DMDc surrogates using PyDMD as linear baselines. DMDc learns `xₖ₊₁ = Axₖ + Buₖ` from the same Phase 2a dataset. Wrap identically to the SINDy surrogate and train PPO inside it. Compare against the SINDy-RL PPO result from Phase 3.

**Expected outcome:** DMDc will likely underperform SINDy due to the nonlinearity of the double pendulum, providing a clear linear vs. nonlinear comparison.

| Surrogate | Model class | One-step error | Rollout error | PPO reward |
|-----------|-------------|---------------|---------------|------------|
| SINDy | Sparse nonlinear | | | (Phase 3 result) |
| DMDc | Linear | | | |
| DMD | Linear (no control) | | | |

**Key tasks:**
- [ ] Fit DMD and DMDc models using PyDMD on the Phase 2a dataset
- [ ] Evaluate one-step and rollout errors vs. SINDy
- [ ] Implement `DMDcSurrogateEnv` wrapper (analogous to SINDy wrapper)
- [ ] Train PPO in DMDc surrogate; add to comparison table

---

## Key Technical Decisions

| Decision | Current plan | When to revisit |
|----------|-------------|-----------------|
| State for SINDy | Raw 6-dim `(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)` | If angular velocity measurements are noisy |
| Feature library | Polynomial degree 2 | Increase to 3 if one-step error is high |
| SINDy formulation | Discrete-time `xₖ₊₁ = f(xₖ, uₖ)` | Could try continuous-time if dt is known exactly |
| RL algorithm | PPO (Stable-Baselines3) | Switch to SAC if PPO is slow to converge |
| Surrogate refresh | Static first | Add Dagger-style refresh if transfer gap > 20% reward drop |
| Reward in surrogate | Replicate MuJoCo formula exactly | May need to tune if surrogate diverges frequently |

---

## Libraries to Install

```bash
uv add pysindy pydmd stable-baselines3 tensorboard
```

- **pysindy** — SINDy model fitting
- **pydmd** — DMD / DMDc
- **stable-baselines3** — PPO implementation
- **tensorboard** — training curve logging

---

## File Structure

```
me595/
├── notebooks/
│   ├── inverted_double_pendulum_intro.ipynb  Phase 1  ✅ done
│   ├── data_collection.ipynb                 Phase 2a — data collection (Andrew)
│   ├── sindy_surrogate.ipynb                 Phase 2a — SINDy fit & validation (Andrew)
│   ├── ppo_fullorder.ipynb                   Phase 2b — PPO full-order baseline (Patrick)
│   ├── ppo_sindy_surrogate.ipynb             Phase 3  — PPO on surrogate (Patrick)
│   ├── evaluation.ipynb                      Phase 4  — comparison (Shared)
│   ├── rl_comparison.ipynb                   Stretch A — RL algorithm comparison (Andrew)
│   └── dmdc_surrogate.ipynb                  Stretch B — DMDc surrogate (Patrick)
├── sindy_rl/
│   ├── envs/
│   │   ├── sindy_env.py                      SINDySurrogateEnv wrapper
│   │   └── dmdc_env.py                       DMDcSurrogateEnv wrapper
│   └── models/
│       ├── sindy_model.pkl                   Saved SINDy model
│       └── dmdc_model.pkl                    Saved DMDc model
├── data/
│   └── trajectories.npz                      Collected trajectory dataset
├── results/
│   ├── ppo_fullorder/                        Checkpoints + learning curves
│   └── ppo_sindy/                            Checkpoints + learning curves
└── eng-docs/
    └── project_plan.md                       This file
```

---

## Next Steps

These two tasks are independent and can proceed in parallel:

**Phase 2a** (<span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>) — Write `notebooks/data_collection.ipynb` to collect and save `(X, U, X_next)` trajectories from `InvertedDoublePendulum-v5`, then fit and validate the SINDy surrogate in `notebooks/sindy_surrogate.ipynb`.

**Phase 2b** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>) — Write `notebooks/ppo_fullorder.ipynb` to train PPO directly on `InvertedDoublePendulum-v5` and record the learning curve as the sample-efficiency baseline.
