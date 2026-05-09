# SINDy-RL for Inverted Double Pendulum — Project Tracker

**Authors:** Andrew Falcone & Patrick Smith  
**Course:** ME595  
**Last updated:** 2026-05-08

---

## Goal

Two parallel tracks run independently and produce separate deliverables:

- **Track A — SINDy Dynamics (Andrew):** Identify a SINDy dynamics surrogate via an iterative active learning loop. An RL controller is co-trained to reach near-equilibrium states the random policy cannot, generating the data SINDy needs to model pole-balance dynamics. The RL controller trained inside the surrogate is evaluated in the real simulator as part of Track A. Output: a validated SINDy dynamics surrogate + the RL controller trained in that surrogate.
- **Track B — Sparse SINDy Policy (Patrick):** Train a reduced-order NN policy directly on the full-order MuJoCo simulator using the 6-dim state. Distill the trained policy into a sparse SINDy polynomial `π(x)→u` via regression on (state, action) pairs. The NN is an intermediate step; the deliverable is the sparse control law.

Phase 3 then joins the two tracks by taking Track A's validated SINDy dynamics model and applying Track B's distillation methodology inside it — training a 6-dim NN policy in the surrogate, then distilling to a sparse SINDy polynomial `π(x)→u`. Output: a fully interpretable closed-loop system where both the dynamics model and the control law are sparse polynomial expressions over the same 6-dim state.

**Comparison:** Phase 4 evaluates four conditions in the real simulator against real-simulator interaction count, wall-clock time, and task success rate:

| Condition | Dynamics model | Policy |
|-----------|---------------|--------|
| Baseline | Black box (MuJoCo) | NN, 9-dim obs |
| Track A | Sparse SINDy equations | NN trained in surrogate |
| Track B | Black box (MuJoCo) | Sparse SINDy π(x)→u |
| Phase 3 Join | Sparse SINDy equations | Sparse SINDy π(x)→u |

---

## Workstream Ownership

| Phase | Area | Owner |
|-------|------|-------|
| 1 | Simulation environment setup | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| B | Full-order PPO baseline — reference policy trained on full dynamics | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| 2a | Track A: iterative active SINDy dynamics + RL controller co-training | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> |
| 2b | Track B: NN policy on full-order MuJoCo → sparse SINDy policy distillation | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> |
| 3 | Join: Train 6-dim NN in Track A's surrogate → distill to sparse SINDy π(x)→u | Shared |
| 4 | Evaluation & comparison (four-way) | Shared |

---

## Project Overview

```
                              ┌─► 2a: Track A:  SINDy Dynamics       [Andrew]  ─────────────┐
1: Setup ──► B: Baseline ─────┤                                                             ├──► 3: Join ──► 4: Compare
             [Patrick]        └─► 2b: Track B: Sparse SINDy π(x)→u   [Patrick] ─────────────┘     [Shared]     [Shared]
```

**Baseline — Full-Order PPO Reference Policy** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>):
- Full-order MLP policy (9-dim obs, [64, 64]) trained directly in MuJoCo — no surrogate, no state reduction
- Result: pending — performance ceiling for Phase 4

**Track A — SINDy Dynamics Identification** (<span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>):
- A random policy cannot reach the near-equilibrium states needed to model pole-balance dynamics; an RL controller is required
- Iteratively co-train SINDy dynamics + RL controller: controller collects near-equilibrium data → refit SINDy → retrain controller in improved surrogate → repeat
- Output: validated SINDy dynamics surrogate (6-dim state, polynomial degree 2) + RL controller trained in the surrogate

**Track B — Sparse SINDy Policy** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>):
- Train an NN policy on the 6-dim reduced state directly in the full-order MuJoCo simulator (intermediate step)
- Distill the trained NN policy into a sparse SINDy polynomial `π(x)→u` via regression on (state, action) pairs
- Output: sparse polynomial control law `π(x)→u` — the NN is not the deliverable

**Phase 3 — Join** (Shared):
- Combines both tracks: using Track A's validated SINDy dynamics as the surrogate environment, apply Track B's approach — train a 6-dim NN policy inside the surrogate, then distill it into a sparse SINDy polynomial `π(x)→u`
- Output: fully interpretable closed-loop system where both the dynamics model and the control law are sparse polynomial expressions over the same 6-dim state

**Phase 4 — Compare** (Shared):
- Four-way comparison against the Baseline: Track A (SINDy dynamics + surrogate-trained controller), Track B (sparse SINDy policy), and Phase 3 Join (SINDy dynamics + sparse SINDy policy)

---

## Phase Summary

| Phase | Description | Owner | Status |
|-------|-------------|-------|--------|
| 1 | Simulation environment setup | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ✅ Complete |
| B | Full-order PPO baseline (`simulation.ipynb`) | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ✅ Complete |
| 2a | Track A: iterative active SINDy + RL controller | <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span> | ⬜ Next |
| 2b | Track B: NN on full-order MuJoCo → sparse SINDy policy distillation | <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span> | ⬜ Next |
| 3 | Join: Train 6-dim NN in Track A's surrogate → distill to sparse SINDy π(x)→u | Shared | ⬜ |
| 4 | Evaluation & comparison (four-way) | Shared | ⬜ |

> Tracks A and B are fully independent and run in parallel. Phase 3 joins them.

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

### Baseline — Full-Order PPO Reference Policy

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>  
**Deliverable:** `notebooks/simulation.ipynb` + `notebooks/checkpoints/baseline/best_model.zip`

Trains a full-order neural network policy (MLP) directly on the MuJoCo simulator — no surrogate model, no state reduction. Provides the performance ceiling for Phase 4: the best achievable result when given unlimited access to the real simulator.

**Policy architecture:**
- Input: 9-dim observation (full MuJoCo obs)
- Network: two hidden layers of 64 units each (`[64, 64]`)
- Output: 1-dim cart force ∈ [−1, 1]

**Training approach:**
- PPO with 8 parallel envs, 1M steps, no BC initialisation (dense reward from step 1, no cold-start problem)
- `EvalCallback` saves best checkpoint; `CheckpointCallback` saves periodic snapshots

**Results:**

| Metric | Value |
|--------|-------|
| Total training steps | pending |
| Mean episode length | pending |
| Task success rate | pending |

---

### Phase 2a — Active SINDy + RL Controller (Track A)

**Owner:** <span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>  
**Deliverable:** `notebooks/trackA_sindy_dynamics.ipynb` + `data/trajectories_trackA_iter{N}.npz` + saved SINDy model + saved controller

The core problem with passive data collection for SINDy: random policy rollouts terminate almost immediately (tip falls within a few steps) and never explore the near-equilibrium state space that matters. This track solves it with an iterative active learning loop — SINDy and the controller co-improve each other.

**Iterative loop:**

```
Bootstrap data (near-upright perturbations)
    └──► Fit SINDy dynamics
              └──► Train RL controller inside SINDy surrogate
                        └──► Run controller in real sim → collect near-equilibrium data
                                  └──► Refit SINDy on expanded dataset
                                            └──► Repeat until rollout error converges
```

**Iteration 0 — Bootstrap:**
- Collect a small dataset (~2 000 transitions) by resetting to the upright equilibrium with small noise on `[θ₁, θ₂, θ̇₁, θ̇₂]` and applying random perturbations
- Fit initial SINDyc `(xₖ, uₖ) → xₖ₊₁`; expect high rollout error at this stage

**Each subsequent iteration:**
- Train a PPO controller inside the current SINDy surrogate until it can sustain pole balancing for > 200 steps in the surrogate
- Roll the trained controller out in the **real simulator** + small random perturbations; collect new `(X, U, X_next)` transitions
- Append new transitions to the dataset; refit SINDy
- Evaluate: one-step RMSE + multi-step rollout error on held-out data
- Stop when rollout error plateaus or a fixed iteration budget is reached (2–3 iterations is a reasonable target)

**State representation (6-dim):**
```
x = [x,                  # cart position
     θ₁, θ₂,             # pole angles from qpos
     ẋ, θ̇₁, θ̇₂]         # velocities from qvel
```

**Control input:** `u = [F]` (1-dim cart force, `Box(-1, 1)`)

**SINDy fit details:**
- Feature library: polynomial degree 2 over 7-dim input (6-dim state + 1-dim action)
- Solver: STLSQ (Sequential Thresholded Least Squares); tune sparsity threshold λ
- Fit one model per output dimension (6 models, each predicting `Δxᵢ`)
- Consider E-SINDy (ensemble) for uncertainty estimates to guide when to collect more data

**Evaluation metrics:**
- One-step prediction RMSE on held-out trajectories
- Multi-step rollout error (surrogate rollout vs. real MuJoCo from same initial condition)
- Qualitative check: does surrogate predict instability correctly near the falling threshold?

**Key tasks:**
- [ ] Collect bootstrap dataset (~2 000 transitions) with near-upright resets + random perturbations
- [ ] Fit initial SINDyc; record baseline rollout error
- [ ] Train PPO inside SINDy surrogate until controller sustains > 200 steps
- [ ] Collect new near-equilibrium data in real sim using trained controller + perturbations
- [ ] Refit SINDy on expanded dataset; compare rollout error to previous iteration
- [ ] Repeat for 2–3 iterations or until rollout error plateaus
- [ ] Save final SINDy model and controller; visualize state space coverage improvement across iterations
- [ ] Plot rollout error vs. iteration number

**`SINDySurrogateEnv` wrapper** (needed for controller training in Step 2+):
```
SINDySurrogateEnv(gymnasium.Env):
    - observation_space: Box(-inf, inf, (9,))   # match MuJoCo obs format
    - action_space:      Box(-1, 1, (1,))
    - step(): SINDy model → 6-dim state → reconstruct 9-dim obs; compute reward identically to MuJoCo
    - reset(): near-upright equilibrium + small noise, or sample from dataset initial conditions
    - terminated: tip height < 1 m or |cart x| > 0.2 m
```

> **Coordination note:** Track A's `SINDySurrogateEnv` and Phase 3 share the same wrapper interface. Implement in `sindy_rl/envs/sindy_env.py` so Phase 3 can load Track A's model directly without code changes.

**Reward replication:**
```python
# tip_height: height of second pole tip (computed from θ₁, θ₂ and pole lengths)
# dist: horizontal distance of tip from upright
# vel: angular velocity magnitudes
r = alive_bonus - dist_penalty - vel_penalty
```

---

### Phase 2b — NN Policy → Sparse SINDy Policy Distillation (Track B)

**Owner:** <span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>  
**Deliverable:** `notebooks/trackB_sindy_rl_policy.ipynb` + saved sparse SINDy policy `π(x)→u`

Train an NN policy in the full-order MuJoCo simulator using the 6-dim state, then distill it into a sparse SINDy polynomial control law. The NN is an intermediate step used to generate (state, action) pairs for SINDy regression. The deliverable is the sparse policy function, not the NN. This track runs entirely on the full-order simulator — no surrogate is built.

**State representation (6-dim):**
```
x = [x,                  # cart position
     θ₁, θ₂,             # pole angles from qpos
     ẋ, θ̇₁, θ̇₂]         # velocities from qvel
```

**Steps:**
1. Implement `ReducedObsEnv`: wrap `InvertedDoublePendulum-v5` to expose the 6-dim state instead of the 9-dim observation
2. Train PPO in the real simulator with reduced obs; verify the 6-dim state is sufficient — this policy is intermediate
3. Collect (state, action) pairs by rolling out the trained NN policy; fit SINDy to regress `x → u`
4. Evaluate the sparse SINDy policy `π(x)→u` in the real simulator

**Key tasks:**
- [ ] Implement `ReducedObsEnv` (extracts 6-dim state from MuJoCo `qpos`/`qvel`)
- [ ] Train PPO in real simulator with 6-dim obs; monitor episode length and reward
- [ ] Roll out trained NN policy; collect (state, action) dataset for SINDy distillation
- [ ] Fit SINDy policy `π(x)→u` using polynomial feature library + STLSQ sparsification
- [ ] Evaluate sparse SINDy policy in real simulator over 20+ episodes; record mean reward and stability
- [ ] Record real-simulator interaction count and wall-clock training time for Phase 4 comparison
- [ ] Save sparse policy and learning curve

---

### Phase 3 — Join: Track A Dynamics + Sparse SINDy Policy

**Owner:** Shared  
**Deliverable:** `notebooks/join_sindy_rl.ipynb` + saved joined policy

Take Track A's actively-learned SINDy dynamics model (higher quality, near-equilibrium data) and apply Track B's distillation methodology to it. This is the "best of both worlds" condition — sparse dynamics from the active loop, sparse policy from distillation, both expressed over the same 6-dim state.

**Approach:**
- Load Track A's final SINDy dynamics model (after iterative refinement)
- Wrap it as a `SINDySurrogateEnv` using the same interface as Track A
- Train a 6-dim NN policy inside the surrogate (same procedure as Track B Step 1–2, but in the surrogate instead of the real sim)
- Collect (state, action) pairs from the trained NN; fit sparse SINDy polynomial `π(x)→u` (same Step 3 as Track B)
- Evaluate in the real MuJoCo simulator

**Why this join matters:**
- Track B's policy distillation is limited by the amount of real-simulator interaction required during NN training
- Track A's dynamics model enables training the NN in the surrogate instead — dramatically fewer real-simulator interactions
- The join tests whether a better training environment (Track A's surrogate) produces a better sparse policy — and whether the combination outperforms either track alone

**Key tasks:**
- [ ] Load Track A SINDy model; wrap as `SINDySurrogateEnv`
- [ ] Train 6-dim NN policy inside Track A surrogate
- [ ] Collect (state, action) pairs from surrogate-trained NN; fit sparse SINDy policy `π(x)→u`
- [ ] Evaluate joined policy in real simulator over 20+ test episodes
- [ ] Compare sparse policy interpretability vs. Track B policy (same distillation, different training environment)
- [ ] Save joined policy and learning curve for Phase 4 comparison

---

### Phase 4 — Evaluation & Comparison

**Owner:** Shared  
**Deliverable:** `notebooks/evaluation.ipynb` + figures for report

Four-way comparison in the real MuJoCo simulator (all vs. Baseline):

| Condition | Dynamics model | Policy |
|-----------|---------------|--------|
| Baseline | Black box (MuJoCo) | NN, 9-dim obs |
| Track A | Sparse SINDy equations | NN trained in surrogate |
| Track B | Black box (MuJoCo) | Sparse SINDy π(x)→u |
| Phase 3 Join | Sparse SINDy equations | Sparse SINDy π(x)→u |

| Metric | Baseline | Track A | Track B | Phase 3 Join |
|--------|----------|---------|---------|--------------|
| Final mean episode reward (real sim) | pending | | | |
| Task success rate (% reaching 1000 steps) | pending | | | |
| Real simulator interactions | 1M (PPO steps) | Track A iterations only | Unlimited (PPO steps) | Track A iterations only |
| Wall-clock training time | pending | | | |
| SINDy one-step prediction RMSE | — | Track A's model | — | Track A's model |
| Policy state dimensionality | 9-dim | 6-dim | 6-dim | 6-dim |
| Policy interpretability | Black-box NN | Black-box NN | Sparse polynomial | Sparse polynomial |

**Key tasks:**
- [ ] Test all four conditions in real simulator for 20+ episodes each; record mean reward and stability
- [ ] Plot reward vs. real simulator interactions (all conditions on same axes)
- [ ] Plot reward vs. wall-clock time (all conditions on same axes)
- [ ] Print Track A's discovered SINDy equations; assess physical interpretability
- [ ] Print Track B's and Phase 3's sparse SINDy policies `π(x)→u`; compare sparsity and interpretability
- [ ] Summarize: how does each condition trade off real-sim cost against interpretability and performance?

---

## Key Technical Decisions

| Decision | Current plan | When to revisit |
|----------|-------------|-----------------|
| State for SINDy | Raw 6-dim `(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)` | If angular velocity measurements are noisy |
| Feature library | Polynomial degree 2 over 7-dim input (6 state + 1 action) | Increase to 3 if one-step error is high |
| SINDy formulation | Discrete-time `xₖ₊₁ = f(xₖ, uₖ)` | Could try continuous-time if dt is known exactly |
| Track A controller | PPO (default); DreamerV3 if PPO is slow to converge in surrogate | Switch based on convergence speed in iteration 1 |
| Track A iteration budget | 2–3 active learning iterations | Add more if rollout error is still improving at iteration 3 |
| Track B NN training | PPO on real sim with 6-dim obs (intermediate step) | Switch to SAC if PPO slow to converge with reduced obs |
| Track B SINDy distillation | Fit SINDy on (state, action) pairs from trained NN policy | Try higher polynomial degree if sparse fit underperforms |
| Phase 3 NN training | PPO in Track A's SINDy surrogate with 6-dim obs (same as Track B but in surrogate) | Switch to SAC if PPO slow to converge in surrogate |
| Phase 3 policy distillation | Fit SINDy on (state, action) pairs from surrogate-trained NN → sparse π(x)→u | Only attempt if Track A's SINDy model achieves acceptable rollout error |
| Reward in surrogate | Replicate MuJoCo formula exactly | May need to tune if surrogate diverges frequently |

---

## Libraries to Install

```bash
uv add pysindy stable-baselines3
```

- **pysindy** — SINDy model fitting (both tracks + Phase 3)
- **stable-baselines3** — PPO (both tracks + Phase 3)

---

## File Structure

```
me595/
├── notebooks/
│   ├── inverted_double_pendulum_intro.ipynb  Phase 1  ✅ — env setup
│   ├── simulation.ipynb                      Baseline ✅ — full-order PPO reference policy (Patrick)
│   ├── trackA_sindy_dynamics.ipynb           Phase 2a ⬜ — SINDy dynamics + RL controller (Andrew)
│   ├── trackB_sindy_rl_policy.ipynb          Phase 2b ⬜ — NN policy → sparse SINDy distillation (Patrick)
│   ├── join_sindy_rl.ipynb                   Phase 3  ⬜ — Track A surrogate + Track B distillation (Shared)
│   └── evaluation.ipynb                      Phase 4  ⬜ — four-way comparison (Shared)
├── sindy_rl/
│   ├── envs/
│   │   └── sindy_env.py                      SINDySurrogateEnv wrapper (Track A + Phase 3)
│   └── models/
│       ├── trackA_sindy_iter{N}.pkl           SINDy dynamics model per iteration (Track A)
│       ├── trackA_controller/                PPO controller (Track A)
│       └── trackB_sindy_policy.pkl           Sparse SINDy policy π(x)→u (Track B)
├── data/
│   └── trajectories_trackA_iter{N}.npz       Controller-generated data per iteration (Track A)
├── results/
│   ├── trackA/                               Track A rollout errors, learning curves, eval
│   ├── trackB/                              Track B learning curves, eval + sparse policy
│   └── join/                                Phase 3 join learning curve + eval
└── eng-docs/
    └── project_tracker.md                    This file
```

---

## Next Steps

Phase 1 is complete. The Baseline and Tracks A and B are all independent and can proceed in parallel:

**Baseline** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>) — Run `notebooks/simulation.ipynb`. Train PPO on `InvertedDoublePendulum-v5` for 1M steps; the notebook handles training, evaluation, and plots automatically. Fill in the Results table in this file once complete.

**Phase 2a** (<span style="background-color: #cfe2ff; color: #084298; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Andrew</span>) — Collect ~2 000 bootstrap transitions using near-upright resets with random perturbations; fit an initial SINDy dynamics model in `notebooks/trackA_sindy_dynamics.ipynb`. Then train a PPO controller inside that surrogate and use it to collect higher-quality near-equilibrium data from the real MuJoCo sim. Refit and iterate.

**Phase 2b** (<span style="background-color: #d1e7dd; color: #0f5132; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: bold;">Patrick</span>) — In `notebooks/trackB_sindy_rl_policy.ipynb`, implement `ReducedObsEnv` (wraps MuJoCo to expose 6-dim state) and train PPO directly in the real simulator. Then collect (state, action) pairs from the trained NN and fit a sparse SINDy policy `π(x)→u`. The sparse polynomial control law is Track B's deliverable — not the NN.

**Phase 3 join** (Shared, after 2a converges) — Load Track A's final SINDy dynamics model, wrap as `SINDySurrogateEnv`, and apply Track B's distillation methodology inside it in `notebooks/join_sindy_rl.ipynb`. Train the NN in the surrogate, distill to sparse `π(x)→u`. This produces the fourth condition for Phase 4 comparison.
