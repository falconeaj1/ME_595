# Preliminary Report — Narrative Arc
**"Interpretable Control for Unstable Systems via SINDy-RL"**
Patrick Smith · Andrew Falcone · ME 595 · Spring 2026

**Source constraint:** all results and claims draw from exactly two notebooks:
- `notebooks/full-order-simulation.ipynb` — baseline PPO oracle
- `notebooks/sindy-rl.ipynb` — SINDy-RL Dyna loop + sparse policy distillation

---

## Narrative Arc (one-line-per-paragraph summary)

- **§1.1** — High-stakes autonomous systems demand controllers that can be audited, certified, and deployed on constrained hardware; neural networks fail on all three counts.
- **§1.2** — SINDy produces governing equations where every term is interpretable — but an unstable system cannot generate useful training data without a controller it doesn't yet have (the chicken-and-egg problem).
- **§1.3** — We implement SINDy-RL (Zolman et al. 2024) on the inverted double pendulum: co-train a dynamics surrogate and RL policy in an iterative Dyna loop, then distill the resulting policy to a sparse polynomial.
- **§2.1** — The inverted double pendulum: 6D state, single cart-force input, two coupled unstable modes, a 0.2 m tip-height band separating upright from fall.
- **§2.2** — SINDy-C: sparse regression over a polynomial library identifies the governing terms of the closed-loop dynamics from data.
- **§2.3** — E-SINDy: a 10-member bootstrap ensemble gives per-state uncertainty estimates at no extra cost — these estimates become active penalties in the RL reward.
- **§2.4** — Dyna-style MBRL: alternating PPO training inside the surrogate with real-environment data collection, so each iteration improves both the model and the policy.
- **§3.1** — Baseline: full-order PPO with unlimited real-environment access sets the performance ceiling only — not used as a distillation teacher.
- **§3.2** — SINDy-RL pipeline: Schroeder bootstrap → Dyna loop (refit E-SINDy, PPO in surrogate, collect real data) → evaluate → distill to sparse polynomial.
- **§3.3** — Metrics: real-environment step count (sample efficiency), mean episode length, success rate (≥500 steps), SINDy RMSE, distillation R².
- **§4.1** — Baseline result: 400k steps, 100% success, mean reward 9,324, 9,731-parameter network — the ceiling.
- **§4.2** — Dyna loop converged in 4 iterations (27,512 real steps, 14.5× fewer than baseline); best NN policy achieves 75% success and mean episode length 763.
- **§4.3** — Three engineering obstacles had to be resolved before convergence: degree-2 RMSE ceiling, a silent filter geometry bug, and surrogate exploitation — each diagnosed and fixed.
- **§4.4** — Sparse policy distillation: degree-3 polynomial with 165 terms achieves 65% success; perturbation augmentation is essential to close the behavioral cloning distribution-shift gap.

---

## Section Structure

---

### §1 — Introduction & Motivation (~0.5 page)

#### §1.1 — The deployment gap

**Message:** Real autonomous systems cannot run neural network controllers that cannot be explained, certified, or deployed on embedded hardware.

**Content:**
- Autonomous systems are entering safety-critical domains (medical, aerospace, industrial); regulators require auditability.
- A 9,731-parameter MLP offers no handle for stability proofs, formal verification, or microcontroller execution.
- If the controller is a polynomial equation, each term can be audited, bounds can be derived analytically, and the policy fits in kilobytes.

**References:** Arrieta et al. 2020 [XAI survey]; Saeed & Omlin 2024 [EU AI Act]; Rudin 2019 [inherently interpretable models]

---

#### §1.2 — The SINDy promise and the data problem for unstable systems

**Message:** SINDy identifies sparse governing equations from data, but for an unstable system the data needed (near-equilibrium transitions) requires a controller that doesn't exist yet.

**Content:**
- SINDy regresses over a polynomial basis to identify which terms drive the dynamics — sparse, interpretable, compact.
- Random policies crash in ~5 steps on the IDP; the near-upright region is entirely unvisited.
- The fix: co-train the controller and model iteratively so each improves the other (Dyna-style active learning).

**References:** Brunton et al. 2016 [SINDy]; Zolman et al. 2024 [SINDy-RL]

---

#### §1.3 — This work

**Message:** We implement Algorithm 1 from Zolman et al. on the inverted double pendulum, working through a series of non-obvious implementation challenges to achieve convergence.

**Content:**
- Two outputs: (1) a data-efficient NN policy trained via SINDy-RL Dyna loop, (2) a sparse polynomial distilled from that policy.
- Contributions beyond the paper: near-upright data filter, warm-start, uncertainty penalty, best-checkpoint rollback, perturbation augmentation for distillation.

---

### §2 — Technical Background (~1 page)

#### §2.1 — The testbed

**Message:** The IDP is a demanding benchmark — two coupled unstable modes, 6D state, tiny region of attraction.

**Sourced from:** `sindy-rl.ipynb` §1 (Schroeder sweep setup), §3 (EnsembleSurrogateEnv constants).

**Key numbers:**
- L₁ = L₂ = 0.6 m; tip height range [0, 1.2] m; fall threshold 1.0 m; near-upright band only 0.2 m wide.
- Reward: alive bonus (≈10/step) minus tip-height penalty `(z − 2)²` — near zero at upright.
- 9-dim observation (sin/cos encoding of angles, velocities, cart position, constraint force).

**Figure F1:** `pendulum_diagram.png` — annotate with state vector x = [x_cart, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] and tip height h.

---

#### §2.2 — SINDy-C: discrete-time sparse dynamics

**Message:** Sparse regression over a polynomial library identifies which terms govern the discrete-time closed-loop dynamics.

**Equation:**
$$x_{k+1} - x_k = \underbrace{\Theta(x_k,\, u_k)}_{\text{120-term library}} \cdot \underbrace{\Xi}_{\text{sparse coefficients}}$$

STLSQ iteratively thresholds small coefficients to exactly zero.

**Key design choice:** degree *d* of the polynomial library is not obvious a priori. Degree-2 (36 features) turned out to be insufficient for the IDP — this became a critical obstacle (see §4.3).

**References:** Brunton et al. 2016; Kaiser et al. 2018 [SINDy-C]; Fasel et al. 2022 [E-SINDy]

---

#### §2.3 — E-SINDy and uncertainty quantification

**Message:** Fitting 10 SINDy models on 80% bootstrap subsamples gives a free uncertainty estimate; ensemble disagreement detects when the surrogate is extrapolating.

**Key mechanism:** At each step, `predict(x, u)` returns `(mean_delta, std_delta)` across 10 models. In the Dyna loop, the RL reward is penalized by `k × mean(std_delta)` — a pessimistic surrogate that steers PPO away from high-uncertainty regions. This directly addresses Zolman et al.'s §3.5 recommendation that ensemble variance serves as a diagnostic signal.

**Sourced from:** `sindy-rl.ipynb` sl-08 (FastEnsemblePredictor), sl-11 (EnsembleSurrogateEnv.step).

---

#### §2.4 — Dyna-style MBRL and behavioral cloning

**Message:** Alternating surrogate training and real data collection improves both model and policy each iteration; behavioral cloning then compresses the NN policy into a sparse polynomial.

**Figure F2:** `figures/rl_loop.svg` — the standard RL control loop, with the Environment box annotated to show that Dyna uses it twice: once as the E-SINDy surrogate (training, cheap) and once as real MuJoCo (evaluation, data collection).

**Algorithm 1 pseudocode** (Zolman 2024, adapted):
```
π₀ ← Schroeder multi-sine sweep          # data-collecting controller
D  ← collect(π₀, real env, 300 episodes)
loop for N_DYNA_ITER iterations:
    Ê  ← E-SINDy(filter_near_upright(D))  # refit ensemble on near-upright data
    π  ← PPO(warm_start=π, Ê, 100k steps) # train in surrogate
    D  ← D ∪ collect(π, real env, 4000 steps)
best_π ← arg max_{iter} real_episode_length
```

**Distillation:**
$$\min_{\Xi}\;\|\Theta_{\text{obs}}(X)\,\Xi - U^*\|_2 \quad \text{STLSQ, threshold 0.05}$$

**References:** Sutton 1990 [Dyna]; Schulman et al. 2017 [PPO]; Ross et al. 2011 [DAgger, theoretical basis for perturbation augmentation]

---

### §3 — Methods (~0.75 page)

#### §3.1 — Baseline: full-order PPO

**Source:** `notebooks/full-order-simulation.ipynb`

PPO with 8 parallel environments, [64,64] MLP tanh, 400k steps total. Saved best checkpoint by evaluation reward (eval every 3,125 steps, 20 episodes). This policy is the **performance ceiling only** — it is not used as a teacher for distillation. The distillation teacher is `best_ppo`, the highest-performing Dyna-loop checkpoint from `sindy-rl.ipynb`.

---

#### §3.2 — SINDy-RL pipeline

**Source:** `notebooks/sindy-rl.ipynb`

Five stages:

1. **Bootstrap** — 300 episodes of Schroeder multi-sine sweep (deterministic band-limited excitation, minimized crest factor to avoid premature falls).
2. **E-SINDy fit** — filter bootstrap data to near-upright transitions (tip height > 1.10 m); fit 10 SINDy-C models on 80% subsamples; stack into `FastEnsemblePredictor`.
3. **Surrogate PPO** — `EnsembleSurrogateEnv` wraps predictor; reward = MuJoCo formula − uncertainty penalty; early-stop if surrogate mean episode length < 5 after 50k steps.
4. **Real data collection** — deploy policy in MuJoCo, collect 4,000 steps, add to dataset.
5. **Repeat** — warm-start PPO from previous checkpoint; rollback to best checkpoint if exploitation detected.

**Distillation** (after Dyna loop converges):
- Teacher: `best_ppo` (highest real episode length checkpoint, not final).
- Collect 50k expert transitions; augment with 5× perturbation queries (Gaussian noise on state, oracle re-query).
- Fit degree-3 polynomial on 8-dim observation (sin/cos encoding) with STLSQ threshold 0.05.

---

#### §3.3 — Metrics

Two goals drive the evaluation: **(1) data efficiency** — can we match baseline performance with far fewer real simulator interactions? **(2) interpretability** — does the final controller use a compact, inspectable representation instead of a neural network?

| Metric | Goal | What a good value means |
|---|---|---|
| Real-env step count | Data efficiency | Lower is better. Baseline PPO needs 400,000 steps. Every step costs real hardware time and wear in a physical deployment. |
| Success rate (≥500 steps) | Task performance | Whether the controller actually works. 500 steps = 25 s of balance; 1000 = full episode. Compared against baseline's 100%. |
| Mean episode length | Task performance (graded) | A controller that falls at step 600 is meaningfully better than one that falls at step 60 — success rate alone misses this. |
| SINDy RMSE | Surrogate quality | Proxy for how much the Dyna loop can trust its internal model. High RMSE → PPO trains against inaccurate physics → policy may not transfer to real env. |
| Term count + R² (distillation) | Interpretability | Fewer terms = more compact, auditable controller. R² measures how faithfully the polynomial captures the NN's decisions before deployment testing. A high R² with a low term count is the target. |

---

### §4 — Preliminary Results (~1.5 pages)

#### §4.1 — Baseline

**Source:** `notebooks/full-order-simulation.ipynb`

**Message:** Full-order PPO is the ceiling — perfect performance, opaque, data-hungry.

**Numbers:** mean reward 9,324 ± 2, 100% success (20/20 episodes), mean length 1,000/1,000, 9,731 parameters, 400,000 real simulator interactions.

> "The policy costs 400,000 simulator interactions and produces a 9,731-parameter network that, while performant, offers no interpretability or analytic stability guarantees. It serves as the performance ceiling against which our data-efficient approach is measured."

---

#### §4.2 — Dyna loop convergence

**Source:** `sindy-rl.ipynb` §4 Dyna loop output

**Message:** Algorithm 1 converged in 4 iterations, using 14.5× fewer real-environment interactions than the baseline.

**Figure F3 (fig5_convergence.png):** Single dual-y-axis plot — x-axis = Dyna iteration (0–4). Left y-axis = mean real episode length (bars, purple). Right y-axis = SINDy RMSE (line + markers, gold). The RMSE rise at iter 3–4 while episode length also rises is the key story: the surrogate is seeing harder states as the policy improves, yet the policy still transfers. Annotate iter-4 bar with "80% success, 805 steps." Keep compact — roughly half a page column wide × 6 cm tall.

**Data table (goes in figure caption or body):**

| Iter | Cumul. real steps | SINDy RMSE | Surrogate mean len | Real mean len† | Real success† |
|---|---|---|---|---|---|
| Bootstrap | 2,897 | 0.021 | — | — | — |
| 1 | 7,023 | 0.015 | 12.6 | ~12 | 0% |
| 2 | 11,150 | 0.013 | 12.7 | ~12 | 0% |
| 3 | 15,461 | 0.094 | 31.1 | ~31 | 0% |
| 4 | 27,512 | 0.090 | **805** | **805** | **80%** |

†Within-loop real-environment evaluation (20 episodes run after each iteration's data collection). A dedicated post-loop evaluation of the iter-4 checkpoint gave mean length 763 steps (75% success); those figures are the definitive results used in §4.4 and §5.

**Inline note on RMSE rise:** The surrogate RMSE jumped from 0.013 to 0.094 at iteration 3. This is not a sign of model degradation — it reflects that the improving policy was exploring states further from the upright equilibrium, where the polynomial dynamics are less accurate. Despite higher RMSE, the surrogate remained accurate enough in the near-upright band to train a transferable policy.

---

#### §4.3 — What didn't work: three obstacles on the path to convergence

**Source:** `sindy-rl.ipynb` §4 tuning table (learning notes)

**Message:** Three non-obvious implementation obstacles prevented convergence on early attempts. Each was diagnosed from observable signals and fixed with targeted interventions.

---

**Obstacle 1 — Degree-2 RMSE ceiling**

Over 25 Dyna iterations with a degree-2 polynomial library (36 features), RMSE oscillated at 0.10–0.16 regardless of how much data was collected (5,825 → 90,055 transitions). Real episode lengths grew from 6 to only 22 steps over 25 iterations — effectively stalled.

*Diagnosis:* RMSE not decreasing with 10× more data = model class capacity ceiling. A degree-2 library cannot express the cross-coupling between the two angular modes that dominates IDP dynamics.

*Fix:* `SINDY_DEGREE = 3` — increases library from 36 to 120 features (C(9,2) vs C(10,3)). RMSE dropped to 0.013 within 2 iterations.

*Reference:* Zolman et al. use degree-2 for all benchmark environments, but the IDP has two coupled unstable modes where quadratic terms cannot capture the necessary nonlinear coupling.

---

**Obstacle 2 — Near-upright filter geometry bug**

The near-upright filter kept only transitions where tip height > `SINDY_H_MIN`. With `SINDY_H_MIN = 1.6` m and the physical maximum tip height of L₁ + L₂ = 1.2 m, no transition could ever pass the filter. The fallback (use all data when filtered count < 50) fired silently every iteration — making the filter a complete no-op.

*Diagnosis:* `fit == data` in every iteration's summary line. The filter should produce `fit < data` when it is active.

*Root cause:* `SINDY_H_MIN = 1.6` was derived from the reward formula's reference height (z = 2.0 m) rather than from the actual pendulum geometry. The reward's reference height is not an achievable physical state.

*Fix:* `SINDY_H_MIN = 1.10` m — within the achievable range (1.0, 1.2) m, corresponding to poles within ≈24° of vertical.

---

**Obstacle 3 — Surrogate exploitation**

*Note: the numbers below are from a diagnostic 30-iteration run conducted before the fixes were in place. The final convergence run (§4.2) never triggered exploitation because both safeguards were active from the start.*

At iteration 6 of that diagnostic run, surrogate reward jumped from 497 to 4,525 (9×) while real episode length collapsed from 414 to 56 steps. The policy had found action sequences that the degree-3 polynomial predicted as highly rewarding but that did not correspond to real physics.

*Why uncertainty penalty alone is insufficient:* All 10 ensemble members share the same polynomial basis. In extrapolated regions, they all make the same wrong prediction — ensemble disagreement (std) is low, but the shared prediction is wrong. The penalty cannot catch what it cannot see.

*Why rollback alone is insufficient:* Rollback detects exploitation after the fact but doesn't prevent it. The exploited iteration still collected bad real-environment data (mostly falls), degrading the SINDy fit for subsequent iterations — a compounding failure.

*Fix:* Both mechanisms together.
1. **Uncertainty penalty:** `reward -= 5.0 × mean(std_delta)` per surrogate step — steers PPO away from high-disagreement states.
2. **Best-checkpoint rollback:** detect `surr_r > 3× prev` AND `real_len < 50% best`; warm-start next iteration from the best-real-env checkpoint, not the exploited policy.

*Reference:* Zolman et al. §3.5 note ensemble variance as a diagnostic signal; this work extends it to an active pessimistic penalty (reward shaping) with a safety-net rollback.

**Figure F3** (Dyna convergence plot) shows the RMSE rise at iter 3–4 that coincided with the exploitation risk window — the uncertainty penalty and rollback prevented this from compounding.

---

#### §4.4 — Sparse policy distillation

**Source:** `sindy-rl.ipynb` §6

**Message:** Behavioral cloning from the best Dyna checkpoint produces a 165-term polynomial achieving 65% success; two implementation choices were required to reach that level.

**Figure F4 (fig6_comparison.png):** Comparison bar chart — success rate (%) and mean episode length for three methods: Baseline PPO (100%, 1000), SINDy-RL NN (75%, 763), Sparse polynomial (65%, 672). Secondary axis or annotation: real-env step count (400k / 27.5k / 77.5k).

---

**Distillation obstacle 1 — Wrong teacher**

Initial distillation used the final loop policy (iter 4 end-state). Mean episode length of the distilled policy: 32.6 steps, 0% success.

*Diagnosis:* the final policy is not the best policy. Warm-start iteration 4 began from the best checkpoint of a previous run; the final state after another 100k surrogate steps may have drifted.

*Fix:* Distill from `best_ppo` — the checkpoint with the highest within-loop real episode length (iter 4, real_len = 805). Result: mean length 136.3 steps after first distillation attempt.

---

**Distillation obstacle 2 — Degree-2 insufficient for NN policy**

Degree-2 polynomial on the 8-dim observation gave OLS R² ≈ 0.905 — a hard ceiling, regardless of data volume. A tanh neural network policy has no natural polynomial sparsity; the degree must match the function's curvature across the full state space.

*Fix:* `DIST_DEGREE = 3` — 165 candidate features on the obs-8 space; OLS R² = 0.992, STLSQ retains all 165 terms at threshold 0.05. The lack of sparsity in the policy (vs. the dynamics) is expected: physics has a sparse structure that neural network policies do not.

---

**Distillation obstacle 3 — Behavioral cloning distribution shift**

With degree-3 and the correct teacher (R² = 0.992), the distilled policy achieved 65% success and mean length 672 — but the teacher NN achieved 75% success and 763 steps. The gap arises from distribution shift: the polynomial is only trained on states the NN naturally visits; in deployment, small errors push the system off-distribution, where the polynomial's predictions are uncalibrated.

*Fix (perturbation augmentation):* For each of the 50k expert transitions, perturb the state with Gaussian noise (σ per dimension: [0.02, 0.02, 0.02, 0.05, 0.10, 0.10]) and re-query the NN oracle for the correct action. No additional MuJoCo rollouts required. Dataset expands 5× (50k → 300k pairs). R² remains 0.992; success rate rises to 65% (up from ~0% without augmentation).

*Reference:* Ross et al. 2011 (DAgger) — the theoretical basis for interactively querying the oracle at off-distribution states to reduce compounding error.

---

### §4.5 — Code repository

**Git repository:** https://github.com/falconeaj1/ME_595
*(Private repo — Michelle Hickner, github: mhickner, added as collaborator.)*

Key notebooks:
- `notebooks/full-order-simulation.ipynb` — baseline PPO training and evaluation
- `notebooks/sindy-rl.ipynb` — SINDy-RL Dyna loop, E-SINDy surrogate, sparse policy distillation

---

### §5 — Summary & Next Steps (~0.25 page)

**Message:** Both deliverables demonstrate interpretable control with a fraction of the data; the open question is combining them in a closed interpretable loop.

**Final comparison table:**

| Approach | Real-env steps | Mean ep len | Success | Interpretable | Parameters |
|---|---|---|---|---|---|
| Baseline PPO | 400,000 | 1,000 | 100% | ✗ | 9,731 |
| SINDy-RL NN (best Dyna checkpoint) | 27,512 | 763 | 75% | ✗ | 9,731 |
| SINDy-RL Sparse (degree-3 poly) | 77,512† | 672 | 65% | ✓ | 165 terms |

†27,512 Dyna real-env steps + 50,000 MuJoCo rollout steps to collect expert transitions for distillation. The 5× perturbation augmentation (250k additional training rows) re-queries the NN oracle on perturbed states — no further MuJoCo interactions required.

**Next steps:** combine the SINDy dynamics model with the sparse distilled policy in a closed interpretable loop; ablation study of STLSQ threshold vs. robustness trade-off.

---

## Figures Summary

| # | Figure | File | Status | Assignment role |
|---|---|---|---|---|
| F1 | Pendulum system diagram | `pendulum_diagram.png` | ✅ exists | System description (§2.1) |
| F2 | RL control loop | `figures/rl_loop.svg` | ✅ **done** | **Workflow diagram (required)** — agent/env/reward loop + Dyna annotation (§2.4) |
| F3 | Dyna convergence plot | `figures/fig5_convergence.png` | ✅ **done** | **Data figure (required)** — episode length + RMSE vs. iteration (§4.2) |
| **F4** | **Method comparison chart** | `figures/fig6_comparison.png` | ✅ **done** | Results summary (§4.4 / §5) |

**All four figures complete.** Minimum for assignment: F2 (workflow) + F3 (data figure). All figures are ready.

---

## References

| # | Paper | Section used | Why |
|---|---|---|---|
| [1] | Zolman et al. 2024 — SINDy-RL (arXiv:2403.09110) | Throughout | Primary method; Algorithm 1 basis; §3.5 uncertainty discussion |
| [2] | Brunton, Proctor & Kutz 2016 — SINDy (PNAS) | §2.2 | Foundational SINDy |
| [3] | Kaiser, Kutz & Brunton 2018 — SINDy-C (Proc. R. Soc. A) | §2.2 | SINDy with control inputs |
| [4] | Fasel et al. 2022 — E-SINDy (Proc. R. Soc. A) | §2.3 | Bootstrap ensemble UQ |
| [5] | Schulman et al. 2017 — PPO (arXiv:1707.06347) | §2.4, §3.1 | RL algorithm |
| [6] | Sutton 1990 — Dyna (ICML proceedings, Elsevier) | §2.4 | Original Dyna MBRL framework — alternating real/model learning |
| [6b] | Sutton, Szepesvári et al. 2008 — Dyna-style planning (UAI proceedings) | §2.4 | Extends Dyna to function approximation; cited by Zolman for "Dyna-style MBRL" framing |
| [10] | Schroeder 1970 — IEEE Trans. Inf. Theory 16(1):85–89 | §3.2, §2.4 pseudocode | Original minimum-crest-factor multi-sine sweep; directly cited by Zolman [56] and notebook §1 |
| [7] | Ross, Gordon & Bagnell 2011 — DAgger (AISTATS) | §4.4 | Theoretical basis for perturbation augmentation |
| [8] | Arrieta et al. 2020 — XAI (Inf. Fusion) | §1.1 | Interpretability requirement |
| [9] | Rudin 2019 — Stop explaining (Nat. Mach. Intell.) | §1.1 | Inherently interpretable models |

---

## CRediT Statement Draft

| Contribution | Patrick Smith | Andrew Falcone |
|---|---|---|
| Conceptualization | ✓ | ✓ |
| Methodology — SINDy-RL Dyna loop | Lead | Supporting |
| Methodology — sparse policy distillation | Lead | Supporting |
| Software — `sindy-rl.ipynb` | Lead | — |
| Software — `full-order-simulation.ipynb` | Lead | — |
| Formal analysis (obstacle diagnosis) | Lead | Supporting |
| Investigation | ✓ | ✓ |
| Writing – original draft | ✓ | ✓ |
| Writing – review & editing | ✓ | ✓ |
| Visualization | ✓ | ✓ |

---

## Notes on Scope

The presentation (`slides.md` / `slides-final.md`) describes a second parallel track: ROM surrogate via SINDy-C linearization + LQR controller. That work is **not** represented in the two specified notebooks and is therefore **excluded** from this report. If Andrew's ROM/LQR notebook is added as a source, §4.2 and §3.1 of the outline can be extended with those results.
