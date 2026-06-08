# Notebooks

All experiments for the ME 595 final project: *Interpretable Control for Unstable Systems via SINDy-RL*.

Notebooks are organized as a dependency graph, not a strict linear pipeline. Saved models and trajectory files live in `../data/` and `../results/`; no notebook overwrites another's artifacts.

**Run order:**
1. `full-order-simulation.ipynb` — independent; produces the baseline checkpoint used by `analysis.ipynb` for comparison.
2. `sindy-rl.ipynb` — must run before any other Dyna variant; produces `bootstrap_schroeder.npz`, which all four variants share.
3. `sindy-rl-sac`, `sindy-rl-lagrangian`, `sindy-rl-sac-lagrangian`, `sindy-rl-no-x` — parallel; each reads only the shared bootstrap data and writes to its own results directory. Run in any order (or concurrently).
4. `analysis.ipynb` — runs after whichever Dyna variants you want evaluated. §1–§5 need only `sindy-rl.ipynb`; §6 additionally needs `sindy-rl-sac.ipynb`; §7 additionally needs `sindy-rl-lagrangian.ipynb`; §8 additionally needs `sindy-rl-sac-lagrangian.ipynb` and `sindy-rl-no-x.ipynb`.

---

## Design Space

The project explores three axes of the RL design space:

**Model-free vs model-based.** The baseline (`full-order-simulation.ipynb`) is **model-free**: PPO interacts directly with the real MuJoCo environment for 400,000 steps. The SINDy-RL variants are **model-based**: a sparse surrogate world model (E-SINDy) is fit from real data, and the RL agent trains inside the surrogate — costing orders of magnitude fewer real-env interactions.

**On-policy vs off-policy.** Within the model-based Dyna loop, the RL optimizer can be on-policy or off-policy:
- **PPO (on-policy)**: samples are collected from the current policy and discarded after each update. Theoretically cleaner in Dyna because the surrogate is refitted each iteration (old buffer entries would be stale), but higher sample variance.
- **SAC (off-policy)**: maintains a replay buffer across iterations and maximises entropy. Theoretically a liability in Dyna (stale buffer), but entropy regularisation appears to compensate — SAC converged faster (13.0× efficiency vs PPO's 6.0×) on this system.

**Surrogate library.** The feature library used to fit the E-SINDy surrogate is a third axis — degree-3 polynomial (120 generic monomials) vs 32 physics-derived Lagrangian atoms.

### Baseline

The baseline (`full-order-simulation.ipynb`) is **model-free** PPO that interacts directly with the real MuJoCo environment for 400,000 steps.

### Dyna variant grid

| | degree-3 polynomial (120 feat) | Lagrangian atoms (32 feat) |
|---|---|---|
| **PPO — on-policy, model-based** | [`sindy-rl.ipynb`](sindy-rl.ipynb) ← baseline | [`sindy-rl-lagrangian.ipynb`](sindy-rl-lagrangian.ipynb) |
| **SAC — off-policy, model-based** | [`sindy-rl-sac.ipynb`](sindy-rl-sac.ipynb) | [`sindy-rl-sac-lagrangian.ipynb`](sindy-rl-sac-lagrangian.ipynb) |

`sindy-rl-no-x.ipynb` is a feature ablation within the PPO / degree-3 cell (drops cart position x, reducing the library from 120 to 84 features).

**Distilled policies.** Each Dyna variant produces a trained NN policy. A second stage distills this NN into a sparse polynomial controller via behavioural cloning (SINDy on expert rollouts). The distilled policy is model-free at inference time — no surrogate needed — and is orders of magnitude smaller than the NN (tens of terms vs 9,731 parameters). The RL taxonomy above describes how the NN teacher was trained, not the final controller.

---

## Pipeline

### 1. `inverted_double_pendulum_intro.ipynb`
**Purpose:** Environment validation.  
Verifies the MuJoCo `InvertedDoublePendulum-v5` geometry (link lengths, tip-height formula, reward formula) against Gymnasium's internal state. Establishes notation and confirms that the environment matches the equations used throughout the remaining notebooks. Run this first to confirm your MuJoCo installation is correct.

---

### 2. `full-order-simulation.ipynb`
**Purpose:** Full-order performance ceiling. **Model-free, on-policy.**  
Trains a standard PPO agent ([64,64] MLP, 9,731 parameters) with unlimited real-environment access (400,000 steps). Saves the best checkpoint to `../data/baseline/`. This is the comparison point for all subsequent data-efficiency claims.

**Key outputs:** 100% success, 1,000 mean episode length, 9,731-parameter neural network.

---

### 3. `sindy-rl.ipynb`
**Purpose:** SINDy-RL Dyna loop — PPO learner, degree-3 polynomial library. **Model-based, on-policy. Baseline Dyna variant.**  
Implements Algorithm 1 from Zolman et al. (2024): Schroeder bootstrap → E-SINDy fit → PPO surrogate training → real data collection → repeat. Saves one PPO checkpoint per iteration to `../results/sindy_rl/ppo_models/`. The in-loop tracker picks the best checkpoint by `real_len` during the run; `analysis.ipynb` does the definitive cross-evaluation of all saved checkpoints.

**Key outputs:** 30 PPO checkpoints (`ppo_iter1.zip`–`ppo_iter30.zip`), bootstrap data (`../data/sindy_rl/bootstrap_schroeder.npz`), convergence table (66,277 real steps at convergence).  
**Note:** The in-loop tracker picks the best checkpoint by `real_len`; the definitive cross-evaluation of all 30 checkpoints is in `analysis.ipynb`.

---

### 4. `sindy-rl-sac.ipynb`
**Purpose:** SINDy-RL Dyna loop — SAC learner, degree-3 polynomial library. **Model-based, off-policy. Algorithm swap from baseline.**  
Replaces PPO with Soft Actor-Critic. Converged in **4 iterations** using **30,735 real steps** (13.0× sample efficiency vs baseline, vs 6.0× for PPO). SAC's off-policy replay buffer is theoretically a liability in Dyna-style MBRL because the surrogate is refitted each iteration, making old buffer entries stale; in practice the entropy regularization appears to compensate, at least on this system.

**Key outputs:** Best NN iter 5, 80% success, 847 mean ep len; sparse polynomial 164/165 terms, R²=0.9813, mean_len=661.

---

### 5. `sindy-rl-lagrangian.ipynb`
**Purpose:** SINDy-RL Dyna loop — **PPO** learner, Lagrangian feature library. **Model-based, on-policy. Library swap from baseline.**  
Replaces the degree-3 polynomial library with 32 physics-derived atoms from the IDP Euler-Lagrange equations. **Only change from `sindy-rl.ipynb` is the feature library.** Ridge regression replaces STLSQ (every atom is genuine; thresholding would incorrectly zero real physics). Includes a **kinematic fidelity check**: rows Δx, Δθ₁, Δθ₂ must recover coefficient ≈ DT = 0.05 without labeled data. Saves checkpoints as `ppo_lag_iter*.zip` to `results/sindy_rl_lagrangian/ppo_models/`.

**Key outputs:** `ppo_lag_iter1.zip`–`ppo_lag_iter30.zip`; definitive evaluation in `analysis.ipynb §7`.

---

### 5b. `sindy-rl-sac-lagrangian.ipynb`
**Purpose:** SINDy-RL Dyna loop — **SAC** learner, Lagrangian feature library. **Model-based, off-policy. Both algorithm and library swapped.**  
Combines both swaps: SAC algorithm (off-policy replay) + Lagrangian atoms (32 features, ridge). Convergence behaviour and kinematic fidelity check identical to `sindy-rl-lagrangian.ipynb`. Saves checkpoints to `results/sindy_rl_lagrangian/sac_models/`.

**Key outputs:** Best SAC iter 2, 100% success, 1,000 mean ep len; sparse policy 29/29 terms, R²=0.9566, 60% success after distillation.

---

### 6. `analysis.ipynb`
**Purpose:** Definitive cross-evaluation and figure generation. **This is the source of the headline numbers in the report.** Evaluates all Dyna variants; generates distilled-policy results.  
Loads all 30 PPO checkpoints from `sindy-rl.ipynb`, evaluates each in real MuJoCo (3-episode quick scan), identifies the best checkpoint, distills it to a sparse polynomial, and runs the STLSQ threshold ablation. Also cross-evaluates the SAC and Lagrangian variant checkpoints for the algorithm and library comparison sections, and produces a unified all-variants comparison (§8). Generates all report figures without re-training. Estimated runtime: 20–40 minutes.

**Key outputs:** Best PPO (100% success, 1,000 mean ep len, 6.0× efficiency); sparse polynomial (state6, degree-3); threshold ablation (λ=0.001–50.0); condition number κ=2.37×10⁴; SAC comparison (13.0× efficiency vs 6.0× PPO); library comparison (degree-3 vs Lagrangian κ, RMSE, RL performance); all-variants comparison with distilled policy sizes; `fig_ep_lengths.png`, `fig_condition_number.png`, `fig_coefficients.png`, `fig_threshold_ablation.png`, `fig_algo_comparison.png`, `fig_library_comparison.png`, `fig_all_variants.png`.

---

## Supporting Notebooks

### `sindy-rl-no-x.ipynb`
**Model-based, on-policy. Feature ablation within the PPO / degree-3 cell.**  
Drops cart position `x` from the SINDy library (translational symmetry: ΔX = ẋ·Δt exactly by kinematics), reducing library from 120 to 84 degree-3 features. Converged in 35,111 real steps (11.4×). Saves best checkpoint to `results/sindy_rl/best_ppo_nox.zip`. Two distilled policies: §6 state6 basis (84/84 terms, R²=0.89) and §6b state5 basis — drops `x` from policy features too (55/56 terms, R²=0.87, 80% success). The near-dense STLSQ result confirms that dropping `x` improves sample efficiency but does not yield a more interpretable policy.

### `sindy_rl_strict_ppo_dyna.ipynb`
**Model-based, on-policy. Stress test of the PPO / degree-3 cell under a strict 100%-success convergence criterion.**  
Repeats the `sindy-rl.ipynb` Dyna loop with stricter settings: convergence requires all 20 eval episodes to survive ≥999 steps (vs. the relaxed 80% criterion used in the main run), 6,000 real transitions per iteration (vs. 4,000), 150k surrogate PPO steps per iteration, and targeted hard-case correction (failed episodes re-run and added to the SINDy fit). Saves per-iteration checkpoints to `results/pure_sindy_rl_strict/ppo_models/`. Additional engineering: uncertainty termination (ensemble std above the 99.5th-percentile threshold terminates the surrogate episode), targeted failure replay, and best-checkpoint rollback on performance regression. The strict criterion was never achieved: the best checkpoint (iteration 12) reached **80% success and 805 mean episode length at 108,251 cumulative real steps**; 20 iterations consumed **270,142 real steps** without exceeding that peak. An optional real-environment PPO fine-tune (50k steps from the best surrogate checkpoint) evaluated at **77% success, 770.7 mean episode length** on 30 seeds — only marginally better than the surrogate-only result. This is the backing data for the "Strict full-horizon stress test" paragraph in §4.2 of the report: it demonstrates that further tuning PPO + polynomial does not close the gap to the full-order baseline, motivating the algorithm and library swaps in subsequent sections.

### `inverted_double_pendulum_intro.ipynb`
See §1 above.

---

## Exploratory/ (not cited in report)

Early-stage experiments that informed design decisions but did not produce final results:

| Notebook | What it explored |
|---|---|
| `physics-informed-library.ipynb` | Static comparison of four SINDy library designs (degree-3, SE(3)+deg-2, SE(3)+deg-3, Lagrangian); conditioning/RMSE analysis absorbed into `analysis.ipynb §7` |
| `se3-forward-kinematics.ipynb` | SE(3) forward kinematics derivation and validation for IDP; supports the SE(3) library experiments |
| `se3-rl-lqr-comparison.ipynb` | Custom SE(3) Gymnasium environment vs MuJoCo LQR comparison; tangential to the SINDy-RL narrative |
| `sindy-hypersearch.ipynb` | STLSQ threshold and polynomial degree sweep for policy distillation |
| `sparse-policy.ipynb` | Behavioral cloning from the full-order PPO baseline (Track B) |
| `sparse-policy-poly.ipynb` | Polynomial policy variants |
| `sindy-dagger.ipynb` | DAgger-style iterative distillation |
| `koopman-dynamics.ipynb` | Koopman EDMD for dynamics identification |
| `koopman-deep.ipynb` | Deep Koopman with neural observables |
| `sindy-feature-engineering.ipynb` | Manual feature selection for SINDy library |
| `ppo-poly.ipynb` | PPO trained directly with polynomial policy head |
| `lqr-controller.ipynb` | LQR baseline for near-upright stabilization |
| `simulation.ipynb` | Early MuJoCo environment exploration |
| `swing-up.ipynb` | Swing-up from rest (far-from-equilibrium, not pursued) |
| `swing-up-v2.ipynb` | Revised swing-up attempt (alternative reward shaping) |
| `trackA_sindy_*.ipynb` | Track A experiments (alternative problem formulation) |
| `autodiff-policy.ipynb` | Automatic differentiation for sparse policy gradient |

---

## Data and Results Layout

```
../data/
  baseline/          # full-order PPO checkpoint and trajectories
  sindy_rl/          # bootstrap_schroeder.npz (shared across all Dyna notebooks)
  distillation/      # distillation dataset from sindy-rl.ipynb

../results/
  sindy_rl/
    ppo_models/      # ppo_iter1.zip … ppo_iter30.zip  (sindy-rl.ipynb and sindy-rl-no-x.ipynb share this dir)
    sac_models/      # sac_iter*.zip                   (sindy-rl-sac.ipynb)
    best_ppo_nox.zip # best no-x checkpoint by in-loop real_len (sindy-rl-no-x.ipynb)
  sindy_rl_lagrangian/
    ppo_models/      # ppo_lag_iter*.zip                (sindy-rl-lagrangian.ipynb)
    sac_models/      # sac_lag_iter*.zip                (sindy-rl-sac-lagrangian.ipynb)
    best_ppo_lagrangian.zip  # best PPO Lagrangian checkpoint by in-loop real_len
    best_policy_lagrangian.zip  # best SAC Lagrangian checkpoint by in-loop real_len
  pure_sindy_rl_strict/
    ppo_models/      # ppo_iter*.zip  (sindy_rl_strict_ppo_dyna.ipynb; best = iter 12)
```
