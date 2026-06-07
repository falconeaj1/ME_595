# Notebooks

All experiments for the ME 595 final project: *Interpretable Control for Unstable Systems via SINDy-RL*.

Notebooks are organized as a pipeline. Run them in the order listed; each stage builds on the outputs of the previous one. Saved models and trajectory files live in `../data/` and `../results/`; no notebook overwrites another's artifacts.

---

## Pipeline

### 1. `inverted_double_pendulum_intro.ipynb`
**Purpose:** Environment validation.  
Verifies the MuJoCo `InvertedDoublePendulum-v5` geometry (link lengths, tip-height formula, reward formula) against Gymnasium's internal state. Establishes notation and confirms that the environment matches the equations used throughout the remaining notebooks. Run this first to confirm your MuJoCo installation is correct.

---

### 2. `full-order-simulation.ipynb`
**Purpose:** Full-order performance ceiling.  
Trains a standard PPO agent ([64,64] MLP, 9,731 parameters) with unlimited real-environment access (400,000 steps). Saves the best checkpoint to `../data/baseline/`. This is the comparison point for all subsequent data-efficiency claims.

**Key outputs:** 100% success, 1,000 mean episode length, 9,731-parameter neural network.

---

### 3. `sindy-rl.ipynb`
**Purpose:** SINDy-RL Dyna loop — PPO learner, degree-3 polynomial library.  
Implements Algorithm 1 from Zolman et al. (2024): Schroeder bootstrap → E-SINDy fit → PPO surrogate training → real data collection → repeat. Saves one PPO checkpoint per iteration to `../results/sindy_rl/ppo_models/`. The in-loop tracker picks the best checkpoint by `real_len` during the run; `analysis.ipynb` does the definitive cross-evaluation of all saved checkpoints.

**Key outputs:** 30 PPO checkpoints (`ppo_iter1.zip`–`ppo_iter30.zip`), bootstrap data (`../data/sindy_rl/bootstrap_schroeder.npz`), convergence table (iters 1–6, 39,616 real steps).  
**Note:** §7 summary in this notebook uses the in-loop best (iter 6, 85% success). The definitive evaluation is in `analysis.ipynb`.

---

### 4. `sindy-rl-sac.ipynb`
**Purpose:** SINDy-RL Dyna loop — SAC learner, degree-3 polynomial library.  
Replaces PPO with Soft Actor-Critic. Converged in **5 iterations** using **30,614 real steps** (13.1× sample efficiency vs baseline, vs 10.1× for PPO). SAC's off-policy replay buffer is theoretically a liability in Dyna-style MBRL because the surrogate is refitted each iteration, making old buffer entries stale; in practice the entropy regularization appears to compensate, at least on this system.

**Key outputs:** Best NN iter 5, 80% success, 847 mean ep len; sparse polynomial 164/165 terms, R²=0.9813, mean_len=661.

---

### 5. `sindy-rl-lagrangian.ipynb`
**Purpose:** SINDy-RL Dyna loop — PPO learner, Lagrangian feature library.  
Replaces the generic degree-3 polynomial library with 32 physics-derived atoms from the IDP Euler-Lagrange equations (gravity, mass-matrix coupling, Coriolis). The feature matrix condition number is κ ≈ 1.2×10³ — 840× better than degree-3's κ ≈ 1×10⁶. Despite better conditioning, surrogate RMSE plateaus at 0.39–0.45 (vs 0.016 for degree-3), and the policy achieves only 50% success. Finding: better-conditioned feature basis does not automatically produce a better surrogate — degree-3 spans a function class that is a better fit for the discrete-time MuJoCo dynamics than the continuous-time Lagrangian atoms.

**Key outputs:** Best NN iter 7, 50% success, 577 mean ep len; sparse policy 29/29 terms (no sparsification), R²=0.9507.

---

### 6. `analysis.ipynb`
**Purpose:** Definitive cross-evaluation and figure generation. **This is the source of the headline numbers in the report.**  
Loads all 30 PPO checkpoints from `sindy-rl.ipynb`, evaluates each in real MuJoCo (3 episodes quick scan), identifies the best checkpoint (iter 7, mean_len≈1000), distills it to a sparse polynomial, and runs the STLSQ threshold ablation. Also cross-evaluates the SAC and Lagrangian variant checkpoints for the algorithm and library comparison sections of the report. Generates all report figures without re-training. Estimated runtime: 10–20 minutes.

**Key outputs:** Best PPO iter 7 (90% success, 904 mean ep len); sparse polynomial 160/165 terms, R²=0.9908; threshold ablation (λ=0.001–50.0); condition number κ=2.37×10⁴; library comparison (degree-3 vs Lagrangian κ, RMSE, RL performance); `fig_ep_lengths.png`, `fig_condition_number.png`, `fig_coefficients.png`, `fig_threshold_ablation.png`, `fig_algo_comparison.png`, `fig_library_comparison.png`.

---

## Supporting Notebooks

### `sindy-rl-no-x.ipynb`
Translational symmetry variant: drops cart position `x` from the SINDy library (since ΔX = ẋ·Δt exactly by kinematics, reducing library from 120 to 84 degree-3 features). Converged in 7 iterations, 43,259 real steps. Best NN: 85% success, 856 mean ep len. Sparse polynomial: 162/165 terms, mean_len=1,000 (100% complete).

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
    ppo_models/      # ppo_iter1.zip … ppo_iter30.zip  (sindy-rl.ipynb)
    sac_models/      # sac_iter1.zip … sac_iter5.zip   (sindy-rl-sac.ipynb)
  sindy_rl_lagrangian/
    ppo_models/      # ppo_lag_iter*.zip                (sindy-rl-lagrangian.ipynb)
    best_ppo_lagrangian.zip  # best Lagrangian checkpoint by in-loop real_len
```
