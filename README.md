# SINDy-RL for Inverted Double Pendulum

**Course:** ME 595 · Spring 2026  
**Authors:** Patrick Smith & Andrew Falcone

This project applies the SINDy-RL framework from Zolman et al. (2025) to the MuJoCo `InvertedDoublePendulum-v5` benchmark. The goal is to reduce real-environment interaction cost while producing a compact, inspectable controller. We evaluate four library–optimizer combinations and find that a Lagrangian feature library combined with SAC simultaneously achieves data efficiency, a reduced-order policy, and interpretability.

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
| 0 | [`notebooks/inverted_double_pendulum_intro.ipynb`](notebooks/inverted_double_pendulum_intro.ipynb) | Optional orientation: MuJoCo environment, state/observation layout, reward, and termination geometry. |
| 1 | [`notebooks/full-order-simulation.ipynb`](notebooks/full-order-simulation.ipynb) | Baseline PPO trained directly in MuJoCo. Produces the 400,000-step, 9,731-parameter reference policy. |
| 2 | [`notebooks/sindy-rl.ipynb`](notebooks/sindy-rl.ipynb) | PPO + polynomial library. Runs the SINDy-RL Dyna loop: Schroeder bootstrap, E-SINDy surrogate fitting, PPO training in the surrogate, real MuJoCo data collection, and polynomial policy distillation. |
| 3 | [`notebooks/sindy-rl-no-x.ipynb`](notebooks/sindy-rl-no-x.ipynb) | PPO + no-x polynomial library. Variant of sindy-rl.ipynb that drops cart position x from the state, reducing the feature library from 84 to 56 terms to probe whether removing a weakly observable state improves conditioning. |
| 4 | [`notebooks/sindy_rl_strict_ppo_dyna.ipynb`](notebooks/sindy_rl_strict_ppo_dyna.ipynb) | PPO + polynomial (strict). Diagnostic variant testing whether stricter real-data correction and tighter convergence criteria can push the surrogate-trained PPO to full-order success rates. |
| 5 | [`notebooks/sindy-rl-sac.ipynb`](notebooks/sindy-rl-sac.ipynb) | SAC + polynomial library. Same Dyna loop with Soft Actor-Critic replacing PPO. |
| 6 | [`notebooks/sindy-rl-lagrangian.ipynb`](notebooks/sindy-rl-lagrangian.ipynb) | PPO + Lagrangian library. Replaces the polynomial feature library with atoms derived from the IDP Euler-Lagrange equations. |
| 7 | [`notebooks/sindy-rl-sac-lagrangian.ipynb`](notebooks/sindy-rl-sac-lagrangian.ipynb) | SAC + Lagrangian library. Combines the Lagrangian feature library with SAC. Best overall result: 22,723 real-env steps, 29-term distilled policy, 100% task success. |
| 8 | [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) | Cross-run analysis. Loads saved checkpoints from all variants, evaluates distilled policies, and generates all report figures. |

To regenerate the final report PDF:

```bash
cd report
bash build.sh --final
```

Requires `pandoc`, `xelatex`, and `ieee.csl` (via the `citation-style-language` TeX package).

## Results Summary

| Variant | Real-env steps | Efficiency | NN success | Distilled terms | Distilled success |
|---|---:|---:|---:|---|---:|
| Baseline PPO | 400,000 | 1.0× | 100% | — | — |
| PPO + poly (strict)† | 323,934 | 1.2× | 95% | 84/84 | 97%† |
| PPO + polynomial | 66,277 | 6.0× | 90% | 82/84 | 85% |
| PPO + no-x | 35,111 | 11.4× | 85% | 55/56 | 85% |
| SAC + polynomial | 30,735 | 13.0× | 100% | 83/84 | 70% |
| PPO + Lagrangian | 25,984 | 15.4× | 100% | 29/29 | 100% |
| **SAC + Lagrangian** | **22,723** | **17.6×** | **100%** | **29/29** | **100%** |

†Strict continuation targeting full-horizon (≥999-step) success. Peak at the harder criterion was 80% after 108,251 steps; the loop continued to 323,934 steps without improvement, confirming that brute-force iteration cannot compensate for representation choice. Distilled success (97%) evaluated at the ≥999-step criterion.

Distillation uses 50k surrogate rollout transitions augmented 5× via perturbation; no additional real MuJoCo interactions. Efficiency is relative to the 400,000-step full-order baseline.

## Exploratory Notebooks

Exploratory notebooks are collected under [`notebooks/Exploratory/`](notebooks/Exploratory/). They are useful for background, diagnostics, and future work, but are not the main report reproduction path.

| Notebook | Purpose |
|---|---|
| [`notebooks/Exploratory/simulation.ipynb`](notebooks/Exploratory/simulation.ipynb) | Earlier baseline PPO training notebook, archived in favour of `full-order-simulation.ipynb`. |
| [`notebooks/Exploratory/lqr-controller.ipynb`](notebooks/Exploratory/lqr-controller.ipynb) | LQR from analytical linearization. R²=1 by construction; motivates the Lagrangian library by showing that physics-grounded bases circumvent the polynomial R²≈0.91 ceiling. |
| [`notebooks/Exploratory/se3-forward-kinematics.ipynb`](notebooks/Exploratory/se3-forward-kinematics.ipynb) | SE(3) product-of-exponentials atoms as an alternative state representation; investigates replacing angle variables with explicit trig features to avoid Taylor-series approximation error. |
| [`notebooks/Exploratory/se3-rl-lqr-comparison.ipynb`](notebooks/Exploratory/se3-rl-lqr-comparison.ipynb) | Custom Gymnasium environment derived from SE(3) first-principles kinematics; side-by-side LQR comparison with MuJoCo to validate the analytical model. |
| [`notebooks/Exploratory/trackA_sindy_dynamics.ipynb`](notebooks/Exploratory/trackA_sindy_dynamics.ipynb) | Track-A iterative loop co-improving SINDy dynamics and a control policy without E-SINDy ensemble uncertainty; precursor to the Dyna-loop design. |
| [`notebooks/Exploratory/koopman-dynamics.ipynb`](notebooks/Exploratory/koopman-dynamics.ipynb) | Koopman operator surrogate for IDP dynamics; alternative to polynomial SINDy that lifts the state into a higher-dimensional observable space to avoid trig truncation. |
| [`notebooks/Exploratory/koopman-deep.ipynb`](notebooks/Exploratory/koopman-deep.ipynb) | Deep Koopman operator control with a learned observable embedding; extends `koopman-dynamics.ipynb` with a neural-network lifting map. |
| [`notebooks/Exploratory/sindy-hypersearch.ipynb`](notebooks/Exploratory/sindy-hypersearch.ipynb) | OLS ceiling search; computes the theoretical maximum R² achievable for a given feature set without sparsity, used to diagnose whether density is a data or representation problem. |
| [`notebooks/Exploratory/physics-informed-library.ipynb`](notebooks/Exploratory/physics-informed-library.ipynb) | Physics-informed library design investigation; evaluates whether SE(3) atoms improve polynomial SINDy distillation term density. Precursor to `sindy-rl-lagrangian.ipynb`. |
| [`notebooks/Exploratory/sindy-feature-engineering.ipynb`](notebooks/Exploratory/sindy-feature-engineering.ipynb) | Feature-library experiments for improving polynomial policy fit. |
| [`notebooks/Exploratory/sparse-policy.ipynb`](notebooks/Exploratory/sparse-policy.ipynb) | Early sparse policy distillation from the baseline PPO (tanh) expert; establishes the R²≈0.91 open-loop ceiling. |
| [`notebooks/Exploratory/sparse-policy-poly.ipynb`](notebooks/Exploratory/sparse-policy-poly.ipynb) | Sparse policy distillation using a polynomial PPO teacher rather than the tanh baseline; tests whether a polynomial teacher closes the ceiling. |
| [`notebooks/Exploratory/ppo-poly.ipynb`](notebooks/Exploratory/ppo-poly.ipynb) | PPO with a polynomial policy architecture (R²=1 by architecture); trains PPO directly using a polynomial action head to produce a distillation teacher with zero function-class gap. |
| [`notebooks/Exploratory/autodiff-policy.ipynb`](notebooks/Exploratory/autodiff-policy.ipynb) | Polynomial policy distillation via gradient descent instead of STLSQ; keeps the same polynomial structure but replaces the open-loop imitation objective with a differentiable closed-loop loss. |
| [`notebooks/Exploratory/sindy-dagger.ipynb`](notebooks/Exploratory/sindy-dagger.ipynb) | DAgger-based closed-loop dataset aggregation; targets the states the learned policy actually visits to push SINDy R² beyond the open-loop expert-trajectory ceiling. |
| [`notebooks/Exploratory/trackA_sindy_lqr_transfer.ipynb`](notebooks/Exploratory/trackA_sindy_lqr_transfer.ipynb) | Track-A SINDy/LQR transfer diagnostic. |
| [`notebooks/Exploratory/trackA_sindy_ppo_checkpoint_sweep.ipynb`](notebooks/Exploratory/trackA_sindy_ppo_checkpoint_sweep.ipynb) | Fixed-surrogate PPO checkpoint and action-penalty sweep. |
| [`notebooks/Exploratory/trackA_sindy_trig_ppo_checkpoint_sweep.ipynb`](notebooks/Exploratory/trackA_sindy_trig_ppo_checkpoint_sweep.ipynb) | Trig-feature branch of the fixed-surrogate PPO sweep. |
| [`notebooks/Exploratory/swing-up.ipynb`](notebooks/Exploratory/swing-up.ipynb) | Two-PPO swing-up and stabilization controller; one agent swings up from hanging, a second catches and holds the upright position. |
| [`notebooks/Exploratory/swing-up-v2.ipynb`](notebooks/Exploratory/swing-up-v2.ipynb) | Height-reward curriculum SAC for swing-up; single agent with a balance bonus that ramps across training phases. |

Paths inside these notebooks are resolved from the project root so they can be launched from the repo root, `notebooks/`, or `notebooks/Exploratory/`.

## Repository Structure

```text
ME_595/
├── notebooks/
│   ├── full-order-simulation.ipynb
│   ├── sindy-rl.ipynb
│   ├── sindy-rl-sac.ipynb
│   ├── sindy-rl-sac-lagrangian.ipynb
│   ├── analysis.ipynb
│   ├── inverted_double_pendulum_intro.ipynb
│   └── Exploratory/
│       └── archived exploratory notebooks
├── report/              Final report source (Markdown), figures, and PDF
├── data/                Saved trajectory datasets
├── results/             Saved trained policies, sweeps, and evaluation outputs
├── prelim-report/       Preliminary report source and PDF (archived)
├── presentation/        Slide deck source and generated HTML/PDF
├── src/                 Shared plotting and rendering utilities
├── references/          Papers
└── eng-docs/            Project notes, planning, and diagrams
```

## Environment

| Property | Value |
|---|---|
| Gym ID | `InvertedDoublePendulum-v5` |
| Observation | 9-dim: `[x, sin(θ₁), sin(θ₂), cos(θ₁), cos(θ₂), ẋ, θ̇₁, θ̇₂, constraint_force]` |
| Physical state for SINDy | 6-dim: `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` |
| Action | 1-dim cart force ∈ [−1, 1] |
| Timestep | `dt = 0.05 s` |
| Max episode | 1,000 steps |
| Termination | Tip height ≤ 1.0 m |

## References

- Zolman et al. 2025, *SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning*
- Brunton, Proctor, and Kutz 2016, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*
- Kaiser, Kutz, and Brunton 2018, *Sparse identification of nonlinear dynamics for model predictive control*
- Fasel et al. 2022, *Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit*
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*
- Haarnoja et al. 2018, *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*
