# Annotated References: ME 595 Lit Review
# Interpretable Control for Unstable Systems via SINDy-RL

Narrative outline of literature review.

## Narrative Arc

- **Paragraph 1** — Safety-critical autonomous systems demand controllers that can be audited, certified, and deployed on edge hardware; neural networks fail all three criteria [1–4].
- **Paragraph 2** — Classical control handles the linear regime but breaks on highly nonlinear unstable systems; deep RL achieves near-perfect performance but produces opaque multi-thousand-parameter policies that cannot be inspected or certified [5–7].
- **Paragraph 3** — SINDy lineage: LASSO → SINDy → SINDy-C → E-SINDy → PySINDy → Koopman comparison [8–13].
- **Paragraph 4** — Two convergent lineages feeding the project: (A) model-based RL — PILCO → PETS → DreamerV3 → SINDy-RL; (B) policy distillation — ALVINN → DAgger → augmented behavioral cloning. SINDy-RL synthesises both; this project demonstrates each technique in isolation and combined [14–20].
- **Paragraph 5** — The inverted double pendulum as proving ground: zero random-policy survival, demanding near-equilibrium data requirements, and a well-understood reward signal isolate the dynamics-modeling challenge; success validates the full technique stack for transfer to robotics, aerospace, and edge autonomy [21–22].

---

## Paragraph 1 — The Case for Interpretable Control

| # | Paper | Key idea | Why included |
|---|-------|----------|--------------|
| [1] | Rudin (2019) | Stop explaining black-box ML — use inherently interpretable models for high-stakes decisions | The foundational argument: post-hoc explanations are insufficient; the model itself must be interpretable |
| [2] | Arrieta et al. (2020) | XAI taxonomy and regulatory landscape across healthcare, finance, and autonomous systems | Establishes the breadth of the interpretability requirement; frames XAI as cross-domain, not niche |
| [3] | Saeed & Omlin (2024) | Explainability requirements in critical decision systems (EU AI Act, FDA guidance) | Regulatory mandate context — interpretability is legally required in deployed systems, not a research nicety |
| [4] | Amodei et al. (2016) | Concrete Problems in AI Safety: reward hacking, distributional shift, safe interruptibility | Grounds the interpretability motivation in formal safety requirements; sparse polynomials make all three problems tractable |

---

## Paragraph 2 — The Control Gap: Classical vs. Neural

| # | Paper | Key idea | Why included |
|---|-------|----------|--------------|
| [5] | Schulman et al. (2017) | PPO: clipped policy gradient; stable, scalable, and the de facto RL oracle for continuous control | Establishes what deep RL achieves — near-perfect balance — and what it costs: a 9,731-parameter black box |
| [6] | Haarnoja et al. (2018) | SAC: maximum-entropy off-policy RL; best sample efficiency among model-free methods | Represents the performance ceiling of model-free continuous control; makes the interpretability gap concrete by showing how much compute and data opaque methods require |
| [7] | Todorov, Erez & Tassa (2012) | MuJoCo: fast, differentiable, contact-rich physics engine for model-based control research | Establishes the high-fidelity simulation substrate; foreshadows why model-based surrogate methods that match this fidelity are valuable |

---

## Paragraph 3 — SINDy Lineage

| # | Paper | Key idea | Why included |
|---|-------|----------|--------------|
| [8] | Tibshirani (1996) | LASSO: L1-regularised regression produces sparse solutions analytically | Mathematical foundation of SINDy's sparsity-promoting regression step |
| [9] | Brunton, Proctor & Kutz (2016) | SINDy: sparse regression over a library of candidate basis functions discovers governing equations from data | The foundational paper — defines the method the entire project is built around |
| [10] | Kaiser, Kutz & Brunton (2018) | SINDy-C / SINDy-MPC: actuation extension; outperforms neural network surrogates in the low-data limit | Direct methodological predecessor to SINDy-RL and Track A's surrogate environment |
| [11] | Fasel et al. (2022) | E-SINDy: bootstrap ensemble of SINDy models; provides robustness and analytic uncertainty estimates from coefficient covariance | The specific SINDy variant used as the surrogate in SINDy-RL; its ensemble spread is the basis for uncertainty-aware model-based RL |
| [12] | de Silva et al. (2020) | PySINDy: full Python package for SINDy, including STLSQ and ensemble solvers | Implementation substrate for both tracks; the STLSQ solver used here is validated against this reference |
| [13] | Brunton et al. (2022) | Modern Koopman theory: lifts nonlinear dynamics to a higher-dimensional linear space | The main competing paradigm for nonlinear system identification; motivates why SINDy's direct sparse identification is preferred for systems with physically interpretable nonlinearities |

---

## Paragraph 4 — Model-Based RL and Policy Distillation

| # | Paper | Key idea | Why included |
|---|-------|----------|--------------|
| [14] | Deisenroth & Rasmussen (2011) | PILCO: Gaussian process surrogate enables policy learning in tens of episodes | Canonical reference for data-efficient model-based RL; establishes the lineage Track A inherits |
| [15] | Chua et al. (2018) | PETS: neural network ensemble dynamics; ensemble uncertainty guides exploration | Bridges PILCO to modern ensemble methods; the uncertainty-guided exploration analogy for Track A's active SINDy loop |
| [16] | Hafner et al. (2023) | DreamerV3: compact latent world model; policy trained entirely through imagined rollouts | State-of-the-art model-based RL alternative; Track A's surrogate-environment loop is an interpretable, explicit analogue of DreamerV3's latent "dreaming" |
| [17] | Zolman et al. (2024) | SINDy-RL: E-SINDy surrogate + RL; 10–100× sample efficiency over model-free baselines; symbolic policy distillation for embedded deployment | The primary paper — the synthesis this project validates on a demanding unstable benchmark |
| [18] | Pomerleau (1989) | ALVINN: the first demonstration that a neural network trained by imitation can control a vehicle | Historical anchor for the behavioral cloning lineage Track B extends; establishes that imitation learning works in principle |
| [19] | Ross, Gordon & Bagnell (2011) | DAgger: reduction of imitation learning to no-regret online learning; proves behavioral cloning's compounding error and corrects it via iterative oracle queries | Directly explains Track B's core failure mode (compounding error under distribution shift) and motivates the oracle-augmentation fix applied here |
| [20] | Fujimoto, van Hoof & Meger (2018) | TD3: dual-critic + delayed updates; highest asymptotic reward on continuous control benchmarks | Key model-free alternative; included to make the comparison complete — TD3 performance situates Track A and Track B results in the broader RL landscape |

---

## Paragraph 5 — Proving Ground and Broader Viability

| # | Paper | Key idea | Why included |
|---|-------|----------|--------------|
| [21] | Brockman et al. (2016) | OpenAI Gym: standardised API for RL environments including classic control and MuJoCo tasks | Establishes the benchmark ecosystem; `InvertedDoublePendulum-v5` is a Gym-standard task, making results directly comparable to the broader RL literature |
| [22] | Raffin et al. (2021) | Stable-Baselines3: peer-reviewed, reproducible implementations of PPO, SAC, TD3 | Implementation substrate for the oracle baseline; peer-reviewed status means oracle performance numbers carry external validity for comparison |

---

## Notes

- [3] (Saeed & Omlin 2024) and [4] (Amodei et al. 2016) carry the safety-certification argument in Paragraph 1. Verify that specific EU AI Act and FDA citation details match the published versions before finalising.
- [16] DreamerV3: the analogy to Track A's surrogate loop is intentional and should be made explicit in the prose — Track A is a physically interpretable DreamerV3, not just a model-based RL method.
- [18] ALVINN is used as the historical anchor for behavioral cloning; if the review needs a more recent behavioral cloning reference, substitute Ho & Ermon (2016) GAIL or Torabi et al. (2018) BCO — but ALVINN's simplicity mirrors what Track B does literally.
- [19] DAgger is the direct theoretical justification for Track B's oracle-augmentation strategy. The augmentation loop (perturb state → query NN oracle → retrain SINDy) is a simplified, non-iterative DAgger.

---

## Unused / Candidate References

| # | Paper | Key idea | Reason not used |
|---|-------|----------|-----------------|
| — | Hinton et al. (2015) | Knowledge distillation via soft targets | Policy-to-polynomial distillation in Track B is regression-based, not KL-based; DAgger is the more precise lineage |
| — | Lillicrap et al. (2016) | DDPG: continuous action RL via deterministic policy gradient | TD3 is the direct successor; citing both adds no new argument |
| — | Tassa et al. (2020) | dm_control suite: DeepMind's MuJoCo control tasks | Overlaps with Gym/MuJoCo references; not needed unless dm_control tasks are used directly |
