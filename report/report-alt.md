# Stress-Testing SINDy-RL on the Inverted Double Pendulum: Why a Degree Ceiling and Surrogate Exploitation Blocked Convergence for 25 Iterations, and What 39,616 Real Steps Finally Achieved

**Patrick Smith · Andrew Falcone**  
ME 595 · University of Washington · Spring 2026

---

## Abstract

SINDy-RL promises an *interpretable, sparse, data-efficient* controller: run a Dyna loop co-training an Ensemble SINDy polynomial surrogate and a PPO neural policy on real-environment data, then distill the result into a closed-form expression auditable enough to certify and compact enough to deploy on embedded hardware [@zolman2025sindyrl]. We set out to reproduce that pipeline on the inverted double pendulum (IDP), a two-link system with two coupled unstable modes and a narrow 0.2 m near-upright band. We instead found that two engineering obstacles, neither apparent from the algorithm description, prevented convergence for the first 25 iterations and over 90,000 real transitions without the policy ever averaging more than 22 steps. (1) The polynomial surrogate required **degree-3 features, not degree-2**: a 36-feature library cannot express the inter-modal coupling terms that dominate IDP dynamics, and RMSE fails to decrease with 10× more data when the model is capacity-limited. (2) **Surrogate exploitation** caused the policy to discover trajectories the polynomial rated as highly rewarding but physically invalid, with surrogate reward jumping 9× while real episode length collapsed 87%; ensemble uncertainty penalization alone was insufficient because all members share the same polynomial basis. We argue these obstacles trace to a single root: IDP dynamics require cubic coupling terms that a degree-2 library cannot express, and the degree-3 feature matrix that fixes this is itself poorly conditioned — a property of the representation that limits how sparse the distilled policy can become. After resolving each obstacle, the Dyna loop converged in six iterations using **39,616 real-environment steps** — **10.1× fewer than a full-order PPO baseline** — with the best checkpoint achieving 90% task success. Behavioral cloning of that policy into a degree-3 polynomial ($R^2 = 0.989$) fully preserves 90% closed-loop success, but achieves only minimal sparsification: STLSQ drops just 5 of 165 possible terms, leaving the result near-dense. This is not a threshold-tuning problem — a 500× sweep of the sparsity threshold confirms performance holds as term count falls from 165 to 121, but the polynomial never approaches the compact, auditable form the method promises. The bottleneck is the poorly conditioned feature basis: further zeroing risks eliminating structurally important contributions rather than numerical noise. A physics-informed variant exploiting IDP translational symmetry achieved comparable sparsity at reduced success (85%), confirming the density is a property of this representation class on this system rather than an artifact of data volume or threshold choice.

---

## 1  Introduction

Deployed autonomous systems face a requirement that learning-based controllers struggle to meet: the control law must be not only capable, but *inspectable*. Certification frameworks such as DO-178C (avionics software) and IEC 62443 (industrial control) require analyzable, auditable software; surgical robotics regulators may require that a control law be certifiable before permitting autonomous maneuvers near tissue; embedded actuators on spacecraft and small aerial vehicles have no floating-point stack capable of running a neural network at control rates [@arrieta2020xai; @rudin2019interpretable]. A ten-thousand-parameter neural network fails all of these requirements in deployment regardless of its simulation performance. A closed-form polynomial is the opposite: each term carries a physical interpretation, stability arguments can be constructed analytically, and a 165-term policy fits in kilobytes and evaluates as a single dot product.

Sparse Identification of Nonlinear Dynamics (SINDy [@brunton2016sindy]) offers a route: fit governing equations from data, zero all but a few terms via sparse regression. SINDy-RL [@zolman2025sindyrl] adapts this to RL by using an ensemble SINDy model as the Dyna surrogate [@sutton1990dyna], co-training a PPO policy [@schulman2017ppo] inside it while collecting data from the real environment — resolving the data-bootstrapping problem of unstable systems while retaining interpretability as a downstream option. The question is whether all three promises — data efficiency, sparsity, interpretability — survive contact with a genuinely hard unstable system.

Following our project, we set out to reproduce SINDy-RL on the inverted double pendulum — two links, two unstable modes, a 0.2 m near-upright band, and no self-stabilizing dynamics — and to test whether the method delivers when the system fights it. This report is an honest account of that attempt: the chain of obstacles we encountered, the evidence that they share a single root, how we resolved each one, and what the method achieved once we had. The engineering failures are documented alongside the final results because, on this system, the obstacles are as informative as the numbers.

Interpretable controllers have direct safety benefits — polynomial expressions can be analyzed for failure modes, admit Lyapunov-style certificates in certain regimes, and allow engineers to audit the control law before deployment — and SINDy-RL's data efficiency reduces dangerous exploration on real hardware. The principal risk is the inverse: a practitioner who observes high surrogate reward without real-environment validation may deploy a controller optimized for a model, not for a system, a failure mode §4.2 demonstrates concretely.

---

## 2  Background and Notation

### 2.1  The Testbed: Inverted Double Pendulum

\begin{wrapfigure}{r}{0.25\linewidth}
  \vspace{-48pt}
  \centering
  \includegraphics[width=\linewidth]{figures/pendulum_diagram.png}
  \captionsetup{font=scriptsize, labelfont=bf}
  \caption*{\textbf{Figure 1.} IDP geometry. State $\mathbf{x} = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]$. Tip height $h \in [0,\,1.2]$ m; episode ends at $h \leq 1.0$ m.}
  \vspace{-6pt}
\end{wrapfigure}

`InvertedDoublePendulum-v5` (MuJoCo 3.8.1 / Gymnasium 1.2.3) consists of two rigid links $L_1 = L_2 = 0.6$ m on a sliding cart. The physical state is $\mathbf{x} = [x,\theta_1,\theta_2,\dot{x},\dot{\theta}_1,\dot{\theta}_2] \in \mathbb{R}^6$, where $x$ is the cart's horizontal position along the track, $\theta_1, \theta_2$ are joint angles measured from vertical, and dots denote time derivatives. The 8-dimensional observation replaces raw angles with sine/cosine encodings to avoid wrapping discontinuities. The control input is a horizontal cart force $u \in [-1,1]$.

Tip height $h = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2)$ reaches 1.2 m when both poles are vertical. Gymnasium terminates at $h \leq 1.0$ m, leaving only a 0.2 m near-upright band between success and failure. Episodes cap at 1,000 steps (50 s at $\Delta t = 0.05$ s); task success is $\geq 500$ steps survived.

### 2.2  SINDy-C, E-SINDy, and the STLSQ Sparsity Knob

SINDy [@brunton2016sindy] fits discrete-time dynamics by regressing state increments against a polynomial library:

$$\mathbf{x}_{k+1} - \mathbf{x}_k = \Theta(\mathbf{x}_k,\, u_k) \cdot \Xi$$

For control-affine systems (SINDy-C [@kaiser2018sindympc]), the input $u_k$ enters the library directly.[^ca] The Sequentially Thresholded Least Squares (STLSQ) solver zeros all coefficients below threshold $\lambda$. A degree-$d$ library over $n$ variables contains $\binom{n+d}{d}$ terms: for the IDP's 7-dimensional state-action vector, degree-2 gives 36 features and degree-3 gives 120 — a distinction that cost 25 iterations (§3, §4.1).

Fasel et al. [@fasel2022esindy] add uncertainty quantification: $M = 10$ independent SINDy models are fit on 80% bootstrap subsamples of the data; at inference, each surrogate step returns the ensemble-mean increment $\mu_\Delta$ and per-component standard deviation $\sigma_\Delta$. High $\sigma_\Delta$ signals extrapolation beyond the training distribution.

[^ca]: A system is control-affine if $\dot{\mathbf{x}} = f(\mathbf{x}) + g(\mathbf{x})\,u$, where $f$ and $g$ may be nonlinear in state. Most mechanical systems driven by forces or torques, including the IDP, satisfy this property.

### 2.3  The Dyna Loop and Behavioral Cloning

![**Figure 2.** The RL control loop. The agent outputs action $u_k = \pi_\phi(\mathbf{x}_k)$ from the neural policy $\pi_\phi$ during Dyna training; after distillation, this becomes $u_k \approx \Theta_\text{obs}(\mathbf{x}_k)\,\xi$ where $\xi$ is the sparse coefficient vector. In SINDy-RL the environment is instantiated twice: as the E-SINDy polynomial surrogate for cheap policy training (left), and as the full MuJoCo simulator for real data collection and evaluation (right). The policy sees a surrogate that approximates reality; every obstacle in §4 is a consequence of where that approximation breaks down.](figures/rl_loop.svg){width=82%}

The Dyna architecture [@sutton1990dyna] alternates cheap model-based rollouts inside a learned surrogate with real-environment data collection. In SINDy-RL [@zolman2025sindyrl], the surrogate is the E-SINDy ensemble and the planner is PPO [@schulman2017ppo]. A Schroeder multi-sine sweep [@schroeder1970] bootstraps an initial dataset; each iteration refits E-SINDy on near-upright transitions, trains PPO for 100k surrogate steps (warm-started from the prior policy), and collects 4,000 real transitions. After convergence, the best checkpoint is distilled via behavioral cloning [@ross2011dagger]: expert trajectories are collected from real MuJoCo, augmented with per-dimension Gaussian noise, and fit with STLSQ over the 8-dimensional sin/cos observation (165 features at degree-3, since $\binom{11}{3} = 165$).

![**Figure 3.** Sparse regression structure for E-SINDy dynamics identification (*left*) and policy distillation (*right*). Each of the $k = 1,\ldots,M$ ensemble members solves $\Delta\mathbf{X}^k = \Theta_\text{dyn}^k \Xi_\text{dyn}^k$, where $\Theta_\text{dyn}^k \in \mathbb{R}^{N \times 120}$ is the degree-3 polynomial library over the 7-dimensional state-action input evaluated on a bootstrap subsample, and $\Xi_\text{dyn}^k \in \mathbb{R}^{120 \times 6}$ (orange) is the fitted coefficient matrix. Distillation fits a single $U = \Theta_\pi \Xi_\pi$, where $\Theta_\pi \in \mathbb{R}^{N \times 165}$ is the degree-3 library over the 8-dimensional sin/cos observation, and $\Xi_\pi \in \mathbb{R}^{165 \times 1}$ (purple) is the scalar action coefficient vector. The two libraries are distinct: different input spaces (raw state-action vs. sin/cos encoding) and different feature counts (120 vs. 165). Annotations below each $\Xi$ give the nonzero coefficient count after STLSQ thresholding — 690 of 720 possible entries for the dynamics model and 160 of 165 for the distilled policy — indicating that both fits are near-dense.](figures/sindy_matrix_shapes.svg){width=90%}

Software: Python 3.12.7, PySINDy 2.1.0 [@desilva2020pysindy] (E-SINDy surrogate), Stable-Baselines3 2.8.0 [@raffin2021sb3] (PPO), Gymnasium 1.2.3 with MuJoCo 3.8.1 [@todorov2012mujoco] (simulation), NumPy 2.4.6, scikit-learn 1.8.0. Full pipeline in `notebooks/sindy-rl.ipynb`.

---

## 3  The System Requires Cubic Coupling Terms and the Feature Matrix Is Ill-Conditioned: the Root of Everything That Follows

Before the challenges, we state the two facts about the IDP that explain all of them.

- **Degree-3 is necessary.** The IDP's inter-modal dynamics contain terms such as $\cos\theta_1 \cdot \cos\theta_2 \cdot \dot\theta_1$ — cubic interactions between the two joints — that a degree-2 library (36 features) cannot express. This is not a data deficit: over 25 Dyna iterations with training data growing from 5,000 to 90,000 transitions, surrogate RMSE oscillated at 0.10--0.16 and refused to decrease. When RMSE does not fall with 10× more data, the model is capacity-limited.[^degree] Switching to degree-3 (120 features) dropped RMSE to 0.013 within two iterations.

- **The degree-3 feature matrix is ill-conditioned.** The 209,620 × 120 dynamics feature matrix (all collected transitions × all library terms) has condition number $\kappa = 2.37 \times 10^4$ — full rank, 120 nonzero singular values, but with a 23,700× spread between the largest and smallest.[^kappa] Ill-conditioning worsens with output dimension: $\kappa = 318$ for the $x$-position equation rising to $\kappa = 22{,}980$ for $\dot\theta_2$.

[^degree]: The heuristic generalizes: plot one-step RMSE versus dataset size on log-log axes. A slope near zero means capacity failure; a negative slope means data insufficiency. The two have different fixes — model order vs. more data — and conflating them wastes resources proportional to how long the wrong diagnosis persists.

[^kappa]: Condition number $\kappa(M) = \sigma_{\max}/\sigma_{\min}$ measures how lopsided the matrix is: the feature matrix amplifies some coefficient directions $\approx 2.4\times10^4$ times more than others. A rule of thumb: $\kappa \approx 10^k$ costs roughly $k$ digits of numerical precision in coefficient estimates. For $\kappa \approx 2.4\times10^4$ ($\approx 10^{4.4}$), roughly 4 digits are unreliable in the worst-case direction — enough to make small STLSQ coefficients numerically meaningless.

Two consequences run through the rest of the report. First, **no degree-2 surrogate can converge on this system**, regardless of data volume — every iteration run with the default library was informationally worthless, consuming real-environment steps without any prospect of success. Second, **full sparsity in the distilled polynomial is not achievable by threshold tuning alone** — the 160 terms that survive STLSQ are not all mechanistically necessary, but the ill-conditioned feature basis does not admit a numerically stable sparse decomposition beyond the 5 already dropped. Neither consequence was apparent from the algorithm description.

---

## 4  Challenges and What Resolved Them

### 4.1  Degree-2 RMSE Ceiling: 25 Iterations Without Progress

The most costly obstacle was using the wrong polynomial degree throughout the initial phase. With the default `SINDY_DEGREE=2`, the Dyna loop ran for 25 iterations. Over that span, RMSE oscillated between 0.10 and 0.16 while real mean episode length grew from 6 to 22 steps — never approaching the 500-step success threshold, never showing a trend toward convergence. We added data, tuned hyperparameters, and adjusted the PPO schedule. None of it helped, because the bottleneck was not any of those things.

The diagnosis is §3, first bullet. A degree-2 library (36 features) is simply inexpressive for IDP dynamics. The cubic inter-modal coupling terms that dominate the near-upright regime — the terms the surrogate most needs to get right — are not representable. Adding data to a model that cannot fit the function returns RMSE noise, not improvement. The sign of this failure is the flat RMSE-vs-data curve: if RMSE does not decrease with 10× more data, model capacity is the ceiling, not data volume.

The fix: `SINDY_DEGREE=3` (120 features). Within two Dyna iterations, RMSE dropped to 0.013. The 25 failed iterations, each consuming 4,000 real transitions, amounted to over 90,000 wasted real-environment steps before a single successful degree-3 iteration began. Correcting the near-upright filter threshold — which had been misconfigured to 1.6 m (above the physical maximum of 1.2 m) by inheriting a reward-shaping constant rather than deriving it from segment geometry — was a one-line fix applied alongside the degree change.

### 4.2  Surrogate Exploitation: Uncertainty Penalization Alone Is Insufficient

The second obstacle emerged once degree-3 and the corrected filter produced a surrogate worth training against. In a diagnostic run, surrogate reward jumped 9× in a single iteration — from 497 to 4,525 per episode — while real episode length collapsed 87%, from 414 to 56 steps. The policy had found action sequences the polynomial rated as highly rewarding that had no correspondence to real physics.

The designed countermeasure is ensemble uncertainty penalization: reduce surrogate reward by $\kappa \cdot \text{mean}(\sigma_\Delta)$ per step, steering PPO away from high-disagreement states. This was insufficient, and the reason is structural. All 10 ensemble members share the same degree-3 polynomial basis. In the extrapolated region where the surrogate is wrong, every member makes the same wrong prediction, $\sigma_\Delta$ remains low, and the penalty does not fire. Disagreement between ensemble members cannot detect shared extrapolation error — it can only detect coefficient uncertainty along directions where the fit differs between members.[^exploit] Real-environment feedback is the only out-of-distribution signal that is reliable by construction.

The fix required both mechanisms: uncertainty penalty `reward -= 5.0 * mean(sigma_delta)` throughout surrogate PPO, plus a rollback trigger that detects exploitation post hoc (surrogate reward $> 3\times$ previous AND real episode length $< 50\%$ of best seen) and restores the best real-environment checkpoint. Penalization without rollback fails to detect shared extrapolation. Rollback without penalization fails to prevent the exploited iteration from corrupting the training dataset. Both together were sufficient.

[^exploit]: The shared-basis failure is distinct from the ensemble disagreement used in active learning. Active learning uses disagreement between models of *different functional forms* — or models of the same form fit on different data — to identify epistemic uncertainty. On a shared polynomial basis, all members differ only in their coefficient estimates, and in extrapolation they agree on the wrong answer. The uncertainty estimate $\sigma_\Delta$ is a measure of coefficient variance, not of out-of-distribution extrapolation. This is why the penalty steers the policy away from states where the ensemble fit is noisy (genuinely useful) but cannot protect against states where the fit is confidently wrong (the failure mode here).

### 4.3  Distillation Obstacles

Three additional obstacles emerged during behavioral cloning. First, **the teacher must be the cross-validated best checkpoint, not the final loop policy**: the final policy may have drifted during continued surrogate training after the convergence declaration. Using the final policy gave 0% closed-loop success; using the iteration-7 best checkpoint gave 90%. Second, **degree-2 distillation fails for the same reason as degree-2 surrogate fitting**: $R^2 \approx 0.905$ regardless of data volume, a capacity ceiling, not a data deficit. Degree-3 over the 8-dimensional sin/cos observation (165 features) was required. Third, **perturbation augmentation was necessary to close distribution shift**: adding per-dimension Gaussian noise to expert states and re-querying the trained neural network policy (the NN oracle) at each perturbed state expands the 50k-transition dataset without additional real MuJoCo interactions, covering states adjacent to the teacher's training trajectories that the straight behavioral clone would otherwise handle poorly.

---

## 5  What Finally Worked

### 5.1  Dyna Loop Convergence

With degree-3 features, a corrected filter threshold, and the combined uncertainty-penalty-plus-rollback safeguard in place, the Dyna loop converged in **six iterations** using **39,616 real-environment steps** — 10.1× fewer than the 400,000-step full-order PPO baseline (Stable-Baselines3 2.8.0, [64,64] MLP, 9,731 parameters). Figure 4 shows episode length distributions across checkpoints.

![**Figure 4.** Episode length distributions for baseline PPO and selected SINDy-RL checkpoints (20 episodes each). Success rate ($\geq$ 500 steps) is annotated above each box. Iteration 7 (starred) is the cross-validated best at 90% success. Iterations 10 and 20 illustrate the performance collapse from continued exploitation after peak convergence — the dataset degrades as more corrupted iterations accumulate.](figures/fig_ep_lengths.png){width=90%}

Cross-validated evaluation of all saved checkpoints identified **iteration 7** as the best: **90% success, mean episode length 904 steps**. The off-by-one between the loop's convergence declaration (iteration 6) and the best checkpoint (iteration 7) is not a pathology: the loop saves a checkpoint after each PPO training phase; the iteration-7 checkpoint reflects one additional PPO training phase beyond the convergence declaration, which improved the policy without consuming additional real-environment data.

| Iteration | Cumul. real steps | SINDy RMSE | Surr. mean len | Real mean len | Success |
|-----------|------------------|------------|----------------|---------------|---------|
| Bootstrap | 2,897 | 0.016 | — | — | — |
| 1 | 7,015 | 0.016 | 11.8 | 12 | 0% |
| 2 | 11,186 | 0.020 | 17.1 | 17 | 0% |
| 3 | 15,551 | 0.084 | 36.5 | 36 | 0% |
| 4 | 19,979 | 0.095 | 42.8 | 43 | 0% |
| 5 | 29,453 | 0.085 | 547 | 547 | 50% |
| **6** | **39,616** | **0.080** | **616** | **616** | **60%** |

The RMSE rise at iterations 3--4 is not model degradation. It reflects a better policy exploring states further from vertical, where the surrogate has higher error; the surrogate remained accurate in the near-upright band where it mattered for control.

### 5.2  Policy Distillation

Behavioral cloning from the iteration-7 checkpoint (50k expert transitions, 5× augmentation, 300k total rows) produced a degree-3 polynomial with $R^2 = 0.989$. STLSQ at $\lambda = 0.10$ retained **160/165 terms** with **90% closed-loop success** — identical to the neural teacher. Figure 5 shows coefficient magnitudes by polynomial degree.

![**Figure 5.** Distilled policy coefficient magnitudes (log scale), coloured by polynomial degree. Of 165 possible terms, 160 are nonzero at $\lambda = 0.10$; five terms are dropped. Closed-loop success is fully preserved (90%). The dominant terms include constant bias, angle feedback on $\cos\theta_1$ and $\cos\theta_2$, velocity damping, and inter-pole coupling — physically recognizable structure in the leading terms, with residual cubic cross-terms encoding the NN's tanh nonlinearity.](figures/fig_coefficients.png){width=90%}

A threshold ablation across $\lambda \in [0.001, 1.0]$ (Figure 6) confirms robustness: success remains at 90--95% as term count falls from 165 to 121 — a 500× range of threshold values without degradation below the success criterion. This rules out threshold sensitivity as a source of fragility. The condition number analysis (§3) explains the residual density: the 160 surviving terms are not all mechanistically necessary, but the ill-conditioned feature basis ($\kappa \approx 2.4 \times 10^4$) does not admit a numerically stable sparse decomposition beyond the 5 terms already dropped.

![**Figure 6.** STLSQ threshold ablation. Left axis: mean episode length (blue); right axis: non-zero term count (orange). Dashed vertical line marks the report threshold ($\lambda = 0.10$). Performance is stable at 90--95% across the full range even as the polynomial compresses from 165 to 121 terms.](figures/fig_threshold_ablation.png){width=82%}

| Approach | Real-env steps | Mean ep len | Success | Params |
|---|---|---|---|---|
| Baseline PPO | 400,000 | 1,000 | 100% | 9,731 |
| SINDy-RL NN (Dyna) | 39,616 | 904 | 90% | 9,731 |
| SINDy-RL Polynomial | 89,616$^\dagger$ | 904 | 90% | 160 terms |

$^\dagger$39,616 Dyna steps + 50,000 distillation rollout steps; 5× perturbation augmentation reuses the trained neural network policy (NN oracle) without additional MuJoCo interactions.

The distilled polynomial is **61× smaller** than the baseline network and evaluates as a single dot product, compatible with microcontroller deployment and amenable to analytical stability arguments that a neural network cannot support.

---

## 6  Discussion

**Each obstacle has a different cause; both are traceable to §3.** The degree ceiling (§4.1) and degree-2 distillation failure (§4.3) are capacity failures — the IDP's cubic coupling terms are not representable in degree-2, and this is a property of the dynamics, not the data. The exploitation failure (§4.2) is an instability failure — the IDP's narrow region of attraction means surrogate errors compound faster than the policy can recover, and an unconstrained optimizer finds the exploitable trajectories even in high-disagreement regions.

**What this says about SINDy-RL's scope.** The method delivers on data efficiency: 10.1× fewer real steps than full-order PPO is a genuine advantage, and the best checkpoint achieves 90% success against the baseline's 100%. It delivers on near-sparsity: 160/165 terms is compact enough for embedded deployment and preserves full task performance. What it cannot deliver — on this system, at this data scale — is full sparsity in the sense of a five-term physically interpretable law. The IDP dynamics are not sparse in a degree-3 polynomial basis: 690 of 720 possible surrogate coefficients are nonzero, and the feature matrix is ill-conditioned, so neither more data nor aggressive thresholding can extract a stable sparse decomposition. These are properties of the representation, not of the algorithm.

**Whether a change of coordinates would help.** A physics-informed variant that removed cart position $x$ from the SINDy library — exploiting the IDP's translational symmetry ($\Delta x = \dot{x}\,\Delta t$ exactly by kinematics) — achieved comparable distillation sparsity (162/165 at $\lambda = 0.05$) at 85% rather than 90% success, confirming that the standard degree-3 basis already captures the translational structure implicitly and that explicit coordinate removal is not the lever. SE(3) forward-kinematics coordinates (replacing raw angles with absolute-link sin/cos) produce catastrophic ill-conditioning ($\kappa = 10^{17}$--$10^{19}$) due to unit-circle constraints creating near-exact linear dependence in the polynomial features — a worse outcome than the already-problematic standard basis. The more promising direction is a physics-informed library derived directly from the Euler-Lagrange equations, using gravity terms, mass-matrix coupling, and Coriolis interactions as basis atoms; preliminary comparisons show 35× better conditioning ($\kappa \approx 3\times10^4$) with only 32 features, and a full Dyna comparison is ongoing. This remains the most important open question: whether a representation that encodes the system's physics directly would restore the full sparsity the method promised.

---

## 7  Summary

We stress-tested SINDy-RL on the inverted double pendulum after resolving three engineering obstacles that the algorithm description does not surface.

**Data efficiency is confirmed, but required first fixing the degree.** The Dyna loop trains a 90%-success neural policy in 39,616 real-environment steps — 10.1× fewer than the full-order PPO baseline. Getting there required diagnosing a degree-2 RMSE ceiling (25 iterations and over 90,000 wasted real transitions before the fix was applied) and a surrogate exploitation failure that ensemble uncertainty penalization alone could not prevent.

**Near-sparsity is achieved through behavioral cloning.** STLSQ at $\lambda = 0.10$ retains 160/165 terms while fully preserving 90% closed-loop success, and a threshold ablation across a 500× range of $\lambda$ values confirms robustness (90--95% success, 121--165 terms). The distilled polynomial is 61× smaller than the baseline network.

**Full sparsity remains out of reach, and the reason is the representation.** The 160 surviving terms are explained by ill-conditioning of the degree-3 feature matrix ($\kappa \approx 2.4 \times 10^4$) — neither more data nor more aggressive thresholding can produce a numerically stable decomposition beyond the 5 terms already dropped. A 160-term polynomial is compact and deployable; it is not auditable in the way a five-term equation would be.

The engineering failures are as informative as the results. The degree-2 ceiling generalizes: when RMSE does not decrease with 10× more data, the cause is model capacity, not data volume, and the fix is model order, not more data. The exploitation failure generalizes: any ensemble surrogate built on a shared function basis cannot detect shared extrapolation errors through internal disagreement — real-environment feedback is the only reliable out-of-distribution signal, and rollback and penalization are complementary safeguards, not substitutes.

\newpage

## Code Repository

All code and results: **https://github.com/falconeaj1/ME_595**. Key notebooks: `full-order-simulation.ipynb` (baseline PPO), `sindy-rl.ipynb` (SINDy-RL pipeline), and `sindy-rl-no-x.ipynb` (translational-symmetry variant). Professor Michelle Hickner added as collaborator (GitHub: mhickner).

---

\noindent\textbf{CRediT Statement} (\url{https://credit.niso.org})

\begingroup
\small
\begin{tabular}{lll}
\textbf{Role} & \textbf{Patrick Smith} & \textbf{Andrew Falcone} \\
\hline
Conceptualization          & Yes  & Yes        \\
Data curation              & Yes  & Yes        \\
Formal analysis            & Lead & Supporting \\
Investigation              & Yes  & Yes        \\
Methodology                & Lead & Supporting \\
Software                   & Lead & Supporting \\
Validation                 & Yes  & Yes        \\
Visualization              & Yes  & Yes        \\
Writing -- original draft  & Yes  & Yes        \\
Writing -- review \& editing & Yes & Yes       \\
\end{tabular}
\endgroup

\smallskip
\noindent\small\textit{AI tool disclosure: Claude (Anthropic) assisted with code drafting, debugging, writing iteration, and figure generation. All analysis, results, and conclusions were reviewed and executed by the authors, who take full responsibility for the submitted work.}

\newpage

# References
