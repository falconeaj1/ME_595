# Stress-Testing SINDy-RL: Data-Efficient Interpretable Control on the Inverted Double Pendulum

**Patrick Smith · Andrew Falcone**  
ME 595 · University of Washington · Spring 2026

---

## Abstract

Safety-critical autonomous systems require controllers that can be formally verified, audited, and deployed
on resource-constrained hardware, properties that large neural networks cannot satisfy. SINDy-RL [@zolman2025sindyrl] addresses this by co-training a sparse surrogate world model and a neural policy in a Dyna loop, then distilling the result into a closed-form expression. The algorithm treats the feature library Θ and the policy optimizer A as practitioner inputs. We stress-test this framework on the inverted double pendulum (IDP), a two-link system with two coupled unstable modes and only a 0.2 m near-upright band, asking whether it delivers its three promised goals: *data economy* via the SINDy surrogate, a *reduced-order policy* via distillation, and *interpretability* via a sparse, auditable controller. After resolving two non-obvious engineering obstacles, we evaluate four library–optimizer combinations spanning the design space. A degree-3 polynomial library with PPO achieves data economy (66,277 real steps, **6.0× fewer** than a full-order baseline) and a reduced-order policy (82-term polynomial, 119× smaller than the baseline neural network), but fails at interpretability: STLSQ retains 82 of 84 terms regardless of threshold, a consequence of ill-conditioning in the polynomial basis ($\kappa \approx 2.4 \times 10^4$), not a data or tuning deficiency. Replacing the polynomial library with 32 atoms derived from the IDP Euler-Lagrange equations, combined with Soft Actor-Critic (SAC) as the policy optimizer, converges in 22,723 real steps (**17.6× fewer** than baseline) with 100% task success and a 29-term distilled controller every term of which is a physically meaningful quantity. **SINDy-RL passes the stress test.** With appropriate choices of Θ and A, the algorithm simultaneously delivers all three goals; the polynomial density is a finding about representation selection, not a limitation of the framework.

---

## 1  Introduction

### 1.1  The Inspectability Problem

A deployed autonomous system faces a constraint that learning-based controllers routinely ignore: the control law must be not only capable, but *inspectable*. Certification frameworks such as DO-178C (avionics) and IEC 62443 (industrial control) require analyzable, auditable software [@arrieta2020xai]. Surgical robotics regulators may require formal certification of the control law before permitting autonomous maneuvers near tissue. Embedded actuators on spacecraft and small aerial vehicles lack the floating-point stack needed to evaluate a neural network at control rates. Compact microcontrollers managing industrial equipment cannot load a 10,000-parameter model into flash storage, much less run it in real time.

A ten-thousand-parameter neural network fails all of these requirements regardless of its simulation performance. A sparse closed-form polynomial is the opposite: each retained term carries a physical interpretation; Lyapunov-style stability arguments can be constructed analytically; and an 82-term polynomial policy fits in kilobytes and evaluates as a single dot product [@rudin2019interpretable]. This gap motivates a growing body of work on *inherently interpretable* controllers, models whose structure is transparent by construction, not approximated after the fact.

### 1.2  SINDy-RL: Co-Training Dynamics and Policy

Sparse Identification of Nonlinear Dynamics (SINDy [@brunton2016sindy]) identifies governing equations from data: fit a library of candidate functions against the observed dynamics, then zero all but a few terms via sparse regression. The result is compact and physically grounded. The difficulty is that for an unstable equilibrium, such as an inverted pendulum, the near-upright transitions that SINDy needs are entirely unvisited without a controller that does not yet exist.

Zolman et al. [@zolman2025sindyrl] resolve this with SINDy-RL: use an ensemble SINDy model as the surrogate in a Dyna architecture [@sutton1990dyna], co-training a neural Reinforcement Learning (RL) policy inside the surrogate while collecting data from the real environment. Each iteration improves both the model and the policy, bootstrapping the unstable system into the near-upright regime. Importantly, Algorithm 1 in [@zolman2025sindyrl] treats the feature library Θ and the RL optimizer A as explicit practitioner inputs, parameterizing the design space rather than fixing it.

### 1.3  Contributions

We stress-test SINDy-RL on the inverted double pendulum (two links, two unstable modes, 0.2 m near-upright tolerance) and characterize its performance across four library–optimizer combinations. Our contributions are:

1. **SINDy-RL passes the stress test.** The SAC+Lagrangian combination achieves all three goals simultaneously: 22,723 real steps (17.6×), a 29-term distilled controller (336× smaller than the baseline neural network), and 100% task success with interpretability by construction.
2. **Data economy is confirmed across all variants.** Every combination substantially outperforms the full-order baseline, from 6.0× (PPO+polynomial) to 17.6× (SAC+Lagrangian).
3. **Interpretability requires a physics-informed library.** The degree-3 polynomial basis is ill-conditioned ($\kappa \approx 2.4 \times 10^4$); Sequentially Thresholded Least Squares (STLSQ) retains 82 of 84 terms at every threshold. This is a property of the polynomial representation on this system, not a failure of SINDy-RL.
4. **Two non-obvious engineering obstacles**, a degree-2 root-mean-square-error (RMSE) ceiling and surrogate exploitation via a shared polynomial basis, are diagnosed and resolved with safeguards applicable to other unstable benchmarks.

### 1.4  Ethics and Safety Considerations

**Benefits.** Interpretable controllers have direct safety advantages: closed-form polynomial expressions can be analyzed for failure modes before deployment, admit Lyapunov-style stability certificates, and allow engineers to audit every term of the control law. SINDy-RL's data efficiency also reduces wear and risk during training; fewer physical interactions means fewer dangerous exploration failures, which matters for systems operating near humans or expensive hardware.

**Risks.** *Surrogate exploitation*, where a policy finds action sequences the polynomial rates as highly rewarding but that do not correspond to real physics, can lead to overconfident deployment decisions if a practitioner observes high surrogate reward without real-environment validation. *Incomplete sparsity* creates a risk that the "interpretability" framing overstates what an engineer can actually verify: an 82-term polynomial is more compact than a neural network, but it is not auditable in the way a five-term equation would be. *Distribution shift* between the training regime and deployment conditions can cause polynomial controllers to fail catastrophically in states not covered by training data.

Any deployment in a safety-critical application should include real-hardware validation across the full intended operating envelope, formal analysis of the polynomial's behavior at boundary conditions, and a conservative fallback for states outside the validated region.

---

## 2  Background

### 2.1  Testbed: The Inverted Double Pendulum

\begin{wrapfigure}{r}{0.25\linewidth}
  \vspace{-48pt}
  \centering
  \includegraphics[width=\linewidth]{figures/pendulum_diagram.png}
  \captionsetup{font=scriptsize, labelfont=bf}
  \caption*{\textbf{Figure 1.} IDP geometry. State $\mathbf{x} = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]$, where $x$ is cart position; $h \in [0,\,1.2]$ m. Episode terminates at $h \leq 1.0$ m.}
  \vspace{-6pt}
\end{wrapfigure}

`InvertedDoublePendulum-v5` (MuJoCo 3.8.1 / Gymnasium 1.2.3) has two rigid links $L_1 = L_2 = 0.6$ m on a sliding cart. The physical state is $\mathbf{x} = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2] \in \mathbb{R}^6$: $x$ is the cart's horizontal position along the track, $\theta_1, \theta_2$ are link angles measured from vertical, and dots denote time derivatives. The 9-dimensional observation replaces raw angles with sine/cosine encodings to avoid wrapping discontinuities. The single control input is a horizontal cart force $u \in [-1, 1]$.

Tip height $h = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2)$ reaches 1.2 m when both links are vertical. Gymnasium terminates an episode when $h \leq 1.0$ m, leaving only a 0.2 m near-upright band between success and failure. The per-step reward is:
$$r_k = 10 - (h_k - 2)^2 - 0.01\,x_\text{tip}^2 - \varepsilon\|\dot{\boldsymbol{\theta}}\|^2$$
The alive bonus ($\approx 10$/step) dominates when the system stays upright. Episodes are capped at 1,000 steps (50 s at $\Delta t = 0.05$ s). Task success is defined as surviving at least 500 steps.

### 2.2  SINDy-C: Sparse Dynamics Identification with Control

SINDy [@brunton2016sindy] fits discrete-time dynamics by regressing state increments against a library of candidate functions:
$$\mathbf{x}_{k+1} - \mathbf{x}_k = \Theta(\mathbf{x}_k, u_k) \cdot \Xi$$
where $\Theta(\mathbf{x}_k, u_k) \in \mathbb{R}^{1 \times p}$ is a row of library features evaluated at the current state-action pair and $\Xi \in \mathbb{R}^{p \times n}$ is the sparse coefficient matrix. For control-affine systems (SINDy-C [@kaiser2018sindympc]), the input $u_k$ enters the library directly.[^ca] The STLSQ solver zeros all coefficients below threshold $\lambda$, promoting sparsity in $\Xi$. A degree-$d$ polynomial library over an $n$-variable input has $\binom{n+d}{d}$ features: for the IDP's 7-dimensional state-action vector, degree-2 gives 36 features and degree-3 gives 120.

[^ca]: A system is control-affine if $\dot{\mathbf{x}} = f(\mathbf{x}) + g(\mathbf{x})\,u$ where $f$ and $g$ may be nonlinear in state. Forces and torques satisfy this, including the IDP.

### 2.3  E-SINDy: Ensemble Uncertainty Quantification

A single SINDy model provides a point estimate with no uncertainty. Fasel et al. [@fasel2022esindy] address this with Ensemble SINDy (E-SINDy): fit $M = 10$ independent SINDy models on 80% bootstrap subsamples, then report the ensemble mean $\mu_\Delta$ and per-component standard deviation $\sigma_\Delta$ at inference. High $\sigma_\Delta$ flags extrapolation beyond the training distribution. Following Zolman et al. [@zolman2025sindyrl], we penalize surrogate reward by $\kappa \cdot \text{mean}(\sigma_\Delta)$ per step ($\kappa = 5.0$), steering the policy away from high-uncertainty regions.

### 2.4  Dyna-Style Learning Loop and Behavioral Cloning

The Dyna architecture [@sutton1990dyna] alternates cheap model-based rollouts inside a learned surrogate with real-environment data collection, so each iteration improves both the surrogate and the policy without expensive real interactions. In SINDy-RL [@zolman2025sindyrl], the surrogate is the E-SINDy ensemble and the planner is a neural policy trained by RL (Proximal Policy Optimization, PPO [@schulman2017ppo], or SAC). Figure 2 shows the RL control loop: during Dyna training, the environment is the E-SINDy surrogate; for data collection and evaluation, it is real MuJoCo.

![**Figure 2.** RL control loop. During Dyna training the environment is the E-SINDy polynomial surrogate; for data collection and evaluation it is the full MuJoCo simulator. The policy $\pi_\phi$ maps observation $\mathbf{x}_k$ to action $u_k = \pi_\phi(\mathbf{x}_k)$.](figures/rl_loop.svg){width=80%}

Once the Dyna loop converges and a neural policy checkpoint is selected, *behavioral cloning*, referred to throughout as *distillation*, compresses the neural policy into a sparse closed-form expression. A large dataset of (observation, action) pairs is collected by running the neural policy (hereafter the *NN oracle*) in the surrogate, augmented via perturbation [@ross2011dagger], and sparse regression is used to fit a polynomial controller $u \approx \Theta_\text{obs}(\mathbf{x})\,\xi$:
$$\min_{\xi}\;\bigl\|\Theta_\text{obs}(X)\,\xi - U^*\bigr\|_2 \quad \text{subject to STLSQ sparsity}$$
where $X$ is a matrix of observations, $U^* \in \mathbb{R}^N$ are the corresponding NN oracle actions, $\Theta_\text{obs}$ is the degree-3 polynomial library over the 6-dimensional raw physical state $\mathbf{x} = [x, \theta_1, \theta_2, \dot{x}, \dot\theta_1, \dot\theta_2]$ (84 features, $\binom{9}{3} = 84$), and $\xi$ is the learned scalar coefficient vector. The distilled polynomial is compact, evaluable as a dot product, and, if sparse, admits physical interpretation of each retained term. Note that two *distinct* libraries are involved: the dynamics library the E-SINDy ensemble fits and the distillation library the policy is cloned into differ in both their input space and their feature count, as Figure 3 makes explicit.

![**Figure 3.** Sparse regression structure for E-SINDy dynamics identification (*left*) and policy distillation (*right*). Each of the $k = 1,\ldots,M$ ensemble members solves $\Delta\mathbf{X}^k = \Theta_\text{dyn}^k \Xi_\text{dyn}^k$, where $\Theta_\text{dyn}^k \in \mathbb{R}^{N \times 120}$ is the degree-3 polynomial library over the 7-dimensional state-action input evaluated on a bootstrap subsample, and $\Xi_\text{dyn}^k \in \mathbb{R}^{120 \times 6}$ (orange) is the fitted coefficient matrix. Distillation instead fits a single $U = \Theta_\pi \Xi_\pi$, where $\Theta_\pi \in \mathbb{R}^{N \times 84}$ is the degree-3 library over the 6-dimensional raw physical state $[x,\theta_1,\theta_2,\dot{x},\dot\theta_1,\dot\theta_2]$ ($\binom{9}{3}=84$ features), and $\Xi_\pi \in \mathbb{R}^{84 \times 1}$ (purple) is the scalar action coefficient vector. The two libraries are distinct: different input spaces (raw state-action vs. raw physical state) and different feature counts (120 vs. 84).](figures/sindy_matrix_shapes.svg){width=90%}

---

## 3  Methods

This section describes the experimental setup. The evaluation proceeds in three stages: (1) train a full-order baseline to establish a performance ceiling; (2) run the SINDy-RL Dyna loop to produce a data-efficient neural policy; and (3) distill that neural policy into a closed-form polynomial controller. We then vary the feature library Θ and RL optimizer A to characterize how these choices affect all three goals.

### 3.1  Baseline

A standard PPO agent (Stable-Baselines3 2.8.0 [@raffin2021sb3]) with unlimited real-environment access serves as the performance ceiling: a two-hidden-layer [64,64] multi-layer perceptron (MLP) with tanh activations (9,731 parameters), trained for 400,000 steps (15,103 episodes). This agent is not used as the distillation teacher; the Dyna loop produces its own neural policy.

### 3.2  SINDy-RL Pipeline

All code is implemented in Python 3.12.7 using PySINDy 2.1.0 [@desilva2020pysindy] (E-SINDy surrogate), Stable-Baselines3 2.8.0 (PPO/SAC), Gymnasium 1.2.3 with MuJoCo 3.8.1 [@todorov2012mujoco] (simulation), NumPy 2.4.6, and scikit-learn 1.8.0. Full pipelines are in `notebooks/sindy-rl.ipynb` (PPO+polynomial), `sindy-rl-sac.ipynb` (SAC+polynomial), and `sindy-rl-sac-lagrangian.ipynb` (SAC+Lagrangian).

**Bootstrap.** A Schroeder multi-sine sweep [@schroeder1970] runs 300 episodes in real MuJoCo, collecting 2,897 state-transition pairs. Transitions with tip height $h > 1.10$ m (poles within $\approx 24°$ of vertical) are retained for SINDy fitting; 87% of the bootstrap data passes this filter. This near-upright filter is derived from segment geometry ($L_1 + L_2 = 1.2$ m) rather than any reward constant; this distinction matters and is discussed in §4.1. Figure 4 shows the resulting angular coverage: all retained transitions lie in a narrow near-upright band ($|\theta_1| \lesssim 48°$, $|\theta_2| \lesssim 67°$), the same region the stabilizing policy will inhabit, so the surrogate need only be accurate locally, where a degree-3 Taylor expansion captures the IDP's cubic inter-modal coupling even though the system is globally nonlinear.

![**Figure 4.** Bootstrap state coverage: 2,897 transitions from 300 Schroeder episodes in joint-angle space. Blue points ($h > 1.10$ m, 87%) are retained for E-SINDy fitting; grey points are excluded. The Schroeder sweep keeps the system near the upright equilibrium, defining the effective operating domain of the surrogate; the degree-3 polynomial is a valid Taylor approximation in this domain even though IDP dynamics are globally nonlinear.](figures/fig_bootstrap_coverage.png){width=50%}

**E-SINDy fit.** Ten SINDy-C models are fit on 80% bootstrap subsamples of the filtered transitions, using a degree-3 polynomial library over the 7-dimensional state-action input (120 features; PySINDy `PolynomialLibrary(degree=3)`, STLSQ threshold 0.05). Their coefficient matrices are stacked into a `FastEnsemblePredictor`[^fep] for efficient parallel inference.

[^fep]: Calling scikit-learn's `PolynomialLibrary.transform()` once per ensemble member costs ~10 ms/step (~13 min per 75k-step PPO phase). Fix: pre-extract the `powers_` exponent matrix at construction; at each step, compute all features via `np.prod(xu ** powers_, axis=1)` and apply all 10 stacked coefficient matrices as a single batched matmul (~0.93 ms/step, 11.5$\times$ speedup).

**Surrogate PPO.** The predictor is wrapped in a Gymnasium environment that replicates MuJoCo's reward formula, applies the uncertainty penalty $\kappa \cdot \text{mean}(\sigma_\Delta)$, and enforces physical bounds ($|x| \leq 2.5$ m, $|\theta| \leq 0.9$ rad, $|\dot\theta| \leq 12$ rad/s). PPO trains for 100k surrogate steps (warm-started from the prior policy), early-stopped if mean episode length stays below 5 after 50k steps.

**Real data collection.** The policy is deployed in real MuJoCo for 4,000 steps, appended to the dataset $\mathcal{D}$.

**Repeat.** Steps 2–4 repeat with a warm-started policy. If exploitation is detected (surrogate reward $> 3\times$ previous and real episode length $< 50\%$ of best seen), the next iteration rolls back to the best real-environment checkpoint.

**Distillation.** The best checkpoint (by cross-validated real-environment success) is distilled: 50k expert transitions from the surrogate under the NN oracle are augmented 5× via per-dimension Gaussian noise [@ross2011dagger] and re-queried from the NN oracle, then fit with STLSQ over the degree-3 polynomial library on the 6-dimensional raw physical state $[x, \theta_1, \theta_2, \dot{x}, \dot\theta_1, \dot\theta_2]$ (84 features at degree-3).

### 3.3  Variants

Beyond the default PPO+polynomial configuration, we evaluate three modifications. The **no-$x$ variant** removes cart position $x$ from the SINDy library, exploiting the IDP's translational symmetry ($\Delta x = \dot{x}\,\Delta t$ exactly by kinematics). The **SAC swap** replaces PPO with Soft Actor-Critic [@haarnoja2018sac], whose off-policy replay buffer accumulates broader state-space coverage across iterations; Algorithm 1 of [@zolman2025sindyrl] treats the policy optimizer $\mathcal{A}$ as a practitioner input, making this a direct exercise of that degree of freedom.

\needspace{18\baselineskip}
The **Lagrangian library** replaces the 120-feature polynomial library with 32 atoms derived from the IDP Euler-Lagrange equations, following the principle that SINDy's library is only limited by domain knowledge [@brunton2016sindy; @brunton2022datadriven] and that incorporating known physical structure improves identification accuracy and interpretability [@loiseau2018constrained]:

- **kinematics + gravity + control** (8): $\dot{x},\,\dot{\theta}_1,\,\dot{\theta}_2,\,\sin\theta_1,\,\cos\theta_1,\,\sin(\theta_1{+}\theta_2),\,\cos(\theta_1{+}\theta_2),\,u$
- **centrifugal/Coriolis velocity products** (6): $\dot{x}^2,\,\dot{\theta}_1^2,\,\dot{\theta}_2^2,\,\dot{\theta}_1\dot{\theta}_2,\,\dot{x}\dot{\theta}_1,\,\dot{x}\dot{\theta}_2$
- **mass-matrix angle–velocity** (9): $\sin\theta_1\cdot\dot{\theta}_1$, $\cos\theta_1\cdot\dot{\theta}_1$, $\sin(\theta_1{+}\theta_2)\cdot\dot{\theta}_1$, $\cos(\theta_1{+}\theta_2)\cdot\dot{\theta}_1$, $\sin(\theta_1{+}\theta_2)\cdot\dot{\theta}_2$, $\cos(\theta_1{+}\theta_2)\cdot\dot{\theta}_2$, $\sin\theta_1\cdot\dot{x}$, $\sin(\theta_1{+}\theta_2)\cdot\dot{x}$, $\cos(\theta_1{+}\theta_2)\cdot\dot{x}$
- **relative angle** (2): $\sin\theta_2,\,\cos\theta_2$
- **cubic Coriolis** (5): $\sin\theta_2\cdot\dot{\theta}_1^2$, $\sin\theta_2\cdot\dot{\theta}_2^2$, $\sin\theta_2\cdot\dot{\theta}_1\dot{\theta}_2$, $\cos\theta_2\cdot\dot{\theta}_1^2$, $\cos\theta_2\cdot\dot{\theta}_2^2$
- **control coupling** (2): $\cos\theta_1\cdot u$, $\cos(\theta_1{+}\theta_2)\cdot u$

Because all atoms are genuinely present in the IDP dynamics, ridge regression (no thresholding) is used for the Lagrangian dynamics fit; STLSQ is applied only during policy distillation.

### 3.4  Metrics

Data efficiency: real-environment step count. Task performance: success rate ($\geq$ 500 steps) and mean episode length. Surrogate quality: E-SINDy one-step RMSE on held-out near-upright transitions. Distillation quality: ordinary least squares (OLS) $R^2$ and STLSQ term count.

---

## 4  Results

This section presents results in three parts: (1) the baseline ceiling; (2) the engineering obstacles that blocked convergence on the polynomial variant and how they were resolved; and (3) the performance of all four library–optimizer combinations, culminating in the SAC+Lagrangian result.

### 4.1  Baseline

Full-order PPO achieves mean reward $9{,}324 \pm 2$, 100% success, and mean episode length 1,000/1,000 steps using 400,000 real interactions and 9,731 parameters. This is the performance ceiling against which Dyna efficiency is measured.

### 4.2  Engineering Obstacles

Two obstacles blocked convergence on the default PPO+polynomial configuration. Both are traceable to properties of the degree-3 polynomial representation on the IDP, not to the algorithm itself.

**Degree-2 RMSE ceiling.** On the default library (`SINDY_DEGREE=2`, 36 features), RMSE oscillated at 0.10–0.16 over 25 Dyna iterations while training data grew from 5k to 90k transitions; real episode length grew from 6 to 22 steps without approaching 500. Adding data had no effect because the bottleneck was *model capacity*, not data volume. A degree-2 library cannot express the inter-modal coupling terms that dominate IDP dynamics (e.g., $\cos\theta_1 \cdot \cos\theta_2 \cdot \dot\theta_1$), which are cubic. The diagnostic signature is a flat RMSE-versus-data curve: when RMSE does not decrease with 10× more data, the cause is capacity rather than data insufficiency. Plotted on log-log axes, a slope near zero means capacity failure; a negative slope means data insufficiency. The two have different fixes, model order versus more data, and conflating them wastes resources proportional to how long the wrong diagnosis persists. Separately, the near-upright filter threshold had been inherited from a reward-shaping constant (`TIP_HEIGHT_TARGET = 2.0`, a dimensionless offset) rather than derived from segment geometry, setting it above the physical maximum $L_1 + L_2 = 1.2$ m; every iteration silently fit on all data. Both were corrected together: `SINDY_DEGREE=3` (120 features) and `SINDY_H_MIN = 1.10` m. Within two iterations RMSE dropped to 0.013.

**Surrogate exploitation.** Once degree-3 produced a useful surrogate, a second failure mode emerged: in a diagnostic run, surrogate reward jumped 9× (from 497 to 4,525 per episode) while real episode length collapsed 87% (from 414 to 56 steps). The policy had found action sequences the polynomial rated as highly rewarding that had no real-physics counterpart. The designed defense, ensemble uncertainty penalization, was insufficient. All 10 ensemble members share the same degree-3 polynomial basis, so in extrapolated regions they all make the same wrong prediction simultaneously; $\sigma_\Delta$ stays low and the penalty does not fire. Ensemble disagreement can only detect coefficient uncertainty, not shared extrapolation error. Real-environment feedback is the only reliable out-of-distribution signal. The fix requires both mechanisms together: the uncertainty penalty `reward -= 5.0 * mean(sigma_delta)` steers PPO away from noisy regions during surrogate training; the rollback trigger (surrogate reward $> 3\times$ previous AND real episode length $< 50\%$ of best seen) catches shared extrapolation errors post hoc and restores the best real-environment checkpoint. Either alone is insufficient.

**Distillation obstacles.** Five additional failures emerged during behavioral cloning. (1) *Teacher selection*: using the final loop policy as the distillation teacher produced 0% closed-loop success; the cross-validated best checkpoint (iteration 7) was required. (2) *Degree-2 distillation*: fitting the policy library at degree-2 hit the same capacity ceiling as the dynamics surrogate: $R^2 \approx 0.905$ regardless of data volume; degree-3 over the 6D raw physical state (84 features) was required. (3) *Observation encoding*: the MuJoCo observation uses sine/cosine angle encodings; near $\theta \approx 0$ these create near-exact linear dependences in every polynomial degree ($\cos^2\theta \approx 1$, $\sin^2\theta + \cos^2\theta = 1$), making STLSQ coefficients on canceling pairs numerically meaningless; switching to raw angles eliminates the collinearities. (4) *Dataset coverage*: the ~28k Dyna transitions cluster tightly along the expert's trajectory; surrogate rollout, running the best-checkpoint policy inside E-SINDy for 50k steps, samples the policy's actual visit distribution at zero additional real-environment cost. (5) *Distribution shift*: perturbation augmentation (per-dimension Gaussian noise, NN oracle re-queried at each perturbed state) was necessary to cover adjacent states the straight behavioral clone otherwise handles poorly.

### 4.3  PPO + Polynomial: Data Economy Without Interpretability

With obstacles resolved, the PPO+polynomial Dyna loop converged in eleven iterations using 66,277 real steps, **6.0× fewer than baseline**. Cross-validated evaluation identified iteration 7 as the best checkpoint at **90% success**. Performance degrades if the loop is continued past the convergence peak, a consequence of continued surrogate exploitation corrupting the dataset, motivating cross-validated best-checkpoint selection rather than using the final policy.

**Why the distilled polynomial is near-dense.** The dynamics feature matrix $\Theta$ (209,620 × 120, all transitions × library terms) has condition number $\kappa = 2.37 \times 10^4$, full rank, but with a ${\approx}24{,}000$× spread between largest and smallest singular values. As a rule of thumb, $\kappa \approx 10^k$ costs roughly $k$ digits of numerical precision in the worst-conditioned direction; at $\kappa \approx 10^{4.4}$, approximately four digits are unreliable, enough to make small STLSQ coefficients numerically meaningless regardless of threshold. Figure 5A shows the singular value spectrum; the five-decade spread means small OLS coefficients along low-singular-value directions are numerically unreliable; STLSQ cannot safely zero them without risking structural contributions. Behavioral cloning from the best checkpoint produced $R^2 = 0.911$ with STLSQ ($\lambda = 0.10$) retaining **82/84 terms**, 119× smaller than baseline but not interpretable in the five-to-thirty-term sense the framework promises. Figure 5B shows the resulting coefficient magnitudes: across all polynomial degrees the retained terms carry comparable weight, with no cluster of negligible coefficients that thresholding could safely prune. A threshold sweep from $\lambda \in [0.001, 50]$ (Figure 5C) confirmed that success stays at 90–95% regardless of threshold, ruling out threshold sensitivity as the source of fragility; the density persists because the feature basis, not the threshold choice, limits sparsification.

![**Figure 5.** Why the degree-3 polynomial policy is dense. **(A)** Singular value spectrum of $\Theta$ ($\kappa = \sigma_{\max}/\sigma_{\min} = 2.37 \times 10^4$); the five-decade spread means small OLS coefficients in the worst-conditioned directions are numerically unreliable. **(B)** Distilled policy coefficient magnitudes (log scale), coloured by polynomial degree; 82 of 84 terms are retained at $\lambda = 0.10$ with no cluster of negligible coefficients. **(C)** Threshold ablation: performance is stable at 90–95% success across the full range of $\lambda$, confirming that density is a property of the polynomial basis rather than the threshold choice.](figures/fig_poly_density_combined.png){width=100%}

### 4.4  Variant Comparison: Library and Optimizer Matter

The strict continuation experiment (Figure 6, "strict 100%" bar) pushed PPO+polynomial to a harder criterion, full-horizon 999-step success, consuming 323,934 real steps, 5× the original budget, without surpassing the standard result. This data efficiency collapse confirmed that iteration count is the wrong lever and motivated the SAC and Lagrangian variants: SAC's off-policy replay buffer accumulates broader state-space coverage without on-policy restarts; the Lagrangian library resolves ill-conditioning at its source. Figure 6 places all configurations against the baseline. Three patterns emerge. (1) Data efficiency improves monotonically as the surrogate becomes more accurate or the optimizer accumulates broader coverage: PPO+poly (6.0$\times$) $\to$ PPO+no-$x$ (11.4$\times$) $\to$ SAC+poly (13.0$\times$) $\to$ PPO+Lagrangian (15.4$\times$) $\to$ SAC+Lagrangian (17.6$\times$). (2) Distilled-polynomial term count is unchanged by the optimizer: both PPO+poly (82/84) and SAC+poly (83/84) produce near-dense distilled controllers, confirming that ill-conditioning is a property of the polynomial representation, not the training optimizer. (3) The Lagrangian library breaks the density: 29/29 terms, all physically interpretable, at the highest efficiency among all variants.

![**Figure 6.** All SINDy-RL variants versus the 400,000-step full-order baseline (20 episodes each). Left: real-environment steps on log scale (fewer = more efficient). Right: task success rate. The "strict 100%" bar is a stress-test continuation experiment under a harder criterion (full-horizon 999 steps) that consumed 323k+ steps to reach 95% success, confirming that brute-force iteration cannot substitute for appropriate library and optimizer choices.](figures/fig_all_variants.png){width=90%}

\FloatBarrier
### 4.5  SAC + Lagrangian: All Three Goals

The SAC+Lagrangian variant converged in two Dyna iterations using **22,723 real steps (17.6× fewer than baseline)**, with the neural policy achieving **100% task success** and mean episode length 1,000 steps (matching the full-order ceiling). The Lagrangian basis reduces the feature count from 120 to 32 and cuts the condition number by approximately a factor of six relative to the polynomial basis. The Lagrangian surrogate has higher next-state RMSE than the polynomial surrogate (it only models physically relevant dynamics), yet achieves substantially better closed-loop performance, confirming that RMSE on generic next-state prediction is a poor quality metric for physics-informed surrogates.

Behavioral cloning from the SAC+Lagrangian best checkpoint produced a 29-term controller with $R^2 = 0.958$ and **100% task success** at mean episode length 715 steps. Every retained term has a direct physical interpretation: velocity damping terms ($\dot\theta_1$, $\dot\theta_2$), gravitational restoring forces ($\sin\theta_1$, $\sin(\theta_1+\theta_2)$), Coriolis coupling ($\sin\theta_2 \cdot \dot\theta_1^2$, $\sin\theta_2 \cdot \dot\theta_1\dot\theta_2$), and control-affine coupling ($\cos\theta_1 \cdot u$, $\cos(\theta_1+\theta_2) \cdot u$). The controller is 336× smaller than the baseline neural network.

| Variant | Real-env steps | Efficiency | NN success | Distilled terms | Distilled success |
|---|---:|---:|---:|---|---:|
| Baseline PPO | 400,000 | 1.0× | 100% | — | — |
| PPO + poly (strict)$^\ddagger$ | 323,934 | 1.2× | 95% | 84/84 | 97%$^\ddagger$ |
| PPO + polynomial | 66,277 | 6.0× | 90% | 82/84 | 85% |
| PPO + no-$x$ | 35,111 | 11.4× | 85% | 55/56 | 85% |
| SAC + polynomial | 30,735 | 13.0× | 100% | 83/84 | 70% |
| PPO + Lagrangian | 25,984 | 15.4× | 100% | 29/29 | 100% |
| **SAC + Lagrangian** | **22,723** | **17.6×** | **100%** | **29/29** | **100%** |

$^\ddagger$Strict continuation targeting full-horizon (≥999-step) success; 95% at the standard ≥500-step criterion. Peak at the harder criterion was 80% after 108,251 steps; the loop continued to 323,934 steps without improvement. Distilled success (97%) also evaluated at the ≥999-step criterion.

### 4.6  Code Repository

All code and results: **https://github.com/falconeaj1/ME_595**. Key notebooks: `full-order-simulation.ipynb` (baseline PPO), `sindy-rl.ipynb` (PPO+polynomial), `sindy-rl-sac.ipynb` (SAC+polynomial), `sindy-rl-sac-lagrangian.ipynb` (SAC+Lagrangian).

---

## 5  Summary

We stress-tested SINDy-RL on the inverted double pendulum and found that **the algorithm passes the stress test**: with the right choices of feature library and RL optimizer, it delivers data economy, a reduced-order policy, and interpretability simultaneously on a genuinely hard unstable benchmark.

**Data economy (Goal 1) is confirmed across all variants.** Every library–optimizer combination substantially outperforms the full-order PPO baseline. The gains are structural to the Dyna architecture: the surrogate provides cheap training rollouts that substitute for real environment interactions. Efficiency scales with surrogate quality and optimizer coverage, from 6.0× (PPO+polynomial) to 17.6× (SAC+Lagrangian). Running PPO+polynomial harder, stricter convergence requiring 323k+ steps, does not close the interpretability gap, confirming that brute-force iteration cannot compensate for the wrong representation choice.

**Reduced-order policy (Goal 2) is confirmed.** The SAC+Lagrangian distilled controller is 29 terms, 336× smaller than the baseline network, with 100% task success. The PPO+polynomial controller (82 terms, 119× smaller) also confirms the reduced-order goal, though its density limits direct physical interpretability.

**Interpretability (Goal 3) requires a physics-informed library.** The degree-3 polynomial basis is ill-conditioned on the IDP ($\kappa \approx 2.4 \times 10^4$): STLSQ retains 82 of 84 terms regardless of threshold, and a threshold sweep confirms this cannot be tuned away. This is a finding about *representation selection*, not a failure of SINDy-RL. The 32-atom Lagrangian library, derived from the IDP Euler-Lagrange equations, delivers interpretability by construction, because the basis encodes what the system actually does. Every retained term in the distilled controller is a recognizable physical quantity: velocity damping, gravitational restoring forces, inter-link Coriolis coupling, and control-affine actuation.

**Two obstacles generalize.** The degree-2 RMSE ceiling illustrates a general diagnostic: when RMSE does not decrease with 10× more data, the cause is model capacity, not data volume; the fix is model order, not more iterations. The exploitation failure illustrates a structural limitation of ensemble surrogates: when all members share a function basis, they agree on the same wrong prediction in extrapolated regions; real-environment feedback is the only reliable out-of-distribution signal, and uncertainty penalization and rollback are complementary safeguards, not substitutes.

**The practical implication** is that library selection dominates outcome on this system. On the IDP, the Lagrangian library resolved ill-conditioning at its source, reducing feature count, improving conditioning, and delivering interpretability without thresholding. For other unstable mechanical systems with a known Lagrangian, a physics-informed library derived from the equations of motion is a natural first choice of Θ in Algorithm 1, though whether it generalizes to systems with more complex dynamics or unknown Lagrangians remains an open question.

Future work should characterize what happens as the system departs further from equilibrium (swing-up tasks), quantify whether the 29-term Lagrangian controller admits formal stability guarantees in the verified region, and explore autoencoder-based latent SINDy representations for systems where the governing equations are unknown. A complementary open question is whether interpretability can be recovered from the polynomial basis without switching libraries: constrained sparse regression [@loiseau2018constrained] enforces known physical symmetries directly on the coefficient matrix, and trapping SINDy [@kaptanoglu2021stability] promotes global stability via a modified loss, both of which could in principle reduce the ill-conditioned density we observed and are worth evaluating as alternatives to the Lagrangian library on the IDP.

\newpage

\noindent\textbf{CRediT Statement} (\url{https://credit.niso.org})

\begingroup
\small
\begin{tabular}{lll}
\textbf{Role} & \textbf{Patrick Smith} & \textbf{Andrew Falcone} \\
\hline
Conceptualization          & Yes  & Yes        \\
Data curation              & Yes  & Yes        \\
Formal analysis            & Yes  & Yes        \\
Investigation              & Yes  & Yes        \\
Methodology                & Yes  & Yes        \\
Software                   & Lead & Supporting \\
Validation                 & Yes  & Yes        \\
Visualization              & Yes  & Yes        \\
Writing -- original draft  & Yes  & Yes        \\
Writing -- review \& editing & Yes & Yes       \\
\end{tabular}
\endgroup

\smallskip
\noindent\small\textit{AI tool disclosure: Claude (Anthropic) assisted with code drafting, debugging, writing iteration, and figure generation. All analysis, results, and conclusions were reviewed and executed by the authors, who take full responsibility for the submitted work.}


# References
