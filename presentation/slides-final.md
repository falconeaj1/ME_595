---
marp: true
theme: default
math: katex
paginate: true
footer: 'ME 595 · University of Washington · Spring 2026'
size: 16:9
---

<style>
/* ═══════════════════════════════════════════════════
   UW Brand System
   Husky Purple  #4B2E83
   Metallic Gold #B7A57A
   ═══════════════════════════════════════════════════ */

:root {
  --uw-purple:    #4B2E83;
  --uw-gold:      #B7A57A;
  --uw-gold-dark: #8B6914;
  --uw-light:     #F0EDF7;
  --uw-gold-bg:   #FDF8EC;
  --uw-dark:      #1a0a3d;
}

/* ── Base ── */
section {
  font-family: 'Open Sans', 'Helvetica Neue', Arial, sans-serif;
  font-size: 28px;
  line-height: 1.55;
  color: #1a1a1a;
  background: #ffffff;
  padding: 52px 72px;
  position: relative;
}

section::after {
  color: #b0a0c8;
  font-size: 0.58em;
}

footer {
  color: #b0a0c8;
  font-size: 0.55em;
}

/* ── Headings ── */
h1 {
  color: var(--uw-purple);
  font-size: 1.55em;
  font-weight: 700;
  letter-spacing: -0.01em;
  border-bottom: 3px solid var(--uw-gold);
  padding-bottom: 0.15em;
  margin-bottom: 0.45em;
  margin-top: 0;
}

h2 {
  color: var(--uw-purple);
  font-size: 1.05em;
  font-weight: 600;
  margin: 0.25em 0;
}

h3 {
  font-size: 0.9em;
  color: var(--uw-gold-dark);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin: 0.2em 0;
}

/* ══ TITLE SLIDE ══ */
section.title {
  background: var(--uw-purple);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px 100px;
}

section.title h1 {
  color: #ffffff;
  font-size: 2em;
  border: none;
  border-bottom: 3px solid var(--uw-gold);
  padding-bottom: 0.25em;
  margin-bottom: 0.6em;
  max-width: 1020px;
  line-height: 1.25;
}

section.title p {
  color: var(--uw-gold);
  font-size: 0.88em;
  margin: 0.18em 0;
}

/* ══ SECTION DIVIDER ══ */
section.section {
  background: var(--uw-purple);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px 100px;
}

section.section h1 {
  color: var(--uw-gold);
  font-size: 2.2em;
  border: none;
  line-height: 1.2;
}

section.section p {
  color: rgba(255,255,255,0.7);
  font-size: 0.82em;
  margin-top: 0.4em;
}

/* ══ BIG STATEMENT ══ */
section.statement {
  background: var(--uw-light);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 60px 100px;
}

section.statement h1 {
  color: var(--uw-purple);
  font-size: 1.9em;
  border: none;
  line-height: 1.3;
  max-width: 800px;
}

section.statement p {
  color: #555;
  font-size: 0.82em;
  max-width: 680px;
  margin-top: 0.6em;
}

/* ══ COMPONENTS ══ */

/* Purple accent box */
.box {
  background: var(--uw-light);
  border-left: 4px solid var(--uw-purple);
  padding: 0.55em 1em;
  border-radius: 0 5px 5px 0;
  margin: 0.3em 0;
  font-size: 0.88em;
}

/* Gold accent box */
.gold-box {
  background: var(--uw-gold-bg);
  border-left: 4px solid var(--uw-gold-dark);
  padding: 0.55em 1em;
  border-radius: 0 5px 5px 0;
  margin: 0.3em 0;
  font-size: 0.88em;
}

/* Placeholder */
.placeholder {
  border: 3px dashed var(--uw-gold);
  background: var(--uw-gold-bg);
  padding: 1em 2em;
  text-align: center;
  border-radius: 8px;
  color: var(--uw-gold-dark);
  font-style: italic;
  font-size: 0.88em;
  margin: 0.5em 0;
}

/* Two column */
.cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.6em;
  align-items: start;
}

.cols-3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1.2em;
  align-items: start;
}

/* Stat block */
.stat { text-align: center; padding: 0.4em 0; }
.stat-n { font-size: 2.6em; font-weight: 900; color: var(--uw-purple); line-height: 1; }
.stat-l { font-size: 0.7em; color: #666; margin-top: 0.15em; }

/* Flow row */
.flow {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-wrap: nowrap;
  gap: 0.3em;
  margin: 0.6em 0;
  font-size: 0.82em;
}

.fn {
  background: var(--uw-purple);
  color: #fff;
  padding: 0.38em 0.6em;
  border-radius: 6px;
  text-align: center;
  min-width: 60px;
  line-height: 1.3;
}

.fn.gold {
  background: var(--uw-gold-dark);
}

.fn.light {
  background: var(--uw-light);
  color: var(--uw-purple);
  border: 1.5px solid var(--uw-purple);
}

.fa { color: var(--uw-gold-dark); font-size: 1.3em; font-weight: 700; }

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.73em;
  margin-top: 0.3em;
}

th {
  background: var(--uw-purple);
  color: #fff;
  padding: 5px 10px;
  text-align: left;
  font-weight: 600;
}

td {
  padding: 4px 10px;
  border-bottom: 1px solid #e8e0f0;
}

tr:nth-child(even) td { background: var(--uw-light); }

.check  { color: #2e7d32; font-weight: 700; }
.cross  { color: #c62828; font-weight: 700; }
.partial{ color: #e65100; font-weight: 700; }

/* Equation callout */
.eq-box {
  background: #fff;
  border: 2px solid var(--uw-purple);
  border-radius: 8px;
  padding: 0.7em 1.6em;
  text-align: center;
  margin: 0.5em auto;
  display: inline-block;
}

/* UW W badge on title */
.uw-w {
  font-size: 3.5em;
  font-weight: 900;
  color: var(--uw-gold);
  letter-spacing: -0.05em;
  line-height: 1;
  font-style: italic;
  margin-bottom: 0.2em;
  font-family: Georgia, serif;
}

/* Timeline */
.timeline {
  display: flex;
  flex-direction: column;
  gap: 0;
  font-size: 0.82em;
  margin-top: 0.4em;
}

.tl-row {
  display: flex;
  align-items: center;
  gap: 0.8em;
  padding: 0.3em 0;
}

.tl-dot {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--uw-purple);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 0.7em;
  font-weight: 700;
}

.tl-dot.gold { background: var(--uw-gold-dark); }

.tl-line {
  width: 2px;
  height: 16px;
  background: var(--uw-light);
  margin-left: 10px;
}

.tl-label {
  font-weight: 600;
  color: var(--uw-purple);
  min-width: 200px;
}

.tl-desc { color: #555; }

/* Center standalone block images */
img[alt~="chicken-egg"] { display: block; margin: 0.3em auto; }
img[alt~="swing-up"]    { display: block; margin: 0.4em auto; }
img[alt~="sindy-loop"]  { display: block; margin: 0.3em auto; }
</style>

<!-- ─────────────────────────────────────────────────
  SLIDE 1 · TITLE
───────────────────────────────────────────────── -->
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="uw-w">W</div>

# Interpretable Control for<br>Unstable Systems via SINDy&#8209;RL

**Patrick Smith &nbsp;·&nbsp; Andrew Falcone**

ME 595 &nbsp;·&nbsp; University of Washington &nbsp;·&nbsp; Spring 2026

<!--
Welcome — I'm Patrick Smith, and this is Andrew Falcone. We're going to walk you through our project applying SINDy-RL to the inverted double pendulum. Andrew's been working on the dynamics identification side; I've been working on the control policy side. Let's get into it.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 2 · THE VISION
  Patrick — 1:00
───────────────────────────────────────────────── -->

# What if the controller was just an equation?

<br>

$$u(x) = \underbrace{-2.4\,\theta_1}_{\text{balance}}
       - \underbrace{0.9\,\theta_2}_{\text{balance}}
       - \underbrace{0.5\,\dot\theta_1}_{\text{damp}}
       - \underbrace{0.2\,\dot\theta_2}_{\text{damp}}
       + \cdots$$

<br>

<div class="gold-box">
~200 terms &nbsp;·&nbsp; 47× fewer parameters than the NN &nbsp;·&nbsp; most coefficients have physical interpretation
</div>

<!--
Think about where autonomous control needs to go: a surgical robot that a regulator has to certify before it touches a patient. A delivery drone flying over populated areas with no cloud connection and a microcontroller for a brain. In each of these cases, a ten-thousand-parameter neural network is a dead end — you can't certify what cannot be explained, and you can't run a GPU on a battery-powered UAV. But if the controller is a polynomial equation, you can reason about every term, prove stability bounds, audit every decision, and flash it onto embedded hardware. That's what we're trying to demonstrate — the performance of deep RL, in a form that can actually be deployed and trusted. And this is the goal: a polynomial governing equation where every coefficient is physically meaningful. For model-based control or faster RL training, that sparsity is critical — instead of running thousands of rollouts in an expensive full simulator, you run them in this equation. That's what SINDy gives us.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 3 · THE SYSTEM
  Andrew — 0:40
───────────────────────────────────────────────── -->

# The Inverted Double Pendulum

<div class="cols" style="align-items:center;">

<div>

**State** &nbsp; $x \in \mathbb{R}^6$

$$x = \begin{bmatrix} x_{\text{cart}} \\ \theta_1 \\ \theta_2 \\ \dot x_{\text{cart}} \\ \dot\theta_1 \\ \dot\theta_2 \end{bmatrix}$$

**Action** &nbsp; $u \in [-1,\,1]$ — cart force

**Unstable equilibrium** at $\theta_1 = \theta_2 = 0$

</div>

<div style="text-align:center;">

![pendulum diagram w:220](pendulum_diagram.png)

<div style="font-size:0.72em; color:#888; margin-top:0.3em;">
  L₁ = L₂ = 0.6 m &nbsp;·&nbsp; dt = 0.05 s &nbsp;·&nbsp; MuJoCo physics
</div>

</div>
</div>

<!--
Our testbed is the inverted double pendulum — two poles balanced upright on a cart. Six-dimensional state: cart position, two joint angles, and their rates. The only input is a horizontal cart force. What makes this hard: the upright equilibrium is unstable. A random policy crashes in about five steps — you never reach the near-equilibrium region. And that's exactly where data-driven dynamics identification is most needed.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 4 · SINDY
  Andrew — 0:55


───────────────────────────────────────────────── -->

# SINDy — Discovering equations from data

**Key idea:** dynamics live in a low-dimensional function space.

$$\dot X \;=\; \underbrace{\Theta(X,U)}_{\substack{\text{candidate} \\ \text{feature library}}} \cdot \underbrace{\Xi}_{\substack{\text{sparse} \\ \text{coefficients}}}$$

$$\Theta = \bigl[\;1 \;\big|\; x \;\big|\; \theta_1,\,\theta_2 \;\big|\; x^2,\,x\theta_1,\,\theta_1^2,\;\ldots\;\bigr] \quad \text{degree-}d\text{ polynomial library}$$

<div class="cols" style="gap:1em; margin-top:0.4em;">
<div class="gold-box">
<b>STLSQ solver</b> drives most coefficients to <em>exactly zero</em> —  leaving a sparse equation with only the dominant governing terms.
</div>
<div class="box">
<b>Library degree <em>d</em> is a design choice</b> — it must match the complexity of the system and is not obvious a priori.
</div>
</div>

<!--
<div class="box" style="margin-top:0.5em; text-align:center; font-size:0.8em;">
  <span style="color:#888;">Lineage: &nbsp;</span>
  LASSO <span style="color:var(--uw-gold-dark);">'96</span>
  &nbsp;→&nbsp; SINDy <span style="color:var(--uw-gold-dark);">'16</span>
  &nbsp;→&nbsp; SINDy-C <span style="color:var(--uw-gold-dark);">'18</span>
  &nbsp;→&nbsp; E-SINDy <span style="color:var(--uw-gold-dark);">'22</span>
  &nbsp;→&nbsp; <strong>SINDy-RL <span style="color:var(--uw-gold-dark);">'24</span></strong>
  &nbsp;&nbsp;|&nbsp;&nbsp; Koopman as the competing paradigm
</div>
-->

<!--
SINDy is the core of our dynamics approach. The idea: nonlinear dynamics live in a low-dimensional space of basis functions. We construct a polynomial library Θ over the state and control inputs, then use sparse regression — specifically STLSQ — to identify which terms actually drive the dynamics. Most coefficients get driven to exactly zero, leaving only the governing terms. One critical design choice is the polynomial degree — it must be rich enough to capture the system's nonlinearities.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 5 · OBJECTIVES
  Both — 0:35
───────────────────────────────────────────────── -->

# Two objectives

<br>

<img src="objectives_diagram.png" style="display:block; margin:1.2em auto 0; width:1060px;" alt="chicken-egg diagram">

<!--
[Andrew]: "We have 2 objectives. Objective one: learn a reduced-order model. SINDyC identifies dynamics with control inputs — giving us an explicit, interpretable surrogate we can run RL inside." [Patrick]: "And, Objective two: distill the NN policy trained in that surrogate down to a sparse polynomial.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 6 · CHICKEN-AND-EGG
  Andrew — 0:50
───────────────────────────────────────────────── -->

# SINDyC is a chicken-and-egg problem

You can't train near equilibrium without reaching it — and you can't reach it without a controller.

![chicken-egg diagram w:750](chicken_egg_diagram.png)

<br>

<div class="gold-box" style="font-size:0.83em;">
  Solution: co-train the controller and surrogate in an iterative active-learning loop.
</div>


<!--
However, we hit a wall! Since the inverted pendulum is inherently unstable, using SINDy (open loop dynamics), we could not gate data near equilibrium. Instead, we had to train SINDy with conrol (SINDyC). We start by bootstrapping, and over multiple iterations incrementally improve the controller which allows us to better explore the dynamics near equilibirum.

We essentially had to redesign the whole approach: co-train the controller and surrogate in an iterative active-learning loop.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 7 · BASELINE
  Patrick — 0:30
───────────────────────────────────────────────── -->

# Baseline — the performance ceiling

<br>

A standard PPO agent trained with **full simulator access**.

<div class="cols-3" style="margin-top:1em; text-align:center;">

<div class="box" style="padding:1em;">
  <div class="stat-n">100%</div>
  <div class="stat-l">Task success rate<br>(20/20 eval episodes)</div>
</div>

<div class="box" style="padding:1em;">
  <div class="stat-n">9359</div>
  <div class="stat-l">Mean reward<br>(max possible ≈ 10,000)</div>
</div>

<div class="box" style="padding:1em;">
  <div class="stat-n">9,731</div>
  <div class="stat-l">Policy parameters<br>MLP [64 × 64]</div>
</div>

</div>

<!--
We started by defining the baseline. We trained a full-order Proximal Policy Optimization agent with unlimited simulator access. It achieved 100% success rate, with mean reward 9,359 — essentially perfect. It took 400,000 simulator interactions and produced a 9,731-parameter Multi-Layer Perceptron neural net. That's what we're trying to match with something interpretable and much more data-efficient.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 8 · SPARSE POLICY DISTILLATION
  Patrick — 0:50
───────────────────────────────────────────────── -->

# Sparse policy distillation — the compounding error trap

<div class="cols" style="align-items:start;">

<div>

**Behavioral cloning** on 50,000 oracle transitions:

$$\min_{\Xi} \;\bigl\|\Theta(X)\,\Xi - U^*\bigr\|_2 \;+\; \lambda\|\Xi\|_1$$

210-term degree-4 library → STLSQ retains **206 terms** · **47× fewer params** than NN


</div>

<div>

**But the policy fails under noise:**



| Noise σ | Mean episode length |
|---|---|
| 0 (training) | ~1000 steps ✓ |
| 0.1 | ~200 steps |
| 0.3 | ~20 steps ✗ |

<div class="gold-box" style="font-size:0.8em; margin-top:0.4em;">
  Off-distribution states produce small errors → errors compound → catastrophic failure.
</div>

</div>
</div>

<!--
Since we already have a trained expert from the baseline run, we get the sparse policy for free, no retraining needed. It's a one-shot regression: we collected 50,000 transitions from the expert, build the polynomial library, solve for the sparse coefficients. In our case, a degree-4 polynomial captures the cross-coupling nonlinearities — 210 possible terms, of which regressor (STLSQ) retained 206. That may not be sparse in the traditional SINDy sense, but it's still 47 times fewer parameters than the nearly 10k-parameter NN, and every most have a physical interpretation.
Distilling the NN expert into a sparse policy dictionary succeeded, but as the data shows, is vulnerable to noise. This is the compounding error problem common to Behavior Cloning approaches mentioned in the Zolman paper.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 9 · THE FIX
  Patrick — 0:40
───────────────────────────────────────────────── -->

# The fix: query the oracle, for free

<div class="cols" style="align-items:start; grid-template-columns: 3fr 2fr;">

<div>

**Key insight:** the NN policy is a pure function of state — no memory, no rollouts needed. Query it at *any* state we choose.

For each of 3 rounds:

<div style="font-size:0.85em;">

1. Perturb states: $\tilde{x} \sim \mathcal{N}(x,\, \sigma^2)$
2. Query oracle: $\tilde u = \pi_\text{NN}(\tilde x)$
3. Append $(\tilde x,\tilde u)$ to dataset

</div>

Dataset grows **4×** (50k → 200k pairs). STLSQ re-fit recovers cross-coupling terms.

</div>

<div>

|  | σ = 0.1 | σ = 0.3 |
|---|---|---|
| Base policy | ~200 steps | ~20 steps ✗ |
| Augmented | ~1000 steps ✓ | ~500–900 steps |

<br>

<br>

<div class="box" style="margin-top:0.6em; font-size:0.83em;">
  <strong>25–45× more robust</strong> — zero additional simulator interactions.
</div>

</div>

</div>

<!--
The fix suggested in the Zolman paper came from a simple insight: the NN policy has no memory — it's a pure function of state. Since we have an expert that was trained, we can essentially ask it what it would do. So we can query it at any state we want, no simulator rollouts needed. We perturb states by adding Gaussian noise (σ=0.15), query the oracle for the correct action, and add those labeled pairs to the dataset. We chose three rounds resulting in four times the original data. The result: 25 to 45 times more robust at deployment noise, with zero additional simulator interactions.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 10 · ROM SURROGATE — ACTIVE LOOP
  Andrew — 0:50
───────────────────────────────────────────────── -->

# ROM surrogate — SINDy dynamics + LQR control

![sindy-loop w:900](sindy_loop_diagram.png)

<div class="cols" style="align-items:start; margin-top:0.5em; gap:0.9em;">

<div class="box" style="font-size:0.72em; padding:0.4em 0.9em;">
  <strong>PPO on SINDy fails to transfer</strong><br>
  Exploits polynomial approximation errors — actions that score well in the model don't generalize: <strong>24 steps</strong> average in MuJoCo.
</div>

<div class="gold-box" style="font-size:0.72em; padding:0.4em 0.9em;">
  <strong>Why LQR transfers</strong><br>
  Uses only the <em>Jacobian at the upright fixed point</em> — accurate near equilibrium in both model and real system. No approximation errors to exploit.
</div>

</div>

<!--
We learn sparse polynomial dynamics with SINDy near the upright equilibrium, linearize the learned model, and compute an LQR controller from that local approximation.
The workflow is iterative: collect near-equilibrium trajectories in MuJoCo, fit SINDy, derive an LQR gain, deploy the controller, gather improved data, and repeat.
As an initial controller, LQR transferred reliably because it depends only on the local Jacobian near the upright fixed point, where the learned dynamics are most accurate.
We also explored PPO training directly inside the surrogate model. While those policies performed well inside the learned dynamics, transferring robustly back to MuJoCo remains an active area of investigation and ongoing work in the project.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 11 · ROM RESULTS
  Andrew — 0:40
───────────────────────────────────────────────── -->

# ROM surrogate — results

<div class="cols" style="align-items:start;">

<div>

**SINDy RMSE convergence** (fixed validation set):

| Iteration | RMSE | Cumulative transitions |
|---|---|---|
| 0 (bootstrap) | 0.188 | 5,000 |
| 1 | 0.182  | 10,000 |
| 2 | 0.184 | 15,000 |

<div class="gold-box" style="margin-top:0.5em; font-size:0.83em;">
  Baseline NN: <strong>400,000</strong> real-sim steps.<br>
  SINDy-LQR: <strong>15,000</strong> — <strong>27× more data-efficient.</strong>
</div>

</div>

<div>

**LQR transfer — real MuJoCo evaluation** (10 episodes per iteration):

| Iteration | Mean return | Mean length | Success |
|---|---|---|---|
| 0 | 9359.88 | 1,000 | <span class="check">100%</span> |
| 1 | 9359.87 | 1,000 | <span class="check">100%</span> |
| 2 | 9359.90 | 1,000 | <span class="check">100%</span> |

<div style="font-size:0.76em; color:#555; margin-top:0.3em;">
  LQR gain computed from SINDy linearized around upright.<br>
  Mean return matches the full-order baseline (9,359) exactly.
</div>

</div>
</div>

<!--
The main result so far is that relatively small amounts of targeted near-equilibrium data were sufficient to learn a locally accurate dynamics model.
Controllers derived from the SINDy linearization consistently stabilized the real MuJoCo system and achieved performance comparable to the neural-network baseline, while requiring substantially fewer real simulator interactions.
Current results are strongest in the local stabilization regime using SINDy-LQR. Extending this to reinforcement learning policies trained directly inside the surrogate remains ongoing work, particularly for improving robustness and transfer back to the full MuJoCo environment.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 12 · COMPARISON
  Both — 0:40
───────────────────────────────────────────────── -->

# How do they compare?

| Approach | Real-sim steps | Policy type | Mean length | Success | Interpretable |
|---|---|---|---|---|---|
| **Baseline NN** | 400,000 | NN (9,731 params) | 1,000 | 100% | <span class="cross">✗</span> | - |
| **Sparse policy (base)** | 400,000* | Polynomial (206 terms) | ~20 @ σ=0.3 | Low | <span class="check">✓</span> |
| **Sparse policy (augmented)** | 400,000* | Polynomial (206 terms) | ~500–900 | ~50–90% | <span class="check">✓</span> |
| **SINDy + PPO-in-surrogate** | ~15,000 | PPO (NN) | ~24 | ~0% | <span class="cross">✗</span> |
| **SINDy-LQR** | **15,000** | LQR from SINDy | **1,000** | **100%** | <span class="check">✓</span> |
| **Phase 3 (stretch)** | TBD | Polynomial | — | — | <span class="check">✓</span> |

<div style="font-size:0.72em; color:#888; margin-top:0.3em;">
  * Inherits baseline training data — no additional agent training, one-shot regression from oracle queries.
</div>

<!--
[Patrick]: "Here's the full picture. The baseline NN is the ceiling — perfect performance, but opaque and data-hungry. Behavioral cloning gives us an interpretable sparse polynomial, essentially for free — but it's fragile at deployment noise. Augmenting the data recovers most of that robustness, still at no additional simulator cost."

[Andrew]: "On the dynamics side, we also explored training PPO directly inside the SINDy surrogate. That approach was extremely data-efficient, but transferring the learned policy robustly back to real MuJoCo remains an ongoing challenge — the policy could exploit inaccuracies in the learned model and averaged about 24 steps after transfer.
As a more stable first step, we switched to LQR computed from the linearized SINDy dynamics near the upright equilibrium. That controller successfully stabilized the real environment using only about 5,000 real simulator interactions. We continued the active-learning loop for two additional refinement rounds — about 15,000 total transitions — but the controller was already transferring reliably after the initial bootstrap stage.
Overall, the SINDy-LQR approach achieved stable transfer with roughly 27× fewer real simulator interactions than the baseline neural-network policy."
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 13 · STRETCH GOAL
───────────────────────────────────────────────── -->
<!-- _backgroundColor: #F0EDF7 -->

# Stretch Goal

### Phase 3 · Fully Interpretable Closed-Loop Control

<br>

Combine the interpretable **dynamics** (ROM) with the interpretable **policy** (sparse dictionary)

<!--
Phase 3 is the stretch goal: combine the interpretable dynamics model from the ROM track with the sparse polynomial policy from the distillation track — a fully interpretable closed loop, end to end.
-->

---


<!-- ─────────────────────────────────────────────────
  SLIDE 14 · BONUS
───────────────────────────────────────────────── -->
<!-- _backgroundColor: #F0EDF7 -->

![bg right:48%](swing_up_animation_cropped.gif)

# Bonus Stretch Goal

### If time permits

<br>

<p style="margin-top:0.5em;">Start from the down equilibrium position.</p>

<br>

<!--
<div class="eq-box" style="font-size:0.75em; margin-top:0.1em; background: white"><strong>NN baseline:</strong> Swing-up PPO → handoff → Stabilizer PPO<br>304 steps &nbsp;·&nbsp; 15.2 s &nbsp;·&nbsp; <strong>SUCCESS</strong></div>
-->

<div class="gold-box" style="font-size:0.82em; margin-top:0.5em;"><strong>Goal:</strong> reproduce this with SINDy-RL — interpretable swing-up + interpretable stabilizer, end to end.</div>

<!--
If time permits, we want to apply this technique to stabilizing the inverted double pendulum when starting from the down equilibrium posittion.
-->

---

<!-- ─────────────────────────────────────────────────
  SLIDE 15 · CLOSING
  Andrew — 0:15
───────────────────────────────────────────────── -->
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="uw-w">W</div>

# Interpretable control for unstable systems

###  Thank you

<br>

**Patrick Smith &nbsp;·&nbsp; Andrew Falcone**

ME 595 &nbsp;·&nbsp; University of Washington &nbsp;·&nbsp; Spring 2026

<!--
Interpretable control for unstable systems. It works — and the central challenge turned out not to be the math, but the data: you can't learn the dynamics you need without the controller you don't yet have.

Thank you

-->
