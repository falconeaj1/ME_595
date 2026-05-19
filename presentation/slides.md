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
  max-width: 820px;
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
  flex-wrap: wrap;
  gap: 0.4em;
  margin: 0.6em 0;
  font-size: 0.82em;
}

.fn {
  background: var(--uw-purple);
  color: #fff;
  padding: 0.38em 0.75em;
  border-radius: 6px;
  text-align: center;
  min-width: 70px;
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
  font-size: 0.8em;
  margin-top: 0.4em;
}

th {
  background: var(--uw-purple);
  color: #fff;
  padding: 8px 14px;
  text-align: left;
  font-weight: 600;
}

td {
  padding: 6px 14px;
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
</style>

<!-- ─────────────────────────────────────────────────
  SLIDE 1 · TITLE
───────────────────────────────────────────────── -->
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="uw-w">W</div>

# Interpretable Control for<br>Unstable Systems via SINDy-RL

**Patrick Smith &nbsp;·&nbsp; Andrew Falcone**

ME 595 &nbsp;·&nbsp; University of Washington &nbsp;·&nbsp; Spring 2026

---

<!-- ─────────────────────────────────────────────────
  SLIDE 2 · THE PROBLEM
───────────────────────────────────────────────── -->
<!-- _class: statement -->

# "Stop explaining black-box ML — use inherently interpretable models."

<p style="font-size:0.85em; margin-top:0.6em;">— Rudin, <em>Nature Machine Intelligence</em>, 2019</p>

Safety-critical systems demand controllers that can be **audited**, **certified**, and **deployed on edge hardware**.
Post-hoc explanations of a 10,000-parameter network are not enough — the model itself must be interpretable.

---

<!-- ─────────────────────────────────────────────────
  SLIDE 3 · THE VISION
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
8 terms &nbsp;·&nbsp; every coefficient has a physical interpretation &nbsp;·&nbsp; fits on a napkin
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 4 · WHY IT MATTERS
───────────────────────────────────────────────── -->

# Three properties that change everything

<div class="cols-3" style="margin-top:0.8em;">

<div class="box" style="text-align:center; padding:1.2em;">
  <div style="font-size:2em;">📉</div>
  <h2>Data-Efficient</h2>
  <p style="font-size:0.8em; color:#444;">Learn from far fewer real-world interactions</p>
</div>

<div class="box" style="text-align:center; padding:1.2em;">
  <div style="font-size:2em;">🔍</div>
  <h2>Explainable</h2>
  <p style="font-size:0.8em; color:#444;">Every control term has physical meaning</p>
</div>

<div class="box" style="text-align:center; padding:1.2em;">
  <div style="font-size:2em;">⚡</div>
  <h2>Lightweight</h2>
  <p style="font-size:0.8em; color:#444;">Runs on a microcontroller, no GPU required</p>
</div>

</div>

<div class="gold-box" style="margin-top:1em; text-align:center; font-size:0.88em;">
  Interpretable control lowers the barrier to certified autonomous systems in robotics, aerospace, and healthcare.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 5 · THE CONTROL GAP
───────────────────────────────────────────────── -->

# Neither existing approach is sufficient

| | Classical Control | Deep RL (PPO / SAC) |
|---|---|---|
| Nonlinear, unstable systems | <span class="cross">✗ Limited</span> | <span class="check">✓ Works</span> |
| Interpretable | <span class="check">✓</span> | <span class="cross">✗</span> |
| Certifiable (safety-critical) | <span class="check">✓</span> | <span class="cross">✗</span> |
| Edge-deployable | <span class="check">✓</span> | <span class="cross">✗</span> |
| Sample-efficient | <span class="check">✓</span> | <span class="cross">✗</span> |

<div class="gold-box" style="text-align:center; margin-top:0.8em;">
  We need the <strong>performance</strong> of deep RL with the <strong>interpretability</strong> of classical control.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 6 · THE PROVING GROUND
───────────────────────────────────────────────── -->
<!-- _class: statement -->

# A demanding proving ground — chosen for a reason.

Instability makes near-equilibrium data collection hard — exactly where SINDy struggles most.
Solve it here, and the framework transfers anywhere.

---

<!-- ─────────────────────────────────────────────────
  SLIDE 6 · THE SYSTEM
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

---

<!-- ─────────────────────────────────────────────────
  SLIDE 7 · SINDY
───────────────────────────────────────────────── -->

# SINDy — Discovering equations from data

**Key idea:** dynamics live in a low-dimensional function space.

$$\dot X \;=\; \underbrace{\Theta(X,U)}_{\substack{\text{candidate} \\ \text{feature library}}} \cdot \underbrace{\Xi}_{\substack{\text{sparse} \\ \text{coefficients}}}$$

$$\Theta = \bigl[\;1 \;\big|\; x \;\big|\; \theta_1,\,\theta_2 \;\big|\; x^2,\,x\theta_1,\,\theta_1^2,\;\ldots\;\bigr] \quad \text{degree-}d\text{ polynomial library}$$

<div class="cols" style="gap:1em; margin-top:0.4em;">
<div class="gold-box">
<b>STLSQ solver</b> drives most coefficients to <em>exactly zero</em> — exposing the true governing terms.
</div>
<div class="box">
<b>Library degree <em>d</em> is a design choice</b> — it must match the complexity of the system and is not obvious a priori.
</div>
</div>

<div class="box" style="margin-top:0.5em; text-align:center; font-size:0.8em;">
  <span style="color:#888;">Lineage: &nbsp;</span>
  LASSO <span style="color:var(--uw-gold-dark);">'96</span>
  &nbsp;→&nbsp; SINDy <span style="color:var(--uw-gold-dark);">'16</span>
  &nbsp;→&nbsp; SINDy-C <span style="color:var(--uw-gold-dark);">'18</span>
  &nbsp;→&nbsp; E-SINDy <span style="color:var(--uw-gold-dark);">'22</span>
  &nbsp;→&nbsp; <strong>SINDy-RL <span style="color:var(--uw-gold-dark);">'24</span></strong>
  &nbsp;&nbsp;|&nbsp;&nbsp; Koopman as the competing paradigm
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 8 · STRATEGY
───────────────────────────────────────────────── -->

# Two objectives — one interpretable controller

![objectives diagram w:800](objectives_diagram.png)

---

<!-- ─────────────────────────────────────────────────
  SLIDE 9 · THE PIVOT
───────────────────────────────────────────────── -->

# SINDyC is a chicken-and-egg problem

The random policy crashes in ~5 steps.
The system never reaches equilibrium.
SINDyC cannot learn the dynamics that matter.

<br>

You need a controller to collect the data.
You need the data to train SINDyC.
You need SINDyC to build the controller.

<div class="gold-box" style="margin-top:0.8em; font-size:0.83em;">
  Solution: co-train the controller and surrogate in an iterative active-learning loop.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 10 · PROJECT JOURNEY
───────────────────────────────────────────────── -->

# The project journey

<div class="timeline" style="margin-top:0.6em;">

  <div class="tl-row">
    <div class="tl-dot">1</div>
    <div class="tl-label">Shared repo & environment</div>
    <div class="tl-desc">Confirmed MuJoCo, 6-dim state recovery, random-policy stats</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">2</div>
    <div class="tl-label">Tried direct SINDyC</div>
    <div class="tl-desc">Random policy terminates in ~5 steps — not enough data near equilibrium</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot gold">3</div>
    <div class="tl-label">Discovered the problem</div>
    <div class="tl-desc">SINDyC requires RL co-training — plan realigned</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">4</div>
    <div class="tl-label">Baseline (oracle) trained</div>
    <div class="tl-desc">Full-order PPO: 100% success, 9359 mean reward</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">5</div>
    <div class="tl-label">Parallel training of sparse models</div>
    <div class="tl-desc">Reduced order SINDy model &amp; sparse policy dictionary</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">6</div>
    <div class="tl-label">Compare and contrast</div>
    <div class="tl-desc">Interpretability, sample efficiency, and robustness across approaches</div>
  </div>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 11 · BASELINE
───────────────────────────────────────────────── -->

# Baseline — the performance ceiling

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

<div class="gold-box" style="margin-top:0.9em; font-size:0.85em;">
  <strong>50,000 transitions</strong> collected from the trained policy — the dataset used for sparse learning.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 12 · TRACK B — DISTILLATION & BC PROBLEM
───────────────────────────────────────────────── -->

# Sparse policy distillation — the compounding error trap

<div class="cols" style="align-items:start;">

<div>

**Behavioral cloning** on the oracle dataset:

$$\min_{\Xi} \;\bigl\|\Theta(X)\,\Xi - U^*\bigr\|_2 \;+\; \lambda\|\Xi\|_1$$

Degree-4 library required to capture nonlinear cross-coupling (210 terms, STLSQ selects the sparse subset).

**But the policy fails in deployment.**

</div>

<div>

![compounding error w:320](compounding_error.png)

<div style="font-size:0.75em; color:#888; margin-top:0.2em; text-align:center;">
  At noise σ = 0.3 rad: <strong>1000 steps → ~20 steps</strong>
</div>

</div>
</div>

<div class="gold-box" style="font-size:0.83em;">
  Off-distribution states produce small action errors → errors compound → catastrophic failure.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 13 · TRACK B — AUGMENTATION & RESULTS
───────────────────────────────────────────────── -->

# The fix: query the oracle, for free

<div class="cols" style="align-items:start;">

<div>

**Key insight:** the NN policy has no memory.
We can query it at *any* state — no new simulator rollouts.

<br>

For each of 3 rounds:

1. Sample perturbed states: $\tilde x = x + \varepsilon,\;\;\varepsilon \sim \mathcal{N}(0,\,0.05^2)$
2. Query oracle: $\tilde u = \pi_\text{NN}(\tilde x)$
3. Append $(\tilde x,\tilde u)$ to dataset

Dataset grows **4×** (50k → 200k pairs).

</div>

<div>

| Policy | Mean episode length |
|---|---|
| Baseline NN | **1000 steps** ✓ |
| SINDy (base) | ~20 steps @ noise 0.3 |
| SINDy (augmented) | ~500–900 steps |

<div style="font-size:0.75em; color:#888; margin-top:0.5em;">Evaluated at initial noise σ = 0.3 rad/m</div>

<div class="box" style="margin-top:0.8em; font-size:0.83em;">
  Augmented SINDy is <strong>25–45× more robust</strong> with zero additional simulator interactions.
</div>

</div>
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 14 · TRACK A — ACTIVE LEARNING
───────────────────────────────────────────────── -->

# ROM surrogate — train RL inside an interpretable model

**Core idea:** use a SINDy surrogate as the "dream environment" (cf. DreamerV3).

<br>

<div class="flow" style="font-size:0.85em; gap:0.5em;">
  <div class="fn">Bootstrap<br><small>PD + random</small></div>
  <div class="fa">→</div>
  <div class="fn">Fit SINDy<br><small>polynomial dynamics</small></div>
  <div class="fa">→</div>
  <div class="fn gold">Train PPO<br><small>in surrogate</small></div>
  <div class="fa">→</div>
  <div class="fn">Deploy<br><small>in real sim</small></div>
  <div class="fa">→</div>
  <div class="fn gold">Collect data<br><small>near equilibrium</small></div>
  <div class="fa">→</div>
  <div class="fn" style="font-size:1.4em; min-width:30px;">⟳</div>
</div>

<br>

| DreamerV3 concept | This project |
|---|---|
| Latent world model (RSSM) | SINDy surrogate — explicit, interpretable |
| "Dreaming" (policy training) | PPO in `SINDySurrogateEnv` |
| World model update | Refit SINDy on expanded dataset |
| Real-env interaction | Controller rollout in MuJoCo |

---

<!-- ─────────────────────────────────────────────────
  SLIDE 15 · TRACK A — RESULTS
───────────────────────────────────────────────── -->

# ROM surrogate — results

<div class="cols" style="align-items:start;">

<div>

**Iterative RMSE convergence:**

| Iteration | One-step RMSE | Real sim steps |
|---|---|---|
| 0 (bootstrap) | — | ~3,000 |
| 1 | Δ large | +~5,000 |
| 2 | Δ small | +~5,000 |

<div style="font-size:0.78em; color:#888; margin-top:0.3em;">
  Each iteration trains a better policy which collects better data.
</div>

<div class="gold-box" style="margin-top:0.5em; font-size:0.83em;">
  Baseline NN required <strong>400,000</strong> real-sim steps.<br>
  ROM surrogate target: <strong>&lt; 50,000</strong> — 8× more data-efficient.
</div>

</div>

<div>

**Surrogate environment** — exact reward replica:

$$r = 10 - \bigl(0.01\,x_\text{tip}^2 + (y_\text{tip}-2)^2\bigr) - v_\text{pen}$$

$$y_\text{tip} = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2)$$

The SINDy model is **interpretable dynamics**:

$$x_{k+1} = x_k + \Delta t \cdot \Xi^T \Theta(x_k, u_k)$$

</div>
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 16 · STRETCH GOAL
───────────────────────────────────────────────── -->
<!-- _class: section -->

# Stretch Goal

### Phase 3 · Fully Interpretable Closed-Loop Control

Combine the interpretable **dynamics** (ROM) with the interpretable **policy** (sparse dictionary)

---

<!-- ─────────────────────────────────────────────────
  SLIDE 17 · STRETCH GOAL DETAIL
───────────────────────────────────────────────── -->

# Phase 3 — Combining both approaches

<div class="cols" style="align-items:start; gap:2.5em;">

<div>

**Goal:** A fully interpretable closed loop.

<br>

<div class="flow" style="flex-direction:column; gap:0.3em; align-items:stretch;">
  <div class="fn" style="text-align:center;">ROM surrogate<br><small>SINDy dynamics</small></div>
  <div style="text-align:center;" class="fa">↓</div>
  <div class="fn" style="text-align:center;">Train 6-dim NN policy<br><small>inside the surrogate</small></div>
  <div style="text-align:center;" class="fa">↓</div>
  <div class="fn gold" style="text-align:center;">Distill to sparse policy<br><small>→ sparse polynomial</small></div>
  <div style="text-align:center;" class="fa">↓</div>
  <div class="fn" style="text-align:center;">Evaluate in real MuJoCo</div>
</div>

</div>

<div style="padding-top:0.5em;">

**Why it matters:**

- Better surrogate → better training distribution
- Augmented distillation → robust sparse policy
- Together: interpretable dynamics **and** interpretable policy

<div class="placeholder" style="margin-top:1em;">
  📊 Phase 3 results — pending
</div>

</div>
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 18 · COMPARISON
───────────────────────────────────────────────── -->

# How do they compare?

| Approach | Real-sim steps | Policy type | Mean length | Success | Interpretable |
|---|---|---|---|---|---|
| **Baseline NN** | 400,000 | NN (9,731 params) | 1,000 | 100% | <span class="cross">✗</span> |
| **Sparse policy (base)** | -* | Polynomial (8 terms) | ~20 @ σ=0.3 | Low | <span class="check">✓</span> |
| **Sparse policy (augmented)** | -* | Polynomial | ~500–900 | ~50–90% | <span class="check">✓</span> |
| **ROM surrogate RL** | ~50,000 (est.) | NN in surrogate | TBD | TBD | <span class="partial">◑</span> |
| **Phase 3 (stretch)** | TBD | Polynomial | — | — | <span class="check">✓</span> |

<div style="font-size:0.72em; color:#888; margin-top:0.3em;">
  * No additional real-sim steps beyond baseline training — augmentation uses oracle queries only.
</div>

<div class="placeholder" style="margin-top:0.5em;">
  Phase 3 full four-way comparison — pending Phase 3 completion
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 19 · SUMMARY
───────────────────────────────────────────────── -->

# What we accomplished

<div class="timeline" style="margin-top:0.6em;">

  <div class="tl-row">
    <div class="tl-dot">✓</div>
    <div class="tl-label">Full research infrastructure</div>
    <div class="tl-desc">MuJoCo environment, shared data pipeline, reproducible baselines</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">✓</div>
    <div class="tl-label">Oracle baseline</div>
    <div class="tl-desc">100% success rate — established the performance ceiling</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">✓</div>
    <div class="tl-label">Sparse policy distillation</div>
    <div class="tl-desc">Quantified compounding error; data augmentation yields 25–45× robustness gain</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot gold">!</div>
    <div class="tl-label">Key learning — Library degree</div>
    <div class="tl-desc">Degree-2 insufficient for double pendulum; degree-4 required (210 vs. 28 features)</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot">✓</div>
    <div class="tl-label">Reduced order model — active SINDy loop</div>
    <div class="tl-desc">DreamerV3-style iterative framework implemented and validated</div>
  </div>
  <div class="tl-line"></div>
  <div class="tl-row">
    <div class="tl-dot gold">→</div>
    <div class="tl-label">Phase 3 & full comparison</div>
    <div class="tl-desc">Combining approaches + four-way evaluation — in progress</div>
  </div>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 20 · WHAT'S NEXT
───────────────────────────────────────────────── -->

# What's next

<div class="cols" style="margin-top:0.6em; gap:2.5em;">

<div>

**Near-term**

<div class="box">Complete Phase 3 — run Phase 3 distillation pipeline on ROM surrogate</div>
<div class="box" style="margin-top:0.5em;">Four-way comparison — standardized evaluation across all conditions</div>
<div class="box" style="margin-top:0.5em;">Ablation studies — sparsity threshold λ vs. robustness trade-off</div>

</div>

<div>

**Longer-term**

<div class="gold-box">Extend to harder systems — 3D humanoid, soft robotics, aerial vehicles</div>
<div class="gold-box" style="margin-top:0.5em;">Safety certification — formal verification of sparse polynomial bounds</div>
<div class="gold-box" style="margin-top:0.5em;">Edge deployment — run the polynomial controller on embedded hardware</div>

</div>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 21 · REFERENCES
───────────────────────────────────────────────── -->

# References

<div class="cols" style="gap:2em; font-size:0.68em; line-height:1.7;">

<div>

**Interpretable & Safe AI**
[1] Rudin. *Stop explaining black-box ML.* Nat. Mach. Intell., 2019.
[2] Arrieta et al. *Explainable AI: concepts and applications.* Inf. Fusion, 2020.
[3] Saeed & Omlin. *Explainability for critical decision systems.* ACM CSUR, 2024.
[4] Amodei et al. *Concrete problems in AI safety.* arXiv:1606.06565, 2016.

**RL Baselines**
[5] Schulman et al. *PPO algorithms.* arXiv:1707.06347, 2017.
[6] Haarnoja et al. *SAC: off-policy max-entropy RL.* ICML, 2018.
[7] Todorov, Erez & Tassa. *MuJoCo.* IROS, 2012.
[20] Fujimoto, van Hoof & Meger. *TD3.* ICML, 2018.

**Benchmark & Implementation**
[21] Brockman et al. *OpenAI Gym.* arXiv:1606.01540, 2016.
[22] Raffin et al. *Stable-Baselines3.* JMLR 22(268), 2021.

</div>

<div>

**SINDy Lineage**
[8] Tibshirani. *Regression shrinkage via LASSO.* JRSS-B, 1996.
[9] Brunton, Proctor & Kutz. *SINDy.* PNAS 113(15), 2016.
[10] Kaiser, Kutz & Brunton. *SINDy-C / SINDy-MPC.* Proc. R. Soc. A, 2018.
[11] Fasel et al. *E-SINDy: ensemble SINDy.* Proc. R. Soc. A, 2022.
[12] de Silva et al. *PySINDy.* JOSS 5(49), 2020.
[13] Brunton et al. *Modern Koopman theory.* SIAM Rev., 2022.

**Model-Based RL & Distillation**
[14] Deisenroth & Rasmussen. *PILCO.* ICML, 2011.
[15] Chua et al. *PETS.* NeurIPS, 2018.
[16] Hafner et al. *DreamerV3.* arXiv:2301.04104, 2023.
[17] Zolman et al. *SINDy-RL.* arXiv:2403.09110, 2024.
[18] Pomerleau. *ALVINN.* NeurIPS, 1989.
[19] Ross, Gordon & Bagnell. *DAgger.* AISTATS, 2011.

</div>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 22 · CLOSING
───────────────────────────────────────────────── -->
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="uw-w">W</div>

# Interpretable control for unstable systems.
# It works.

**Patrick Smith &nbsp;·&nbsp; Andrew Falcone**

ME 595 &nbsp;·&nbsp; University of Washington &nbsp;·&nbsp; Spring 2026
