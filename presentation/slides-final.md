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

/* Center standalone block images */
img[alt~="chicken-egg"] { display: block; margin: 0.3em auto; }
img[alt~="swing-up"]    { display: block; margin: 0.4em auto; }
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
8 terms &nbsp;·&nbsp; every coefficient has a physical interpretation &nbsp;·&nbsp; fits on a napkin
</div>

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
  SLIDE 5 · OBJECTIVES
  Both — 0:35
───────────────────────────────────────────────── -->

# Two objectives — one interpretable controller

<br>
<br>

<img src="objectives_diagram.png" style="display:block; margin:0.3em auto; width:800px;" alt="chicken-egg diagram">

---

<!-- ─────────────────────────────────────────────────
  SLIDE 6 · CHICKEN-AND-EGG
  Both — 0:50
───────────────────────────────────────────────── -->

# SINDyC is a chicken-and-egg problem

You can't train near equilibrium without reaching it — and you can't reach it without a controller.

![chicken-egg diagram w:380](chicken_egg_diagram.png)

<div class="gold-box" style="font-size:0.83em;">
  Solution: co-train the controller and surrogate in an iterative active-learning loop.
</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 7 · BASELINE
  Patrick — 0:30
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
  SLIDE 8 · SPARSE POLICY DISTILLATION
  Patrick — 0:50
───────────────────────────────────────────────── -->

# Sparse policy distillation — the compounding error trap

<div class="cols" style="align-items:start;">

<div>

**Behavioral cloning** on the oracle dataset:

$$\min_{\Xi} \;\bigl\|\Theta(X)\,\Xi - U^*\bigr\|_2 \;+\; \lambda\|\Xi\|_1$$

Degree-4 library: 210 terms → STLSQ selects **8 terms** ✓

**But the policy fails in deployment.**

<div style="font-size:0.75em; color:#888; margin-top:0.2em; text-align:center;">
  At noise σ = 0.3 rad: <strong>1000 steps → ~20 steps</strong>
</div>

</div>

<div>

![compounding error w:400](compounding_error.png)

| Noise σ | Mean episode length |
|---|---|
| 0 (training) | ~1000 steps ✓ |
| 0.1 rad | ~200 steps |
| 0.3 rad | ~20 steps ✗ |

</div>
</div>

<div class="gold-box" style="font-size:0.83em;">
  Off-distribution states produce small action errors → errors compound → catastrophic failure.
</div>


</div>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 9 · THE FIX
  Patrick — 0:40
───────────────────────────────────────────────── -->

# The fix: query the oracle, for free

<div class="cols" style="align-items:start;">

<div>

**Key insight:** the NN policy is a pure function of state — no memory, no rollouts needed. Query it at *any* state we choose.

For each of 3 rounds:

1. Perturb states: $\tilde x = x + \varepsilon,\;\;\varepsilon \sim \mathcal{N}(0,\,0.15^2)$
2. Query oracle: $\tilde u = \pi_\text{NN}(\tilde x)$
3. Append $(\tilde x,\tilde u)$ to dataset

Dataset grows **4×** (50k → 200k pairs). STLSQ re-fit recovers cross-coupling terms.

</div>

<div>

|  | σ = 0.1 | σ = 0.3 |
|---|---|---|
| Base policy | ~200 steps | ~20 steps ✗ |
| Augmented | ~1000 steps ✓ | ~500–900 steps |

<div style="font-size:0.75em; color:#888; margin-top:0.4em;">Baseline NN: 1000 steps at all noise levels</div>

<div class="box" style="margin-top:0.6em; font-size:0.83em;">
  <strong>25–45× more robust</strong> — zero additional simulator interactions.
</div>

<div class="gold-box" style="margin-top:0.5em; font-size:0.74em;">
  R²≈0.97 ceiling: Tanh NN ≠ polynomial — structural mismatch.<br>
  <span class="check">✓</span> <strong>Polynomial actor</strong>: degree-2 library, 44→22 terms (STLSQ) → R² = 0.999, 1000/1000 steps at all noise.
</div>

</div>

</div>


---

<!-- ─────────────────────────────────────────────────
  SLIDE 10 · ROM SURROGATE — ACTIVE LOOP
  Andrew — 0:50
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
  SLIDE 11 · ROM RESULTS
  Andrew — 0:40
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
  SLIDE 12 · COMPARISON
  Both — 0:40
───────────────────────────────────────────────── -->

# How do they compare?

| Approach | Real-sim steps | Policy type | Mean length | Success | Interpretable |
|---|---|---|---|---|---|
| **Baseline NN** | 400,000 | NN (9,731 params) | 1,000 | 100% | <span class="cross">✗</span> |
| **Sparse policy (base)** | 400,000 | Polynomial (8 terms) | ~20 @ σ=0.3 | Low | <span class="check">✓</span> |
| **Sparse policy (augmented)** | 400,000* | Polynomial | ~500–900 | ~50–90% | <span class="check">✓</span> |
| **ROM surrogate RL** | ~50,000 (est.) | NN in surrogate | TBD | TBD | <span class="partial">◑</span> |
| **Phase 3 (stretch)** | TBD | Polynomial | — | — | <span class="check">✓</span> |

<div style="font-size:0.72em; color:#888; margin-top:0.3em;">
  * No additional real-sim steps beyond baseline training — augmentation uses oracle queries only.
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
  SLIDE 13 · Bonus
───────────────────────────────────────────────── -->

# Bonus

Make it works from down initial position

![swing-up w:250](swing_up_animation.gif)

<div class="gold-box" style="text-align:center; font-size:0.85em; margin-top:0.4em;">

**Currently**: Swing-up PPO → handoff → Stabilizer PPO &nbsp;·&nbsp; 304 steps (15.2 s) &nbsp;·&nbsp; <strong>SUCCESS</strong>

</div>

---

<!-- ─────────────────────────────────────────────────
  SLIDE 14 · CLOSING
  Andrew — 0:15
───────────────────────────────────────────────── -->
<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

<div class="uw-w">W</div>

# Interpretable control for unstable systems.
# It works.

**Patrick Smith &nbsp;·&nbsp; Andrew Falcone**

ME 595 &nbsp;·&nbsp; University of Washington &nbsp;·&nbsp; Spring 2026
