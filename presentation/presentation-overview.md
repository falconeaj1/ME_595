# Presentation Overview
## SINDy-RL for Inverted Double Pendulum Control
**Target: ~9:40 · 15 slides · Patrick Smith & Andrew Falcone**

---

<table>
<thead>
<tr>
<th>#</th>
<th>Main idea</th>
<th>Presenter</th>
<th>How to communicate</th>
<th>Slide title</th>
<th>Time</th>
</tr>
</thead>
<tbody>

<tr>
<td>1</td>
<td>Title / housekeeping</td>
<td>Both on screen</td>
<td>Title slide; Patrick speaks first</td>
<td>Interpretable Control for Unstable Systems via SINDy-RL</td>
<td>0:15</td>
</tr>
<tr><td colspan="6"><em>"Welcome — I'm Patrick Smith, and this is Andrew Falcone. We're going to walk you through our project applying SINDy-RL to the inverted double pendulum. Andrew's been working on the dynamics identification side; I've been working on the control policy side. Let's get into it."</em></td></tr>

<tr>
<td>2</td>
<td>The vision: a sparse governing equation is the goal — and it's what makes fast, interpretable control possible</td>
<td>Patrick</td>
<td>Show the 8-term equation. Explain that this is what SINDy produces for the dynamics: a compact surrogate you can actually run RL inside. The sparsity is not just elegant — it's what enables model-based control and faster training. Deep RL without this surrogate requires orders of magnitude more simulator interactions.</td>
<td>What if the controller was just an equation?</td>
<td>1:00</td>
</tr>
<tr><td colspan="6"><em>"Think about where autonomous control actually needs to go: a surgical robot that a regulator has to certify before it touches a patient. A drone flying a power-line inspection with no cloud connection and a microcontroller for a brain. A spacecraft controller that has to be formally verified before launch. In every one of those cases, a ten-thousand-parameter neural network is a dead end — you can't certify what you can't explain, and you can't run a GPU on a battery-powered UAV. But if the controller is eight terms, you can prove its stability bounds, audit every decision, and flash it onto the cheapest embedded chip. That's what we're trying to build — the performance of deep RL, in a form that can actually be deployed and trusted. And this is the goal: a sparse governing equation. Eight terms, every coefficient physically meaningful. For model-based control or faster RL training, that sparsity is critical — instead of running thousands of rollouts in an expensive full simulator, you run them in this equation. That's what SINDy gives us."</em></td></tr>

<tr>
<td>3</td>
<td>The system: inverted double pendulum</td>
<td>Andrew</td>
<td>Show the pendulum diagram and state vector. State why this benchmark is appropriate: instability means a random policy never reaches equilibrium, which is exactly where data-driven dynamics identification is hardest.</td>
<td>The Inverted Double Pendulum</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>"Our testbed is the inverted double pendulum — two poles balanced upright on a cart. Six-dimensional state: cart position, two joint angles, and their rates. The only input is a horizontal cart force. What makes this hard: the upright equilibrium is unstable. A random policy crashes in about five steps — you never reach the near-equilibrium region. And that's exactly where data-driven dynamics identification is most needed."</em></td></tr>

<tr>
<td>4</td>
<td>SINDy: discovering equations from data</td>
<td>Andrew</td>
<td>Walk through the library equation (<code>Ẋ = Θ(X,U)·Ξ</code>), explain STLSQ sparsity, note that library degree <em>d</em> is a design choice — it must match the system's nonlinearities and is not obvious a priori.</td>
<td>SINDy — Discovering equations from data</td>
<td>0:55</td>
</tr>
<tr><td colspan="6"><em>"SINDy is the core of our dynamics approach. The idea: nonlinear dynamics live in a low-dimensional space of basis functions. We construct a polynomial library Θ over the state and control inputs, then use sparse regression — specifically STLSQ — to identify which terms actually drive the dynamics. Most coefficients get driven to exactly zero, leaving only the governing terms. One critical design choice is the polynomial degree — it must be rich enough to capture the system's nonlinearities, and for the double pendulum that bar is high."</em></td></tr>

<tr>
<td>5</td>
<td>Two objectives — one interpretable controller</td>
<td>Both</td>
<td>Show the objectives diagram. Andrew introduces the ROM objective (his work); Patrick introduces the sparse dictionary policy objective (his work). Natural handoff slide.</td>
<td>Two objectives — one interpretable controller</td>
<td>0:35</td>
</tr>
<tr><td colspan="6"><em>[Andrew]: "Objective one: learn a reduced-order model. SINDyC identifies dynamics with control inputs — giving us an explicit, interpretable surrogate we can run RL inside." [Patrick]: "Objective two: distill the NN policy trained in that surrogate down to a sparse polynomial. The result is a controller you can write on a napkin and deploy on a microcontroller."</em></td></tr>

<tr>
<td>6</td>
<td>The obstacle: SINDyC is a chicken-and-egg problem</td>
<td>Both</td>
<td>State the three-line problem plainly. This was a shared discovery that reshaped the plan — both should speak to it briefly. End with the solution (iterative co-training).</td>
<td>SINDyC is a chicken-and-egg problem</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>[Patrick]: "Before we could do any of that, we hit a wall. The random policy crashes in five steps — the system never reaches equilibrium. SINDyC cannot learn the dynamics that matter." [Andrew]: "But you need a controller to collect near-equilibrium data. You need the data to train SINDyC. You need SINDyC to build the controller. We had to redesign the whole approach: co-train the controller and surrogate in an iterative active-learning loop."</em></td></tr>

<tr>
<td>7</td>
<td>Baseline: the performance ceiling</td>
<td>Patrick</td>
<td>Show the three stat boxes (100%, 9359, 9731 params). Frame this as the oracle we're trying to match — at a fraction of the parameters and data.</td>
<td>Baseline — the performance ceiling</td>
<td>0:30</td>
</tr>
<tr><td colspan="6"><em>"Before we could measure progress we needed a ceiling. We trained a full-order PPO agent with unlimited simulator access: 100% success rate, mean reward 9,359 — essentially perfect. It took 400,000 simulator interactions and produced a 9,731-parameter MLP. That's what we're trying to match with something interpretable and data-efficient."</em></td></tr>

<tr>
<td>8</td>
<td>Sparse policy distillation — and the compounding error trap</td>
<td>Patrick</td>
<td>Walk through behavioral cloning formulation. Degree-4 library required — 210 terms, STLSQ selects 8. Distillation succeeds at σ=0. Show the three-curve compounding error graph (σ=0, 0.1, 0.3). Key message: small action errors accumulate — every off-distribution state makes the next one worse.</td>
<td>Sparse policy distillation — the compounding error trap</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>"We already trained the PPO baseline — so we get the sparse policy for free, no retraining needed. It's a one-shot regression: collect 50,000 transitions from the oracle, build the polynomial library, solve for the sparse coefficients. Degree-4 is required — the double pendulum's cross-coupling nonlinearities can't be captured at degree-2. That gives 210 library terms; STLSQ selects eight. Distillation succeeds — at σ=0 it runs 1,000 steps. But at deployment noise it falls apart. At σ=0.1 we get about 200 steps. At σ=0.3, roughly 20. The training data is near-equilibrium only. Off-distribution states produce small action errors, and those errors compound — every bad step puts us further from the training hull, making the next step worse."</em></td></tr>

<tr>
<td>9</td>
<td>The fix: query the oracle, augment the data</td>
<td>Patrick</td>
<td>Explain the key insight (NN policy is stateless — query it at any state for free). Show the 3-round augmentation loop and the before/after robustness result (25–45×). Close with the Tanh/polynomial ceiling and the polynomial actor as the path to full performance.</td>
<td>The fix: query the oracle, for free</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>"The fix came from a simple insight: the NN policy has no memory — it's a pure function of state. So we can query it at any state we want, no simulator rollouts needed. We perturb states by adding Gaussian noise (σ=0.15), query the oracle for the correct action, and add those labeled pairs to the dataset. Three rounds, four times the data — 50k to 200k pairs. STLSQ re-fit recovers cross-coupling terms that were incorrectly pruned before. Result: 25 to 45 times more robust at deployment noise, with zero additional simulator interactions. There is a ceiling though — R²≈0.97 — because a polynomial can't exactly represent the Tanh activations in the NN. To break through it, the expert itself needs to be polynomial. A polynomial actor achieves R²=0.999 with a degree-2 library — 44 terms, STLSQ retains 22 — and runs 1,000 steps at all noise levels."</em></td></tr>

<tr>
<td>10</td>
<td>ROM surrogate: active SINDy loop</td>
<td>Andrew</td>
<td>Explain the iterative loop (bootstrap → fit SINDy → train PPO in surrogate → deploy → collect → repeat). Draw the DreamerV3 analogy: this is an interpretable latent world model. This is Andrew's core contribution.</td>
<td>ROM surrogate — train RL inside an interpretable model</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>"On the dynamics side, the key idea is to train the RL policy inside the SINDy surrogate rather than the real simulator — the surrogate is fast, cheap to run, and runs anywhere. Instead of paying the cost of 400,000 real environment steps, you pay it once to bootstrap, then iterate almost entirely inside the equation. We bootstrap with a PD controller plus random perturbations, fit SINDy, train PPO inside the surrogate, deploy to the real simulator, collect near-equilibrium data, and repeat. Each iteration produces a better policy, which produces better data, which improves the surrogate. If you've seen DreamerV3, this is the same loop — but instead of a learned latent neural network model, the world model is an explicit polynomial equation you can read."</em></td></tr>

<tr>
<td>11</td>
<td>ROM results so far</td>
<td>Andrew</td>
<td>Show RMSE convergence table and surrogate reward equations. Be honest: framework is implemented and validated; full policy evaluation is in progress. State the data efficiency target (8× vs. baseline).</td>
<td>ROM surrogate — results</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>"The iterative framework converges — each round reduces one-step RMSE and improves policy quality in the surrogate. We're on track toward our target of under 50,000 real simulator interactions — an 8× improvement over the 400,000 the full-order baseline required. Full end-to-end policy evaluation in the real simulator is in progress."</em></td></tr>

<tr>
<td>12</td>
<td>How do they compare?</td>
<td>Both</td>
<td>Walk the comparison table row by row. Acknowledge TBD entries honestly. Mention Phase 3 stretch goal (combining both) in one sentence.</td>
<td>How do they compare?</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>[Patrick]: "Here's where things stand. The baseline NN is the ceiling — perfect, but opaque. Basic behavioral cloning is interpretable but fragile at deployment noise. Augmented distillation recovers most of the robustness — 50 to 90% success — with zero extra simulator interactions." [Andrew]: "The ROM surrogate RL is on track for 8× data efficiency; full results pending. Phase 3 stretch goal would combine both into a single interpretable closed loop."</em></td></tr>

<tr>
<td>13</td>
<td>Stretch goal: fully interpretable closed-loop control</td>
<td>Both</td>
<td>Section divider. One sentence: combine the interpretable dynamics (ROM) with the interpretable policy (sparse dictionary).</td>
<td>Stretch Goal — Phase 3 · Fully Interpretable Closed-Loop Control</td>
<td>0:10</td>
</tr>
<tr><td colspan="6"><em>"Phase 3 is the stretch goal: combine the interpretable dynamics model from the ROM track with the sparse polynomial policy from the distillation track — a fully interpretable closed loop, end to end."</em></td></tr>

<tr>
<td>14</td>
<td>Bonus: swing-up from hanging — it already works</td>
<td>Patrick</td>
<td>Play the animation. Two-agent hybrid controller: swing-up PPO brings the pendulum from hanging (θ₁≈π) to near-upright, then hands off to stabilizer PPO. 304 steps, 15.2 s, SUCCESS.</td>
<td>Bonus — swing-up from hanging initial position</td>
<td>0:30</td>
</tr>
<tr><td colspan="6"><em>"As a bonus — we went beyond just balancing. The stabilization work we've shown assumes you start near equilibrium. But what about starting from all the way down? We trained a two-agent hybrid: a swing-up PPO that pumps energy into the system from a hanging start, and the stabilizer PPO that takes over once the poles are close enough to vertical. Here it is — 304 steps, about 15 seconds, successful balance."</em></td></tr>

<tr>
<td>15</td>
<td>Closing</td>
<td>Andrew</td>
<td>Restate the thesis in one or two sentences. Both on screen.</td>
<td>Interpretable control for unstable systems. It works.</td>
<td>0:15</td>
</tr>
<tr><td colspan="6"><em>[Both]: "Interpretable control for unstable systems. It works — and the central challenge turned out not to be the math, but the data: you can't learn the dynamics you need without the controller you don't yet have."</em></td></tr>

</tbody>
</table>

**Total: ~9:40**

---

## Notes

- **Handoff at slide 5** (objectives diagram): Andrew introduces the ROM side; Patrick introduces the sparse policy side. Natural transition between the two workstreams.
- **Degree-4 is Patrick's finding only** — degree-4 was required for the policy library. The dynamics library degree for SINDyC is a separate question. Andrew should not foreshadow a degree finding he doesn't own; the degree-4 payoff lives entirely in slide 8 (Patrick).
- **Slide 8 focuses on distribution shift / compounding errors only** — the Tanh/polynomial mismatch is introduced on slide 9 as the explanation for the R²≈0.97 ceiling after augmentation. Keep these two problems on their respective slides.
- **Slide 9 polynomial actor note** — degree-2 sufficient because the actor IS a degree-2 polynomial by construction (no Tanh). OLS recovers it exactly (R²=1.000); STLSQ at thr=0.01 retains 22/44 terms at R²=0.999.
- **Slide 13 (Stretch Goal)** is a section divider (_class: section, purple background). Keep it brief — 10 seconds max.
- **Slide 14 (Bonus animation)** — GIF animates in HTML export; PDF shows first frame (pendulum hanging). When recording off HTML, let the animation play through before speaking.
- **Trig library experiment** (not on slides, useful for Q&A): polynomial + trig features (appending sin/cos of joint angles) was tested but achieved only R²≈0.90 — worse than pure degree-4 polynomial (R²≈0.97). The ceiling is from Tanh, not from polynomial vs. trig choice.
