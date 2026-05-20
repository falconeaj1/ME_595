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
<tr><td colspan="6"><em>"Think about where autonomous control needs to go: a surgical robot that a regulator has to certify before it touches a patient. A delivery drone flying over populated areas with no cloud connection and a microcontroller for a brain. In each of these cases, a ten-thousand-parameter neural network is a dead end — you can't certify what cannot be explained, and you can't run a GPU on a battery-powered UAV. But if the controller is a few terms, you can prove its stability bounds, audit every decision, and flash it onto embedded chips. That's what we're trying to demonstrate — the performance of deep RL, in a form that can actually be deployed and trusted. And this is the goal: a sparse governing equation where every coefficient is physically meaningful. For model-based control or faster RL training, that sparsity is critical — instead of running thousands of rollouts in an expensive full simulator, you run them in this equation. That's what SINDy gives us."</em></td></tr>

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
<tr><td colspan="6"><em>[Andrew]: "We have 2 objectives. Objective one: learn a reduced-order model. SINDyC identifies dynamics with control inputs — giving us an explicit, interpretable surrogate we can run RL inside." [Patrick]: "And, Objective two: distill the NN policy trained in that surrogate down to a sparse polynomial. The result is a controller you can write on a napkin and deploy on a microcontroller."</em></td></tr>

<tr>
<td>6</td>
<td>The obstacle: SINDyC is a chicken-and-egg problem</td>
<td>Andrew</td>
<td>State the three-line problem plainly. This was a shared discovery that reshaped the plan — both should speak to it briefly. End with the solution (iterative co-training).</td>
<td>SINDyC is a chicken-and-egg problem</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>"Before we could do any of that, we hit a wall. The random policy crashes in five steps — the system never reaches equilibrium. SINDyC cannot learn the dynamics that matter. You need a controller to collect near-equilibrium data. You need the data to train SINDyC. You need SINDyC to build the controller. We had to redesign the whole approach: co-train the controller and surrogate in an iterative active-learning loop."</em></td></tr>

<tr>
<td>7</td>
<td>Baseline: the performance ceiling</td>
<td>Patrick</td>
<td>Show the three stat boxes (100%, 9359, 9731 params). Frame this as the oracle we're trying to match — at a fraction of the parameters and data.</td>
<td>Baseline — the performance ceiling</td>
<td>0:30</td>
</tr>
<tr><td colspan="6"><em>"We started by defining the baseline. We trained a full-order PPO agent with unlimited simulator access: 100% success rate, mean reward 9,359 — essentially perfect. It took 400,000 simulator interactions and produced a 9,731-parameter MLP. That's what we're trying to match with something interpretable and much more data-efficient."</em></td></tr>

<tr>
<td>8</td>
<td>Sparse policy distillation — and the compounding error trap</td>
<td>Patrick</td>
<td>Walk through behavioral cloning formulation. Degree-4 library required — 210 terms, STLSQ selects 8. Distillation succeeds at σ=0. Show the three-curve compounding error graph (σ=0, 0.1, 0.3). Key message: small action errors accumulate — every off-distribution state makes the next one worse.</td>
<td>Sparse policy distillation — the compounding error trap</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>"Since we already have a trained expert from the PPO baseline, we get the sparse policy for free, no retraining needed. It's a one-shot regression: collect 50,000 transitions from the oracle, build the polynomial library, solve for the sparse coefficients. Degree-4 is required — the double pendulum's cross-coupling nonlinearities can't be captured at degree-2. That gives 210 library terms; STLSQ selects eight. Distillation succeeds — at σ=0 it runs 1,000 steps. But at deployment noise it falls apart. At σ=0.1 we get about 200 steps. At σ=0.3, roughly 20. The training data is near-equilibrium only. Off-distribution states produce small action errors, and those errors compound — every bad step puts us further from the training hull, making the next step worse."</em></td></tr>

<tr>
<td>9</td>
<td>The fix: query the oracle, augment the data</td>
<td>Patrick</td>
<td>Explain the key insight (NN policy is stateless — query it at any state for free). Show the 3-round augmentation loop and the before/after robustness result (25–45×). Close with the Tanh/polynomial ceiling and the polynomial actor as the path to full performance.</td>
<td>The fix: query the oracle, for free</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>"The fix suggested in the Zolman paper came from a simple insight: the NN policy has no memory — it's a pure function of state. Since we have an expert that was trained, we can essentially ask it what it would do. So we can query it at any state we want, no simulator rollouts needed. We perturb states by adding Gaussian noise (σ=0.15), query the oracle for the correct action, and add those labeled pairs to the dataset. Three rounds, four times the data — 50k to 200k pairs. STLSQ re-fit recovers cross-coupling terms that were incorrectly pruned before. Result: 25 to 45 times more robust at deployment noise, with zero additional simulator interactions. There is a ceiling though — R²≈0.97 — because a polynomial can't exactly represent the Tanh activations in the NN. To break through it, the expert itself needs to be polynomial. A polynomial actor achieves R²=0.999 with a degree-2 library — 44 terms, STLSQ retains 22 — and runs 1,000 steps at all noise levels."</em></td></tr>

<tr>
<td>10</td>
<td>ROM surrogate: SINDy dynamics + LQR control — and the PPO transfer failure that motivated LQR</td>
<td>Andrew</td>
<td>Walk through the SVG loop diagram (Bootstrap → Fit SINDy → Linearize → LQR gain → Deploy → Collect data → repeat). Then explain the two-panel roadblock/solution: (1) PPO converged inside the surrogate but exploited polynomial approximation errors — 24 steps in real MuJoCo; (2) LQR only uses the Jacobian at the upright fixed point, which is accurate in both model and real system. This is Andrew's core contribution.</td>
<td>ROM surrogate — SINDy dynamics + LQR control</td>
<td>0:50</td>
</tr>
<tr><td colspan="6"><em>"On the dynamics side: we learn sparse polynomial dynamics with SINDy, linearize around the upright equilibrium, and compute an LQR gain. We bootstrap with near-upright probe data, fit SINDy, linearize, deploy LQR in MuJoCo, collect near-equilibrium data, and repeat. But we didn't start with LQR — we first tried training PPO directly inside the polynomial surrogate. PPO converged, scoring high reward in the model. But when we deployed it in MuJoCo, it averaged 24 steps. The policy had learned to exploit the polynomial's approximation errors — actions that look optimal in the equation don't generalize to the real physics. LQR sidesteps this entirely: it only needs the Jacobian at the upright fixed point. Near equilibrium, that linearization is accurate in both the model and the real system — so there are no model errors to exploit."</em></td></tr>

<tr>
<td>11</td>
<td>ROM results — 27× data efficiency, 100% success</td>
<td>Andrew</td>
<td>Show RMSE convergence table (0.188→0.182→0.085 over three iterations, 5k/10k/15k transitions). Show LQR transfer table: all three iterations achieve 100% success, 1,000 steps, mean return 9,359 — matching the full-order baseline exactly. Gold box: 15,000 real-sim steps vs. baseline's 400,000 — 27× more data-efficient.</td>
<td>ROM surrogate — results</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>"Here are the results. The iterative framework converges — each round reduces RMSE on a fixed validation set: 0.188 at bootstrap, 0.182 after the first round, 0.085 after the second. And the LQR controller? 100% success rate at every iteration — 1,000 steps, mean return 9,359 — matching the full-order baseline exactly. Total real simulator interactions: 15,000. The baseline required 400,000. That's 27 times more data-efficient."</em></td></tr>

<tr>
<td>12</td>
<td>How do they compare?</td>
<td>Both</td>
<td>Walk the table row by row. Key callouts: (1) sparse base and augmented both carry the 400k baseline cost (*) but require no additional training; (2) polynomial actor achieves 100% but at 1M steps — more than the baseline; (3) SINDy+PPO-in-surrogate row is the roadblock — cheap data but fails; (4) SINDy-LQR wins on every dimension. Phase 3 is a one-sentence close.</td>
<td>How do they compare?</td>
<td>0:40</td>
</tr>
<tr><td colspan="6"><em>[Patrick]: "Here's the full picture. The baseline NN is the ceiling — perfect performance, but opaque and data-hungry. Behavioral cloning gives us an interpretable 8-term polynomial, essentially for free — but it's fragile at deployment noise. Augmenting the data recovers most of that robustness, 50 to 90% success, still at no additional simulator cost. The polynomial actor breaks through the Tanh ceiling and hits 100% — but it required a million simulator interactions, more than the baseline itself." [Andrew]: "On the dynamics side: we tried PPO inside the SINDy surrogate — cheap data, but the policy exploited the model and averaged 24 steps in MuJoCo. Switching to LQR from the linearized model fixed the transfer problem entirely — 100% success at 15,000 real-sim steps. That's 27 times more data-efficient than the baseline. Phase 3 would close the loop: interpretable dynamics and interpretable policy, end to end."</em></td></tr>

<tr>
<td>13</td>
<td>Stretch goal: fully interpretable closed-loop control</td>
<td>Both</td>
<td>Section divider. One sentence: combine the interpretable dynamics (ROM) with the interpretable policy (sparse dictionary).</td>
<td>Stretch Goal</td>
<td>0:10</td>
</tr>
<tr><td colspan="6"><em>"Phase 3 is the stretch goal: combine the interpretable dynamics model from the ROM track with the sparse polynomial policy from the distillation track — a fully interpretable closed loop, end to end."</em></td></tr>

<tr>
<td>14</td>
<td>Bonus: what if we started from the pendulum down position? NN PPO achieves it — SINDy-RL is the goal.</td>
<td>Patrick</td>
<td>Pose the question: all the stabilization work assumed a near-upright start — what about hanging? Show the animation as the NN baseline: hybrid PPO (swing-up → handoff → stabilizer) achieves 304 steps, 15.2 s, SUCCESS. Frame the goal explicitly: reproduce this with SINDy-RL — interpretable swing-up + interpretable stabilizer, end to end.</td>
<td>Bonus Stretch Goal</td>
<td>0:30</td>
</tr>
<tr><td colspan="6"><em>"One more question before we close. Everything we've shown assumed you start near the upright equilibrium. But what about starting all the way down — pendulum hanging, zero energy? That's a fundamentally harder problem. You have to pump energy into the system, swing the poles up through a chaotic trajectory, then hand off to a stabilizer at just the right moment. A neural network hybrid — swing-up PPO into stabilizer PPO — can do it: 304 steps, 15 seconds, success. That's the NN baseline. The goal is to do the same thing with SINDy-RL: an interpretable swing-up controller handing off to the interpretable LQR stabilizer we've already built. That's the next step."</em></td></tr>

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
- **Why degree-4 for Tanh NN but degree-2 for polynomial actor (Q&A):** The policy itself does not require degree-4 — the polynomial actor proves degree-2 is sufficient to represent a good stabilizing policy. Degree-4 was needed specifically when fitting a *Tanh NN* with STLSQ. Near equilibrium, Tanh ≈ identity and the state is near zero, so the cross-coupling terms (θ₁·θ₂, θ₁·θ̇₂, etc.) appear small in the NN's output. STLSQ's threshold treats them as negligible and prunes them — even though they matter off-equilibrium. A richer degree-4 library spreads the cross-coupling signal across more basis functions, giving STLSQ room to retain the right combination. When the expert IS a polynomial (polynomial actor), there is no projection error and no over-pruning problem — OLS recovers the true coefficients exactly at degree-2.
- **Slide 10 — LQR, not PPO** — Andrew's actual result is SINDy → linearize → LQR. PPO-in-surrogate was tested but failed in real MuJoCo (24 steps avg). Do not present PPO as the approach; LQR is the one that transfers.
- **Slide 11 numbers** — RMSE: 0.188 / 0.182 / 0.085 at 5k / 10k / 15k transitions. LQR: 100% success, 1,000 steps, mean return 9,359.88/87/90 at all three iterations. 27× data efficiency vs. baseline (15k vs. 400k).
- **Slide 12 table rows** — 7 data rows + header. Key distinctions: sparse base/augmented have * (inherits 400k, no retraining); polynomial actor has NO * (fresh 1M-step PPO run); SINDy+PPO row exists to show the roadblock (PPO exploited model errors → 24 steps); SINDy-LQR is the payoff row. The contrast between SINDy+PPO and SINDy-LQR on adjacent rows makes the design decision legible.
- **Slide 13 (Stretch Goal)** uses a light purple background (_backgroundColor: #F0EDF7, no section class). Keep it brief — 10 seconds max.
- **Slide 14 (Bonus animation)** — GIF animates in HTML export; PDF shows first frame (pendulum hanging). When recording off HTML, let the animation play through before speaking. The framing is: NN hybrid is the *baseline target* — the goal is to reproduce it with SINDy-RL (interpretable swing-up + LQR stabilizer). Don't present the NN result as the final achievement; present it as what we're trying to match interpretably.
- **Trig library experiment** (not on slides, useful for Q&A): polynomial + trig features (appending sin/cos of joint angles) was tested but achieved only R²≈0.90 — worse than pure degree-4 polynomial (R²≈0.97). The ceiling is from Tanh, not from polynomial vs. trig choice.
