# Final Report

**URL:** https://canvas.uw.edu/courses/1903055/assignments/11188992

---

## Assignment Details

Final Report
    
Due: Mon Jun 8, 2026 11:59pmDue: Mon Jun 8, 2026 11:59pm6/8/2026 (2026-06-09T06:59:59.000Z)Ungraded, 80 Possible Points80 Points PossibleAdd CommentDetailsSubmit on [gradescope](http://gradescope.com); please remember to add all team members when submitting. The final project report should be no more than 10 pages, not including the optional bibliography and nomenclature sections. Remember to review and incorporate feedback from previous assignments. This report should stand alone; do not assume that your audience has seen previous assignments.

Your report should include the following sections and information:

1. **Title, authors, date**
2. [Abstract](https://canvas.uw.edu/courses/1903055/pages/recipe-for-an-abstract)
3. **Introduction**

- Provide background information on the field, and motivation for the project.
- Give a short summary of what your goals were and what you tried or accomplished, which will be expanded on in the rest of the report.
- Either here or later in the report, discuss ethics and safety considerations around general approaches similar to your project. This should include both potential benefits and risks.
4. **Background**

- Use this section to get your audience up to speed on key technical concepts and demonstrate to the teaching team that you fully understand the algorithms you implemented.
- Give readers context of how the technical background connects to the methods and goals of your project.
- If this section has a lot of math, consider including a nomenclature section (glossary) at the beginning or end of your report.
5. **Methods**

- Describe the methods you used, including approaches that worked and those that didn’t.
- Methods sections can be dry and confusing: give some context for how these specific methods relate to your goals and results, to keep the reader oriented. Your goal is to be concise, while also providing all the necessary information for someone to replicate your work, even if they don't have access to your code.
- Provide the names of any software packages used, including version.
6. **Results**or **Demonstration of X Method**

- What goes in this section will depend quite a bit on how far you got in your implementation, but should have a strong connection to your goals and your methods.
- Display your results. Depending on your project, you will likely need 2-5 figures (graphs, tables, annotated images). Use tables sparingly; plots are a better choice if you are interested in trends.
- Remember that people who are skimming through papers and reports may look at figures but not read the text, so it’s worth spending some extra time to clearly display your main results and write descriptive captions.
7. **Summary**

- Recap your report.
- This section is in some ways a slightly expanded and rephrased version of your abstract, but it is also a place where you can speculate a little bit about other ways that you could have approached the problem, and what you would do in the future if you were continuing with this work.
8. **Link to code** **repository**
9. [CRediT statement](https://credit.niso.org/)
10. (Optional)**Bibliography**

- Cite sources for code or data that you did not write yourself. ****
- You may want to also cite sources of further information to support your methods or technical background sections, or to support statements of fact in your introduction.


---

## Instructor feedback from prelim-report-v2


**Introduction & Motivation**
>8 / 8 pts
>applied rubric item
>− 0 pts
>Grading comment:
>Sufficient motivation, including how the results could be used (with further work) to improve lives, increase productivity, etc. 
>
>Grading comment:
>Good introduction.


**Question 3**
>Technical Background
>11 / 12 pts
>applied rubric item
>− 1 pt
>Grading comment:
>Undefined variables or inconsistent notation
>
>Grading comment:
>Don't forget to actually define x (little x), since it doesn't appear in figure 1. Also define U* in section 2.4.
>I was glad to see the footnote defining control-affine, given the intended audience.
>Your graphics are very nice-looking.
>In Figure 2, should the equations in the purple box be u_n = pi(Theta Xi), where pi represents the policy?
>Do you define or explain the term "NN Oracle" before its use in the last sentence of section 2.4?
>
>Good technical background given the page limit for this assignment. For the final paper, I think that section 2.4 could use more elaboration.


**Question 4**
>Methods
>13 / 15 pts
>applied rubric item
>− 2 pts
>Grading comment:
>Includes undefined jargon, acronyms, or other language not appropriate for intended broad engineering audience
>
>Grading comment:
>It would be nice if the methods section started with a few sentences to recap and provide context for the following subsections. I had to go back to the introduction to figure out what "Baseline" referred to.
>Overall, the methods section could use more narrative "glue". I appreciate that you are trying to get a lot of technical details across with a page limit, but without additional context, this section is quite difficult to follow. It's OK if some of that narrative glue is more-or-less repeated from the technical background or introduction sections; strategic repetition is good for readability.
>The term "distillation" shows up in section 3.1 without having been previously defined or described. I also don't think that a single sentence warrants its own subsection.


**Question 5**
>Preliminary Results
>16 / 16 pts
>applied rubric item
>− 0 pts
>Grading comment:
>Sufficient description of approaches taken and results to date. 
>
>Grading comment:
>You've made good progress, given that SINDy-RL is a deceptively simple wrapper around a few >complicated interlocking pieces.
>
>Similar comment to the methods; this section could use a few sentences of overview before diving into the specific sections, to prepare your reader for the more technical details.
>
>I'm not convinced that the IDP requires 690 coefficients; to me this indicates that further tuning is needed of hyperparameters or that STLSQ might be doing something undesirable. I recognize that doing better than this might be outside the scope of this quarter's project, so this comment is meant to push you to consider what might be causing this, rather than insisting that you learn a sparser expression. I would be interested in seeing the distribution of coefficients, with an eye towards seeing how many of them are very small.
>
>Thank you for including a section on the challenges you faced. I had some trouble following the surrogate exploitation section, having to read it twice to understand; if this is an important point, I recommend revising the wording for this point.

---

## Throughline

The problem. Safety-critical systems need controllers that can be inspected, verified, and deployed on constrained hardware. Deep RL delivers capable but opaque neural networks. SINDy-RL promises an alternative: a Dyna loop that co-trains a sparse surrogate and a neural policy using a fraction of the real-environment interactions, then distills the result into a compact, readable closed-form expression. Three goals: data economy, reduced-order policy, interpretability.

The stress test. The inverted double pendulum is a demanding benchmark — two coupled unstable modes, a narrow region of attraction, and dynamics that punish any gap between the surrogate and reality. If SINDy-RL can deliver all three goals here, the framework is credible for safety-critical deployment.

What we found. Two non-obvious engineering obstacles had to be resolved first: a degree-2 polynomial library lacks the capacity to express IDP's inter-modal coupling terms (fix: degree-3), and ensemble surrogates built on a shared basis cannot self-detect shared extrapolation errors through internal disagreement (fix: uncertainty penalty plus real-environment rollback, together). With those resolved, the core question became library and optimizer selection — both explicit inputs to Algorithm 1, left to the practitioner.

The default polynomial library achieves data economy and reduced order but fails at interpretability. The feature basis is ill-conditioned (κ ≈ 2.4×10⁴): STLSQ retains 160 of 165 terms, and a full threshold sweep confirms this cannot be tuned away. Pushing harder — stricter convergence criterion, more data per iteration, 270k real steps — does not close the gap. The density is a property of the polynomial representation on this system, not a data or tuning problem.

The Lagrangian library replaces generic monomials with atoms hand-crafted from the IDP Euler-Lagrange equations. Every retained term is a physically meaningful quantity — gravitational restoring forces, Coriolis coupling, velocity damping — so interpretability is delivered by the basis itself, not by thresholding. SAC, as an off-policy optimizer, accumulates broader state-space coverage across Dyna iterations and converges faster.

The result. SAC + Lagrangian, following Algorithm 1, achieves all three goals simultaneously: 17.6× data economy (22,723 real steps vs. 400,000 for the full-order baseline), a 29-term reduced-order controller that is 336× smaller than the baseline neural network, and interpretability by construction — a policy every term of which a practitioner can read and reason about physically. The distilled sparse policy achieves 100% task success.

SINDy-RL passes the stress test. The finding about the polynomial library is not a failure of the framework — it is a result about library selection: for mechanical systems with known Lagrangians, a physics-informed basis is the appropriate choice of Θ, and it is precisely the kind of practitioner judgment Algorithm 1 was designed to accommodate.


---

## Obstacles

- degree 2 -> degree 3
- Uncertainty Penalization Alone Is Insufficient -> plus rollback trigger
- exploit
  
- the teacher must be the cross-validated best checkpoint, not the final loop policy
- degree-2 distillation fails for the same reason as degree-2 surrogate fitting
- perturbation augmentation was necessary to close distribution shift
- the sin/cos observation encoding is ill-conditioned for polynomial regression near the upright equilibrium
- the Dyna datastore provides insufficient coverage for behavioral cloning