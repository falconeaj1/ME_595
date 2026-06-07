# Final Report

**URL:** https://canvas.uw.edu/courses/1903055/assignments/11188992

---

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




# Instructor feedback from prelim-report-v2

let's address instructor feedback:

---
Introduction & Motivation
8 / 8 pts
applied rubric item
− 0 pts
Grading comment:
Sufficient motivation, including how the results could be used (with further work) to improve lives, increase productivity, etc. 

Grading comment:
Good introduction.

---

Question 3
Technical Background
11 / 12 pts
applied rubric item
− 1 pt
Grading comment:
Undefined variables or inconsistent notation

Grading comment:
Don't forget to actually define x (little x), since it doesn't appear in figure 1. Also define U* in section 2.4.
I was glad to see the footnote defining control-affine, given the intended audience.
Your graphics are very nice-looking.
In Figure 2, should the equations in the purple box be u_n = pi(Theta Xi), where pi represents the policy?
Do you define or explain the term "NN Oracle" before its use in the last sentence of section 2.4?

Good technical background given the page limit for this assignment. For the final paper, I think that section 2.4 could use more elaboration.

---

Question 4
Methods
13 / 15 pts
applied rubric item
− 2 pts
Grading comment:
Includes undefined jargon, acronyms, or other language not appropriate for intended broad engineering audience

Grading comment:
It would be nice if the methods section started with a few sentences to recap and provide context for the following subsections. I had to go back to the introduction to figure out what "Baseline" referred to.
Overall, the methods section could use more narrative "glue". I appreciate that you are trying to get a lot of technical details across with a page limit, but without additional context, this section is quite difficult to follow. It's OK if some of that narrative glue is more-or-less repeated from the technical background or introduction sections; strategic repetition is good for readability.
The term "distillation" shows up in section 3.1 without having been previously defined or described. I also don't think that a single sentence warrants its own subsection.

---

Question 5
Preliminary Results
16 / 16 pts
applied rubric item
− 0 pts
Grading comment:
Sufficient description of approaches taken and results to date. 

Grading comment:
You've made good progress, given that SINDy-RL is a deceptively simple wrapper around a few complicated interlocking pieces.

Similar comment to the methods; this section could use a few sentences of overview before diving into the specific sections, to prepare your reader for the more technical details.

I'm not convinced that the IDP requires 690 coefficients; to me this indicates that further tuning is needed of hyperparameters or that STLSQ might be doing something undesirable. I recognize that doing better than this might be outside the scope of this quarter's project, so this comment is meant to push you to consider what might be causing this, rather than insisting that you learn a sparser expression. I would be interested in seeing the distribution of coefficients, with an eye towards seeing how many of them are very small.

Thank you for including a section on the challenges you faced. I had some trouble following the surrogate exploitation section, having to read it twice to understand; if this is an important point, I recommend revising the wording for this point.

---


