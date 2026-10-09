# Pre-registration — R9: does the production objective prefer the anatomically CORRECT solution,
# or the production solution?

**Written and committed BEFORE the run.** Bars fixed here are not revised after seeing a number.

Date: 2026-09-08. Folder: `diagnostics/groundtruth/`.

---

## 1. The question, and why it's not answered by anything already on record

G6 ([[project-g6-case-b-confirmed]]) showed the objective can't recover correct joint *positions*
even when told where they are — an objective-vs-search question about pose. R3/R4
(`RESULTS_R3_20260908.md`, `RESULTS_R4_20260908.md`) showed named anatomical parts can be relocated
onto the real anatomy when supervision names the correct part (34.7%→0.2% error), robust across
12/12 landmarks and 11/12 specimens — an existence proof that the machinery CAN reach correct
anatomy. Neither answers the question this audit asks:

> **When the production objective (all terms, D1_PROD.yaml's Stage_3_deform_fine weights) is
> evaluated AT R3's anatomically-correct solution, does it score better or worse than at the actual
> production solution?**

R3/R4 measured only total chamfer (1.25x for CORRECT vs 1.00x production) and distance-to-part.
They never decomposed the full objective — edge, normal, laplacian, offset, sym, midline,
scale/allo, limit — term by term. That decomposition is the entire content of this audit.

## 2. Design — reuses R3's exact machinery, not a fresh build

`r3_correct_parts.py` already computes the full `comp` dict (every `MoonshotStage.forward()` term,
RAW/unweighted) at each arm's converged state via `stage.forward(src, it=0)` — it only ever
extracted `comp["chamfer"]`. This audit (`r9_objective_audit.py`) is that same script, same 12
specimens, same production-parameter source (`diagnostics/moonshot/runs/Z8_W*/Stage_3_deform_fine.npz`),
same optimization (800 Adam iterations, same landmark-constraint arms), with the FULL comp dict
captured and reported for every arm, both raw and weighted (`lw[term] * comp[term]`), plus the
total weighted objective `tot`.

Arms, unchanged from R3: **production** (actual fitted state, no further optimization), **CORRECT**
(optimized with the anatomically-correct-part landmark constraint added on top of the full
production objective), **CONSENSUS** and **FREE** reported alongside for context, gating nothing.

**STATUS (2026-09-09): CASE 2.** Production objective is 67% higher at the anatomically correct solution than at production's own fit, every term (9/9) costs more, none reverses. Full result: `RESULTS_R9_objective_audit_20260909.md`. The objective actively prefers the anatomically wrong solution -- not merely a search failure.

## 3. Decision rule — fixed now, per the three-way split already proposed in this thread

Primary comparison: **total weighted production objective**, `production` vs `CORRECT`, paired
across the 12 specimens.

- **Case 1 — CORRECT's total objective is LOWER than production's**: the objective is compatible
  with the anatomically correct solution. The remaining problem is search/initialization/discovery
  — getting the optimizer to that basin without being told the answer. Points toward Priority 2
  (semantic anatomical identity recovery), not toward redesigning the objective.
- **Case 2 — CORRECT's total objective is HIGHER than production's**: the objective itself ranks
  the anatomically wrong solution more favorably. Not merely a search failure — redesigning the
  objective becomes necessary before semantic recovery would even help, since a correct-anatomy
  detector would hand the optimizer a solution the objective actively fights.
- **Case 3 — mixed / ambiguous** (e.g. total is close, or sign is inconsistent across specimens):
  decompose per-term. Report which term(s) score better for CORRECT and which score worse. A term
  that consistently favors production over CORRECT is a candidate for "the term fighting anatomy."

**Amendment, before running, not after seeing a number**: a per-specimen paired sign test (the
X1/X2/G-series standard) is NOT available here without deeper surgery — `MoonshotStage.forward()`
batch-means every term internally (chamfer, edge, etc. are pytorch3d batch losses; scale/allo/limit
are `.mean()` over the whole batch), so there is no per-specimen total to pair on from this
instrumentation. This audit therefore produces **one total-objective number per arm** (a single
comparison, not n=12 paired points), not a significance-tested result. That is a real limitation,
stated now: a large, decisive gap (e.g. an order of magnitude, matching G6/G7's scale of effect) is
still informative; a small gap cannot be told apart from noise without per-specimen instrumentation
this script does not have. If the result comes back close, the right move is building per-specimen
extraction before trusting a verdict, not reading the sign of a single number as significant.

## 4. What this does NOT establish

CORRECT's landmark-constraint loss (`LAM * loss(v, REG["CORRECT"])`) is added ON TOP OF the
production objective during CORRECT's optimization — it is not a free re-fit under the production
objective alone. So CORRECT's total production-objective value reflects "the best the optimizer can
do under the production objective while ALSO being forced to the right anatomy," not "the unforced
optimum under the production objective that happens to land on correct anatomy." This is the right
experiment for the stated question (does the objective, evaluated there, prefer it or fight it) but
it is not evidence about whether the production objective's OWN unconstrained optimum would ever
find that point — R6/R7/R8/W5 already answer that route is closed by other means.

n=12, one run per arm, same limits as R3/R4 (reachability under full landmark supervision, not
generalization; correct-part assignment is anatomical judgement).
