# Pre-registration — F2: a *trained* 1-of-5 selector, barred at the fitter level

Written **before** the selector is built or run, 2026-08-28.

## The gap, and the number the bar is anchored to

Every ranking attempt so far used **one existing heuristic score directly as the ranker**: cosine
similarity (flat, uninformative) and cycle-consistency. None trained a selector for the task.
F1 supplies the missing precondition: circumferential angle is linearly decodable from `fp1` at
52.89° against a 90° chance level, so the raw material a learned ranker needs is demonstrably
present in the representation.

Measured baselines, re-read from `out_multihyp_20260827/multihypothesis.json` rather than recalled
(the figure quoted in discussion was 24%/20%; the actual values are):

| segment | top1 err | oracle@5 err | cycle err | **gap closed by cycle** |
|---|---|---|---|---|
| `fe` | 0.1991 | 0.1361 | 0.1821 | **27.0%** |
| `tr` | 0.1351 | 0.0964 | 0.1290 | **15.7%** |

The original multi-hypothesis pre-registration set H2 at "a selector closes ≥ 30% of the
top1 → oracle@5 gap". Cycle-consistency failed that on both segments. F2 keeps that threshold.

## Design

Small classifier (gradient-boosted trees) scoring each of the 5 candidates per point, trained on
measured recoverability — i.e. supervised by which candidate is actually nearest the true vertex.
Features are ones already computed in `multihypothesis_feasibility_20260827.py`, **combined for the
first time** rather than used singly:

- embedding similarity **margin** (candidate score minus the next candidate's, not the raw score —
  cosine's absolute value was already shown to be flat)
- cycle-consistency distance (the current best single selector)
- **local consistency**: agreement between this point's candidate and the candidates chosen by its
  k-nearest neighbours on the target surface. Note `coherence` already exists as a *standalone*
  selector and failed (fe 0.1955, barely below top1's 0.1991); what is untried is combining it,
  where an isolated point's unreliable vote and a whole patch agreeing are different signals.
- candidate rank, and the segment/chain-position context already available per point

**Split at the SPECIMEN level, never the point level.** Points from one specimen are not
independent — a point-level split leaks a specimen's geometry across train and test and would
inflate accuracy. Held-out specimens only, same tail split as C3/C11/F1.

## Two-tier bar, both fixed now

**Tier 1 — retrieval.** The selector must close **≥ 40%** of the top1 → oracle@5 gap on **both**
`tr` and `fe`. This clears the original 30% threshold *and* beats cycle's measured 27.0% / 15.7%
by a real margin, so "slightly better than cycle" cannot be reported as success.

**Both tiers are scored PER SEGMENT, and a pooled average never counts as a pass.** The two
segments start from very different places — cycle closes 27.0% on `fe` but only 15.7% on `tr` — so
a classifier trained across both could ride `fe`'s cleaner signal to a pooled pass while leaving
`tr` exactly where cycle left it. That is the same aggregate-hides-segment-failure shape that the
n=12 `leg_acc` artefact and the `ta`-is-limiting inference both had. **`tr` and `fe` must each
clear independently, on both tiers.** A `fe`-only pass is reported as a `fe`-only result.

**Tier 2 — the fitter, and this is the one that decides.** Selected correspondences fed through the
same D1 pipeline must move `leg_acc`/`seg_acc` beyond what top-1-by-cycle-consistency already
achieves, paired per specimen at n=48, sign + Wilcoxon + paired-t.

- **PASS** — Tier 1 **and** Tier 2 both clear **on both segments individually**, with `leg_acc`
  improving at sign p < 0.05.
- **DECOUPLED** — Tier 1 clears, Tier 2 does not. This is **not a win**. It is the *fourth*
  instance of this investigation's recurring pattern (C11/C12 retrieval-vs-fitter; bench50's
  geometric-fit-vs-correspondence framing; H_A's surface-vs-skeleton), and will be reported under
  that heading, not as a selector result.
- **FAIL** — Tier 1 does not clear. The ~30% headroom is not capturable by a learned ranker on
  these features, and the multi-hypothesis line closes.

Mechanism check, mandatory regardless of outcome (the H_A lesson — that arm cleared its endpoint by
127% and was still wrong): report `deform_verts` magnitude and coxa `joint_rot` error vs GT
alongside the endpoint, so a surface-metric gain bought by deformation cannot be mistaken for an
anatomical one.

## What will NOT be claimed

- Nothing about real scans; this is the synthetic corpus throughout.
- No claim that Tier-1 success implies anything about the fitter — that is exactly what Tier 2
  exists to test, and three prior instances say the implication does not hold.
- Oracle@5 is a ceiling built with ground truth and is never a deployable number.
