# Register — advanced-methods series (F4–F6), 2026-08-28

One row per arm, updated as each lands. Bars below are fixed **before** the corresponding run.
Everything here is buildable from artifacts already on disk; nothing waits on new data.

| id | idea | needs | status | verdict |
|---|---|---|---|---|
| F4 | Sinkhorn / optimal-transport assignment instead of per-point argmax | nothing (inference-only) | built | — |
| F5 | Conformal calibration of a GT-free fit-quality signal | nothing (existing runs) | built | — |
| F6 | Geodesic-softened targets replacing one-hot vertex identity | retrain (job 3280205) | training | — |

Closed earlier and **not** revisited, each on its own evidence: equivariant backbone (F1 — the
signal already reaches the head), further selectors (F2 — rank accuracy is not error reduction),
pose initialisers (C14p — ground-truth pose is the ceiling and buys nothing).

Standing rule for all three, from this session's six decoupling instances: **every arm reports a
mechanism check alongside its endpoint.** An endpoint alone is not evidence.

---

## F4 — optimal-transport assignment

**Why.** F2 established that *per-point independent* selection is a dead end: a trained ranker beat
chance on rank accuracy and bought ~3% of the oracle gap, because the cases it fixed and broke
cancelled. OT attacks the structure that failure exposed — points compete for targets, so the
assignment is globally consistent instead of 10,235 independent argmaxes. Standard since SuperGlue.

**Method.** Sinkhorn-normalise the existing (N × V) query/key similarity before taking the
assignment. No retraining, no new data; operates on the C11 head's own output.

**Bar.** Held-out synthetic specimens, per segment, error as a fraction of segment length, against
plain argmax (`top1`).
- **PASS** — median per-point retrieval error falls on **both `tr` and `fe` individually**, sign
  p < 0.05, with **≥5% relative** reduction on each. Pooled averages do not count (F2's lesson).
- **FAIL** — otherwise. OT is not the missing structure and this line closes.
- Mechanism check: report the assignment's *entropy* and the fraction of template vertices
  receiving mass. A gain achieved by collapsing onto few vertices is a degenerate solution, not a
  correspondence improvement, and is reported as such.
- Chance/no-op control: Sinkhorn at very low regularisation must approach argmax; if it does not,
  the implementation is wrong and the arm is void.

## F5 — conformal calibration of fit quality

**Why.** Nothing in SMILify outputs uncertainty, and there is no way to know a fit is wrong on a
real scan. The catastrophic specimens (0.366, 0.560) were found only because synthetic data had
ground truth. Two ingredients already exist: a GT-free triage signal that works
(cycle-consistency degrades per-specimen, lab record §8) and 4000 synthetic specimens with ground
truth to calibrate against.

**Method.** Split-conformal prediction. Calibrate the triage signal against measured error on a
held-out synthetic calibration set; emit a per-specimen prediction interval and a flag with
distribution-free coverage guarantees. No retraining.

**Bar.** On a held-out synthetic test split disjoint from the calibration split:
- **PASS** — empirical coverage of the 90% interval lies within **[0.85, 0.95]**, *and* the
  intervals are informative: mean interval width must be **< 80%** of the width a constant
  (signal-free) predictor needs for the same coverage. Coverage alone is trivially achievable by
  predicting a huge interval, so the width condition is the one that matters.
- **FAIL** — coverage outside the band, or width ≥ 80% of the constant baseline.
- Mechanism check: report coverage **per segment** and for the worst-decile specimens separately.
  Marginal coverage that holds on average while failing exactly on the specimens a user needs
  flagged is worse than useless, and is reported as a failure of the method's purpose even if the
  marginal number passes.
- **Not claimed:** conformal guarantees hold under exchangeability. Synthetic-calibrated intervals
  transferring to real scans is an *assumption*, not a guarantee, and will be stated as such
  wherever a real-scan number is quoted.

## F6 — geodesic-softened targets

**Why.** The loss is `F.cross_entropy` against one true vertex, so on a 10,235-vertex mesh the
vertex 0.1 mm away is punished exactly as hard as one on the opposite leg. Three independent
results say that is wrong: F1 (information present at 52.9° while retrieval top-1 is 0.071), E0
(the metric's own floor is 0.0908 because near-misses are genuinely unavoidable), and C3's own
docstring (exact-vertex accuracy is "harsh on a dense mesh"; weigh 3D error instead).

**Method.** Target mass spread over the true vertex's **graph** neighbourhood — graph, not
Euclidean, because adjacent coxae are physically close but not edge-connected, and Euclidean
smoothing would teach exactly the cross-leg confusion E0 measured. Verified before launch:
σ = 0.011780 read off the mesh (median 3rd-neighbour graph distance), mean self-weight 0.172
against 0.042 uniform, and cross-leg target mass masked to exactly 0.0 (it was 0.46 on one coxa
vertex before masking).

**Bar.** Held-out synthetic retrieval, against C11's top-1 = 0.0709 and median 3D error 0.02687.
- **PASS** — **median 3D retrieval error** falls by ≥15% relative, i.e. ≤ 0.02284. This is the
  endpoint, *not* top-1: the whole premise is that exact-vertex accuracy is the wrong measure, so
  scoring on it would contradict the hypothesis being tested.
- **PARTIAL** — 3D error falls but by <15%.
- **FAIL** — 3D error does not fall.
- Mechanism check: top-1 is reported alongside and is **expected to fall or hold**, not rise. A
  soft target deliberately trades exact-vertex precision for neighbourhood correctness; top-1
  rising would mean the loss did something other than what it was designed to do.
- Fitter-level follow-up is **conditional on PASS** and is a separate pre-registration — retrieval
  gains have failed to reach the fitter three times (C11, C12, F2).
