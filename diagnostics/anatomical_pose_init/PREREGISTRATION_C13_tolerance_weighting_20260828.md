# Pre-registration — C13: tolerance-weighted dense correspondence

Written **before** the arm is generated or run. 2026-08-28.

## What the evidence established first (all before this document)

Four probes, in order, each one correcting the premise of the last:

1. `leg_acc` **does** move under correspondence. The session-long "nothing has ever touched
   leg_acc, not even the dense oracle" claim was an **n=12 power artifact**, the same failure mode
   as the earlier optimistic n=12 seg_acc number. At n=48: **+0.0578, 34/48, sign p=0.0055,
   Wilcoxon p=5.4e-05** — a *larger* effect than seg_acc (+0.0310).

2. The residual is **not** a wrong-basin problem. Correspondence rescues precisely the
   catastrophic specimens (synth_012 0.560→0.936, synth_036 0.694→0.972; specimens below 0.8:
   10→1, below 0.7: 3→0), and the leg confusion matrix puts ~0.000 mass on non-adjacent legs.

3. The addressable residual is **proximal**, against a *measured paired ceiling* (GT mesh
   substituted for the fit on the same 48 specimens): `co` 57.8% + `tr` 27.7% = **85.5%**.
   Distal (`ti`+`ta`+`pt`) is **4.8%**, and correspondence already improves distal *most*.

4. The mechanism is **tolerance, not correspondence**. The coxa is the *best-placed* segment in
   absolute terms (0.0201 of body diagonal vs `pt`'s 0.0397) but sits at
   **risk = placement_error / inter-leg_tolerance = 0.995**, against ~0.21 for every other
   segment. Its error is **92% rigid offset**, and correspondence moved that offset by 3%
   (0.0192→0.0186) while nearly halving it distally.

Not expressiveness: `make_synth_corpus.py` generates targets **from the model**, so parameters
placing the coxa exactly exist in the search space. This is an optimization/weighting failure.

## Hypothesis

The dense term is `mean_v ||fitted_v − target_v||²`, isotropic in absolute model units. Its
per-vertex gradient is proportional to the residual, which is *smallest* exactly where the
tolerance is *tightest*. The fitter therefore trades coxal precision for distal precision, and
that trade is wrong for `leg_acc`. Weighting the residual by inverse inter-leg tolerance —
optimizing **risk** rather than absolute error — should reallocate gradient to the coxa.

Literature anchor: this is the OKS (object keypoint similarity) construction from COCO keypoint
evaluation, where each keypoint carries a measured per-keypoint tolerance sigma precisely because
a fixed absolute error means different things at different keypoints. Applied per-vertex here,
with the tolerance **measured** (`interleg_tolerance_20260828.py`) rather than assumed.

## Arms

All three share C11 correspondences, the P48 corpus, and every other setting. They differ only
in the per-vertex weight `w_v` on the dense term (which is a weighted mean, so `w` scale is
irrelevant — only its *shape* across vertices matters):

| arm | weight |
|---|---|
| `C13_uniform` | `w_v = 1` (reproduces the current `P48_cse_all` mechanism) |
| `C13_invtol`  | `w_v = 1 / tol_seg(v)` |
| `C13_invtol2` | `w_v = 1 / tol_seg(v)²` — the exact translation of "minimize squared risk" |

`tol_seg` is the measured per-segment inter-leg tolerance, a corpus constant, **not** anything
derived per-specimen from ground truth — so nothing here leaks target information into the fit.
`invtol2` puts ~91× more weight on `co` than `pt`; `invtol` ~9.5×. Both are registered because
the aggressive variant may destabilize, and I want that distinguishable from "the idea is wrong."

## Readings fixed in advance

Primary endpoint: **paired per-specimen `co` leg-level error**, `C13_invtol*` vs `C13_uniform`,
n=48, reported as sign test + Wilcoxon + paired-t together.

- **PASS** — `co` leg error falls by **≥ 0.05 absolute** (0.404 → ≤ 0.354) with sign p < 0.05.
  0.05 is ~16% of the 0.309 addressable coxal residual; below that the reallocation is not worth
  the complexity it adds.
- **PARTIAL** — `co` leg error falls with sign p < 0.05 but by < 0.05 absolute. Mechanism is real
  but the ceiling on it is low; report as a bounded effect, do not ship.
- **FAIL** — `co` leg error does not fall, or rises. Then coxal placement is NOT gradient-starved,
  and the remaining explanation is that the coxa's rigid offset is set by a part of the parameter
  space the dense term cannot steer (joint regressor / thorax betas) even when weighted. That
  redirects to shape-space, and I will say so rather than re-running variants.

Guard-rails, fixed now so a coxal win cannot be bought silently:

- **Distal must not regress.** If `ti`/`ta`/`pt` leg error rises by > 0.02 absolute, the arm is
  reported as a **trade**, not an improvement, regardless of the coxal number.
- **Overall `leg_acc` must not fall.** A coxal gain paid for elsewhere is not a result.
- `seg_acc` reported alongside, but it is **not** the endpoint here and will not be used to
  claim success.

## What I will NOT claim

- Not that this addresses `leg_acc`'s remaining gap in general — it targets 57.8% of it.
- Not any conclusion about `ta`, circumferential retrieval, or the CSE head's quality. C13 changes
  **only** how the existing correspondences are weighted; the network is untouched.
- Not a real-scan claim. P48 is synthetic and model-generated; bench50 is a separate question.
