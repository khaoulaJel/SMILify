# Pre-registration — X2: does an externally-validated allometric prior fix the anterior
# anatomy where the scale cap didn't?

**Written and committed BEFORE the runs.** Bars fixed here are not revised after seeing a number.

Date: 2026-09-07. Folder: `diagnostics/anterior_mechanism/`.

---

## 1. The question

X1 (`RESULTS_X1_scalecap_mechanism_20260830.md`) showed that `scale_cap` (`w_scale=0.052`, a barrier
on `log_beta_scales`) reproduces its own proxy (anterior scale-tail compression) but leaves the
mechanism unmoved (head deform ratio 0.793→0.802, sign p=0.32). The diagnosis in that result: the
term punishes distance from a magnitude of 1.0 in log-space — it has no notion of what an ant's head
*should* be sized relative to its body, so it can suppress outliers without teaching the model
anything about plausible anatomy.

The Atta head-width replication (`REPORT_ATTA_HEADWIDTH.md`) subsequently produced something X1
didn't have: a **content-bearing** constraint. The model's own fits, once measured, reproduce a
real published allometric law — `log₁₀(HW) = 1.2341·log₁₀(BL) − 0.5012` (reference, n=20,
R²=0.9868) vs. this repo's replicated `log₁₀(HW) = 1.1888·log₁₀(BL) − 0.4700` (R²=0.9872,
residual scatter 0.0232 log-units, identical to the reference's own noise). That is an externally
sourced anatomical prior, not an arbitrary constant.

> **H2: penalizing deviation from the validated allometric line moves the head-carriage mechanism
> that penalizing deviation from a scale magnitude did not.**

Design principle, carried over unchanged from X1: **the proxy is the positive control for the
mechanism.** If the new proxy (fitted ratio → allometric line) doesn't move, the mechanism reading
is void, exactly as X1 treated its own proxy.

## 2. Constraints discovered before designing this, each a live trap

1. **`head_width` is not differentiable today.** `b_h_l`/`b_h_r` are Blender-only reporter bones
   (inverse-distance-weighted to each bone's 10 nearest rest-pose vertices, added post-hoc via the
   SMIL Model Importer addon) — they do not exist in the pkl's `J_regressor`/`J_names`, and grep of
   `fitter_3d/` and `smal_model/` for `head_width|b_h_l|b_h_r` returns nothing. Today HW is only ever
   produced by a Blender export → CSV → offline numpy script
   (`diagnostics/atta_reference/compare_arms.py`). **A loss term needs HW inside the autograd graph
   every step; this doesn't exist and must be built (§3, Step 0) before anything else is possible.**
2. **`body_length` IS differentiable already**, just not assembled anywhere as a named quantity.
   `b_t` and `b_a_5` are ordinary trained joints in `J_regressor`; `joints = J_regressor @ verts` is
   already computed in `smal_model/smal_torch.py:394-399` every forward pass. `body_length =
   ‖joints[b_t] - joints[b_a_5]‖` is a two-line addition, not new machinery.
3. **The reference coefficients, not the replicated ones, must anchor the penalty.** The replicated
   line (exponent 1.1888) was fit from specimens that were themselves produced under `w_scale=0.052`
   or its precursors — using it as the target would be circular (penalizing the model for deviating
   from its own past output). The penalty target is the **reference** line (exponent 1.2341/1.2352,
   intercept −0.5012/−0.5019), an external constraint, sourced independent of any SMILify fit.
4. **The allometric law was validated on Atta specifically (20 specimens).** Applying it as a prior
   during fitting of a general/mixed-genus corpus assumes the HW~BL exponent generalizes across
   genera. This is untested and must be flagged, not assumed — see §8.

## 3. Design

### Step 0 — build and validate the differentiable head-width reporter (prerequisite, gates everything else)

**STATUS (2026-09-08): PASSED.** After the noise-floor-relative bar below FAILED (0/20) against
the original, provenance-mismatched CSV, a fresh `b_h_l`/`b_h_r` placement + CSV export was done in
one uninterrupted Blender session against a matched npz — mean \|rel err\| dropped from 4.09% to
**0.070%**, max from 8.32% to **0.121%**, 20/20 specimens within ±1%, well inside the fitter's own
measured noise floor. Confirms the original gap was a stale-snapshot artifact, not a reporter bug or
a structural Blender-vs-numpy discrepancy. Full result:
`RESULTS_X2_step0_reporter_validation_rematch_20260908.md`. **Step 1 is unblocked.**

- Replicate the Blender addon's bone placement in torch: for each of `b_h_l`/`b_h_r`, find its 10
  nearest **rest-pose** (Basis-shape) vertices and their inverse-distance weights, exactly as the
  addon computes them, and freeze this as a static sparse matrix `R` (shape `(2, V)`) computed once
  from the same rest-pose vertex positions the addon used.
- Reporter position each forward pass: `head_bone_pos = R @ verts` (posed/deformed verts), giving
  `b_h_l`, `b_h_r` positions the same way `joints = J_regressor @ verts` gives ordinary joints — same
  pattern, different fixed matrix.
- `head_width = ‖head_bone_pos[b_h_l] - head_bone_pos[b_h_r]‖`.
- **Validation (mandatory before this is trusted for anything):** run this torch reporter on the 20
  Atta rest-pose/fitted meshes already on disk and diff its output against
  `diagnostics/atta_reference/blender_export/OmniAnt_25PCs_joint_limited_joint_distances.csv`
  (the ground truth this whole allometric law was validated against). Report max and mean absolute
  and relative error across the 20 specimens.

  **Bar revised 2026-09-07, before the noise-floor run below, on new evidence — not loosened after a
  failing result.** The original flat ≤1% bar (inherited from the whole-*skeleton* aggregate check,
  which averages over 55 joint pairs and was never exposed to this problem) FAILED at 4.09% mean.
  Three-way diagnosis at the time (exact at rest pose; the *same* magnitude of error appears on
  trained joints using verbatim-correct regressor rows with zero reporter code involved — mandible
  4.85%, waist 13.8%, antenna 6.3%; whole-skeleton error shape matches the original report) already
  pointed away from a reporter bug and toward small bilateral distances being structurally exposed to
  a fixed absolute discrepancy via a small denominator. The follow-up determinism check then
  confirmed a second, independent source of exactly that kind of noise: re-running the identical
  commit/config/meshes (`b5bf9565`, `cfg_arm_a_master.yaml`, job 3790072) produced a fit differing
  from the original by mean abs 0.00275 / max abs 0.0565 model units — non-zero even with nothing
  changed. A single flat percentage cannot distinguish "reporter is wrong" from "this specimen's
  head-width scale is small enough that ordinary fitter noise produces a large relative error" — so
  the bar is redefined as a **noise-floor-relative** test instead of an absolute tolerance:

  1. Re-run the ARM_A fit **K=5 times** (fresh, no code changes, same commit/config/meshes as job
     3790072) to characterize the fitter's own run-to-run spread on `head_width` specifically (not
     inferred from other pairs).
  2. Per specimen, compute mean and sample std of `head_width` across the 5 runs (`ddof=1`, t
     distribution, 4 df).
  3. Per specimen, test whether the Blender CSV's ground-truth value is a plausible 6th draw from
     that run's own distribution: **95% t-based prediction interval**,
     `mean_i ± t_crit(0.975, df=4) · std_i · sqrt(1 + 1/5)`.
  4. **PASS bar (fixed now):** the CSV ground truth falls inside its specimen's 95% prediction
     interval for **≥17/20 (85%) specimens**. This is the noise-floor analog of the original ≤1%
     bar — a specimen whose reporter value is a plausible sample of the fitter's own noise is
     validated, not miscounted as a reporter defect.
  5. **If PASS:** Step 1 unblocks — the reporter is shown correct up to the fitter's own
     irreducible noise, which is the tightest standard achievable given this measurement pipeline.
  6. **If FAIL** (systematically outside the interval, not just occasionally): that is still a
     real, informative result — it means the discrepancy exceeds ordinary optimizer noise and some
     other explanation (a residual reporter bug, or a genuine addon-vs-numpy computation difference)
     remains open, and Step 1 stays blocked pending that diagnosis.

  **Superseded:** manually re-importing a freshly matched snapshot into Blender
  (`blender_bundle_rematch_20260907/`) to chase byte-identity is no longer pursued — the diagnosis
  above shows a perfectly matched snapshot would still sit inside the same noise floor, so it would
  not have distinguished anything this test doesn't already settle more directly. The bundle is left
  on disk in case a single clean data point is wanted later for the record, but it does not gate
  anything.

### Step 1 — the loss term

- `allo_target(BL) = 10^(intercept_ref + exponent_ref · log10(BL))`, coefficients fixed at the
  **reference** values (exponent 1.2341, intercept −0.5012 — the paper's own fit, not this repo's).
- `l_allo = (log10(head_width) - (intercept_ref + exponent_ref · log10(body_length)))**2`, mirroring
  `scale_barrier`'s per-batch `.mean()` reduction (`fitter_3d/joint_limits.py:79-110`), added in the
  same `if lw.get(...) > 0` chain as `w_scale` in `trainer_moonshot.py` (~line 738-741).
- `w_allo` calibrated by a small pilot sweep so its gradient magnitude at the start of the relevant
  stage is comparable to `w_scale · d(scale_barrier)/d(params)` at the same point — not copied
  verbatim from 0.052, since the two terms have different units and curvature.

### Step 2 — the three-arm run

- **Corpus: `bench50_clean`**, same as X1, for direct comparability of the mechanism metric.
  (Separately, as a generalization check, also run on the 20-specimen Atta corpus itself, where the
  allometric prior is *known* to hold by construction — see §8 for why this arm alone cannot license
  a general claim.)
- **Arm A** — `A_nocap.yaml`, reused verbatim from X1 (`w_scale: 0.0`, no allo term).
- **Arm B** — `B_scalecap.yaml`, reused verbatim from X1 (`w_scale: 0.052`, shipped baseline).
  Confirmed byte-identical to the current `D1_PROD.yaml` as of this writing — see §7.7.
- **Arm C** — `C_allometric.yaml`: `A_nocap.yaml` base (`w_scale: 0.0`) plus `w_allo` at the
  calibrated value from Step 1. Scale_cap and the allometric term are not combined in the primary
  comparison — C is a **replacement**, not an addition, matching the framing "replace the
  content-free penalty," and keeps the arm count interpretable. (A fourth arm, B+C combined, is
  reportable but not gating.)
- Verified before running: A/B/C configs differ **only** in the scale/allo keys — same diff
  discipline as X1 (`diff` must show exactly the intended lines).

## 4. The two measurements, computed side by side, same script pattern as X1

- **PROXY (new, this experiment's own positive control):** absolute residual
  `|log10(HW) - allo_target_log(BL)|` per specimen, arm C vs A vs B. Expected: C ≪ A ≈ B, since only
  C penalizes this residual directly.
- **MECHANISM (X1's metric, reused unchanged):** head deform ratio (head/thorax), computed exactly
  as `x1_anterior_mechanism.py:92-115` — same `PART_GROUPS_COARSE` vertex assignment, same
  thorax-relative `mean(‖deform_verts‖)` per group. **Head is the endpoint, same as X1**, so the two
  results sit on the same scale and are directly comparable in one table.

## 5. Pre-registered bar — identical to X1's, for comparability

Endpoint: **head deform ratio (head/thorax), arm C vs arm A**, paired across specimens.
X1's arms sat at B≈0.802 vs A≈0.793 (both far from X1's target of 1.0; §6.10 baseline was 0.72–0.76×).

- **PASS (mechanism corrected)** — head ratio in C is closer to 1.0 than in A by **≥0.05 absolute**,
  with paired sign test **p < 0.05**; the PROXY confirms C actually reduced the allometric residual
  relative to A/B (positive control holds); *and* the **cost condition** below holds. All three are
  gating — mechanism movement bought by blowing the cost budget is not a PASS, it's a relabeled cost
  tradeoff (see below).
- **PARTIAL** — moves toward 1.0 significantly, by <0.05, or clears ≥0.05/p<0.05 but fails the cost
  condition (i.e. the mechanism gain is real but not affordable at this `w_allo`).
- **FAIL** — no significant movement toward 1.0, despite the proxy holding.
- **VOID** — the proxy itself doesn't move (C fails to reduce its own target residual vs A/B). Then,
  exactly as in X1, nothing is concluded about H2 — the term was not shown to do the one thing it was
  designed to do, so its effect on the mechanism is uninterpretable.

**Cost condition (gating, not advisory):** arm C's Chamfer cost must be **≤1.4× arm B's**. This is
not the "catastrophic" backstop a first draft of this document used (>2×, which would rubber-stamp
almost any result) — it is a pre-committed exchange-rate bar in the same spirit as G6's Pareto
framing, chosen because G6's own accepted tradeoff (11.0% vs 38.9% joint residual) came in far
steeper, at 34×, for a *joint-position* term addressing a problem the objective structurally could
not reach any other way. An allometric *shape prior* competing for the same gradient budget as
Chamfer every step has no comparable structural excuse to spend that much; 1.4× is the number this
document is committing to defend, not a number chosen after seeing a result. If `w_allo` (calibrated
in Step 1) cannot clear the mechanism bar within this cost budget, that is itself the finding — it
means the prior only helps by dominating the objective, not by supplying more correct information —
and the outcome is PARTIAL, not a tuned-until-it-fits PASS.

Reported alongside, gating nothing else: mandible and antenna ratios (X1 found antenna 1.668→1.704,
should move toward 1.0 if the anterior representation defect is genuinely a shape-plausibility
problem), gaster and waist for completeness, and edge/normal/laplacian cost relative to arm B for
context alongside the gating Chamfer number.

## 6. What each outcome licenses — fixed now

| outcome | reading |
|---|---|
| proxy holds **and** head ratio → 1.0 | the anterior defect is a shape-plausibility problem, and an externally-sourced prior — not an arbitrary magnitude penalty — is the right class of fix. Worth porting into the production recipe as a replacement for scale_cap. |
| **proxy holds, head ratio unmoved** | decisive negative, and a genuinely informative one: it narrows the search by ruling out "shape-plausibility" as the failure mode entirely — the defect must be something the objective still doesn't reach even with a semantically correct prior in place (revisit G6/G7's joint-position-term direction instead). |
| proxy does not hold | VOID — we have not shown the allometric term does the one thing it is designed to do on this corpus; nothing is concluded about H2, and the run must be re-diagnosed (calibration of `w_allo`, or a reporter-matrix bug) before re-attempting. |

The third row is a voiding condition, not a result — same discipline as X1 §6.

## 7. Mechanism checks

1. **Step 0 validation passes before Step 1 is written.** The torch head-width reporter must
   reproduce the Blender CSV to a stated tolerance (to be fixed as ≤1% relative error per specimen,
   matching the existing whole-skeleton replication precedent in `REPORT_ATTA_HEADWIDTH.md`) across
   all 20 Atta specimens with an existing Blender export. If it doesn't, stop and fix the reporter —
   do not proceed to a loss term built on an unvalidated measurement (this is the same rule the
   project's own CLAUDE.md states for I/O swaps: verify byte/numeric equivalence, don't trust that a
   measurement is "close enough" by construction).
2. **Arms differ, and only in the intended way (VOIDS).** Same diff-discipline assertion as X1 §7.1.
3. **Reference coefficients used, not replicated ones (VOIDS).** Assert the hardcoded target in the
   loss matches `REPORT_ATTA_HEADWIDTH.md`'s stated *reference* row, not the *replicated* row — a
   copy-paste of the wrong column silently makes the whole experiment circular.
4. **`deform_verts` present and non-trivial in all arms**, same check as X1 §7.3.
5. **Thorax reference non-degenerate**, same check as X1 §7.4.
6. **body_length non-degenerate.** Report its absolute value per arm; a collapsed or exploded
   body_length would make the allo residual meaningless in the same way a near-zero thorax
   denominator would.
7. **Arm A/B provenance — checked 2026-09-07, confirmed current.** `diff B_scalecap.yaml
   diagnostics/moonshot/cfg/D1_PROD.yaml` is **byte-identical** (zero diff); `diff A_nocap.yaml
   D1_PROD.yaml` differs in **exactly** the two `w_scale` lines (45, 65: `0.0` vs `0.052`), nothing
   else. `git log --since=2026-08-31 -- diagnostics/moonshot/cfg/D1_PROD.yaml` returns no commits —
   the production recipe has not moved since X1. **Arm B is the currently-shipped recipe, not a
   stale Aug-31 snapshot**; the comparison's scope is exactly what it appears to be. (Re-run this
   check if Step 2 is delayed long enough that D1_PROD.yaml could plausibly have changed again.)

## 8. Not claimed

- This tests whether the allometric prior moves the mechanism **on `bench50_clean`** (X1's corpus,
  for comparability) and, separately, on the 20 Atta specimens the law was validated on. A PASS on
  Atta alone would not be surprising — the prior is true there by construction — and cannot be used
  to claim the fix generalizes; PASS is only informative on `bench50_clean`, where the prior is an
  out-of-sample constraint being imposed on specimens it wasn't fit to.
- This does not test whether the HW~BL exponent transfers to genera outside Atta. **Confirmed:**
  `diagnostics/moonshot/bench50_clean/` contains 50 distinct species across ≥30 genera (Acromyrmex,
  Anochetus, Cephalotes, Eciton, Dorylus, Lasius, ... ) and **zero** Atta specimens. So a PASS on
  `bench50_clean` is real evidence for the *class* of fix (external allometric prior vs. arbitrary
  magnitude penalty) generalizing out-of-genus, which is the stronger and more interesting claim —
  but it is not validation of this specific exponent for any one of those genera individually. A FAIL
  on `bench50_clean` could equally mean "wrong exponent for this genus mix" rather than "wrong class
  of fix," and the two must be told apart using the Atta-only arm (where the exponent is known
  correct by construction) before concluding against H2 in general: FAIL on both = the class of fix
  doesn't help; FAIL on `bench50_clean` but PASS on Atta-only = the class of fix works but this
  exponent doesn't transfer across genera, a narrower and still useful finding.
- A VOID or FAIL here does not retract G6/G7's joint-position-term finding, which remains the other
  standing, positive lead in this record.
