# Pre-registration — E1/E2: frozen pose, and is `log_beta_scales` causal?

Written **before** E1/E2 are run. 2026-08-28. Two controls were measured first (E0 and the
hier-only baselines); both are controls, not treatments, and both are reported below because the
bar is meaningless without them.

## Control 1 (E0) — the metric has a coxa-specific floor, and it is not zero

`E0_identity` audits the **ground-truth mesh as if it were the fit**. Placement is exactly correct
by construction; no fit is run. It measures the metric's own floor.

| | leg_acc | co | tr | fe | ti | ta | pt |
|---|---|---|---|---|---|---|---|
| `E0_identity` (perfect placement) | 0.9926 | **0.0908** | 0.0110 | 0.0043 | 0.0003 | 0.0021 | 0.0010 |

**A perfect fit still scores 0.0908 coxal error** — 8× the next-worst segment and ~300× `ti`.
Adjacent coxae sit close enough that nearest-vertex matching misassigns ~9% of coxal points even
on an exact mesh. This is the measured, direct form of the "inter-leg tolerance risk = 0.995"
result.

Two consequences, both fixed before seeing E1/E2:

1. **The addressable coxal residual is smaller than every previous document said.** Not 0.41, but
   `0.41 − 0.09 = 0.32`. Every past percentage attributed to the coxa is inflated by this floor.
   **Correction to this document (added 2026-08-28, before E1/E2 results):** this floor was
   **already measured**. `LAB_RECORD_correspondence_20260827.md` §15 recorded it as a paired
   ceiling of **0.0949** and computed every *addressable* share against it (co 57.8%, tr 27.7%).
   E0 is an independent confirmation by a different route, not a new discovery, and the lab
   record's percentages stand. The claim "every past percentage attributed to the coxa is inflated
   by this floor" was **wrong as written** and applies only to `RESULTS_C13_*` and
   `RESULTS_C14p_*`, which quoted 0.41 without subtracting the known floor. Both now carry a
   corrective banner. Nothing about the bar below changes.

2. **Representability is confirmed, not open.** A perfect parameter set scores 0.09, not 0.41, so
   parameters that place the coxa well plainly exist and are measurable. This is expected — P48
   targets are generated *from* the model — but it is now measured rather than asserted.
   **Therefore E1 is a REACHABILITY test, not a representability test**, and the "genuine model
   limitation" branch is ruled out on this corpus *a priori*. A null in E1 will mean "no shape
   configuration this optimizer reaches places the coxa", never "the model cannot express it".
   E1/E2 will not be reported as evidence of a model limitation whatever they show.

## Control 2 — the D1 stage is where the coxa is lost

E1/E2 must be hier-only: `optimise_moonshot` has no freeze flags and would unfreeze pose. So
hier-only (`H2_joint`) baselines were audited to pair against. They show something unplanned:

| run | stage | leg_acc | **co** |
|---|---|---|---|
| `C13_uniform` (CSE) | H2_joint | 0.9649 | **0.2564** |
| `C13_uniform` (CSE) | after D1 | 0.9415 | **0.4115** |
| `C14p_gtinit_nocse` (GT seed) | H2_joint | 0.9413 | **0.4564** |
| `C14p_gtinit_nocse` (GT seed) | after D1 | 0.9468 | **0.4114** |

With correspondence, the hierarchical stage reaches **0.2564** and the D1 refinement then
**degrades it to 0.4115**, taking leg_acc down with it (0.9649 → 0.9415). The coxa is not
uniformly stuck at 0.41; it is reached and then lost. This is recorded here, before E1/E2, so it
cannot later be presented as an E1 finding. It is a separate result and is not what E1/E2 test.

## Arms

Both hier-only, `--deform_its 0`, no correspondence term, otherwise byte-identical to the recipe
`C14p_gtinit_nocse` used — so `C14p_gtinit_nocse_hier_H2` is the paired control, differing from
E1 **only** by the freeze.

| arm | pose | log_beta_scales | everything else |
|---|---|---|---|
| `C14p_gtinit_nocse_hier_H2` (control, on disk) | GT-**seeded**, free | free | free |
| `E1_frozen_pose` | GT-seeded, **hard-frozen** | free | free |
| `E2_frozen_pose_gtscale` | GT-seeded, **hard-frozen** | GT-seeded, **hard-frozen** | free |

Free in both arms: `betas`, `global_rot`, `trans`, `betas_trans` (and `log_beta_scales` in E1).
`deform_verts` is disabled — free-form per-vertex offsets would satisfy any placement and make the
test vacuous.

The freeze is verified, not assumed: a 15-iteration smoke run leaves `joint_rot` and
`log_beta_scales` **bit-identical** to GT (max |Δ| = 0.0) while `betas` and `trans` move.

## Readings fixed in advance

Primary endpoint: paired per-specimen `co` leg-level error vs `C14p_gtinit_nocse_hier_H2`
(0.4564), n=48, sign test + Wilcoxon + paired-t reported together. Addressable gap to the E0
floor is **0.4564 − 0.0908 = 0.3656**.

**E1 — does holding pose fixed let shape place the coxa?**
- **PASS** — `co` ≤ **0.20** (closes ≥ 70% of the addressable gap) with sign p < 0.05. Then coxal
  failure is an optimization/loss-shaping problem that pose freedom was driving, and one targeted
  fix is warranted.
- **PARTIAL** — falls with sign p < 0.05 but stays above 0.20. Pose freedom contributes but is not
  the driver; report as bounded, do not pursue.
- **FAIL** — does not fall, or rises. Then no shape configuration this optimizer reaches places
  the coxa even with pose removed as a confound, and the thread closes as a characterized
  optimization limitation.

**E2 — is `log_beta_scales` causal, or a compensating symptom?**
- **CAUSAL** — `co`(E2) ≤ `co`(E1) − **0.10** with sign p < 0.05. The r=+0.503 correlation
  reflects a real missing degree of freedom.
- **SYMPTOM** — no significant drop. The optimizer was reaching for the scale block to compensate
  for something else, and that something else is still unidentified. Reported as such, not chased.

Guard-rails, fixed now:
- **Distal must not regress.** `ti`/`ta`/`pt` rising > 0.02 makes any coxal gain a trade, not an
  improvement.
- **`leg_acc` must not fall.**
- All numbers are read against the **E0 floor of 0.0908**, never against 0.

## What will NOT be claimed

- No representability or model-limitation claim (see Control 1). This corpus cannot support one.
- Nothing about real scans. P48 is synthetic and model-generated.
- No claim about the D1 degradation found in Control 2 — that is a separate, untested observation.
- Per the standing rule: **this closes the coxal thread.** Whichever way E1/E2 land, no fourth
  branch is opened from them.
