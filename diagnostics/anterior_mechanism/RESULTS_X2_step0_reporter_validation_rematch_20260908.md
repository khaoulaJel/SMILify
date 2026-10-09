# X2 Step 0 — reporter re-validation against a properly matched Blender export

**Date: 2026-09-08. Resolves the open question in `RESULTS_X2_step0_reporter_validation.md` §5/§6
and `RESULTS_X2_rematch_20260907.md`: is the original ~4% gap a stale-snapshot artifact, or a
structural Blender-vs-numpy computational discrepancy?**

## Verdict: **stale snapshot, confirmed. Not a structural discrepancy. Step 0 now PASSES
comfortably — well inside the fitter's own noise floor.**

---

## What changed

A fresh `b_h_l`/`b_h_r` placement and CSV export were done in one uninterrupted Blender session,
using the exact trilaterated rest-pose coordinates already computed in the original Step 0 work
(residual ~3.6e-9 model units — reused unchanged here, not re-derived):

- `b_h_l`: `[0.40507574, 0.16124934, 0.06171087]`
- `b_h_r`: `[0.39643314, -0.16170239, 0.05901498]`

Exported to `diagnostics/atta_reference/blender_export_rematch_20260907/OmniAnt_25PCs_joint_limited_joint_distances.csv`,
compared against `diagnostics/atta_reference/blender_bundle_rematch_20260907/ATTA20_ARM_A_rematch.npz`
(the SAME npz the Blender session imported — zero gap between generation and export this time,
which is the entire point).

## Result

| check | old (mismatched snapshot) | new (matched, same-sitting) |
|---|---:|---:|
| mean \|rel err\| | 4.086% | **0.070%** |
| max \|rel err\| | 8.159% | **0.121%** |
| specimens within ±1% | 3/20 | **20/20** |
| Base (rest-pose) error | 0.0047% | 0.0047% (unchanged, algorithm was always exact here) |

Full per-specimen table (all 20, `abs_err` in model units, `rel_err_pct` signed):

| specimen | pred | gt | abs_err | rel_err_pct |
|---|--:|--:|--:|--:|
| 01.obj | 0.31698 | 0.31707 | -0.00008 | -0.026 |
| 02.obj | 0.38301 | 0.38302 | -0.00001 | -0.004 |
| 03.obj | 0.32865 | 0.32884 | -0.00019 | -0.057 |
| 04.obj | 0.31763 | 0.31755 | +0.00009 | +0.027 |
| 05.obj | 0.35431 | 0.35441 | -0.00010 | -0.028 |
| 06.obj | 0.33417 | 0.33433 | -0.00015 | -0.046 |
| 07.obj | 0.35068 | 0.35091 | -0.00023 | -0.065 |
| 08.obj | 0.32821 | 0.32842 | -0.00021 | -0.063 |
| 09.obj | 0.31729 | 0.31767 | -0.00038 | -0.121 |
| 10.obj | 0.28828 | 0.28855 | -0.00027 | -0.093 |
| 11.obj | 0.28713 | 0.28742 | -0.00028 | -0.098 |
| 12.obj | 0.26877 | 0.26899 | -0.00022 | -0.082 |
| 13.obj | 0.30596 | 0.30627 | -0.00030 | -0.099 |
| 14.obj | 0.32415 | 0.32447 | -0.00032 | -0.100 |
| 15.obj | 0.37036 | 0.37060 | -0.00024 | -0.064 |
| 16.obj | 0.32880 | 0.32901 | -0.00021 | -0.063 |
| 17.obj | 0.36736 | 0.36776 | -0.00039 | -0.107 |
| 18.obj | 0.33754 | 0.33780 | -0.00026 | -0.077 |
| 19.obj | 0.35760 | 0.35786 | -0.00026 | -0.072 |
| 20.obj | 0.38874 | 0.38914 | -0.00040 | -0.103 |

Note the residual is not only small but **consistently signed** (19/20 specimens negative, pred
slightly below gt) — a small, uniform bias of this shape is far more consistent with a residual
measurement-pipeline offset (e.g. minor float32-vs-float64 rounding differences between numpy's
computation and Blender's internal evaluation, or a sub-pixel rest-pose placement difference) than
with anything resembling the original ~4-8% scattered, per-specimen-idiosyncratic gap. This
residual is well inside the fitter's own measured noise floor (mean CV 0.165%,
`RESULTS_X2_step0_noisefloor_20260907.md` §6) and is not investigated further — it is not the
open question this test was designed to answer.

## What this settles

Combined with the noise-floor test (`RESULTS_X2_step0_noisefloor_20260907.md`, which ruled out
"just optimizer noise" as an explanation for the *original* CSV's ~4% gap by direct measurement),
this result completes the diagnosis: the original Blender export
(`blender_export/OmniAnt_25PCs_joint_limited_joint_distances.csv`) was generated from a vertex
snapshot that was not the one in any npz obtainable in this environment — a genuine
provenance/timing gap (bundle written 2026-09-06 00:01, CSV exported 12:47 the same day, per
`RESULTS_X2_step0_reporter_validation.md` §5) — **not** a flaw in the reporter algorithm, and
**not** a structural Blender-vs-numpy computational mismatch. The reporter formula itself
(`x2_head_width_reporter.py`, unmodified since Step 0) is now validated end-to-end: exact at rest
pose, gradient-checked, and now confirmed on posed specimens to well within the fitter's own noise
floor when the snapshot is properly matched.

## Status: Step 0 **PASSES**. Step 1 (the `l_allo` loss term) is unblocked.

Per `PREREGISTRATION_X2_allometric_prior.md` §3 Step 0's gating rule, this validated reporter can
now be trusted as the measurement instrument for Step 1's `l_allo` loss term and the X2 experiment
proper (§4-§8 of the preregistration). The noise-floor-relative bar from the 2026-09-07 revision
was a legitimate and necessary fallback given the tools available at the time (no Blender access);
it is superseded here by a stronger, more direct result now that a matched export exists, and
neither retroactively invalidates the other — the revised bar's own diagnostic work (isolating
"noise" from "systematic offset") is exactly what correctly predicted this outcome.
