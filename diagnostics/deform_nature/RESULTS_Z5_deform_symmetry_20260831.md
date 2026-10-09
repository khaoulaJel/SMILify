# Z5 — PARTIAL. A real, dose-responsive effect that the fit bound disqualifies.

Bar fixed in `PREREGISTRATION_Z5_deform_symmetry.md` before the run. Job 3353632, 12 min 25 s,
COMPLETED. `bench50_clean` — **50 real workers**. `D1_PROD.yaml`, seed 0; arms differ only in
`w_deform_sym`.

## Result

| arm | symmetric share | **gen@20/spread** | chamfer | vs A | §6.10 head ratio |
|---|---:|---:|---:|---:|---:|
| A — as shipped | 0.6070 | **0.6044** | 0.00115 | — | 0.8118 |
| **B — `w_deform_sym` 2.0** | 0.8500 | **0.5713** | 0.00124 | **+7.8%** | 0.7800 |
| C — `w_deform_sym` 10.0 | 0.9724 | **0.5545** | 0.00128 | **+11.3% → VOID** | 0.7582 |

All four voiding checks: involution exact (0.00e+00, 100% of vertices), the term binds
(0.607 → 0.850 → 0.972), shape space open in every arm (|z| 1.12–1.15), fit bounded — and C fails
it by 1.3 percentage points.

**Verdict PARTIAL.** B improves gen@20 by **+0.0331** at an acceptable fit cost but does not reach
the 0.56 bar. C reaches **0.5545**, which *would* have passed, and is disqualified for fit
collapse.

## What is actually established

**The effect is real and dose-responsive.** Symmetric share, transfer, and fit all move
monotonically together across three arms: constrain the field more → it transfers better → the fit
gets worse. This is not the proxy-without-mechanism pattern that X1, Z2 and Z3 hit — there the
intervention moved its own quantity and the outcome sat still. Here the *endpoint* moves, cleanly,
in dose order.

**It is a trade-off, not a free win.** The post-hoc projection bound was 0.5292; C gets to 0.5545
of that only by spending 11.3% of chamfer. The antisymmetric half of the free-form field is
therefore **partly load-bearing for the data term** — it is not pure noise that can simply be
removed, which qualifies Z4's reading.

**It makes the head defect worse.** The §6.10 ratio moves *away* from 1.0 in dose order
(0.8118 → 0.7800 → 0.7582). Whatever sustains the head defect, symmetrising the free-form field
is not a route to it — consistent with Z2, which also failed on that endpoint.

## The caveat that bounds all of the above

Arm A here reads **0.6044**; the identical configuration in Z3 read **0.6147**. Same recipe, same
seed, same corpus — the fitter resamples the target every iteration as a deliberate stochastic
regulariser (`trainer_moonshot`), so runs are not deterministic. **Run-to-run spread on this
endpoint is therefore ~0.01, estimated from n = 2.** B's +0.0331 is about three times that, so the
effect is very unlikely to be noise, but **two samples is not a variance estimate** and no claim
here rests on a difference smaller than ~0.03.

## What this licenses, per §5 of the pre-registration

**PARTIAL → report; do not ship; do not sweep further weights hunting for the bar.**

That clause is load-bearing and is being honoured. A weight between 2 and 10 would very likely land
below 0.56 with chamfer under 10%, and finding it by tuning on n = 50 is exactly the procedure that
produced seven proxy-without-mechanism results in this project. The right way to settle it is the
standard `scale_cap` was held to (`EXECUTION_PLAN.md` §4): **the full 757-worker corpus, with the
weight treated as a deployment parameter and the decision rule fixed in advance** — including
whether ~0.03–0.05 of gen@20 is worth 8–11% of chamfer at all, which is a question about the
downstream morphometrics, not about this metric.

`w_deform_sym` stays **off by default** in `D1_PROD.yaml`.

## Standing position on workers after Z1–Z5

| proposed cause of the worker gap | closed by |
|---|---|
| worker scans are unregisterable | Z1 — the 0.98 anchor was measured with the shape space frozen; the real figure is ~0.61 |
| anterior scale magnitude | X1 — proxy moved, mechanism did not |
| anterior rotation freedom | Z2 — bands bound hard, head ratio unmoved (p = 0.322) |
| joint scale/translation unmodelled | Z3 — fully modelled (0% → 100% driven), transfer got worse |
| free-form field is missing anatomy | Z4 — 0.60 symmetric against a 0.500 null, and **neither half transfers**; it is predominantly per-specimen noise, so more shape modes will not recover it |
| free-form asymmetry is removable noise | **Z5 — removing it helps (+0.033) but is partly load-bearing; a trade, not a free win** |

Workers sit at **gen@20/spread ≈ 0.60**, against an instrument floor of 0.00 at k = 20 under exact
correspondence. The §6.10 head ratio sits at **0.81** against a corrected 1.0 and has now resisted
four separate interventions.

The remaining gap is, on this evidence, **dominated by irreducible per-specimen scan and
correspondence error** rather than by any parameterisation or capacity defect. The honest next step
is not another intervention on the fitter: it is to establish what `gen@20/spread` can actually
attain on real worker scans — i.e. an achievable target — so that the residual can be judged
against something real instead of against 0.00.

## Code

`trainer_moonshot.py` gains `w_deform_sym` and `deform_symmetry_penalty`, off by default. The
involution is built from `v_template` (not from a fitted mesh — `build_shape_space.py` is explicit
that matching a fitted specimen conflates the mirror map with fit error, 7.68e-02 vs the
template's true 0) and is **asserted exact at construction, raising rather than warning**.

The Z5 scorer reads fit quality from the fitter's own log and **treats a missing value as a
failure**; Z3's equivalent check passed vacuously on `nan` because `optimise_moonshot` only writes
`metrics.csv` when asked to evaluate.

## Artifacts

`z4_deform_signal_or_noise.py`, `z4_out.txt`, `out_Z4/`; `z5_score.py`, `B_dsym2.yaml`,
`C_dsym10.yaml`, `submit_Z5_20260831.sbatch`, `sbatch_logs/Z5_3353632.log`, `out_Z5/`; fits at
`diagnostics/moonshot/runs/Z5_{A_off,B_dsym2,C_dsym10}`.
