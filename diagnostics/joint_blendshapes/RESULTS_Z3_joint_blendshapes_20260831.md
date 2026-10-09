# Z3 — REFUTED. The shape space's joint channels work exactly as designed, and shape still does not transfer.

Bar fixed in `PREREGISTRATION_Z3_joint_blendshapes.md` before the run. Job 3353400, 12 min 26 s,
COMPLETED. `bench50_clean`, 50 real workers, `D1_PROD.yaml`, seed 0; arms differ only in
`SMILIFY_COUPLE_JOINT_BLENDSHAPES` and `w_jresid`.

## The intervention did everything it was built to do

| arm | betas sd | mean \|z\| | driven share of joint log-scale | free residual sd |
|---|---:|---:|---:|---:|
| A (as shipped) | 0.37779 | 1.101 | **0.0%** | 0.2368 |
| B (coupling) | 0.29205 | 0.858 | **54.7%** | 0.2335 |
| C (coupling + `w_jresid` 5.0) | **0.52049** | **1.556** | **100.0%** | **0.0006** |

**Primary endpoint PASS.** In C the shape space drives 100% of the applied per-joint log-scale,
the free residual is suppressed to sd 0.0006, and the betas are used *more* than in the shipped
arm — |z| 1.101 → 1.556. `b_a_5`'s p99 |log s| goes 0.6967 → 0.0000: the 28× scale runaway
disappears entirely, because that parameter no longer exists as a free variable.

This is not a marginal effect. §6.6's proposed root cause was implemented correctly, verified
active (vertices move 2.30× further at ±3σ with the joint channels connected), and it did exactly
what the hypothesis said it would do to the parameterisation.

## And shape transfer got worse

| arm | chamfer (fitter, Stage_3 it 999) | vs A | **gen@20/spread** | §6.10 head ratio |
|---|---:|---:|---:|---:|
| A | 0.00116 | — | **0.6147** | 0.7919 |
| **B** | 0.00119 | **+2.6%** | **0.7036** | 0.8695 |
| C | 0.00162 | **+39.7%** | 0.6322 | 0.7989 |

*Lower is better. The reference is the instrument floor — synthetic exact correspondence, 0.00 at k=20.*

**Arm B is the valid arm and it refutes the hypothesis.** Its fit is essentially unchanged
(+2.6%, inside the pre-registered 10% bound), the shape space drives the majority of joint scale
— and `gen@20/spread` moves from 0.6147 to **0.7036**, clearly *worse*. Moving joint scale out of
330 free parameters and into the betas made the recovered shape structure less transferable, not
more.

**Arm C is VOID on voiding check 3.3.** Its chamfer is **+39.7%** against a 10% bound. Forcing the
betas to carry 100% of joint scale costs 40% of fit accuracy, so C's 0.6322 is measured on a
substantially worse registration and cannot be read as a transfer result. The pre-registration
bounded this in advance precisely so it could not be excused afterwards.

### A defect in this experiment's own instrument, and what it cost

Voiding check 3.3 initially reported **PASS on `nan`**. `optimise_moonshot` writes `metrics.csv`
only when asked to evaluate, the sbatch did not ask, and `abs(nan) > 10` is `False` — so a missing
file read as "within bounds". The scorer now treats non-finite values as a failure. The check was
re-run from the fitter's own final chamfer in the log, which is what produced the VOID on C above.
**The first automated verdict of this run was wrong** and would have reported C as PARTIAL rather
than VOID. Z2's check 4.3 has the same gap and its "fit quality reported" element was likewise
never populated; Z2's verdict is unaffected because it was a FAIL on the endpoint itself.

## What this settles

Per §5 of the pre-registration, this is the **FAIL** branch, and it is a real result:

> §6.6's root cause — "shape is stored in a parameter that cannot transfer" — is **refuted on the
> pipeline where it could finally be tested.** The joint channels are closed.

§6.7 reached the same conclusion for the wrong reason: it tested coupling against a betas channel
frozen at sd 0.0005, where the test could not have returned a positive. Z3 tested it against a
betas channel at sd 0.38–0.52, where the mechanism demonstrably engaged — 0% → 100% driven share —
and the outcome still did not follow. The conclusion is now supported by evidence that could have
gone the other way.

**Do not ship `COUPLE_JOINT_BLENDSHAPES`.** It stays off by default. Do not sweep `w_jresid`
hunting for a weight that buys back the 40% chamfer — the pre-registration rules that out, and
arm B already shows the effect is adverse at zero fit cost.

## The standing conclusion after Z1–Z3

The unmodelled-parameter account of the registration problem is now exhausted:

| what was proposed as the cause | how it was closed |
|---|---|
| worker scans are unregisterable | Z1 — the 0.98 anchor was measured with the shape space frozen; the real figure is 0.5975 |
| anterior scale magnitude | X1 — proxy moved, mechanism did not |
| anterior rotation freedom | Z2 — bands bound hard, head ratio unmoved (p = 0.322) |
| joint scale/translation unmodelled | **Z3 — fully modelled, transfer got worse** |

On **real workers** — the only corpus that matters here — `gen@20/spread` sits at **0.6147**
against an instrument floor of 0.00 at k=20 under exact correspondence, and the §6.10 head ratio
at **0.79** against a corrected 1.0. Both gaps are real, both are measured on the shipped
pipeline and on workers, and none of the four parameterisation hypotheses explains them.

**ALL_ANTS_CLEAN is not the target and should not be quoted as one.** It is a different corpus
(0/50 overlap with `bench50_clean`), and REFUTE1's R6 showed it is in-sample for this shape space
— 7.27% of its free-form field lies inside `span(shapedirs)` against 0.90% for workers. Beating it
would not mean the worker problem was solved.

## Code left in the repo

The coupling is implemented and off by default, which is the correct end state regardless of this
result — the model file has always shipped `scaledirs`/`transdirs` and the fitter silently ignored
them. `smal_model/smal_torch.py` now loads them and exposes `compose_log_scale`/`compose_trans` as
the single implementation; `fitter_3d/joint_limits.composed_*` call those instead of duplicating
the math; `trainer_moonshot.py` gained `w_jresid`. Verification lives in
`verify_coupling_PROBE.py`: OFF is bit-identical (max|diff| 0.000e+00), ON == OFF at beta = 0, ON
moves vertices 2.30× further at ±3σ.

`fitter_3d/part_groups.py` was corrected in the same series: j24/25/44/45 are `wing` (bilateral,
dorsal, thorax-parented, no skinning mass on a wingless worker template), `waist` is [1, 2] —
petiole `b_a_1` and postpetiole `b_a_2` — and `gaster` is [3, 4, 5].

## Artifacts

`verify_coupling_PROBE.py`, `z3_score.py`, `C_couple_jresid.yaml`, `submit_Z3_20260831.sbatch`,
`sbatch_logs/Z3_3353400.log`, `out_Z3/z3_results.json`; fits at
`diagnostics/moonshot/runs/Z3_{A_off,B_couple,C_jresid}`.
