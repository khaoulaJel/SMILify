# X2 Step 0 — noise-floor prediction-interval test result

**Date: 2026-09-07/08. Gate for `PREREGISTRATION_X2_allometric_prior.md` §3 Step 0 (revised bar).**

Follow-up to `RESULTS_X2_step0_reporter_validation.md` (single-run FAIL at the original flat ≤1%
bar) and `RESULTS_X2_rematch_20260907.md` (determinism finding that motivated this test). This
document runs the exact pre-registered noise-floor test: is the CSV ground truth a plausible 6th
draw from the fitter's own 5-run distribution?

## Verdict: **FAIL — 0/20 specimens (0%), far below the ≥17/20 (85%) bar. Step 1 stays BLOCKED.**

---

## 1. Provenance of the 5 fresh runs

Same commit, config, and meshes as job 3790072 (`RESULTS_X2_rematch_20260907.md`):
worktree `/hpcwork/nao48500/atta20_master` @ `b5bf9565`, `cfg_noisefloor_run{1..5}_20260907.yaml`
(each byte-identical to `cfg_arm_a_master.yaml` except `results_dir`), 20 meshes at
`/hpcwork/nao48500/atta20/{01..20}.obj`.

Independently verified (not taken on trust from the scheduling report):

| run | job ID | state | exit | output |
|---|---|---|---|---|
| 1 | 3838254 (`nf1_c23g`) | COMPLETED | 0:0 | `runs/noisefloor_run1_20260907/Stage_3_deform_fine.npz` |
| 2 | 3838307 (`nf2_c23g`) | COMPLETED | 0:0 | `runs/noisefloor_run2_20260907/Stage_3_deform_fine.npz` |
| 3 | 3838308 (`nf3_c23g`) | COMPLETED | 0:0 | `runs/noisefloor_run3_20260907/Stage_3_deform_fine.npz` |
| 4 | 3838309 (`nf4_c23g`) | COMPLETED | 0:0 | `runs/noisefloor_run4_20260907/Stage_3_deform_fine.npz` |
| 5 | 3838312 (`nf5_c23g`) | COMPLETED | 0:0 | `runs/noisefloor_run5_20260907/Stage_3_deform_fine.npz` |

Confirmed via `sacct` (`ExitCode 0:0` for each), and each npz independently loaded and checked:
`verts.shape == (20, 10235, 3)`, `labels == ["01.obj", ..., "20.obj"]` in order, for all 5 runs.
(The originally-submitted array job, 3799698, was cancelled during scheduling troubleshooting and
re-submitted as 5 separate jobs on partition `c23g` instead of `c25g` — a scheduling-only change;
commit, configs, and meshes are unchanged, so nothing about validity of the comparison is affected.)

## 2. Method

`head_width()` from `diagnostics/anterior_mechanism/x2_head_width_reporter.py` (reused unmodified)
applied to each run's `Stage_3_deform_fine.npz["verts"]` (20 specimens × 5 runs → a (5, 20) matrix).
Per specimen: `mean_i`, `std_i` (`ddof=1`) across the 5 runs; `t_crit = t.ppf(0.975, df=4) = 2.7764`;
`interval_i = mean_i ± t_crit · std_i · sqrt(1 + 1/5)`. Ground truth and specimen-to-CSV-row mapping
reused verbatim from `x2_head_width_reporter.py::validate_against_blender_csv` (labels `NN.obj`
matched directly against the CSV's `Shape` column), exactly as instructed — no new mapping was
derived.

## 3. Per-specimen table

`bias%` = `(mean_i − gt) / gt × 100` (signed); `CV%` = `std_i / mean_i × 100`.

| specimen | mean (5 runs) | std (ddof=1) | 95% PI | CSV gt | inside PI? | CV% | bias% |
|---|--:|--:|--:|--:|:--:|--:|--:|
| 01 | 0.31753 | 0.00074 | [0.31529, 0.31977] | 0.33204 | **NO** | 0.232% | −4.37% |
| 02 | 0.38295 | 0.00031 | [0.38200, 0.38389] | 0.35352 | **NO** | 0.081% | +8.32% |
| 03 | 0.32790 | 0.00016 | [0.32741, 0.32838] | 0.33387 | **NO** | 0.049% | −1.79% |
| 04 | 0.31808 | 0.00069 | [0.31597, 0.32019] | 0.33862 | **NO** | 0.218% | −6.06% |
| 05 | 0.35457 | 0.00027 | [0.35375, 0.35540] | 0.33201 | **NO** | 0.077% | +6.79% |
| 06 | 0.33442 | 0.00049 | [0.33294, 0.33590] | 0.31976 | **NO** | 0.146% | +4.58% |
| 07 | 0.35154 | 0.00026 | [0.35073, 0.35234] | 0.33193 | **NO** | 0.075% | +5.91% |
| 08 | 0.32864 | 0.00058 | [0.32689, 0.33040] | 0.32409 | **NO** | 0.176% | +1.40% |
| 09 | 0.31922 | 0.00052 | [0.31763, 0.32081] | 0.33967 | **NO** | 0.164% | −6.02% |
| 10 | 0.28888 | 0.00039 | [0.28769, 0.29008] | 0.30462 | **NO** | 0.136% | −5.17% |
| 11 | 0.28728 | 0.00039 | [0.28610, 0.28846] | 0.27470 | **NO** | 0.135% | +4.58% |
| 12 | 0.26878 | 0.00027 | [0.26795, 0.26961] | 0.27161 | **NO** | 0.102% | −1.04% |
| 13 | 0.30576 | 0.00030 | [0.30486, 0.30665] | 0.30353 | **NO** | 0.096% | +0.73% |
| 14 | 0.32567 | 0.00028 | [0.32483, 0.32651] | 0.31172 | **NO** | 0.085% | +4.48% |
| 15 | 0.36991 | 0.00069 | [0.36779, 0.37202] | 0.35261 | **NO** | 0.188% | +4.91% |
| 16 | 0.32747 | 0.00071 | [0.32530, 0.32963] | 0.34086 | **NO** | 0.217% | −3.93% |
| 17 | 0.36833 | 0.00141 | [0.36404, 0.37261] | 0.36309 | **NO** | 0.382% | +1.44% |
| 18 | 0.33769 | 0.00103 | [0.33457, 0.34082] | 0.35312 | **NO** | 0.304% | −4.37% |
| 19 | 0.35923 | 0.00106 | [0.35602, 0.36244] | 0.35584 | **NO** | 0.294% | +0.95% |
| 20 | 0.39161 | 0.00058 | [0.38985, 0.39337] | 0.37387 | **NO** | 0.148% | +4.75% |

**Specimens with gt inside their 95% prediction interval: 0/20 (0%).**

## 4. Pre-registered bar applied exactly as written

`PREREGISTRATION_X2_allometric_prior.md` §3 Step 0, bar 4: **PASS requires ≥17/20 (85%)** of
specimens to have their CSV ground truth fall inside their own 95% prediction interval.

**Result: 0/20 (0%). FAIL, by the widest possible margin** — not a borderline miss.

## 5. Verdict, stated explicitly

**FAIL. Step 1 (the `l_allo` loss term) stays BLOCKED**, exactly per the pre-registration's §3 Step 0
outcome 6 ("if FAIL ... that is still a real, informative result").

**Diagnosis of what remains unexplained:** the fitter's own run-to-run noise is not the explanation
for the ≤1%-bar failure documented in `RESULTS_X2_step0_reporter_validation.md`. Two things are now
established, not merely suspected:

1. **The noise floor is roughly two orders of magnitude too small to cover the gap.** Mean `std_i`
   across specimens is 0.00056 model units (mean CV 0.165%, max CV 0.382% — see §6), while the
   mean absolute bias between the 5-run mean and the CSV ground truth is **4.08%** (max 8.32%,
   specimen 02) — essentially identical to the single-run FAIL already reported (mean 4.086%,
   max 8.159%). Five independent fresh fits landed within a tight, mutually consistent cluster,
   and that cluster sits **4% away from the CSV value, not adjacent to it** — every single
   specimen's interval excludes the ground truth, several by 10+ standard errors.
2. **The discrepancy is a systematic bias, not noise.** Because the 5-run std is so small relative
   to the gap, the earlier candidate explanation ("the specific `ATTA20_ARM_A.npz` snapshot used in
   Step 0 validation just happened to be a noisy draw, off by ordinary optimizer variance from
   whatever snapshot produced the CSV") is now ruled out by direct measurement, not just
   argued from a determinism side-check. The bias has the same *sign pattern and magnitude* as the
   original single run per specimen (e.g. specimen 02: original +8.159%, here +8.32%; specimen 01:
   original −4.764%, here −4.37%) — reinforcing that this is a **specimen-level systematic offset**,
   reproducible across independent optimizer runs, not run-to-run scatter.

This points back to the diagnosis already on record in `RESULTS_X2_step0_reporter_validation.md`
§5: the CSV's ground truth was very likely exported from a *different* vertex snapshot than any
`Stage_3_deform_fine.npz` this environment can produce (the original bundle was stamped ~12h45m
before its CSV export), and that gap is a fixed, roughly-constant *absolute* discrepancy per
specimen that gets amplified into a large *relative* error specifically because `head_width` is a
small bilateral distance (~0.27–0.39 model units) with a small denominator — exactly the same
mechanism §5 identified using zero reporter-specific code (verbatim trained-joint bilateral pairs
showed the same inflation). The **remaining unexplained gap is a snapshot/provenance mismatch
between the npz families available here and whatever produced the Blender CSV**, not fitter
stochasticity and not a bug in the reporter algorithm (which is independently gradcheck-verified and
exact at Base/rest pose to 5 decimal places). Closing it would require either (a) a Blender-side
regeneration of the ground-truth CSV from one of the npz snapshots actually on disk in this
environment (ruled out as low-value in `RESULTS_X2_rematch_20260907.md`, since the rematch bundle
sits in the same noise floor as everything measured here), or (b) accepting the noise-floor test as
having definitively shown the current CSV is not a valid target for a ≤1%-style bar on this
quantity, and either loosening the bar to reflect a snapshot-tolerance term (not proposed here) or
sourcing a fresh, contemporaneous CSV export the next time Blender access is available.

## 6. Noise-floor summary (standalone fact, reusable elsewhere in this project)

Computed on `head_width` across the 5 fresh runs, before comparison to any ground truth:

| statistic | value |
|---|---|
| mean `std_i` (5 runs, ddof=1) | 0.000556 model units |
| max `std_i` | 0.001409 model units (specimen 17) |
| mean CV (`std_i / mean_i`) | 0.165% |
| max CV | 0.382% (specimen 17) |

**Takeaway for other parts of this project:** on this quantity, run-to-run optimizer noise alone is
~0.2–0.4% CV — small in absolute terms, consistent with the Z4/Z5 project-memory finding
("~0.01 run-to-run on gen@20" for whole-vertex fields) scaled down to a single bilateral distance.
This confirms small local distances do **not** inherently need looser *optimizer-noise* tolerances
than whole-skeleton aggregates — the noise floor here is tiny. What *does* need looser tolerance for
small local distances is **measurement/snapshot-provenance mismatch** (§5): the same absolute
per-vertex discrepancy that is negligible relative to a large aggregate distance (e.g. `b_t`–`b_a_5`,
0.90 model units, 0.095% mean error in the original Step 0 report) becomes a large relative error on
a small bilateral distance (0.27–0.39 model units here) purely from the smaller denominator. The two
failure modes (optimizer noise vs. snapshot mismatch) are now cleanly separated by this test: noise
is ruled out as negligible; snapshot mismatch remains open.
