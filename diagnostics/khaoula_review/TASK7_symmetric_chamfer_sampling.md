# TASK 7 — symmetric chamfer sampling: causal test + gate-hypothesis probe

Tests the SECOND of FINDINGS.md step 3's two independent, confirmed-but-untouched causes of
the count/F-score problem (chamfer vertex-density asymmetry — legs 2.36x denser than gaster in
raw template vertices), decoupled entirely from penetration/matching-primitive (TASK6's
subject). New `symmetric_chamfer_sampling` flag on `Stage` (`fitter_3d/trainer.py`), default
`False`/byte-identical, ports `trainer_moonshot.py`'s already-working symmetric sampling
(area-weighted-sample BOTH sides of chamfer, not just the target) into this trainer. Same
discipline as every arm before it: TASK5's full config held exactly fixed (pair-scoped
gaster-legs `w_penetration`, `w_offset`, `w_scale`/`w_trans`), `w_sdf=0` explicit, single
variable swapped.

## Part 1 — original 3-specimen causal test (same specimens as TASK6)

| specimen | count: control→**symmetric** | Δcount | f_score: control→**symmetric** | Δf_score |
|---|---|---:|---|---:|
| Acanthostichus | 753→**686** | −8.9% | 0.890→**0.907** | +0.016 |
| Acromyrmex | 964→**1230** | **+27.6%** | 0.675→**0.647** | **−0.028** |
| Solenopsis | 305→**474** | **+55.4%** | 0.684→**0.693** | +0.009 |

Not a clean fix at n=3 — same "Acanthostichus wins, Acromyrmex/Acromyrmex-and-Solenopsis don't"
shape as TASK6's GWN result, despite targeting a completely different, unrelated mechanism.
That repetition across two architecturally unrelated fixes (inside/outside detection vs.
correspondence sampling density) was read as evidence the variable that matters is a property
of the SPECIMEN, not of which lever gets pulled — motivating the gate investigation below.

## Part 2 — gate hypothesis: density ratio + baseline concentration

Two candidate per-specimen predictors, computed from data already on disk (no new GPU cost for
the variables themselves):
- **Legs:gaster vertex-density ratio** — the direct mechanism item 1 corrects. Computed from
  fitted mesh surface area (fixed template vertex COUNT per part ÷ specimen-specific fitted
  AREA). Pose-only proxy (Stage_0/1, no deform) tried first and **rejected**: up to +149% error
  and flipped rank order vs. the real Stage_3 value on the 3 known specimens, since
  `deform_verts` (Stage_2+) is what actually corrects individual proportions away from the
  shared template. Stage_2-only **validated** as a cheap-enough substitute: 7.8%/9.3%/−2.3%
  error, rank order preserved. Computed for all 50 bench50_clean specimens on Stage_2
  (`fitter_3d/run_density_ratio_probe.py`, `diagnostics/density_ratio_bench50.csv`).
- **Baseline penetration concentration** — fraction of a specimen's total baseline (no
  penetration loss) penetration count sitting in its single largest pair/direction. No proxy
  needed, read directly from existing per-pair CSVs. Computed for all 50.

At n=50, ratio spans 0.87–8.00 (median 2.89), concentration spans 0.277–0.966 (median 0.684).
Solenopsis sits at rank 50/50 on concentration — the single most extreme specimen in the whole
corpus, a real outlier not just "a bit high." Acanthostichus and Acromyrmex, by contrast, are
both unremarkable on concentration (ranks 25 and 28 of 50) despite opposite outcomes — the
variable that separates them is density ratio (3.84 vs. 2.42), not concentration. This
motivated treating the two variables as separate, independent checks (fix-relevance vs.
collateral-risk) rather than one combined score, per the pre-registered design.

**8 new specimens chosen deliberately from the (ratio, concentration) map** — 5 as nearest-
neighbor replications of the 3 known points (tests whether "nearby in variable-space" means
"same outcome"), 3 as untested corner combinations (conflicting-signal case, extreme-ratio
tail, inverted-ratio/lowest-concentration case). Not a random or dense sample — see
`fitter_3d/run_symmetric_chamfer_experiment_gate_probe.py` docstring for the exact selection
reasoning per specimen.

| specimen | ratio | conc | Δcount | Δf_score | selected as |
|---|---:|---:|---:|---:|---|
| Cephalotes_simillimus | 4.01 | 0.679 | −31.5% | −0.001 | near Acanthostichus |
| Strumigenys_sp.appretiata | 2.42 | 0.808 | **−79.6%** | **+0.057** | near Acromyrmex |
| Cephalotes_spinosus | 2.71 | 0.802 | −33.2% | +0.002 | near Acromyrmex |
| Mayriella_sp. | 2.76 | 0.930 | −59.4% | +0.055 | near Solenopsis |
| Crematogaster_nawai | 2.20 | 0.875 | +17.7% | +0.030 | near Solenopsis |
| Nesomyrmex_angulatus | 5.12 | 0.954 | −13.8% | +0.055 | corner: high ratio + high conc |
| Ectatomma_brunneum | 8.00 | 0.808 | **+65.1%** | +0.009 | corner: most extreme ratio in corpus |
| Eciton_hamatum | 0.87 | 0.277 | −8.9% | **+0.100** | corner: inverted ratio, lowest conc |

## Reading

> **CORRECTION (see Part 3 below) — read before citing this claim.** The paragraph immediately
> below reports Strumigenys_sp.appretiata's single unseeded run as a −79.6% count win,
> directly contradicting Acromyrmex's near-identical position and used as the evidence this
> gate hypothesis fails replication. Part 3's 3-seed rerun of the same specimen/config found
> that result **was seed noise, not a stable outcome**: +12.7%/+2.9%/−49.4% across seeds 0/1/2
> (mean −11.3%, only 1/3 seeds actually a win) — nothing like the single run's −79.6%.
> Acromyrmex's own result IS seed-stable (unanimous loss, all 3 seeds). The "two identical
> points, opposite outcomes" contradiction below rests on one anchor point that wasn't real.
> The gate is still not validated (a not-reliable anchor isn't evidence FOR the gate either),
> but this specific falsification is weaker than it reads below — see Part 3 for the full
> correction.

**The replication test failed at its sharpest point.** Strumigenys_sp.appretiata was selected
specifically as Acromyrmex's nearest neighbor in (ratio, concentration) space — ratio 2.42 vs.
2.42 (identical), concentration 0.808 vs. 0.762 (close) — and it produced the single BEST
result of all 11 specimens tested (−79.6% count, +0.057 F-score), while Acromyrmex itself is
the worst. Two specimens essentially co-located in this 2D space produced opposite outcomes.
**Whatever actually determines the outcome is not captured by density ratio or baseline
concentration** — the core hypothesis this gate investigation set out to test is falsified by
direct replication, not just weakly supported.

**Ectatomma_brunneum breaks the "higher ratio → bigger win" extrapolation.** At ratio=8.00 (2x
Acanthostichus's 3.84, the most extreme in the whole 50-specimen corpus), the mechanism
predicted the strongest win — instead it's the second-worst result (+65.1% count, worse than
Acromyrmex's +27.6%). The relationship is not monotonic; if density ratio matters at all, it
saturates or reverses at the extreme rather than extrapolating.

**Despite the gate hypothesis failing, the aggregate result across all 11 specimens is the
strongest in the entire investigation**: aggregate count −5.1%, mean F-score **+0.0276** — both
sides of TASK5's own pre-registered bar (aggregate ≤0%, mean F ≥ −0.01) cleared simultaneously,
which no arm across TASK1-6 achieved. 7/11 win on count, 9/11 improve on F-score; Acromyrmex is
the only specimen with a real loss on both axes at once. Heavy caveats apply: n=11, no seeds,
and the sample was deliberately chosen near known "bad" points and untested corners — not a
random or representative draw, so this number is not a population estimate. If anything the
selection bias should have pushed the aggregate toward the pessimistic side (half the new
picks were chosen specifically because they neighbored a losing specimen), which makes the net
positive result more notable, not less.

## Part 3 — seeded scale validation (10 specimens x 3 seeds, random-sampled)

n=11 was single-seed and deliberately curated (near known points + untested corners) --
exactly the profile FINDINGS.md's own history says not to trust before multi-seed validation.
Ran the full 10x3-seed panel: 2 fixed specimens (Acromyrmex_coronatus, Strumigenys_sp.appretiata
-- the pair that directly contradicted the gate hypothesis in Part 2, kept in deliberately so
they sit inside a real distribution rather than only as curated outliers) + 8 drawn by an actual
random sample (`Random(42).sample`, reproducible, documented) over the remaining 48 bench50_clean
specimens. NOT representative-by-construction of the full 50-specimen population any more than
any single random n=10 draw is -- stated explicitly so this isn't over-read as a population
estimate either. Same isolation discipline as every arm before it: TASK5's config held fixed,
`w_sdf=0` explicit, `symmetric_chamfer_sampling` the only variable, seeds 0/1/2 (same convention
as FINDINGS.md's own reseeded rerun and `optimise.py --seed`).
Script: `fitter_3d/run_symmetric_chamfer_experiment_seeded_scale.py`.

| seed | aggregate count: control→exp | Δcount | mean F: control→exp | ΔF |
|---|---|---:|---|---:|
| 0 | 5337→4920 | −7.8% | 0.7364→0.7944 | +0.0580 |
| 1 | 3747→4158 | +11.0% | 0.7519→0.7804 | +0.0284 |
| 2 | 4335→4363 | +0.6% | 0.7523→0.7818 | +0.0295 |
| **mean** | | **+1.3%** | | **+0.0386** |

**F-score wins robustly, in every seed** -- this part of the n=11 curated result holds up at
scale. **Aggregate count does not** -- it averages to roughly flat (+1.3%, i.e. a small net
LOSS not a win), swinging from -7.8% to +11.0% seed to seed. This does not clear TASK5's own bar
(aggregate <=0% AND mean F >=-0.01, simultaneously): F clears it comfortably and consistently,
count does not clear it at all.

**Per-specimen, count is largely seed noise, same as this project's original gentle-proximity
finding.** Only 3/10 specimens have a stable count-direction across all 3 seeds (3 unanimous
win, 1 unanimous loss, 6/10 flip sign between seeds -- e.g. Carebara_trechideros:
-35.9%/+557.4%/+353.5%, same specimen, no consistent story). This directly echoes FINDINGS.md's
own "44% of specimens don't have a seed-stable answer" result for the unrelated gentle-proximity
arm -- not unique to this fix, a property of the count metric's variance at this specimen-count
scale generally.

**This also puts an asterisk on Part 2's gate-hypothesis falsification.** Strumigenys_sp.
appretiata -- the single curated run that beat everything and was the direct evidence the gate
hypothesis failed replication against Acromyrmex -- turns out to be **seed noise, not a stable
result**: +12.7%/+2.9%/-49.4% across the 3 seeds (mean -11.3%, only 1/3 actually a win, nothing
like the single run's -79.6%). Acromyrmex, by contrast, IS stable: unanimous loss, all 3 seeds
(+16.8%/+52.3%/+64.9%). One of the two anchor points in Part 2's "near-identical position,
opposite outcome" contradiction wasn't real -- the replication test's premise (that both points
were reliable single-seed reads) doesn't hold. This doesn't resurrect the gate (Acromyrmex being
real and stable doesn't mean density ratio/concentration explain it), but it does mean Part 2's
falsification was more equivocal than it looked at the time -- worth knowing if this gets
revisited, not corrected by re-running the gate now per this task's own scoping discipline.

## Part 4 — why is count so seed-unstable? (short, focused check, not a fourth fix attempt)

Two independent arms now (gentle-proximity, per FINDINGS.md's reseeded rerun; symmetric-chamfer,
Part 3 above) show the same seed-instability in penetration count across a large fraction of
specimens, regardless of which loss mechanism produced the fit. That repetition across
unrelated mechanisms points at the metric/fitting stochasticity itself, not any one loss term --
worth a short mechanistic check before treating it as unexplained noise.

**Method**: per-vertex penetration classification (gaster-into-legs, the dominant pair) compared
between seed 0 and seed 1's control-arm fits, using the already-saved Stage_3 vertices (CPU-only,
zero new GPU compute) -- a numpy port of `penetration_loss.py`'s own matching/threshold logic
(nearest-triangle-centroid, signed distance to that triangle's plane, `proximity_tau_fraction`
gate). For each specimen: which query vertices flip in/out of "penetrating" between the two
seeds, and whether flipped vertices sit closer to the sign=0 classification boundary than
stably-classified ones.

| specimen | close verts | flip 0→1 (%) | mean\|sign\| flip | mean\|sign\| stable |
|---|---:|---:|---:|---:|
| Acromyrmex_coronatus (unanimous lose) | 1132 | 26.6% | 0.0032 | 0.0218 |
| Strumigenys_sp.appretiata (unstable) | 883 | 30.5% | 0.0367 | 0.0607 |
| Strumigenys_alberti (unanimous win) | 566 | 28.4% | 0.0063 | 0.0117 |
| Centromyrmex_brachycola (unanimous win) | 1011 | 37.1% | 0.0230 | 0.0227 |
| Acromyrmex_lobicornis (unanimous win) | 990 | **40.5%** | 0.0069 | 0.0083 |
| Eciton_hamatum (unstable) | 1231 | 39.6% | 0.0051 | 0.0083 |
| Dilobocondyla_fouqueti (unstable) | 1205 | 36.7% | 0.0243 | 0.0199 |
| Cyphomyrmex_cf.minutus (unstable) | 291 | 15.8% | 0.0269 | 0.0418 |
| Cephalotes_minutus (unstable) | 534 | 24.7% | 0.0194 | 0.0171 |
| Carebara_trechideros (unstable) | 1214 | 28.4% | 0.0030 | 0.0082 |

**Confirmed, with a real limit.** In 7/10 specimens, flipped vertices sit measurably closer to
the classification boundary than stably-classified ones (Acromyrmex_coronatus: 6.7x closer;
Carebara_trechideros: 2.7x) -- direct evidence the near-tie mechanism drives individual-vertex
flips, not a stability artifact of this check. But it does NOT explain aggregate-count
stability on its own: 3/10 specimens (Centromyrmex_brachycola, Dilobocondyla_fouqueti,
Cephalotes_minutus) show flip-vertices no closer to (sometimes farther from) the boundary than
stable ones, and per-vertex flip rates are large even in specimens whose AGGREGATE count
direction is seed-stable -- Acromyrmex_lobicornis ("unanimous win") has the single highest flip
rate of all 10 specimens (40.5%) despite a consistent net direction across seeds. Aggregate
stability looks like it depends on whether the underlying per-vertex churn happens to net out
symmetrically (flip-in roughly balancing flip-out) rather than on how much churn exists in
absolute terms -- a second-order question, not pursued further here per the "short, focused
look, not a fifth fix" scoping this was given.

## Part 5 — re-scored under soft count (zero new runs, same seeded data)

`penetration_soft_num_penetrating` (added to `penetration_loss.py`/`trainer.py` after Part 4)
recomputed retroactively on the same 6 saved Stage_3 npz files from Part 3 -- no new GPU
compute. Directly answers whether Part 3's null result on count was the real effect or hard-count
noise masking a real one.

| seed | hard: control->exp (delta) | soft: control->exp (delta) |
|---|---|---|
| 0 | 5341->4925 (-7.8%) | 7569.4->6686.6 (**-11.7%**) |
| 1 | 3748->4160 (+11.0%) | 5843.5->6132.4 (+4.9%) |
| 2 | 4340->4365 (+0.6%) | 6314.6->6137.5 (-2.8%) |
| **mean** | **+1.3%** | **-3.2%** |

Neither of the two framings this was meant to distinguish is fully right on its own. The
instability is real -- soft count still flips sign across seeds (seed 1 stays positive/worse
even under the lower-noise metric), so this isn't "hard count was just noise, the fix always
worked." But the hard count's noise WAS partially masking a real average effect: the mean flips
from +1.3% (fails TASK5's aggregate<=0% bar) to -3.2% (passes it on average, though not in
every seed). Per-specimen, soft count has lower cross-seed CV in **10/10 specimens** (control
arm) -- e.g. Carebara_trechideros 114.3%->80.1%, Dilobocondyla_fouqueti 58.3%->37.6% -- a clean,
complete confirmation the metric fix works as designed, independent of what it says about this
specific fix. Reading: `symmetric_chamfer_sampling` likely has a real, modest, positive average
effect on count that single-seed hard-count noise was capable of hiding entirely (as it did in
the original n=3 and even shaded n=11) -- genuinely more promising than Part 3 alone suggested,
but "modest average effect, one seed still unfavorable" is still not an unconditional pass.

## Verdict

**Ship the F-score result on its own terms -- it doesn't need to wait on count.** These are two
separate claims about one flag, not one combined verdict: (1) `symmetric_chamfer_sampling`
improves surface fit, seed-stably, at n=10x3; (2) it does not reliably reduce penetration count
at n=10x3. Claim 1 is real, positive, and independently useful regardless of how or whether
claim 2 ever resolves -- it corrects a genuine, previously-diagnosed asymmetry
(FINDINGS.md step 3) in the chamfer term specifically, which has nothing to do with penetration
count as a metric. `symmetric_chamfer_sampling=True` is available now as an opt-in flag
(`fitter_3d/trainer.py`, default `False`, zero effect on any existing config) for anyone who
wants better surface fit and doesn't need the penetration-count question solved first.

`symmetric_chamfer_sampling` reliably improves surface fit (mean F-score, positive in all 3
seeds, +0.028 to +0.058) but does not reliably reduce penetration count at scale (mean +1.3%,
sign-flipping across seeds) -- a genuine partial result, not the clean simultaneous pass the
n=11 curated sample suggested. Does not clear TASK5's own acceptance bar. Standing
recommendation (`REVIEW_penetration_update.md`: ship as a gated option, not a default) holds,
now on stronger, seeded evidence than any other arm across TASK1-7 -- and with a real, positive
F-score effect worth keeping in mind for future combination attempts, even though count alone
isn't there. The two-variable gate (density ratio x baseline concentration) does not survive
its Part 2 replication test and should not be built as designed -- though per Part 3, that
specific test's evidentiary weight is weaker than it first appeared, since one of its two
anchor points wasn't seed-stable. Acromyrmex remains a genuine, reproducible outlier -- worth
understanding on its own terms if this is picked up again, not as a representative of a
filterable "region" in (ratio, concentration) space, which Part 2's data doesn't support.
Natural next step if this is picked up again: understand WHY count is this seed-unstable for
most specimens (mirroring, and possibly sharing a root cause with, the same instability already
documented for the unrelated gentle-proximity arm) -- that's a different, likely more
fundamental question than anything specific to this fix, and answering it might matter more for
this project's penetration work generally than any single arm's pass/fail result.
