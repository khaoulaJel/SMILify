# TASK 6 — GWN-based matching primitive as a training signal: causal test

Answers the question `TASK3_gwn_disagreement_baseline_vs_gentle.md` §5 and
`scripts/penetration_joint_study/FINDINGS.md` Step 6 both left open: Step 6 confirmed the
proximity-test's inside/outside verdict is measurably less accurate specifically in the
direction that fails to improve under training, and proposed (never tested) replacing it with
a GWN-based signal. This is that test — not another diagnostic, an actual intervention.

## Provenance

Ran on JURECA (`/p/home/jusers/jellal1/jureca/SMILify/`), not this branch's local checkout —
flagging per this project's branch-reconciliation discipline, since the local checkout has no
record of these job outputs (only the sbatch scripts that submitted them). Chain of jobs:
15520880 (uncalibrated `w_penetration_gwn=0.02/0.05` by analogy from the unrelated proximity
loss — catastrophic, GWN loss hit 70.5% of total weighted loss) → 15521221 (calibration probe,
`fitter_3d/gwn_weight_calibration_probe.py`, landed on `w=0.001`) → 15521230 (calibrated
1000-iteration Stage_3, "essentially flat" on Acromyrmex per the sbatch comment) →
**longstage3** (this result: same calibrated weight, Stage_3 extended 1000→3000 iterations,
`fit3d_results_gwn_gentle_calibrated_longstage3/three_way_comparison.csv`) — submitted because
a checkpoint-bracketed check on the 1000-iteration run found the GWN loss and `B_into_A`
proximity count still declining substantially through Stage_3, not plateaued: a rate problem,
not a weight-ceiling problem. A local reproduction attempt was started on this branch
(`fitter_3d/run_gwn_penetration_experiment.py`, `GWN_PEN_STAGE3_NITS=3000`) before this result
was located on the cluster; killed early (10/300 into Stage_1) once confirmed redundant.

## Method (unchanged from the original TASK-3-adjacent design, not re-litigated here)

3 specimens, chosen in the original design specifically to span the mechanism: Acanthostichus
(baseline case), Acromyrmex (**strongest** proximity-test disagreement asymmetry per Step 6's
pilot), Solenopsis (**no** disagreement signal — contrast case). `w_penetration_gwn` trains
**only** the gaster-legs pair (`GWN_PAIRS = [("gaster", "legs")]`); `w_penetration=0` throughout
(isolated, one-variable-at-a-time — proximity loss and GWN loss are never combined). Evaluation
metrics (`penetration_num_penetrating` etc.) are the **all-non-adjacent-pairs aggregate**, same
convention as TASK1-5, scored identically regardless of which loss trained the geometry.

## Results (all-pairs aggregate, 3 specimens, no seeds)

| specimen | count: baseline→proximity→**gwn** | gwn vs proximity (count) | gwn vs baseline (count) | f_score: baseline→proximity→**gwn** | gwn vs proximity (f_score) |
|---|---|---:|---:|---|---:|
| Acanthostichus | 711→599→**517** | **−13.7%** | **−27.3%** | 0.9421→0.9731→**0.9617** | −0.0114 |
| Acromyrmex (strongest signal) | 1044→857→**971** | **+13.3%** | −7.0% | 0.8984→0.8799→**0.8021** | **−0.0778** |
| Solenopsis (no signal, contrast) | 537→656→**803** | **+22.4%** | **+49.5%** | 0.8600→0.8379→**0.8365** | −0.0014 |

Depth (`penetration_mean_depth_among_penetrating`) mostly stayed flat or improved slightly under
GWN relative to proximity (Acanthostichus 0.0126→0.0189 worse, Acromyrmex 0.0181→0.0190 roughly
flat, Solenopsis 0.0234→0.0305 worse) — no clean depth win either, unlike the proximity loss's
own depth-improves/count-doesn't split from TASK1.

## Reading

**Not the clean fix the hypothesis needed.** Only Acanthostichus wins on both count and F-score.
Acromyrmex — deliberately the specimen with the *strongest* proximity-test accuracy asymmetry,
the one this mechanism was most expected to help — instead takes the worst F-score hit of the
three (−0.078 vs. the existing proximity loss, a genuinely large surface-fit cost) while its
count is *worse* than proximity, not better, and only 7% better than doing nothing at all.
Solenopsis, the contrast case with no original disagreement signal, gets meaningfully worse on
count under GWN (+22.4% vs. proximity, +49.5% vs. baseline) with no offsetting benefit anywhere.

**Open question this table cannot answer on its own**: whether Solenopsis's and Acromyrmex's
aggregate-count regressions are collateral damage on OTHER pairs while gaster-legs itself
(the only pair GWN actually trained) improved — the exact TASK1 shape ("contact gets shallower
on the targeted pair, more widespread elsewhere"), just recurring under a different matching
primitive. This CSV only has the all-pairs aggregate, not the per-pair breakdown TASK1/TASK3
used to establish that pattern. If this thread is picked up again, a per-pair count table (same
method as TASK1 §1, or the disagreement diagnostic's per-pair table) on these same 3 specimens'
saved output would settle it directly, at zero new GPU cost.

**Answered** (per-pair CSVs recovered from `fit3d_results_all_baseline`,
`fit3d_results_all_gentle`, and this task's own `fit3d_results_gwn_gentle_calibrated_longstage3`,
same 3 specimens, split into gaster-legs (the trained pair) vs. the sum of the other 4 pairs that
touch `legs` and so share deformation with it):

| specimen | gaster-legs: baseline→proximity→**gwn** | other pairs: baseline→proximity→**gwn** |
|---|---|---|
| Acanthostichus | 504→313→**306** | 207→286→**211** |
| Acromyrmex | 823→512→**702** | 221→345→**269** |
| Solenopsis | 535→291→**482** | 2→365→**321** |

Both counts independently sum to the previously-reported aggregates (exact match, e.g.
504+207=711=baseline's published total) -- an internal consistency check, not just a new claim.
**Important correction to how this comparison reads**: `fit3d_results_all_gentle` ("proximity"
above) is TASK1's original ALL-10-PAIRS gentle arm, not TASK4/5's pair-scoped one -- it directly
trains gradients on every pair, GWN trains only gaster-legs. So proximity's own "other pairs"
number partly reflects direct (net-unhelpful, per TASK1) training pressure on those pairs, not
pure collateral. GWN's "other pairs" number, by contrast, is **pure geometric coupling** -- GWN
never computes a gradient for any pair but gaster-legs, so every one of those 211/269/321
instances arose purely from deforming the mesh to fix gaster-legs perturbing vertices shared with
neighboring parts, not from misdirected training pressure.

On the trained pair itself, GWN is the *weaker* fix on 2 of 3 specimens (proximity's own
gaster-legs reduction is bigger: Acanthostichus roughly tied, −37.9% vs. GWN's −39.3%; Acromyrmex
−37.8% vs. GWN's only −14.7%; Solenopsis −45.6% vs. GWN's only −9.9%) while ALSO still generating
real new collateral on pairs it never trains (flat on Acanthostichus, real on Acromyrmex +21.7%,
large in absolute terms on Solenopsis: baseline near-zero at 2 instances, up to 321 under GWN --
comparable in scale to proximity's own 365, even though proximity is the one actively training
those pairs and GWN is not). Solenopsis's mandible-antenna anomaly flagged above is now
explained, and sharpened, not resolved in GWN's favor: baseline 0 → proximity 111 → **GWN 223**
-- GWN's collateral on this one sub-pair is roughly DOUBLE proximity's, even though GWN's total
"other pairs" collateral (321) is slightly below proximity's (365). GWN doesn't avoid collateral,
it concentrates it differently -- more narrowly, almost entirely into mandible-antenna, instead
of proximity's broader spread.

**Revised reading**: this is not "GWN was mis-scoped, pair-scope it and the aggregate result
would clean up" (last turn's hopeful framing). GWN's collateral arises purely from deformation
coupling -- fixing gaster-legs moves vertices that neighboring parts share, regardless of which
loss mechanism did the fixing -- so scoping GWN's *evaluation* differently would not remove it;
the coupling is a property of the mesh deformation itself, not of misattributed gradient credit.
And GWN is simultaneously the weaker fix on the pair it does train, on 2 of 3 specimens. Net:
GWN is gentler in both directions at once (smaller target-pair win, but also smaller-to-comparable
collateral) rather than a primitive that cleanly outperforms the existing proximity test once
correctly scoped. Whether TASK4/5's own PAIR-SCOPED proximity loss (not this all-pairs "gentle"
arm) shows the same deformation-coupling collateral on these 3 specimens remains untested -- that
per-pair CSV was not part of this recovery and would be the natural next comparison if this
mechanism is investigated further, since it would isolate deformation-coupling collateral (shared
by any pair-scoped loss, primitive-independent) from primitive-specific behavior cleanly.

**The longstage3 extension answered its own question, unfavorably**: it was submitted because
1000 iterations looked "still declining, not plateaued" — hoping more compute would resolve it
cleanly. 3000 iterations later, the aggregate result is mixed-to-negative across 2 of 3
specimens, not resolved. "Give it more iterations" bought Acromyrmex a modest further count
improvement (1041-ish at 1000 iters, per the sbatch comment, down to 971 here) but did not fix
its F-score cost or Solenopsis's regression — consistent with this being a primitive with a
real gradient-quality problem (see the GauWN literature note in
`DELIVERABLE_penetration_fix_TASK1-5.md`'s torch-mesh-isect addendum: plain discrete GWN's
gradients near the inside/outside boundary are known to be rough, motivating smoothed variants
in recent work), not merely an undertrained one.

## Caveats (unchanged in kind from every other arm in this investigation)

n=3, no seeds — this project's own established bar throughout TASK1-5 is 10 specimens x 3 seeds
with a sign test before treating a direction as reliable, not met here. Single pair
(gaster-legs) trained; all-pairs aggregate evaluated. No manifoldness risk (per
`gwn_disagreement.py`'s module docstring, structurally guaranteed for SMIL fits, not re-derived
here). Not combined with `w_offset`/pair-scoped/barriers (TASK2/4/5's stack) — this tests the
matching primitive swap in isolation against the ORIGINAL gentle-proximity baseline, not against
the current best (TASK5) config, so it is not directly comparable to the −16.9% aggregate
headline number.

## Verdict

The GWN-based matching primitive, as an isolated causal intervention, does not fix Step 6's
proximity-test accuracy asymmetry cleanly — on this small sample it trades one specimen's clean
win for a large F-score cost on the specimen the mechanism was expected to help most, and a
substantial count regression on the specimen it was expected to leave untouched. Does not
change the standing default recommendation (`REVIEW_penetration_update.md`, reaffirmed through
TASK5): default stays no-penetration-loss, soft losses stay gated/optional. Adds a second,
independent negative-leaning data point (alongside the torch-mesh-isect addendum above) against
this investigation's two candidate matching-primitive fixes; the per-pair breakdown described
above is the cheapest remaining thread if this line of work continues.
