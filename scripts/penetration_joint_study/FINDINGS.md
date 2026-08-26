# Penetration-loss count/depth split: investigation chain and findings

Investigates why the gentle-weight penetration loss (`fitter_3d/penetration_loss.py`)
makes inter-part contact shallower on most specimens without reducing how often
collisions occur (Khaoula's original finding). Five steps, each falsifying or
narrowing the previous explanation. Data for all steps is preserved on disk;
paths given throughout.

## 1. Pose-freeze hypothesis -- falsified

**Hypothesis** (Fabian): pose is frozen during the deform stages where the
penetration loss is active, so the optimizer can only resolve a collision by
denting the mesh locally, never by rotating a limb away.

**Test**: re-ran the baseline-vs-gentle ablation with `scheme: 'all'` in the
deform stages instead of `scheme: 'deform'`, so `joint_rot` stays available
throughout. 50 specimens (`diagnostics/moonshot/bench50_clean/`), full
Stage_0-3 schedule. Configs: `fitter_3d/ants_cfg_all_baseline.yaml`,
`fitter_3d/ants_cfg_all_penetration_gentle.yaml`. Results:
`fit3d_results_all_baseline/`, `fit3d_results_all_gentle/`.

**Pre-registered prediction**: if pose-freezing is the real mechanism,
unfreezing it should let collision COUNT shrink or invert, not just depth.

**Result**: rotation was genuinely exploited (converged-pose `joint_rot`
differs by 6.9 deg on average between arms, max 61 deg, 89% of joints moved
>1 deg under the gentle loss) -- not just nominally available. Despite that:

| metric | baseline | gentle | delta |
|---|---|---|---|
| mean penetrating instances/specimen | 565 | 660 | **+16.8%** |
| mean depth among penetrating instances | 0.0290 | 0.0171 | **-41.0%** |
| per-specimen count improved/worse | -- | -- | 23 / 27 (coin flip) |

Depth replicates the original finding almost exactly; count does not shrink
or invert. **Falsified**: pose being frozen was not what was preventing count
from improving.

**Correction, now resolved**: `w_limit: 100.0` was originally set only in
`Stage_1_default`'s `loss_weights` in both configs -- `Stage_2`/`Stage_3`
(where the 61 deg swings happened) silently fell back to the trainer
default of `0.0`, so "rotation was available" above should have read
"*unconstrained* rotation was available." Fixed (`w_limit: 100.0` now set
in all four stages of both `ants_cfg_all_*.yaml` configs) and reconfirmed
via a 3-seed x 2-arm reseeded rerun -- see "Reseeded rerun: falsification
confirmed under biomechanically-bounded rotation" below. The headline
result is unchanged; the per-specimen "coin flip" framing is sharpened.

**Second finding, independent of the count/depth split**: chamfer distance,
computed post-hoc from saved verts (all 50 specimens), shows the gentle
arm is not cost-free even though the mean looks flat. Mean delta
gentle-vs-baseline is +1.32%, but the spread is real: max +7.02%, min
-4.23%, and **17/50 specimens (34%) exceed the |3%| chamfer-delta bound**
used elsewhere in this project's review process. Stated plainly: *the
gentle penetration weight measurably degrades surface fit on over a third
of specimens, even though the mean chamfer number alone would suggest
nothing changed.*

## 2. The depth clamp (mechanism 1) -- ruled out

**Hypothesis**: `penetration_max_depth_fraction` clamps loss contribution
per vertex; `torch.clamp`'s gradient is zero beyond the clamp, so severely
penetrating vertices might never be pushed to resolve.

**Test**: direct measurement -- observed `penetration_max_depth` vs. the
configured clamp ceiling (`0.08 * bbox_diag`), all 50 specimens, gentle arm.

**Result**: max observed depth never exceeds 37.1% of the clamp ceiling
(mean 36.2%, 0/50 specimens above even 50%). The clamp is never active.
**Ruled out** -- more decisive than the persistence check alone, since it
shows the mechanism's precondition (hitting the clamp) never occurs.

## 3. Per-pair persistence + directional asymmetry -- descriptive finding

**Test**: for each (specimen, part-pair, direction) instance, classified as
resolved (penetrating in baseline, clean in gentle), created (clean ->
penetrating), or persisted (penetrating in both). Same-day analysis, no new
GPU time, using `Stage_3_deform_fine_per_pair_penetration.csv` from step 1.

**Result**: 92.1% of baseline-penetrating instances persist in the gentle
arm; only 7.9% fully resolve; 13.2% of previously-clean instances get newly
created. Pair identity is stable (Spearman rho=0.929 on pair totals -- same
anatomical pairs dominate in both arms), but **direction within a pair
often flips**: for (part_a=gaster, part_b=legs), `A_into_B` ("gaster verts
penetrating into legs") worsens baseline 1528 -> gentle 5558 (+4030), while
`B_into_A` ("legs verts penetrating into gaster") improves baseline 19205
-> gentle 10869 (-8336) -- in the same pair, same arm. Within the 293
persisted instances specifically: depth improved in 87% but count got worse
in 59%.

**Candidate mechanism for the directional flip, tested against saved
geometry (no new GPU time)**: stock chamfer compares a fixed 3000-point
area-weighted sample of the TARGET against all 10,229 raw TEMPLATE
vertices -- not a matched, area-weighted sample on both sides (documented
defect, this project's chamfer-asymmetry writeup; Fabian's own symmetric-
sampling fix dropped Stage_2 chamfer by 48%, so the asymmetry is real and
measured, not speculative). Under this asymmetry, a part with denser raw
template topology per unit surface area gets pulled toward the target
harder than an equally-sized sparse part.

Measured directly on the SMAL rest-pose template (`v_template`) and this
project's own `part_faces`: **legs are 2.36x more vertex-dense than
gaster** (8984 vs. 3808 vertices per unit surface area). This is exactly
the direction needed: the vertex-dense side (legs) is the one that
IMPROVES (`B_into_A`, legs retreating out of gaster, -8336) -- consistent
with legs being anchored more strongly to their true target position by
the chamfer asymmetry. The vertex-sparse side (gaster) is the one that
WORSENS (`A_into_B`, gaster advancing into legs, +4030) -- consistent with
gaster being anchored more weakly, giving it more freedom to drift into
leg-occupied space once legs have retreated there. **Plausible, evidenced
mechanism, not proven**: the vertex-density asymmetry and its predicted
direction both check out against real data, but no ablation has isolated
it yet (e.g. rerunning with symmetric-sampled chamfer and checking whether
the directional flip disappears). That ablation is the natural next test
if this thread is picked back up.

## 4. Neighbor-drag hypothesis -- adjacency-confirmed (correlational)

**Hypothesis**: local smoothness terms (edge/laplacian) drag previously-clean
neighboring vertices along when a badly-penetrating vertex retreats,
converting them from just-outside to just-inside the collision threshold.

**Test**: recomputed `penetrating_vertex_mask` from already-saved final
vertex positions (no retraining) for both arms. For each "created"
(newly-penetrating) vertex, measured mesh-hop distance to the nearest
"resolved" vertex, vs. a null of random same-part vertices.

**Result**: created vertices sit at median 7.0 hops from a resolved site,
vs. 14.0 hops for the random-same-part null (~2x enrichment, holds in 43/50
specimens individually). Correlational support for neighbor-drag.

## 5. Neighbor-drag causal intervention -- does not support the hypothesis

**Design**: `fitter_3d/local_smoothness.py` + `Stage`'s optional
`local_downweight` (opt-in, every existing config unaffected). Two arms,
matched on touched-vertex COUNT and per-vertex weight every iteration,
differing only in spatial arrangement:
- **local**: down-weight (factor=0.1) vertices within k=2 mesh-hops of a
  currently-penetrating vertex.
- **global (control)**: down-weight the SAME COUNT of vertices, same
  factor, but a FRESH random subset of the whole mesh drawn every
  iteration (not a fixed subset -- deliberate choice, isolates
  clustered-vs-scattered at the cost of giving the control zero temporal
  persistence at any one location).

3 specimens: `Acanthostichus_aff.brevicornis`, `Acromyrmex_coronatus`
(strongest neighbor-drag signal in step 4), `Solenopsis_invicta` (no signal
-- contrast case). Full Stage_0-3 schedule identical to step 1's gentle
config. Script: `fitter_3d/run_neighbor_drag_experiment.py`. Results:
`neighbor_drag_results/`.

**Result**:

| specimen | local count | global count | no-downweight (step 1 gentle) |
|---|---|---|---|
| Acanthostichus | 1086 | 885 | **599** |
| Acromyrmex | 1849 | 912 | **857** |
| Solenopsis | 338 | 391 | 656 |

Chamfer guardrail (post-hoc, vs. the no-downweight reference): Acanthostichus
local +6.9% / global +4.1%; Acromyrmex local +1.9% / global -4.2%; Solenopsis
local -3.2% / global +0.6%. No consistent direction, but 2/6 (arm, specimen)
combinations already exceed the |3%| bound at n=3 -- another data point
against the interventions being a clean win, on top of the count/F-score
results below.

1. **local vs. global-control**: local is worse than the matched scattered
   control on 2/3 specimens (Acromyrmex: ~2x worse), better only on
   Solenopsis -- the specimen already flagged as not showing the
   neighbor-drag adjacency signal in step 4.
2. **both interventions vs. doing nothing**: on 2/3 specimens, both local
   and global-control show MORE penetrating instances, and worse F-score,
   than the untouched gentle arm from step 1.
3. **Coverage confound**: k=2 touched 24-31% of the mesh on average
   (2487-3193 / 10229 vertices/iteration) in the real run -- not a small
   local perturbation. Likely structural: an ant mesh's thin, elongated
   parts (legs, antennae, mandibles) have few vertices around their
   circumference, so a k-hop neighborhood on a leg covers most of the
   leg within 1-2 hops regardless of k. A smaller k would not obviously
   fix this (see "not pursued further" below).
4. **Mesh integrity**: `folded_face_frac` (~8-12%) is statistically
   indistinguishable across local/global/no-downweight -- normal for a
   fully-articulated ant pose relative to rest-pose, not caused by the
   intervention. `edge_logratio_mean_abs` runs 10-25% higher under local
   than no-downweight on 2/3 specimens (global sits in between) -- a real
   but modest cost, not catastrophic, with no offsetting benefit in count.

**Conclusion**: falsifies the specific fix (locally releasing edge/laplacian
constraints near collisions) on its own terms -- both variants underperform
doing nothing, independent of the locality question. This is the headline
result: releasing smoothness, clustered or scattered, does not help resolve
penetration and measurably costs surface fit and edge distortion. The
local-vs-global comparison is a secondary, confounded finding (via the
coverage issue above).

**Not pursued further, and why**: a coverage-capped or smaller-k rerun would
only clarify the confounded local-vs-global sub-question, not rescue the
already-falsified core idea (both variants lose to baseline). Solenopsis
being the one specimen where local beat global-control is consistent with
it already being a known outlier (no adjacency signal in step 4), not a
promising thread.

## 6. Proximity-test accuracy asymmetry -- confirmed mechanism

**Hypothesis**: `penetration_loss.py`'s inside/outside test is an explicitly
acknowledged approximation (nearest-triangle-CENTROID matching, not the true
closest point, per its own docstring). If that approximation is
systematically less accurate in one direction of a pair than the other, it
would feed the optimizer a miscalibrated gradient specifically in the
direction that fails to improve -- a mechanism distinct from (and testable
independently of) step 3's chamfer vertex-density story.

**Method**: an offline, CPU-only diagnostic, no new GPU time. For the
gaster<->legs pair (gentle arm, using already-saved Stage_3 verts), compared
the CURRENT proximity-based verdict (`_directional_penalty`'s own
`penetrating_mask`, the exact code path the training loss uses) against a
TRUE generalized winding-number inside/outside signal (Jacobson et al. 2013;
reused `winding_number()` from `diagnostics/moonshot/joint_placement_common.py`,
pure numpy, no `igl` dependency needed -- `igl` is not installed in this
project's conda env).

Two methodological steps were necessary before the comparison meant anything:
- **Closure**: `winding_number()` only gives a valid signal for a CLOSED
  mesh. A single anatomical part cut from the full mesh has open boundary
  loops wherever it attached to neighboring parts (e.g. "legs" alone is not
  closed), so each part was capped with a fan triangulation from each
  boundary loop's centroid, oriented outward, before use as the "true
  surface." This is a real geometric approximation local to the cap itself,
  not exact.
- **Manifoldness gate**: checked first, since GWN reliability degrading on
  ill-formed meshes is a documented, evidenced risk (see
  `diagnostics/moonshot/refute_gwn_validity_PROBE.py`, which found this on
  raw, independently-reconstructed SCAN meshes). Reimplemented
  `mesh_integrity_report`/`report_mesh_degeneracy`'s checks in pure numpy
  (those take a `bpy.types.Object`; `bpy` is also not installed here) and
  ran them on the rest-pose template plus 6 fitted Stage_3 outputs (3
  specimens x baseline/gentle): all came back perfectly manifold, watertight,
  and degeneracy-free (0 boundary edges, 0 non-manifold edges/verts, 0
  zero-length edges, 0 zero-area faces, 0 duplicate verts, uniformly). This
  is structural, not incidental: SMIL's fixed topology means pose/shape/
  `deform_verts` only move vertices, never face-vertex incidence, so
  manifoldness can't degrade under this model regardless of deformation --
  the moonshot scan-topology caveat doesn't transfer to this object.
- **Trustworthy band**: rather than a fixed absolute winding-number margin,
  gated each query vertex by whether `|w - 0.5|` exceeds `2.5x` that
  vertex's own local mean incident edge length (edge lengths vary ~30x
  across the mesh, finest near antennae/mandible tips at ~0.0004-0.0006 --
  the same order of magnitude as the penetration depths being measured, so
  a fixed threshold would have been meaningless in the finest regions).

**Result, 3-specimen pilot then 50-specimen replication (per the discipline
of this investigation: replicate before writing up)**:

Trustworthy fraction came back ~100% at both scales (17801/17808 in the
pilot, 50/50 specimens valid in the full run) -- query points sit at actual
mesh vertices, not arbitrary sub-edge-length offsets, so margin degradation
was not the limiting factor here, contrary to what the edge-length-scale
concern might have predicted.

Pilot (n=3) found disagreement between the proximity verdict and the true
GWN signal was 3-8x higher in the worsening direction (`A_into_B`,
gaster->legs) than the improving direction (`B_into_A`) for the 2 specimens
that showed the directional-asymmetry pattern, absent in the 1 that didn't.
**Replicated at full scale (all 50 specimens, using already-saved step-1
data, no new GPU time)**:

- Disagreement asymmetry (`rate_AtoB - rate_BtoA`) correlates with the
  actual count-worsening magnitude in `A_into_B`: **Pearson r=0.947,
  Spearman rho=0.935, both p<0.0001**.
- 44/50 specimens show `A_into_B` worsening; their mean asymmetry is
  **+0.082**, vs. **-0.007** for the 6 that didn't worsen (Mann-Whitney U,
  p=0.0004) -- a clean separation, not a handful of outliers driving it.
- The effect is specific to the worsening direction's own accuracy, not a
  general per-specimen noise confound: mean `rate_AtoB` is ~12x higher in
  the worsened group (9.5% vs 0.8%), while `rate_BtoA` barely differs
  between groups (1.2% vs 1.5%).

**Conclusion**: confirmed mechanism, not a pilot observation. The proximity-
based penetration test is measurably, significantly less accurate
specifically in whichever direction of a pair fails to improve under
training -- independent of, and additional to, step 3's chamfer
vertex-density story. This directly motivates a concrete fix, now evidenced
rather than only theoretically SOTA-motivated: replace the nearest-
triangle-centroid proximity test in `penetration_loss.py` with a
winding-number-based (or true SDF) inside/outside test as the loss's core
signal. **Not yet tested as an intervention** -- this diagnostic shows the
current test disagrees with ground truth, not yet that replacing it fixes
the count problem. That causal test (analogous to step 5's neighbor-drag
intervention) is the natural next step if this thread is picked up.
Scripts: `diagnostics/gwn_diagnostic.py` (3-specimen pilot),
`diagnostics/gwn_diagnostic_full.py` (50-specimen replication),
`diagnostics/manifoldness_check.py` (the manifoldness gate). Per-specimen
results: `diagnostics/gwn_full_results.csv`. Raw run output preserved as
`diagnostics/gwn_diagnostic_pilot_out.txt`, `diagnostics/gwn_diagnostic_full_out.txt`,
`diagnostics/manifoldness_check_out.txt`.

## What remains open

The count/severity split (depth improves, count does not) now has TWO
confirmed, evidenced (not yet fixed) mechanisms: chamfer vertex-density
asymmetry (step 3) and proximity-test accuracy asymmetry (step 6) -- both
real, not mutually exclusive. Ruled out: pose-freezing (falsified, confirmed
under biomechanically-bounded rotation), the depth clamp (ruled out), local
smoothness release (falsified). Step 6's proposed fix (replace the proximity
test with a winding-number-based loss) has since been tested as a causal
intervention -- see `diagnostics/khaoula_review/TASK6_gwn_matching_primitive_causal_test.md`:
**not a clean fix**, real wins on some specimens, real new F-score/count costs
on others (worst on the specimen the mechanism was expected to help most), no
consistent direction across a 3-specimen pilot. A part-filtered BVH/conical
loss (torch-mesh-isect) was also scoped as an alternative and found to not run
correctly on this project's current environment
(`diagnostics/khaoula_review/DELIVERABLE_penetration_fix_TASK1-5.md`'s
addendum) -- not currently a viable near-term substitute either. The
remaining untried thread from this step: a symmetric-sampled-chamfer ablation
(step 3's mechanism, independent of step 6's) -- to see if the directional
flip disappears when THAT asymmetry alone is corrected.

## Compute cost (all jobs run for this investigation)

| job | elapsed | notes |
|---|---|---|
| step 1 baseline (50 specimens) | 20:33 | |
| step 1 gentle (50 specimens) | 33:01 | |
| step 5 neighbor-drag (3 specimens x 2 arms) | 25:05 | |

All three jobs requested `--gres=gpu:1 --cpus-per-task=8`, but `sacct`'s
`AllocTRES` shows `gres/gpu=4, cpu=256` for every one of them -- the
`dc-gpu` partition appears to grant node-exclusive allocation regardless
of the smaller request. Reporting this as the raw observed fact; whether
the account is billed for the full node or a fair-share of it hasn't been
checked against this cluster's billing configuration.

## Reseeded rerun: falsification confirmed under biomechanically-bounded rotation

Both corrections folded into one rerun: `w_limit: 100.0` now active in all
four stages of both `ants_cfg_all_*.yaml` configs (was Stage_1-only
before), and 3 seeds x 2 arms (`--seed` support added to
`fitter_3d/optimise.py`, did not exist prior to this investigation).
Results: `fit3d_results_seeded/{baseline,gentle}_seed{0,1,2}/`.

One infrastructure bug hit and fixed along the way, worth recording since
it could recur: with several seeded jobs running concurrently against the
same shared conda environment, `plot_meshes()`'s `multiprocessing.Pool`
workers (each re-importing torch/numpy/matplotlib via `spawn`) raced on
the shared filesystem and hung 3 of the 6 jobs with partially-initialized-
module `ImportError`s, after training had already finished successfully
(only the post-hoc mesh-PNG rendering was affected). Fixed with an opt-in
`SMIL_DISABLE_PLOTTING=1` env var in `optimise.py` (default off, zero
effect on normal single-job runs); the 3 affected jobs were killed and
resubmitted with it set. No training results were lost or corrupted --
only wasted wall-clock on the 3 killed jobs.

**Results, all three of Fabian's aggregation methods, `penetration_num_penetrating`:**

| method | result |
|---|---|
| (A) mean-of-burdens -> % | pooled: 571.4 -> 653.5, **+14.36%**. Per-specimen mean +36.40% (median +8.22%) |
| (B) majority-of-seeds | **20/50 improved, 30/50 worse**. Of those: only 12/50 unanimous improved, 16/50 unanimous worse -- **22/50 (44%) flip direction depending on the seed** |
| (C) mean-of-per-seed-% | mean +48.13% (median +8.81%); every individual seed shows the same worsening direction (seed0 +62.15%, seed1 +39.15%, seed2 +43.08%), though the magnitude varies by ~1.6x seed to seed |

**Chamfer guardrail, per seed**: mean delta +1.65%/+1.81%/+2.17% (small,
consistent with the unseeded check's +1.32%), but 11/50, 10/50, 15/50
specimens (22-30%) exceed the |3%| bound per seed -- comparable to or
higher than the unseeded check's 17/50 (34%). The surface-fit cost
replicates across seeds, it isn't a single-run artifact.

**Conclusion**: the pooled-level falsification is unchanged and robust
(+14.36% vs the original +16.8%, same direction, similar magnitude, now
under proper joint-limit constraints). The per-specimen "coin flip" framing
from the original single-seed run needs sharpening, not reversing: this
isn't a clean 50/50 split, it's 60/40 toward worse by majority vote, AND
close to half of all specimens (44%) don't have a seed-stable answer to
"does gentle penetration loss help or hurt this specimen's count" at all --
a meaningful chunk of the original single-seed 23/27 result was seed noise,
not a real per-specimen effect. Both the depth improvement and the count/
surface-fit costs are real, reproducible, population-level effects; the
individual-specimen story is noisier than the original single-seed run
suggested.

Step 5's sample size (3 -> 8-10 specimens, with seeds) remains a separate,
independently-sized decision -- not pursued further in this pass, since
step 5's causal conclusion (releasing smoothness doesn't help, regardless
of locality) is already a clean negative result on its own terms and
doesn't depend on step 1's exact per-specimen split.
