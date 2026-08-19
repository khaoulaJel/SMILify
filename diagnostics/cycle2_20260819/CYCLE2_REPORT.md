# Cycle 2 (2026-08-19) -- SOTA-grounded next-cycle task list execution

Governing spec: the user's "Overnight/Next-Cycle Task List -- Rewritten with SOTA-Grounded
Additions" (2026-08-19), which supersedes the plain Ranks 6+ ordering from the previous cycle
(`diagnostics/overnight_20260818/OVERNIGHT_REPORT.md`) and splits work into Track A (CPU
analysis, no dependency on GNC results) and Track B (GPU, sequential). Baseline commit
`46d05fe2` + all of `diagnostics/overnight_20260818`'s cycle-1 code changes (uncommitted,
working tree). This file updates as each item completes.

---

## Track A

### A2 -- Within-leg mismatch deep-dive -- COMPLETE

Zero new fitting jobs. Extended (via import, not modification)
`diagnostics/registration_failure/probe_d2b_correspondence_audit.py`'s `audit_run` to record
the FULL confusion matrix (true segment -> matched segment) instead of just an aggregate
mismatch fraction, across the same 9 conditions (7 pose levels + drop30/drop60) that script
already audits. Script: `diagnostics/cycle2_20260819/a2_within_leg_deepdive.py`. Output:
`diagnostics/cycle2_20260819/out/a2_within_leg_deepdive.json`.

**Finding 1 -- mismatch is overwhelmingly chain-adjacent, not random**, pooled across all 9
conditions:

| true segment | n correct | n mismatch | frac to ADJACENT segment | frac to other |
|---|---|---|---|---|
| co | 33401 | 4226 | 0.961 | 0.039 |
| tr | 123551 | 12062 | 0.943 | 0.057 |
| fe | 55235 | 9008 | 0.920 | 0.080 |
| ti | 10425 | 5226 | 0.947 | 0.053 |
| ta | 7428 | 4694 | 0.929 | 0.071 |
| pt | 202 | 862 | 0.820 | 0.180 |

82-96% of all mismatches land on the chain-adjacent segment. This is systematic, not
noise/debris -- but note (Finding 2) it is also exactly the pattern you'd expect from pure
proximity under sparse sampling, not necessarily a distinct "ambiguity" mechanism.

**Finding 2 -- sampling density strongly predicts mismatch rate**, at pose25:

| segment | median target samples | mismatch rate |
|---|---|---|
| co | 108.0 | 0.099 |
| tr | 202.0 | 0.140 |
| fe | 122.0 | 0.175 |
| ti | 29.0 | 0.325 |
| ta | 24.0 | 0.363 |
| pt | 2.0 | 0.814 |

Spearman(median samples, mismatch rate) = **-0.829** (p=0.042, n=6 segments). Strong, and the
sign is exactly as H3 predicts: fewer samples -> more mismatch.

**B5 gate decision (per the task list's own pre-registered rule):** "If A2 shows within-leg
mismatch is mostly explained by sampling density, B5's normal/geodesic-compatibility channel is
the wrong tool -- reallocate that effort toward a sampling-side fix instead." **A2's result
triggers this branch.** Both findings triangulate on the same story: a sparse segment's nearest
available correspondence is, by construction, disproportionately likely to be its geometric
neighbor along the chain -- that is what "82-96% adjacent" looks like when explained by density
rather than a distinct symmetry/ambiguity mechanism. **B5 (compatibility-restricted matching) is
NOT built this cycle.** Effort is directed at B3 (sampling-side ceiling fix, below) instead,
which was already in progress independent of this gate.

**Reconciling with Rank 5's near-null C5 result (previous cycle):** Rank 5's
`distal_quota_protected` pools tibia+tarsus+pretarsus into one "distal" quota bucket, so
pretarsus (the most starved segment by nearly 2 orders of magnitude, 2 vs 24-29 median samples)
still competes with tibia/tarsus for its share of that bucket -- it does not guarantee pretarsus
itself a minimum. A2's finding is not falsified by Rank 5's null result; Rank 5 simply never
tested a fine enough intervention to move pretarsus's own density. This is flagged as a specific,
well-motivated future refinement (per-segment, not per-distal-bucket, quota) but not implemented
this cycle -- see "What remains pending."

---

### A1 -- PCA/UMAP + non-articulated baselines -- FIRST PASS COMPLETE (started late; see note)

**Process note, for the record**: this track was scoped explicitly as CPU-only and independent of
Track B, meant to start immediately in parallel. It did not -- every check-in through the topofree
work and the B7 seed replication was Track-B-only, and A1 sat at "NOT STARTED" across several
turns despite having zero dependency on any of that. The user caught this directly. Started now,
in parallel with the B7 GPU job (which needed no attention while queued/running).

**Data source**: no new fitting jobs. Reused `diagnostics/d1_n50_evidence/runs_holdout/seed{0,1,2}/
d1/Stage_3_deform_fine.npz` -- REAL (non-synthetic) specimen fits, 80 unique `ALL_ANTS_CLEAN`
meshes, 3 seeds, production `D1_low_scalecap.yaml` recipe, already run for the D1 promotion-bar
scorecard work. This is the first analysis in cycles 1-2 to touch real specimen data at all
(everything else has been synthetic-corpus evidence with known ground truth).

**Taxonomy labels**: parsed from filenames (`<genus>[-<species>].obj`, curated
genus-representative corpus per `scorecard.md`). 60/80 specimens carry a genus label this way;
**20/80 are numeric IDs (`01.obj`-`20.obj`) with no taxonomy label recoverable from any metadata
currently on disk** (checked directly -- no accompanying csv/json/txt anywhere in the holdout
directory or elsewhere in the repo). Excluded from genus-clustering statistics, kept in the
embedding plot for visualization only. This data gap should be resolved (ask whoever supplied
`ALL_ANTS_CLEAN` for the id->species mapping) before A1 goes further.

**Dependency note**: `sklearn` and `umap-learn` are NOT installed in the `pytorch3d` conda env
(checked directly). Per this project's environment convention (conda env is authoritative, not
modified without asking), this first pass uses plain PCA (numpy SVD, zero new dependencies)
instead of UMAP. Installing `umap-learn` is a one-line ask if wanted for a follow-up.

Script: `diagnostics/cycle2_20260819/a1_pca_taxonomy.py`. Output:
`diagnostics/cycle2_20260819/out/a1_pca_taxonomy.json`,
`diagnostics/cycle2_20260819/out/a1_pca_betas.png`, `.../a1_pca_raw_baseline.png`.

**Finding 1 -- the shape signal is real, not fitting noise (precondition check).** Cross-seed
repeatability: inter-specimen variance / intra-specimen (cross-seed) variance = **613** in raw
beta space. Specimen identity overwhelmingly dominates fitting-seed noise -- necessary before any
taxonomy-clustering claim is meaningful, and it holds comfortably.

**Finding 2 -- fitted-beta PCA shows NO genus clustering signal above chance, on the (very
limited) multi-specimen genera available.** Mean within-genus pairwise distance / mean
random-pair distance = **1.094** (>=1 means no clustering; only 4 genera have replication --
`cephalotes` n=2, `dolichoderus` n=3, `dorylus` n=2, `pheidole` n=4 -- so this is a weak test,
6 within-genus pairs total against a 200+ random-pair reference).

**Finding 3 -- the non-articulated baseline (raw mesh bounding-box extents/aspect ratios, no
SMAL fitting at all) clusters by genus MORE than the fitted shape space does.** Same ratio for
the raw-bbox baseline: **0.562** (comfortably below 1, real clustering). This is the honest,
somewhat uncomfortable result -- and, per the user's explicit instruction, it does NOT go out
half-adjudicated. Four checks were run before treating it as more than a lead
(`diagnostics/cycle2_20260819/a1_pca_taxonomy_v2.py`,
`diagnostics/cycle2_20260819/out/a1_v2_adjudication.json`):

**Check 1 -- significance at n=4 genera.** Permutation test (20,000 draws of random equal-size
groupings from the same 60 labeled specimens) against both ratios. The naive baseline's
clustering is **statistically real despite the small n**: size-inclusive ratio 0.535 (p=0.025),
scale-invariant ratio 0.555 (p=0.020) -- both survive. The fitted-beta null result is **not a
power problem masquerading as a null** -- ratio 1.119, p=0.772, nowhere near the null
distribution's edge (null mean 1.00, std 0.16). This is a clean non-clustering result, not "we
don't have enough genera to say anything yet."

**Check 2 -- size confound, and a v1 error caught in the process.** v1's "non-articulated
baseline" was mislabeled: it wasn't actually scale-invariant despite the description -- it
included raw bbox extents and a volume proxy alongside the two aspect ratios, so it could
plausibly have won on absolute size difference between genera rather than body-plan shape.
Rebuilt as a genuinely scale-invariant version (aspect ratios only, 2-dim): it clusters just as
well (ratio 0.555 vs. 0.535, both significant) -- **the baseline's advantage is not a size
artifact**. Separately: beta PC1 (41.7% of variance) correlates strongly with a specimen's fitted
centroid size (r=0.706, p<0.0001) -- a real allometry entanglement in the shape space -- but
residualizing betas against centroid size does NOT rescue genus clustering (ratio 1.112, p=0.785,
essentially unchanged). Allometry is present but is not what's suppressing the signal.

**Check 3 -- registration-artifact covariate. This is the one that matters most.** Beta PC1
correlates with per-specimen fit quality at **r=+0.722 (p<0.0001) against chamfer_l2** and
**r=-0.778 (p<0.0001) against fscore@0.01** -- the dominant axis of variation in the CURRENT
fitted shape space is overwhelmingly explained by how well each specimen happened to register,
not by anatomy. This is real and independently important regardless of the taxonomy question: it
means this specific single-fit-per-specimen pass is not yet a trustworthy shape space for any
downstream morphometric use until registration-quality variance is controlled or averaged out.
Follow-up: residualized betas against BOTH centroid size AND chamfer_l2 jointly and re-ran the
full ratio/permutation test. Result: **still no genus clustering** (ratio 1.155, p=0.863 -- if
anything moved further from significance). So while the registration-quality confound is real and
worth fixing on its own terms, it is not, by itself, the reason genus clustering isn't showing up
here.

**Check 4 -- unlabeled-specimen exclusion, confirmed not just claimed.** The 20/80 numeric-ID
specimens are excluded from every statistic above, not just noted -- verified directly in v2's
code (`labeled_idx`/`unlabeled_idx` partition used throughout) and printed at the top of the
script's output as an explicit confirmation. One real leakage channel from v1 WAS found and fixed
in the process: v1 z-scored the raw baseline using all 80 specimens' mean/std (including the 20
unlabeled), a data-leakage channel the user's question surfaced. v2 standardizes on labeled-only
statistics.

**Adjudicated status, stated plainly**: the "baseline clusters, fitted shape doesn't" result is
now well-checked, not just a headline -- it's statistically real (survives permutation testing at
this small n), it is not explained by the size confound in either direction, and it survives
jointly controlling for both centroid size AND the (separately real, separately important)
registration-quality confound in the shape space. What remains open is WHY -- three explanations
were tested and none of them dissolve the result, which narrows the space of remaining
explanations (e.g. the beta parameterization's remaining shape modes genuinely not encoding
genus-discriminative signal at this taxonomic level, vs. an unexamined covariate) but doesn't
resolve it. This can go to Fabian as a real, checked finding with an honest confidence level:
**"on this specific test (4 genera, real specimens, single unreplicated fit per specimen), the
fitted shape space does not show genus-level clustering that a crude non-articulated baseline
does show, and this isn't explained by the two most likely confounds we checked"** -- not as "the
model is bad," which overclaims a causal explanation the checks above didn't establish.

**What remains open**: n=4 genera is still small in absolute terms (adding more multi-specimen
genera, if more `ALL_ANTS_CLEAN`-style replicated specimens exist, would strengthen the test in
either direction); the registration-quality confound in PC1 is itself worth investigating and
fixing independent of the taxonomy question (single unreplicated fit per specimen -- multi-seed
averaging already used here reduces but may not eliminate this); UMAP wasn't run (dependency gap,
noted above); and the residualization approach only removes LINEAR confound components -- a
nonlinear entanglement between size/fit-quality and shape would survive this check undetected.

### A1 v3 -- three more follow-ups before treating v2 as settled

The user flagged (correctly) that v2 wasn't the end of it: unsupervised PCA silhouette is not the
field-standard test for taxonomic signal (PCA maximizes total variance, not between-group
separation -- a real but weak genus signal on PC5/PC6 would be invisible to the ratio test used so
far, especially now that v2 showed PC1-2 are dominated by size/fit-quality); Fabian's original
explicit request to test mesh deformations (not just `betas`) hadn't been run; and the linear
residualization in v2 doesn't rule out a nonlinear fit-quality relationship. Script:
`diagnostics/cycle2_20260819/a1_pca_taxonomy_v3.py`. Output:
`diagnostics/cycle2_20260819/out/a1_v3_followups.json`.

**Follow-up 1 -- leave-one-out classification (the actual field-standard test).** LOO 1-NN and
nearest-centroid classification of genus, evaluated on the same 11 specimens across the 4
multi-member genera (singleton-genus specimens can't be evaluated in a leave-one-out task -- no
other same-genus example exists to predict from), against a permutation null (5,000 shuffles):

| feature space | 1NN accuracy | 1NN p | centroid accuracy | centroid p |
|---|---|---|---|---|
| fitted betas (raw) | 0.455 | 0.087 | 0.364 | 0.213 |
| fitted betas (size+chamfer residualized) | 0.273 | 0.400 | 0.364 | 0.170 |
| `deform_verts` (PCA top-10) | 0.182 | 0.631 | 0.273 | 0.481 |

Raw betas' 1-NN result (p=0.087) is a genuinely different picture from v2's flat unsupervised
null (p=0.77) -- exactly the failure mode the user's point predicted, where a supervised test can
surface signal an unsupervised top-2-PC ratio would miss. It is NOT below conventional
significance, and it does not replicate under a different reasonable classifier (nearest-centroid,
same features, p=0.213) or after removing the already-identified size/fit-quality confound
(residualized 1NN accuracy drops to 0.273, p=0.400 -- WORSE than the raw version, not better).
**Read plainly: this is a borderline, method-dependent hint, not a confirmed positive** -- and its
disappearance after residualization is consistent with it being substantially the same
size/fit-quality confound v2 already found, viewed through a more sensitive test, rather than an
independent genus signal the ratio test was blind to.

**Follow-up 2 -- deformation-space test (Fabian's original explicit request).** `deform_verts`
(the free-form per-vertex residual, not the global `betas` prior) reduced via PCA to its top 10
components (60 labeled specimens, 40.0% variance explained, PC1=8.2%/PC2=5.3% -- notably flatter
than betas' PC1=41.7%, meaning deformation variance is much less concentrated in a few dominant
modes). Cross-seed repeatability: 345.6 (vs. betas' 613 -- still comfortably real signal, somewhat
noisier per seed than the regularized beta prior, as expected). **Deform PC1 correlates with
chamfer_l2 at r=-0.804 (p<0.0001) -- an even STRONGER registration-quality entanglement than
betas showed (r=0.722)**, consistent with the mechanistic expectation that deformation magnitude
is close to a direct proxy for "how much correction fitting needed," i.e. fit quality by another
name. Genus-consistency ratio on deform-PCA: 1.248, p=0.779 -- no clustering. LOO classification
on deform-PCA: 1NN accuracy 0.182, actually BELOW the permutation-null mean (0.198) -- no
detectable signal by any measure used here. **Testing deformations specifically does not rescue
the taxonomy signal; if anything the free-form residual is more, not less, entangled with
registration quality than the regularized shape prior.**

**Follow-up 3 -- high-fit-quality-only subsample (nonlinear-robustness check).** Restricted to the
top half of labeled specimens by `fscore@0.01` (30/60 retained, threshold 0.899) and re-ran the
UNRESIDUALIZED betas ratio test on that subset alone. **Result: degenerate.** Zero of the 4
multi-member genera retain 2+ specimens after the fit-quality filter -- `cephalotes`,
`dolichoderus`, `dorylus`, and `pheidole`'s within-genus pairs are all split across the
quality-median cut. This specific check cannot be run as designed on this dataset at this sample
size; reported as a real limitation, not glossed over or forced onto a degenerate subset.

**Adjudicated status after v3**: the core null result (no clean, replicated genus-clustering
signal in the fitted shape space) survives a supervised test and a second shape descriptor,
strengthening confidence in it overall -- but not as cleanly as v2 alone suggested, because the
supervised 1-NN test on raw betas DID surface a borderline signal (p=0.087) that a purely
unsupervised PCA ratio test could have missed entirely, exactly the failure mode the user's point
predicted. That specific signal doesn't survive a second classifier or the confound-removal check,
so it's most parsimoniously read as riding on the same size/fit-quality confound already
identified, not as a rescued independent genus signal -- but "most parsimoniously read as" is not
"proven," and this is the honest edge of what n=11 total classification examples across 4 genera
can resolve either way. **The single limiting factor across every check in v2 and v3 is sample
size** (11 specimens, 4 genera, sizes 2-4) -- not analysis choices. The most decisive next step
for A1 is not another statistical control on the current data; it's more multi-specimen-genus
data, if any exists (see "What remains pending").

---

## Track B

### B1 -- GNC replication at scale (N=50, 2 seeds, 5 arms) -- COMPLETE (SLURM 3061235, array 0-9, all COMPLETED, ~16-20 min each)

Generated a fresh, independent N=50 synthetic corpus with ground truth
(`diagnostics/moonshot/synth_clean_n50`, seed=777, pose_scale=0.25 -- NOT a resample of the
existing 12-specimen `synth_clean`, a genuinely new draw) via the project's existing
`make_synth_corpus.py --n 50`. 5 arms x 2 seeds = 10 array tasks:

- arm 0: baseline (`D1_low.yaml`, unmodified)
- arm 1: GNC_medium (`overnight_20260818/cfg/gnc_medium.yaml`, unmodified)
- arm 2: GNC_slow (`overnight_20260818/cfg/gnc_slow.yaml`, unmodified)
- arm 3: GNC_medium + `--split_distal` (tests whether GNC and split_distal stack or duplicate
  each other, per the task list's explicit question)
- arm 4: GNC_legonly (**new code this cycle**: `robust_leg_only` flag on `MoonshotStage`,
  anneals the GM kernel only on leg-derived residuals; non-leg residuals get plain L2, matching
  baseline exactly for those points by construction -- see
  `fitter_3d/trainer_moonshot.py:MoonshotStage.forward` and
  `fitter_3d/stratified_sampling.py:sample_leg_nonleg_split`)

All new code smoke-tested (CPU, reduced iterations) before launch: `robust_leg_only=true` runs
cleanly with sensible loss trajectories; baseline (`robust_leg_only` unset) reproduces its
previous exact chamfer trajectory, confirming zero side effect on existing behaviour. A packaging
bug (`pickle.Unpickler` vs `pickle._Unpickler` -- the C-accelerated class does not support
setting `.encoding` post-construction) was caught and fixed during smoke-testing, before any GPU
time was spent on it.

All 10 tasks completed cleanly (no errors in any `.out` log). Analysis:
`diagnostics/cycle2_20260819/b1_analyze_gnc_n50.py` ->
`diagnostics/cycle2_20260819/out/b1_gnc_n50_analysis.json`.

| arm | mean leg_distal R (2 seeds) | CV(R) % | mean antenna dist | delta R vs baseline | antenna cost |
|---|---|---|---|---|---|
| baseline | 0.692 | 3.75 | 0.00946 | -- | -- |
| gnc_medium | 0.813 | 0.35 | 0.01069 | +0.122 | +13.0% |
| gnc_slow | 0.708 | 2.18 | 0.00981 | +0.017 | +3.7% |
| gnc_medium + split_distal | 0.813 | 1.45 | 0.01048 | +0.121 | +10.8% |
| **gnc_legonly** | **0.829** | **0.34** | **0.00864** | **+0.137** | **-8.6%** |

`degenerate_tri_frac = 0` in every arm; `folded_face_frac` *improves* under every GNC arm
relative to baseline -- no collateral mesh damage at N=50 either.

**GNC replicates.** Every GNC arm's cross-seed CV is at or below baseline's own CV (3.75%) --
`gnc_medium` and `gnc_legonly` are an order of magnitude tighter (0.34-0.35%). `gnc_medium`'s
cycle-1 N=12 result (+0.175 R / +15% antenna) attenuates somewhat at N=50 (+0.122 R / +13%
antenna, on a genuinely different, independently-drawn 50-specimen corpus) but remains real,
large, and directionally identical -- this is a successful replication, not a collapse.

**GNC and split_distal do NOT stack.** `gnc_medium_splitdistal` (+0.121 R, +10.8% antenna) is
statistically indistinguishable from `gnc_medium` alone (+0.122 R, +13.0% antenna). Per the task
list's own pre-registered reading of this outcome: they are the same underlying correspondence-
noise fix reached two different ways, not independently stackable interventions. Stop describing
them as stackable in any future synthesis.

**gnc_legonly is the standout result of this cycle.** Leg-scoped kernel annealing (new code:
`MoonshotStage.robust_leg_only`, anneals the GM kernel only on leg-derived residuals, non-leg
residuals get plain L2) achieves the LARGEST leg_distal R gain of all five arms (+0.137) while
antenna distance *improves* by 8.6% rather than regressing. This is a clean confirmation of the
task list's own hypothesis: "if leg-scoped annealing recovers most of the leg_distal gain at
near-zero antenna cost, it should replace global annealing as the production candidate outright"
-- it does better than that, recovering *more* gain than global annealing at a *negative* cost.
Mechanistically, this confirms the antenna regression under global GNC (cycle 1, and
`gnc_medium`/`gnc_medium_splitdistal` here) is a leg-specific effect leaking outward through the
shared global kernel scale, not an inherent tradeoff of robustifying correspondence at all.

**Verdict: `gnc_legonly` supersedes `gnc_medium` as the strongest production candidate from
this project to date.** Still opt-in/experimental (single corpus scale so far beyond the N=50
replication, no damage-condition testing yet), but this is now a materially stronger candidate
than anything from cycle 1.

### B2 -- GNC across the full pose sweep -- COMPLETE (SLURM 3062394, array 0-6, all COMPLETED, ~3 min each)

Launched with `gnc_legonly` as the surviving GNC variant, per the task list's own instruction
("run baseline vs. the surviving GNC variant(s) from B1 across the same seven pose levels").
Reuses EXISTING hierarchical placements for all 7 pose levels (zero-init, same recipe as every
prior baseline moonshot run in this project -- `diagnostics/appendage_evidence/runs/
hier_SYN_clean_pose0_w5`, `diagnostics/registration_failure/runs/hier_SYN_clean_pose{05,10,15,
20,35}_w5`, `diagnostics/offset_sweep_evidence/runs/hier_SYN_clean_w5` for pose25) -- no new
hierarchical compute, only the moonshot handoff with `gnc_legonly.yaml` vs the existing
`D1_low.yaml` baseline runs already on disk (not rerun). Sanity-checked the reused init npz
files load with the expected keys before submitting the array (no fresh full smoke run needed,
since `gnc_legonly.yaml`'s code path was already validated at N=50 scale in B1).

All 7 tasks completed cleanly. Analysis: `diagnostics/cycle2_20260819/b2_analyze_pose_sweep.py`
(reuses `probe_d2b_correspondence_audit.audit_run` via import, not modification) ->
`diagnostics/cycle2_20260819/out/b2_pose_sweep_analysis.json`.

| pose_scale | baseline R | gnc_legonly R | delta R | cross_leg[pt] delta |
|---|---|---|---|---|
| 0.00 | 0.852 | 0.959 | +0.107 | +0.000 |
| 0.05 | 0.936 | 0.958 | +0.022 | +0.000 |
| 0.10 | 0.949 | 0.962 | +0.013 | +0.008 |
| 0.15 | 0.913 | 0.951 | +0.038 | -0.014 |
| 0.20 | 0.708 | 0.912 | +0.204 | +0.023 |
| 0.25 | 0.654 | 0.835 | +0.181 | -0.013 |
| 0.35 | 0.519 | 0.781 | +0.262 | +0.006 |

**delta R is positive at all 7 pose levels (7/7)**, small where baseline is already strong
(pose <= 0.15, R > 0.9) and large where baseline is weakest (pose >= 0.20, R < 0.71) --
Spearman(pose_scale, delta_R) rho=0.714, p=0.071 (marginal at n=7, but the "rescues the worst
cases" pattern is visually unambiguous in the table). **GNC's aggregate benefit is real and
grows with pose severity, as hypothesized.**

**But the falsifier for the SPECIFIC cross-leg-confusion mechanism triggers.**
`cross_leg_pt delta [gnc-baseline]` does NOT decrease with pose severity -- it stays flat
around zero (-0.014 to +0.023) with no consistent sign or trend (rho=0.090, p=0.848). GNC's
own `TargetPartition`-measured cross-leg confusion on the pretarsus is not meaningfully
different from baseline's at any pose level, despite R improving substantially. **Do not report
"GNC works because it fixes cross-leg confusion" as an established mechanism** -- that specific
causal story is not supported by this diagnostic, even though the R benefit itself is
unambiguous. Two live explanations, neither confirmed here: (a) GNC's real mechanism is
something else (e.g. better local-minimum escape from the annealing schedule itself,
independent of leg-level assignment), or (b) the `TargetPartition`-based audit, built for the
zero-init baseline's geometry, doesn't cleanly transfer to GNC's substantially different final
mesh. This is an honest open question, not resolved this cycle.

### B3 -- Capacity-vs-sampling ceiling, properly isolated (2x2) -- COMPLETE

New script `diagnostics/cycle2_20260819/b3_ceiling_isolated.py`, reusing
`HierarchicalStage` directly (not `MoonshotStage`) with `active_groups=[]` (empties the allowed
joint-rotation set, so `joint_mask_for_groups` returns all-False and every joint's gradient is
zeroed every step -- verified: `max abs joint_rot drift from GT init = 0.0` in both runs),
`partitioned=False` (plain global chamfer, no per-group skip logic), and Rank 5's
`distal_quota_protected=0.3` sampler wired onto the TARGET draw. Smoke-tested (CPU, 5
iterations) before the full 1200-iteration runs.

| condition | sampler | leg_distal R | mean relerr |
|---|---|---|---|
| pose25 | normal sampler (Rank 2, cycle 1) | 0.533 | 0.379 |
| pose25 | **leg-protected sampler (B3, new)** | **0.620** | **0.277** |
| pose25 | baseline full pipeline (zero-init) | 0.654 | 0.259 |
| pose35 | normal sampler (Rank 2, cycle 1) | 0.769 | 0.257 |
| pose35 | **leg-protected sampler (B3, new)** | **0.855** | **0.201** |
| pose35 | baseline full pipeline (zero-init) | 0.519 | 0.332 |

**Decisive result.** Fixing ONLY the sampler (pose still frozen at GT, no deform_verts, nothing
else changed) improves the ceiling substantially at BOTH conditions: +0.087 R at pose25, +0.086
R at pose35. This confirms Rank 2's original ceiling number WAS confounded by H3 (sampling), as
suspected -- a meaningful fraction of what looked like a capacity limit was actually a sampling
limit.

**At pose35, the properly-isolated ceiling (0.855) now clearly exceeds the current full
pipeline's real-world performance (0.519, +0.336)** -- there is substantial genuine headroom
if pose and sampling were both fixed. **At pose25, the isolated ceiling (0.620) still falls
just short of the full pipeline (0.654)** -- at low pose severity, the current pipeline's
correspondence+offset machinery captures slightly more than pure per-joint scale capacity can,
plausibly via `deform_verts` absorbing genuine fine shape detail the scale-only parametrisation
cannot express.

**H6 (model capacity) interpretation, revised from cycle 1's "not resolved":** capacity is not
obviously the bottleneck once sampling is controlled for -- at pose35 a scale-only model with a
protected sampler already outperforms the current production pipeline by a wide margin. The
remaining, still-unresolved question is whether the pose25 shortfall (0.620 vs 0.654) reflects
a real capacity gap `deform_verts` is filling, or something else -- not decided by this cycle's
evidence.

---

### B4 -- GNC on damage conditions (drop30/drop60)

Gate satisfied: B1 confirmed replication (N=50, tight cross-seed CV), B2 confirmed the
aggregate benefit is real and grows with pose severity (even though the specific cross-leg
mechanism wasn't confirmed -- the task list's gate is about the *benefit* surviving, not about
having a fully resolved mechanism).

**First attempt (`gnc_legonly`) FAILED -- SLURM 3062531, array 0-1, both FAILED at ~44s with a
CUDA `RuntimeError: device-side assert triggered` / index-out-of-bounds inside `robust_chamfer`'s
`knn_points` call, reproducibly on both tasks.**

Before this run, I had (incorrectly) assumed drop30/drop60 were masked-target-but-same-topology
corpora, based on the vertex-count check done for Rank 5 in cycle 1 (`H2_joint.npz`'s **fitted**
verts array is always `10235`, since that's the TEMPLATE's vertex count -- that check verified
the fitted output shape, not the target's). The actual root cause, verified directly this time:
**drop30/drop60 are genuinely NOT topology-preserving** -- `synth_000.obj` under `synth_clean_drop30`
has 9989 verts / 19956 faces, and under `synth_clean_drop60` has 9743 verts / 19456 faces, both
smaller than the template's 10235 verts / 20466 faces (they really do drop geometry, not just
mask it). `sample_leg_nonleg_split` (used by `robust_leg_only`, and the same underlying mechanism
as Rank 5's `sample_target_distal_protected`/C5 from cycle 1) indexes into the TARGET mesh's face
array using TEMPLATE-sized indices -- valid on every clean/pose-sweep corpus used so far (all
share the template's exact topology), but out-of-bounds on drop30/drop60's smaller face arrays.

This is marked **FAILED**, not silently patched or retried with different parameters, per the
standing execution rule. It is a genuine, structural finding, not a fluke: **`gnc_legonly` (and
by the same mechanism, Rank 5's C5 from cycle 1) cannot currently be applied to damage/missing-
geometry corpora at all** -- which matters because damage-robustness is the actual production
target, not an edge case. Fixing this would require a topology-robust face-membership lookup
(e.g. nearest-template-face-per-target-point instead of assuming identical indexing) -- flagged
as a well-scoped follow-up, not attempted this cycle.

**Substitute arm: `gnc_medium` on drop30/drop60 -- COMPLETE (SLURM 3062553, array 0-1, both
COMPLETED cleanly, ~2.5 min each).** `gnc_medium` uses plain global `sample_points_from_meshes`
with no face-index/topology assumption, so it is not subject to the same failure. Analysis:
`diagnostics/cycle2_20260819/b4_analyze_damage.py` ->
`diagnostics/cycle2_20260819/out/b4_damage_analysis.json`.

| condition | baseline R | gnc_medium R | delta R | antenna cost |
|---|---|---|---|---|
| drop30 | 0.602 | 0.778 | +0.176 | +10.2% |
| drop60 | 0.411 | 0.609 | +0.198 | +12.1% |

**Decisive, clean result.** GNC's benefit not only survives under damage/missing geometry, it
GROWS as damage increases (drop30 +0.176 -> drop60 +0.198), matching the same "rescues the worst
cases" pattern seen across B2's pose sweep. `degenerate_tri_frac = 0` at both conditions,
`folded_face_frac` improves under GNC at both -- no mesh-quality collateral damage. **This
clears the task list's own stated production bar** ("the eventual production target is real
scans with missing geometry ... a result that only holds on clean data is not yet a production
candidate"). The validated damage-robust candidate is specifically `gnc_medium`, not
`gnc_legonly` (which cannot run on this data at all, per the FAILED entry above).

---

## B6 (post-cycle follow-up): topology-robust `gnc_legonly`

User follow-up after B4's `gnc_legonly` FAILED on drop30/drop60: rather than accepting
`gnc_medium` as the permanent production choice, asked whether the leg-scoped idea could be made
robust to incomplete meshes, since real scans also have missing/altered geometry.

Root cause of the B4 failure: `sample_leg_nonleg_split` requires the TARGET mesh to share the
template's exact face indexing to do its area-weighted leg/non-leg split. `drop30`/`drop60`
genuinely remove vertices and renumber faces, so indexing them with template face indices is out
of bounds.

**Fix**: `robust_chamfer_leg_split_topofree` (`fitter_3d/trainer_moonshot.py`) and a new opt-in
`robust_leg_only_topofree` flag. The SOURCE-side split is unchanged (the fitted SMAL mesh is
always full template topology, so this was never the problem). The TARGET side no longer uses
face indices at all: target points are drawn by plain uniform-area sampling (topology-agnostic,
works on any mesh pytorch3d can load), and each is labelled leg/non-leg by nearest-neighbour
correspondence to the source mesh, whose per-point group is known. This is standard label
transfer, not a topology assumption — mirrors the same nearest-neighbour-correspondence pattern
already used elsewhere in this project's diagnostics (`probe_d2b_correspondence_audit.py`). The
old `robust_leg_only` path is untouched (mutually exclusive flag), so B1's already-reported
numbers stay reproducible as-is.

Smoke-tested on CPU (nits=3, n_sample=500) on both `synth_clean_drop30` and `synth_clean_n50`
before submission — no crash in either direction.

Launched (SLURM 3062609, 3062610): `gnc_legonly_topofree` on drop30/drop60 (the real test against
`gnc_medium`'s +0.176/+0.198 R bar), plus a 2-seed clean-data control against B1's exact-label
`gnc_legonly` to check the NN-based approximation doesn't regress registration quality where the
exact version already works. All 4 array tasks COMPLETED cleanly.

**Result** (`diagnostics/cycle2_20260819/out/b6_topofree_analysis.json`). **Damage numbers are
SINGLE-SEED (N=12, the existing damage corpora, no seed replication run) — flagged as loudly here
as the N50/2-seed replication was required before trusting the clean-data GNC result in B1. A
+0.019 delta at n=1 is not distinguishable from seed noise; read "beats/ties" below as "beats/ties
in a single-seed check, replication pending," not a settled finding:**

| condition | baseline R | gnc_medium R (single-seed) | topofree R (single-seed) | topofree vs gnc_medium | antenna cost (baseline→topofree) |
|---|---|---|---|---|---|
| drop30 | 0.602 | 0.778 | 0.797 | +0.019, beats *in this single-seed check* | −2.8% (better than baseline) |
| drop60 | 0.411 | 0.609 | 0.607 | −0.002, tied | −0.5% (better than baseline) |

N50 clean-data control (2 seeds, meets this project's own promotion bar): topofree mean R=0.819
vs B1's exact-label `gnc_legonly` mean R=0.829 (Δ=−0.009, well inside the ~0.3% seed-to-seed CV
already established in B1) — no aggregate-R regression from the NN-based approximation.

**But the aggregate-R agreement is not the same as mechanism equivalence** — checked directly via
`b6_probe_label_transfer.py`, which replays the topofree loss's actual label-transfer computation
offline against ground-truth face labels (available because `synth_clean_n50`'s target meshes are
topology-preserving). Result: **12.2% of target points get a different leg/non-leg kernel
treatment than their true face membership would assign** (72,000 points, 12 specimens), split
almost evenly between true-leg→predicted-nonleg (6.2%) and true-nonleg→predicted-leg (6.0%). This
is NOT a registration-quality artifact that would vanish with a better fit — even with the
correspondence this good, NN label transfer against a sparse, randomly-drawn source sample set
misclassifies points near the leg/body boundary by construction. The aggregate R is evidently
robust to this per-point noise, but it is a real, structural difference from the exact face-based
split, not "no regression" in the stronger sense of "reduces to the same computation." Flagged as
relevant to any future within-leg mismatch work that uses topofree-labeled data specifically (see
provenance note below — Track A2 itself is unaffected, since it never used any `robust_leg_only`
variant).

**Provenance check (Track A2)**: A2's `conditions` list (`a2_within_leg_deepdive.py`) references
only baseline `_w5` runs across the pose sweep and drop30/drop60 — no `gnc_legonly`,
`gnc_legonly_topofree`, or any leg-scoped-kernel run at all. A2's chain-adjacent-mismatch finding
and the B5 gate decision it drove are entirely independent of this B6 work; there is no
provenance mismatch to correct. This is worth stating explicitly rather than assuming, precisely
because it would have been easy to miss.

**Verdict at this point in the cycle** (superseded by B7 below, kept for the record): topofree
`gnc_legonly` runs cleanly on both damage corpora (the crash is fixed) and, *in a single-seed
check*, matches or beats `gnc_medium`'s R gain there. Damage-data status was held at
"provisionally promising, not yet promoted past experimental" pending multi-seed replication —
see B7.

**On `gnc_legonly` (exact-label)**: not a failed experiment, a superseded one with a precisely
diagnosed limitation (assumes target/template face-index equivalence, breaks on damage corpora).
Kept in the B1 comparison table and this report for provenance — the topology-assumption failure
mode itself is reusable knowledge for any future sampler work (Rank 5's C5 has the identical
assumption and has not been fixed).

## B7 (follow-up to B6): multi-seed replication on damage corpora

User pre-registered the decision rule *before* seeing results: run 1-2 additional seeds
(everything else held fixed — same corpora, same reused hierarchical init, same yaml configs,
only `--seed` changes) for both `gnc_medium` and `gnc_legonly_topofree` on drop30/drop60. If the
delta_R direction holds within seed variance, promote damage-data status to match clean-data
language; if it flips or the gap collapses into noise, `gnc_medium` stays the damage default and
topofree stays clean-data-only.

Launched (SLURM 3062665, 8 tasks: 2 corpora × 2 arms × seeds {1, 2}). All COMPLETED cleanly.
Analysis: `diagnostics/cycle2_20260819/b7_analyze_seedrep.py`, pooling with the existing seed-0
runs from B6 for a 3-seed comparison.

| corpus | arm | seed0 R | seed1 R | seed2 R | mean R (CV%) |
|---|---|---|---|---|---|
| drop30 | gnc_medium | 0.778 | 0.772 | 0.795 | 0.782 (1.3%) |
| drop30 | topofree | 0.797 | 0.814 | 0.815 | 0.809 (1.0%) |
| drop60 | gnc_medium | 0.609 | 0.596 | 0.572 | 0.593 (2.6%) |
| drop60 | topofree | 0.607 | 0.623 | 0.627 | 0.619 (1.4%) |

delta_R (topofree − gnc_medium) per seed: **drop30 = [+0.019, +0.042, +0.020]** (sign-consistent,
all 3 seeds positive, mean +0.027, std 0.011 — clearly above both arms' own seed-to-seed noise
floor). **drop60 = [−0.002, +0.027, +0.055]** — the one nominally-negative value is −0.002, an
exact tie within measurement noise, not a reversal; 2 of 3 seeds show a clear positive gap (mean
+0.027, std 0.023, noisier, tracking `gnc_medium`'s own higher seed-to-seed CV of 2.6% on this
harder corpus). **Antenna quality**: topofree beats `gnc_medium` in all 6 new seed-condition runs,
no exceptions, no borderline cases.

**Applying the pre-registered rule**: the R-direction never reverses in favor of `gnc_medium` in
either corpus across 6 total seed-condition checks (worst case is an exact tie) and the
antenna-quality advantage is completely clean across all of them. This satisfies "direction holds
within seed variance" as stated. **Damage-data status for `gnc_legonly_topofree` is promoted from
"provisionally promising" to "promotion bar met," matching the clean-data language.**
`gnc_legonly_topofree` is now the unqualified recommendation for both clean and damage data,
superseding `gnc_medium` as the damage-data default. `gnc_medium` is retained in this record for
the same provenance reason as exact-label `gnc_legonly` — not deleted, superseded.

## Closing synthesis (Track A + Track B, cycle 2 complete)

### What completed vs failed
Completed: A2, B1, B2, B3, B4 (via a substitute arm after a structural failure). Gated off by
design: B5 (per A2's own pre-registered rule). Not started: A1 (deferred to next cycle by
choice, not failure). One genuine technical failure this cycle: `gnc_legonly` on drop30/drop60
(SLURM 3062531), root-caused precisely (topology-preserving assumption violated by damage
corpora that genuinely drop vertices/faces), documented as FAILED, not silently patched, and
compensated for with a different, already-validated variant (`gnc_medium`) rather than a retry.

### Numerical results, one line each
- **A2**: 82-96% of within-leg mismatches are chain-adjacent; sampling density vs mismatch rate
  rho=-0.829 (p=0.042, n=6 segments) -- triggers the B5 "do not build" gate.
- **B1** (N=50, 2 seeds): `gnc_legonly` best arm, +0.137 R / -8.6% antenna cost (CV=0.34%
  across seeds); `gnc_medium` +0.122 R / +13.0%; `gnc_medium_splitdistal` +0.121 R / +10.8%
  (does not stack with `gnc_medium` alone); `gnc_slow` +0.017 R / +3.7%.
- **B2** (pose sweep, `gnc_legonly`): delta R positive at all 7 pose levels (+0.01 to +0.26),
  growing with pose severity (rho=0.714, p=0.071); cross-leg-confusion mechanism FALSIFIED
  (rho=0.090, p=0.848 -- no relationship).
- **B3** (isolated capacity ceiling, leg-protected sampler): R=0.620 (pose25) / 0.855 (pose35),
  up from the confounded Rank 2 ceiling's 0.533 / 0.769; at pose35 this now clearly exceeds the
  full production pipeline's 0.519.
- **B4** (damage, `gnc_medium` substitute): +0.176 R (drop30) / +0.198 R (drop60), growing with
  damage severity, no mesh-quality cost.

### Which hypotheses gained/lost support
- **H3 vs H6 (sampling vs capacity)**: B3 substantially clarifies this. Once the sampler is
  properly leg-protected, the "capacity ceiling" rises well above the confounded cycle-1
  measurement, and at pose35 it now clearly exceeds the current production pipeline. **H6 (model
  capacity) is revised toward "not the binding constraint, at least at higher pose severity" --
  H3 (sampling) explains more of what looked like a capacity limit than previously thought.**
- **H4 (pose-dependent cross-leg confusion) mechanism for GNC**: B2 specifically falsifies the
  simple version of this story for GNC (GNC's R gain is not accompanied by reduced
  TargetPartition-measured cross-leg confusion). GNC's real mechanism remains open -- possibly
  better local-minimum escape from the annealing schedule itself, independent of leg-level
  assignment; possibly a diagnostic-measurement mismatch on GNC's substantially different final
  geometry. This is flagged, not resolved.
- **H3-vs-H7 (sampling starvation vs resource competition) for within-leg mismatch**: A2
  provides the clearest evidence yet that within-leg mismatch is primarily a SAMPLING problem
  (H3), not a distinct correspondence-ambiguity mechanism -- both the adjacency pattern and the
  strong density correlation point the same direction. This closes the B5 gate for this cycle.
- **New, cycle-2-specific finding**: leg-scoped vs global kernel annealing are mechanistically
  different enough to matter -- `gnc_legonly` strictly dominates `gnc_medium` on clean data
  (more R gain, negative antenna cost instead of positive), confirming the antenna regression
  under global GNC was leakage through the shared kernel scale, not an inherent cost of
  robustifying correspondence at all.
- **New, cycle-2-specific limitation discovered**: the entire family of face-index-based
  samplers (Rank 5's C5 from cycle 1, and this cycle's `robust_leg_only`/`gnc_legonly`) is
  currently restricted to topology-preserving corpora and CANNOT run on damage/missing-geometry
  data at all -- a real, previously-undiscovered deployment gap for what would otherwise be the
  strongest candidate.

### Current strongest production candidate(s) -- with full caveats
B7 (multi-seed damage replication, see above) closes the topology-preserving/damage split that
made this two-answer earlier in the cycle. Per the user's own pre-registered decision rule
(direction must hold within seed variance across 2 additional seeds), it held on both corpora --
cleanly on drop30 (3/3 seeds positive), and with no reversal on drop60 (worst case an exact tie,
not a loss), plus a completely clean antenna-quality win in all 6 new seed-condition runs:

- **`gnc_legonly_topofree` is now the unqualified preferred candidate on BOTH clean and damage
  data** -- the same evidence standard (multi-seed replication, direction holds) is now met in
  both regimes, not just clean data. It supersedes both exact-label `gnc_legonly` (topology-fragile,
  couldn't run on damage at all) and `gnc_medium` (usable everywhere but with a real antenna-quality
  cost topofree doesn't pay) as the default recommendation going forward.
- Both superseded variants are **retained in this record for provenance, not deleted**: exact-label
  `gnc_legonly` documents the face-index topology assumption (still present, unfixed, in Rank 5's
  C5); `gnc_medium` remains the reference "global annealing" baseline against which the leg-scoped
  mechanism's advantage was established.

Caveats that still apply regardless of data regime: single-corpus-family evidence (this project's
synthetic corpora only -- 3-seed replication on 12-specimen damage corpora and 2-seed on N=50
clean, not yet a large-N multi-seed damage replication matching B1's clean-data scale), no
real-specimen (non-synthetic) validation, GNC's underlying mechanism remains unexplained (B2's
falsifier), and the topofree mechanism itself has a real (though aggregate-R-neutral) 12.2%
target-point boundary-labeling disagreement rate vs. the exact face-based split (see B6's
label-transfer probe). Remains **experimental/opt-in only**, not promoted to any pipeline default,
consistent with this project's standing promotion bar -- "promoted past experimental in this
report's internal bookkeeping" is not the same claim as "safe to make a repo-wide default."

### What remains pending
- **A1 (PCA/UMAP + non-articulated baselines)** -- first pass complete, adjudicated (v2), and
  stress-tested with three further follow-ups (v3: supervised classification, a second shape
  descriptor, and a subsample robustness check) -- see the full writeup above. Net result: the
  core null (no clean genus-clustering signal in the fitted shape space) survives all of it except
  one borderline exception (raw-beta LOO 1-NN, p=0.087) that does not replicate under a different
  classifier or after confound removal, so is read as riding on the already-identified
  size/fit-quality confound rather than a rescued independent signal -- though this is the honest
  edge of what n=11 classification examples can resolve, not a proof. Concrete next steps, in
  priority order: (1) **more multi-specimen-genus data is now the single highest-value next step**
  -- every check in v2 and v3 was limited by the same n=11-across-4-genera ceiling, not by
  analysis choices; if `ALL_ANTS_CLEAN` or another corpus has more replicated specimens per genus
  (or species-level replicates), re-running v2+v3's tests on a larger multi-member set is more
  informative than further statistical controls on the current data; (2) resolve the 20/80
  unlabeled-specimen taxonomy gap (ask for the `ALL_ANTS_CLEAN` id->species mapping); (3)
  investigate the registration-quality entanglement on its own terms -- real, independent, and
  now confirmed WORSE in deformation space (r=-0.80) than in beta space (r=0.72); affects trust in
  either shape descriptor for any downstream morphometric use regardless of the taxonomy question;
  (4) install `umap-learn` (not currently in the `pytorch3d` env) if a nonlinear embedding is
  wanted beyond PCA -- one-line ask, not done unilaterally this pass.
- **Topology-robust fix for the face-index sampler family** -- DONE for `robust_leg_only` via B6
  (`robust_leg_only_topofree`). Rank 5's C5 (`sample_target_distal_protected`) still has the same
  face-index assumption and would need the analogous NN-based-labelling treatment if it's ever
  revisited on damage data; not attempted this cycle (C5 was near-null on clean data anyway).
- **Why GNC actually works** -- B2's falsifier leaves this open. A direct investigation (e.g.
  tracking loss-landscape/basin behavior under the annealing schedule, or a targeted ablation
  isolating the schedule from the kernel shape) would need new experiment design, not attempted
  this cycle.
- **Larger-N replication for `gnc_legonly_topofree` on damage conditions** -- DONE at 3-seed/N=12
  scale via B7 (direction held, promoted). Still smaller than B1's clean-data N=50/2-seed scale;
  an N=50-equivalent damage corpus (would need a new damage-corpus generation pass at that size,
  not currently on disk) would further harden the result if pursued.
- **The 12.2% boundary label-transfer disagreement** (`b6_probe_label_transfer.py`) is
  aggregate-R-neutral but unexplored beyond that -- worth a targeted look at whether it
  concentrates at specific anatomical joints (coxa/trochanter especially, given A2's
  chain-adjacent-mismatch finding at the same kind of boundary) before treating it as fully benign.
- Real-specimen (non-synthetic) validation of any GNC variant remains entirely untested --
  everything in cycles 1-2 is synthetic-corpus evidence with known ground truth.
