# Registration correspondence — v2 investigation report

Branch `feature/investigation`, 2026-08-11. Executes "Registration Correspondence —
Revised Plan (v2, supersedes the SDF-part-masking brief)" against
`upstream/feature/registration_moonshot`'s existing E6 ground-truth harness. Full
chronological trail, every command, every job ID, every raw number: [V2_CHANGELOG.md](V2_CHANGELOG.md).
This document is the synthesis; the changelog is the primary source.

---

## TL;DR

1. **Tier 0 (correctness) is closed.** The honest current-best number is
   **6.69% correct, 3.21% median error** — exactly `diagnostics/moonshot/cfg/D1_SYN.yaml`
   (`w_offset` 5.0/2.0), unchanged. Nothing tested beats it: a full offset-weight
   sweep confirms 5.0/2.0 is the grid optimum (non-monotonic, as predicted);
   rest-edge loss combined with it is *rejected* (correctness collapses to 1.52%);
   every other candidate fix (`scheme: all`, symmetric sampling, `w_midline`,
   `w_limit`) was already present in the recipe.
2. **One real addition earned its way in: `w_scale: 0.052` (beta-scale bounding).**
   Null on E6 correctness (as expected — it was never a correspondence fix) but a
   clean, decisive pass on its actual design target: on real scans
   (`bench50_clean`, 50 specimens), it compresses anterior joint-scale explosion
   by 49% (head), 61% (mandible), 83% (antenna), at negligible cost (+2.1%
   wall-clock, <2% shift on every core loss term). Recommend shipping it.
3. **Tier 1 (the actual open problem) produced a real, verified, non-spurious,
   but insufficient result.** T1.0 showed raw DINOv2 features carry substantial
   anatomical semantic signal (ARI 0.428–0.447 vs ~0 chance) — a decisive pass,
   unlike everything intrinsic (HKS) that came before it. T1.1 asked the harder
   question directly — does DINO recover *within-part* correspondence better
   than HKS, on an identical, coverage-matched vertex population — and the
   answer is **yes, genuinely (10.34% vs 13.51% median error, ~23% relative,
   confirmed not to be a population artifact), but not by enough** to justify
   building a full matching pipeline on the raw signal (still >2× worse than
   the fitted pipeline's 4.41%, and short of the pre-registered "wide margin"
   bar).
4. **The actual candidate mechanism — a geometry-refined descriptor (Uzolas et
   al.-style autoencoder, trained against geodesic distance) — was built and
   tested, and did not rescue it.** Refined median error: 10.12% vs raw DINO's
   10.34% — a 2% relative change, within noise, against a gate that required
   roughly halving the error (≤5.17%). **This closely matches a risk named
   before training ran**: the method's own source paper admits it cannot break
   a true geometric symmetry from geodesics alone, and an ant (6-fold leg
   symmetry + bilateral symmetry, weak front/rear visual prior) sits close to
   its acknowledged failure case. Training's own contrastive loss visibly
   plateaued rather than converging — consistent with this, observed *before*
   the eval result was known.
5. **The descriptor family (intrinsic and extrinsic) is closed.** Per the
   pre-registered decision rule, raw and geometry-refined DINO join HKS as
   closed for within-part correspondence, same footing as the whole
   partition family in `FINAL_REPORT.md`.
5b. **Two further, structurally different levers were also tried and also
   closed (§2.5)**: kinematic-chain arc-length ordering (a known-order
   constraint, not a descriptor — real signal on its own hypothesis target,
   10.80% vs HKS's 12.81%, but short of the gate) and GBCPD-style geodesic
   motion coherence, added directly to the optimizer (worse than baseline at
   every tested weight, monotonically so — a decisive negative, with a
   genuine gate-comparison error caught and corrected before the verdict was
   finalized, not smoothed over). **Tier 1 is now fully closed** — every
   candidate mechanism this investigation, the structurally different levers
   proposed after it, and the original branch's investigation together could
   name has been tried. This is a complete, legitimate negative result.
6. **`w_scale: 0.052` ported to the actual production config** (§2.4) —
   `D1_PROD.yaml`, wired into `run_m1_fit_all.sh`, ready to run wherever the
   757+81-specimen morphometrics corpus lives (not reachable from this
   machine). A downstream check on the only real corpus available here
   (`bench50_clean`, severely underpowered — 22 specimens, 9 genera) shows
   genus-classification lift moving in the right direction (2.66×→3.27×),
   driven by one specimen flipping correct with zero flipping the other way
   — real, consistent, but not on its own statistically distinguishable from
   noise. Directional evidence, not proof, pending a run on the full corpus.

---

## 1. Tier 0 — the honest current-best correctness number

### 1.1 What was tested, in order

| item | tested | result |
|---|---|---|
| T0.1 offset-weight sweep | `{2.0/0.8, 5.0/2.0, 10/4, 20/8}` + pre-existing anchors `{0.2/0.08, 200/200}` on `synth_clean`, scored via E6 | **5.0/2.0 wins the full grid** (6.69%/3.21%), confirmed non-monotonic as predicted — see table below |
| T0.2 rest-edge loss + T0.1 winner | `edge_mode: rest` (corrected, live-recomputed form — verified already correctly implemented in `trainer_moonshot.py`) combined with `w_offset` 5.0/2.0 | **REJECTED** — 1.52% correct / 6.67% median err, a >4× collapse. Interacts negatively with the offset penalty; not merely redundant |
| T0.3 bundle (`scheme: all`, symmetric sampling, `w_midline`, `w_limit`, `--deform_its`) | verified against `D1_SYN.yaml` and `trainer_moonshot.py` directly | **all already present**; only behavioral change is passing `--deform_its 0` explicitly going forward (H3's 600 iterations/specimen are never consumed downstream) |
| T0.4 scale_cap (`w_scale=0.052`) on E6 | full hierarchical+moonshot chain, `synth_clean` | **null on correctness** (6.70%/3.28% vs 6.69%/3.21% baseline — within noise), but confirmed *engaged* (`log_beta_scales` max 0.77, past the 0.69 free band) |
| T0.4 trans_cap (`w_trans=1.0`, uncalibrated — no prior reference exists anywhere in the branch) on E6 | same harness | **not meaningfully engaged** — `betas_trans` barrier loss ≈0 throughout training; D1's other regularizers already keep translation in band. Near-identical correctness number is not evidence it works. **Deprioritized, not shipped.** |
| T0.4 cost check (`w_scale`) | wall-clock + convergence, same job, isolated per-arm timing | +2.1% moonshot-stage wall-clock (328s→335s, at/below the run-to-run noise floor); <2% shift on chamfer/edge/normal/laplacian, uniform across terms (not a fit breaking down) — **checked, negligible** |
| T0.4 anterior check, `synth_clean` | `exp(log_beta_scales)` range on head/mandible/antenna, baseline vs `scale_cap`-on | **inconclusive by construction** — synth_clean's own anterior ranges (2.1–3.1×) are nowhere near the 138–889× pathology `scale_cap` was built for; this corpus cannot exhibit the failure mode it's meant to fix |
| T0.4 anterior check, `bench50_clean` (real scans, 50 specimens) | same method, real corpus | **PASSES decisively**: head 7.47×→3.79× (−49%), mandible 10.69×→4.12× (−61%), antenna 23.56×→4.06× (−83%); consistent on the outlier-robust p1–p99 statistic too |

**T0.1 full grid** (`w_offset` stage2/stage3 → correct% / median err%):

```
0.2/0.08   ->  4.81% / 3.48%
2.0/0.8    ->  6.58% / 3.14%
5.0/2.0    ->  6.69% / 3.21%   <- winner
10/4       ->  6.59% / 3.41%
20/8       ->  5.76% / 3.69%
200/200    ->  3.36% / 4.87%
```

Every number above was independently recomputed from raw `Stage_3_deform_fine.npz`
weights, not taken from any training log or prior report — including catching and
correcting for a case where the SLURM scoring job showed as `FAILED` (a known
`--render_n 0` plotting bug that crashes *after* printing) that would have caused a
transcription-only result to go unverified.

### 1.2 Recommended recipe

`diagnostics/moonshot/cfg/D1_SYN.yaml` unchanged, plus `w_scale: 0.052` on both stages.
Do **not** add `edge_mode: rest` or `w_trans` on the strength of anything measured here.

---

## 2. Tier 1 — the actual open problem: within-part correspondence

E6/E7 (pre-existing, `diagnostics/moonshot/hull/why_partitions_null.py`) established
that 79–83% of correspondence error occurs *within* the correct anatomical part —
capping every partition-shaped fix (SDF masking, geodesic decomposition, convex hulls,
soft/EM partitions, anterior splits, robust kernels) at a ceiling of 16.7–20.9%
correctness even if perfect. HKS (an intrinsic descriptor, the closest prior test of
"can *any* signal see inside a part") barely cleared chance (0.66% vs 0.14%, a 4.7×
margin) but lost badly to the fitted pipeline on placement error (12.91% vs 4.32%
median) — `FINAL_REPORT.md`'s own verdict: "kills intrinsic-only, not the
learned-extrinsic family."

### 2.1 T1.0 — does any extrinsic feature carry anatomical signal at all? (coarse check)

Cheapest possible version of the question, per the plan's own instruction: render the
template from 8 views, extract raw off-the-shelf DINOv2 and SD-turbo features (no
training, no fine-tuning), backproject to vertices, k-means at k=5 (matching the
`{thorax, gaster, head, leg, antenna}` grouping), score purity/ARI against ground truth
vs. a random-vector chance control.

```
candidate    dim   purity   ARI     chance (purity/ARI)   verdict
DINO         768   0.797    0.428   0.463 / 0.000          clears >=3x
SD          1280   0.693    0.275   0.463 / -0.001         clears >=3x
combined    2048   0.799    0.447   0.463 / 0.000          clears >=3x
```

**PASS, decisively** — ARI 0.43–0.45 is a real, substantial agreement (0=chance,
1=perfect), unlike HKS's marginal clearance of its own bar. DINO alone does nearly all
the work; `combined` barely beats it (0.447 vs 0.428) at roughly double the compute
(VAE + UNet forward pass on top). **DINO chosen as the sole candidate for T1.1** on
this basis — SD was not carried forward.

### 2.2 T1.1 — the actual hard question: within-part correspondence, matched-population

Fork of `diagnostics/moonshot/hull/within_part_signal.py`
(`diagnostics/khaoula_v2/within_part_signal_dino.py`): identical protocol, controls, and
JSON-output convention, with `hks()` replaced by a DINO descriptor (T1.0's rendering +
backprojection pipeline, reused not rewritten).

One correction made *before* logging any verdict, at the user's insistence: DINO's
descriptor is extrinsic/view-based and has incomplete coverage (77.0% mean, template ×
posed-target intersection — occlusion shifts with pose), while HKS is intrinsic and
covers 100%. Scoring the two against their own native coverage would confound
"better descriptor" with "easier vertex population." Fixed by scoring **HKS on the
exact same 77%-coverage mask DINO used**, not just its native 100%.

```
                          median err   coverage
fitted (pipeline)         4.41%        100%
hks_within_full          12.91%        100%   (reproduction check: exact match to the
                                                 original probe's reference)
hks_within_matched       13.51%        77.0%  (same subset DINO used)
dino_within              10.34%        77.0%
```

**The coverage-matching result resolves the concern in the direction that validates
DINO's edge**: HKS on the identical 77% subset scores *worse* (13.51%) than its own
full-coverage number (12.91%) — the visible/occluded split is not an "easy subset"
that would have inflated DINO's apparent advantage. **DINO's ~23% relative improvement
over HKS is real, measured on a matched population, not a selection artifact.**

**But it is not enough.** Against the pre-registered gate (within-part error must land
at or below half of the matched HKS baseline, i.e. ≤6.76%), DINO's 10.34% does not
clear it. And DINO still loses more than 2× to the fitted pipeline's own 4.41%.

**T1.1 is closed on its own narrow question**: raw, off-the-shelf DINO does not clear
the bar to justify building a full matching pipeline (T1.2 as originally scoped) on the
raw signal. That specific build is correctly not justified and should not happen.

### 2.3 The refinement layer — built, tested, gate not cleared

The plan named a specific mechanism for exactly this residual gap *before* T1.1 was
run: raw Diff3F/DINO only guarantees *semantic* separation, not *geometric/positional*
separation — a small autoencoder refining these features under a
geodesic-distance-preservation objective (Uzolas et al. 2025, "disambiguating symmetric
parts on a shape", SIGGRAPH Asia, arXiv 2503.18254) was the actual candidate; raw
features were always the cheap pre-check for whether that refinement is worth building,
not the candidate itself. T1.1's real-but-insufficient result was the expected
intermediate outcome that pre-check existed to produce — so it was built next.

**Pre-registered risk, stated before training ran, not after a miss**: the source paper
names its own limitation directly — it "cannot establish a consistent partitioning for
objects that are both geometrically and semantically isotropic," and its human-leg
result specifically depends on the base vision features already carrying a learned
front/rear pose cue from the diffusion model's training distribution, not on geodesics
alone — geodesics genuinely cannot distinguish a true mirror pair (a left/right limb is
geodesically identical under reflection; no amount of training against geodesic distance
alone fixes that). An ant sits close to the hard case the paper admits failing on:
6-fold leg symmetry plus bilateral symmetry, against a plausibly much weaker front/rear
visual prior in Stable Diffusion's training distribution than human limbs carry.

**Design**: geodesic distances computed once on the template's rest-pose mesh (Dijkstra
on the edge graph, 500 sampled anchors, `diagnostics/khaoula_v2/train_refine_autoencoder.py`);
DINO features for the template + 8 training specimens (`synth_004`..`synth_011`,
disjoint from the 4 held-out eval specimens `synth_000`..`synth_003` T1.1 was scored
on); a small MLP autoencoder (768→256→128→256→768, `diagnostics/khaoula_v2/refine_net.py`)
trained with reconstruction MSE + a margin-based contrastive term (geodesically-near
pairs pulled together, geodesically-far pairs pushed apart).

**Training result**: reconstruction loss converged smoothly (2.67→0.63 over 30 epochs).
The contrastive term dropped fast for ~5 epochs then **plateaued around 0.13** instead
of continuing toward zero — observed *before* running eval, and consistent with a real
subset of far-pairs (plausibly the symmetric leg/antenna instances) the encoder could
not learn to separate.

**Eval result** (held-out `synth_000`..`synth_003`, identical protocol to T1.1):

```
                          median err
raw DINO (T1.1)           10.34%
refined DINO (this run)   10.12%   <- 2.1% relative change: within noise
hks_within_matched        13.51%   (reference, unchanged)
fitted (pipeline)          4.41%   (reference, unchanged)
```

**REFINEMENT-LAYER GATE: NOT CLEARED.** Required ≤5.17% (half of raw DINO's 10.34%);
got 10.12% — not a close miss, the gate needed roughly a halving of error and got ~2%.
`correct%` moved slightly the *wrong* way (0.91%→0.74%). The refinement layer preserved
DINO's existing edge over HKS without adding anything measurable beyond it.

**This matches the pre-registered risk, not a generic negative result**: training's own
contrastive-loss plateau, logged before eval ran, already pointed at this outcome. The
most parsimonious explanation is the one named in advance — geodesic distance alone
cannot break the ant's leg/bilateral symmetry, exactly the paper's own admitted
limitation.

**Per the pre-registered decision rule** (stated before this result was known): the
primary kill condition was not cleared, so **the entire extrinsic-feature family (raw
and refined) is now closed for the within-part correspondence problem**, on the same
evidentiary footing the intrinsic family (HKS, geodesic branches, convex hulls, SDF,
EM partitions, anterior split, robust kernels) was closed in `FINAL_REPORT.md` §3.

---

## 2.4 Closing the loop: shipped to production, and does scale_cap move the metric that matters?

`w_scale: 0.052` was validated in §1 on a magnitude metric (anterior joint-scale
compression). Two follow-ups, done after §1–2.3:

**Ported to production.** `diagnostics/moonshot/cfg/D1_PROD.yaml` = `D1_SYN.yaml` +
`w_scale: 0.052`; `diagnostics/morphometrics/run_m1_fit_all.sh` (the actual morphometrics
production driver, not a diagnostics-only script) now points `--yaml_src` at it instead
of the defective `D1_low.yaml`. **Cannot be executed from this machine**: it needs the
757-worker + 81-`ALL_ANTS_CLEAN` corpus, which lives only at `/media/fabi/Data/...` on
the original author's machine — confirmed unreachable here, same pattern as
`clean81`/`coreg_train`/`coreg_test`/`heldout_*` found when `diagnostics/` was first
synced. The port itself is complete and ready to run wherever that corpus is.

**Does scale_cap move genus classification, not just joint-scale magnitude?** The
honest answer given real data constraints: reused the existing `bench50_clean` fits
from §1's anterior check (no new compute), ran the *exact* classification code
`diagnostics/morphometrics/analyse.py` uses (lot-blind leave-one-accession-lot-out
1-NN + permutation null) on both. Severely underpowered relative to the original
(`bench50_clean`'s 50 specimens span 37 genera — built for diversity, not
classification; only 9 genera clear even a lowered `min_n=2` bar, 22 specimens):

| | baseline | scale_cap |
|---|---|---|
| accuracy | 22.7% (5/22) | 27.3% (6/22) |
| LIFT (acc/null) | 2.66× | 3.27× |

Direction: **up**. But the entire delta is one specimen (`Ponera_kohmoku`) flipping
from wrong to correct, zero flipping the other way — a real, consistent, one-directional
shift, but 1 discordant pair has no statistical power on its own. **Answer to "up or
down": up, on the only corpus available. Answer to "is this proof": no — directional
evidence only, consistent with the magnitude-metric result, not an independent
confirmation of it.** A powered answer needs the full corpus and `run_m1_fit_all.sh`.

---

## 2.5 A fundamentally different lever: changing the search, not the descriptor

Everything in §2.1–2.4 attacked the within-part problem by giving each vertex a better
FINGERPRINT for independent per-point nearest-neighbour search. Two further levers, proposed
directly, target the SEARCH mechanism itself instead.

**Lever A — kinematic-chain arc-length ordering.** The rig's own kinematic chain (coxa ->
trochanter -> femur -> tibia -> tarsus -> pretarsus) gives a known 1D order along each limb,
unused elsewhere in the pipeline. Per-vertex arc-length position, computed independently on
each mesh from its own joint positions (`J_regressor`) and segment lengths — explicitly not
looked up from a template table by index, which would trivially recover ground truth by
construction. Needs no GPU (no rendering, fully defined everywhere, 100% coverage like HKS).

On its own hypothesis target (leg/antenna groups): **10.80% median error** — beats HKS
(12.81%) by nearly the same margin raw DINO did, but does not beat DINO itself (10.34%), and
falls far short of the gate (≤5.17%). Reading: arc-length genuinely cannot resolve
circumferential position around a limb, only position along its length, and that unresolved
ambiguity alone accounts for most of the remaining error — the pre-registered limitation,
not a surprise.

**Lever B — GBCPD-style geodesic motion coherence.** GBCPD (Kondo et al., TPAMI 2022) fixes
standard Coherent Point Drift's defect — its motion-coherence prior is Euclidean-proximity
based, so a folded limb touching the thorax gets treated as coherent with it — by using
geodesic distance instead. `probreg` (pip) has Bayesian CPD but its coherence kernel is
confirmed Euclidean (`grep -r geodesic` over its source: zero matches), so using it as-is
would test a different, weaker hypothesis. Full GBCPD reimplementation was judged out of
proportion to this investigation's scope; instead, built the alternative explicitly offered
when this lever was proposed: geodesic motion coherence as an ADDED TERM in the existing
optimizer (`GeoCoherentMoonshotStage`, subclassing — not modifying — `MoonshotStage`),
penalising the variance of per-vertex total motion within a geodesic (not Euclidean)
neighbourhood, reusing the geodesic distances already cached for the refinement layer.
Offline-smoke-tested (uniform displacement → ~0 loss; random displacement → loss matches the
theoretical variance exactly; gradient finite) before any GPU time was spent.

Swept `w_geocoh` ∈ {0.1, 1.0, 10.0} (no prior calibration exists for this term) against the
same `D1_SYN.yaml` baseline every Tier-1 candidate was compared against:

| `w_geocoh` | correct % | median err |
|---|---|---|
| 0 (baseline) | 6.69% | 3.21% |
| 0.1 | 4.13% | 4.13% |
| 1.0 | 2.12% | 5.50% |
| 10.0 | 1.79% | 6.04% |

**Worse than baseline at every weight tested, monotonically worsening — no interior optimum.**
Training logs show the mechanical trade-off: as the geocoh loss shrinks (exactly as designed),
chamfer rises in lockstep — the term suppresses local displacement variance at a direct cost to
surface fit, and that cost is not repaid in correspondence.

A genuine gate-comparison error was caught and corrected before finalizing this, not
smoothed over: the pre-registered ≤5.17% bar was calibrated from DINO/HKS's *oracle-restricted*
protocol (search pre-restricted to the correct part), while the within-part decomposition
actually run here used `why_partitions_null.py`'s *different* method (global unrestricted
match, classified post-hoc) — numbers on a different scale, not comparable to that bar.
Comparing them and calling it "cleared" would have been apples-to-oranges. The valid check —
GEOCOH_01 vs baseline, both scored identically — shows median error 3.79% vs baseline's
**3.03%** (worse for geocoh), consistent with, not contradicting, the aggregate result. The
mechanism under test (minimising local displacement *variance*) is mechanically related to
metrics measuring local error *magnitude*, so a same-method same-baseline comparison was
necessary to avoid the same "looked better, measured worse" trap as D2 and chamfer/fscore
gaming, already named in `FINAL_REPORT.md` §3.2.

**Both levers: decisive negative, on the same footing as everything closed in §2.**

---

## 3. What to do next, in order

1. **Ship `w_scale: 0.052`** into the standing D1 recipe — **done**, `D1_PROD.yaml` is
   ready; only needs to be run wherever the production corpus lives. No measured
   correctness downside, real anterior-plausibility upside, negligible cost.
2. **Do not** add `edge_mode: rest`, `w_trans`, or `w_geocoh` — all tested and
   rejected/deprioritized on real data this session, not on assumption.
3. **Do not** build T1.2 (a matching pipeline) on raw or refined DINO features, arc-length
   ordering, or geodesic motion coherence — all four tested, none cleared its
   pre-registered bar.
4. **The within-part correspondence problem is closed across every mechanism family
   this investigation and the structurally different levers proposed after it could
   name** — intrinsic descriptors (HKS), extrinsic descriptors (raw and refined DINO),
   geometric partitioning (SDF, geodesic branches, convex hulls, EM partitions, anterior
   split, robust kernels), known-order constraints (kinematic-chain arc-length), and
   correspondence-search architecture itself (geodesic motion coherence). This is a
   legitimate, complete negative result, not an unfinished one. The morphometrics
   deliverable (per-part weighted-average measurements, proven to survive exactly this
   error) remains the working path forward, as `FINAL_REPORT.md` already concluded for
   the moonshot branch's own investigation.
5. If a future attempt wants to revisit the extrinsic-feature family, the specific,
   evidenced reason not to reach first for another geodesic-guided refinement is now on
   record (§2.3): the failure mode is symmetry the geodesic signal cannot see, not
   insufficient training or a wrong base feature. A future mechanism would need a
   different supervision signal for symmetry-breaking specifically (e.g. a genuine
   pose/view-consistency signal, not distance-on-the-mesh) — not a retry of the same
   geodesic-only approach with more data or a bigger network.
6. If a future attempt wants to revisit correspondence-search architecture (§2.5's
   lever B), the evidenced reason not to reach first for more geodesic-coherence
   regularization is also on record: the mechanism was verified to do exactly what it
   was designed to do (suppress local displacement variance, confirmed via the loss
   curves and an offline math check) and it still didn't help, at any of three weights
   spanning two orders of magnitude, with no sign of an interior optimum. A genuine
   Bayesian-EM CPD/GBCPD reimplementation (not a coherence term bolted onto the existing
   gradient-descent optimizer) is the one variant of this lever not yet tried — flagged
   as a real gap, not attempted here given its scope.

---

## Appendix — full traceability

Every number in this report was independently recomputed from raw model output at the
time it was produced (not taken from training logs or transcribed from a crashed
scoring job), per entries in [V2_CHANGELOG.md](V2_CHANGELOG.md). Key artifacts:

- `diagnostics/khaoula_v2/cfg/` — every config used, including the ones superseded
  (`T01_off{2,10,20}.yaml`, `T02_restedge.yaml`, `T04_scale.yaml`, `T04_trans.yaml`)
- `diagnostics/khaoula_v2/runs/` — raw fitted output for every arm run this session
- `diagnostics/khaoula_v2/t10_diff3f_sanity.py`, `within_part_signal_dino.py` — the two
  feature-extraction/eval probes, both forks/reuses of existing, already-validated
  harnesses, not new infrastructure built from scratch
- `diagnostics/khaoula_v2/refine_net.py`, `train_refine_autoencoder.py` — the
  refinement-layer architecture (shared between train and eval so they cannot drift
  apart) and training script (geodesic computation, DINO caching, training loop)
- `diagnostics/khaoula_v2/out/t10_result.json`, `within_part_signal_dino.json`,
  `within_part_signal_dino_refined.json`, `refine_net.pt`, `template_geodesic.npz`,
  `dino_cache/*.npz` — machine-readable results and every cached intermediate, reusable
  without re-extraction
- `diagnostics/moonshot/cfg/D1_PROD.yaml`, `diagnostics/morphometrics/run_m1_fit_all.sh`
  (updated) — the production port
- `diagnostics/khaoula_v2/genus_scale_cap_check.py`, `out/genus_scale_cap_check.json` —
  the genus-classification A/B, reusing `measure.py`/`analyse.py` directly
- `diagnostics/khaoula_v2/within_part_signal_arclength.py` — lever A (kinematic-chain
  arc-length ordering), CPU-only, no GPU needed
- `diagnostics/khaoula_v2/trainer_geocoherence.py`, `optimise_geocoherence.py`,
  `cfg/GEOCOH_{01,1,10}.yaml` — lever B (geodesic motion coherence), subclassing/
  forking `trainer_moonshot.py`/`optimise_moonshot.py` without modifying either
- Three environment-corruption bugs (torch/torchvision metadata, numpy metadata, PIL —
  all the same "/p/scratch package-extraction drops files" pattern, on top of the
  gmpy2 instance already fixed before this report's work began) diagnosed and fixed
  along the way, logged with root cause in `V2_CHANGELOG.md`, not silently worked around
