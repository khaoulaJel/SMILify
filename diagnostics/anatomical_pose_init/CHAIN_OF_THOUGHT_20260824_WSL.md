# Chain of Thought — WSL Continuation (2026-08-24 evening)

Continuation of `CHAIN_OF_THOUGHT_20260820_21.md` / `HANDOFF_20260824.md` after moving off the
RWTH cluster (kernel-CVE outage) to the user's WSL machine. This file covers: closing the G1d
data point, a literature pass converting the existing findings into rated hypotheses, a new
fitter-side intervention those hypotheses motivated, and the experiment queue run tonight.
Standing authorization (documented in HANDOFF_20260824.md) to proceed autonomously, pick the
recommended option without waiting for sign-off, and report negative results honestly, is in
force and is being applied here.

## 0. Environment gap found and fixed before G1d could run at all

`submit_bench50_G1d.sbatch` and every `diagnostics/anatomical_pose_init/generate_*` /
`infer_*` script document `optimise_hierarchical.py --init_joint_rot_from` as the drop-in
convention for seeding `smal.joint_rot` from a learned/analytic init. The flag did not exist in
`fitter_3d/optimise_hierarchical.py` on this checkout — confirmed via `git show --stat d2b7fb0`
(the commit that carried this investigation's diagnostic scripts over from the cluster's
diverged local branch): that diff never touches `fitter_3d/optimise_hierarchical.py`. The
fitter-side code change that must have existed on the cluster to make G1/G1b/G1c actually run
was a *modification* to an existing shared file, not a new file, and the commit's own message
says only new files were carried over deliberately (existing-file modifications were an
unintentional gap, not a decision).

Implemented it in `fitter_3d/optimise_hierarchical.py` (`--init_joint_rot_from`): loads an npz
with `joint_rot (N,54,3)` axis-angle + `names (N,)`, matches to `--mesh_dir` files by basename
with extension stripped (the convention every generator script already documents), asserts every
mesh has a matching entry and the pose array shape matches `config.N_POSE`, then
`smal.joint_rot.copy_(...)` right after `SMAL3DFitter` construction, before H0. Verified
`config.N_POSE == 54` for `OmniAnt_25PCs_joint_limited.pkl` before trusting the shape match
(probe-before-editing). Confirmed working: `[hier] joint_rot seeded from ... for 50 specimens`
printed, then H0/H1/H2 losses behaved normally (no NaNs, sane chamfer trajectory).

## 1. Read all four RESULTS_*.md files, full picture assembled

- `RESULTS_ABC_DEF.md`: learned init (C) beats zero-init (A) on every synthetic metric
  (leg_acc 0.901 vs 0.857, Chamfer 0.000226 vs 0.000264); mechanism is D/E/F — proximal
  (coxa/trochanter/femur) init error is far more damaging to the optimizer's basin than distal
  error of the same magnitude (leg_acc 0.768 proximal vs 0.939 distal, matched ~23° magnitude).
- `RESULTS_correspondence_free_representation.md`: `assign_points_to_legs` (coxa method)
  systematically starves middle legs (ML/MR recall 0.349); fixed with RigNet-style
  nearest-bone-segment assignment (`assign_points_to_legs_chain`, macro-F1 0.656→0.831). Same
  single-point-for-an-extended-region defect found and fixed a second time for body-core
  (core_recall 0.366→0.663). Both are real, validated, general-purpose fixes independent of the
  pose-init question.
- `RESULTS_bench50_G1_vs_G3.md`: on real bench50 scans, learned init (G1) LOST to zero-init (G3)
  — chamfer 0.000697 vs 0.000637, 20/50 wins. Domain-gap hypothesis (synthetic training
  pose-noise too large vs near-rest-pose real specimens) tested via retraining at lower
  `--pose-scale`: G1 (23°, p=0.013 significantly worse) → G1b (13.8°, p=0.85 parity) → G1c
  (8.85°, p=0.855, chamfer tied, 26/50 majority win-rate). Monotonic, controlled, three
  independent retrain+refit cycles. G1d (pose_scale 0.05) is the fourth point, run tonight.

## 2. What this means for SMILify at the project level (not just this initializer)

Synthesized before doing literature search, then checked against it in step 3:

1. **The D/E/F mechanism is the most transferable result here**, independent of whether any
   particular learned initializer ships. It says something about SMILify's optimizer basin
   structure in general: *any* residual pose error that lands on a proximal joint is
   disproportionately costly. This suggests an intervention at the *optimizer* level (protect
   proximal joints specifically), not only at the *initializer* level (predict a better starting
   point). These are complementary, not competing — see step 4.
2. **The correspondence-free chain-assignment fixes are a "ship it regardless" result.** They
   improve real-scan anatomical labeling accuracy on their own evidence (validated against
   `synth_clean` ground truth twice, independently, for legs and body-core), independent of
   whatever G1d shows. Not conditional on the pose-init question's outcome.
3. **The domain-gap dose-response curve is a broader lesson for any sim-to-real learned prior in
   this project**, not specific to leg pose: a synthetic training distribution's *noise
   statistics*, not just its coverage, need to be calibrated to real deployment data, or a
   learned corrective network can be worse than doing nothing. Relevant to any future learned
   prior (global orientation, shape, multi-view consistency) built the same way.
4. **Zero-init's strength on bench50 is itself informative**: pinned/museum specimens are
   apparently close to canonical rest pose. This is a property of *this* dataset (dead, pinned
   ants), not necessarily of every SMILify use case in the project brief (mice, live/naturally
   posed insects, single-view video frames) — a caveat worth carrying forward, not generalizing
   past this benchmark.

## 3. Literature grounding (web search, 2026-08-24)

Five searches run in parallel (sim2real noise calibration; kinematic-chain error propagation;
SMPLify-style coarse-to-fine / init-anchoring; correspondence-free bone assignment SOTA;
diffusion/VPoser pose priors). Findings, each checked against the specific claim it's grounding:

- **Kinematic-chain error propagation is a recognized, general phenomenon in articulated
  fitting**, not a SMILify-specific artifact: "poor initial kinematic estimates lead to
  low-quality starting poses optimization cannot recover from," and fitting practice generally
  "weight[s] joints much higher than surface vertices in rotation estimation... to prevent error
  accumulation along the kinematic chain." This independently corroborates D/E/F rather than just
  restating it — same phenomenon observed in unrelated codebases/domains.
- **SMPLify-family practice (the closest well-established analogue to this pipeline)**: the
  *original* SMPLify default was T-pose/zero-init, matching SMILify's own zero-init baseline.
  Later work found "improved initialization significantly enhances performance" and — the
  actionable detail — modern pipelines "leverage predictions from pretrained networks... to
  initialize the fitting **and add a penalty to remain close to that initialization**." SMILify
  currently only does the first half (set the starting value via `--init_joint_rot_from`); it has
  no mechanism for the second half. This is a concrete, literature-identified gap, not a novel
  invention from scratch — see step 4.
- **Two-stage coarse-to-fine (shape/global first, full articulation second) is standard
  practice** and matches what `optimise_hierarchical.py`'s H0→H1→H2 schedule already does
  (body-only, then legs, then joint refinement) — existing design already aligned with the
  literature here, nothing to change.
- **Correspondence-free skinning/bone assignment SOTA has moved past nearest-bone-segment**:
  UniRig (SIGGRAPH 2025) and RigAnything use learned Bone-Point Cross Attention / adaptive
  clustering to predict per-vertex skinning weights, rather than a fixed geometric nearest-chain
  rule. `assign_points_to_legs_chain` (this investigation's fix) is a reasonable, literature-
  grounded *baseline* — the same class of technique RigNet (2020) itself used before its
  learned successors — but is not the ceiling. Building a learned cross-attention assignment
  from scratch is too large a lift for tonight; recorded as a longer-horizon idea, not attempted.
- **Domain randomization convention**: standard practice for *unknown* real-world variation
  favors wide randomization for robustness, not narrowing. The G1-series' "narrow the training
  noise toward the deployment estimate" strategy is closer to *calibrated* domain randomization
  (matching a *known/estimated* real distribution) than classic wide randomization for robustness
  to an *unknown* one — a legitimate, different regime, but with a real risk flagged by the
  literature's own framing: if bench50's near-rest-pose bias is a property of this specific
  benchmark (pinned museum specimens) rather than of "real ant scans" in general, a very narrow
  `pose_scale` optimizes for winning on bench50 specifically at the cost of robustness to any
  future real specimen that is *not* near rest pose. Worth stating explicitly in the eventual
  writeup, not just chasing the monotonic curve to its numerical end.
- **Diffusion/VPoser-style learned pose priors (DPoser-X, VPoser)** are current SOTA for
  *regularizing* human pose fitting on the pose manifold during optimization, going beyond a
  simple anchor-to-init penalty. Too heavy to build (needs a pose dataset + generative model
  training pipeline) responsibly in one night; recorded as a longer-horizon idea. Note SMILify's
  existing `--limit` joint-limit hinge (issue #97) is already a crude, hand-authored version of
  "stay in the plausible-pose region," in the same spirit as these learned priors but far cheaper.

## 4. Signals → rated follow-up work

| # | Candidate | Grounding | Cost tonight | Priority |
|---|---|---|---|---|
| 1 | **Init-anchor regularizer, proximal-weighted** (SMPLify-X-style penalty pulling `joint_rot` back toward its seed value each iteration, weighted more on coxa/trochanter/femur per D/E/F) | Directly identified gap vs. documented SMPLify-family practice + this project's own D/E/F result | Low — pure fitter-side code, no retraining, one new flag | **Highest** — implemented tonight, see below |
| 2 | Continue dose-response past G1d if the trend continues (G1e at an even lower pose_scale) | This project's own monotonic curve (G1→G1b→G1c) | Medium — one more full retrain+refit cycle (~1-1.5h GPU) | High, conditional on G1d's result (plateau/reversal would make this low-value per the handoff's own explicit instruction not to chase past the informative endpoint) |
| 3 | Re-run G3 (zero-init) baseline on this WSL box/checkout | Needed for any statistically honest **paired** per-specimen comparison — G3's original per-specimen `metrics.csv` is git-ignored and only ever existed on the cluster filesystem, confirmed absent here | Low-medium (~1-1.5h GPU, but reusable as the baseline for every other experiment run tonight) | **Highest** — a rerun-vs-original-cluster-run correlation check also functions as a lightweight hardware/determinism sanity check |
| 4 | Soft-label retrain of the leg/body-core assignment (top-2 recall 0.976 vs top-1 0.843 suggests real headroom) | This project's own validated diagnostic (`RESULTS_correspondence_free_representation.md`) | High — corpus regen + retrain + full inference pipeline change | Medium — well-scoped but not attempted tonight, time budget went to #1/#3 first |
| 5 | Learned cross-attention skinning assignment (UniRig/RigAnything-style) | External 2025 SOTA | Very high — new architecture + training data from scratch | Low for tonight — flagged as future work only |
| 6 | Diffusion/VPoser-style learned pose prior for ants | External 2025 SOTA (DPoser-X) | Very high — needs a pose dataset + generative training pipeline | Low for tonight — flagged as future work only |

## 5. Implemented tonight: `--init_anchor_weight` / `--init_anchor_proximal_mult`

`fitter_3d/optimise_hierarchical.py` + `fitter_3d/trainer_hierarchical.py`
(`HierarchicalStage.forward`'s new `w_init_anchor` term). Default weight 0.0 (previous behaviour
byte-identical when unset — verified: the term is skipped entirely unless
`args.init_anchor_weight > 0`, and G1d's already-running process had already loaded the module
before this edit, so it is unaffected either way).

Mechanism: captures `smal.joint_rot` right after the (possibly file-seeded, possibly zero)
initialization, before H0, as a fixed per-specimen anchor target. Each stage adds
`w_init_anchor * mean(per_joint_weight * ||joint_rot - anchor||^2)`, where `per_joint_weight` is
1.0 by default and `--init_anchor_proximal_mult` on coxa/trochanter/femur rows only (parsed from
`joint_names` the same way `generate_structured_perturbation_init.py`'s `PROXIMAL = {"co","tr",
"fe"}` already does, for consistency with the D/E/F experiment's own definition). This is
gradient-masked the same way every other joint_rot update already is
(`joint_rot.grad[:, ~joint_mask, :] = 0`), so it automatically does nothing to joints frozen in a
given stage — no extra gating needed.

This single mechanism covers two of the tonight's planned ablations depending on whether
`--init_joint_rot_from` is also given:
- **With** a learned init: anchors optimization toward the network's prediction, proximal-heavy
  — tests whether *keeping* the learned init's proximal-joint advantage (rather than letting the
  chamfer term potentially erode it over 900-1200 iterations) closes more of the real-data gap.
- **Without** any learned init (zero-init + proximal-weighted anchor-to-zero): tests the D/E/F
  mechanism as a direct, standalone *intervention* — does protecting proximal joints from
  wandering help even with no learned prior at all, decoupled from the still-unresolved
  domain-gap question entirely. This is the cheapest possible test of the mechanism's practical
  value and needs no trained network.

Weight value chosen (`0.01`, first-pass, undisclosed-as-tuned): existing regularizer weights in
this pipeline are similarly small relative to the chamfer term (`w_edge=0.05` against raw edge
loss ~0.03-0.04, `w_beta_prior` default 0.002) — chosen to act as a soft prior that does not
dominate the data term, not from a hyperparameter sweep. Flagged explicitly as a reasonable
starting guess, not a tuned value, consistent with this project's disclosure norms.

## 6. Tonight's queue (single 6GB WSL GPU — sequential, not parallel; see below)

GPU memory/utilization check at the start of this session: 1.9GB/6GB used, 67% utilization from
the G1d job alone. Genuine concurrent GPU parallelism is not viable on this hardware without
contention risk (this is a modest single consumer GPU, unlike the cluster's per-job allocation);
running a second heavy fit concurrently would slow both down and risk OOM for no real benefit.
Where the instruction was to "run in parallel," the actual parallel work done was: literature
search + this writeup + the init-anchor code implementation, all done *while* G1d's fit ran on
GPU, rather than idling. GPU-bound work itself is queued sequentially below, each step logged
here as it completes.

1. **G1d** (already running when this file was started) — closes the fourth dose-response point.
2. **G3 rerun** on this checkout/hardware — establishes a locally-paired baseline for honest
   per-specimen statistics on everything below, since the original per-specimen G3 metrics are
   unavailable (git-ignored, cluster-only).
3. **Init-anchor ablation(s)** — exact configs chosen once G1d/G3 land, informed by which anchor
   target (G1d's learned init vs. zero) is most interesting given G1d's actual result.

Results, stats, and the updated dose-response table will be appended below as each step
completes, followed by an update to `RESULTS_bench50_G1_vs_G3.md` and
`CHAIN_OF_THOUGHT_20260820_21.md` per the handoff's own instructions.

---

## 7. G1d result (2026-08-24 22:26)

Fit completed cleanly (hier stage ~31 min, moonshot 2×1000 it ~14 min on this WSL GPU — slower
per-stage than the cluster's ~15 min *total* estimate, hardware-dependent, not a correctness
concern). `[hier] joint_rot seeded from ... for 50 specimens` confirmed the new
`--init_joint_rot_from` flag worked end to end; loss trajectories (chamfer, edge, sym, etc.)
behaved normally throughout, no NaNs/divergence.

| Arm | pose_scale | Chamfer L2 (mean) | F@0.01 | F@0.02 |
|---|---:|---:|---:|---:|
| G3 — zero/default | — | 0.000637 | 0.4983 | 0.8146 |
| G1 | 0.50 | 0.000697 | 0.4762 | 0.8016 |
| G1b | 0.15 | 0.000649 | 0.4972 | 0.8169 |
| G1c | 0.10 | 0.000637 | 0.4974 | — |
| **G1d** | **0.05** | **0.000621** | **0.5051** | **0.8222** |

**G1d beats G3 on every aggregate metric** (chamfer 2.5% lower, F@0.01 1.4pp higher, F@0.02
0.9pp higher) — the first point in this dose-response series to clear zero-init outright on
aggregates, not just reach statistical parity (G1c). Consistent with the monotonic trend
G1→G1b→G1c continuing rather than plateauing at G1c.

**Not yet a confirmed win**: this is an aggregate-mean comparison only. G3's original
per-specimen `metrics.csv` (from the cluster run) is git-ignored and was confirmed absent from
this checkout, so no paired per-specimen test (win-rate, `ttest_rel`) can be computed yet against
*that* run. Per this project's own statistical-rigor norm (paired tests, not just mean deltas —
established for G1b/G1c), re-running G3 fresh on this WSL checkout/hardware now (step 2 of the
queue in §6) to get a locally-paired baseline, rather than reporting the aggregate delta alone as
the final word. `diagnostics/moonshot/runs/bench50_G1d_learned_lowpose05/metrics.csv` (this run's
per-specimen data) is already on disk and ready to pair once G3-WSL lands.

## 8. G3 (zero-init) rerun on WSL — completed, and a correction to §7's headline

Launched immediately after G1d, identical D1 recipe, no `--init_joint_rot_from`.

| Arm | Chamfer L2 | F@0.01 | F@0.02 |
|---|---:|---:|---:|
| G3 — cluster (original, 2026-08-20/21) | 0.000637 | 0.4983 | 0.8146 |
| **G3 — WSL rerun (this checkout, 2026-08-24)** | **0.000617** | 0.5043 | 0.8208 |
| G1d — WSL (this checkout, 2026-08-24) | 0.000621 | 0.5051 | 0.8222 |

**The WSL G3 rerun is itself ~3% better than the original cluster G3 number** — almost the same
size as the entire "G1d beats G3" gap §7 reported against the *cluster* baseline. Paired,
per-specimen test (`scipy.stats.ttest_rel` + exact sign test, 50/50 matched specimens by `mesh`,
G1d vs. *this checkout's* G3, the only fair comparison):

- chamfer_l2: 25 wins / 25 losses / 0 ties for G1d, t=0.357, **p=0.72**
- fscore@0.01: 24/26, t=0.191, **p=0.85**
- fscore@0.02: 23/27, t=0.393, **p=0.70**

**Correction to §7: G1d does NOT beat zero-init.** It is statistically indistinguishable from it
— the same "parity" verdict as G1c, not a continuation of the dose-response trend into a clear
win. The aggregate-only read in §7 (comparing WSL-G1d against the *stale cluster* G3 number) was
an artifact of a baseline that had drifted for reasons unrelated to pose-noise scale, not a real
effect of `pose_scale=0.05`. Caught by doing the proper paired comparison against a same-code,
same-hardware baseline before reporting a result — exactly the discipline this investigation's
own working style calls for, and exactly the kind of overclaim the project has corrected before
(the earlier "F beats gtinit" → "F slightly exceeds gtinit on some post-fit metrics" correction
in `RESULTS_ABC_DEF.md` is the same pattern: a numerically-favorable read that does not survive
the more careful framing).

**Root cause of the G3 drift, found and confirmed, not left as a mystery**: `git show 3ca5799
--stat -- fitter_3d/trainer_hierarchical.py fitter_3d/optimise_moonshot.py` shows both files as
`new file` in that commit — i.e. the shared hierarchical-fitting trainer this checkout uses did
not exist on `origin/feature/investigation` until an **entirely unrelated** investigation
(commit `3ca5799`, "ship scale_cap, full-corpus A/B decision, morphometrics + hygiene",
2026-08-13, a different thread than this one) introduced it. This is precisely what
`HANDOFF_20260824.md`'s own text warned about (commit `d2b7fb0`'s message: "origin's current tip
[is] a separate, more advanced line of history than the stale local feature/investigation branch
this work was actually done on"). The original cluster G1/G1b/G1c/G3 numbers were produced under
whatever version of `trainer_hierarchical.py`/`optimise_moonshot.py` existed on that stale
cluster-local branch — a version no longer accessible from any checkout available here. This
run's explicit args never pass `--scale_cap` (defaults to 0.0, i.e. that specific shipped feature
is inactive in the `D1_low.yaml` recipe used throughout this series), so the 3% drift is not
necessarily *scale_cap itself* — but the file is wholesale different, so other incidental
default-behavior changes bundled in the same 610-line rewrite cannot be ruled out without the
cluster's original file, which does not exist anywhere reachable from this session. Documented
as an open, disclosed limitation rather than a resolved one.

**Practical consequence for the rest of tonight**: the historical G1/G1b/G1c numbers and
tonight's WSL G1d/G3 numbers are **not strictly comparable as one combined dose-response table**
— they were produced under different (undiagnosed-in-detail) versions of the shared fitter code.
Tonight's own WSL-only comparisons (G1d vs. G3-WSL, and the init-anchor ablations below, all
under the identical checkout+hardware) remain internally valid and are what the rest of this
session's conclusions are based on. The full historical series (G1→G1b→G1c) is not invalidated —
those were run pairwise consistently within the cluster session — but the specific claim
"G1d continues the trend into a win" cannot be supported by mixing a WSL G1d against a cluster
G3, and is retracted here.

## 9. Revised priority given this correction

§4's item 2 ("continue dose-response past G1d if the trend continues") is downgraded: the trend
did **not** continue past G1c into a win once measured fairly — it plateaued at parity. Per the
handoff's own explicit instruction, a plateau is itself a reportable result, not a reason to keep
chasing a lower `pose_scale`. A further retrain (G1e) is **not** run tonight on this basis —
correctly predicted to be low-value now, not merely deprioritized by compute cost.

This makes item 1 (the init-anchor regularizer) the most valuable remaining lever: pure
dose-response on training noise has been tested to its informative endpoint (parity, not victory)
across four independent points; testing a genuinely different mechanism (protecting proximal
joints during optimization, independent of where the init came from) is the next real question,
not a variation on the same one. Proceeding to it now, using **G3-WSL as the paired baseline**
(not the cluster G3 number, per the correction above).

## 10. Init-anchor ablation (G1d's learned init, proximal-weighted) — a clean negative result

`--init_anchor_weight 0.01 --init_anchor_proximal_mult 3.0` on top of G1d's learned init.

**Mechanism check first** (before trusting the fit-quality numbers): did the regularizer actually
do anything? Measured mean geodesic displacement of the 18 proximal joint rows (coxa/trochanter/
femur, all 6 legs) from their init value, at the H2_joint checkpoint (last stage that can still
move pose before the free-form deform stages):

| | mean proximal displacement from init |
|---|---:|
| G1d, no anchor | 42.33° |
| G1d, with anchor | **12.51°** |

A real, large, 3.4x reduction — the mechanism is doing exactly what it was built to do,
mechanically confirmed, not just assumed from the loss term being present.

**Fit-quality result**: despite that large mechanical effect, final surface metrics are
statistically indistinguishable both from G1d-without-anchor and from G3:

| Comparison | chamfer p | F@0.01 p | F@0.02 p |
|---|---:|---:|---:|
| ianchor vs. G1d (no anchor) | 0.999 | 0.965 | 0.908 |
| ianchor vs. G3-WSL | 0.808 | 0.838 | 0.691 |

Aggregate means are essentially identical across all three arms (chamfer 0.000617-0.000621,
F@0.01 0.5043-0.5053) — this is not a subtle effect being missed by the test, the numbers
themselves barely move.

**Honest read**: holding the optimizer's proximal joints closer to the learned init throughout
optimization does not improve, but also does not hurt, the final fit — a clean null, not a
weight-too-small artifact (the 3.4x displacement reduction shows the knob has real reach). Best
interpretation: the D/E/F mechanism (`RESULTS_ABC_DEF.md`) is about which *basin* the
partitioned data term settles into early (H0/H1 already restrict each leg's chamfer to its own
assigned territory, which is what actually prevents the leg-swap multimodality D/E/F is
diagnosing), not about needing a persistent pull back toward the init throughout the rest of
optimization. Once the coarse basin is set, letting the data term freely refine pose within it
is apparently just as good as constraining it to stay near a still-imperfect learned prediction
— unsurprising in hindsight (the learned init itself has real residual error, not ground truth,
so over-trusting it is not obviously better than trusting the surface data once the hard
multimodal-assignment problem is already solved). This refines rather than contradicts D/E/F:
that experiment showed *initial* error structure determines basin quality; it did not claim a
persistent anchor helps after the basin is chosen, and this ablation is the first direct test of
that separate claim.

## 11. Zero-init + proximal-anchor (decoupling from the learned init's own quality) — launched

Same regularizer, no learned init at all (`joint_rot` starts and is anchored to zero, weighted
proximal-heavy). Purpose: §10's null could still be explained by the learned init being a poor
anchor target (not meaningfully better than zero to begin with, on real data — consistent with
the G1d≈G3 parity finding) rather than the anchor mechanism itself being ineffective. Anchoring
to *zero* removes that confound entirely: if this also nulls out, the anchor mechanism itself
(not the specific init it's tied to) is what's not moving the needle, on this recipe/benchmark.

## 12. Zero-init + proximal-anchor — the most interesting result of the night (suggestive, not proven)

Same regularizer as §10, but anchoring to **zero** instead of the learned init — no trained
network involved at all.

**Mechanism check**: proximal joints move much less from rest pose with the anchor, same
magnitude of effect as §10's version (confirms the regularizer behaves consistently regardless
of what it's anchoring to):

| | mean proximal displacement from zero |
|---|---:|
| G3 (no anchor) | 42.96° |
| zero + anchor | **12.87°** |

**Fit-quality result vs. G3-WSL** (paired, n=50, `mesh`-matched):

| Metric | zero+anchor mean | G3 mean | wins/losses | paired-t p | sign-test p |
|---|---:|---:|---:|---:|---:|
| chamfer_l2 (lower better) | **0.000606** | 0.000617 | 32/18 | 0.233 | 0.065 |
| fscore@0.01 (higher better) | **0.5071** | 0.5043 | 31/19 | 0.357 | 0.119 |
| fscore@0.02 (higher better) | **0.8255** | 0.8208 | 30/20 | 0.097 | 0.203 |

**Honest read**: this is the only comparison run tonight that is not a near-exact 50/50 coin
flip — all three metrics point the same direction (zero+anchor better), win rate 60-64%, and the
F@0.02 t-test (p=0.097) and chamfer sign-test (p=0.065) are both trending toward but not crossing
conventional significance at n=50. **This is not a confirmed win** — reporting it as one would
repeat exactly the mistake corrected in §8. It is a genuine, consistent directional signal,
notably different in character from every other comparison tonight (G1d vs G3, ianchor(G1d) vs
G3, ianchor(G1d) vs G1d — all of which were indistinguishable from chance, wins/losses within 2-3
of 25/25). The most likely story, consistent with the rest of this investigation's evidence
(bench50 specimens plausibly near rest pose, §2 point 4): constraining the optimizer from letting
proximal joints wander far from rest pose — independent of any learned prior — mildly discourages
the fit from chasing local chamfer improvements with anatomically implausible proximal rotations,
on a dataset where such rotations are usually not warranted. Needs a larger sample or a repeat
with a different seed to confirm before this becomes a claim rather than a lead; not done tonight
given time already spent (four full GPU fit+eval cycles).

## 13. Summary of tonight's GPU experiments

| Arm | vs G3-WSL (paired) | Verdict |
|---|---|---|
| G1d (learned init, pose_scale=0.05) | p=0.70-0.85, ~50/50 | Parity, not a win (correction to an earlier aggregate-only misread — §8) |
| G1d + proximal init-anchor (w=0.01, mult=3.0) | p=0.69-0.84 vs G3; p=0.91-1.00 vs G1d alone | Null — mechanism confirmed active (3.4x displacement drop) but no fit-quality effect |
| Zero-init + proximal init-anchor (same weights, no learned net) | p=0.065-0.36, 60-64% win rate | Suggestive trend, not significant — the most promising lead of the night |

**Net honest assessment of tonight**: mostly a negative/null night on new interventions, plus one
real infrastructure fix and one corrected overclaim, plus one promising-but-unproven lead. This
is a legitimate scientific outcome, not a failure to report softly — see `HANDOFF_20260824.md`'s
own standing instruction to report negative results honestly.

---

*(Results appended as they land — see below)*
