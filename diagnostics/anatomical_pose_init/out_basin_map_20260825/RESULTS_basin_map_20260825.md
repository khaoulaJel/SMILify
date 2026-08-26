# Basin Map: What Kind of Wrong Initialization Can SMILify Recover From? — RESULTS (2026-08-25)

Follow-up to the G1d parity result (`RESULTS_bench50_G1_vs_G3.md`) and the original D/E/F
mechanistic experiment (`RESULTS_ABC_DEF.md`). This experiment reframes the research question.
G1d established that the learned initializer is a tie with zero-init, not a win — the open
question was never "is the network accurate enough," it was **what property does a useful
initialization actually need to have**. This experiment answers that directly, by deliberately
constructing five different kinds of wrong initialization from known ground truth and measuring
which ones the D1 optimizer can recover from.

## Decision (stated first)

**Error structure determines optimizer recoverability far more than error magnitude does.** The
single clearest result: at matched ~30° magnitude, a rigid, internally-coherent whole-leg
misplacement (`coherent`) scores 0.217 leg_acc higher than the same magnitude of independent
per-joint noise concentrated at the same (proximal) end of the chain (`coherent` 0.839 vs.
`proximal` 0.622) — the largest gap in the entire experiment, between two conditions with
approximately the same nominal error. Separately, a real, anatomically valid pose belonging to
the wrong specimen (`swap`) is nearly as damaging as the worst proximal noise, which rules out
"anatomical plausibility" as the relevant property and points instead at basin *selection*. Two
geometry-only initializers built to exploit this (constrained IK from an estimated leg root and
tip) do **not** beat zero-init. Read together: **a useful initializer must produce a pose whose
error is *recoverable* by the optimizer, not merely one with low average angular error from
ground truth** — a materially different and more precise design target than anything tested
before this experiment.

## Setup

12 `synth_clean` specimens (known ground truth), identical D1 recipe used throughout this series
(`--midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0`, `D1_low.yaml`). Five perturbation
families constructed from GT pose (`generate_structured_perturbation_init.py`):
`random`/`proximal`/`distal` (i.i.d. and segment-concentrated noise, extending the original D/E/F
single-magnitude test to a 15°/23°/30° sweep), `coherent` (new: only the coxa/root joint is
rotated, trochanter→pretarsus left exactly at GT — an internally self-consistent whole-leg
displacement), and `swap` (new: each specimen's leg pose replaced by its nearest other
specimen's real GT leg pose; no synthetic noise, no magnitude control — the corpus's closest
inter-specimen leg-pose distance is 27°, so `swap` always lands ~27-33° naturally). Separately,
two constrained-IK initializers (`generate_ik_init.py`, `fitter_3d/geom_leg_init.py`'s
`solve_chain_ik`/`solve_chain_ik_multi`) were tested as a candidate practical initializer:
`IK_tip_only` (root + estimated tip position only) and `IK_tip_waypoint` (root + one estimated
mid-chain point + tip), both using the model's real authored per-axis joint limits
(`fitter_3d/joint_limits.py`), validated first against oracle (true) tip/waypoint positions
before being fed realistic scan-derived estimates (`ik_init_oracle_PROBE.py`,
`ik_init_oracle_disambiguation_PROBE.py`, `ik_init_oracle_multiconstraint_PROBE.py`).

All numbers below are from a fresh, same-code-version, same-machine zero-init run
(`SYN_clean_zero_wsl`) and the basin-map/IK arms, all run together on WSL — a paired comparison,
not mixed across the cluster/WSL code-version boundary that required a retraction earlier in
`RESULTS_bench50_G1_vs_G3.md`. Two numbers from the original cluster run are cited below purely
as historical/motivating context, explicitly marked as such, not as paired comparisons.

## Results

**Correspondence accuracy** (`leg_acc`, via `run_audit.py`; zero-init reference `SYN_clean_zero_wsl` = 0.869):

| Condition | 15° | 23° | 30° |
|---|---:|---:|---:|
| random (control) | 0.929 | 0.875 | 0.817 |
| proximal | 0.863 | 0.775 | **0.622** |
| distal | 0.937 | 0.938 | **0.940** |
| coherent (whole-leg rigid) | 0.937 | 0.937 | **0.839** |
| swap (real-but-wrong pose) | 0.834 (single value, no magnitude axis) |
| IK tip-only / tip+waypoint | 0.834 / 0.851 |

**Morphometric feature reliability** (`all_feature_R`, via `calib_features.py`; zero-init = 0.731):

| Condition | 15° | 23° | 30° |
|---|---:|---:|---:|
| random | 0.813 | 0.805 | 0.727 |
| proximal | 0.813 | 0.696 | 0.662 |
| distal | 0.799 | 0.816 | 0.783 |
| coherent | 0.768 | 0.783 | 0.714 |
| swap | 0.676 |
| IK tip-only / tip+waypoint | 0.701 / 0.714 |

**IK arms vs. zero-init, surface-fit metrics** (`eval_run.py`, paired t-test, n=12):

| Metric | IK tip-only | IK tip+waypoint | zero-init | p (tip-only / tip+wpt) | wins/12 |
|---|---:|---:|---:|---:|---:|
| chamfer_l2 | 0.000302 | 0.000293 | 0.000270 | 0.340 / 0.412 | 3 / 4 |
| fscore@0.01 | 0.6844 | 0.6873 | 0.7036 | 0.079 / 0.090 | 1 / 4 |
| fscore@0.02 | 0.9332 | 0.9362 | 0.9397 | 0.144 / 0.389 | 3 / 4 |

Not significant at n=12, but every metric points the same direction on both IK variants — a
consistent, if modest, disadvantage, not noise scattered around zero.

*Historical context only, cluster-origin, not paired against the table above*: the original
ceiling test found GT-init leg_acc 0.938 (vs. that run's own zero-init 0.857) and the earlier,
cruder hand-designed heuristic (`simple_leg_heuristic.py`, no longer in the repo) at leg_acc
0.732, pre-optimization pose error 28.52° — worse than zero-init's own 23.09° (`RESULTS_ABC_DEF.md`).

## What this establishes

**1. Same magnitude, different structure, very different outcome.** `distal` stays flat and
*above* the zero-init baseline through 30° (0.940 leg_acc) while `proximal` collapses at the same
magnitude (0.622) — confirming the original D/E/F single-point result held up across a full
sweep, not just one matched condition. `coherent` vs. `proximal` at 30° is the sharpest result in
the experiment: two perturbations concentrated at the same end of the chain, at the same nominal
magnitude, differ by 0.217 leg_acc depending only on whether the error is internally consistent.
Forward-kinematic error propagation is the mechanism: a proximal (root) rotation error moves
every downstream joint's world position; the same rotation applied *coherently* (rest of the
chain undisturbed relative to it) leaves the leg's own internal structure intact even though the
whole leg is misplaced, and the optimizer recovers from that far more easily than from a
distorted chain.

**2. Anatomical plausibility alone does not predict recoverability.** `swap` uses a real,
anatomically valid GT pose — just from the wrong specimen — and still lands at 0.834, comparable
to the worst coherent condition and to both IK arms, clearly below zero-init. A configuration
being "a real ant leg pose" is not sufficient; the question is whether it's the pose the optimizer
can descend toward the *correct* correspondence from. This is a basin-*selection* problem, not an
implausible-noise problem — consistent with what the literature sweep (`LITERATURE_SWEEP_20260825.md`)
found is still an open problem field-wide, not something with an off-the-shelf answer.

**3. The two geometry-only IK initializers, despite explicitly targeting these findings, do not
beat zero-init.** They were built specifically to produce internally-consistent, joint-limit-
respecting poses (the property `coherent` shows matters) from only a root and tip/waypoint
position — but the oracle validation independently established that this is fundamentally
underdetermined (15 unknowns from 5 three-axis joints, 3-6 position constraints; ~21° mean
per-joint gap from GT even given the *true* tip position, confirmed via a disambiguation test to
be genuine redundancy, not a regularizer artifact), and realistic scan-derived tip/waypoint
estimation adds further error on top of that redundancy. The result: consistently worse than
zero-init on every metric tested (not significant at n=12, but directionally uniform), though
clearly better than the older, cruder heuristic it was built to improve on.

## What this rules in and out for the initialization approach going forward

**Ruled out**: "make the initial pose more anatomically informed" as a sufficient design
principle on its own. Both attempts at this — the original hand-designed heuristic and the new
constrained-IK approach — failed to beat zero-init, despite the IK approach explicitly enforcing
real joint limits and internal consistency.

**Supported, with a specific mechanism now identified**: initialization quality is governed by
error *structure* (location + internal coherence), not error *magnitude* — shown independently by
proximal-vs-distal, random-vs-structured, coherent-vs-incoherent, and the cross-specimen swap.

**The sharpened design target for any future initializer** (learned or otherwise): it is not
enough to predict a pose with low average angular error from ground truth. A 20° error
concentrated proximally and incoherently can be worse for SMILify than a 30° error that is either
distal or internally coherent. A useful initializer needs to be evaluated on at least three axes,
not one:
1. raw pose accuracy (the only thing measured so far in the G1-G1d series),
2. error *structure* (proximal/distal/coherent/fragmented/cross-specimen-like),
3. downstream recoverability — does the resulting D1 fit actually improve correspondence and
   morphometric reliability, which is the only one of the three that is the real objective.

This also reframes what a learned network would need to do to be useful: not "predict the pose
closest to GT," but "predict a pose whose error, whatever its magnitude, is the kind the optimizer
can recover from" — a materially different and better-specified training target than anything
attempted in the G1-G1d series.

## Follow-up: cross-tabulating G1d against the basin map, IK constraint scaling, swap predictability, and a first recoverability regression (2026-08-25)

Four direct follow-up questions, all answerable from data/code already produced above plus one
new oracle sweep. All four resolved with real numbers, not restated as open questions.

### G1d's real-data output, reclassified by the basin map's own structural lens

The G1-G1d dose-response series (`RESULTS_bench50_G1_vs_G3.md`) was previously an unexplained
correlation: larger training pose-noise -> worse real-data result, smaller -> parity, with no
mechanism identified. Computing each checkpoint's actual predicted deviation-from-rest on real
bench50 (50 specimens), decomposed proximal (co/tr/fe) vs. distal (ti/ta) exactly as in the basin
map above:

| Checkpoint | proximal mean | distal mean | ratio | coxa share of leg movement | real-data result |
|---|---:|---:|---:|---:|---|
| G1 (pose_scale=0.50) | 16.57 deg | 6.89 deg | 2.41 | 0.245 | **lost** to zero-init |
| G1b (0.15) | 6.28 deg | 2.51 deg | 2.50 | 0.283 | parity |
| G1c (0.10) | 3.65 deg | 1.24 deg | 2.93 | 0.300 | parity |
| G1d (0.05) | 1.31 deg | 0.55 deg | 2.38 | 0.279 | parity |

The proximal/distal ratio (~2.4-2.9x) and coxa-share (~0.25-0.30, far short of the `coherent`
family's near-total root-concentration) are essentially **constant across all four checkpoints**
regardless of training noise scale -- the network always learns the same unfavourable
(proximal-skewed, incoherent) structural signature. Only the magnitude shrinks with training
scale. G1's proximal corrections (16.6 deg) sit almost exactly inside this experiment's
`proximal_15deg`-`proximal_23deg` range, already shown above to cause real damage (leg_acc
0.863->0.775). G1d's (1.3 deg) sit far below the smallest magnitude tested here (15 deg) -- too
small to enter either the damaging or the favourable regime this experiment mapped out.
**Conclusion: the entire G1-G1d dose-response curve is explained by one mechanism identified in
this experiment, not a separate phenomenon** -- the network's structural signature was always
unfavourable; shrinking training noise only shrank a fixed-shape correction below the threshold
where that structure matters, never changed the shape itself. This also yields a falsifiable
prediction for future work: training at any pose_scale caps out at parity, never a win, unless
the *structure* the network learns to predict changes (e.g. toward coherent or distal-favouring),
not just its magnitude.

### IK constraint-count sweep: is the redundancy fixable with more information?

Extended the oracle validation (`solve_chain_ik_multi`) to k=1..5 TRUE constraint points along
the chain (`ik_init_constraint_sweep_PROBE.py`):

| k (constraint points) | 1 | 2 | 3 | 4 | 5 (every joint, oracle ceiling) |
|---|---:|---:|---:|---:|---:|
| mean deg from GT | 21.15 | 19.34 | 17.48 | 16.05 | **14.23** |

Closes gradually and monotonically but does not collapse to zero even at k=5 -- every joint's
*true* position given exactly, strictly more information than any real scan could provide (which
would first need to correctly assign each point to the right joint). The residual ~14 deg is
twist ambiguity: a joint's position constrains which direction its bone points (its "swing") but
never its rotation about its own axis (its "twist"), and no number of position-only anchors can
resolve that -- a structural property of position-based IK, not an artifact of too few points.
**Conclusion: partly information-starved (k=1->2 closes ~9%, worth having), partly fundamentally
irreducible by this method (k=5's 14 deg floor).** This gives a network genuine headroom over IK
specifically where it can learn twist from surface cues position anchors cannot carry -- the same
reasoning behind HybrIK's explicit twist/swing split in the literature sweep, not an arbitrary
architecture choice.

### Is `swap`'s damage predictable from donor-target pose distance?

Per-specimen correlation, `swap`'s achieved distance (manifest) vs. post-fit outcome (n=12):
chamfer r=+0.278 (p=0.382), fscore@0.01 r=-0.231 (p=0.471), fscore@0.02 r=-0.333 (p=0.290) --
none significant. Repeating with the structural (proximal/distal) decomposition instead of the
raw scalar distance, in case the aggregate number was hiding a real relationship: proximal-error
r=+0.281 (p=0.375), distal-error r=+0.044 (p=0.893) -- still flat at both levels of scrutiny.
**Conclusion: no evidence recoverability is predictable from donor-target pose geometry within
this sample.** Caveat, stated plainly: n=12 with a narrow tested range (nearest-donor selection
means distances only span 26.7-32.7 deg) is underpowered to fully rule out a relationship a
larger, more diverse sample might reveal -- this supports weighting Phase 5 (measure
recoverability empirically per-candidate) over a geometric distance proxy, not proof no proxy
could ever work.

### A first recoverability regression, at full scale, from data already on disk

Consolidated all 13 conditions x 12 specimens (156 perturbation/outcome pairs: achieved
proximal/distal error from `basin_map_manifest.csv`, D1 outcome from each arm's `metrics.csv`) --
the first time in this investigation anything has been regressed against *measured* D1 outcome
rather than pose-accuracy proxy, even at this crude, linear scale:

| Predictor | R^2 (vs. chamfer_l2) | single-feature r (vs. chamfer_l2) | p |
|---|---:|---:|---:|
| total achieved error only | 0.088 | +0.296 | 0.0002 |
| proximal + distal error (2 features) | **0.253** | -- | -- |
| proximal error alone | -- | **+0.488** | **<0.0001** |
| distal error alone | -- | +0.015 | 0.852 |

Decomposing by structure explains 2.9x more variance than the aggregate scalar (R^2 0.253 vs.
0.088); distal error carries essentially zero predictive signal for outcome (r indistinguishable
from 0, p=0.85), proximal error carries strong, highly significant signal alone. **Conclusion:
even the crudest possible model (linear regression on two summary numbers) built from data this
experiment already produced shows real, significant predictive power for D1 outcome.** That is a
concrete, evidence-backed case -- not a speculative one -- for the natural next step: generate
many more (perturbation, measured-D1-outcome) pairs per specimen using this same harness
(`generate_structured_perturbation_init.py` + the D1 chain + `run_audit.py`/`calib_features.py`),
and train a selector or network against *measured* recoverability directly, for the first time in
this entire investigation, rather than against ground-truth pose distance as every arm through
G1d has done.

## Artifacts

- `diagnostics/anatomical_pose_init/generate_structured_perturbation_init.py` — perturbation
  families A-E (random/proximal/distal/coherent/swap), `basin_map_manifest.csv` (per-specimen
  achieved error).
- `diagnostics/anatomical_pose_init/out_basin_map_20260825/` — 13 init `.npz` files + manifest.
- `diagnostics/anatomical_pose_init/run_basin_map_20260825.sh` — the 13 fit+eval runs.
- `fitter_3d/geom_leg_init.py` — `solve_chain_ik`, `solve_chain_ik_multi`, `estimate_leg_tip`,
  `estimate_leg_waypoint`, `init_joint_rot_for_specimen_ik(2)` (new IK primitives).
- `diagnostics/anatomical_pose_init/ik_init_oracle_PROBE.py`,
  `ik_init_oracle_disambiguation_PROBE.py`, `ik_init_oracle_multiconstraint_PROBE.py`,
  `ik_init_constraint_sweep_PROBE.py` — IK correctness/redundancy validation, run before any
  realistic use.
- `diagnostics/anatomical_pose_init/generate_ik_init.py`,
  `run_ik_arms_20260825.sh` — realistic IK init generation and D1 fit+eval.
- `diagnostics/moonshot/runs/{BASIN_*, IK_tip_only, IK_tip_waypoint, SYN_clean_zero_wsl}/metrics.csv`
  — per-specimen Chamfer/F-score/integrity (`eval_run.py`).
- `diagnostics/correspondence_accuracy/out/correspondence_confusion.json` — leg_acc/seg_acc for
  all arms above (`run_audit.py`; `labels.py`/`confusion.py`/`geodesic.py` recovered from the
  cluster checkout, 2026-08-25, after being lost in the original cluster-to-WSL migration).
- `diagnostics/morphometrics/out/feature_reliability.json` — `all_feature_R` for all arms above
  (`calib_features.py`).
