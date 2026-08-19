# Task 3 — improve measurement of thin articulated appendages (leg_distal, R=0.136)

## Verdict: REJECT both C1 (joint-chain sum) and C2 (mesh-edge geodesic) as replacements for the
## current joint-to-joint bone length (C0) on the distal-leg span. Neither clears gate 1
## (ground-truth reliability) with a consistent margin across the required battery, so gate 2
## (cross-corpus replication) was correctly never run. `CORE_BLOCKS`/production code are
## unchanged. This is a legitimate, evidence-based null result, not a failed experiment — the
## task explicitly anticipated it ("do not assume geodesic length is automatically superior").

## 1. Why: Phase 1 diagnosis (full detail in `PHASE1_FINDING.md`)

The ceiling-test failure (R=0.136, zero domain gap) is a scattered, catastrophic-outlier
correspondence problem concentrated on thin, low-support distal geometry (82/38/8 dominant
vertices on tibia/tarsus/pretarsus vs 150-275 proximally), strongly aggravated by pose bending
(refitting at `pose_scale=0` nearly triples `leg_distal` R, 0.144→0.461, and closes most of the
distal/proximal joint-localization gap) — but pose is not a universal fix (antenna gets *worse* at
`pose_scale=0`). **This already predicts that a measurement definition built from the same
corrupted vertex region — geodesic included — is unlikely to fix a correspondence-level failure,
and a summed-edge geodesic path could plausibly be more fragile than a single joint-regressor
average.** Phase 3 tests this prediction rather than assuming it.

## 2. Phase 3: the required battery, same ground truth and scoring for every candidate

`diagnostics/appendage_evidence/score.py` runs C0/C1/C2 through the identical `measure_all` code
path on the fitted-vertex array and on the (always intact/undamaged) ground-truth array, scored
with the R/SNR/bias formula copied verbatim from `calib_features.py` — no candidate can get a
different ground-truth definition or scoring rule than any other (`candidates.py`/`score.py`
docstrings have the exact contract). Four conditions, matching the plan:

- **clean_pose25** — `SYN_clean_w5`/`synth_clean`, the existing ceiling test (R=0.136's own
  condition, pose_scale=0.25).
- **pose0** — `SYN_clean_pose0_w5`/`synth_clean_pose0` (Phase 1b), pose_scale=0.0, everything else
  identical. Reported here as one more evidence point, per your instruction — **not used to
  cherry-pick a candidate**; it does not carry more weight than the other three conditions below.
- **drop30 / drop60** — new: `diagnostics/appendage_evidence/make_damage_corpus.py` removes the
  geometrically most-distal 30%/60% of each leg's tibia-tarsus-pretarsus patch (246/492 vertices
  per specimen, ranked by TEMPLATE geodesic distance from the tibia joint, frozen and identical
  across specimens — the same style of frozen rule as C2's endpoints) from the **fitting target**
  before optimisation (SLURM job 3041347); ground truth stays the true, undamaged length (copied
  verbatim from `synth_clean/ground_truth.npz`) — the question is "can each measurement recover
  the animal's actual length from damaged input," not a redefinition of truth.

Family-median R / SNR / |bias| per condition (`out/phase3_full_comparison.json` has every
individual column):

| condition | C0 R / SNR / bias | C1 chain_distal R / SNR / bias | C2 geo_distal R / SNR / bias |
|---|---|---|---|
| clean_pose25 | 0.141 / 0.35 / 12.0% | 0.110 / 0.35 / **8.6%** | **0.155** / **0.42** / 24.1% |
| pose0 | **0.351** / 0.59 / 8.2% | 0.321 / 0.51 / **5.8%** | 0.151 / **0.77** / 18.3% |
| drop30 | 0.123 / 0.25 / **25.6%** | **0.171** / **0.38** / 27.7% | 0.093 / **0.38** / 38.2% |
| drop60 | -0.103 / 0.25 / **41.7%** | -0.147 / **0.38** / **34.7%** | **-0.016** / **0.35** / 46.3% |

(bold = beats C0 on that cell; `_full` whole-leg variants excluded from this table — see §3.)

## 3. Scoring against the two-gate criterion: does either candidate GENUINELY beat C0?

Counting wins across all 4 conditions × 3 metrics (12 cells each):

- **C1 chain_distal: 6/12 wins**, scattered with no coherent pattern — wins bias at
  clean/pose0/drop60 but loses bias at drop30; wins R/SNR only at drop30/drop60 but loses both at
  clean/pose0. A candidate that is ahead exactly as often as it is behind, with the wins and
  losses landing on different metrics in different conditions, is not a genuine improvement — it
  is noise around parity with C0 at n=12.
- **C2 geo_distal: 6/12 wins, but a clean pattern, not noise — and the pattern is bad.** SNR wins
  in **4/4** conditions (a real, consistent property: geodesic length tracks between-specimen
  variance relative to residual noise better than the raw two-point distance). But **bias loses in
  4/4 conditions, every time by a large and growing margin as damage increases** (24.1%→18.3%
  under pose alone, but 38.2%→46.3% under damage, vs C0's 25.6%→41.7% — worse in absolute terms
  at every damage level). This is exactly the failure mode Phase 1 predicted: summing many noisy
  mesh edges systematically inflates the path length (geodesic ≥ straight-line by construction,
  confirmed on the clean template in Phase 2 — 1.5-2.5% longer there already), and that inflation
  compounds under exactly the missing/damaged-geometry condition the task asked to test.

**Neither candidate clears gate 1 with a consistent margin.** Per the two-gate principle: a better
ground-truth number would still need to translate into better downstream behaviour to matter to
the project, but that question does not even arise here, because neither candidate reliably
produces a better ground-truth number in the first place. Running Phase 4 (cross-corpus
replication) on either would burn the multi-genus corpus requirement (§4 of `analyse_source.py`,
already noted as scarce in Task 2's `REPORT.md`) on candidates with no demonstrated reliability
edge — explicitly not done, per the plan ("only run this phase for a candidate that already beat
C0 in Phase 3").

## 4. A side observation, explicitly out of scope for this verdict

`chain_full`/`geo_full` (whole leg, coxa-to-pretarsus) score far higher everywhere (R 0.31-0.79,
bias 2-21%) than any distal-only candidate. This is expected — they lean heavily on the
well-supported proximal segments — and it is **not evidence for or against C1/C2 on the
leg_distal question**, because whole-leg length answers a different anatomical question than
"how long is the tibia+tarsus" (per the task's own instruction not to conflate a segment length
with a different quantity). If there is future interest in a *new* "total leg length" measurement
for `CORE_BLOCKS`, that would need its own ground-truth validation and cross-corpus replication
pass — not implied or pre-validated by anything in this report.

## 5. Recommendation

Leave `measure.py`/`calib_features.py`/`CORE_BLOCKS` exactly as they are. `leg_distal` stays
excluded from `CORE_BLOCKS`, and the underlying cause (scattered correspondence failure on thin,
pose-sensitive distal geometry, Phase 1) is an optimizer/correspondence problem, not a measurement
-definition problem — fixing it, if pursued, belongs in the fitting/optimization pipeline (e.g.
loss weighting or joint-limit work specific to distal leg segments), not in how length is computed
from the output. No code in `diagnostics/absolute_scale/` was modified in this task.

## Reproducing

```
conda activate pytorch3d
# Phase 1 (already reproducible per PHASE1_FINDING.md)

# Phase 2 candidates + sanity check
python diagnostics/appendage_evidence/candidates.py

# Phase 3 damage corpora (already generated + fit, SLURM job 3041347)
python diagnostics/appendage_evidence/make_damage_corpus.py --frac 0.30 --out synth_clean_drop30
python diagnostics/appendage_evidence/make_damage_corpus.py --frac 0.60 --out synth_clean_drop60
sbatch diagnostics/appendage_evidence/run_damage_fit.sbatch

# Full comparison, any condition
python diagnostics/appendage_evidence/score.py --run SYN_clean_w5 --corpus synth_clean
python diagnostics/appendage_evidence/score.py --run SYN_clean_pose0_w5 --corpus synth_clean_pose0
python diagnostics/appendage_evidence/score.py --run SYN_clean_drop30_w5 --corpus synth_clean_drop30
python diagnostics/appendage_evidence/score.py --run SYN_clean_drop60_w5 --corpus synth_clean_drop60
```

Artifacts: `diagnostics/appendage_evidence/out/phase3_full_comparison.json` (every individual
column's R/SNR/bias per condition), fit logs and outputs under
`diagnostics/appendage_evidence/logs/` and `diagnostics/appendage_evidence/runs/` +
`diagnostics/moonshot/runs/SYN_clean_{pose0,drop30,drop60}_w5/`.
