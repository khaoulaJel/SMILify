# Morphometrics under D1_PROD + scale_cap — full-corpus run, 2026-08-13

Author: this investigation (branch `feature/investigation`), HPC cluster. **Separate from, not
an edit of, `diagnostics/morphometrics/REPORT_MORPHOMETRICS.md`** (Fabian's report, generated on
his own machine, worker-only corpus, D1 without `scale_cap`, 616 workers/82 genera). That report
is the methodology reference for every measurement definition, bug fix, and calibration claim
below — nothing here repeats or restates its content; this document covers only what's new in
this run: the `scale_cap` A/B's morphometrics deliverable, on the full combined corpus.

## 0. What's different about this run

- **Corpus**: 838 specimens total (757 `worker_ALT` + 81 `ALL_ANTS_CLEAN`), synced from the
  shared UM6P_2026 Google Drive to this HPC cluster 2026-08-13 — not Fabian's 616-worker-only
  set. Classification tests below pool both corpora (`--min_n 3`/`5` on the combined set),
  where Fabian's report tests workers and `ALL_ANTS_CLEAN` separately (§3 vs §5 there). This is
  the single largest reason the headline numbers differ from his — not a regression, a
  different (bigger, combined) test population.
- **Recipe**: `D1_PROD.yaml`, now including `w_scale: 0.052` (`scale_cap`), adopted 2026-08-13
  after passing the execution plan's §4.4 decision rule — see
  `diagnostics/EXECUTION_PLAN.md` §5 and `diagnostics/morphometrics/ab_scale_cap/` for the full
  A/B trail. Fabian's report predates `scale_cap` entirely.
- **Tags**: fits live under `diagnostics/moonshot/runs/AB_B_*`, symlinked as the canonical
  `MORPH_W0..W11`/`MORPH_CLEAN` (matching `run_all.sh`'s auto-discovery and
  `run_m1_fit_all.sh`'s default `TAG_PREFIX`) so future default invocations pick them up
  without extra flags. No fresh refit was run: arm B's fit *is* content-identical to a fresh
  `D1_PROD.yaml` run (weights diff only in an unused `results_dir` field — verified by diff),
  and the optimizer is deterministic, so re-running would have produced the same numbers at
  ~3h of pure GPU cost. Symlink + corrected recipe-fingerprint stamp instead (see
  `diagnostics/EXECUTION_PLAN.md` §5 provenance block for job IDs/commit hash).

## 1. Genus classification (primary endpoint, lot-blind)

Combined corpus, `analyse.py`, `--min_n 3` (92 genera / 678 specimens clear the bar):

| protocol | accuracy | null | p | lift |
|---|---|---|---|---|
| leave-one-specimen-out | 15.3% | 2.1% | <0.0001 | 7.1× |
| **leave-one-lot-out (defensible, primary)** | **11.4%** | 2.1% | <0.0001 | **5.3×** |
| subfamily, lot-blind | 47.5% | 28.3% | <0.0001 | 1.7× |

Full logs: `diagnostics/morphometrics/ab_scale_cap/analyse_B.log` (this arm),
`analyse_A.log` (baseline, no `scale_cap`, for comparison — 11.7%/5.4×, statistically flat
against this arm, see `EXECUTION_PLAN.md` §5).

### 1.1 Lot confound, quantified on this run

Same pattern Fabian's §3.4 established, reproduced here at full combined-corpus scale: the
specimen-blind protocol overstates the signal relative to the lot-blind one by
**(7.1−5.3)/7.1 ≈ 25%** (this arm) and **(7.2−5.4)/7.2 ≈ 25%** (baseline arm) — consistent
across both A/B arms, so `scale_cap` does not change the confound's magnitude. The lot-blind
number (5.3×/5.4×) is the one every decision in `EXECUTION_PLAN.md` §5 is judged against, per
Fabian's own established protocol.

### 1.2 Quality gate: column, not filter

Per the existing M1 design (confirmed, not just assumed — `analyse.py`'s own
`exclusion: none a priori`): the fittability/quality gate is carried as a column on every row,
never used to drop specimens before the primary classification test. This run follows that
convention unchanged.

## 2. Cross-corpus genus index table

`genus_table.py --min_n 5`, this arm's fits. Full CSV/JSON:
`diagnostics/morphometrics/ab_scale_cap/out_D1_PROD/` (`genus_indices.csv`, `genus_table.json`,
`fig_genus_indices.png`, `fig_crosscorpus.png`). 38 genera present in both corpora:

| index | R | | index | R |
|---|---|---|---|---|
| gaster_slenderness | 0.786 | | funiculus_scape | 0.277 |
| mesosoma_slenderness | 0.573 | | petiole_index | 0.297 |
| waist_constriction | 0.572 | | hindfemur_index | 0.269 |
| cephalic_index | 0.624 | | foreleg_hindleg | 0.193 |
| gaster_mesosoma | 0.374 | | scape_index | 0.117 |
| head_mesosoma | 0.359 | | tibia_femur | −0.035 |
| head_flatness | 0.351 | | mandible_index | −0.126 |
| mandible_slenderness | 0.294 | | | |

Median cross-corpus R = 0.297. Same qualitative split as Fabian's §5.2: head/body-shape
indices replicate better than appendage indices (leg-adjacent ratios weakest or negative here
too — `tibia_femur`, `mandible_index`). Consistent with his finding, not a new one; recorded
here because the underlying fits are new.

## 3. Absolute-scale status — unchanged, known gap

Same as Fabian's §2.1/§9, restated because it still applies: **no absolute size is recoverable
from either corpus.** `ALL_ANTS_CLEAN` arrives pre-normalised (bounding diagonal ≈ 1.6);
worker mesh units are arbitrary per scan. This run changes nothing about that — it is a known,
documented gap (original scan metadata / voxel spacing, if recovered, would fix it), not
something `scale_cap` or this A/B touches. All measurements here are Mosimann log-shape-ratios,
same as his report.

## 4. Integrity + anterior-scale pathology (this arm, full corpus)

Not in Fabian's report (predates `scale_cap`). Full numbers in `EXECUTION_PLAN.md` §5 and
`diagnostics/morphometrics/ab_scale_cap/integrity_anterior_full_corpus.json`. Summary: every
`metrics.py` integrity term moved <1% vs the no-`scale_cap` baseline; anterior joint-scale
outlier tail (the actual mechanism `scale_cap` targets) compressed 58–70% (max ratio,
head/mandible/antenna). Region-stratified integrity dashboard (via `sdf_stratify.py`, fixed —
see below):`diagnostics/morphometrics/ab_scale_cap/dashboard/integrity_dashboard_D1_PROD.csv`.
Failure gallery (top-20 worst `deform_mag`, by region): 13/20 in `leg`, 5/20 `body`, 1/20 each
`head`/`leg_distal` — deformation failures concentrate in legs, consistent with Fabian's §2.2
finding that distal-leg reliability is the weakest block (R=0.136).

**Bug found and fixed in `sdf_stratify.py` while building this dashboard**: `region_labels_for_fit()`
read raw pickle `dd["weights"]`/`v_template` (10,229 verts) instead of the model class's actual
post-symmetrisation arrays (10,235 verts, matching every real fit including Fabian's own
`CLEAN_M7`/`GEOCOH_01` runs) — a pre-existing bug (not introduced by this session's fits), now
fixed by loading through `smal_model.smal_torch.SMAL` directly. See the script's own docstring
for the full trace.

## 5. embed.py — not run, known environment blocker

`embed.py --quality_top 50` (PCA/t-SNE/UMAP + HDBSCAN, Fabian's §6) was attempted on this run's
fits but blocked: `import umap` → `cffi` module missing `FFI`/`__version__` attributes despite
correct pip metadata (`cffi 1.17.1` installed, broken at import). Same failure class already
documented elsewhere in this investigation ("`/p/scratch` package-extraction drops files",
`diagnostics/khaoula_v2/V2_CHANGELOG.md` appendix) — not attempted to fix here given the shared
conda env was actively in use by other queued/running jobs at the time; fixing it blind risked
breaking those. Left as a known gap, not silently skipped: the PCA/clustering embedding
question is answered by Fabian's §6 already (no cluster structure, local-signal-only) and
nothing about `scale_cap` gives a specific reason to expect that conclusion changed — this is
flagged as unconfirmed on the new fits, not assumed unchanged.

## 6. Reproducing this specific report

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
python diagnostics/morphometrics/analyse.py --worker_runs MORPH_W0 MORPH_W1 MORPH_W2 MORPH_W3 \
  MORPH_W4 MORPH_W5 MORPH_W6 MORPH_W7 MORPH_W8 MORPH_W9 MORPH_W10 MORPH_W11 \
  --clean_run MORPH_CLEAN --min_n 3
python diagnostics/morphometrics/genus_table.py --worker_runs MORPH_W0 MORPH_W1 MORPH_W2 \
  MORPH_W3 MORPH_W4 MORPH_W5 MORPH_W6 MORPH_W7 MORPH_W8 MORPH_W9 MORPH_W10 MORPH_W11 \
  --clean_run MORPH_CLEAN --min_n 5
python -m diagnostics.moonshot.sdf_stratify --run_dir diagnostics/moonshot/runs/MORPH_W0 \
  --out /tmp/region_deform_W0.csv   # repeat per chunk, or see dashboard/ for the merge script
```

(`--worker_runs`/`--clean_run` passed explicitly because `MORPH_*` here means the specific
`scale_cap` arm, not whatever the auto-discovery default happens to find later.)
