# D1 vs stock scorecard

- to-ship commit: `47165ae32bcf0ca390100ee43e6747c5f018af8a`
- Model: `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` (`SMILIFY_SMAL_FILE`), required by both
  arms since `ants_cfg.yaml` already sets `w_limit>0`/`w_midline>0`, which need authored
  `joint_limits`/`sym_verts` — the stock default `SMIL_OmniAnt.pkl` has neither.
- Hardware: 1x NVIDIA H100 (Slurm partition `c23g`, RWTH Aachen cluster), full iteration
  budgets (no reduction from smoke-test counts).
- Mesh set: N=10, `diagnostics/d1_evidence/mesh_list.txt` — first 10 (alphabetical) of the
  50-specimen `bench50_clean` set, materialized read-only from
  `feature/registration_moonshot:diagnostics/moonshot/bench50_clean/*.obj` (that raw .obj set
  does not exist on `to-ship`; the `comparison_output/bench50_clean_*` directory already on
  `to-ship` only retains per-specimen logs, not meshes). Local copy at
  `diagnostics/d1_evidence/mesh_set/` (gitignored — 10 raw scan `.obj` files, ~35MB, not
  committed).
- Metric suite: `diagnostics/moonshot/metrics.py`, ported verbatim (read-only `git show`,
  branch not checked out) from `feature/registration_moonshot` for this comparison — it is a
  pure evaluation module (surface agreement, edge/triangle distortion, bilateral symmetry,
  per-part coverage) with no dependency on any excluded mechanism (no hull/robust
  kernel/correspondence code). Arm D scores it via `optimise_moonshot.py --eval`; Arm S has no
  built-in eval pass, so `diagnostics/d1_evidence/eval_stage_npz.py` (new, mirrors
  `optimise_moonshot.py`'s `run_eval()` verbatim) scores its final-stage `.npz` the same way.
- Deviation from spec, disclosed: `config.PLOT_RESULTS` (legacy global flag, not an ants_cfg
  default) was temporarily set `False` for the Arm S run only. `True` (the checked-in default)
  makes `fitter_3d/optimise.py` render/save a 10-mesh matplotlib point-cloud plot after every
  stage; on this cluster that render stalled for 40+ minutes between Stage_2 and Stage_3 with
  0% GPU utilization (confirmed via `srun --overlap nvidia-smi`/`ps`), a rendering-backend
  issue unrelated to the fit itself. The toggle affects only PNG diagnostic plots, not any
  loss weight, schedule, or model parameter, and was reverted immediately after the run (see
  `git diff config.py` — clean). Arm D's hierarchical/moonshot code paths do not read this
  flag at all, so it could not have affected Arm D.

## Exact commands run

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
MESH_DIR=diagnostics/d1_evidence/mesh_set

# Arm D, step 1/2 (full budget: body_its=900 leg_its=1200 joint_its=900 deform_its=600, n_sample=8000)
python -u -m fitter_3d.optimise_hierarchical --mesh_dir $MESH_DIR \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir diagnostics/d1_evidence/runs/hier

# Arm D, step 2/2 (full budget: 1000+1000 its, n_sample=6000, from diagnostics/moonshot/cfg/D1_low.yaml)
python -u -m fitter_3d.optimise_moonshot --mesh_dir $MESH_DIR \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from diagnostics/d1_evidence/runs/hier/H2_joint.npz \
  --results_dir diagnostics/d1_evidence/runs/d1 --eval

# Arm S (full budget, ants_cfg.yaml unmodified aside from args.results_dir)
python -u -m fitter_3d.optimise --mesh_dir $MESH_DIR \
  --yaml_src diagnostics/d1_evidence/ants_cfg_arm_s.yaml

python -u diagnostics/d1_evidence/eval_stage_npz.py \
  --npz diagnostics/d1_evidence/runs/stock/Stage_3_deform_fine.npz \
  --mesh_dir $MESH_DIR --out_csv diagnostics/d1_evidence/metrics_stock.csv
```

Full transcripts: `diagnostics/d1_evidence/logs/arm_d_hier.log`, `arm_d_moonshot.log`,
`arm_s_stock.log`, `arm_s_eval.log`. No `Traceback`/`Error` in any of the three optimiser logs;
all 10x2 specimens produced a complete metrics row (no crashes, no dropped specimens).


## Per-specimen table

| specimen | arm | deform_mag | edge_logratio | fscore@0.01 | chamfer_l2 |
|---|---|---|---|---|---|
| Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj | D | 0.00399 | 0.16359 | 0.6669 | 0.00031 |
| Acromyrmex_coronatus_CASENT0744365_processed.obj | D | 0.00449 | 0.14625 | 0.5618 | 0.00051 |
| Acromyrmex_lobicornis_CASENT0878007_processed.obj | D | 0.00587 | 0.17290 | 0.3591 | 0.00075 |
| Anochetus_risii_CASENT0877608_processed.obj | D | 0.00327 | 0.12405 | 0.7296 | 0.00021 |
| Anochetus_risii_CASENT0877609_processed.obj | D | 0.00592 | 0.15585 | 0.3651 | 0.00085 |
| Aphaenogaster_pallida_CASENT0745235_processed.obj | D | 0.00267 | 0.11681 | 0.7859 | 0.00018 |
| Atopomyrmex_mocquerysi_CASENT0744784_processed.obj | D | 0.00349 | 0.16213 | 0.7007 | 0.00024 |
| Carebara_trechideros_CASENT0877591_processed.obj | D | 0.00360 | 0.14039 | 0.6780 | 0.00030 |
| Centromyrmex_brachycola_CASENT0744052_processed.obj | D | 0.00318 | 0.15086 | 0.7154 | 0.00021 |
| Cephalotes_minutus_CASENT0709253_processed.obj | D | 0.00366 | 0.15280 | 0.6781 | 0.00024 |
| Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj | S | 0.00373 | 0.20908 | 0.7152 | 0.00023 |
| Acromyrmex_coronatus_CASENT0744365_processed.obj | S | 0.00426 | 0.20172 | 0.5822 | 0.00044 |
| Acromyrmex_lobicornis_CASENT0878007_processed.obj | S | 0.00604 | 0.20842 | 0.3533 | 0.00072 |
| Anochetus_risii_CASENT0877608_processed.obj | S | 0.00369 | 0.17324 | 0.6378 | 0.00040 |
| Anochetus_risii_CASENT0877609_processed.obj | S | 0.00625 | 0.18742 | 0.3250 | 0.00126 |
| Aphaenogaster_pallida_CASENT0745235_processed.obj | S | 0.00366 | 0.17202 | 0.6693 | 0.00034 |
| Atopomyrmex_mocquerysi_CASENT0744784_processed.obj | S | 0.00378 | 0.20233 | 0.6759 | 0.00037 |
| Carebara_trechideros_CASENT0877591_processed.obj | S | 0.00409 | 0.20370 | 0.6238 | 0.00038 |
| Centromyrmex_brachycola_CASENT0744052_processed.obj | S | 0.00351 | 0.16888 | 0.6617 | 0.00026 |
| Cephalotes_minutus_CASENT0709253_processed.obj | S | 0.00369 | 0.19877 | 0.6760 | 0.00026 |

## Summary (mean / median)

| metric | mean D | median D | mean S | median S |
|---|---|---|---|---|
| deform_mag_mean | 0.00401 | 0.00363 | 0.00427 | 0.00375 |
| edge_logratio_absmean | 0.14856 | 0.15183 | 0.19256 | 0.20024 |
| fscore@0.01 | 0.62406 | 0.67806 | 0.59203 | 0.64978 |
| chamfer_l2 | 0.00038 | 0.00027 | 0.00046 | 0.00037 |

## Pass/fail vs bars

- D finishes without NaN/collapse on >=80% of meshes: PASS (see log excerpts / crash notes below)
- mean deform_mag(D) <= mean deform_mag(S): PASS (0.00401 vs 0.00427)
- mean F@0.01(D) not worse than S by more than 0.02: PASS (0.6241 vs 0.5920)

## Recommendation

**Use D1 (hierarchical -> moonshot) as the experimental mainline candidate: yes**, with the
caveat that this is one N=10 run, not a multi-seed study.

- Integrity (the metric this task weights higher than chamfer): D1's `edge_logratio_absmean`
  is 23% lower than stock (0.149 vs 0.193 mean) — the D1 mesh is measurably less distorted
  relative to its own rest geometry, on every one of the 10 specimens (see per-specimen table;
  D < S for `edge_logratio` on all 10 rows).
- `deform_mag_mean` is also lower for D1 on 8/10 specimens (mean 0.0040 vs 0.0043) — D1 is not
  winning integrity by hiding behind a larger free-form offset; it is using less of it.
- Surface agreement did not have to be sacrificed for that integrity gain: `fscore@0.01` is
  *higher* for D1 on 8/10 specimens (mean 0.624 vs 0.592), and `chamfer_l2` is lower for D1 on
  7/10 (mean 0.00038 vs 0.00046).
- All three pass bars defined for this task are met (10/10 specimens finished clean on both
  arms, deform_mag(D) <= deform_mag(S), F@0.01(D) within 0.02 of S — it's actually ahead).

What this run does NOT establish: single seed, N=10 (not the full bench50), and hierarchical's
per-leg partitioning is fit-derived rather than validated against any ground-truth
correspondence — the metric suite's part-coverage and midline numbers are proxies, not ground
truth (see `diagnostics/moonshot/metrics.py` docstring). Before treating D1 as the shipped
default (as opposed to an experimental opt-in, which is its current status per
`fitter_3d/README.md`), repeat at N=50 and >=2 seeds.

