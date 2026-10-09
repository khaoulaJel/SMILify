# Provenance: run_cpu1 (the Task 2 figure run)

| item | value |
|---|---|
| recipe | D1_PROD: `fitter_3d.optimise_hierarchical --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 --skip_h3` then `fitter_3d.optimise_moonshot --yaml_src diagnostics/joint_alignment_benchmark/cfg/A_prod.yaml --init_from H2_joint.npz --eval` |
| model | `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl`, md5 `08b69daf119a6eda0cd699b8894ee7f8` |
| moonshot config | `A_prod.yaml`, md5 `1efa7a1481119aaf05e32a89375f4218` (weights identical to `diagnostics/moonshot/cfg/D1_PROD.yaml`) |
| code | HEAD `302d14396998d735992b324899b9e28931ed87c2` + uncommitted `fitter_3d/`+`smal_model/` patch, sha256 of `git diff` `04d27f2127f028dd749116e57a7067f1ef9ce208dbff9a75ef6196c36deab33d` (line-identical to JAB's `data/code_state_uncommitted.patch`) |
| pre-registration | `../PREREGISTRATION.md`, sha256 `f4f265e29a88fd0c098314e74e97d1252eae29041fe2300db7febf43a14fffb0` (frozen before any fit) |
| deviations | `../DEVIATIONS.md` D1 (CPU, account `default`) and D2 (one specimen per job, batch 1) |
| specimen set | 20 *A. vollenweideri* workers, `/hpcwork/nao48500/atta20/01..20.obj` (WOLO) |
| figure specimen | `13.obj`, md5 `e40551f0868b8d85f358d91dbfe9389d`, 12.5 mg, 5.52 mm (pre-registered input-only rule) |
| seed | 0 (every specimen) |
| fit jobs | Slurm array 4890224 (tasks 1-20), c23ms, account `default`, 8 CPU cores each, 2026-10-09 11:43:46-12:14:13; identical HEAD / md5 / patch hash in all 20 logs |
| fit outputs | `/hpcwork/nao48500/fig_registration_stages/D1_s0_cpu1/NN/{hier,moon}/`, stacked by `../merge_singles.py` into `D1_s0_cpu1_merged{,_hier}` |
| scoring job | 4890848 (c23ms, `default`), `../score_stages.py` + `../render_stages.py` (first attempt 4890225 failed on a label assertion, see DEVIATIONS.md) |

## Gates (all pass; `probes/gates.json`)

1. Saved-parameter reproduction: every stage rebuilds to <= 2.4e-7 (bar 1e-4); skeleton stages carry zero deform.
   The pre-registered "init equals v_template" check does NOT hold: the default state differs by up to
   0.317, entirely from unnormalised skinning weights on 134 left-mandible vertices (see
   `../probes/skinning_weight_defect_PROBE*`). The init panel is still the true start of H0.
2. Stage_3 vs D1_PROD's own `metrics.csv` (specimen 13): chamfer 7.025e-5 vs 7.043e-5; F@0.01 0.9471 vs 0.9462.
3. Independent numpy + scipy recompute at all four panel stages: chamfer within 1.4%, F@0.01 within 0.003.
4. Overlays inspected (`probes/overlay_*_PROBE.png`): orientation correct at every stage; Stage_2/3 surfaces
   coincide with the scan on body, head and gaster; legs stay thin blades inside the scan's tubular legs.
5. Penetration by pair (`probes/penetration_by_pair_PROBE_out.txt`): init 18 = mandible-antenna contacts
   already present in raw `v_template`; pose 46 = waist-legs 32 + mandible-antenna 14; gaster-legs 0 at every
   stage (proximity and GWN); no penetrating vertex belongs to the weight-defect set.
6. Orientation (`../probes/orientation_template_vs_scan13_PROBE.png`, `../probes/early_hier_spec13_cpu1_PROBE.png`).
7. Specimen 13 among the 20 at Stage_3: F@0.01 rank 6/20 (0.947; median 0.936, range 0.897-0.973).

## Metric definitions

- chamfer: `diagnostics/moonshot/metrics.surface_metrics` `chamfer_l2`, mean over 5 sampling seeds of 30k
  area-weighted points per surface; fitter frame (scan centred, max |coord| = 1).
- F@0.01: same function, tau = 0.01 absolute (1% of half-extent). `fscore@0.01bboxdiag` in the CSV is the
  other repo definition (tau = 1% of bbox diagonal; 0.996 at Stage_3 for specimen 13).
- penetration: `penetration_loss_batched` hard `num_penetrating` (vertex x direction x pair triples, here equal
  to unique vertices) with trainer.py eval arguments; soft count, gaster-legs and GWN columns in the CSV.

## Final render (style B)

Blender 5.0.1 (bpy module, `/hpcwork/nao48500/bpy_venv`), Cycles CPU, 1600 px, 128 samples, denoised,
`blender_render.py` via job 4892744 (c23ms, `default`, 2026-10-09). Input `probes/focus_meshes.npz` (the same
vertices that were scored). One orthographic camera = the pre-registered dorsolateral view (elev 22, azim 140,
half-width 1/1.15), checked against the pytorch3d panel for orientation and mirroring
(`../probes/blender_test_PROBE/mirror_check_PROBE.png`). Fit `#2a78d6`, scan `#b4b3ae`. Composed by
`../compose_figure.py`; numbers read from `../stages.csv` unchanged. The pytorch3d draft is kept as
`fig_registration_stages_pytorch3d_DRAFT.png`.
