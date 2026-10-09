# Cluster continuation handoff (2026-08-26)

Continuing the Phase 10 correspondence-network investigation on the RWTH HPC cluster after
switching from WSL. State as found, verified before touching anything (per project discipline):

## Gap found and fixed

`fitter_3d/pointcloud2smil/smil_correspondence_net.py` (the `SMILCorrespondenceNet` module B1
built, that `train_correspondence_net_B2_20260825.py` imports) was **missing** from this
checkout. Root cause: `.gitignore` blanket-ignored the whole `fitter_3d/pointcloud2smil/`
directory (originally meant for vendored PointNet++ code); the file was created on WSL but never
committed before the machine switch, so the WSL→cluster merge (`4f9076ba`) could not carry it
over. This is the same failure class already flagged in project memory re: `probe_*.py`.

- Reconstructed the file verbatim from the prior session's transcript.
- Verified it is not just plausible but **exactly correct**: loaded the existing
  `out_B2_correspondence_net_20260825/best_model.pt` checkpoint into a fresh instance with
  `strict=True` — zero missing/unexpected keys. Checkpoint is from epoch 16/120 (best val during
  a run that logged through epoch 20/120 before the WSL session ended).
- Fixed `.gitignore`: removed the blanket directory ignore (nothing vendored remains in that
  directory today; `__pycache__` there is still covered by the global `**/__pycache__/` rule).
  Note for future self: a per-file `!` negation does NOT work here — git does not allow
  re-including a file under an ignored parent directory, so the fix had to be removing the
  parent-level ignore itself, not adding exceptions under it.

## What was NOT recoverable and had to be redone

- The 236MB training corpus (`diagnostics/moonshot/synth_b2_train_corrb06.npz`) was deliberately
  excluded from the WSL→cluster push (over GitHub's file size limit) — regenerated via
  `make_b2_training_corpus_20260825.py --n 4000 --sampler correlated --rho 0.6`.
- `train_correspondence_net_B2_20260825.py` has no checkpoint-resume flag, so the epoch-20
  partial run could not be continued — training resubmitted from scratch for the full 120 epochs,
  per the script's default.

## Resubmitted job

`sbatch diagnostics/anatomical_pose_init/sbatch_logs/submit_B2_resubmit.sbatch` -> job **3167672**
(corpus regen + full 120-epoch training in one job, `c23g_low` partition — account is over its
core-hour quota this period, so this landed on the low-priority queue and may sit pending for a
while). Output going to
`diagnostics/anatomical_pose_init/out_B2_correspondence_net_20260826/` (new date suffix, so the
existing epoch-20 partial run/checkpoint under `..._20260825/` is preserved, not overwritten, per
the "preserve diagnostic artifacts" discipline).

## Still not done (per the original Phase A-D plan)

B3's pre-registered success bar already exists
(`out_B2_correspondence_net_20260825/PREREGISTRATION_B3_success_bar_20260825.md`) and does not
need to be rewritten. Once job 3167672 finishes: evaluate the new checkpoint against that bar
(convert to a candidate via C2's rigid-alignment reuse, run through `run_audit.py`, report
leg_acc/seg_acc with paired-t + sign + Wilcoxon, full-sample and outlier-excluded). C1 (sampler
ablation, old i.i.d. vs corrected) and Phase D (sim-to-real, synthesis doc) have not been started.
